#!/usr/bin/env python3
"""Fast, local screenshot OCR with conservative layout reconstruction."""

from __future__ import annotations

import argparse
import bisect
import io
import math
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class Token:
    text: str
    left: float
    top: float
    right: float
    bottom: float
    confidence: float = 1.0
    source: str = "ocr"

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.bottom - self.top)


@dataclass(frozen=True)
class Grid:
    x_lines: tuple[int, ...]
    y_lines: tuple[int, ...]


@dataclass(frozen=True)
class BorderlessTable:
    start: int
    end: int
    separators: tuple[float, ...]
    score: float


@dataclass(frozen=True)
class CodeBlock:
    top: float
    bottom: float
    base_left: float
    character_width: float


@dataclass(frozen=True)
class ListMarkerCandidate:
    center_x: float
    content_left: float
    marker: Token | None = None
    replacement: Token | None = None


class RegionRole(str, Enum):
    PROSE = "prose"
    LIST = "list"
    CODE = "code"
    TABLE = "table"


@dataclass(frozen=True)
class RegionMetrics:
    base_left: float
    character_width: float
    line_height: float
    line_gap: float


@dataclass(frozen=True)
class LayoutRegion:
    role: RegionRole
    start: int
    end: int
    block: int
    separators: tuple[float, ...] = ()


class OCRError(RuntimeError):
    pass


class ClipboardError(OCRError):
    def __init__(self, message: str, output: str) -> None:
        super().__init__(message)
        self.output = output


LIST_MARKERS = frozenset({"•", "◦", "▪", "▫", "‣", "∙", "●", "○"})
ORDERED_LIST_MARKER_RE = re.compile(r"^\d+[.)]$")
DASH_LIKE = "-−–—―一"
ASCII_TABLE_BORDER_CHARS = frozenset(
    "+-|_=十─━┄┈┼┬┴├┤┌┐└┘╋┳┻┣┫"
)
ASCII_TABLE_INTERSECTIONS = frozenset(
    "+十┼┬┴├┤┌┐└┘╋┳┻┣┫"
)
SEMANTIC_BORDER_OPERATORS = frozenset(
    {"+", "-", "|", "=", "++", "--", "||", "=="}
)
SHELL_VARIABLE_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")


def _is_list_marker(text: str) -> bool:
    stripped = text.strip()
    return bool(
        stripped[:1] in LIST_MARKERS
        or ORDERED_LIST_MARKER_RE.fullmatch(stripped)
    )


def _bounds(box) -> tuple[float, float, float, float]:
    points = np.asarray(box, dtype=float).reshape(-1, 2)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


@lru_cache(maxsize=1)
def _rapidocr_engine():
    from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

    return RapidOCR(
        params={
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType.SMALL,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Rec.model_type": ModelType.SMALL,
        }
    )


def _modern_rapidocr_tokens(image: Image.Image) -> list[Token]:
    trap = io.StringIO()
    with redirect_stdout(trap), redirect_stderr(trap):
        result = _rapidocr_engine()(
            # Keep RGB as a deliberate recognition hypothesis. Although RapidOCR's
            # ndarray loader documents BGR, BGR regresses colored UI screenshots
            # by turning decorative icons into confident text.
            np.asarray(image),
            use_det=True,
            use_cls=False,
            use_rec=True,
            return_word_box=True,
            box_thresh=0.2,
            text_score=0.2,
        )

    # A detection-only result object has no `txts` attribute at all.
    if result is None or not getattr(result, "txts", None):
        return []

    tokens: list[Token] = []
    if result.word_results and len(result.word_results) == len(result.txts):
        for line_text, line_words in zip(result.txts, result.word_results):
            line_tokens = []
            positions = []
            cursor = 0
            positions_valid = True
            for text, confidence, box in line_words:
                text = str(text).strip()
                if not text:
                    continue
                left, top, right, bottom = _bounds(box)
                line_tokens.append(
                    Token(
                        text,
                        left,
                        top,
                        right,
                        bottom,
                        float(confidence),
                    )
                )
                position = str(line_text).find(text, cursor)
                if position < 0:
                    positions_valid = False
                else:
                    positions.append((position, position + len(text)))
                    cursor = position + len(text)

            if positions_valid and len(positions) == len(line_tokens):
                reconstructed = [line_tokens[0]] if line_tokens else []
                for index, token in enumerate(line_tokens[1:], start=1):
                    between = str(line_text)[positions[index - 1][1] : positions[index][0]]
                    if between and any(character.isspace() for character in between):
                        reconstructed.append(token)
                        continue
                    previous = reconstructed[-1]
                    reconstructed[-1] = Token(
                        previous.text + token.text,
                        min(previous.left, token.left),
                        min(previous.top, token.top),
                        max(previous.right, token.right),
                        max(previous.bottom, token.bottom),
                        min(previous.confidence, token.confidence),
                    )
                line_tokens = reconstructed
            tokens.extend(line_tokens)
        return tokens

    for box, text, confidence in zip(result.boxes, result.txts, result.scores):
        text = str(text).strip()
        if text:
            left, top, right, bottom = _bounds(box)
            tokens.append(
                Token(text, left, top, right, bottom, float(confidence))
            )
    return tokens


def _strip_ascii_table_artifacts(tokens: list[Token]) -> list[Token]:
    """Remove OCR text produced by ASCII/terminal table border strokes."""
    if not tokens:
        return tokens

    def compact_border(token: Token) -> str:
        compact = re.sub(r"\s+", "", token.text)
        return (
            compact
            if compact
            and all(character in ASCII_TABLE_BORDER_CHARS for character in compact)
            else ""
        )

    text_heights = [
        token.height
        for token in tokens
        if any(character.isalnum() for character in token.text)
    ]
    typical_height = median(text_heights or [token.height for token in tokens])

    def has_inline_text_support(token: Token) -> bool:
        return any(
            other is not token
            and any(character.isalnum() for character in other.text)
            and abs(other.center_y - token.center_y) <= typical_height * 0.45
            and max(
                0.0,
                max(token.left, other.left) - min(token.right, other.right),
            )
            <= typical_height * 4
            for other in tokens
        )

    def is_semantic_operator(token: Token) -> bool:
        compact = re.sub(r"\s+", "", token.text)
        if compact not in SEMANTIC_BORDER_OPERATORS:
            return False
        if len(compact) == 1:
            return True
        return bool(
            (
                token.confidence >= 0.8
                and token.height >= typical_height * 0.6
            )
            or has_inline_text_support(token)
        )

    # Border-only rows can survive when the OCR engine changes which corner
    # glyphs it recognizes. Their large horizontal span distinguishes them
    # from compact operator expressions.
    border_row_tokens: set[int] = set()
    for line in cluster_lines(tokens):
        if len(line) < 2:
            continue
        border_like = []
        for token in line:
            compact = compact_border(token)
            tiny_corner_digit = bool(
                re.fullmatch(r"[1Il]", token.text.strip())
                and token.confidence < 0.8
                and token.height < typical_height * 0.65
            )
            border_like.append(bool(compact or tiny_corner_digit))
        span = max(token.right for token in line) - min(
            token.left for token in line
        )
        if (
            all(border_like)
            and span >= typical_height * 6
            and (
                median(token.confidence for token in line) < 0.85
                or median(token.height for token in line)
                < typical_height * 0.75
            )
        ):
            border_row_tokens.update(id(token) for token in line)

    # Tiny low-confidence strokes are safe to reject even if one of the outer
    # corners was not recognized, which keeps vertical separators out of cells.
    micro_fragments = {
        id(token)
        for token in tokens
        if compact_border(token)
        and not is_semantic_operator(token)
        and token.confidence < 0.78
        and token.height < typical_height * 0.85
    }
    always_drop = border_row_tokens | micro_fragments

    corners = [
        token
        for token in tokens
        if (compact := compact_border(token))
        and any(character in ASCII_TABLE_INTERSECTIONS for character in compact)
        and compact != "+"
    ]
    if (
        len(corners) < 2
        or not any(len(compact_border(token)) >= 2 for token in corners)
        or max(token.center_y for token in corners)
        - min(token.center_y for token in corners)
        < typical_height * 2
    ):
        return [token for token in tokens if id(token) not in always_drop]
    top = min(token.center_y for token in corners) - typical_height
    bottom = max(token.center_y for token in corners) + typical_height

    def is_border_artifact(token: Token) -> bool:
        compact = compact_border(token)
        if not compact:
            return False
        if is_semantic_operator(token):
            return False
        if len(compact) >= 2:
            return True
        return (
            any(
                character in ASCII_TABLE_INTERSECTIONS
                for character in compact
            )
            or (
                token.confidence < 0.75
                and token.height < typical_height * 0.85
            )
        )

    return [
        token
        for token in tokens
        if id(token) not in always_drop
        and not (
            top <= token.center_y <= bottom and is_border_artifact(token)
        )
    ]


def _token_recognition_variants(
    image: Image.Image, token: Token
) -> tuple[np.ndarray, ...]:
    pad = max(2, round(token.height * 0.08))
    box = (
        max(0, math.floor(token.left) - pad),
        max(0, math.floor(token.top) - pad),
        min(image.width, math.ceil(token.right) + pad),
        min(image.height, math.ceil(token.bottom) + pad),
    )
    crop = image.crop(box)
    upscaled = crop.resize(
        (max(1, crop.width * 2), max(1, crop.height * 2)),
        Image.Resampling.LANCZOS,
    )
    gray = ImageOps.grayscale(upscaled)
    if float(np.median(np.asarray(gray))) < 128:
        gray = ImageOps.invert(gray)
    contrasted = ImageOps.autocontrast(gray, cutoff=1).convert("RGB")
    sharpened = contrasted.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=2)
    )
    return (
        np.asarray(upscaled),
        np.asarray(contrasted),
        np.asarray(sharpened),
    )


def _balanced_wrapping_quote(text: str) -> bool:
    return len(text) >= 2 and text[0] in {'"', "'"} and text[-1] == text[0]


def _accept_crop_consensus(
    base: Token,
    candidate: str,
    votes: int,
    score: float,
    pair_support: int,
    shell_occurrences: Counter[str],
) -> bool:
    if votes < 2 or score < 0.75 or not candidate or candidate == base.text:
        return False
    base_compact = re.sub(r"\s+", "", base.text)
    candidate_compact = re.sub(r"\s+", "", candidate)
    if min(len(base_compact), len(candidate_compact)) <= 1:
        return False

    if (
        base_compact == candidate_compact
        and candidate.count(" ") > base.text.count(" ")
    ):
        return True
    if (
        not _balanced_wrapping_quote(base.text)
        and _balanced_wrapping_quote(candidate)
        and SequenceMatcher(None, base.text, candidate).ratio() >= 0.8
    ):
        return True

    base_variables = SHELL_VARIABLE_RE.findall(base.text)
    candidate_variables = SHELL_VARIABLE_RE.findall(candidate)
    case_only_shell_change = (
        len(base_variables) == len(candidate_variables) == 1
        and base.text.casefold() == candidate.casefold()
        and base_variables[0].casefold() == candidate_variables[0].casefold()
    )
    if case_only_shell_change:
        return (
            votes == 3
            and score >= 0.9
            and _supports_case_only_crop_correction(
                base_variables[0], candidate_variables[0]
            )
            and (
                shell_occurrences[base_variables[0].casefold()] == 1
                or score >= base.confidence + 0.005
            )
        )

    similarity = SequenceMatcher(None, base.text, candidate).ratio()
    if (
        pair_support >= 2
        and votes == 3
        and score >= 0.9
        and len(base_compact) == len(candidate_compact)
        and similarity >= 0.45
    ):
        return True
    return (
        votes == 3
        and score >= base.confidence + 0.08
        and similarity >= 0.7
    )


def _refine_token_recognition(
    image: Image.Image, tokens: list[Token]
) -> list[Token]:
    """Re-read word crops and apply only independently supported consensus."""
    eligible = [
        (index, token)
        for index, token in enumerate(tokens)
        if len(re.sub(r"\s+", "", token.text)) >= 2
    ]
    if not eligible:
        return tokens

    crops = [
        variant
        for _, token in eligible
        for variant in _token_recognition_variants(image, token)
    ]
    trap = io.StringIO()
    try:
        with redirect_stdout(trap), redirect_stderr(trap):
            result = _rapidocr_engine().recognize_txt(crops)
        crop_texts, crop_scores = result.txts, result.scores
    # Crop refinement is optional. RapidOCR may surface backend-specific
    # exceptions here; retain the successful primary pass for any of them.
    except Exception:  # noqa: BLE001
        return tokens
    if (
        crop_texts is None
        or crop_scores is None
        or len(crop_texts) != len(crops)
        or len(crop_scores) != len(crops)
    ):
        return tokens

    suggestions: dict[int, tuple[str, int, float]] = {}
    pair_counts: Counter[tuple[str, str]] = Counter()
    for position, (index, token) in enumerate(eligible):
        start = position * 3
        texts = [
            unicodedata.normalize("NFC", str(crop_texts[start + offset]).strip())
            for offset in range(3)
        ]
        candidate, votes = Counter(texts).most_common(1)[0]
        scores = [
            float(crop_scores[start + offset])
            for offset, text in enumerate(texts)
            if text == candidate
        ]
        score = sum(scores) / len(scores)
        suggestions[index] = (candidate, votes, score)
        if candidate and candidate != token.text:
            pair_counts[(token.text, candidate)] += 1

    shell_occurrences: Counter[str] = Counter(
        match.group().casefold()
        for token in tokens
        for match in SHELL_VARIABLE_RE.finditer(token.text)
    )
    refined = []
    for index, token in enumerate(tokens):
        suggestion = suggestions.get(index)
        if suggestion is None:
            refined.append(token)
            continue
        candidate, votes, score = suggestion
        if not _accept_crop_consensus(
            token,
            candidate,
            votes,
            score,
            pair_counts[(token.text, candidate)],
            shell_occurrences,
        ):
            refined.append(token)
            continue
        refined.append(
            Token(
                candidate,
                token.left,
                token.top,
                token.right,
                token.bottom,
                max(token.confidence, score),
                token.source,
            )
        )
    return refined


def _has_short_column_support(
    token: Token, tokens: list[Token], typical_height: float
) -> bool:
    """Preserve uncertain one-character cells when a real column supports them."""
    matches = 0
    for other in tokens:
        if other is token or other.confidence < 0.75:
            continue
        compact = re.sub(r"\s+", "", other.text)
        if not compact or len(compact) > 2:
            continue
        if abs(other.center_x - token.center_x) > typical_height * 0.75:
            continue
        if abs(other.center_y - token.center_y) < typical_height * 0.75:
            continue
        matches += 1
        if matches >= 2:
            return True
    return False


def _is_dash_run(text: str) -> bool:
    if not text or len(text) > 3 or any(character not in DASH_LIKE for character in text):
        return False
    return "一" not in text or len(text) >= 2


def _is_option_fragment(text: str) -> bool:
    return bool(re.fullmatch(rf"[{re.escape(DASH_LIKE)}]{{1,2}}[A-Za-z]", text))


def _clean_tokens(tokens: list[Token]) -> list[Token]:
    """Drop model fragments conservatively, then suppress contained duplicates."""
    if not tokens:
        return []

    reference_tokens = [
        token
        for token in tokens
        if token.confidence >= 0.8
        and any(unicodedata.category(character)[0] in {"L", "N"} for character in token.text)
    ]
    typical_height = median(
        [token.height for token in reference_tokens]
        or [token.height for token in tokens]
    )
    plausible: list[Token] = []
    for token in tokens:
        text = unicodedata.normalize("NFC", token.text).strip()
        compact = re.sub(r"\s+", "", text)
        if not compact:
            continue
        nearby_heights = [
            other.height
            for other in reference_tokens
            if abs(other.center_y - token.center_y)
            <= max(typical_height * 1.5, token.height * 1.5)
        ]
        local_height = (
            median(nearby_heights)
            if len(nearby_heights) >= 2
            else typical_height
        )
        is_short = len(compact) <= 2
        is_wordlike = any(
            unicodedata.category(character)[0] in {"L", "N"} for character in compact
        )
        is_fragment = token.height < local_height * 0.72
        preserves_syntax = bool(
            _is_dash_run(compact)
            or _is_option_fragment(compact)
            or compact in SEMANTIC_BORDER_OPERATORS
        )
        has_column_support = (
            is_short
            and is_wordlike
            and _has_short_column_support(token, tokens, local_height)
        )
        if (
            is_short
            and is_wordlike
            and token.height < local_height * 0.5
            and token.confidence < 0.99
            and not has_column_support
        ):
            continue
        if (
            is_short
            and token.confidence < 0.75
            and is_fragment
            and not preserves_syntax
        ):
            continue
        if (
            len(compact) == 1
            and token.confidence < 0.55
            and is_wordlike
            and not has_column_support
        ):
            continue
        plausible.append(
            Token(
                text,
                token.left,
                token.top,
                token.right,
                token.bottom,
                token.confidence,
                token.source,
            )
        )

    kept: list[Token] = []
    for i, t1 in enumerate(plausible):
        a1 = (t1.right - t1.left) * (t1.bottom - t1.top)
        is_contained = False
        for j, t2 in enumerate(plausible):
            if i == j or a1 <= 0:
                continue
            a2 = (t2.right - t2.left) * (t2.bottom - t2.top)
            inter_left = max(t1.left, t2.left)
            inter_top = max(t1.top, t2.top)
            inter_right = min(t1.right, t2.right)
            inter_bottom = min(t1.bottom, t2.bottom)
            if inter_right > inter_left and inter_bottom > inter_top:
                inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
                if inter_area / a1 > 0.65 and a2 > a1 * 1.3:
                    is_contained = True
                    break
        if not is_contained:
            kept.append(t1)
    return kept


def _supports_case_only_crop_correction(variant: str, candidate: str) -> bool:
    """Limit crop-based case correction to an ambiguous mixed-case spelling."""
    name, target = variant.lstrip("$"), candidate.lstrip("$")
    return not (name.isupper() or name.islower()) and (
        target.isupper() or target.islower()
    )


def recognize(image: Image.Image) -> list[Token]:
    try:
        tokens = _modern_rapidocr_tokens(image)
        tokens = _strip_ascii_table_artifacts(tokens)
        tokens = _refine_token_recognition(image, tokens)
        # Shell identifiers are case-sensitive. Only direct crop evidence may
        # change their case; page-level majority voting can corrupt valid pairs
        # such as `$MyVar` and `$MYVAR`.
        return _clean_tokens(tokens)
    except Exception as exc:
        raise OCRError(f"OCR engine failed: {exc}") from exc


def _cluster_positions(records: list[tuple[float, ...]], axis: int) -> list[tuple]:
    groups: list[list[tuple[float, ...]]] = []
    for record in sorted(records, key=lambda value: value[axis]):
        if not groups:
            groups.append([record])
            continue
        group_center = sum(value[axis] for value in groups[-1]) / len(groups[-1])
        if abs(record[axis] - group_center) <= 4:
            groups[-1].append(record)
        else:
            groups.append([record])

    clustered = []
    for group in groups:
        position = round(sum(value[axis] for value in group) / len(group))
        clustered.append((position, group))
    return clustered


def detect_grid(image: Image.Image) -> Grid | None:
    """Detect a dominant ruled table using its actual border lines."""
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    edges = cv2.Canny(gray, 10, 50)
    min_dimension = min(height, width)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(20, int(min_dimension * 0.12)),
        minLineLength=max(20, int(min_dimension * 0.25)),
        maxLineGap=max(4, int(min_dimension * 0.01)),
    )
    if lines is None:
        return None

    horizontal: list[tuple[float, ...]] = []
    vertical: list[tuple[float, ...]] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        if abs(y2 - y1) <= 2:
            y = (y1 + y2) / 2
            if 3 < y < height - 4:
                horizontal.append((y, min(x1, x2), max(x1, x2)))
        elif abs(x2 - x1) <= 2:
            x = (x1 + x2) / 2
            if 3 < x < width - 4:
                vertical.append((x, min(y1, y2), max(y1, y2)))

    if not horizontal or not vertical:
        return None

    longest_horizontal = max(record[2] - record[1] for record in horizontal)
    longest_vertical = max(record[2] - record[1] for record in vertical)
    horizontal = [
        record
        for record in horizontal
        if record[2] - record[1] >= max(width * 0.2, longest_horizontal * 0.65)
    ]
    vertical = [
        record
        for record in vertical
        if record[2] - record[1] >= max(height * 0.2, longest_vertical * 0.65)
    ]

    horizontal_groups = _cluster_positions(horizontal, axis=0)
    vertical_groups = _cluster_positions(vertical, axis=0)
    if len(horizontal_groups) < 3 or len(vertical_groups) < 3:
        return None

    x_lines = [position for position, _ in vertical_groups]
    y_lines = [position for position, _ in horizontal_groups]
    x_min, x_max = min(x_lines), max(x_lines)
    y_min, y_max = min(y_lines), max(y_lines)

    horizontal_groups = [
        group
        for group in horizontal_groups
        if max(record[2] for record in group[1]) - min(record[1] for record in group[1])
        >= (x_max - x_min) * 0.7
    ]
    vertical_groups = [
        group
        for group in vertical_groups
        if max(record[2] for record in group[1]) - min(record[1] for record in group[1])
        >= (y_max - y_min) * 0.7
    ]
    x_lines = sorted(position for position, _ in vertical_groups)
    y_lines = sorted(position for position, _ in horizontal_groups)

    if horizontal_groups:
        h_min = min(min(rec[1] for rec in grp[1]) for grp in horizontal_groups)
        h_max = max(max(rec[2] for rec in grp[1]) for grp in horizontal_groups)
        edge_tolerance = max(8, round(min_dimension * 0.03))
        if x_lines and (
            x_lines[0] - h_min > edge_tolerance
            or h_max - x_lines[-1] > edge_tolerance
        ):
            return None

    if len(x_lines) < 3 or len(y_lines) < 3:
        return None
    if any(right - left < 8 for left, right in pairwise(x_lines)):
        return None
    if any(bottom - top < 8 for top, bottom in pairwise(y_lines)):
        return None

    table_area = (x_lines[-1] - x_lines[0]) * (y_lines[-1] - y_lines[0])
    if table_area < width * height * 0.08:
        return None
    return Grid(tuple(x_lines), tuple(y_lines))


def _colored_icons(image: Image.Image, grid: Grid) -> list[Token]:
    """Recover common colored UI icons that text recognizers omit."""
    rgb = np.asarray(image)
    red = rgb[:, :, 0].astype(float)
    green = rgb[:, :, 1].astype(float)
    blue = rgb[:, :, 2].astype(float)
    masks = {
        "✅": (green > 90) & (green > red * 1.25) & (green > blue * 1.15),
        "❌": (red > 120) & (red > green * 1.35) & (red > blue * 1.25),
        "⚠️": (red > 150) & (green > 110) & (blue < 120) & (red > green * 0.8),
    }

    icons: list[Token] = []
    median_cell_height = median(
        bottom - top for top, bottom in zip(grid.y_lines, grid.y_lines[1:])
    )
    max_size = max(64, int(median_cell_height * 0.8))
    for symbol, raw_mask in masks.items():
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            raw_mask.astype(np.uint8), 8
        )
        for component in range(1, count):
            left, top, width, height, area = map(int, stats[component])
            fill = area / max(1, width * height)
            if not (5 <= width <= max_size and 5 <= height <= max_size):
                continue
            if not (0.45 <= width / height <= 2.2 and area >= 20 and fill >= 0.12):
                continue
            center_x = left + width / 2
            center_y = top + height / 2
            if not (
                grid.x_lines[0] < center_x < grid.x_lines[-1]
                and grid.y_lines[0] < center_y < grid.y_lines[-1]
            ):
                continue
            icons.append(
                Token(
                    symbol,
                    left,
                    top,
                    left + width,
                    top + height,
                    1.0,
                    "icon",
                )
            )
    return icons


def merge_colored_icons(
    tokens: list[Token], image: Image.Image, grid: Grid
) -> list[Token]:
    glyphs = {
        "✅": {"v", "√", "✓", "✔"},
        "❌": {"x", "×"},
        "⚠️": {"a", "!", "！"},
    }
    merged = list(tokens)
    for icon in _colored_icons(image, grid):
        matching_glyphs = []
        for token in merged:
            overlaps = not (
                token.right < icon.left - 3
                or token.left > icon.right + 3
                or token.bottom < icon.top - 3
                or token.top > icon.bottom + 3
            )
            if (
                overlaps
                and token.source == "ocr"
                and token.text.strip().lower() in glyphs[icon.text]
            ):
                matching_glyphs.append(token)
        if not matching_glyphs:
            continue

        retained = [
            token
            for token in merged
            if not any(token is glyph for glyph in matching_glyphs)
        ]
        retained.append(icon)
        merged = retained
    return merged


def recover_horizontal_punctuation(
    tokens: list[Token], image: Image.Image
) -> list[Token]:
    """Recover isolated hyphens that OCR text detectors commonly discard."""
    if not tokens:
        return tokens

    lines = cluster_lines(tokens)
    ocr_heights = [token.height for token in tokens if token.source == "ocr"]
    typical_height = median(ocr_heights or [token.height for token in tokens])
    mask = _foreground_mask(image)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    recovered = list(tokens)
    for component in range(1, component_count):
        left, top, width, height, area = map(int, stats[component])
        if not (
            4 <= width <= typical_height * 1.25
            and 1 <= height <= max(3, typical_height * 0.25)
            and width / height >= 1.2
            and area >= 3
        ):
            continue
        right = left + width
        bottom = top + height
        if any(
            not (
                token.right < left
                or token.left > right
                or token.bottom < top
                or token.top > bottom
            )
            for token in recovered
        ):
            continue

        center_x = left + width / 2
        center_y = top + height / 2
        possible_lines = []
        for line in lines:
            line_top = min(token.top for token in line)
            line_bottom = max(token.bottom for token in line)
            line_left = min(token.left for token in line)
            line_right = max(token.right for token in line)
            if (
                line_top <= center_y <= line_bottom
                and line_left - typical_height <= center_x <= line_right + typical_height
            ):
                possible_lines.append(
                    (abs(center_y - (line_top + line_bottom) / 2), line_top, line_bottom)
                )
        if not possible_lines:
            continue
        _, line_top, line_bottom = min(possible_lines)
        relative_y = (center_y - line_top) / max(1.0, line_bottom - line_top)
        if 0.25 <= relative_y <= 0.7:
            recovered.append(
                Token("-", left, top, right, bottom, 1.0, "punctuation")
            )
    return recovered


def _foreground_mask(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    background = float(np.median(gray))
    if background >= 128:
        return gray < background - 30
    return gray > background + 30


def recover_list_markers(tokens: list[Token], image: Image.Image) -> list[Token]:
    """Recover or correct repeated compact bullets to the left of text lines."""
    if not tokens:
        return tokens

    lines = cluster_lines(tokens)
    ocr_heights = [token.height for token in tokens if token.source == "ocr"]
    typical_height = median(ocr_heights or [token.height for token in tokens])
    mask = _foreground_mask(image)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    components = []
    for component in range(1, component_count):
        left, top, width, height, area = map(int, stats[component])
        fill = area / max(1, width * height)
        if not (
            typical_height * 0.12 <= width <= typical_height * 0.5
            and typical_height * 0.12 <= height <= typical_height * 0.5
            and 0.55 <= width / height <= 1.8
            and fill >= 0.18
        ):
            continue
        components.append((left, top, width, height))

    candidates: list[ListMarkerCandidate] = []
    for line in lines:
        line = sorted(line, key=lambda token: token.left)
        first = line[0]
        if first.text[:1] in LIST_MARKERS:
            content_left = (
                line[1].left
                if len(line) >= 2
                else first.right + typical_height * 0.5
            )
            candidates.append(
                ListMarkerCandidate(first.center_x, content_left)
            )
            continue
        line_top = min(token.top for token in line)
        line_bottom = max(token.bottom for token in line)
        line_center = (line_top + line_bottom) / 2
        matches = []
        for left, top, width, height in components:
            right = left + width
            bottom = top + height
            center_x = left + width / 2
            center_y = top + height / 2
            if (
                abs(center_y - line_center)
                > max(typical_height, line_bottom - line_top) * 0.3
            ):
                continue
            overlapping = [
                token
                for token in line
                if not (
                    token.right < left
                    or token.left > right
                    or token.bottom < top
                    or token.top > bottom
                )
            ]
            replacement = None
            if overlapping:
                compact = re.sub(r"\s+", "", first.text)
                if not (
                    first in overlapping
                    and len(line) >= 2
                    and len(compact) <= 2
                    and (
                        first.height < typical_height * 0.65
                        or first.confidence < 0.75
                    )
                ):
                    continue
                replacement = first
                content_left = line[1].left
            else:
                content_left = first.left

            gap = content_left - right
            if not typical_height * 0.1 <= gap <= typical_height * 1.1:
                continue
            if any(
                not (
                    token.right < left
                    or token.left > right
                    or token.bottom < top
                    or token.top > bottom
                )
                for token in tokens
                if token is not replacement
            ):
                continue
            matches.append(
                (
                    gap,
                    ListMarkerCandidate(
                        center_x,
                        content_left,
                        Token("•", left, top, right, bottom, 1.0, "marker"),
                        replacement,
                    ),
                )
            )
        if matches:
            _, candidate = min(matches, key=lambda item: item[0])
            candidates.append(candidate)

    recovered = list(tokens)
    for candidate in candidates:
        support = sum(
            abs(other.center_x - candidate.center_x) <= typical_height * 0.3
            and abs(other.content_left - candidate.content_left)
            <= typical_height * 0.35
            for other in candidates
        )
        if support < 2 or candidate.marker is None:
            continue
        if candidate.replacement is not None:
            recovered = [
                token
                for token in recovered
                if token is not candidate.replacement
            ]
        recovered.append(candidate.marker)
    return recovered


def cluster_lines(tokens: list[Token]) -> list[list[Token]]:
    if not tokens:
        return []
    ocr_heights = [token.height for token in tokens if token.source == "ocr"]
    typical_height = median(ocr_heights or [token.height for token in tokens])
    tolerance = max(3.0, typical_height * 0.55)
    lines: list[dict] = []

    for token in sorted(tokens, key=lambda item: (item.center_y, item.left)):
        candidates = [
            (abs(token.center_y - line["center"]), index)
            for index, line in enumerate(lines)
            if abs(token.center_y - line["center"]) <= tolerance
        ]
        if not candidates:
            lines.append({"tokens": [token], "centers": [token.center_y], "center": token.center_y})
            continue
        _, index = min(candidates)
        lines[index]["tokens"].append(token)
        lines[index]["centers"].append(token.center_y)
        lines[index]["center"] = median(lines[index]["centers"])

    return [
        sorted(line["tokens"], key=lambda item: item.left)
        for line in sorted(lines, key=lambda item: item["center"])
    ]


def _normalize_join_text(
    text: str,
    code_like: bool,
    role: RegionRole | None = None,
) -> str:
    code_context = role == RegionRole.CODE or (role is None and code_like)
    if _is_option_fragment(text):
        text = re.sub(
            rf"^[{re.escape(DASH_LIKE)}]+",
            lambda match: "-" * len(match.group()),
            text,
        )
    elif code_context and _is_dash_run(text):
        text = "-" * len(text)
    if (
        code_context
        and len(text) >= 3
        and text[0] in {'"', "'"}
        and text[-1] in {'"', "'"}
        and text[0] != text[-1]
        and "/" in text
    ):
        text = text[:-1] + text[0]
    if re.fullmatch(r"[A-Z][A-Za-z ]{1,24}:/[^/].*", text):
        text = text.replace(":/", ": /", 1)
    return text


def _tight_fragment_join(
    left: str,
    right: str,
    gap: float,
    character_width: float,
    code_context: bool,
) -> bool:
    """Join lexical fragments only when punctuation and geometry agree."""
    if left.endswith("."):
        stem = left[:-1]
        basename = re.split(r"[/\\]", stem)[-1]
        looks_like_filename = bool(
            re.fullmatch(r"[A-Za-z0-9_.~$/\\-]+", stem)
            and re.fullmatch(r"[A-Za-z0-9_~-]{4,}", basename)
            and re.fullmatch(r"[a-z0-9_]{1,5}[,;:]?", right)
        )
        if looks_like_filename and gap <= max(4.0, character_width * 0.75):
            return True
    if gap > max(3.0, character_width * 0.45):
        return False
    if left == "." and re.match(r"^[A-Za-z0-9_]", right):
        return True
    if left.endswith(("/", "\\", "~", "$", "@")):
        return bool(
            re.match(r"^[A-Za-z0-9_.~-]", right)
            and (code_context or left not in {"/", "\\"})
        )
    if right.startswith(("/", "\\")):
        return bool(
            re.search(r"[.~$@/\\]$", left)
            or code_context
        )
    return False


def join_line(
    tokens: list[Token], role: RegionRole | None = None
) -> str:
    ordered = sorted(tokens, key=lambda item: item.left)
    raw_texts = [token.text.strip() for token in ordered if token.text.strip()]
    code_like = any(
        "/" in text
        or re.match(rf"^[{re.escape(DASH_LIKE)}]{{1,2}}[A-Za-z]", text)
        for text in raw_texts
    )
    code_context = role == RegionRole.CODE or (role is None and code_like)
    width_estimates = [
        (token.right - token.left) / len(compact)
        for token in ordered
        if len(compact := re.sub(r"\s+", "", token.text)) >= 2
    ]
    character_width = median(width_estimates) if width_estimates else 8.0
    parts: list[str] = []
    previous_token: Token | None = None
    for token in ordered:
        text = _normalize_join_text(token.text.strip(), code_like, role)
        if not text:
            continue
        if not parts:
            parts.append(text)
        elif (
            re.fullmatch(r"[,.;:!?%)\]}\'\"]+", text)
            or parts[-1].endswith(("(", "[", "{"))
            or (
                previous_token is not None
                and _tight_fragment_join(
                    parts[-1],
                    text,
                    max(0.0, token.left - previous_token.right),
                    character_width,
                    code_context,
                )
            )
        ):
            parts[-1] += text
        else:
            parts.append(text)
        previous_token = token
    result = " ".join(parts)
    if code_context:
        # A leading dot is commonly read as a short dash at screenshot scale.
        # `-/path` is not a valid relative shell path, while `./path` is.
        result = re.sub(
            rf"^[{re.escape(DASH_LIKE)}]\s*(?=/[A-Za-z0-9_.~])",
            ".",
            result,
            count=1,
        )
    return result


def cell_text(tokens: list[Token]) -> str:
    lines = cluster_lines(tokens)
    text = " ".join(
        join_line(line, RegionRole.TABLE) for line in lines if line
    )
    return re.sub(r"\s+", " ", text).strip()


def escape_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").strip()


def markdown_table(rows: list[list[str]]) -> str:
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    output = ["| " + " | ".join(map(escape_cell, rows[0])) + " |"]
    output.append("| " + " | ".join(["---"] * width) + " |")
    output.extend(
        "| " + " | ".join(map(escape_cell, row)) + " |" for row in rows[1:]
    )
    return "\n".join(output)


def _recover_table_glyphs(
    cells: list[list[list[Token]]],
) -> list[list[list[Token]]]:
    """Recover horizontally oriented infinity glyphs from numeric columns."""
    if len(cells) < 4:
        return cells
    recovered = [[list(cell) for cell in row] for row in cells]
    column_count = max((len(row) for row in cells), default=0)
    for column in range(column_count):
        records = [
            (row, cell[0])
            for row, row_cells in enumerate(cells)
            if column < len(row_cells)
            and len(cell := row_cells[column]) == 1
            and re.fullmatch(r"\d", cell[0].text.strip())
        ]
        reference = [
            token for _, token in records if token.text.strip() != "8"
        ]
        if len(reference) < 3:
            continue
        typical_ratio = median(
            (token.right - token.left) / token.height
            for token in reference
        )
        typical_height = median(token.height for token in reference)
        typical_width = median(token.right - token.left for token in reference)
        for row, token in records:
            ratio = (token.right - token.left) / token.height
            if not (
                token.text.strip() == "8"
                and ratio >= typical_ratio * 1.2
                and token.height <= typical_height * 0.92
                and token.right - token.left >= typical_width * 1.08
            ):
                continue
            recovered[row][column] = [
                Token(
                    "∞",
                    token.left,
                    token.top,
                    token.right,
                    token.bottom,
                    token.confidence,
                    "table-glyph",
                )
            ]
    return recovered


def _normalize_table_identifier_case(
    cells: list[list[list[Token]]],
) -> list[list[list[Token]]]:
    """Resolve case-only disagreements in repeated table identifier families."""
    if len(cells) < 4:
        return cells
    normalized = [[list(cell) for cell in row] for row in cells]
    column_count = max((len(row) for row in cells), default=0)

    header_groups: dict[
        str, list[tuple[int, Token, str, str]]
    ] = {}
    if cells:
        for column, cell in enumerate(cells[0]):
            if len(cell) != 1:
                continue
            token = cell[0]
            match = re.fullmatch(r"([A-Za-z]+)(\d+)", token.text.strip())
            if match:
                prefix, suffix = match.groups()
                header_groups.setdefault(prefix.casefold(), []).append(
                    (column, token, prefix, suffix)
                )
    family = max(header_groups.values(), key=len, default=[])
    if len(family) < 3:
        return cells

    header_lower = sum(prefix.islower() for _, _, prefix, _ in family)
    header_upper = sum(prefix.isupper() for _, _, prefix, _ in family)
    header_case = (
        "lower"
        if header_lower > header_upper
        else "upper" if header_upper > header_lower else None
    )
    if header_case is not None:
        for column, token, prefix, suffix in family:
            replacement = (
                prefix.lower() if header_case == "lower" else prefix.upper()
            ) + suffix
            normalized[0][column] = [
                Token(
                    replacement,
                    token.left,
                    token.top,
                    token.right,
                    token.bottom,
                    token.confidence,
                    "table-case",
                )
            ]

    categorical: list[tuple[int, list[tuple[int, Token]]]] = []
    for column in range(column_count):
        records: list[tuple[int, Token]] = []
        occupied = 0
        for row in range(1, len(cells)):
            if column >= len(cells[row]) or not cells[row][column]:
                continue
            occupied += 1
            cell = cells[row][column]
            if len(cell) == 1 and re.fullmatch(
                r"[A-Za-z]", cell[0].text.strip()
            ):
                records.append((row, cell[0]))
        if (
            len(records) >= 3
            and len(records) / max(1, occupied) >= 0.8
        ):
            categorical.append((column, records))
    if len(categorical) < 2:
        return normalized

    values = [
        token.text.strip()
        for _, records in categorical
        for _, token in records
    ]
    lower_count = sum(value.islower() for value in values)
    upper_count = sum(value.isupper() for value in values)
    preferred_case = (
        "lower"
        if lower_count > upper_count
        else "upper"
        if upper_count > lower_count
        else header_case
    )
    if preferred_case is None:
        return normalized

    for column, records in categorical:
        for row, token in records:
            value = token.text.strip()
            replacement = (
                value.lower() if preferred_case == "lower" else value.upper()
            )
            normalized[row][column] = [
                Token(
                    replacement,
                    token.left,
                    token.top,
                    token.right,
                    token.bottom,
                    token.confidence,
                    "table-case",
                )
            ]
    return normalized


def _format_grid(tokens: list[Token], grid: Grid) -> str:
    row_count = len(grid.y_lines) - 1
    column_count = len(grid.x_lines) - 1
    cells = [[[] for _ in range(column_count)] for _ in range(row_count)]
    before: list[Token] = []
    after: list[Token] = []
    inside: list[Token] = []

    for token in tokens:
        if token.center_y <= grid.y_lines[0]:
            before.append(token)
            continue
        if token.center_y >= grid.y_lines[-1]:
            after.append(token)
            continue
        if not (grid.x_lines[0] < token.center_x < grid.x_lines[-1]):
            if token.center_x <= grid.x_lines[0]:
                before.append(token)
            else:
                after.append(token)
            continue
        row = bisect.bisect_right(grid.y_lines, token.center_y) - 1
        column = bisect.bisect_right(grid.x_lines, token.center_x) - 1
        if 0 <= row < row_count and 0 <= column < column_count:
            cells[row][column].append(token)
            inside.append(token)

    cells = _normalize_table_identifier_case(_recover_table_glyphs(cells))
    rows = [[cell_text(cell) for cell in row] for row in cells]
    sections = []
    if before:
        sections.append(format_plain_lines(cluster_lines(before)))
    table = markdown_table(rows)
    if table:
        sections.append(table)
    elif inside:
        # Too few populated rows for a Markdown table; keep the text as text.
        sections.append(format_plain_lines(cluster_lines(inside)))
    if after:
        sections.append(format_plain_lines(cluster_lines(after)))
    return "\n\n".join(section for section in sections if section)


def _line_center(line: list[Token]) -> float:
    return median(token.center_y for token in line)


def _line_height(line: list[Token]) -> float:
    return median(token.height for token in line)


def _split_line_blocks(lines: list[list[Token]]) -> list[tuple[int, int]]:
    if not lines:
        return []
    if len(lines) == 1:
        return [(0, 1)]
    centers = [_line_center(line) for line in lines]
    gaps = [right - left for left, right in pairwise(centers)]
    typical_height = median(_line_height(line) for line in lines)
    baseline_gap = min(gaps) if len(gaps) < 3 else median(gaps)
    split_at = max(baseline_gap * 2.1, typical_height * 2.2)
    blocks = []
    start = 0
    for index, gap in enumerate(gaps, start=1):
        if gap > split_at:
            blocks.append((start, index))
            start = index
    blocks.append((start, len(lines)))
    return blocks


def _gap_runs(values: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(values)
    if not len(indices):
        return []
    runs = []
    start = previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index > previous + 1:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    return runs


def _evaluate_separators(
    lines: list[list[Token]], separators: tuple[float, ...]
) -> float | None:
    column_count = len(separators) + 1
    occupied = np.zeros((len(lines), column_count), dtype=bool)
    for row, line in enumerate(lines):
        for token in line:
            column = bisect.bisect_right(separators, token.center_x)
            occupied[row, column] = True

    support = occupied.mean(axis=0)
    cells_per_row = occupied.sum(axis=1)
    multi_column_rows = float((cells_per_row >= 2).mean())
    if support.min() < 0.55 or multi_column_rows < 0.8:
        return None
    return (
        len(lines) * (float(cells_per_row.mean()) - 1)
        + float(support.min()) * 4
        + multi_column_rows * 4
        - column_count * 0.15
    )


def _is_valid_table_row(line: list[Token], separators: tuple[float, ...]) -> bool:
    for token in line:
        for sep in separators:
            if token.left < sep < token.right:
                return False
    cells = [[] for _ in range(len(separators) + 1)]
    for token in line:
        cells[bisect.bisect_right(separators, token.center_x)].append(token)
    occupied = [bool(cell) for cell in cells]
    return sum(occupied) >= 2


def _column_alignment_score(
    lines: list[list[Token]],
    separators: tuple[float, ...],
    character_width: float,
) -> float | None:
    """Score columns by their best shared left, center, or right alignment."""
    column_count = len(separators) + 1
    columns: list[list[list[Token]]] = [[] for _ in range(column_count)]
    for line in lines:
        cells: list[list[Token]] = [[] for _ in range(column_count)]
        for token in line:
            cells[bisect.bisect_right(separators, token.center_x)].append(token)
        for column, cell in enumerate(cells):
            if cell:
                columns[column].append(cell)

    score = 0.0
    for cells in columns:
        if len(cells) < 2:
            return None
        lefts = [min(token.left for token in cell) for cell in cells]
        rights = [max(token.right for token in cell) for cell in cells]
        centers = [(left + right) / 2 for left, right in zip(lefts, rights)]
        dispersions = []
        for values in (lefts, centers, rights):
            center = median(values)
            dispersions.append(
                median(abs(value - center) for value in values)
                / max(1.0, character_width)
            )
        best = min(dispersions)
        if best > 2.5:
            return None
        score += 2.5 - best
    return score


MAX_TABLE_WINDOW = 30


def _score_table_window(
    sub_lines: list[list[Token]], start: int, end: int
) -> BorderlessTable | None:
    """Score one candidate span; the result never depends on its surroundings."""
    marker_rows = sum(
        _is_list_marker(min(line, key=lambda token: token.left).text)
        for line in sub_lines
    )
    if marker_rows / len(sub_lines) >= 0.6:
        return None
    left = math.floor(min(token.left for line in sub_lines for token in line))
    right = math.ceil(max(token.right for line in sub_lines for token in line))
    if right - left < 40:
        return None
    coverage = np.zeros(right - left + 1, dtype=np.int16)
    for line in sub_lines:
        row_coverage = np.zeros_like(coverage, dtype=bool)
        for token in line:
            token_left = max(0, math.floor(token.left) - left)
            token_right = min(len(coverage), math.ceil(token.right) - left)
            row_coverage[token_left:token_right] = True
        coverage += row_coverage

    low_coverage = coverage <= max(0, int(len(sub_lines) * 0.15))
    typical_char_w = _estimate_character_width(sub_lines)
    min_gutter = max(12.0, typical_char_w * 1.5)

    candidates = []
    for run_left, run_right in _gap_runs(low_coverage):
        gutter = run_right - run_left
        if gutter < min_gutter:
            continue
        separator = left + (run_left + run_right) / 2
        relative = (separator - left) / (right - left)
        if not 0.03 < relative < 0.97:
            continue
        left_support = sum(
            any(token.center_x < separator for token in line) for line in sub_lines
        ) / len(sub_lines)
        right_support = sum(
            any(token.center_x > separator for token in line) for line in sub_lines
        ) / len(sub_lines)
        if min(left_support, right_support) >= 0.65:
            candidates.append((separator, gutter))

    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:6]
    candidates.sort()
    best: BorderlessTable | None = None
    for mask in range(1, 1 << len(candidates)):
        separators = tuple(
            candidates[index][0]
            for index in range(len(candidates))
            if mask & (1 << index)
        )
        if len(separators) > 7:
            continue
        score = _evaluate_separators(sub_lines, separators)
        if score is None:
            continue
        if not all(_is_valid_table_row(line, separators) for line in sub_lines):
            continue
        alignment_score = _column_alignment_score(
            sub_lines,
            separators,
            typical_char_w,
        )
        if alignment_score is None:
            continue
        score += (len(sub_lines) * 2.0) + alignment_score + sum(
            candidates[index][1]
            for index in range(len(candidates))
            if mask & (1 << index)
        ) / max(1, right - left)
        if best is None or score > best.score:
            best = BorderlessTable(start, end, separators, score)
    return best


def score_table_windows(lines: list[list[Token]]) -> tuple[BorderlessTable, ...]:
    """Score every candidate table span once so ranges can be queried cheaply."""
    scored = []
    for start in range(len(lines) - 2):
        for end in range(start + 3, min(len(lines) + 1, start + MAX_TABLE_WINDOW)):
            window = _score_table_window(lines[start:end], start, end)
            if window is not None:
                scored.append(window)
    return tuple(scored)


def best_table_within(
    windows: tuple[BorderlessTable, ...], start: int, end: int
) -> BorderlessTable | None:
    contained = [
        window
        for window in windows
        if window.start >= start and window.end <= end
    ]
    return max(contained, key=lambda window: window.score, default=None)


def merge_adjacent_tables(
    tables: list[BorderlessTable], lines: list[list[Token]]
) -> list[BorderlessTable]:
    """Rejoin a long table that the bounded window search reported in parts."""
    merged: list[BorderlessTable] = []
    for table in tables:
        previous = merged[-1] if merged else None
        if previous is None or previous.end != table.start:
            merged.append(table)
            continue
        if len(previous.separators) != len(table.separators):
            merged.append(table)
            continue
        span = lines[previous.start : table.end]
        tolerance = max(8.0, _estimate_character_width(span))
        columns = list(zip(previous.separators, table.separators))
        if any(abs(above - below) > tolerance for above, below in columns):
            merged.append(table)
            continue
        separators = tuple((above + below) / 2 for above, below in columns)
        if _evaluate_separators(span, separators) is None or not all(
            _is_valid_table_row(line, separators) for line in span
        ):
            merged.append(table)
            continue
        merged[-1] = BorderlessTable(
            previous.start,
            table.end,
            separators,
            previous.score + table.score,
        )
    return merged


def _estimate_character_width(lines: list[list[Token]]) -> float:
    estimates = []
    for line in lines:
        for token in line:
            visible = re.sub(r"\s+", "", token.text)
            if len(visible) >= 2:
                estimates.append((token.right - token.left) / len(visible))
    return median(estimates) if estimates else 8.0


def _row_background(rgb: np.ndarray, y: float) -> np.ndarray:
    center = round(y)
    top = max(0, center - 2)
    bottom = min(rgb.shape[0], center + 3)
    return np.median(rgb[top:bottom].reshape(-1, 3), axis=0)


def _page_background(rgb: np.ndarray) -> np.ndarray:
    border = max(1, min(rgb.shape[:2]) // 50)
    samples = np.concatenate(
        (
            rgb[:border].reshape(-1, 3),
            rgb[-border:].reshape(-1, 3),
            rgb[:, :border].reshape(-1, 3),
            rgb[:, -border:].reshape(-1, 3),
        )
    )
    return np.median(samples, axis=0)


def _color_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(float) - right.astype(float))))


def _looks_like_code_line(line: list[Token]) -> bool:
    text = join_line(line).strip()
    if not text:
        return False
    if text.startswith(("#", "//", "/*", "* ", "$", "./", "../", "~/")):
        return True
    if "/" in text or "\\" in text:
        return True
    return bool(re.search(r"[=\[\]{}()]|(?:^|\s)--?[A-Za-z]", text))


def detect_code_blocks(
    lines: list[list[Token]], image: Image.Image
) -> tuple[CodeBlock, ...]:
    """Detect text groups rendered on a distinct, continuous panel background."""
    if not lines:
        return ()

    rgb = np.asarray(image)
    page_background = _page_background(rgb)
    groups: list[list[tuple[int, np.ndarray]]] = []
    for index, line in enumerate(lines):
        signature = _row_background(rgb, _line_center(line))
        if _color_distance(signature, page_background) < 4:
            continue
        if groups:
            previous_index, previous_signature = groups[-1][-1]
            midpoint = (_line_center(lines[previous_index]) + _line_center(line)) / 2
            midpoint_signature = _row_background(rgb, midpoint)
            if (
                index == previous_index + 1
                and _color_distance(signature, previous_signature) <= 3
                and _color_distance(midpoint_signature, signature) <= 3
            ):
                groups[-1].append((index, signature))
                continue
        groups.append([(index, signature)])

    code_blocks = []
    for group in groups:
        group_lines = [lines[index] for index, _ in group]
        if not any(_looks_like_code_line(line) for line in group_lines):
            continue
        code_blocks.append(
            CodeBlock(
                top=min(token.top for line in group_lines for token in line),
                bottom=max(token.bottom for line in group_lines for token in line),
                base_left=min(token.left for line in group_lines for token in line),
                character_width=max(3.0, _estimate_character_width(group_lines)),
            )
        )
    return tuple(code_blocks)


def _code_block_for_line(
    line: list[Token], code_blocks: tuple[CodeBlock, ...]
) -> CodeBlock | None:
    center = _line_center(line)
    return next(
        (block for block in code_blocks if block.top <= center <= block.bottom),
        None,
    )


def recover_code_continuations(
    lines: list[list[Token]],
    code_blocks: tuple[CodeBlock, ...],
    image: Image.Image,
) -> list[list[Token]]:
    """Recover visually supported trailing backslashes in code panels."""
    if not lines or not code_blocks:
        return lines

    mask = _foreground_mask(image).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, 8
    )
    components = [
        (component, *map(int, stats[component]))
        for component in range(1, component_count)
    ]
    recovered = [list(line) for line in lines]

    for block in code_blocks:
        indices = [
            index
            for index, line in enumerate(recovered)
            if _code_block_for_line(line, (block,)) is not None
        ]
        for index, next_index in pairwise(indices):
            line = sorted(recovered[index], key=lambda token: token.left)
            next_line = sorted(
                recovered[next_index], key=lambda token: token.left
            )
            if not line or not next_line:
                continue
            if join_line(line, RegionRole.CODE).rstrip().endswith("\\"):
                continue
            line_center = _line_center(line)
            next_center = _line_center(next_line)
            if next_center - line_center > block.character_width * 4:
                continue
            if next_line[0].left <= block.base_left + block.character_width * 0.5:
                continue

            terminal = line[-1]
            terminal_text = re.sub(r"\s+", "", terminal.text)
            suspect = bool(
                terminal_text
                and len(terminal_text) <= 3
                and all(character in DASH_LIKE for character in terminal_text)
            )
            content_right = line[-2].right if suspect and len(line) >= 2 else terminal.right
            line_top = min(token.top for token in line)
            line_bottom = max(token.bottom for token in line)
            line_height = max(1.0, line_bottom - line_top)

            candidates = []
            for component, left, top, width, height, area in components:
                right = left + width
                bottom = top + height
                if not (
                    content_right + block.character_width * 0.1 <= left
                    <= content_right + block.character_width * 2
                    and line_top - line_height * 0.1 <= top
                    and bottom <= line_bottom + line_height * 0.1
                    and line_height * 0.35 <= height <= line_height
                    and 0.2 <= width / max(1, height) <= 0.9
                    and area >= 5
                ):
                    continue
                overlapping = [
                    token
                    for token in line
                    if not (
                        token.right < left
                        or token.left > right
                        or token.bottom < top
                        or token.top > bottom
                    )
                ]
                if any(token is not terminal for token in overlapping):
                    continue
                component_pixels = labels[
                    top : top + height, left : left + width
                ] == component
                ys, xs = np.nonzero(component_pixels)
                if len(xs) < 5 or np.std(xs) == 0 or np.std(ys) == 0:
                    continue
                correlation = float(np.corrcoef(xs, ys)[0, 1])
                if correlation < 0.65:
                    continue
                candidates.append((left - content_right, left, top, right, bottom))

            if not candidates:
                continue
            _, left, top, right, bottom = min(candidates)
            if suspect:
                recovered[index] = [
                    token for token in recovered[index] if token is not terminal
                ]
            recovered[index].append(
                Token(
                    "\\",
                    left,
                    top,
                    right,
                    bottom,
                    1.0,
                    "continuation",
                )
            )
    return recovered


def recover_code_delimiters(
    lines: list[list[Token]], code_blocks: tuple[CodeBlock, ...]
) -> list[list[Token]]:
    """Recover tightly constrained missing delimiters inside visual code panels."""
    recovered = [list(line) for line in lines]
    for block in code_blocks:
        indices = [
            index
            for index, line in enumerate(recovered)
            if _code_block_for_line(line, (block,)) is not None
        ]
        bracket_balance = 0
        parenthesis_balance = 0
        previous_left: float | None = None
        for position, index in enumerate(indices):
            line = recovered[index]
            line_left = min(token.left for token in line)
            text = join_line(line)
            stripped = text.strip()
            if (
                len(line) == 1
                and len(stripped) == 1
                # Only repair an implausible lone glyph, never a delimiter the
                # recognizer already read.
                and stripped not in "([{)]}"
                and line_left <= block.base_left + block.character_width * 0.75
                and previous_left is not None
                and previous_left - line_left >= block.character_width * 1.5
            ):
                expected = (
                    "]"
                    if bracket_balance > 0
                    else ")" if parenthesis_balance > 0 else None
                )
                if expected is None or stripped == expected:
                    expected = None
            else:
                expected = None
            if expected is not None:
                token = line[0]
                recovered[index] = [
                    Token(
                        expected,
                        token.left,
                        token.top,
                        token.right,
                        token.bottom,
                        token.confidence,
                        "delimiter",
                    )
                ]
                text = expected

            stripped = text.strip()
            if (
                stripped[:1] in {'"', "'"}
                and len(re.findall(rf"(?<!\\){re.escape(stripped[0])}", stripped))
                % 2
                == 1
                and parenthesis_balance > 0
                and position + 1 < len(indices)
                and not stripped.endswith("\\")
            ):
                next_line = recovered[indices[position + 1]]
                next_left = min(token.left for token in next_line)
                if abs(next_left - line_left) <= block.character_width * 0.75:
                    next_text = join_line(next_line).lstrip()
                    suffix = stripped[0]
                    if not next_text.startswith((stripped[0], ")", "]", "}")):
                        suffix += ","
                    last = max(line, key=lambda token: token.right)
                    recovered[index].append(
                        Token(
                            suffix,
                            last.right,
                            last.top,
                            last.right + block.character_width * len(suffix),
                            last.bottom,
                            last.confidence,
                            "delimiter",
                        )
                    )
                    text += suffix

            bracket_balance += text.count("[") - text.count("]")
            bracket_balance = max(0, bracket_balance)
            parenthesis_balance += text.count("(") - text.count(")")
            parenthesis_balance = max(0, parenthesis_balance)
            previous_left = line_left
    return recovered


def _region_metrics(lines: list[list[Token]]) -> RegionMetrics:
    character_width = max(3.0, _estimate_character_width(lines))
    line_height = median(_line_height(line) for line in lines)
    centers = [_line_center(line) for line in lines]
    gaps = [right - left for left, right in pairwise(centers)]
    return RegionMetrics(
        base_left=min(token.left for line in lines for token in line),
        character_width=character_width,
        line_height=line_height,
        line_gap=median(gaps) if gaps else line_height * 1.3,
    )


def _classify_plain_regions(
    lines: list[list[Token]],
    start: int,
    end: int,
    block: int,
    code_blocks: tuple[CodeBlock, ...],
) -> list[LayoutRegion]:
    if start >= end:
        return []
    span = lines[start:end]
    metrics = _region_metrics(span)
    continuation_gap = max(metrics.line_gap * 1.4, metrics.line_height * 1.8)
    roles: list[RegionRole] = []
    active_list: tuple[float, float] | None = None
    for line in span:
        ordered = sorted(line, key=lambda token: token.left)
        center = _line_center(ordered)
        if _code_block_for_line(ordered, code_blocks) is not None:
            roles.append(RegionRole.CODE)
            active_list = None
            continue

        first_text = ordered[0].text.strip()
        if _is_list_marker(first_text):
            content_left = (
                ordered[1].left
                if len(ordered) >= 2
                else ordered[0].left + metrics.character_width * 2
            )
            active_list = (content_left, center)
            roles.append(RegionRole.LIST)
            continue

        if active_list is not None:
            content_left, previous_center = active_list
            if (
                center - previous_center <= continuation_gap
                and abs(ordered[0].left - content_left)
                <= max(8.0, metrics.character_width * 0.8)
            ):
                active_list = (content_left, center)
                roles.append(RegionRole.LIST)
                continue
        active_list = None
        roles.append(RegionRole.PROSE)

    regions = []
    region_start = start
    role = roles[0]
    for offset, next_role in enumerate(roles[1:], start=1):
        if next_role == role:
            continue
        regions.append(
            LayoutRegion(role, region_start, start + offset, block)
        )
        region_start = start + offset
        role = next_role
    regions.append(LayoutRegion(role, region_start, end, block))
    return regions


def analyze_layout(
    lines: list[list[Token]],
    code_blocks: tuple[CodeBlock, ...],
) -> tuple[LayoutRegion, ...]:
    """Assign mutually exclusive, locally measured roles before formatting."""
    if not lines:
        return ()

    table_spans: list[BorderlessTable] = []
    windows = score_table_windows(lines)

    def collect_tables(start: int, end: int) -> None:
        if end - start < 3:
            return
        table = best_table_within(windows, start, end)
        if table is None:
            return
        table_start = table.start
        table_end = table.end
        table_lines = lines[table_start:table_end]
        code_rows = sum(
            _code_block_for_line(line, code_blocks) is not None
            for line in table_lines
        )
        if code_rows / len(table_lines) >= 0.5:
            return
        collect_tables(start, table_start)
        table_spans.append(
            BorderlessTable(
                table_start,
                table_end,
                table.separators,
                table.score,
            )
        )
        collect_tables(table_end, end)

    collect_tables(0, len(lines))
    table_spans.sort(key=lambda table: table.start)
    table_spans = merge_adjacent_tables(table_spans, lines)

    visual_blocks = _split_line_blocks(lines)
    line_blocks = [0] * len(lines)
    for block, (start, end) in enumerate(visual_blocks):
        for index in range(start, end):
            line_blocks[index] = block

    regions: list[LayoutRegion] = []
    cursor = 0

    def append_plain(start: int, end: int) -> None:
        while start < end:
            block = line_blocks[start]
            block_end = min(
                end,
                next(
                    (
                        index
                        for index in range(start + 1, end)
                        if line_blocks[index] != block
                    ),
                    end,
                ),
            )
            regions.extend(
                _classify_plain_regions(
                    lines, start, block_end, block, code_blocks
                )
            )
            start = block_end

    for table in table_spans:
        append_plain(cursor, table.start)
        block = min(line_blocks[table.start : table.end])
        regions.append(
            LayoutRegion(
                RegionRole.TABLE,
                table.start,
                table.end,
                block,
                table.separators,
            )
        )
        cursor = table.end
    append_plain(cursor, len(lines))
    return tuple(regions)


def _code_indent_anchors(
    lines: list[list[Token]], code_blocks: tuple[CodeBlock, ...]
) -> dict[CodeBlock, tuple[float, ...]]:
    """Cluster nearly equal code-line starts before deriving indentation."""
    anchors_by_block: dict[CodeBlock, tuple[float, ...]] = {}
    for block in code_blocks:
        lefts = sorted(
            min(token.left for token in line)
            for line in lines
            if _code_block_for_line(line, (block,)) is not None
        )
        if not lefts:
            continue
        tolerance = max(2.0, block.character_width * 0.75)
        clusters: list[list[float]] = []
        for left in lefts:
            if (
                clusters
                and abs(left - median(clusters[-1])) <= tolerance
            ):
                clusters[-1].append(left)
            else:
                clusters.append([left])
        anchors_by_block[block] = tuple(
            float(median(cluster)) for cluster in clusters
        )
    return anchors_by_block


def format_plain_lines(
    lines: list[list[Token]],
    code_blocks: tuple[CodeBlock, ...] = (),
    role: RegionRole | None = None,
    base_left_override: float | None = None,
) -> str:
    if not lines:
        return ""
    blocks = _split_line_blocks(lines)
    code_indent_anchors = _code_indent_anchors(lines, code_blocks)
    formatted_blocks = []
    for start, end in blocks:
        block_lines = lines[start:end]
        character_width = max(3.0, _estimate_character_width(block_lines))
        base_left = (
            base_left_override
            if base_left_override is not None
            else min(min(token.left for token in line) for line in block_lines)
        )
        centers = [_line_center(line) for line in block_lines]
        line_gaps = [right - left for left, right in pairwise(centers)]
        continuation_gap = max(
            (median(line_gaps) if line_gaps else 0) * 1.4,
            median(_line_height(line) for line in block_lines) * 1.8,
        )
        output = []
        active_list: tuple[int, float, float, int] | None = None
        for line in block_lines:
            line = sorted(line, key=lambda token: token.left)
            text = join_line(line, role)
            first_text = line[0].text.strip()
            starts_bullet = first_text[:1] in LIST_MARKERS
            starts_ordered = bool(
                ORDERED_LIST_MARKER_RE.fullmatch(first_text)
            )
            starts_list = starts_bullet or starts_ordered
            content_left: float | None = None
            hanging_width = 2
            if starts_bullet:
                if first_text in LIST_MARKERS:
                    body = join_line(line[1:], role)
                    content_left = (
                        line[1].left
                        if len(line) >= 2
                        else line[0].right + character_width
                    )
                else:
                    replacement = Token(
                        first_text[1:].lstrip(),
                        line[0].left,
                        line[0].top,
                        line[0].right,
                        line[0].bottom,
                        line[0].confidence,
                        line[0].source,
                    )
                    body = join_line([replacement, *line[1:]], role)
                    content_left = line[0].left + character_width * 2
                text = "- " + body if body else "-"
            elif starts_ordered:
                content_left = (
                    line[1].left
                    if len(line) >= 2
                    else line[0].right + character_width
                )
                hanging_width = len(first_text) + 1
            code_block = _code_block_for_line(line, code_blocks)
            if code_block:
                anchors = code_indent_anchors.get(
                    code_block, (code_block.base_left,)
                )
                line_left = line[0].left
                anchor = min(anchors, key=lambda value: abs(value - line_left))
                indent = round(
                    (anchor - anchors[0])
                    / code_block.character_width
                )
            else:
                raw_indent = round((line[0].left - base_left) / character_width)
                if raw_indent >= 2:
                    snapped = int(round(raw_indent / 4) * 4)
                    indent = (
                        snapped if abs(snapped - raw_indent) <= 1 else raw_indent
                    )
                else:
                    indent = 0

            line_center = _line_center(line)
            if starts_list and content_left is not None:
                active_list = (
                    indent,
                    content_left,
                    line_center,
                    hanging_width,
                )
            elif active_list is not None:
                (
                    list_indent,
                    list_content_left,
                    previous_center,
                    hanging_width,
                ) = active_list
                if (
                    line_center - previous_center <= continuation_gap
                    and abs(line[0].left - list_content_left)
                    <= max(8.0, character_width * 0.8)
                ):
                    indent = list_indent + hanging_width
                    active_list = (
                        list_indent,
                        list_content_left,
                        line_center,
                        hanging_width,
                    )
                else:
                    active_list = None
            output.append(" " * max(0, indent) + text)
        formatted_blocks.append("\n".join(output))
    return "\n\n".join(block for block in formatted_blocks if block)


def _format_table_lines(
    lines: list[list[Token]], separators: tuple[float, ...]
) -> str:
    cells_by_row = []
    for line in lines:
        cells = [[] for _ in range(len(separators) + 1)]
        for token in line:
            cells[bisect.bisect_right(separators, token.center_x)].append(token)
        cells_by_row.append(cells)
    cells_by_row = _normalize_table_identifier_case(
        _recover_table_glyphs(cells_by_row)
    )
    rows = [
        [cell_text(cell) for cell in cells]
        for cells in cells_by_row
    ]
    return markdown_table(rows)


def format_layout_regions(
    lines: list[list[Token]],
    regions: tuple[LayoutRegion, ...],
    code_blocks: tuple[CodeBlock, ...],
) -> str:
    block_bases: dict[int, float] = {}
    for region in regions:
        if region.role == RegionRole.TABLE:
            continue
        left = min(
            token.left
            for line in lines[region.start : region.end]
            for token in line
        )
        block_bases[region.block] = min(
            block_bases.get(region.block, left), left
        )

    block_outputs: list[str] = []
    current_block: int | None = None
    current_output = ""
    previous_role: RegionRole | None = None
    for region in regions:
        if region.role == RegionRole.TABLE:
            text = _format_table_lines(
                lines[region.start : region.end], region.separators
            )
        else:
            text = format_plain_lines(
                lines[region.start : region.end],
                code_blocks,
                region.role,
                (
                    block_bases.get(region.block)
                    if region.role == RegionRole.LIST
                    else None
                ),
            )
        if not text:
            continue
        if current_block != region.block:
            if current_output:
                block_outputs.append(current_output)
            current_block = region.block
            current_output = text
            previous_role = region.role
            continue
        separator = (
            "\n\n"
            if RegionRole.TABLE in {previous_role, region.role}
            else "\n"
        )
        current_output += separator + text
        previous_role = region.role
    if current_output:
        block_outputs.append(current_output)
    return "\n\n".join(block_outputs)


def format_document(tokens: list[Token], image: Image.Image) -> str:
    grid = detect_grid(image)
    tokens = recover_list_markers(tokens, image)
    tokens = recover_horizontal_punctuation(tokens, image)
    if grid:
        tokens = merge_colored_icons(tokens, image, grid)
        return _format_grid(tokens, grid)

    lines = cluster_lines(tokens)
    code_blocks = detect_code_blocks(lines, image)
    lines = recover_code_continuations(lines, code_blocks, image)
    lines = recover_code_delimiters(lines, code_blocks)
    regions = analyze_layout(lines, code_blocks)
    return format_layout_regions(lines, regions, code_blocks)


def normalize_output(text: str) -> str:
    """Apply lossless Unicode and whitespace canonicalization."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        character
        for character in text
        if character in {"\n", "\t"}
        or not unicodedata.category(character).startswith("C")
    )
    lines = [line.rstrip() for line in text.splitlines()]
    normalized = []
    blank = False
    for line in lines:
        if not line:
            if normalized and not blank:
                normalized.append("")
            blank = True
            continue
        normalized.append(line)
        blank = False
    return "\n".join(normalized).strip()


def copy_to_clipboard(text: str) -> None:
    subprocess.run(
        ["/usr/bin/pbcopy"],
        input=text,
        text=True,
        check=True,
    )


def notify_copied() -> None:
    try:
        subprocess.run(
            [
                "/usr/bin/osascript",
                "-e",
                'display notification "" with title "OCR copied"',
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        # Notifications are optional and must not turn a successful copy into
        # a failed OCR run.
        pass


def process_ocr(image_path: str | Path, copy: bool = True) -> str:
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise OCRError(f"image does not exist: {path}")
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except Exception as exc:
        raise OCRError(f"cannot open image: {exc}") from exc

    tokens = recognize(image)
    if not tokens:
        raise OCRError("no text was detected")
    output = normalize_output(format_document(tokens, image))
    if not output:
        raise OCRError("text was detected but layout reconstruction was empty")
    if copy:
        try:
            copy_to_clipboard(output)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ClipboardError(
                f"OCR succeeded, but clipboard copy failed: {exc}",
                output,
            ) from exc
        else:
            notify_copied()
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="screenshot image to OCR")
    parser.add_argument(
        "--no-clipboard",
        action="store_true",
        help="print without changing the clipboard (for tests)",
    )
    args = parser.parse_args(argv)
    no_clipboard = args.no_clipboard or os.environ.get("SMART_OCR_NO_CLIPBOARD") == "1"
    try:
        output = process_ocr(args.image, copy=not no_clipboard)
    except ClipboardError as exc:
        print(exc.output)
        print(f"ocr-smart: {exc}", file=sys.stderr)
        return 1
    except OCRError as exc:
        print(f"ocr-smart: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())