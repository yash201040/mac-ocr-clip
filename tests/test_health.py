"""Self-contained functional health checks for Smart OCR."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module, metadata
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ocr_smart = import_module("ocr_smart")
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/SFNSMono.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
)


def load_test_font(size: int = 36) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(path, size)
    raise AssertionError("no suitable built-in macOS test font was found")


def format_lines(
    lines: list[list[ocr_smart.Token]],
    code_blocks: tuple[ocr_smart.CodeBlock, ...] = (),
) -> str:
    """Format clustered lines through the shipping layout pipeline."""
    regions = ocr_smart.analyze_layout(lines, code_blocks)
    return ocr_smart.format_layout_regions(lines, regions, code_blocks)


def detect_table(
    lines: list[list[ocr_smart.Token]],
) -> ocr_smart.BorderlessTable | None:
    windows = ocr_smart.score_table_windows(lines)
    return ocr_smart.best_table_within(windows, 0, len(lines))


class FormattingTests(unittest.TestCase):
    def test_markdown_cells_are_escaped(self) -> None:
        output = ocr_smart.markdown_table(
            [["Name", "Value"], ["path", r"one\two|three"]]
        )
        self.assertEqual(
            output,
            "| Name | Value |\n"
            "| --- | --- |\n"
            r"| path | one\\two\|three |",
        )

    def test_borderless_table_reconstruction(self) -> None:
        values = (
            ("Name", "Count", "Status"),
            ("Alpha", "10", "Ready"),
            ("Beta", "20", "Ready"),
            ("Gamma", "30", "Done"),
        )
        tokens = []
        for row, cells in enumerate(values):
            top = 20 + row * 42
            for text, left in zip(cells, (20, 230, 440)):
                tokens.append(
                    ocr_smart.Token(
                        text=text,
                        left=left,
                        top=top,
                        right=left + max(30, len(text) * 18),
                        bottom=top + 24,
                    )
                )

        lines = ocr_smart.cluster_lines(tokens)
        self.assertEqual(
            format_lines(lines),
            "| Name | Count | Status |\n"
            "| --- | --- | --- |\n"
            "| Alpha | 10 | Ready |\n"
            "| Beta | 20 | Ready |\n"
            "| Gamma | 30 | Done |",
        )

    def test_colored_icon_recovery(self) -> None:
        image = Image.new("RGB", (300, 150), "white")
        ImageDraw.Draw(image).line(
            ((40, 44), (48, 52), (62, 30)),
            fill=(0, 180, 0),
            width=5,
        )
        tokens = [ocr_smart.Token("v", 38, 28, 64, 54)]
        grid = ocr_smart.Grid((0, 150, 299), (0, 75, 149))

        recovered = ocr_smart.merge_colored_icons(tokens, image, grid)

        self.assertEqual([token.text for token in recovered], ["✅"])

    def test_colored_status_dot_is_not_invented_as_an_icon(self) -> None:
        image = Image.new("RGB", (300, 150), "white")
        ImageDraw.Draw(image).ellipse((40, 30, 58, 48), fill=(0, 180, 0))
        grid = ocr_smart.Grid((0, 150, 299), (0, 75, 149))

        self.assertEqual(
            ocr_smart.merge_colored_icons([], image, grid),
            [],
        )

    def test_isolated_hyphen_recovery(self) -> None:
        image = Image.new("RGB", (300, 100), "white")
        ImageDraw.Draw(image).line((115, 45, 126, 45), fill="black", width=3)
        tokens = [
            ocr_smart.Token("alpha", 20, 30, 100, 60),
            ocr_smart.Token("beta", 150, 30, 230, 60),
        ]

        recovered = ocr_smart.recover_horizontal_punctuation(tokens, image)

        self.assertEqual(ocr_smart.join_line(recovered), "alpha - beta")

    def test_text_to_the_right_of_a_grid_stays_after_it(self) -> None:
        grid = ocr_smart.Grid((100, 200, 300), (100, 150, 200))
        tokens = [
            ocr_smart.Token("H1", 120, 110, 150, 130),
            ocr_smart.Token("H2", 220, 110, 250, 130),
            ocr_smart.Token("A", 120, 160, 140, 180),
            ocr_smart.Token("B", 220, 160, 240, 180),
            ocr_smart.Token("SIDE", 320, 160, 370, 180),
        ]

        self.assertEqual(
            ocr_smart._format_grid(tokens, grid),
            "| H1 | H2 |\n"
            "| --- | --- |\n"
            "| A | B |\n\n"
            "SIDE",
        )

    def test_contained_duplicate_token_suppression(self) -> None:
        tokens = [
            ocr_smart.Token("•", 85, 559, 99, 594),
            ocr_smart.Token("1", 90, 574, 98, 580),
            ocr_smart.Token("Copy", 114, 559, 181, 594),
        ]
        cleaned = ocr_smart._clean_tokens(tokens)
        self.assertEqual([token.text for token in cleaned], ["•", "Copy"])

    def test_path_and_option_spacing_in_join_line(self) -> None:
        tokens = [
            ocr_smart.Token("PYTHONDONTWRITEBYTECODE=1", 10, 20, 150, 40),
            ocr_smart.Token("./.venv/bin/python", 160, 20, 280, 40),
        ]
        self.assertEqual(
            ocr_smart.join_line(tokens),
            "PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python",
        )

    def test_relative_path_prefix_uses_code_context(self) -> None:
        for glyph in ("-", "一"):
            with self.subTest(glyph=glyph):
                tokens = [
                    ocr_smart.Token(glyph, 10, 10, 15, 30),
                    ocr_smart.Token("/setup.sh", 22, 10, 100, 30),
                ]
                self.assertEqual(
                    ocr_smart.join_line(tokens, ocr_smart.RegionRole.CODE),
                    "./setup.sh",
                )
                self.assertEqual(
                    ocr_smart.join_line(tokens, ocr_smart.RegionRole.LIST),
                    f"{glyph} /setup.sh",
                )

    def test_tight_filename_fragments_use_geometry_conservatively(self) -> None:
        filename = [
            ocr_smart.Token("smartocr.", 10, 10, 110, 30),
            ocr_smart.Token("log", 118, 10, 150, 30),
        ]
        prose = [
            ocr_smart.Token("mo.", 10, 10, 40, 30),
            ocr_smart.Token("for", 43, 10, 75, 30),
        ]
        self.assertEqual(ocr_smart.join_line(filename), "smartocr.log")
        self.assertEqual(ocr_smart.join_line(prose), "mo. for")

    def test_crop_consensus_rejects_isolated_glyph_changes(self) -> None:
        self.assertFalse(
            ocr_smart._accept_crop_consensus(
                ocr_smart.Token("o", 0, 0, 10, 10),
                "0",
                3,
                0.99,
                2,
                Counter(),
            )
        )
        self.assertTrue(
            ocr_smart._accept_crop_consensus(
                ocr_smart.Token("OCRis", 0, 0, 50, 10),
                "OCR is",
                3,
                0.98,
                1,
                Counter(),
            )
        )
        self.assertTrue(
            ocr_smart._accept_crop_consensus(
                ocr_smart.Token("Ul", 0, 0, 20, 10),
                "UI",
                3,
                0.95,
                2,
                Counter(),
            )
        )

    def test_role_specific_normalization_preserves_prose_typography(self) -> None:
        tokens = [
            ocr_smart.Token("tool.py", 10, 10, 70, 30),
            ocr_smart.Token("—", 80, 10, 90, 30),
            ocr_smart.Token("./path", 100, 10, 160, 30),
        ]
        self.assertEqual(
            ocr_smart.join_line(tokens, ocr_smart.RegionRole.LIST),
            "tool.py — ./path",
        )

    def test_output_normalization_is_lossless_and_canonical(self) -> None:
        self.assertEqual(
            ocr_smart.normalize_output("Cafe\u0301  \r\n\r\n\r\nnext\u200b\r\n"),
            "Café\n\nnext",
        )

    def test_table_region_can_cross_visual_block_boundaries(self) -> None:
        lines = [
            [
                ocr_smart.Token("Name", 20, 10, 80, 30),
                ocr_smart.Token("Value", 220, 10, 280, 30),
            ],
            [
                ocr_smart.Token("Alpha", 20, 100, 80, 120),
                ocr_smart.Token("10", 220, 100, 250, 120),
            ],
            [
                ocr_smart.Token("Beta", 20, 140, 70, 160),
                ocr_smart.Token("20", 220, 140, 250, 160),
            ],
        ]
        self.assertGreater(len(ocr_smart._split_line_blocks(lines)), 1)
        regions = ocr_smart.analyze_layout(lines, ())
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].role, ocr_smart.RegionRole.TABLE)

    def test_borderless_table_excludes_leading_caption(self) -> None:
        lines = [
            [ocr_smart.Token("Table", 10, 10, 50, 30), ocr_smart.Token("1", 60, 10, 70, 30)],
            [ocr_smart.Token("H1", 20, 50, 40, 70), ocr_smart.Token("H2", 120, 50, 140, 70), ocr_smart.Token("H3", 220, 50, 240, 70)],
            [ocr_smart.Token("A1", 20, 90, 40, 110), ocr_smart.Token("A2", 120, 90, 140, 110), ocr_smart.Token("A3", 220, 90, 240, 110)],
            [ocr_smart.Token("B1", 20, 130, 40, 150), ocr_smart.Token("B2", 120, 130, 140, 150), ocr_smart.Token("B3", 220, 130, 240, 150)],
            [ocr_smart.Token("C1", 20, 170, 40, 190), ocr_smart.Token("C2", 120, 170, 140, 190), ocr_smart.Token("C3", 220, 170, 240, 190)],
        ]
        table = detect_table(lines)
        self.assertIsNotNone(table)
        self.assertEqual(table.start, 1)
        self.assertEqual(table.end, 5)

    def test_three_row_two_column_feature_grid(self) -> None:
        lines = [
            [
                ocr_smart.Token("Memory", 20, 20, 90, 40),
                ocr_smart.Token("Storage", 230, 20, 300, 40),
            ],
            [
                ocr_smart.Token("Power", 21, 60, 80, 80),
                ocr_smart.Token("Keyboard", 231, 60, 315, 80),
            ],
            [
                ocr_smart.Token("Adapter", 19, 100, 85, 120),
                ocr_smart.Token("Backlit", 229, 100, 295, 120),
            ],
        ]
        self.assertEqual(
            format_lines(lines),
            "| Memory | Storage |\n"
            "| --- | --- |\n"
            "| Power | Keyboard |\n"
            "| Adapter | Backlit |",
        )

    def test_grid_text_survives_when_it_cannot_form_a_table(self) -> None:
        grid = ocr_smart.Grid((100, 200, 300), (100, 150, 200))
        tokens = [
            ocr_smart.Token("Caption", 10, 60, 90, 80),
            ocr_smart.Token("H1", 120, 110, 180, 130),
            ocr_smart.Token("H2", 220, 110, 280, 130),
        ]
        output = ocr_smart._format_grid(tokens, grid)
        self.assertIn("H1 H2", output)
        self.assertIn("Caption", output)

    def test_recognized_closing_delimiter_is_never_replaced(self) -> None:
        block = ocr_smart.CodeBlock(
            top=0, bottom=200, base_left=20, character_width=10
        )
        recognized = [
            [ocr_smart.Token("print(", 20, 10, 80, 30)],
            [ocr_smart.Token('"[",', 60, 50, 120, 70)],
            [ocr_smart.Token(")", 20, 90, 30, 110)],
        ]
        self.assertEqual(
            [
                ocr_smart.join_line(line)
                for line in ocr_smart.recover_code_delimiters(recognized, (block,))
            ],
            ["print(", '"[",', ")"],
        )

        misread = [
            [ocr_smart.Token("values = [", 20, 10, 120, 30)],
            [ocr_smart.Token('"a",', 60, 50, 120, 70)],
            [ocr_smart.Token("_", 20, 90, 30, 110)],
        ]
        self.assertEqual(
            ocr_smart.join_line(
                ocr_smart.recover_code_delimiters(misread, (block,))[2]
            ),
            "]",
        )

    def test_page_level_recognition_preserves_shell_variable_case(self) -> None:
        tokens = [
            ocr_smart.Token("$MyVar/bin", 0, 0, 110, 20, 0.97),
            ocr_smart.Token("$MYVAR/src", 0, 30, 110, 50, 0.97),
            ocr_smart.Token("$HoME/tmp", 0, 60, 110, 80, 0.90),
            ocr_smart.Token("$HOME/lib", 0, 90, 110, 110, 0.99),
        ]
        with (
            patch.object(
                ocr_smart, "_modern_rapidocr_tokens", return_value=tokens
            ),
            patch.object(
                ocr_smart,
                "_refine_token_recognition",
                side_effect=lambda _image, values: values,
            ),
        ):
            recognized = ocr_smart.recognize(
                Image.new("RGB", (140, 130), "white")
            )
        self.assertEqual(
            [token.text for token in recognized],
            ["$MyVar/bin", "$MYVAR/src", "$HoME/tmp", "$HOME/lib"],
        )

    def test_ordered_and_bullet_lists_share_one_region_role(self) -> None:
        def region_roles(marker: str) -> list[ocr_smart.RegionRole]:
            lines = [
                [
                    ocr_smart.Token(marker, 20, 20, 40, 40),
                    ocr_smart.Token("First item text", 60, 20, 240, 40),
                ],
                [ocr_smart.Token("wrapped continuation", 60, 60, 250, 80)],
            ]
            return [
                region.role for region in ocr_smart.analyze_layout(lines, ())
            ]

        self.assertEqual(region_roles("1."), region_roles("•"))
        self.assertEqual(region_roles("1."), [ocr_smart.RegionRole.LIST])

    def test_degenerate_token_box_is_kept(self) -> None:
        tokens = [
            ocr_smart.Token("A", 10, 10, 10, 30, 0.9),
            ocr_smart.Token("Bee", 40, 10, 90, 30, 0.9),
            ocr_smart.Token("Cee", 100, 10, 150, 30, 0.9),
        ]
        self.assertEqual(
            [token.text for token in ocr_smart._clean_tokens(tokens)],
            ["A", "Bee", "Cee"],
        )

    def test_long_table_is_reported_as_one_table_with_one_header(self) -> None:
        rows = 45
        headers = ("Name", "Count", "Status")
        lines = []
        for row in range(rows):
            top = 20 + row * 40
            cells = (
                headers
                if row == 0
                else (f"Item{row}", str(row * 3), "Ready")
            )
            lines.append(
                [
                    ocr_smart.Token(text, 20 + column * 200, top, 90 + column * 200, top + 20)
                    for column, text in enumerate(cells)
                ]
            )
        regions = ocr_smart.analyze_layout(lines, ())
        tables = [
            region
            for region in regions
            if region.role == ocr_smart.RegionRole.TABLE
        ]
        self.assertEqual(len(tables), 1)
        self.assertEqual((tables[0].start, tables[0].end), (0, rows))
        output = format_lines(lines).splitlines()
        separators = [line for line in output if set(line) <= set("| -")]
        self.assertEqual(len(separators), 1)
        self.assertEqual(len(output), rows + 1)
        self.assertTrue(output[0].startswith("| Name | Count | Status |"))

    def test_layout_analysis_stays_linear_on_long_pages(self) -> None:
        def build(rows: int) -> list[list[ocr_smart.Token]]:
            return [
                [
                    ocr_smart.Token(f"c{column}r{row}", 20 + column * 160, 20 + row * 30, 90 + column * 160, 40 + row * 30)
                    for column in range(4)
                ]
                for row in range(rows)
            ]

        def elapsed(rows: int) -> float:
            lines = build(rows)
            start = time.perf_counter()
            ocr_smart.analyze_layout(lines, ())
            return time.perf_counter() - start

        small = max(elapsed(40), 1e-3)
        large = elapsed(160)
        self.assertLess(large / small, 12)

    def test_unaligned_prose_is_not_a_borderless_table(self) -> None:
        lines = [
            [
                ocr_smart.Token("First phrase", 20, 20, 120, 40),
                ocr_smart.Token("continues here", 230, 20, 340, 40),
            ],
            [
                ocr_smart.Token("Second phrase", 20, 60, 130, 80),
                ocr_smart.Token("has another ending", 300, 60, 450, 80),
            ],
            [
                ocr_smart.Token("Third phrase", 20, 100, 120, 120),
                ocr_smart.Token("ends elsewhere", 390, 100, 500, 120),
            ],
        ]
        self.assertIsNone(detect_table(lines))

    def test_block_aware_plain_lines_formatting(self) -> None:
        lines = [
            [ocr_smart.Token("Header", 10, 10, 80, 30)],
            [ocr_smart.Token("Block", 300, 100, 350, 120)],
            [ocr_smart.Token("Sub-item", 330, 130, 400, 150)],
        ]
        output = ocr_smart.format_plain_lines(lines)
        self.assertEqual(output, "Header\n\nBlock\n    Sub-item")

    def test_low_confidence_graphic_fragments_are_suppressed(self) -> None:
        raw_tokens = [
            ocr_smart.Token("Label", 40, 10, 100, 30, 0.99),
            ocr_smart.Token("O", 10, 10, 30, 30, 0.31),
            ocr_smart.Token("_", 110, 16, 118, 22, 0.61),
            ocr_smart.Token("T", 120, 17, 126, 22, 0.95),
            ocr_smart.Token("Value", 130, 10, 180, 30, 0.99),
        ]
        cleaned = ocr_smart._clean_tokens(raw_tokens)
        self.assertEqual([t.text for t in cleaned], ["Label", "Value"])

    def test_compact_semantic_operators_survive_fragment_cleanup(self) -> None:
        tokens = [
            ocr_smart.Token("left", 10, 10, 60, 30, 0.99),
            ocr_smart.Token("--", 70, 15, 90, 25, 0.70),
            ocr_smart.Token("||", 100, 15, 120, 25, 0.70),
            ocr_smart.Token("==", 130, 15, 150, 25, 0.70),
            ocr_smart.Token("right", 160, 10, 220, 30, 0.99),
        ]
        cleaned = ocr_smart._clean_tokens(tokens)
        self.assertEqual(
            [token.text for token in cleaned],
            ["left", "--", "||", "==", "right"],
        )

    def test_local_text_scale_preserves_a_small_text_row(self) -> None:
        tokens = [
            ocr_smart.Token(f"Body{index}", index * 70, 10, index * 70 + 60, 40)
            for index in range(5)
        ]
        tokens.extend(
            [
                ocr_smart.Token("A", 10, 100, 18, 110, 0.95),
                ocr_smart.Token("B", 30, 100, 38, 110, 0.95),
                ocr_smart.Token("C", 50, 100, 58, 110, 0.95),
            ]
        )
        cleaned = ocr_smart._clean_tokens(tokens)
        self.assertEqual([token.text for token in cleaned[-3:]], ["A", "B", "C"])

    def test_clean_shell_variable_rejects_case_only_crop_change(self) -> None:
        self.assertFalse(
            ocr_smart._accept_crop_consensus(
                ocr_smart.Token(
                    '"$path/bin"', 0, 0, 100, 20, confidence=0.800
                ),
                '"$PATH/bin"',
                3,
                0.999,
                3,
                Counter({"$path": 1}),
            )
        )

    def test_clipboard_failure_is_nonzero_and_keeps_recognized_output(
        self,
    ) -> None:
        failure = ocr_smart.ClipboardError(
            "OCR succeeded, but clipboard copy failed: unavailable",
            "recognized text",
        )
        stdout, stderr = StringIO(), StringIO()
        with (
            patch.object(ocr_smart, "process_ocr", side_effect=failure),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            status = ocr_smart.main(["unused.png"])
        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "recognized text\n")
        self.assertIn("clipboard copy failed", stderr.getvalue())

    def test_repeated_shell_variable_can_use_stronger_crop_evidence(self) -> None:
        self.assertTrue(
            ocr_smart._accept_crop_consensus(
                ocr_smart.Token(
                    '"$HoME/path"', 0, 0, 100, 20, confidence=0.970
                ),
                '"$HOME/path"',
                3,
                0.980,
                1,
                Counter({"$home": 4}),
            )
        )

    def test_uncertain_short_cell_survives_repeated_column_support(self) -> None:
        tokens = [
            ocr_smart.Token("0", 20, 10, 35, 30, 0.45),
            ocr_smart.Token("Alpha", 80, 10, 140, 30),
            ocr_smart.Token("1", 20, 50, 35, 70),
            ocr_smart.Token("Beta", 80, 50, 130, 70),
            ocr_smart.Token("2", 20, 90, 35, 110),
            ocr_smart.Token("Gamma", 80, 90, 140, 110),
        ]
        cleaned = ocr_smart._clean_tokens(tokens)
        self.assertIn("0", [token.text for token in cleaned])

    def test_repeated_list_markers_are_recovered_from_geometry(self) -> None:
        for background, foreground in (("#181818", "white"), ("white", "black")):
            for scale in (0.75, 1.0, 2.0):
                with self.subTest(background=background, scale=scale):
                    image = Image.new(
                        "RGB",
                        (round(220 * scale), round(120 * scale)),
                        background,
                    )
                    draw = ImageDraw.Draw(image)
                    tokens = []
                    for index, text in enumerate(("Alpha", "Beta", "Gamma")):
                        center_y = (20 + index * 40) * scale
                        draw.ellipse(
                            (
                                round(25 * scale),
                                round(center_y - 3 * scale),
                                round(31 * scale),
                                round(center_y + 3 * scale),
                            ),
                            fill=foreground,
                        )
                        tokens.append(
                            ocr_smart.Token(
                                text,
                                50 * scale,
                                center_y - 10 * scale,
                                120 * scale,
                                center_y + 10 * scale,
                            )
                        )

                    recovered = ocr_smart.recover_list_markers(tokens, image)
                    output = ocr_smart.format_plain_lines(
                        ocr_smart.cluster_lines(recovered)
                    )
                    self.assertEqual(output, "- Alpha\n- Beta\n- Gamma")

    def test_overlapping_glyph_is_corrected_to_a_repeated_list_marker(self) -> None:
        image = Image.new("RGB", (220, 130), "white")
        draw = ImageDraw.Draw(image)
        tokens = []
        for index, text in enumerate(("Alpha", "Beta", "Gamma")):
            center_y = 20 + index * 40
            draw.ellipse((25, center_y - 3, 31, center_y + 3), fill="black")
            marker_text = "C" if index == 1 else "•"
            marker_height = 6 if index == 1 else 20
            tokens.extend(
                (
                    ocr_smart.Token(
                        marker_text,
                        25,
                        center_y - marker_height / 2,
                        31,
                        center_y + marker_height / 2,
                        0.9,
                    ),
                    ocr_smart.Token(
                        text,
                        50,
                        center_y - 10,
                        120,
                        center_y + 10,
                    ),
                )
            )

        recovered = ocr_smart.recover_list_markers(tokens, image)
        self.assertNotIn("C", [token.text for token in recovered])
        output = ocr_smart.format_plain_lines(ocr_smart.cluster_lines(recovered))
        self.assertEqual(output, "- Alpha\n- Beta\n- Gamma")

    def test_wrapped_list_lines_use_hanging_indentation(self) -> None:
        lines = [
            [
                ocr_smart.Token("•", 20, 10, 28, 30),
                ocr_smart.Token("First line", 50, 10, 140, 30),
            ],
            [ocr_smart.Token("continuation", 51, 50, 160, 70)],
            [
                ocr_smart.Token("•", 20, 90, 28, 110),
                ocr_smart.Token("Second item", 50, 90, 150, 110),
            ],
        ]
        self.assertEqual(
            ocr_smart.format_plain_lines(lines),
            "- First line\n"
            "  continuation\n"
            "- Second item",
        )

    def test_wrapped_ordered_list_uses_marker_width_for_hanging_indent(self) -> None:
        lines = [
            [ocr_smart.Token("References", 0, 10, 100, 30)],
            [
                ocr_smart.Token("1.", 40, 50, 60, 70),
                ocr_smart.Token("First line", 70, 50, 170, 70),
            ],
            [ocr_smart.Token("continuation", 71, 90, 181, 110)],
        ]
        self.assertEqual(
            ocr_smart.format_plain_lines(lines),
            "References\n"
            "    1. First line\n"
            "       continuation",
        )

    def test_ordered_list_marker_column_is_not_a_borderless_table(self) -> None:
        lines = [
            [
                ocr_smart.Token(f"{index}.", 20, top, 40, top + 20),
                ocr_smart.Token(name, 60, top, 130, top + 20),
            ]
            for index, (top, name) in enumerate(
                ((10, "Alpha"), (50, "Beta"), (90, "Gamma")),
                start=1,
            )
        ]
        self.assertIsNone(detect_table(lines))

    def test_list_marker_column_is_not_a_borderless_table(self) -> None:
        lines = [
            [
                ocr_smart.Token("•", 20, 10, 28, 30),
                ocr_smart.Token("Alpha", 60, 10, 120, 30),
            ],
            [
                ocr_smart.Token("•", 20, 50, 28, 70),
                ocr_smart.Token("Beta", 60, 50, 110, 70),
            ],
            [
                ocr_smart.Token("•", 20, 90, 28, 110),
                ocr_smart.Token("Gamma", 60, 90, 120, 110),
            ],
        ]
        self.assertIsNone(detect_table(lines))

    def test_prefixed_list_marker_uses_the_same_markdown_style(self) -> None:
        lines = [[ocr_smart.Token("•Input", 20, 10, 90, 30)]]
        self.assertEqual(ocr_smart.format_plain_lines(lines), "- Input")

    def test_cli_dash_normalization_is_contextual(self) -> None:
        tokens = [
            ocr_smart.Token("−V", 10, 10, 30, 30),
            ocr_smart.Token("./tool.py", 40, 10, 110, 30),
        ]
        self.assertEqual(ocr_smart.join_line(tokens), "-V ./tool.py")

    def test_visual_code_panel_preserves_monospace_indentation(self) -> None:
        for page, panel in (("white", "#fafafa"), ("#181818", "#292929")):
            with self.subTest(page=page):
                image = Image.new("RGB", (400, 220), page)
                ImageDraw.Draw(image).rectangle(
                    (10, 20, 390, 200),
                    fill=panel,
                )
                lines = [
                    [ocr_smart.Token("items = [", 60, 40, 140, 60)],
                    [ocr_smart.Token("value", 100, 80, 150, 100)],
                    [ocr_smart.Token("1", 60, 120, 70, 140)],
                    [ocr_smart.Token("./next", 80, 160, 140, 180)],
                ]
                blocks = ocr_smart.detect_code_blocks(lines, image)
                self.assertEqual(len(blocks), 1)
                recovered = ocr_smart.recover_code_delimiters(lines, blocks)
                self.assertEqual(
                    ocr_smart.format_plain_lines(recovered, blocks),
                    "items = [\n"
                    "    value\n"
                    "]\n"
                    "  ./next",
                )

    def test_code_indentation_clusters_absorb_horizontal_jitter(self) -> None:
        lines = [
            [ocr_smart.Token("print(", 59, 20, 120, 40)],
            [ocr_smart.Token('"message",', 122, 60, 220, 80)],
            [ocr_smart.Token("value,", 126, 100, 190, 120)],
            [ocr_smart.Token(")", 52, 140, 62, 160)],
            [ocr_smart.Token("next()", 59, 180, 120, 200)],
        ]
        block = ocr_smart.CodeBlock(20, 200, 52, 16.7)
        self.assertEqual(
            ocr_smart.format_plain_lines(
                lines, (block,), ocr_smart.RegionRole.CODE
            ),
            "print(\n"
            '    "message",\n'
            "    value,\n"
            ")\n"
            "next()",
        )

    def test_visual_code_continuation_is_recovered_or_corrected(self) -> None:
        for terminal in (None, "一"):
            with self.subTest(terminal=terminal):
                image = Image.new("RGB", (240, 120), "white")
                ImageDraw.Draw(image).line(
                    (145, 22, 155, 42), fill="black", width=3
                )
                first_line = [
                    ocr_smart.Token("command", 60, 20, 140, 45)
                ]
                if terminal is not None:
                    first_line.append(
                        ocr_smart.Token(terminal, 145, 20, 155, 45)
                    )
                lines = [
                    first_line,
                    [ocr_smart.Token("--option", 80, 60, 150, 85)],
                ]
                block = ocr_smart.CodeBlock(20, 85, 60, 10)
                recovered = ocr_smart.recover_code_continuations(
                    lines, (block,), image
                )
                self.assertEqual(recovered[0][-1].text, "\\")
                self.assertNotIn("一", [token.text for token in recovered[0]])

    def test_missing_code_string_terminator_is_conservatively_recovered(self) -> None:
        lines = [
            [ocr_smart.Token("print(", 60, 20, 120, 40)],
            [ocr_smart.Token('"message', 100, 60, 180, 80)],
            [ocr_smart.Token("value,", 100, 100, 160, 120)],
            [ocr_smart.Token(")", 60, 140, 70, 160)],
        ]
        block = ocr_smart.CodeBlock(20, 160, 60, 10)
        recovered = ocr_smart.recover_code_delimiters(lines, (block,))
        self.assertEqual(
            ocr_smart.format_plain_lines(recovered, (block,)),
            "print(\n"
            '    "message",\n'
            "    value,\n"
            ")",
        )

    def test_dedented_isolated_glyph_recovers_expected_code_closer(self) -> None:
        for glyph in ("_", "一"):
            with self.subTest(glyph=glyph):
                lines = [
                    [ocr_smart.Token("print(", 20, 20, 80, 40)],
                    [ocr_smart.Token("value", 60, 60, 110, 80)],
                    [ocr_smart.Token(glyph, 20, 100, 30, 120)],
                ]
                block = ocr_smart.CodeBlock(20, 120, 20, 10)
                recovered = ocr_smart.recover_code_delimiters(lines, (block,))
                self.assertEqual(recovered[-1][0].text, ")")

    def test_numeric_table_geometry_recovers_infinity(self) -> None:
        cells = [
            [[ocr_smart.Token("df", 10, 10, 30, 30)]],
            [[ocr_smart.Token("8", 8, 50, 33, 72)]],
            [[ocr_smart.Token("1", 10, 90, 31, 116)]],
            [[ocr_smart.Token("2", 10, 130, 31, 156)]],
            [[ocr_smart.Token("3", 10, 170, 31, 196)]],
        ]
        recovered = ocr_smart._recover_table_glyphs(cells)
        self.assertEqual(recovered[1][0][0].text, "∞")

    def test_table_identifier_family_uses_consistent_case(self) -> None:
        values = [
            ("", "X1", "x2", "x3", "x4"),
            ("0", "1", "a", "30", "X"),
            ("1", "2", "b", "29", "Z"),
            ("2", "3", "b", "28", "Z"),
            ("3", "4", "a", "27", "y"),
            ("4", "5", "d", "26", "y"),
            ("5", "6", "a", "25", "Z"),
            ("6", "7", "a", "24", "X"),
            ("7", "8", "b", "23", "Z"),
            ("8", "9", "d", "22", "X"),
        ]
        cells = [
            [
                []
                if not value
                else [
                    ocr_smart.Token(
                        value,
                        column * 50,
                        row * 30,
                        column * 50 + 40,
                        row * 30 + 20,
                    )
                ]
                for column, value in enumerate(row_values)
            ]
            for row, row_values in enumerate(values)
        ]
        normalized = ocr_smart._normalize_table_identifier_case(cells)
        self.assertEqual(
            [normalized[0][column][0].text for column in range(1, 5)],
            ["x1", "x2", "x3", "x4"],
        )
        self.assertEqual(
            [normalized[row][4][0].text for row in range(1, 10)],
            ["x", "z", "z", "y", "y", "z", "x", "z", "x"],
        )

    def test_ascii_table_border_tokens_are_removed_only_with_frame_evidence(
        self,
    ) -> None:
        tokens = [
            ocr_smart.Token("十", 0, 0, 10, 10),
            ocr_smart.Token("Name", 40, 20, 90, 40),
            ocr_smart.Token("_", 20, 25, 28, 32, 0.54),
            ocr_smart.Token("+", 100, 45, 110, 65),
            ocr_smart.Token("-", 120, 45, 130, 65),
            ocr_smart.Token("|", 140, 45, 150, 65),
            ocr_smart.Token("=", 160, 45, 170, 65),
            ocr_smart.Token("++", 180, 45, 200, 65),
            ocr_smart.Token("--", 210, 45, 230, 65),
            ocr_smart.Token("||", 240, 45, 260, 65),
            ocr_smart.Token("==", 270, 45, 290, 65),
            ocr_smart.Token("---", 300, 45, 330, 55, 0.60),
            ocr_smart.Token("十-", 0, 100, 20, 110),
            ocr_smart.Token("+", 40, 200, 50, 220),
        ]
        stripped = ocr_smart._strip_ascii_table_artifacts(tokens)
        self.assertEqual(
            [token.text for token in stripped],
            [
                "Name",
                "+",
                "-",
                "|",
                "=",
                "++",
                "--",
                "||",
                "==",
                "+",
            ],
        )

    def test_border_only_rows_do_not_require_recognized_corners(self) -> None:
        tokens = [
            ocr_smart.Token("Name", 80, 40, 130, 60),
            ocr_smart.Token("Value", 240, 40, 300, 60),
            ocr_smart.Token("+-", 10, 10, 35, 18, 0.60),
            ocr_smart.Token("--", 150, 10, 175, 18, 0.60),
            ocr_smart.Token("1", 340, 10, 348, 18, 0.60),
            ocr_smart.Token("_", 145, 45, 152, 52, 0.55),
        ]
        stripped = ocr_smart._strip_ascii_table_artifacts(tokens)
        self.assertEqual([token.text for token in stripped], ["Name", "Value"])

    def test_open_grid_is_left_for_borderless_reconstruction(self) -> None:
        image = Image.new("RGB", (600, 300), "white")
        draw = ImageDraw.Draw(image)
        for x in (150, 300, 450):
            draw.line((x, 40, x, 240), fill="black", width=3)
        for y in (40, 90, 140, 190, 240):
            draw.line((40, y, 560, y), fill="black", width=3)
        self.assertIsNone(ocr_smart.detect_grid(image))


class EndToEndHealthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="smartocr-health-")
        cls.directory = Path(cls.temporary.name)
        font = load_test_font()

        cls.light_image = cls.directory / "light.png"
        light = Image.new("RGB", (900, 140), "white")
        ImageDraw.Draw(light).text(
            (40, 40),
            "SMART OCR HEALTH CHECK 12345",
            font=font,
            fill="black",
        )
        light.save(cls.light_image)

        cls.dark_image = cls.directory / "dark.png"
        dark = Image.new("RGB", (900, 140), "#181818")
        ImageDraw.Draw(dark).text(
            (40, 40),
            "DARK MODE WORKS 67890",
            font=font,
            fill="white",
        )
        dark.save(cls.dark_image)

        cls.small_image = cls.directory / "small.png"
        small = Image.new("RGB", (700, 100), "white")
        ImageDraw.Draw(small).text(
            (30, 25),
            "SMALL SCALE CHECK 2468",
            font=load_test_font(24),
            fill="black",
        )
        small.save(cls.small_image)

        cls.large_image = cls.directory / "large.png"
        large = Image.new("RGB", (1200, 190), "#181818")
        ImageDraw.Draw(large).text(
            (40, 40),
            "LARGE SCALE CHECK 13579",
            font=load_test_font(54),
            fill="white",
        )
        large.save(cls.large_image)

        cls.table_image = cls.directory / "table.png"
        table = Image.new("RGB", (900, 360), "white")
        draw = ImageDraw.Draw(table)
        x_lines = (30, 430, 870)
        y_lines = (30, 130, 230, 330)
        for x in x_lines:
            draw.line((x, y_lines[0], x, y_lines[-1]), fill="black", width=3)
        for y in y_lines:
            draw.line((x_lines[0], y, x_lines[-1], y), fill="black", width=3)
        for row, cells in enumerate(
            (("Name", "Status"), ("Alpha", "Ready"), ("Beta", "Good"))
        ):
            draw.text((55, y_lines[row] + 30), cells[0], font=font, fill="black")
            draw.text((455, y_lines[row] + 30), cells[1], font=font, fill="black")
        table.save(cls.table_image)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_locked_dependencies_are_installed(self) -> None:
        for raw_line in (ROOT / "requirements.txt").read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            name, expected = line.split("==", maxsplit=1)
            with self.subTest(package=name):
                self.assertEqual(metadata.version(name), expected)

    def test_light_and_dark_ocr(self) -> None:
        cases = (
            (self.light_image, "SMART OCR HEALTH CHECK 12345"),
            (self.dark_image, "DARK MODE WORKS 67890"),
            (self.small_image, "SMALL SCALE CHECK 2468"),
            (self.large_image, "LARGE SCALE CHECK 13579"),
        )
        for image, expected in cases:
            with self.subTest(image=image.name):
                self.assertEqual(
                    ocr_smart.process_ocr(image, copy=False),
                    expected,
                )

    def test_ruled_table_ocr(self) -> None:
        self.assertEqual(
            ocr_smart.process_ocr(self.table_image, copy=False),
            "| Name | Status |\n"
            "| --- | --- |\n"
            "| Alpha | Ready |\n"
            "| Beta | Good |",
        )

    def test_clipboard_failure_raises_before_notification(self) -> None:
        token = ocr_smart.Token("recognized", 20, 20, 160, 50)
        with (
            patch.object(ocr_smart, "recognize", return_value=[token]),
            patch.object(
                ocr_smart,
                "copy_to_clipboard",
                side_effect=OSError("pbcopy unavailable"),
            ),
            patch.object(ocr_smart, "notify_copied") as notify,
            self.assertRaises(ocr_smart.ClipboardError) as raised,
        ):
            ocr_smart.process_ocr(self.light_image, copy=True)
        self.assertEqual(raised.exception.output, "recognized")
        notify.assert_not_called()

    def test_trigger_accepts_shortcuts_binary_stdin(self) -> None:
        log_file = self.directory / "successful-trigger.log"
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "SMART_OCR_LOCK_FILE": str(self.directory / "trigger.lock"),
                "SMART_OCR_LOG_FILE": str(log_file),
                "TMPDIR": str(self.directory),
            }
        )
        result = subprocess.run(
            [str(ROOT / "trigger.sh")],
            input=self.light_image.read_bytes(),
            capture_output=True,
            check=False,
            env=environment,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode("utf-8", errors="replace"),
        )
        self.assertEqual(
            result.stdout.decode("utf-8").strip(),
            "SMART OCR HEALTH CHECK 12345",
        )
        self.assertFalse(log_file.exists())

    def test_duplicate_trigger_fails_without_empty_clipboard_output(self) -> None:
        lock_file = self.directory / "active-trigger.lock"
        ready_file = self.directory / "active-trigger.ready"
        holder = subprocess.Popen(
            [
                "/usr/bin/lockf",
                "-k",
                str(lock_file),
                sys.executable,
                "-c",
                (
                    "import sys, time; "
                    "from pathlib import Path; "
                    "Path(sys.argv[1]).touch(); "
                    "time.sleep(30)"
                ),
                str(ready_file),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            if ready_file.exists():
                break
            if holder.poll() is not None:
                self.fail("lock holder exited before acquiring the test lock")
            time.sleep(0.02)
        else:
            holder.terminate()
            holder.wait(timeout=5)
            self.fail("timed out waiting for the test lock")

        environment = os.environ.copy()
        environment.update(
            {
                "SMART_OCR_LOCK_FILE": str(lock_file),
                "SMART_OCR_LOG_FILE": str(self.directory / "duplicate.log"),
                "TMPDIR": str(self.directory),
            }
        )
        try:
            result = subprocess.run(
                [str(ROOT / "trigger.sh")],
                input=self.light_image.read_bytes(),
                capture_output=True,
                check=False,
                env=environment,
                timeout=10,
            )
        finally:
            holder.terminate()
            holder.wait(timeout=5)
            lock_file.unlink(missing_ok=True)
            ready_file.unlink(missing_ok=True)

        self.assertEqual(result.returncode, 75)
        self.assertEqual(result.stdout, b"")

    def test_unlocked_stale_lock_file_does_not_block(self) -> None:
        lock_file = self.directory / "stale-trigger.lock"
        lock_file.write_text("999999\n")
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "SMART_OCR_LOCK_FILE": str(lock_file),
                "SMART_OCR_LOG_FILE": str(self.directory / "stale.log"),
                "TMPDIR": str(self.directory),
            }
        )
        result = subprocess.run(
            [str(ROOT / "trigger.sh")],
            input=self.light_image.read_bytes(),
            capture_output=True,
            check=False,
            env=environment,
            timeout=30,
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode("utf-8", errors="replace"),
        )
        self.assertEqual(
            result.stdout.decode("utf-8").strip(),
            "SMART OCR HEALTH CHECK 12345",
        )
        self.assertTrue(lock_file.exists())
        lock_file.unlink()

    def test_trigger_reports_ocr_failure_to_its_caller(self) -> None:
        unreadable = self.directory / "corrupt.png"
        unreadable.write_bytes(b"not a real screenshot")
        log_file = self.directory / "failure.log"
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "SMART_OCR_LOCK_FILE": str(self.directory / "failure.lock"),
                "SMART_OCR_LOG_FILE": str(log_file),
                "TMPDIR": str(self.directory),
            }
        )
        result = subprocess.run(
            [str(ROOT / "trigger.sh"), str(unreadable)],
            capture_output=True,
            check=False,
            env=environment,
            timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn("OCR failed with status", log_file.read_text())

    def test_cli_failure_is_clean(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "ocr_smart.py"),
                "--no-clipboard",
                str(self.directory / "missing.png"),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("ocr-smart: image does not exist:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
