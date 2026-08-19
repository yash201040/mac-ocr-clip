#!/bin/bash

set -u

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:${PATH:-}"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
OCR="$ROOT/ocr_smart.py"
LOG_FILE="${SMART_OCR_LOG_FILE:-$ROOT/smartocr.log}"
LOCK_FILE="${SMART_OCR_LOCK_FILE:-${TMPDIR:-/private/tmp}/smartocr.lock}"
TEMP_DIR=""
INPUT_SOURCE="capture"
NONINTERACTIVE=0

log_error() {
    printf '%s %s\n' "$(/bin/date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG_FILE"
}

if [ ! -x "$PYTHON" ] || [ ! -f "$OCR" ]; then
    log_error "Smart OCR is not installed; run $ROOT/setup.sh from Terminal."
    exit 1
fi

if [ "${SMART_OCR_LOCKED:-0}" != "1" ]; then
    SMART_OCR_LOCKED=1 \
        /usr/bin/lockf -s -t 0 -k "$LOCK_FILE" "$ROOT/trigger.sh" "$@"
    LOCK_STATUS=$?
    if [ "$LOCK_STATUS" -eq 75 ]; then
        log_error "Skipped a duplicate trigger because Smart OCR was already active."
    elif [ "$LOCK_STATUS" -eq 126 ] || [ "$LOCK_STATUS" -eq 127 ]; then
        # trigger.sh never exits with these, so lockf itself could not run.
        log_error "Could not run /usr/bin/lockf (status $LOCK_STATUS)."
    fi
    exit "$LOCK_STATUS"
fi

cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -f \
            "$TEMP_DIR/capture.png" \
            "$TEMP_DIR/capture.err" \
            "$TEMP_DIR/ocr.err" \
            "$TEMP_DIR/shortcut-input"
        rmdir "$TEMP_DIR" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' HUP INT TERM

TEMP_DIR="$(mktemp -d "${TMPDIR:-/private/tmp}/smartocr.XXXXXX")"
if [ -z "$TEMP_DIR" ] || [ ! -d "$TEMP_DIR" ]; then
    log_error "Could not create a temporary directory."
    exit 1
fi

# Accept ordinary image-path arguments for Terminal and other trusted callers.
# Shortcuts should use stdin because its Group Container path is not readable
# by the standalone Python process on macOS 26.
IMAGE_FILE=""
for CANDIDATE in "$@"; do
    case "$CANDIDATE" in
        file://*)
            CANDIDATE="$("$PYTHON" -c 'import sys, urllib.parse; print(urllib.parse.unquote(urllib.parse.urlparse(sys.argv[1]).path))' "$CANDIDATE")"
            ;;
    esac
    if [ -f "$CANDIDATE" ] && [ -s "$CANDIDATE" ]; then
        IMAGE_FILE="$CANDIDATE"
        INPUT_SOURCE="argument"
        break
    fi
done

if [ "$#" -gt 0 ] && [ -z "$IMAGE_FILE" ]; then
    log_error "Input was supplied, but it was not a readable image file."
    exit 1
fi

# Shortcuts' binary stdin is materialized into this run's private temporary
# directory before the standalone OCR process is launched.
if [ ! -t 0 ]; then
    NONINTERACTIVE=1
fi
if [ -z "$IMAGE_FILE" ] && [ "$NONINTERACTIVE" -eq 1 ]; then
    STDIN_FILE="$TEMP_DIR/shortcut-input"
    /bin/cat >"$STDIN_FILE"
    if [ -s "$STDIN_FILE" ]; then
        IMAGE_FILE="$("$PYTHON" -c '
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()
try:
    candidate = data.decode("utf-8").strip()
except UnicodeDecodeError:
    candidate = ""
print(candidate if candidate and os.path.isfile(candidate) else path)
' "$STDIN_FILE")"
        INPUT_SOURCE="stdin"
    fi
fi

# Never call screencapture from a background Shortcuts shell. Missing input is
# a Shortcut wiring error, not a reason to re-enter the unreliable TCC path.
if [ -z "$IMAGE_FILE" ] && [ "$NONINTERACTIVE" -eq 1 ]; then
    log_error "No screenshot input reached trigger.sh; connect Take Screenshot and pass its result to Run Shell Script."
    exit 1
fi

# No input in an interactive Terminal means the user requested a fresh capture.
if [ -z "$IMAGE_FILE" ]; then
    IMAGE_FILE="$TEMP_DIR/capture.png"
    CAPTURE_ERROR_FILE="$TEMP_DIR/capture.err"
    /usr/sbin/screencapture -i -s -x "$IMAGE_FILE" 2>"$CAPTURE_ERROR_FILE"
    CAPTURE_STATUS=$?
    if [ ! -s "$IMAGE_FILE" ]; then
        CAPTURE_ERROR=""
        if [ -s "$CAPTURE_ERROR_FILE" ]; then
            IFS= read -r CAPTURE_ERROR <"$CAPTURE_ERROR_FILE"
        fi
        if [ -n "$CAPTURE_ERROR" ]; then
            log_error "Capture failed (status $CAPTURE_STATUS): $CAPTURE_ERROR"
        else
            log_error "Capture was cancelled or produced an empty image."
        fi
        exit 0
    fi
fi

OCR_ERROR_FILE="$TEMP_DIR/ocr.err"
if [ "$INPUT_SOURCE" = "capture" ]; then
    # A capture this script started owns the pasteboard and its notification.
    "$PYTHON" "$OCR" -- "$IMAGE_FILE" 2>"$OCR_ERROR_FILE"
    OCR_STATUS=$?
else
    # Supplied images print to stdout instead; Shortcuts copies that result
    # with its native clipboard action, so only that path is announced.
    SMART_OCR_NO_CLIPBOARD=1 "$PYTHON" "$OCR" -- "$IMAGE_FILE" 2>"$OCR_ERROR_FILE"
    OCR_STATUS=$?
    if [ "$OCR_STATUS" -eq 0 ] && [ "$INPUT_SOURCE" = "stdin" ]; then
        /usr/bin/osascript -e 'display notification "" with title "OCR copied"' >/dev/null 2>&1 || true
    fi
fi
if [ -s "$OCR_ERROR_FILE" ]; then
    /bin/cat "$OCR_ERROR_FILE" >>"$LOG_FILE"
fi
if [ "$OCR_STATUS" -ne 0 ]; then
    log_error "OCR failed with status $OCR_STATUS."
    exit "$OCR_STATUS"
fi