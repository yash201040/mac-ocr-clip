#!/bin/bash

set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"

if [ "$(/usr/bin/uname -s)" != "Darwin" ]; then
    printf 'Smart OCR requires macOS.\n' >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [ -x "/opt/homebrew/bin/uv" ]; then
    UV="/opt/homebrew/bin/uv"
elif [ -x "/usr/local/bin/uv" ]; then
    UV="/usr/local/bin/uv"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    printf 'uv is required. Install it with: brew install uv\n' >&2
    exit 1
fi

if [ ! -e "$VENV" ]; then
    "$UV" venv --python 3.13 "$VENV"
elif [ ! -x "$PYTHON" ]; then
    printf '%s exists but is not a usable environment; remove it and rerun setup.sh.\n' "$VENV" >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 13))'; then
    printf '%s must use Python 3.13; remove it and rerun setup.sh.\n' "$VENV" >&2
    exit 1
fi

"$UV" pip sync --python "$PYTHON" "$ROOT/requirements.txt"
/bin/chmod +x "$ROOT/setup.sh" "$ROOT/trigger.sh"
/bin/bash -n "$ROOT/trigger.sh"

PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v

printf '\nSmart OCR is installed and healthy.\n'
printf 'Shortcut command: %s\n' "$ROOT/trigger.sh"
