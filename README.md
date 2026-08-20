# Smart OCR

Accuracy-first, fully local screenshot OCR for macOS. It is tuned for light and
dark developer interfaces, code, plain text, lists, and Markdown table
reconstruction. Images and recognized text never leave the Mac.

Every recognized word of two or more characters is reread from upscaled,
contrast-normalized, and sharpened crops. Only independently supported changes
are accepted; isolated-character substitutions are rejected. The formatter then assigns local prose, list, code,
and table regions before applying role-specific spacing and normalization.
A small selection therefore takes a few seconds, and a dense full-window
capture takes roughly ten.

## Install

Requirements:

- macOS
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv`)
- The built-in Shortcuts app

Clone this repo at `~`, then run:

```bash
cd "$HOME/mac-ocr-clip"
/bin/bash setup.sh
```

`setup.sh` creates a private Python 3.13 environment in `.venv`, installs only
the locked runtime dependencies, validates the shell integration, and runs
self-contained checks covering light and dark mode, small and large text,
ruled and borderless tables, icons, punctuation, multilayer consensus,
bullet and ordered list wrapping, ASCII-table cleanup, code panels, and the
`trigger.sh` stdin and locking behavior. Interactive capture needs a person
in front of the screen, so it is the one path the checks cannot exercise. The
OCR models are included in the installed package, so OCR is offline after
setup.

To reinstall, update, or verify the tool later, run the same command again.
Delete only `.venv` before rerunning setup if its Python installation becomes
damaged.

## Configure the macOS Shortcut

Use these actions in this order inside the macOS Shortcuts app:

1. **Take Screenshot**
  - Type: **Interactive**
  - Show More → Selection: **Custom**
2. **Run Shell Script**
  - Shell: `/bin/bash`
  - Script: `"$HOME/.ocr-env/trigger.sh"`
  - Input: the result of **Take Screenshot**
  - Pass input: **to stdin**
3. **Copy to Clipboard**
  - Copy **Shell Script Result** to the clipboard.

Do not add `"$@"`, do not select **as arguments**, and do not call
`screencapture` in the Shortcut. Binary stdin avoids macOS denying Python
access to Shortcuts' private temporary image path. The native **Take
Screenshot** action also avoids the intermittent `could not create image from rect` failure caused by launching `screencapture` from a Shortcuts
shell.

A notification confirms each successful copy. Failures stay silent and are
written to the log instead, so a missing notification means the run failed.

Assign the finished Shortcut to the desired keyboard shortcut or Touch Bar button (preferred if available). macOS may ask for Screen Recording permission from Shortcuts app and terminal the first time it runs or running after next macOS restart.

## Use from Terminal

Run an interactive area capture and copy its OCR text:

```bash
"$HOME/.ocr-env/trigger.sh"
```

OCR an existing image and print it without changing the clipboard:

```bash
"$HOME/.ocr-env/trigger.sh" "/path/to/image.png"
```

The clipboard and its notification are reserved for captures this tool starts
and for the Shortcut, which copies the printed result itself.

## Health check

The normal setup command also repairs dependency drift and runs every check:

```bash
cd "$HOME/.ocr-env"
./setup.sh
```

To run only the checks:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

Tests generate temporary screenshots at runtime. No personal screenshots or
expected-output logs are required or tracked; the local `debug/` corpus is
ignored by Git.

## Troubleshooting

Failures are appended to `~/.ocr-env/smartocr.log`; the file is created only
when needed.

- **No clipboard text:** confirm that **Copy to Clipboard** receives **Shell
Script Result**.
- **Operation not permitted:** pass the screenshot **to stdin**, not **as
arguments**.
- **Could not create image from rect:** use Shortcuts' native **Take
Screenshot** action instead of shelling out to `screencapture`.
- **Smart OCR is not installed:** run `/bin/bash "$HOME/.ocr-env/setup.sh"`.
- **Terminal capture is denied:** allow Terminal under System Settings →
Privacy & Security → Screen & System Audio Recording.



## Files

- `trigger.sh` — Shortcuts/Terminal entry point, locking, and temporary files
- `ocr_smart.py` — OCR and layout reconstruction
- `setup.sh` — reproducible install and health check
- `requirements.in` — direct dependency versions
- `requirements.txt` — complete locked runtime environment
- `tests/test_health.py` — generated-image functional checks
- `.gitignore` — local environments, debug data, logs, caches, and build output
- `LICENSE` — MIT

`.venv/`, `debug/`, logs, Python/tool caches, test reports, package builds, and
editor metadata are generated locally and are not source files.

To update pinned packages after editing `requirements.in`:

```bash
uv pip compile requirements.in --python-version 3.13 \
  --output-file requirements.txt
./setup.sh
```



## Known limits

- Pixel-identical glyphs such as lowercase `o` and zero `0` cannot always be
disambiguated without application-specific context. Multilayer recognition
deliberately does not rewrite isolated glyphs.
- Very low-confidence isolated glyphs are omitted to avoid turning UI icons
into letters or digits; genuinely ambiguous standalone characters can also
be omitted.
- Very small, low-resolution subscripts and superscripts may remain ambiguous;
the formatter does not invent a character when the recognition passes disagree.
- The formatter does not dictionary-correct recognized words. Missing code
delimiters are recovered only when both a detected visual code panel and
tightly constrained balancing evidence support the correction.
- Dash width is recognized, not inferred, so a hyphen, en dash, and minus
sign can be reported interchangeably at screenshot resolution.
- Ruled tables use detected border lines. Borderless tables require at least
three rows with persistent column gaps and repeated alignment.
- The first detected table row is treated as the Markdown header.
- Decorative UI icons are normally omitted; only supported colored status
symbols inside detected tables may be recovered.

