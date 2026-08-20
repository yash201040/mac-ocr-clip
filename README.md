# Smart OCR

Accuracy-first, fully local screenshot OCR for macOS. Tuned for light and dark
developer interfaces, code, prose, lists, and Markdown table reconstruction.
Images and recognized text never leave your Mac.

Every word of two or more characters is reread from upscaled, contrast-normalized
crops, and only independently supported corrections are accepted. A small
selection takes a few seconds; a dense full-window capture takes roughly ten.

## Install

Requires macOS, the built-in Shortcuts app, and
[uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv`).

```bash
git clone https://github.com/yash201040/mac-ocr-clip.git ~/mac-ocr-clip
cd ~/mac-ocr-clip
/bin/bash setup.sh
```

`setup.sh` builds a private Python 3.13 environment in `.venv`, installs the
locked dependencies, and runs 59 self-contained checks. It finishes by printing
the exact path to paste into your Shortcut. OCR models ship with the
dependencies, so recognition is fully offline afterwards.

Any clone location works — the scripts resolve their own directory. Rerun the
same command any time to update, repair dependency drift, or re-verify.

## Configure the macOS Shortcut

In the Shortcuts app, add these three actions in order:

1. **Take Screenshot** — Type: **Interactive**, Show More → Selection: **Custom**
2. **Run Shell Script** — Shell `/bin/bash`, Script `"$HOME/mac-ocr-clip/trigger.sh"`,
   Input: result of **Take Screenshot**, Pass input: **to stdin**
3. **Copy to Clipboard** — copy **Shell Script Result**

Do not add `"$@"`, do not choose **as arguments**, and do not call `screencapture`
in the Shortcut. Binary stdin avoids macOS denying Python access to Shortcuts'
private image path, and the native **Take Screenshot** action avoids intermittent
`could not create image from rect` failures.

Then assign the Shortcut a keyboard shortcut or Touch Bar button. macOS may ask
for Screen Recording permission on first run and after a restart.

A notification confirms each successful copy. Failures are silent and go to the
log, so a missing notification means the run failed.

## Use from Terminal

```bash
# Interactive area capture, copied to the clipboard
~/mac-ocr-clip/trigger.sh

# OCR an existing image, printed without touching the clipboard
~/mac-ocr-clip/trigger.sh /path/to/image.png
```

The clipboard and its notification are reserved for captures this tool starts
and for the Shortcut, which copies the printed result itself.

To run only the checks:

```bash
cd ~/mac-ocr-clip
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Troubleshooting

Failures are appended to `smartocr.log` inside the clone; it is created only when
needed.

- **No clipboard text** — confirm **Copy to Clipboard** receives **Shell Script Result**.
- **Operation not permitted** — pass the screenshot **to stdin**, not **as arguments**.
- **Could not create image from rect** — use the native **Take Screenshot** action
  rather than shelling out to `screencapture`.
- **No such file or directory** — the **Run Shell Script** action holds a path from
  another machine or account. Replace it with `"$HOME/mac-ocr-clip/trigger.sh"`.
- **Smart OCR is not installed** — run `/bin/bash ~/mac-ocr-clip/setup.sh`.
- **Terminal capture is denied** — allow Terminal under System Settings → Privacy
  & Security → Screen & System Audio Recording.

## Files

`trigger.sh` (Shortcuts/Terminal entry point, locking, temp files) ·
`ocr_smart.py` (OCR and layout reconstruction) · `setup.sh` (install and health
check) · `requirements.in` / `requirements.txt` (direct and locked dependencies) ·
`tests/test_health.py` (generated-image checks)

`.venv/`, `debug/`, and logs are generated locally and are not tracked. To update
pins after editing `requirements.in`:

```bash
uv pip compile requirements.in --python-version 3.13 --output-file requirements.txt
./setup.sh
```

## Known limits

- Pixel-identical glyphs such as `o` and `0` cannot always be disambiguated without
  application context; isolated glyphs are deliberately never rewritten.
- Very low-confidence isolated glyphs are omitted rather than guessed, so UI icons
  do not become letters — genuinely ambiguous standalone characters may drop too.
- Small, low-resolution subscripts and superscripts may stay ambiguous.
- Recognized words are not dictionary-corrected. Missing code delimiters are
  recovered only with both a detected code panel and tight balancing evidence.
- Dash width is recognized, not inferred, so hyphen, en dash, and minus can be
  reported interchangeably at screenshot resolution.
- Ruled tables use detected borders; borderless tables need at least three rows
  with persistent column gaps. The first detected row becomes the Markdown header.
- Decorative icons are omitted; only supported colored status symbols inside
  detected tables are recovered.

## License

MIT — see [LICENSE](LICENSE).
