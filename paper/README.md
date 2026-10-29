# Paper / one-page summary

`onepager.pdf` — a one-page research summary (for the advisor meeting and as a seed
for the full write-up). Source: `onepager.html`.

Rebuild the PDF (headless Chrome, no extra deps):

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(pwd)/paper/onepager.pdf" \
  "file://$(pwd)/paper/onepager.html"
```

Fill in `[Name] · [Student ID] · [Course]` in `onepager.html` before sharing.
Figures come from `docs/assets/fig_*.png` (regenerate via `dev/make_result_figs.py`).
