# Figures

Figures in `docs/` are reproducible from scripts in this directory. The convention:

- Each figure has a generator script: `<chapter>_<figure>.py`
- Generators write their output to `docs/figures/_generated/<name>.svg`
- Markdown pages reference figures by the generated path: `![…](../figures/_generated/<name>.svg)`
- Hand-drawn schematics (Inkscape, etc.) live alongside generators as `<name>.svg` and are referenced directly

## Regenerating

```bash
uv run python docs/figures/_make_all.py
```

(Will be added when the first generator script lands.)

## Tooling

- Python: `matplotlib` for plots; `numpy` for data
- For 3D robot/frame visualisations, `matplotlib.mpl_toolkits.mplot3d` is sufficient for static figures
- Output format: SVG (vector, scales cleanly in light/dark mode)
- Style: keep figures legible at the small size mkdocs-material renders inline (~600 px wide)

## Why scripts and not committed PNGs

- Figures stay reproducible across edits to data or notation
- Differences in regenerated figures show up in PR diffs (visually obvious)
- Light/dark mode adjustments can be parametrised
