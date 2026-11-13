# Figure 1: Study workflow

Publication figure for **Reusing Automatically Optimized Prompts Across Datasets and Models: An Empirical Study of Security Requirement Classification**.

The diagram separates SecReq-only class-definition optimization, frozen source outputs, and GitReq evaluation for RQ1–RQ3. Dashed branches indicate the supporting RQ2 analyses. Target labels never feed back into source optimization. This release updates the artwork; it does not change experimental data or analysis code.

## Files

- [PDF](Fig1-study-workflow.pdf): final vector figure, 130.74 × 168.14 mm, embedded fonts and no raster objects.
- [SVG](Fig1-study-workflow.svg): vector figure with embedded subset Arial fonts.
- [PNG](Fig1-study-workflow.png): inspection preview.
- [Draw.io](Fig1-study-workflow.drawio): editable native diagrams.net cells and connectors.
- [YAML](Fig1-study-workflow.yaml): canonical source with explicit geometry, labels and styles.
- [Inclusion block](figure1-inclusion.tex): LaTeX reference, separate caption, and original figure label. It expects the figure PDF beside the manuscript source.

## Rebuild

Open the `.drawio` file in diagrams.net for visual editing. To reproduce the published exports, edit the canonical YAML and run:

```sh
uv run --no-project build_figure.py \
  --drawio-cli /path/to/drawio/scripts/cli.js \
  --font-dir /path/to/Arial-font-directory
```

The builder uses the Draw.io skill CLI v2.8.0, Node.js, system Cairo, and Arial font files named `Arial.ttf`, `Arial Bold.ttf`, and `Arial Italic.ttf`. The fonts must be available locally; standalone font files are not distributed here. The Python script pins CairoSVG, Matplotlib, PyMuPDF, PyYAML, and fontTools through inline dependency metadata. Default paths target macOS; use the options above on other systems. STIX mathematical symbols are exported as vector paths.

A YAML rebuild regenerates the Draw.io file. Transfer manual diagram edits back to the YAML before rebuilding if both sources must remain synchronized.

## Verification

`content-audit.json` records the study-design checks. `export-checks.json` records label widths, physical dimensions, embedded fonts, and the absence of raster images in the PDF. `SHA256SUMS` covers the files in this directory.

Use the figure at full manuscript text width; reducing this layout to an 84 mm column makes the text too small. The published PDF is ready for inclusion without rebuilding it.
