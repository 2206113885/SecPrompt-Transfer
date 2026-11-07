"""Export the compiled TikZ diagram as vector SVG; requires pymupdf.

latexmk -pdf -outdir=figures figures/study_flow.tex
uv run --isolated --no-project --with pymupdf python figures/export_study_flow.py
"""
from pathlib import Path
import pymupdf

here = Path(__file__).resolve().parent
doc = pymupdf.open(here / 'study_flow.pdf')
assert len(doc) == 1
assert abs(doc[0].rect.width / 72 - 372 / 72.27) < 0.01
(here / 'study_flow.svg').write_text(doc[0].get_svg_image(text_as_path=True))
doc[0].get_pixmap(dpi=150, alpha=False).save(here / 'study_flow.png')
print('Study-flow vector SVG and preview PNG exported at the final printed width.')
