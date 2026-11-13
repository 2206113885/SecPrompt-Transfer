# RQ3 publication figures

Revised on 14 November 2026 to match the manuscript's shared typography and
grid style. Figures 8 and 10 use producer colors; Figure 9 retains favorable
and adverse screening directions. All numerical positions and intervals come
from the unchanged CSV inputs in `../data/rq3-systematic/`.

From the repository root:

```sh
uv run --isolated --no-project --with numpy==2.5.4 --with matplotlib==3.11.2 python paper/figures/build_rq3_figures.py
```

The generated `rq3_transfer_t_all_pe2/btape` files correspond to Figure 8;
`rq3_error_tradeoffs` to Figure 9; `rq3_transfer_t_nfr_pe2/btape` to Figure 10.
PDF, SVG, and 300 dpi PNG outputs are included. The native width is 372 TeX
points; labels use Helvetica Neue at 8--8.5 pt, with installed-font fallbacks
in the style module. PDFs embed their fonts; SVG text remains editable.
`rq3_split_placements.csv` preserves all 420 split values.
`rq3_marker_checks.json` checks marker bounds, and `rq3_figure_audits.json`
checks text collisions and clipping. Figure 11 and its source are under
[`analysis/source_duplicates`](../../analysis/source_duplicates/README.md).
