# SecReq duplicate-text sensitivity analysis

This supplement supports Sections 3.2, 3.5, and 4.2.4 and Appendix D of
*Reusing Automatically Optimized Prompts Across Datasets and Models: An Empirical Study of Security Requirement Classification*.

The analysis rescores stored predictions from the original 30 source cells and
380 cell–candidate combinations. It performs no model inference or prompt
optimization and does not replace the historical source-selected definitions.

## Reproduce

Use the data and predictions in the
[SecPrompt-Transfer replication repository](https://github.com/2206113885/SecPrompt-Transfer).
From that repository's root, run the following command, replacing the script
path if this supplement is stored elsewhere:

```sh
uv run --isolated --no-project --with numpy==2.5.4 python \
  analysis/source_duplicates/audit_duplicates.py \
  --repository-root . \
  --output-dir /tmp/secreq-duplicate-audit
```

Python and NumPy 2.5.4 are the only runtime dependencies. An existing compatible
Python environment with that NumPy version can run the script directly without
uv. The script accepts explicit input and output paths and records input
SHA-256 hashes in `audit-results.json`.

## Files

| File | Contents |
| --- | --- |
| `audit_duplicates.py` | Standalone audit and rescoring script |
| `audit-results.json` | Aggregate results, checks, input hashes, and limitations |
| `split-overlap.csv` | Text overlap and retained sample sizes for seeds 40–44 |
| `candidate-test-scores.csv` | Candidate scores under each source-test weighting scheme |
| `cell-test-sensitivity.csv` | Per-cell scores, ranking agreement, and selection gaps |
| `validation-reselection-sensitivity.csv` | Fixed-pool reselection after excluding validation overlap |
| `figure/` | Figure 11 in PDF, SVG, and PNG; generating script, style module, plotted values, and checks |

To rebuild Figure 11 from these saved results:

```sh
uv run --isolated --no-project --with matplotlib==3.11.2 --with numpy==2.5.4 python \
  analysis/source_duplicates/figure/build_figure.py
```

The plot is drawn at the manuscript's 372-TeX-point text width. It uses the
Helvetica Neue font and the same blue, orange, and green model colors as the
other manuscript figures. Marker shapes and fill styles also encode the groups
for grayscale reading. The PDF embeds its fonts; the SVG retains editable text.
The style module can use its documented installed-font fallbacks on another
machine. The figure shows individual cells and descriptive summaries; it does
not treat the five overlapping splits as independent replicates or add
confidence intervals.

Figure 11 uses two stacked horizontal panels. Within each scoring group,
summary diamonds occupy a separate row above the cell markers and their values
appear in the right column. Deterministic vertical packing separates equal or
nearby cell values; numerical horizontal positions are unchanged.
`figure/point-placements.csv` records the placement of all 150 cell markers.
`figure/figure-checks.json` includes checks of actual marker bounds and edge
strokes, in addition to text collisions, clipping, and summary agreement.

## Matching and sensitivity checks

Texts are normalized with Unicode NFC, case folding, and collapsed whitespace.
Exact matching identifies the same 66 duplicate groups. Each contains two
records with consistent labels: 510 records represent 444 distinct texts.

Three source-test evaluations are compared:

1. **Original rows:** the original 204-row source-test partition.
2. **Unseen rows:** exclude a test record if its normalized text occurs in that
   split's optimization pool or validation set. This retains 172–177 rows.
3. **Equal text weights:** apply the same exclusion and give each remaining
   distinct text total weight one, divided equally among its records. There are
   159–166 distinct texts per split.

The original candidate pools and validation-selected definitions remain fixed
in these comparisons. A separate diagnostic excludes validation rows matching
the optimization pool, then reselects within each frozen pool using weighted-F1
and the original latest-iteration tie rule. The selected definition is unchanged
in 27 of 30 cells. Those diagnostic selections do not replace the historical
definitions in the main study.

## Selected results

All values below summarize the 30 cells with equal cell weights. F1 differences
are percentage points. The original evaluation is reproduced before applying
either sensitivity check.

| Metric | Original rows | Unseen rows | Equal text weights |
| --- | ---: | ---: | ---: |
| Mean selected-definition macro-F1 (%) | 80.752 | 80.050 | 79.436 |
| Mean gain over the initial definition (points) | 3.073 | 2.805 | 2.756 |
| Median selection gap (points) | 0.769 | 1.112 | 1.181 |
| Median validation–test Kendall tau-b | 0.568 | 0.497 | 0.507 |
| Cells within one point of their test-pool maximum | 17/30 | 13/30 | 13/30 |

## Interpretation limits

These are descriptive checks on fixed predictions from five overlapping splits.
Excluding records also changes sample size and class composition, so the
before–after differences do not isolate a causal effect of duplication. The
checks retain evidence of selection losses within SecReq. They do not establish
what candidate pools or transfer results would follow from a new optimization
using splits that keep identical texts together. No semantic near-duplicate or
pretraining-contamination analysis was performed.
