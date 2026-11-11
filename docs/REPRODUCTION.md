# Reproduction notes

## Offline workflow

From the repository root, run:

```bash
uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python reproduce.py --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python reproduce.py --full
```

The first command checks the compact predictions and their correspondence to manuscript tables. It also independently reconstructs all 300 source-test ranking records and 180 source-test selection-gap records. The second reruns the original RQ1/RQ2 statistical analyses and the RQ3 uncertainty analysis. Thread settings bound CPU parallelism; they do not alter predictions or bootstrap counts.

Outputs:

| Location | Contents |
| --- | --- |
| `reproduced/verification.json` | Counts, vote/metric checks, and overall pass/fail |
| `reproduced/source_test_rank_reference.csv` | Recomputed supplement correlations |
| `reproduced/source_test_selection_gaps.csv` | Recomputed supplement selection losses |
| `reproduced/rq1/tables/` | Source/target scores, paired effects, repository-cluster intervals |
| `reproduced/rq2/tables/` | Rankings, candidate diversity, selection losses, OOF checks, bootstrap diagnostics |
| `reproduced/rq3/` | Cell/group effects, pointwise/simultaneous intervals and bootstrap draws |

RQ1 checks deterministic table hashes when rerun. RQ3 refuses to overwrite a completed analysis: its compact adapter reports that an existing result is complete. To rerun it independently, move `reproduced/rq3` aside and invoke `analysis/rq3/from_compact.py` again. Reference inputs and published manuscript statistics remain under `data/`, `analysis/*/data/` and `paper/data/`.

The checks use a score tolerance of `1e-12`; ranks use the retained tie handling, not rounded displayed scores. There are no model calls in the offline workflow.

## Manuscript tables and figures

The README lists asset-generation commands. They use the exact frozen statistics and the same final-size plotting code retained with the manuscript. To test them without replacing bundled PDFs, run:

```bash
uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python verify_assets.py
```

This builds assets in `reproduced/paper/`. The TikZ flow diagram additionally requires a LaTeX distribution with `standalone` and `tikz`:

```bash
latexmk -pdf -outdir=paper/figures paper/figures/study_flow.tex
```

PDF metadata can differ between renders even when the data and geometry agree. The numerical comparison audit covers the statistical tables; render tests check that the supplied scripts work and their own layout audits pass.

## GPU inference code

`inference/` contains the retained research implementation for source optimization, original-model deployment, cross-model deployment and the source-test supplement, plus the historical prompt-template snapshots accompanying the audit. The supplied JSON configurations preserve the original machine paths as provenance. **Do not run them unmodified on a new machine.**

For a new inference run, install the model/provider dependencies separately, obtain weights under their original licenses, change only storage/resource paths in a new configuration, and check the per-route runtime contracts against `data/rq2/route_manifest.jsonl` or `data/rq3/route_manifest.jsonl`. These contracts record the model revisions, tokenizer/chat-template fingerprints, wrappers, stops, context limits, decoding and vote policy.

Reference import path (without starting a run):

```bash
PYTHONPATH=inference/src:inference/scripts python -m unittest discover -s inference/tests -p 'test_source_test_supplement.py'
```

The code snapshot and saved contracts are supplied for inspection; fresh GPU execution was not performed during packaging. The default replication entry point is the tested offline analysis, not a claim that every historical server path or runtime is immediately portable. Original model weights and long raw response objects are intentionally outside this deposit.
