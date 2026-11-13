# SecPrompt-Transfer

Replication package for **Reusing Automatically Optimized Prompts Across Datasets and Models: An Empirical Study of Security Requirement Classification**.

**Paper version:** [v1.0.1](https://github.com/2206113885/SecPrompt-Transfer/releases/tag/v1.0.1). Use this release when reproducing the submitted manuscript; `main` may change. This documentation update explains the auxiliary normalized gap in [data formats](data/README.md); data, code and frozen results are identical to v1.0.0.

## Reproduce

Python 3.12; CPU only. The commands use frozen predictions and make no model calls.

```bash
uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python reproduce.py --check
uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python reproduce.py --all
```

Alternatively, install `requirements.txt` in a Python 3.12 virtual environment and run the same Python commands.

- `--check`: verify file hashes, recorded votes, source prompt selection, classification scores, candidate ranks/gaps on the source test set, and SecReq duplicate-text sensitivity.
- `--all`: also recompute RQ1–RQ3 statistical tables and bootstrap intervals, compare them against frozen reference tables (absolute tolerance `1e-12`), and regenerate the manuscript tables and figures.

Outputs appear in `results/verification.json`, `results/tables/`, `results/figures/`, `results/source_test/` and `results/source_duplicates/`. Temporary expanded inputs are removed after execution. The source-test and duplicate analyses are post hoc diagnostics; the historical prompts and predictions remain fixed. A successful check does not establish that newly generated model outputs will be identical.

## Contents

| Path | Purpose |
| --- | --- |
| `data/` | Datasets, split IDs, candidate texts, saved predictions/votes and run provenance; [formats](data/README.md) |
| `src/analysis/` | RQ1–RQ3 statistical engines and duplicate-text sensitivity |
| `src/paper/` | Table and figure generators, sharing `src/figure_style.py` |
| `src/workflow/` | Editable Figure 1 (`.drawio`) and its vector PDF |
| `src/inference/` | Original optimization/deployment implementation, configurations and protocol tests; [usage](src/inference/README.md) |
| `CITATION.cff`, `LICENSES/` | Citation metadata and component attribution |

The package stores each target label/ID vector once per stage and compresses text inputs losslessly. `data/reference.zip` contains deduplicated comparison tables and auxiliary prediction records required by the analysis engines. The entry point restores their original paths automatically. Generated previews, duplicate styles, unused configurations and intermediate revision materials are excluded. Figures use the recorded style with installed font fallbacks; numerical checks do not compare PDF bytes.

## Attribution

- SecReq: [original dataset](https://doi.org/10.5281/zenodo.4530183).
- GitReq v3: [original dataset](https://doi.org/10.6084/m9.figshare.31669477.v3).
- PE2: [original implementation](https://github.com/INK-USC/PE2).
- BT-APE: [original paper](https://doi.org/10.48550/arXiv.2607.00427).

Please cite the accompanying paper, this version and the original datasets/methods. Component terms are recorded in [LICENSES](LICENSES/README.md); no blanket license is asserted for the authors' original code.
