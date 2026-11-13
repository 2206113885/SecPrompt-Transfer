# SecPrompt-Transfer

Replication package for **Reusing Automatically Optimized Prompts Across Datasets and Models: An Empirical Study of Security Requirement Classification**.

Yizheng He, Zhiwei Fei, Jidong Ge, and Liguo Huang.

## Study workflow (Figure 1)

The redesigned study workflow is available as a [vector PDF](paper/figures/study-workflow/Fig1-study-workflow.pdf), [SVG](paper/figures/study-workflow/Fig1-study-workflow.svg), and [PNG preview](paper/figures/study-workflow/Fig1-study-workflow.png). The [editable Draw.io source](paper/figures/study-workflow/Fig1-study-workflow.drawio), canonical YAML, and rebuild instructions are in [paper/figures/study-workflow](paper/figures/study-workflow/README.md).

## Reproduce the results

The package includes the SecReq and GitReq study datasets, all frozen prompt candidates, saved prediction votes, analysis code, and figure-generation scripts. The statistical results can be reproduced on a CPU:

```bash
uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python reproduce.py --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python reproduce.py --full
```

`--check` verifies the datasets, prompt-selection rule, recorded votes, classification scores, and agreement between the RQ1 endpoints and RQ2 candidate routes. `--full` regenerates the RQ1–RQ3 statistical tables and intervals. Results are written to `reproduced/`.

To regenerate the manuscript figures and supplementary tables:

```bash
uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python verify_assets.py
```

The generated assets are written to `reproduced/paper/`. The saved vote labels and predictions are the inputs to these commands; no model download is needed for offline reproduction. `inference/` contains the source-optimization and deployment implementation for a new model run.

## SecReq duplicate-text sensitivity

The [duplicate-text supplement](analysis/source_duplicates/README.md) quantifies
text overlap in the original source partitions and rescores the frozen predictions
after excluding that overlap. It includes per-cell results, a standalone
reproduction script, and Figure 11 in vector PDF/SVG and PNG. This post hoc
analysis retains the original candidate pools and historical selected definitions.

## Contents

| Directory | Contents |
| --- | --- |
| `data/` | SecReq and GitReq, 30 frozen source trajectories, split IDs, candidate-route manifests, prediction arrays and vote labels |
| `analysis/` | RQ1, RQ2 and RQ3 statistical analyses and their comparison tables |
| `paper/` | Figure and supplementary-table generators with the summary data they use |
| `inference/` | Source optimization, target deployment, configurations and tests |
| `docs/DATA_DICTIONARY.md` | Prediction-array fields, labels and evaluation views |

`data/source/source_30cells_analysis_slim_v1.zip` contains all 630 source candidate occurrences and their validation predictions. The deduplicated target predictions are stored once per route in `data/rq2/` and `data/rq3/`; the route manifests map them back to the experimental comparisons.

## Data sources

- SecReq: Knauss et al., [Zenodo](https://doi.org/10.5281/zenodo.4530183).
- GitReq v3: Kamal, Kabir and Islam, [Figshare](https://doi.org/10.6084/m9.figshare.31669477.v3).
- BT-APE: Zadenoori et al., [paper](https://doi.org/10.48550/arXiv.2607.00427).
- PE2: Ye et al., [code](https://github.com/INK-USC/PE2).

Third-party notices are in `LICENSES/`. Please cite the paper and the original datasets when using this package.
