# SecPrompt-Transfer

Materials for **How Well Do Automatically Optimized Prompts Transfer? An Empirical Study of Security Requirement Classification**

Yizheng He¹, Zhiwei Fei¹, Jidong Ge¹, and Liguo Huang².

¹ National Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China.

² Department of Computer Science, Southern Methodist University, Dallas, USA.

**Research data and code, 7 November 2026.** SecPrompt-Transfer studies whether automatically optimized security-requirement prompts transfer across datasets and deployment models. The repository accompanies the manuscript above; publication details will be added when available.

Repository: [2206113885/SecPrompt-Transfer](https://github.com/2206113885/SecPrompt-Transfer).

## What this package reproduces

The study optimizes requirement-class definitions on SecReq and evaluates the unchanged definitions on GitReq. This package contains the frozen inputs, candidate texts and lineage, saved predictions, scoring and statistical analysis code, and manuscript figure/table scripts. It supports offline reproduction without downloading model weights or calling an API.

| Part | Included evidence |
| --- | --- |
| Source optimization | 30 cells, 630 candidate occurrences, 600 iteration events, validation votes, original endpoint test votes, and five split manifests |
| RQ1 | 30 source-selected prompts versus the initial definition on their original models; paired predictions and analysis |
| RQ2 | 350 deduplicated target routes, 2,205,700 route–item predictions and their three recorded votes; candidate mappings, grouped target folds and analysis |
| RQ3 | 189 deduplicated target routes across seven deployers; 210 logical winner cells plus seven baselines; one recorded vote per item |
| Source-test supplement | All 380 cell–candidate routes on the original 204-item test partitions, including the reused endpoints; 77,520 predictions and three votes per item |
| Concise-definition supplement | Three target routes, 18,906 predictions and three votes per item; the exact fixed definition |

Repeated prompts retain their original cell/iteration identities, while identical deployment routes are stored once. Route counts therefore differ from the number of logical comparisons. RQ1 endpoint predictions are checked against the corresponding RQ2 routes; they are not separate new inference runs in this deposit.

## Quick start (CPU only)

Use Python 3.12 and the pinned packages in `requirements.txt`. With `uv`:

```bash
uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python reproduce.py --check
uv run --python 3.12 --isolated --no-project --with-requirements requirements.txt python reproduce.py --full
```

Alternatively:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py --check
.venv/bin/python reproduce.py --full
```

`--check` verifies package checksums, votes and majority labels, all target prediction dimensions, source-validation scores, historical source winners, RQ1/RQ2 endpoint agreement, and prediction-level scores for RQ2, RQ3 and both supplements. It writes `reproduced/verification.json` and reconstructed supplement statistics. `--full` additionally reruns the retained RQ1 and RQ2 analyses and the RQ3 repository-cluster bootstrap, including their test suites. **Neither command uses a GPU or invokes a model.** Outputs go to `reproduced/`; original evidence is retained.

RQ1 and RQ2 use 2,000 bootstrap replicates with their original analysis seeds. RQ3 uses 10,000 replicates, seed 20261023, and the original statistical engine with a compact-input adapter. The adapter replaces only the raw-response loader, not the estimands, resampling, metrics or intervals. See `docs/REPRODUCTION.md` for outputs and rerun behavior.

## Layout

```text
data/
  datasets/              Exact SecReq and enriched GitReq CSV versions
  source/                Frozen 30-cell archive: prompts, votes and lineage
  splits/                Five shared source split manifests
  rq2/                   Target predictions, votes, route contracts and provenance
  rq3/                   Seven-model predictions, route and logical-cell mappings
  source_test/           All 380 source-test candidate prediction vectors
  concise/               Fixed concise-definition predictions and contract
analysis/
  rq1/                   Portable original offline analysis and input evidence
  rq2/                   Portable original offline reanalysis and grouped folds
  rq3/                   Original statistical engine and compact-input adapter
paper/
  data/                  Frozen tables used by the manuscript
  tables/                Generated LaTeX tables
  figures/               Vector PDF figures and RQ3 plotting scripts
  tools/                 Table/figure and case-analysis scripts
inference/               Research implementation and historical template snapshots
docs/                    Data dictionary, execution notes and release checklist
```

Some small inputs are intentionally repeated inside the historical analysis directories so their relative-path-based scripts remain runnable. Model weights, environments, API credentials, SSH settings, unrelated pilot runs, article PDFs and the complete server backup are excluded. Full long-form inference responses remain in the authors' local archive; this package contains the recorded vote labels, predictions and file-level provenance needed for independent scoring.

## Recreate manuscript assets

```bash
python paper/tools/review_supplements.py
python paper/tools/test_review_supplements.py
python paper/tools/build_source_test_reference.py
python paper/tools/build_concise_definition_table.py
python paper/tools/build_results_assets.py
python paper/figures/build_rq3_figures.py
python paper/figures/build_rq3_matrix_overview.py
```

Run these commands in the environment installed above. They operate on copied evidence inside this repository; they never edit the original experiment archive. Vector PDFs are included for convenience. The scripts that render plots use the recorded reference statistics rather than launching inference.

## Data sources and attribution

- **SecReq:** Knauss et al., [Zenodo record 4530183](https://doi.org/10.5281/zenodo.4530183), CC BY 4.0. The exact 510-row BT-APE-derived CSV used by this study is retained, not silently deduplicated or repartitioned.
- **GitReq:** Kamal, Kabir and Islam, [Figshare v3](https://doi.org/10.6084/m9.figshare.31669477.v3), recorded as CC BY 4.0 in the original dataset audit. The enriched CSV retains the requirement text, category, issue URL and repository; it joins the official NFR and Functional partitions and adds study IDs and binary labels. Security is positive; the other seven categories are negative. `T_All` contains 6,302 items and `T_NFR` excludes the 531 Functional items.
- **BT-APE:** Zadenoori et al., [paper](https://doi.org/10.48550/arXiv.2607.00427). The locally preserved author package was consulted for its installation, execution, configuration and test organization; its MIT notice is retained in `LICENSES/BTAPE-MIT.txt`. The historical GitHub link returned 404 when this release was assembled.
- **PE2:** Ye et al., [author repository](https://github.com/INK-USC/PE2). Its separation of environment, data/preprocessing and runnable experiment examples informed the package organization. This study's PE2 is the requirements-classification adaptation described in the manuscript, not an assertion that the original PE2 implementation was run unchanged.

See `LICENSES/README.md` for component-specific rights. Existing dataset and third-party code licenses are retained; no blanket license has been assigned to the authors' original code. No archive DOI has been assigned.

## Scientific scope

The historical source winners, candidate text, source splits and target interfaces are unchanged. Source-test supplement scores refer to the original 204-row partitions. RQ2 reports candidate-pool ranking and selection diagnostics; it is not a new prompt optimizer. RQ3 uses one-vote deployment after the recorded consistency checks, while RQ1/RQ2 and both supplements retain three votes. Read `docs/DATA_DICTIONARY.md` before combining these stages.

Please cite the manuscript and the original datasets and methods. `CITATION.cff` identifies this research repository; the manuscript's publication metadata will be added after acceptance.
