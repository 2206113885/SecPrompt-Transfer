# Original inference implementation

Offline reproduction uses the root `reproduce.py` and does not import this code.

This directory retains the source optimizer (`src/pe2_btape/formal_cell.py`), PE2 and BT-APE implementations, deployment scripts, recorded configurations and protocol tests. The archived run environment used vLLM 0.6.4, PyTorch 2.5.1 with CUDA 12.4 and Transformers 4.46.3; model revisions and stage-specific runtime details are in `data/*/model_runtimes.json[.xz]` and provenance records.

For a new inference run, obtain the checkpoints from their providers, install the appropriate GPU environment, and adapt checkpoint/data/output paths in a copy of the configuration to the local machine. Recorded absolute paths are historical provenance. These are not portable download locations. Run source cells through `scripts/run_formal_matrix.py`; target deployment and the source-test supplement have separate scripts. Consult each script's `--help` before execution. This package does not download weights or rerun inference automatically.

Run protocol tests from this directory with `PYTHONPATH=src python -m unittest discover -s tests`. The lightweight tests and the numerical reproduction command do not verify a fresh GPU run. Earlier prompt-template code is retained under `historical_templates/`; the later template snapshot equals `src/pe2_btape/prompts.py` and is not stored twice.
