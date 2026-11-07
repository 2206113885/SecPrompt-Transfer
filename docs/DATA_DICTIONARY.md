# Data dictionary

## Binary encoding and prediction arrays

`data/{rq2,rq3,source_test,concise}/predictions.npz` is a NumPy compressed archive, readable with `numpy.load(path, allow_pickle=False)`.

| Array | Meaning |
| --- | --- |
| `predictions` | `[route, item]`, unsigned byte; 0 = Non-security, 1 = Security |
| `votes` | `[route, item, vote]`; actual saved vote labels, not duplicated majority predictions |
| `gold` | `[route, item]`, same encoding |
| `item_ids` | `[route, item]`; GitReq string IDs, or SecReq zero-based row indices |
| `route_ids` | `[route]`, identifiers in route-manifest order |

Each array's first axis matches the companion `route_manifest.jsonl` (or the `routes` array of `route_manifest.json` for the concise baseline). Target stages have the same item order as the enriched GitReq CSV. Source-test item order is specific to each source split and is listed in its route manifest. `source_test` route IDs in the NPZ combine the full `cell_id` and prompt hash; no cross-cell validation score is reused.

For each stage, `prediction_provenance.json` records original prediction-file names and hashes, vote count, row count and reuse/inference origin. Original completion markers are included where present. These are original-record hashes, not hashes of the compact export. Full original responses are not required for metric recomputation and are not included.

Example (no inference):

```python
import numpy as np
a = np.load('data/source_test/predictions.npz', allow_pickle=False)
i = 0
for item, gold, pred, votes in zip(a['item_ids'][i], a['gold'][i],
                                 a['predictions'][i], a['votes'][i]):
    print(a['route_ids'][i], item, gold, pred, votes.tolist())
```

## Frozen source archive

`data/source/source_30cells_analysis_slim_v1.zip` contains:

- `cells.jsonl`: 30 origin-model × optimizer × split-seed cells, stored winner, split IDs and endpoint metrics.
- `candidates.jsonl`: 630 prompt occurrences (iterations 0–20), exact text, prompt ID/hash, validation metrics and selected flags.
- `iteration_events.jsonl`: 600 proposal events with parent/current iteration, backtracking, history and error-feedback identifiers.
- `validation_predictions.jsonl`: 96,390 per-item prediction rows, each with three saved vote labels.
- `source_test_predictions.jsonl`: 12,240 original p0/p* endpoint rows; the later full-pool source-test results are in `data/source_test/`.
- Matrix manifests, export audit, and the exact source CSV.

Some legacy compact-source metadata fields are null. Runtime/model revisions and classifier interfaces are recorded in the target route contracts; null source fields are not inferred or filled with invented provenance. Historical prompt-template snapshots are supplied separately.

## Statistical units and views

The 630 occurrences map to 380 within-cell unique prompt texts and 350 origin-model/target deployment routes. A repeated text is not a new independent candidate. The main RQ2 analysis includes p0; generated-only and occurrence-weighted analyses remain separately named sensitivities. Historical winners maximize source-validation weighted-F1, with newer iterations winning exact ties.

`T_All` and `T_NFR` are fixed masks of the same saved target predictions. Both retain all 1,646 Security requirements. Grouped target-calibration/evaluation fold IDs are in `analysis/rq2/data/original_tables/target_group_folds.csv`. Source split seeds are 40–44, not five-fold source cross-validation.

No test partition, candidate, selected prompt or prediction was changed during packaging. The complete raw archives remain separately preserved by the authors.
