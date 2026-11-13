# Frozen inputs

Binary labels: `0 = Non-security`, `1 = Security`. NumPy archives use `allow_pickle=False`.

| NPZ array | Shape |
| --- | --- |
| `predictions` | `[route, item]`, majority prediction |
| `votes` | `[route, item, vote]`, actual recorded labels |
| `route_ids` | `[route]`, companion manifest order |
| `gold`, `item_ids` | `[item]` for target stages; `[route, item]` for `source_test` |

Target stages (`rq2`, `rq3`, `concise`) share the GitReq CSV row order. Source-test items are zero-based SecReq row indices, specific to each split. The reproduction adapter expands shared target vectors for the original statistical engines without changing any observation.

Large CSV/JSON/JSONL inputs use lossless XZ compression, readable with Python's `lzma.open`. Runtime and provenance records retain the original file hashes and reuse/inference origins; those hashes describe original records, not the compressed exports. Full model responses and weights are not distributed.

`source/source_30cells_analysis_slim_v1.zip` retains 30 cells, 630 candidate occurrences, exact prompt texts/hashes, 600 proposal events, 96,390 source-validation prediction rows and 12,240 original source-test endpoint rows. Candidate occurrences map to 380 within-cell unique texts and 350 origin-model target routes. Historical selection maximizes validation weighted-F1, breaking ties in favor of later iterations.

`reference.zip` stores comparison tables and paired/cross-validation prediction records. Its `manifest.json` maps each original path to a stored member, content SHA-256 and original encoding; byte-identical tables share a member. The grouped target folds are under the manifest path `analysis/rq2/data/original_tables/target_group_folds.csv`. The reproduction command restores the required layout automatically.

SecReq split seeds are 40–44 (five repeated splits). GitReq has 6,302 rows; its NFR view has 5,771. Both views use the same saved predictions and retain all 1,646 Security rows. No target labels participate in source optimization or historical prompt selection. Target-supervised reselection is a separate analytical comparator.

Some historical source metadata fields are null; these have not been filled with inferred provenance. Historical template code and final inference configurations are retained in `src/inference/`.
