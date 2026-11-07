# Frozen-candidate SecReq source-test diagnostic, v1

This is a **post-hoc supplementary diagnostic**, approved after inspecting RQ1,
RQ2 and RQ3. It is not preregistration, a target-blind study, a fresh independent
test of the entire research process, or a correction of the original PE2 runs.

## Unchanged evidence and inference

Preserve the 30 original trajectories, 630 occurrences and 380 cell-unique
texts, original split membership/order, definitions, p0 and historical p*.
Do not run proposers, extract/clean generated definitions, add demonstrations,
select a new deployed champion, or overwrite source/RQ1/RQ2/RQ3 artifacts.
Evaluate every cell-unique text on that cell's original 204 SecReq test items.
Repeated texts within a cell share one inference route but retain all lineage.
There is no cross-cell reuse in this supplement: conservative endpoint reuse
is confined to each original cell and its identical prompt text.

Use the original HF weights and tokenizer, FP16 vLLM 0.6.4, XFORMERS, batch 1,
temperature 0, top-p 1, top-k -1, engine seed 0, max model length 8192,
classification context 2048, generation cap 64, original stop strings and
guided binary choice. Llama/Granite use appendix_f; Qwen uses package_minimal.
Every new item has **three actual model calls**. No vote replication. Only
GPU0 is used, with Llama, Qwen, Granite in sequential model processes.

The existing CSV loader's strip operation is retained; the exact stored prompt
string is not stripped. Gold, IDs and split metadata never enter the request.
All requests are token-audited before inference; no new truncation rule.
Only stop-terminated valid binary votes complete new tasks. Errors, truncation
or invalid labels are retained as failures, not coerced to a label.

## Audit and durable resume

Check the archive and dataset hashes, current full-result hashes against the
previously frozen RQ3 manifest, 30/630/380 counts, split disjointness, shared
seed splits, source definitions, empty demo lists, vote-derived validation
metrics and unchanged historical best (latest iteration for exact ties).
For endpoints check all three raw/parsed votes, majority, original request,
rendered request hash, model runtime and metrics before reuse. Reuse preserves
original vote metadata and points to the original full-result SHA-256.

Freeze configuration, code hashes, manifests and runtime fingerprints. Cache
keys include cell, exact prompt, ordered test IDs, dataset hash, model runtime,
wrapper, decoding and votes. Each returned vote is fsynced to an append-only
journal. A completion marker is atomic and includes prediction-file SHA-256;
it is valid only for 204 distinct expected IDs with three valid votes each.
A truncated final journal fragment is preserved separately before removing
only the fragment; semantic errors in complete records block recovery.
File locking prevents two controllers; GPU0 availability is checked before
each model process. PAUSE stops between votes. Restart uses the same immutable
manifest, revalidates finished routes and completes only missing votes.

## Offline analyses after full coverage

Main unit: a unique frozen prompt **within its original cell**. Keep 630
occurrence mappings. All 30 cells, including low-diversity PE2, remain reported.
Join unchanged RQ2 target scores for T_All and T_NFR; do not run target inference.

Report per-cell Kendall tau-b and Spearman average-rank rho:

- validation weighted-F1 -> source-test weighted-F1;
- validation Macro-F1 -> source-test Macro-F1;
- source-test weighted-F1 -> target weighted-F1;
- source-test Macro-F1 -> target Macro-F1;
- validation weighted-F1 -> target Macro-F1 (original operational comparison).

Numerical ties use 1e-12 anchored groups; constant vectors produce NA and
n<5 is marked low_resolution. Historical p* remains fixed irrespective of
new scores. Report its source-test weighted/Macro-F1 and target Macro-F1,
p0 gains, within-pool oracle gap, range, normalized gap (NA for zero range),
uniform-unique expected score, and rank. Oracle gaps are in-sample descriptive
pool comparisons, not deployable improvements or unbiased maxima. Summaries
weight cells equally and expose all five seeds. No nominal candidate-level
p-values or claims that seed/candidate/vote observations are independent.

The automatic report provides descriptive diagnostics, not new confidence
intervals. Further inferential analysis, if added, must be separately versioned
and labelled post-hoc. These comparisons provide a within-corpus reference;
they do not causally separate dataset shift, adaptive validation overfitting
and finite-sample noise. The original source test was already viewed at p0/p*.
