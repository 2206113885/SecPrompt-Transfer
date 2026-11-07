# RQ2 protocol: candidate ranking transferability

Version: rq2-ranking-v1. This is a sequential analysis plan frozen after the
observed RQ1 and prior GitReq pilot results. It is not a preregistration and
does not claim the research process was target-blind.

## Question and boundary

RQ2 asks whether the ranking of frozen prompts on SecReq validation data
predicts their ranking on GitReq, and how much target performance is lost by
choosing the source winner. The unit is one unique exact definition text within
one frozen source cell. A cell is origin model × optimizer × source split seed.
There are 30 source cells, each with p0 plus up to 20 generated occurrences.

RQ1 evaluated p0 and the source-selected p* on GitReq. RQ2 evaluates every
remaining frozen definition in the original producer model and uses the same
GitReq observations to analyse ranking and selection loss. RQ3 cross-model
deployment is excluded. No proposer is run, no candidate is generated or
rewritten, and no target information is returned to source optimization.
Fixed/Broken error transitions are supplementary RQ1 analyses and are not RQ2.

## Source and target

The source of truth is the frozen 30-cell tar archive with SHA-256
6980c68109639305d08f7cbaa4669ed3ccbeb5f3fed1908914a1efc0ec95ad06.
SecReq contains 510 rows; every seed splits all rows into 153 pool, 153
validation, and 204 test rows. Source rankings use the original per-cell
validation predictions and are independently recomputed from three-vote
majorities. Historical p* remains the maximum validation weighted-F1
candidate, breaking exact ties in favor of the newer iteration.

The target is GitReq v3 enriched CSV with SHA-256
d765fb052feccb0fc0d031595b855834637d03662d798c090c309e32b6b772e7.
Security is positive and all seven other original categories are negative.
Both fixed views are reported from one prediction pass: T_All has 6,302 rows;
T_NFR removes the 531 Functional rows and has 5,771 rows. A case-insensitive
whole-word security lexical mask is a sensitivity analysis, intersected with
each view while retaining both binary classes.

## Candidate and route deduplication

All 630 iteration records and their order remain in candidate lineage. Main
ranking pools deduplicate only byte-identical UTF-8 definition text within a
cell. Similar wording is never clustered. Repeated occurrences retain all
prompt IDs and iterations. If their source validation prediction vectors
differ, the latest occurrence represents that text in the unique-text main
analysis; a sensitivity analysis keeps all occurrences.

Deployment routes are deduplicated across cells only when origin model,
complete RQ1 interface/runtime fingerprint, exact definition text, GitReq
item mapping, and vote protocol match. Cache identity includes model weights
and revision, tokenizer/chat template, classifier wrapper, stop/output
settings, dataset hash, and prompt bytes. A prompt hash by itself is not a
cache key. RQ1 p0/p* predictions are reused only after validating route
markers, prediction file hashes, all ordered item/gold/repository mappings,
three valid votes, and exact rendered request hashes.

## Inference contract

Use the frozen RQ1 model-specific classifier interface and runtime unchanged:
local pinned HF safetensors, canonical local tokenizer/chat template,
vLLM 0.6.4, V100, FP16, temperature 0, top-p 1, top-k -1, engine seed 0,
max model length 8,192, classification context cap 2,048, generation cap 64,
dispatch batch size 1, and lm-format-enforcer choices
Non-security/Security. Llama and Granite use the RQ1 Appendix-F wrapper;
Qwen uses the RQ1 package_minimal wrapper. Each item gets three actual
logical votes and a majority prediction. No target label, category, or
repository metadata is sent in the classifier request.

Before any new candidate request, audit the complete frozen source archive and
target mapping, verify exact RQ1 reuse, and check every unique route × target
item token length against the context cap. Then run a fixed three-item
serialization/classification smoke check per model without using its accuracy
to select settings. Failed, empty, truncated, or invalid responses are
preserved in attempt logs and retried at most twice per logical vote; only
three valid accepted votes produce a completed item. Partial routes are
resumable by ordered item prefix. No invalid response is converted to a class.

## Ranking analyses

Within each cell's unique-text pool, separately report each GitReq view and:

1. Primary: source validation weighted-F1 versus target Macro-F1.
2. Metric-aligned sensitivity: source validation Macro-F1 versus target
   Macro-F1.
3. Supplementary: source validation weighted-F1 versus target weighted-F1.

Use average ranks for Spearman rho and explicit tie-aware Kendall tau-b.
Scores within 1e-12 are ties. Constant-score and single-candidate cases are
NA with a reason; pools of 2–4 candidates are retained and marked
low-resolution. Report all cells and a separate n_unique≥5 sensitivity.
Occurrence-level ranking is supplementary only.

For each cell/view, report Top-1/3/5 overlap when k≤n, tie-inclusive Jaccard,
complete first-place tie sets, source winner's target average/best/worst rank,
and normalized rank position. If k=n, mark the result trivial and exclude it
from effective Top-k summaries.

The historical source p* stays fixed. Report its target Macro-F1, the best
score among the same frozen candidates, target score range, absolute
in-sample candidate-pool oracle gap in percentage points, normalized gap
only when range is nonzero, and the unique-candidate uniform-random expected
score. Report within 0.5, 1, and 2 pp of observed pool best; 1 pp is a
descriptive practical-nearness threshold, not a statistical equivalence
margin. Pool-best gap intervals recompute the maximum in each bootstrap and
remain conditional/descriptive, not optimism-corrected estimates.

## Target-selected diagnostic comparator

Before candidate scores are examined, assign all repositories to five
mutually exclusive folds using seed 20261013 and the recorded
greedy_group_multilabel_balance_v1 algorithm. All cells and both views share
the same repository assignment. In each fold, choose among the frozen
candidates using only the other four folds' target Macro-F1; evaluate the
chosen candidate, historical p*, and p0 only on the held-out fold. Report
fold-paired scores and pooled out-of-fold metrics from concatenated held-out
predictions. This is a target-supervised diagnostic on the known GitReq
corpus, not a source-only method or an external-dataset estimate.

## Uncertainty

Use 2,000 repository-cluster bootstrap replicates and analysis seed 20261013.
Each replicate resamples repository IDs with replacement and preserves item
multiplicity. Every candidate, cell, and view uses the same repository draw;
T_All and T_NFR are masks of the same resampled rows. Percentile 95% intervals
and valid replicate counts are reported for cell ranking and selection-gap
statistics. Undefined replicates remain NA.

Separately use 2,000 stratified item bootstrap replicates inside each
153-item SecReq validation set. The same per-seed draw is shared by its six
cells and all candidates. Report p* selection frequency, candidate winner
frequency, and bootstrap ranking stability. These are conditional diagnostics
for the fixed observed candidates; they do not remove adaptive-search bias or
represent new APE runs. The five source splits overlap and are not treated as
independent datasets.

Summarize first by cell, then equally across the five observed seeds within
each origin model × optimizer. Candidate rows, routes, votes, and item
predictions are not independent experimental replicates. Do not use nominal
candidate-level correlation p-values as primary evidence.

## Interpretation limits

The same-corpus pool maximum is an in-sample candidate-pool oracle, not a
deployable target upper bound. Low ranking correlation can coexist with a
small practical gap when candidate scores are close; high correlation can
coexist with a meaningful first-place gap. The analysis is conditional on
these three origin models, two optimizers, SecReq and GitReq, and the frozen
candidate pools. Model wrappers differ, so absolute cross-model score
differences are not pure model effects. No full-candidate source-test scores
are inferred from the historical p0/p* endpoint-only test records.
