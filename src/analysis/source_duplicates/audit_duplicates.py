#!/usr/bin/env python3
"""Descriptive re-scoring of frozen predictions; no inference or source edits.

Run: uv run --isolated --no-project --with numpy python audit_duplicates.py \
       --repository-root PATH --output-dir PATH
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import unicodedata
import zipfile

import numpy as np

TOL = 1e-12


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(gold, pred, weights=None):
    gold, pred = np.asarray(gold), np.asarray(pred)
    w = np.ones(len(gold)) if weights is None else np.asarray(weights, dtype=float)
    assert len(gold) == len(pred) == len(w) and len(gold)
    assert set(gold.tolist()) == {0, 1} and set(pred.tolist()) <= {0, 1}
    f1, supports = [], []
    for label in [0, 1]:
        tp = w[(gold == label) & (pred == label)].sum()
        fp = w[(gold != label) & (pred == label)].sum()
        fn = w[(gold == label) & (pred != label)].sum()
        f1.append(float(2 * tp / (2 * tp + fp + fn)))
        supports.append(float(w[gold == label].sum()))
    return {
        "macro_f1": sum(f1) / 2,
        "weighted_f1": float(np.dot(f1, supports) / w.sum()),
        "accuracy": float(w[gold == pred].sum() / w.sum()),
    }


def tied_ranks(values):
    # Same anchored 1e-12 tie grouping as the original source-test analysis.
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    p = 0
    while p < len(order):
        q = p + 1
        while q < len(order) and abs(values[order[q]] - values[order[p]]) <= TOL:
            q += 1
        for idx in order[p:q]:
            ranks[idx] = (p + 1 + q) / 2
        p = q
    return ranks


def tau(a, b):
    x, y = tied_ranks(a), tied_ranks(b)
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )


def distribution(values):
    v = [float(x) for x in values if x is not None]
    return {
        "n": len(v), "mean": statistics.mean(v),
        "sd_descriptive": statistics.stdev(v) if len(v) > 1 else 0,
        "median": statistics.median(v), "min": min(v), "max": max(v),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.repository_root, args.output_dir
    out.mkdir(exist_ok=True, parents=True)
    inputs = [
        "data/datasets/SecReq_510_BTAPE.csv", "data/datasets/gitreq_enriched_v3.csv",
        "data/source/source_30cells_analysis_slim_v1.zip",
        "data/source_test/predictions.npz", "data/source_test/route_manifest.jsonl",
        "paper/data/source-test-supplement/analysis/candidate_scores.csv",
        "paper/data/source-test-supplement/analysis/cell_rank_reference.csv",
        "paper/data/source-test-supplement/analysis/cell_selection_gaps.csv",
        *[f"data/splits/seed{s}.json" for s in range(40, 45)],
    ]
    hashes = {p: sha(root / p) for p in inputs}
    rows = read_csv(root / inputs[0])
    texts = [normalized(row["text"]) for row in rows]
    labels = np.array([int(row["label"] == "Security") for row in rows])
    assert len(rows) == 510
    groups = defaultdict(list)
    exact = defaultdict(list)
    for i, row in enumerate(rows):
        groups[texts[i]].append(i)
        exact[row["text"]].append(i)
    assert sorted(sorted(x) for x in groups.values()) == sorted(sorted(x) for x in exact.values())
    assert all(len(set(labels[g])) == 1 for g in groups.values())
    duplicate_groups = [g for g in groups.values() if len(g) > 1]
    assert len(groups) == 444 and len(duplicate_groups) == 66
    splits = {s: json.loads((root / f"data/splits/seed{s}.json").read_text()) for s in range(40, 45)}
    split_audit = []
    masks, unique_weights, val_masks = {}, {}, {}
    for seed, split in splits.items():
        pool, val, test = [split[k] for k in ["pool_ids", "val_ids", "test_ids"]]
        assert len(pool) == len(val) == 153 and len(test) == 204
        assert sorted(pool + val + test) == list(range(510))
        pool_text = {texts[i] for i in pool}
        seen = pool_text | {texts[i] for i in val}
        keep = np.array([texts[i] not in seen for i in test])
        vkeep = np.array([texts[i] not in pool_text for i in val])
        freq = Counter(texts[i] for i in np.asarray(test)[keep])
        weights = np.array([1 / freq[texts[i]] for i in np.asarray(test)[keep]])
        masks[seed], unique_weights[seed], val_masks[seed] = keep, weights, vkeep
        split_audit.append({
            "seed": seed, "pool_rows": len(pool), "validation_rows": len(val), "test_rows": len(test),
            "validation_rows_matching_pool": int((~vkeep).sum()),
            "validation_overlap_percent": float((~vkeep).mean() * 100),
            "test_rows_matching_pool_or_validation": int((~keep).sum()),
            "test_overlap_percent": float((~keep).mean() * 100),
            "unseen_test_rows": int(keep.sum()), "unseen_unique_test_texts": len(freq),
            "original_security_rows": int(labels[test].sum()),
            "unseen_security_rows": int(labels[np.asarray(test)[keep]].sum()),
            "unseen_unique_security_texts": sum(labels[g[0]] for t, g in groups.items() if t in freq),
        })
    archive = root / "data/source/source_30cells_analysis_slim_v1.zip"
    with zipfile.ZipFile(archive) as z:
        assert z.read("SecReq_510_BTAPE.csv") == (root / inputs[0]).read_bytes()
        cell_records = [json.loads(line) for line in z.read("cells.jsonl").splitlines()]
        candidates = [json.loads(line) for line in z.read("candidates.jsonl").splitlines()]
        val_predictions = defaultdict(dict)
        for line in z.read("validation_predictions.jsonl").splitlines():
            item = json.loads(line)
            assert item["gold"] == rows[item["item_id"]]["label"]
            assert item["prediction"] in ["Security", "Non-security"]
            assert Counter(item["vote_labels"]).most_common(1)[0][0] == item["prediction"]
            assert item["item_id"] not in val_predictions[item["prompt_id"]]
            val_predictions[item["prompt_id"]][item["item_id"]] = int(item["prediction"] == "Security")
    assert len(candidates) == len(val_predictions) == 630 and len(cell_records) == 30
    for cell in cell_records:
        assert cell["split"] == {k: splits[cell["seed"]][k] for k in ["pool_ids", "val_ids", "test_ids"]}
    candidates_by_id = {c["prompt_id"]: c for c in candidates}
    manifest = [json.loads(line) for line in (root / "data/source_test/route_manifest.jsonl").read_text().splitlines()]
    archive_predictions = np.load(root / "data/source_test/predictions.npz", allow_pickle=False)
    assert len(manifest) == len(archive_predictions["route_ids"]) == 380
    assert set(np.unique(archive_predictions["votes"])) <= {0, 1}
    assert np.array_equal(archive_predictions["predictions"], (archive_predictions["votes"].sum(axis=2) >= 2).astype("uint8"))
    previous = read_csv(root / "paper/data/source-test-supplement/analysis/candidate_scores.csv")
    old = {(r["cell_id"], r["prompt_sha256"], r["view"]): r for r in previous}
    cell_routes = defaultdict(list)
    candidate_results = []
    max_score_error = 0.0
    duplicate_prediction_disagreements = 0
    for ri, route in enumerate(manifest):
        cid, ph, seed = route["cell_id"], route["prompt_sha256"], route["seed"]
        assert archive_predictions["route_ids"][ri] == cid + "__" + ph
        ids = archive_predictions["item_ids"][ri]
        assert list(ids) == splits[seed]["test_ids"] == route["test_ids"]
        y, pred = archive_predictions["gold"][ri], archive_predictions["predictions"][ri]
        assert np.array_equal(y, labels[ids])
        keep = masks[seed]
        views = {
            "original_rows": score(y, pred),
            "unseen_rows": score(y[keep], pred[keep]),
            "unseen_equal_text_weights": score(y[keep], pred[keep], unique_weights[seed]),
        }
        v_ids = splits[seed]["val_ids"]
        occurrence = candidates_by_id[route["candidate_id"]]
        assert occurrence["prompt_sha256"] == ph
        vdict = val_predictions[route["candidate_id"]]
        assert set(vdict) == set(v_ids)
        vpred = np.array([vdict[i] for i in v_ids])
        for oid in route["occurrence_ids"]:
            assert val_predictions[oid] == vdict, "Repeated candidate text has different validation predictions"
        vs = score(labels[v_ids], vpred)
        vkeep = val_masks[seed]
        vs_clean = score(labels[np.asarray(v_ids)[vkeep]], vpred[vkeep])
        ref = old[cid, ph, "T_All"]
        for metric in ["macro_f1", "weighted_f1", "accuracy"]:
            err = abs(views["original_rows"][metric] - float(ref["source_test_" + metric]))
            max_score_error = max(max_score_error, err)
            assert err < TOL
            assert abs(vs[metric] - float(ref["validation_" + metric])) < TOL
            assert abs(vs[metric] - route["source_validation_metrics"][metric]) < TOL
        for inds in groups.values():
            positions = [j for j, i in enumerate(ids) if i in inds and keep[j]]
            if len(positions) > 1 and len(set(pred[positions])) > 1:
                duplicate_prediction_disagreements += 1
        route_result = {
            "route": route, "scores": views, "validation": vs,
            "validation_without_pool_overlap": vs_clean,
            "target": {v: {metric: float(old[cid, ph, v]["target_" + metric]) for metric in ["macro_f1", "weighted_f1"]} for v in ["T_All", "T_NFR"]},
        }
        cell_routes[cid].append(route_result)
        for view, scores in views.items():
            candidate_results.append({"cell_id": cid, "prompt_sha256": ph, "seed": seed, "source_test_view": view, **scores})
    cell_results, validation_results = [], []
    for cid, rr in sorted(cell_routes.items()):
        selected = next(i for i, r in enumerate(rr) if r["route"]["is_selected_best"])
        initial = next(i for i, r in enumerate(rr) if r["route"]["is_p0"])
        assert sum(bool(r["route"]["is_selected_best"]) for r in rr) == 1
        assert sum(bool(r["route"]["is_p0"]) for r in rr) == 1
        r0 = rr[0]["route"]
        meta = {k: r0[k] for k in ["cell_id", "model", "optimizer", "seed"]}
        val = [r["validation"]["macro_f1"] for r in rr]
        full = [r["scores"]["original_rows"]["macro_f1"] for r in rr]
        for view in ["original_rows", "unseen_rows", "unseen_equal_text_weights"]:
            v = [r["scores"][view]["macro_f1"] for r in rr]
            wf = [r["scores"][view]["weighted_f1"] for r in rr]
            cell_results.append({
                **meta, "source_test_view": view, "candidate_count": len(rr),
                "pstar_macro_f1_pp": 100 * v[selected], "p0_macro_f1_pp": 100 * v[initial],
                "gain_over_p0_pp": 100 * (v[selected] - v[initial]),
                "selection_gap_pp": 100 * (max(v) - v[selected]),
                "selection_gap_weighted_f1_pp": 100 * (max(wf) - wf[selected]),
                "selected_is_pool_best": max(v) - v[selected] <= TOL,
                "selected_within_one_pp": 100 * (max(v) - v[selected]) <= 1 + TOL,
                "validation_test_tau_b": tau(val, v),
                "test_target_all_tau_b": tau(v, [r["target"]["T_All"]["macro_f1"] for r in rr]),
                "test_target_nfr_tau_b": tau(v, [r["target"]["T_NFR"]["macro_f1"] for r in rr]),
                "original_subset_test_tau_b": tau(full, v),
            })
        original_val = [r["validation"]["weighted_f1"] for r in rr]
        clean_val = [r["validation_without_pool_overlap"]["weighted_f1"] for r in rr]
        assert abs(original_val[selected] - max(original_val)) < TOL
        original_tie_winner = max((i for i, v in enumerate(original_val) if max(original_val) - v <= TOL), key=lambda i: rr[i]["route"]["last_iteration"])
        assert original_tie_winner == selected, "Historical selection rule was not reproduced"
        winner = max((i for i, v in enumerate(clean_val) if max(clean_val) - v <= TOL), key=lambda i: rr[i]["route"]["last_iteration"])
        validation_results.append({
            **meta, "historical_prompt_sha256": rr[selected]["route"]["prompt_sha256"],
            "reselected_prompt_sha256": rr[winner]["route"]["prompt_sha256"],
            "definition_changes": selected != winner,
            "historical_still_tied_for_best": max(clean_val) - clean_val[selected] <= TOL,
            "old_new_validation_tau_b": tau(original_val, clean_val),
            "historical_validation_score_change_pp": 100 * (clean_val[selected] - original_val[selected]),
            "reselection_target_all_change_pp": 100 * (rr[winner]["target"]["T_All"]["macro_f1"] - rr[selected]["target"]["T_All"]["macro_f1"]),
            "reselection_target_nfr_change_pp": 100 * (rr[winner]["target"]["T_NFR"]["macro_f1"] - rr[selected]["target"]["T_NFR"]["macro_f1"]),
        })
    original_cells = {r["cell_id"]: r for r in cell_results if r["source_test_view"] == "original_rows"}
    saved_gaps = read_csv(root / "paper/data/source-test-supplement/analysis/cell_selection_gaps.csv")
    for row in saved_gaps:
        if row["view"] == "T_All" and row["metric"] == "source_test_macro_f1":
            assert abs(float(row["selection_gap_pp"]) - original_cells[row["cell_id"]]["selection_gap_pp"]) < TOL
    saved_ranks = read_csv(root / "paper/data/source-test-supplement/analysis/cell_rank_reference.csv")
    for row in saved_ranks:
        if row["view"] == "T_All" and row["metric_pair"] == "validation_macro_f1 -> source_test_macro_f1":
            assert abs(float(row["tau_b"]) - original_cells[row["cell_id"]]["validation_test_tau_b"]) < TOL
    summary = {}
    metric_keys = ["pstar_macro_f1_pp", "p0_macro_f1_pp", "gain_over_p0_pp", "selection_gap_pp", "selection_gap_weighted_f1_pp", "validation_test_tau_b", "test_target_all_tau_b", "test_target_nfr_tau_b", "original_subset_test_tau_b"]
    for view in ["original_rows", "unseen_rows", "unseen_equal_text_weights"]:
        selected_rows = [r for r in cell_results if r["source_test_view"] == view]
        s = {k: distribution([r[k] for r in selected_rows]) for k in metric_keys}
        s.update({"cells": len(selected_rows), "pool_best_count": sum(r["selected_is_pool_best"] for r in selected_rows), "within_one_pp_count": sum(r["selected_within_one_pp"] for r in selected_rows), "positive_gain_count": sum(r["gain_over_p0_pp"] > TOL for r in selected_rows), "negative_gain_count": sum(r["gain_over_p0_pp"] < -TOL for r in selected_rows)})
        if view != "original_rows":
            for key in ["pstar_macro_f1_pp", "gain_over_p0_pp", "selection_gap_pp"]:
                delta = [r[key] - original_cells[r["cell_id"]][key] for r in selected_rows]
                s[key + "_change_from_original"] = distribution(delta)
                s[key + "_absolute_change"] = distribution([abs(x) for x in delta])
        summary[view] = s
    target = read_csv(root / "data/datasets/gitreq_enriched_v3.csv")
    assert "requirement_text" in target[0], list(target[0])
    target_overlap = set(texts) & {normalized(row["requirement_text"]) for row in target}
    validation_summary = {
        "cells": 30, "changed_definitions": sum(r["definition_changes"] for r in validation_results),
        "historical_still_tied_for_best": sum(r["historical_still_tied_for_best"] for r in validation_results),
        **{k: distribution([r[k] for r in validation_results]) for k in ["old_new_validation_tau_b", "historical_validation_score_change_pp", "reselection_target_all_change_pp", "reselection_target_nfr_change_pp"]},
    }
    result = {
        "scope": "Descriptive sensitivity of fixed candidate predictions. No inference, no new optimization, no replacement of historical pstar.",
        "source_run_id": "75cdb026-dae7-4480-b230-c2e76ea20b43",
        "data_audit_run_id": "d35420a1-9501-4b57-89c1-7fd90270dfa2",
        "normalization": "Unicode NFC; casefold; collapse whitespace; trim. Exact matching gives the same source duplicate groups.",
        "source": {"rows": len(rows), "unique_texts": len(groups), "duplicate_groups": len(duplicate_groups), "rows_in_duplicate_groups": sum(map(len, duplicate_groups)), "excess_copies": len(rows) - len(groups), "conflicting_label_groups": 0, "duplicate_group_sizes": dict(Counter(map(len, duplicate_groups))), "duplicate_rows_security": int(sum(labels[g].sum() for g in duplicate_groups)), "duplicate_rows_nonsecurity": int(sum(len(g) - labels[g].sum() for g in duplicate_groups)), "target_normalized_overlap": len(target_overlap)},
        "split_audit": split_audit,
        "source_test_sensitivity": summary,
        "validation_sensitivity": validation_summary,
        "checks": {"candidate_routes": 380, "candidate_prediction_items": 380 * 204, "original_score_max_error": max_score_error, "original_rank_and_gap_reproduced": True, "vote_majorities_verified": True, "validation_predictions_verified": True, "duplicate_unseen_test_prediction_disagreements": duplicate_prediction_disagreements},
        "limitations": [
            "Descriptive results for the five overlapping splits and frozen candidate pools; cells are not independent replicates.",
            "Removing overlap changes sample size and class composition; before-after differences do not identify causal bias.",
            "Equal-text analysis assigns total weight one per distinct remaining normalized text without arbitrarily dropping disagreeing predictions.",
            "Validation exclusion is a post hoc fixed-pool reselection diagnostic; it does not recreate the search trajectories that grouped splitting would produce.",
            "Unchanged frozen GitReq results are not evidence that a fully deduplicated source optimization would have identical target results.",
            "No near-duplicate semantic or pretrained-model contamination test was performed.",
        ],
        "input_sha256": hashes,
        "software": {"numpy": np.__version__},
    }
    for name, output_rows in [("split-overlap.csv", split_audit), ("candidate-test-scores.csv", candidate_results), ("cell-test-sensitivity.csv", cell_results), ("validation-reselection-sensitivity.csv", validation_results)]:
        write_csv(out / name, output_rows)
    (out / "audit-results.json").write_text(json.dumps(result, indent=2, default=lambda x: x.item()) + "\n")
    print(json.dumps({"source": result["source"], "source_test": {v: {k: s[k] for k in ["pool_best_count", "within_one_pp_count", "positive_gain_count", "negative_gain_count", "selection_gap_pp", "validation_test_tau_b", "test_target_all_tau_b", "original_subset_test_tau_b"]} for v, s in summary.items()}, "validation": validation_summary, "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
