"""Offline, paired repository-cluster uncertainty for frozen RQ3 predictions.

All empirical calculations run through the experiment entry point. --self-test
uses only synthetic arrays. No model packages or network access are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

METRICS = ["macro_f1", "weighted_f1", "accuracy", "security_precision",
           "security_recall", "security_f1", "false_positive_rate"]
MODELS = ["llama3_8b", "qwen2_7b", "granite3_2_8b", "falcon3_7b",
          "ministral_8b", "granite3_3_8b", "qwen3_8b"]
VIEWS = ["T_All", "T_NFR"]


def emit(event, **kwargs):
    print(event + " " + json.dumps(kwargs, sort_keys=True), flush=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def save_csv(path, rows):
    assert rows
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros(np.broadcast_shapes(np.shape(a), np.shape(b))), where=b != 0)


def metrics(counts):
    """Last dimension is TP, FP, FN, TN. Return seven metrics in [0,1]."""
    tp, fp, fn, tn = np.moveaxis(counts, -1, 0)
    pos, neg = tp + fn, tn + fp
    fpos = safe_div(2 * tp, 2 * tp + fp + fn)
    fneg = safe_div(2 * tn, 2 * tn + fp + fn)
    return np.stack(((fpos + fneg) / 2,
                     safe_div(pos * fpos + neg * fneg, pos + neg),
                     safe_div(tp + tn, pos + neg), safe_div(tp, tp + fp),
                     safe_div(tp, pos), fpos, safe_div(fp, neg)), axis=-1)


def cluster_counts(pred, gold, repo, masks, nrepo):
    out = np.zeros((nrepo, len(masks), len(pred), 4), dtype=np.float64)
    for vi, mask in enumerate(masks):
        for ri, p in enumerate(pred):
            for ci, sel in enumerate((p & gold, p & ~gold, ~p & gold, ~p & ~gold)):
                out[:, vi, ri, ci] = np.bincount(repo[mask & sel], minlength=nrepo)
    return out


def self_test():
    c = np.array([[2, 0, 0, 2], [0, 2, 2, 0], [2, 2, 0, 0]], float)
    np.testing.assert_allclose(metrics(c)[:, 0], [1, 0, 1 / 3])
    np.testing.assert_allclose(metrics(c)[2, 3:6], [.5, 1, 2/3])
    # Nonlinear F1 must be computed within each route before averaging.
    c2 = np.array([[8, 1, 1, 90], [9, 90, 0, 1]], float)
    assert abs(metrics(c2).mean(0)[0] - metrics(c2.sum(0))[0]) > .01
    gold = np.array([1, 0, 1, 0, 0], bool)
    pred = np.array([[1, 1, 1, 0, 0], [0, 0, 1, 1, 0]], bool)
    repo = np.array([0, 0, 1, 2, 2])
    masks = [np.ones(5, bool), np.array([1, 1, 1, 0, 0], bool)]
    cc = cluster_counts(pred, gold, repo, masks, 3)
    np.testing.assert_array_equal(cc.sum(0)[0], [[2, 1, 0, 2], [1, 1, 1, 2]])
    # Repeated repositories retain all their rows and all route pairings.
    w = np.array([[2., 0, 1], [0, 3, 0]])
    bt = (w @ cc.reshape(3, -1)).reshape(2, 2, 2, 4)
    np.testing.assert_array_equal(bt[0, 0, 0], [2, 2, 0, 2])
    scores = metrics(bt)
    np.testing.assert_array_equal(scores[:, :, 0] - scores[:, :, 0], 0)
    emit("RQ3_SELF_TEST", passed=True)


def audit(cfg, out):
    assert file_sha(cfg["archive"]) == cfg["archive_sha256"], "archive hash mismatch"
    assert file_sha(cfg["target"]) == cfg["target_sha256"], "target hash mismatch"
    with open(cfg["target"], newline="") as f:
        target = list(csv.DictReader(f))
    assert len(target) == 6302 and len({r["gitreq_id"] for r in target}) == 6302
    expected = [(r["gitreq_id"], r["binary_label"], r["original_label"], r["repository"],
                 r["url"], sha(r["requirement_text"].encode())) for r in target]
    metadata, contexts, context_hashes, context_markers = {}, {}, {}, {}
    markers, predictions, pred_hashes, sources = {}, {}, {}, {}
    routes = {}
    member_hashes = {}
    with tarfile.open(cfg["archive"], "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name
            f = archive.extractfile(member)
            if name.startswith("predictions/") and name.endswith(".jsonl"):
                rid = Path(name).stem
                assert rid in routes and rid in contexts and rid not in predictions
                route = routes[rid]
                ctx = contexts.pop(rid)
                p = np.empty(len(target), dtype=bool)
                h = hashlib.sha256()
                source = set()
                n = 0
                for n, line in enumerate(f, 1):
                    h.update(line)
                    r = json.loads(line)
                    assert n <= len(target)
                    assert (r["gitreq_id"], r["gold_binary"], r["original_label"], r["repository"],
                            r["url"], r["requirement_text_sha256"]) == expected[n-1]
                    assert r["route_id"] == rid and r["model"] == route["model"]
                    assert r["prompt_sha256"] == route["prompt_sha256"]
                    assert r["route_fingerprint_sha256"] == route["route_fingerprint_sha256"]
                    assert len(r["votes"]) == 1
                    v = r["votes"][0]
                    assert v["label"] in ("Security", "Non-security")
                    assert v["raw"].strip() == v["label"] == r["prediction"]
                    assert v["finish_reason"] == "stop"
                    rendered = v.get("rendered_request_sha256", v.get("rendered_sha256"))
                    assert (r["gitreq_id"], v["input_tokens"], rendered) == ctx[n-1]
                    assert v["input_tokens"] + 64 <= 2048
                    p[n-1] = r["prediction"] == "Security"
                    source.add(r["prediction_source"])
                assert n == 6302 and len(source) == 1
                predictions[rid] = p
                pred_hashes[rid] = h.hexdigest()
                sources[rid] = source.pop()
                member_hashes[name] = h.hexdigest()
                if len(predictions) % 27 == 0:
                    emit("RQ3_AUDIT_PROGRESS", routes=len(predictions), total=189)
            else:
                raw = f.read()
                member_hashes[name] = sha(raw)
                if name == "route_manifest.jsonl":
                    routes = {r["route_id"]: r for r in map(json.loads, raw.splitlines())}
                elif name.startswith("context/") and name.count("/") == 2:
                    rid = Path(name).stem
                    c = json.loads(raw)
                    assert c["route_fingerprint_sha256"] == routes[rid]["route_fingerprint_sha256"]
                    contexts[rid] = [(r["gitreq_id"], r["input_tokens"], r["rendered_request_sha256"])
                                     for r in c["items"]]
                    assert len(contexts[rid]) == 6302
                    context_hashes[rid] = sha(raw)
                elif name.startswith("context/") and name.endswith(".complete.json"):
                    context_markers[name] = json.loads(raw)
                elif name.startswith("predictions/") and name.endswith(".complete.json"):
                    c = json.loads(raw)
                    markers[c["route_id"]] = c
                else:
                    metadata[name] = raw
    assert len(routes) == len(predictions) == len(markers) == len(context_hashes) == 189
    assert not contexts
    for c in context_markers.values():
        for r in c["routes"]:
            assert context_hashes[r["route_id"]] == r["sha256"]
    assert len(context_markers) == 7
    for rid, route in routes.items():
        assert sha(route["prompt_text"].encode()) == route["prompt_sha256"]
        assert route["contract"]["target_sha256"] == cfg["target_sha256"]
        assert route["contract"]["votes_per_item"] == 1
        canonical = json.dumps(route["contract"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert sha(canonical.encode()) == route["route_fingerprint_sha256"]
        mark = markers[rid]
        assert mark["prediction_sha256"] == pred_hashes[rid]
        assert mark["route_fingerprint_sha256"] == route["route_fingerprint_sha256"]
        assert mark["rows"] == 6302 and mark["votes_per_item"] == 1
    logical = [json.loads(b) for b in metadata["logical_matrix_manifest.jsonl"].splitlines()]
    assert len(logical) == 217
    assert sum(r["is_baseline"] for r in logical) == 7
    for r in logical:
        assert r["semantic_sha256"] == routes[r["route_id"]]["prompt_sha256"]
        assert r["deployer"] == routes[r["route_id"]]["model"]
    frozen = json.loads(metadata["frozen_prompts.json"])
    assert len(frozen["prompts"]) == 30
    for p in [frozen["p0"], *frozen["prompts"]]:
        assert sha(p["semantic_text"].encode()) == p["semantic_sha256"]
    # Preserve small upstream records and the exact hash manifest.
    md = out / "provenance"
    md.mkdir(exist_ok=True)
    for name, raw in metadata.items():
        dest = md / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    save_json(out / "input_member_hashes.json", member_hashes)
    save_json(out / "route_manifest.json", list(routes.values()))
    save_json(out / "logical_matrix_manifest.json", logical)
    keys = list(routes)
    pred = np.stack([predictions[rid] for rid in keys])
    gold = np.array([r["binary_label"] == "Security" for r in target])
    names, repo = np.unique([r["repository"] for r in target], return_inverse=True)
    assert len(names) == 4080 and (names != "").all()
    masks = np.array([np.ones(len(target), bool),
                      np.array([r["source_partition"] == "NFR" for r in target])])
    assert masks.sum(1).tolist() == [6302, 5771]
    assert [int(gold[m].sum()) for m in masks] == [1646, 1646]
    np.savez_compressed(out / "validated_predictions.npz", predictions=pred, gold=gold,
                        repository_index=repo, repository_names=names, masks=masks,
                        route_ids=np.array(keys), item_ids=np.array([r["gitreq_id"] for r in target]),
                        original_labels=np.array([r["original_label"] for r in target]))
    checks = dict(routes=189, logical_winner_cells=210, baselines=7, items=6302,
                  repositories=4080, unique_prompt_texts=len({r["prompt_sha256"] for r in routes.values()}),
                  prediction_rows=189*6302, prediction_sources=dict(Counter(sources.values())),
                  archive_sha256=cfg["archive_sha256"], target_sha256=cfg["target_sha256"],
                  all_row_context_and_marker_checks_passed=True)
    emit("RQ3_AUDIT", **checks)
    return target, keys, pred, gold, repo, masks, logical, metadata, checks


def interval(draws):
    return np.quantile(draws, [.025, .975], axis=0)


def analyze(cfg):
    started = time.monotonic()
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "analysis_checks.json").exists(), "Refuse to overwrite a completed analysis"
    save_json(out / "analysis_protocol.json", cfg)
    emit("RQ3_CONFIG", **cfg, numpy=np.__version__, python=platform.python_version())
    target, keys, pred, gold, repo, masks, logical, metadata, checks = audit(cfg, out)
    lookup = {rid: i for i, rid in enumerate(keys)}
    baselines = {r["deployer"]: lookup[r["route_id"]] for r in logical if r["is_baseline"]}
    winners = sorted([r for r in logical if not r["is_baseline"]],
                     key=lambda r: (MODELS.index(r["deployer"]), MODELS.index(r["producer"]),
                                    r["optimizer"], r["seed"]))
    group_keys = [(p, o, d) for d in MODELS for p in MODELS[:3] for o in ("pe2", "btape")]
    groups = [[r for r in winners if (r["producer"], r["optimizer"], r["deployer"]) == g]
              for g in group_keys]
    assert all([r["seed"] for r in group] == cfg["source_seeds"] for group in groups)
    group_idx = np.array([[lookup[r["route_id"]] for r in group] for group in groups])
    group_base = np.array([baselines[d] for p, o, d in group_keys])
    cc = cluster_counts(pred, gold, repo, masks, checks["repositories"])
    counts = cc.sum(axis=0)
    score = metrics(counts)  # view, route, metric
    assert np.all(counts.sum(-1) == masks.sum(1)[:, None])
    # Independently calculated point estimates must reproduce all upstream rows.
    upstream = list(csv.DictReader(io.StringIO(metadata["analysis/route_scores.csv"].decode())))
    assert len(upstream) == 378
    max_error = 0.0
    for r in upstream:
        for mi, name in enumerate(METRICS[:-1]):
            err = abs(score[VIEWS.index(r["view"]), lookup[r["route_id"]], mi] - float(r[name]))
            max_error = max(max_error, err)
            assert err < 1e-12
    group_score = score[:, group_idx, :].mean(axis=2)
    effects = score[:, group_idx, :] - score[:, group_base, :][:, :, None, :]
    mean_effect = effects.mean(axis=2)
    upstream_groups = list(csv.DictReader(io.StringIO(metadata["analysis/group_summary.csv"].decode())))
    assert len(upstream_groups) == 84
    for r in upstream_groups:
        gi = group_keys.index((r["producer"], r["optimizer"], r["deployer"]))
        assert abs(mean_effect[VIEWS.index(r["view"]), gi, 0] * 100 - float(r["mean_gain_pp"])) < 1e-10
    checks.update(upstream_route_scores_reproduced=378, upstream_group_means_reproduced=84,
                  maximum_score_absolute_error=max_error)
    emit("RQ3_POINT_CHECK", **checks)
    rng = np.random.default_rng(cfg["seed"])
    b, k = cfg["replicates"], checks["repositories"]
    boot = np.empty((b, 2, len(keys), len(METRICS)), dtype=np.float64)
    flat_cc = cc.reshape(k, -1)
    for start in range(0, b, cfg["batch_size"]):
        stop = min(b, start + cfg["batch_size"])
        weights = rng.multinomial(k, np.full(k, 1/k), size=stop-start).astype(float)
        c = (weights @ flat_cc).reshape(stop-start, 2, len(keys), 4)
        assert (c[..., 0] + c[..., 2] > 0).all() and (c[..., 1] + c[..., 3] > 0).all()
        boot[start:stop] = metrics(c)
        if stop % 1000 == 0:
            emit("RQ3_BOOTSTRAP_PROGRESS", replicates=stop, total=b, elapsed_seconds=round(time.monotonic()-started, 1))
    assert np.isfinite(boot).all()
    group_boot_score = boot[:, :, group_idx, :].mean(axis=3)
    group_boot = group_boot_score - boot[:, :, group_base, :]
    group_lo, group_hi = interval(group_boot)
    # Approximate simultaneous sensitivity over all 84 predeclared primary gains.
    primary = group_boot[:, :, :, 0].reshape(b, 84)
    center = mean_effect[:, :, 0].reshape(84)
    sd = primary.std(axis=0, ddof=1)
    assert (sd > 0).all()
    critical = float(np.quantile(np.max(np.abs((primary-center)/sd), axis=1), .95))
    sim_lo = (center-critical*sd).reshape(2, 42)
    sim_hi = (center+critical*sd).reshape(2, 42)
    np.savez_compressed(out / "group_bootstrap_draws.npz", gains=group_boot,
                        mean_scores=group_boot_score, group_keys=np.array(group_keys), metrics=np.array(METRICS))
    group_rows, cell_rows, baseline_rows, behavior_rows, category_rows = [], [], [], [], []
    route_lo, route_hi = interval(boot)
    for vi, view in enumerate(VIEWS):
        for model, ri in baselines.items():
            for mi, metric in enumerate(METRICS):
                baseline_rows.append(dict(view=view, deployer=model, metric=metric, estimate=score[vi,ri,mi],
                                          ci_low=route_lo[vi,ri,mi], ci_high=route_hi[vi,ri,mi]))
        for gi, (producer, opt, deployer) in enumerate(group_keys):
            base = dict(view=view, producer=producer, optimizer=opt, deployer=deployer,
                        source_seeds=5, origin_diagonal=producer == deployer,
                        additional_deployer=deployer in MODELS[3:])
            for mi, metric in enumerate(METRICS):
                e = effects[vi, gi, :, mi]*100
                loso = (e.sum()-e)/4
                row = dict(**base, metric=metric, baseline_score=score[vi,group_base[gi],mi],
                           mean_score=group_score[vi,gi,mi], mean_gain_pp=mean_effect[vi,gi,mi]*100,
                           ci_low_pp=group_lo[vi,gi,mi]*100, ci_high_pp=group_hi[vi,gi,mi]*100,
                           seed_min_pp=float(e.min()), seed_max_pp=float(e.max()),
                           seed_sd_pp=float(e.std(ddof=1)), positive_seeds=int((e>1e-12).sum()),
                           negative_seeds=int((e < -1e-12).sum()), zero_seeds=int((abs(e)<=1e-12).sum()),
                           leave_one_seed_out_min_pp=float(loso.min()), leave_one_seed_out_max_pp=float(loso.max()))
                if mi == 0:
                    row.update(simultaneous_low_pp=sim_lo[vi,gi]*100, simultaneous_high_pp=sim_hi[vi,gi]*100)
                group_rows.append(row)
            for si, r in enumerate(groups[gi]):
                ri, bi = group_idx[gi,si], group_base[gi]
                draws = boot[:,vi,ri,:] - boot[:,vi,bi,:]
                lo, hi = interval(draws)
                if ri == bi:
                    assert np.all(draws == 0)
                for mi, metric in enumerate(METRICS):
                    cell_rows.append(dict(**base, seed=r["seed"], route_id=r["route_id"], metric=metric,
                                          score=score[vi,ri,mi], gain_pp=effects[vi,gi,si,mi]*100,
                                          ci_low_pp=lo[mi]*100, ci_high_pp=hi[mi]*100,
                                          identical_to_baseline=ri == bi))
            m = masks[vi]
            p = pred[group_idx[gi]][:,m]
            p0, y = pred[group_base[gi]][m], gold[m]
            fixes = ((p == y) & (p0 != y)).sum(axis=1)
            breaks = ((p != y) & (p0 == y)).sum(axis=1)
            recover = (p & ~p0 & y).sum(axis=1)
            misses = (~p & p0 & y).sum(axis=1)
            alarmfix = (~p & p0 & ~y).sum(axis=1)
            newalarm = (p & ~p0 & ~y).sum(axis=1)
            np.testing.assert_array_equal(fixes, recover + alarmfix)
            np.testing.assert_array_equal(breaks, misses + newalarm)
            np.testing.assert_allclose((fixes-breaks).mean()/m.sum(), mean_effect[vi,gi,2], atol=1e-14)
            behavior_rows.append(dict(**base, mean_corrected_errors=fixes.mean(), mean_new_errors=breaks.mean(),
                                      mean_security_recoveries=recover.mean(), mean_new_security_misses=misses.mean(),
                                      mean_false_alarms_removed=alarmfix.mean(), mean_new_false_alarms=newalarm.mean(),
                                      mean_tp=counts[vi,group_idx[gi],0].mean(), mean_fp=counts[vi,group_idx[gi],1].mean(),
                                      mean_fn=counts[vi,group_idx[gi],2].mean(), mean_tn=counts[vi,group_idx[gi],3].mean()))
            labels = np.array([r["original_label"] for r in target])
            for label in sorted(set(labels[m & ~gold])):
                sel = m & (labels == label)
                a, base_rate = pred[group_idx[gi]][:,sel].mean(), pred[group_base[gi],sel].mean()
                category_rows.append(dict(**base, category=label, support=int(sel.sum()), baseline_fpr=base_rate,
                                          mean_fpr=a, mean_fpr_gain_pp=(a-base_rate)*100))
    save_csv(out / "group_intervals.csv", group_rows)
    save_csv(out / "cell_effects.csv", cell_rows)
    save_csv(out / "baseline_intervals.csv", baseline_rows)
    save_csv(out / "paired_error_changes.csv", behavior_rows)
    save_csv(out / "category_false_positive_rates.csv", category_rows)
    # Conditional matched optimizer contrast; no causal attribution to algorithms.
    contrasts = []
    for d in MODELS:
        for p in MODELS[:3]:
            ia, ib = group_keys.index((p,"pe2",d)), group_keys.index((p,"btape",d))
            for vi, view in enumerate(VIEWS):
                lo, hi = interval(group_boot[:,vi,ib,0]-group_boot[:,vi,ia,0])
                contrasts.append(dict(view=view, producer=p, deployer=d,
                                      btape_minus_pe2_pp=(mean_effect[vi,ib,0]-mean_effect[vi,ia,0])*100,
                                      ci_low_pp=float(lo)*100, ci_high_pp=float(hi)*100))
    save_csv(out / "optimizer_contrasts.csv", contrasts)
    # Equal group averages are descriptive aggregates, not extra model trials.
    aggregates = []
    for name, take in (("all", np.ones(42,bool)),
                       ("origin_diagonal", np.array([p==d for p,o,d in group_keys])),
                       ("off_diagonal", np.array([p!=d for p,o,d in group_keys])),
                       ("additional_deployers", np.array([d in MODELS[3:] for p,o,d in group_keys]))):
        for vi, view in enumerate(VIEWS):
            draws = group_boot[:,vi,take,0].mean(axis=1)
            lo, hi = interval(draws)
            aggregates.append(dict(view=view, subset=name, groups=int(take.sum()),
                                   mean_gain_pp=mean_effect[vi,take,0].mean()*100,
                                   ci_low_pp=float(lo)*100, ci_high_pp=float(hi)*100))
    save_csv(out / "aggregate_effects.csv", aggregates)
    summaries = {}
    for vi, view in enumerate(VIEWS):
        e = effects[vi,:,:,0]
        summaries[view] = dict(positive_groups=int((mean_effect[vi,:,0]>1e-12).sum()),
                               positive_pointwise_intervals=int((group_lo[vi,:,0]>0).sum()),
                               negative_pointwise_intervals=int((group_hi[vi,:,0]<0).sum()),
                               positive_simultaneous_intervals=int((sim_lo[vi]>0).sum()),
                               negative_simultaneous_intervals=int((sim_hi[vi]<0).sum()),
                               positive_cells=int((e>1e-12).sum()), negative_cells=int((e < -1e-12).sum()),
                               zero_cells=int((abs(e)<=1e-12).sum()),
                               positive_leave_one_seed_out_groups=sum(r["leave_one_seed_out_min_pp"]>0 for r in group_rows
                                                                       if r["view"]==view and r["metric"]=="macro_f1"))
    checks.update(complete=True, bootstrap_replicates=b, bootstrap_seed=cfg["seed"],
                  simultaneous_family_size=84, simultaneous_critical_value=critical,
                  shared_repository_weights=True, zero_self_differences_verified=True,
                  error_transition_identities_verified=True, elapsed_seconds=time.monotonic()-started,
                  summary=summaries)
    save_json(out / "analysis_checks.json", checks)
    emit("RQ3_FINAL", **checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rq3_systematic_analysis.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        analyze(json.loads(Path(args.config).read_text()))
