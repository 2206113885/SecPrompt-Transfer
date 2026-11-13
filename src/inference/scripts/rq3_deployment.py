"""Frozen seven-deployer RQ3: GPU0, first-vote reuse and per-item resume."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from run_target_matrix import (canonical, create_backend, immutable_json, load_inputs,
                               model_runtime, read_json, read_target, sha_file)
from pe2_btape.prompts import classification_messages
from pe2_btape.metrics import compute_metrics

LABELS = ["Non-security", "Security"]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/rq3_seven_model_v1.json"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(path, obj):
    atomic(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def jl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def immutable_jl(path, rows):
    text = "".join(canonical(r) + "\n" for r in rows)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"Frozen manifest conflict: {path}")
    if not path.exists():
        atomic(path, text)


def emit(event, **data):
    print(event + " " + canonical({"time": now(), **data}), flush=True)


def config(path):
    cfg = read_json(path)
    base = read_json(ROOT / cfg["base_spec"])
    extra = read_json(ROOT / cfg["additional_spec"])
    models = {**base["models"], **extra["models"]}
    preflights = {m: str(Path(s["output_root"]) / "preflight" / f"{m}.json")
                 for s in (base, extra) for m in s["models"]}
    assert len(models) == 7 and set(models) == set(cfg["model_order"])
    assert cfg["gpu"] == 0 and cfg["votes_per_item"] == 1
    return {**base, **cfg, "models": models, "preflights": preflights}


def fingerprint(runtime):
    # Exclude location metadata, retain hardware type and all computation fields.
    ignored = {"cuda_visible_devices", "model_dir", "model_manifest_path",
               "model_manifest_sha256", "model_manifest"}
    out = {k: v for k, v in runtime.items() if k not in ignored}
    out["chat_template_kwargs"] = runtime.get("chat_template_kwargs", {})
    out["local_artifact"] = {k: v for k, v in runtime["local_artifact"].items()
                             if k != "revision_metadata_files"}
    return out


def build_manifests(spec, frozen, runtimes):
    unique = {}
    for p in [frozen["p0"], *frozen["prompts"]]:
        assert sha(p["semantic_text"]) == p["semantic_sha256"]
        unique.setdefault(p["semantic_sha256"], p["semantic_text"])
        assert unique[p["semantic_sha256"]] == p["semantic_text"]
    routes, logical = [], []
    for m in spec["model_order"]:
        model = spec["models"][m]
        for h in sorted(unique, key=lambda h: (h != frozen["p0"]["semantic_sha256"], h)):
            contract = {"model": m, "model_fingerprint": fingerprint(runtimes[m]),
                        "interface": model["interface"], "stop_strings": model["stop_strings"],
                        "labels": LABELS, "runtime": model_runtime(spec, m),
                        "target_sha256": spec["gitreq_enriched_sha256"],
                        "prompt_text_sha256": h, "votes_per_item": 1,
                        "decoding": {"temperature": 0, "top_p": 1, "top_k": -1}}
            key = sha(canonical(contract))
            route = {"route_id": f"{m}__{h[:16]}__{key[:12]}", "model": m,
                     "prompt_sha256": h, "prompt_text": unique[h],
                     "route_fingerprint_sha256": key, "contract": contract}
            routes.append(route)
            for p in [frozen["p0"], *frozen["prompts"]]:
                if p["semantic_sha256"] == h:
                    logical.append({k: v for k, v in {**p, "deployer": m,
                        "route_id": route["route_id"], "is_baseline": p["prompt_id"] == "p0"}.items()
                                    if k != "semantic_text"})
    return routes, logical


def plan(spec):
    out = Path(spec["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    if sha_file(Path(spec["source_csv"])) != spec["source_csv_sha256"]:
        raise RuntimeError("SecReq hash mismatch")
    target = read_target(spec)
    frozen, _ = load_inputs(spec, verify_archive=True)
    runtimes, audits = {}, []
    for m, path in spec["preflights"].items():
        p = read_json(Path(path)); rows = p["predictions"]
        if not (p["status"] == "passed" and p["formal_rq3_votes"] == 1
                and p["preflight_prompt_sha256"] == frozen["p0"]["semantic_sha256"]
                and [r["source_id"] for r in rows] == frozen["seed42_test_ids"]
                and p["interface"] == spec["models"][m]["interface"]):
            raise RuntimeError(f"Preflight contract mismatch: {m}")
        for r in rows:
            assert len(r["votes"]) == 3
            for v in r["votes"]:
                assert v["raw"].strip() == v["label"] in LABELS and v["finish_reason"] == "stop"
        agreement = sum(len({v["label"] for v in r["votes"]}) == 1 for r in rows) / len(rows)
        assert agreement >= .995 and len(rows) == 204
        runtimes[m] = p["model_runtime"]
        audits.append({"model": m, "path": path, "sha256": sha_file(Path(path)),
                       "agreement": agreement, "items": len(rows), "votes": 612})
    routes, logical = build_manifests(spec, frozen, runtimes)
    code_files = ["scripts/rq3_deployment.py", "scripts/run_target_matrix.py",
                  "src/pe2_btape/backend.py", "src/pe2_btape/prompts.py", "src/pe2_btape/metrics.py"]
    immutable_json(out / "frozen_config.json", {"spec": spec,
        "code_sha256": {f: sha_file(ROOT / f) for f in code_files},
        "protocol_sha256": sha_file(ROOT / "RQ3_PROTOCOL.md")})
    protocol = (ROOT / "RQ3_PROTOCOL.md").read_text()
    if (out / "RQ3_PROTOCOL.md").exists() and (out / "RQ3_PROTOCOL.md").read_text() != protocol:
        raise RuntimeError("Protocol changed")
    atomic(out / "RQ3_PROTOCOL.md", protocol)
    immutable_json(out / "frozen_prompts.json", frozen)
    immutable_json(out / "preflight_evidence.json", audits)
    immutable_json(out / "model_runtimes.json", runtimes)
    immutable_jl(out / "route_manifest.jsonl", routes)
    immutable_jl(out / "logical_matrix_manifest.jsonl", logical)
    summary = {"logical_selected_cells": sum(not r["is_baseline"] for r in logical),
        "logical_baselines": sum(r["is_baseline"] for r in logical),
        "unique_texts": len({r["prompt_sha256"] for r in routes}), "unique_routes": len(routes),
        "items_per_route": len(target), "preflight_models_passed": len(audits), "gpu": 0}
    immutable_json(out / "plan.json", summary)
    emit("RQ3_PLAN", **summary)


def paths(spec, route):
    base = Path(spec["output_root"]) / "predictions" / route["model"] / route["route_id"]
    return base.with_suffix(".jsonl"), base.with_suffix(".partial.jsonl"), base.with_suffix(".complete.json")


def rendering(spec, route, row):
    system, user = classification_messages(route["prompt_text"], [], row["requirement_text"],
                                            LABELS, spec["models"][route["model"]]["interface"])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def context_audit(spec, model):
    from transformers import AutoTokenizer
    from pe2_btape.backend import _local_hf_artifact_metadata
    out = Path(spec["output_root"])
    routes = [r for r in jl(out / "route_manifest.jsonl") if r["model"] == model]
    rows = read_target(spec); mod = spec["models"][model]
    runtime = read_json(out / "model_runtimes.json")[model]
    observed = _local_hf_artifact_metadata(mod["model_dir"], mod["revision"])
    for k, v in fingerprint(runtime)["local_artifact"].items():
        assert observed[k] == v, (model, k)
    tok = AutoTokenizer.from_pretrained(mod["model_dir"], local_files_only=True, trust_remote_code=False)
    assert sha(tok.chat_template or "") == runtime["tokenizer_chat_template_sha256"]
    assert tok.__class__.__name__ == runtime["tokenizer_class"]
    reports = []
    cached_routes = {}
    cache_root = Path(spec["context_cache_root"]) if spec.get("context_cache_root") else None
    if cache_root:
        cached_routes = {r["route_id"]: r for r in jl(cache_root / "route_manifest.jsonl")}
        cached_report = read_json(cache_root / "context" / f"{model}.complete.json")
        cached_hashes = {r["route_id"]: r["sha256"] for r in cached_report["routes"]}
    for route in routes:
        dest = out / "context" / model / (route["route_id"] + ".json")
        if not dest.exists() and route["route_id"] in cached_routes:
            assert route == cached_routes[route["route_id"]], "Changed route cannot use old token audit"
            src = cache_root / "context" / model / (route["route_id"] + ".json")
            assert sha_file(src) == cached_hashes[route["route_id"]]
            cached = read_json(src)
            assert cached["route_fingerprint_sha256"] == route["route_fingerprint_sha256"]
            assert [x["gitreq_id"] for x in cached["items"]] == [x["gitreq_id"] for x in rows]
            assert cached["max_input_tokens"] + 64 <= 2048
            immutable_json(dest, cached)
        if dest.exists():
            c = read_json(dest)
            assert c["route_fingerprint_sha256"] == route["route_fingerprint_sha256"]
            assert len(c["items"]) == 6302
            reports.append({"route_id": route["route_id"], "max_input_tokens": c["max_input_tokens"], "sha256":sha_file(dest)})
            continue
        items = []
        for start in range(0, len(rows), spec["tokenizer_batch_size"]):
            chunk = rows[start:start + spec["tokenizer_batch_size"]]
            conv = [rendering(spec, route, row) for row in chunk]
            kwargs = dict(add_generation_prompt=True, **mod.get("chat_template_kwargs", {}))
            texts = tok.apply_chat_template(conv, tokenize=False, **kwargs)
            ids = tok.apply_chat_template(conv, tokenize=True, **kwargs)
            for row, text, tokens in zip(chunk, texts, ids):
                if len(tokens) + 64 > 2048:
                    raise RuntimeError(f"Context overflow {route['route_id']} {row['gitreq_id']}: {len(tokens)}+64")
                items.append({"gitreq_id": row["gitreq_id"], "rendered_request_sha256": sha(text),
                              "input_tokens": len(tokens)})
        c = {"route_fingerprint_sha256": route["route_fingerprint_sha256"], "items": items,
             "max_input_tokens": max(x["input_tokens"] for x in items)}
        immutable_json(dest, c)
        reports.append({"route_id": route["route_id"], "max_input_tokens": c["max_input_tokens"], "sha256":sha_file(dest)})
        emit("RQ3_CONTEXT_PASS", model=model, routes_done=len(reports), routes_total=len(routes))
    immutable_json(out / "context" / f"{model}.complete.json", {"model": model, "routes": reports})
    emit("RQ3_MODEL_CONTEXT_PASS", model=model, routes=len(reports),
         cache_root=str(cache_root) if cache_root else None)


def check_vote(vote, context):
    assert vote["label"] in LABELS
    assert vote["raw"].strip() == vote["label"]
    assert vote["finish_reason"] == "stop"
    rendered = vote.get("rendered_request_sha256", vote.get("rendered_sha256"))
    assert rendered == context["rendered_request_sha256"]
    assert vote["input_tokens"] == context["input_tokens"]


def check_row(record, row, route, context, votes=1):
    assert record["gitreq_id"] == row["gitreq_id"] == context["gitreq_id"]
    assert record["model"] == route["model"] and record["prompt_sha256"] == route["prompt_sha256"]
    assert record["route_id"] == route["route_id"]
    key = route.get("route_fingerprint_sha256", route.get("route_key_sha256"))
    assert record["route_fingerprint_sha256"] == key
    assert record["requirement_text_sha256"] == sha(row["requirement_text"])
    assert record["gold_binary"] == row["binary_label"]
    assert record["original_label"] == row["original_label"]
    assert record["repository"] == row["repository"] and record["url"] == row["url"]
    assert len(record["votes"]) == votes
    for v in record["votes"]:
        check_vote(v, context)
    assert record["prediction"] == Counter(v["label"] for v in record["votes"]).most_common(1)[0][0]


def validate_file(path, rows, route, contexts, votes=1):
    records = jl(path)
    assert len(records) == len(rows)
    for rec, row, ctx in zip(records, rows, contexts):
        check_row(rec, row, route, ctx, votes)
    return records


def completed(spec, route, rows, contexts):
    final, _, mark = paths(spec, route)
    if not mark.exists():
        return False
    marker = read_json(mark)
    assert marker["route_fingerprint_sha256"] == route["route_fingerprint_sha256"]
    assert marker["prediction_sha256"] == sha_file(final)
    assert marker["rows"] == len(rows) and marker["votes_per_item"] == 1
    validate_file(final, rows, route, contexts)
    return True


def finish(spec, route, rows, contexts, provenance):
    final, partial, mark = paths(spec, route)
    source = final if final.exists() else partial
    validate_file(source, rows, route, contexts)
    if source == partial:
        os.replace(partial, final)
    immutable_json(mark, {"route_id": route["route_id"], "route_fingerprint_sha256": route["route_fingerprint_sha256"],
        "rows": len(rows), "votes_per_item": 1, "prediction_sha256": sha_file(final),
        "completed_at": now(), "provenance": provenance})


def contexts_for(spec, route):
    p = Path(spec["output_root"]) / "context" / route["model"] / (route["route_id"] + ".json")
    data = read_json(p)
    assert data["route_fingerprint_sha256"] == route["route_fingerprint_sha256"]
    assert len(data["items"]) == 6302
    return data["items"]


def project_first_vote(rec, route, provenance):
    return {**rec, "route_id": route["route_id"],
            "route_fingerprint_sha256": route["route_fingerprint_sha256"],
            "votes": [rec["votes"][0]], "prediction": rec["votes"][0]["label"],
            "prediction_source": "rq2_first_vote_projection", "source_provenance": provenance}


def legacy_rq1_evidence(spec, marker, route, records):
    """Old RQ2 completion schema stores RQ1 provenance, not a runtime copy."""
    assert marker["prediction_source"] == "rq1_reused"
    root = Path(spec["rq1_root"])
    manifest_path = root / "route-manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["gitreq_enriched_sha256"] == route["contract"]["target_sha256"]
    assert manifest["source_archive_sha256"] == spec["source_archive_sha256"]
    assert manifest["votes_per_item"] == 3 and manifest["target_rows"] == 6302
    prompt_id = marker["source_route_id"]
    match = [x for x in manifest["routes"] if x["deployer"] == route["model"] and x["prompt_id"] == prompt_id]
    assert len(match) == 1 and match[0]["semantic_sha256"] == route["prompt_sha256"]
    source = root / "predictions" / route["model"] / f"{prompt_id}.jsonl"
    source_hash = sha_file(source)
    assert source_hash == marker["source_prediction_sha256"]
    completion = read_json(source.with_suffix(".complete.json"))
    assert completion["prediction_sha256"] == source_hash
    assert completion["model"] == route["model"] and completion["prompt_sha256"] == route["prompt_sha256"]
    assert completion["rows"] == 6302 and completion["votes_per_item"] == 3
    original = jl(source)
    assert len(original) == len(records) == 6302
    for old, normalized in zip(original, records):
        assert old["votes"] == normalized["votes"]
        for k in ("gitreq_id", "gold_binary", "original_label", "repository", "url"):
            assert old[k] == normalized[k]
        assert old["semantic_sha256"] == route["prompt_sha256"]
    runtime = manifest["model_fingerprints"][route["model"]]
    assert fingerprint(runtime) == route["contract"]["model_fingerprint"]
    return {"runtime_manifest": str(manifest_path), "runtime_manifest_sha256": sha_file(manifest_path),
            "source_prediction_file": str(source), "source_prediction_sha256": source_hash,
            "source_run_id": marker["source_run_id"], "source_route_id": prompt_id}


def reuse(spec):
    out = Path(spec["output_root"]); oldroot = Path(spec["rq2_root"])
    rows = read_target(spec); oldroutes = defaultdict(list)
    for r in jl(oldroot / "route_manifest.jsonl"):
        oldroutes[(r["model"], r["prompt_sha256"])].append(r)
    audits = []
    for r in jl(out / "route_manifest.jsonl"):
        candidates = oldroutes[(r["model"], r["prompt_sha256"])]
        if not candidates:
            audits.append({"route_id": r["route_id"], "model": r["model"], "status": "new_inference_required"})
            continue
        contexts = contexts_for(spec, r); oldrecords = None; sources = []
        for old in candidates:
            oc, nc = old["route_contract"], r["contract"]
            for k in ("model", "interface", "stop_strings", "labels", "runtime", "target_sha256", "prompt_text_sha256"):
                assert oc[k] == nc[k], (r["route_id"], k)
            assert old["prompt_text"] == r["prompt_text"] and oc["votes_per_item"] == 3
            assert fingerprint(oc["model_fingerprint"]) == nc["model_fingerprint"]
            path = oldroot / "predictions" / old["model"] / (old["route_id"] + ".jsonl")
            marker = read_json(path.with_suffix(".complete.json")); filehash = sha_file(path)
            assert marker["prediction_sha256"] == filehash and marker["rows"] == 6302
            assert marker["votes_per_item"] == 3 and marker["route_id"] == old["route_id"]
            assert marker["route_fingerprint_sha256"] == old["route_key_sha256"]
            records = validate_file(path, rows, old, contexts, votes=3)
            if "model_fingerprint" in marker:
                assert fingerprint(marker["model_fingerprint"]) == nc["model_fingerprint"]
                legacy = None
            else:
                legacy = legacy_rq1_evidence(spec, marker, r, records)
            if oldrecords is not None:
                assert [x["votes"] for x in oldrecords] == [x["votes"] for x in records], "Conflicting duplicate source"
            oldrecords = records
            sources.append({"path": str(path), "sha256": filehash, "route_id": old["route_id"],
                            "original_prediction_source": marker["prediction_source"],
                            "legacy_rq1_evidence": legacy})
        provenance = {"type": "rq2_first_vote_projection", "vote_index_zero_based": 0, "sources": sources,
                      "source_votes_per_item": 3, "target_votes_per_item": 1}
        if not completed(spec, r, rows, contexts):
            final, partial, mark = paths(spec, r)
            partial.parent.mkdir(parents=True, exist_ok=True)
            projected = [project_first_vote(rec, r, provenance) for rec in oldrecords]
            if final.exists():
                assert jl(final) == projected
            else:
                count = resume_prefix(partial, rows, r, contexts)
                if count:
                    assert jl(partial) == projected[:count]
                with partial.open("a", encoding="utf-8") as f:
                    for rec in projected[count:]:
                        f.write(canonical(rec) + "\n")
                    f.flush(); os.fsync(f.fileno())
            finish(spec, r, rows, contexts, provenance)
        audits.append({"route_id": r["route_id"], "model": r["model"], "status": "reused",
                       "prompt_sha256": r["prompt_sha256"], "rows": 6302, "provenance": provenance,
                       "first_vote_vs_majority_disagreements": sum(x["votes"][0]["label"] != x["prediction"] for x in oldrecords)})
        emit("RQ3_REUSE_VERIFIED", model=r["model"], route_id=r["route_id"])
    immutable_json(out / "reuse_audit.json", audits)
    report_reuse(spec, audits)


def csv_write(path, rows):
    if not rows:
        return
    import io
    columns = list(dict.fromkeys(k for row in rows for k in row))
    f = io.StringIO(); writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader(); writer.writerows(rows)
    atomic(path, f.getvalue())


def report_reuse(spec, audits):
    out = Path(spec["output_root"]); logical = jl(out / "logical_matrix_manifest.jsonl")
    details, summaries = [], []
    for a in audits:
        if a["status"] != "reused":
            continue
        matching = [x for x in logical if x["route_id"] == a["route_id"]]
        details.append({"route_id": a["route_id"], "deployer": a["model"], "prompt_sha256": a["prompt_sha256"],
            "logical_pstar_cells": sum(not x["is_baseline"] for x in matching),
            "baseline_records": sum(x["is_baseline"] for x in matching),
            "source_cell_ids": ";".join(x["prompt_id"] for x in matching),
            "saved_calls": 6302, "first_vote_vs_majority_disagreements": a["first_vote_vs_majority_disagreements"],
            "source_files_and_hashes": canonical(a["provenance"]["sources"])})
    for m in spec["model_order"]:
        n = sum(a["status"] == "reused" and a["model"] == m for a in audits)
        total = sum(a["model"] == m for a in audits)
        summaries.append({"deployer": m, "unique_routes": total, "reused_routes": n,
                          "new_routes_required": total-n, "saved_calls": n*6302,
                          "new_calls_required": (total-n)*6302})
    csv_write(out / "reused_routes.csv", details)
    csv_write(out / "reuse_summary.csv", summaries)
    emit("RQ3_REUSE_SUMMARY", reused_routes=sum(x["reused_routes"] for x in summaries),
         new_routes=sum(x["new_routes_required"] for x in summaries))


def resume_prefix(path, rows, route, contexts):
    if not path.exists():
        return 0
    raw = path.read_bytes(); lines = raw.splitlines(keepends=True); valid = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines)-1 or line.endswith(b"\n"):
                raise
            evidence = path.with_name(path.name + f".torn-{time.time_ns()}")
            os.rename(path, evidence)
            atomic(path, b"".join(lines[:index]).decode("utf-8"))
            return valid
        if index >= len(rows):
            raise RuntimeError("Too many partial rows")
        check_row(record, rows[index], route, contexts[index])
        valid += 1
    if raw and not raw.endswith(b"\n"):
        with path.open("ab") as f:
            f.write(b"\n"); f.flush(); os.fsync(f.fileno())
    return valid


def infer(spec, model):
    out = Path(spec["output_root"]); rows = read_target(spec)
    routes = [r for r in jl(out / "route_manifest.jsonl") if r["model"] == model]
    pending = [r for r in routes if not completed(spec, r, rows, contexts_for(spec, r))]
    if not pending:
        return
    backend = create_backend(spec, model, 0)
    expected = read_json(out / "model_runtimes.json")[model]
    assert fingerprint(backend.runtime_metadata()) == fingerprint(expected), "Runtime drift"
    immutable_json(out / "runtime" / f"{model}.json", backend.runtime_metadata())
    (out / "progress").mkdir(exist_ok=True)
    for route in pending:
        contexts = contexts_for(spec, route)
        final, partial, mark = paths(spec, route); partial.parent.mkdir(parents=True, exist_ok=True)
        provenance = {"type": "rq3_new_inference", "gpu": 0}
        if final.exists():
            finish(spec, route, rows, contexts, provenance)
            continue
        count = resume_prefix(partial, rows, route, contexts)
        start = time.monotonic(); begin = count
        with partial.open("a", encoding="utf-8") as f:
            for index in range(count, len(rows)):
                if (out / "PAUSE").exists():
                    emit("RQ3_PAUSED", model=model, route_id=route["route_id"], rows=index)
                    return
                row, ctx = rows[index], contexts[index]
                sysmsg, usermsg = classification_messages(route["prompt_text"], [], row["requirement_text"],
                                                          LABELS, spec["models"][model]["interface"])
                success = False
                for attempt in range(1, spec["maximum_attempts"] + 1):
                    raw = None; rec = None
                    try:
                        raw, rec = backend.chat(sysmsg, usermsg, kind="classification:target_deployment", num_predict=64)
                        v = {"raw": raw, "label": raw.strip(), "finish_reason": rec["finish_reason"],
                             "rendered_request_sha256": rec["hf_rendered_prompt_sha256"],
                             "input_tokens": rec["prompt_eval_count"], "output_tokens": rec["eval_count"],
                             "duration_ns": rec["total_duration_ns"], "timestamp_utc": now(), "attempt_count": attempt}
                        check_vote(v, ctx); success = True; break
                    except Exception as exc:
                        error = {"route_id": route["route_id"], "gitreq_id": row["gitreq_id"],
                                 "attempt": attempt, "error": repr(exc), "raw": raw, "record": rec, "time": now()}
                        ep = out / "errors" / f"{route['route_id']}.jsonl"; ep.parent.mkdir(exist_ok=True)
                        with ep.open("a") as ef:
                            ef.write(canonical(error)+"\n"); ef.flush(); os.fsync(ef.fileno())
                if not success:
                    raise RuntimeError(f"Invalid/missing prediction after retries: {route['route_id']} {row['gitreq_id']}")
                record = {"gitreq_id": row["gitreq_id"], "gold_binary": row["binary_label"],
                    "original_label": row["original_label"], "repository": row["repository"], "url": row["url"],
                    "requirement_text_sha256": sha(row["requirement_text"]), "model": model,
                    "prompt_sha256": route["prompt_sha256"], "route_id": route["route_id"],
                    "route_fingerprint_sha256": route["route_fingerprint_sha256"], "votes": [v],
                    "prediction": v["label"], "prediction_source": "rq3_new_inference", "timestamp_utc": now()}
                check_row(record, row, route, ctx)
                f.write(canonical(record)+"\n"); f.flush(); os.fsync(f.fileno())
                if (index+1) % spec["progress_interval"] == 0 or index == len(rows)-1:
                    progress = {"model": model, "route_id": route["route_id"], "items": index+1,
                                "total": len(rows), "new_items_this_session": index+1-begin,
                                "elapsed_seconds": time.monotonic()-start, "pid": os.getpid(), "gpu": 0, "time": now()}
                    save(out / "progress" / f"{model}.json", progress)
                    emit("RQ3_PROGRESS", **progress)
                    backend._calls.clear(); backend._render_cache.clear()
        finish(spec, route, rows, contexts, provenance)
        emit("RQ3_ROUTE_DONE", model=model, route_id=route["route_id"])


def status(spec):
    out = Path(spec["output_root"])
    if not (out / "route_manifest.jsonl").exists():
        emit("RQ3_STATUS", stage="not_planned"); return
    counts = Counter(); per = defaultdict(Counter)
    for r in jl(out / "route_manifest.jsonl"):
        final, partial, mark = paths(spec, r)
        if mark.exists():
            kind = read_json(mark)["provenance"]["type"]
            counts["complete"] += 1; per[r["model"]]["complete"] += 1
            counts[kind] += 1; per[r["model"]][kind] += 1
        else:
            counts["remaining"] += 1; per[r["model"]]["remaining"] += 1
    emit("RQ3_STATUS", counts=dict(counts), models={m:dict(v) for m,v in per.items()},
         progress=[read_json(p) for p in sorted((out/"progress").glob("*.json"))])


def analyze(spec):
    out = Path(spec["output_root"]); rows = read_target(spec); scores = {}; route_rows = []
    routes = jl(out / "route_manifest.jsonl"); logical = jl(out / "logical_matrix_manifest.jsonl")
    for r in routes:
        if not completed(spec, r, rows, contexts_for(spec, r)):
            continue
        final, _, marker = paths(spec, r); predictions = jl(final)
        kind = read_json(marker)["provenance"]["type"]
        for view in spec["views"]:
            idx = [i for i,row in enumerate(rows) if view == "T_All" or row["original_label"] != "Functional"]
            assert len(idx) == (6302 if view == "T_All" else 5771)
            met = compute_metrics([predictions[i]["prediction"] for i in idx], [rows[i]["binary_label"] for i in idx], LABELS).as_dict()
            score = {"route_id": r["route_id"], "deployer": r["model"], "prompt_sha256": r["prompt_sha256"],
                     "view": view, "n": len(idx), "macro_f1": met["macro_f1"], "weighted_f1": met["weighted_f1"],
                     "accuracy": met["accuracy"], "security_precision": met["per_class"]["Security"]["precision"],
                     "security_recall": met["per_class"]["Security"]["recall"], "security_f1": met["per_class"]["Security"]["f1"],
                     "prediction_source": kind}
            scores[(r["route_id"], view)] = score; route_rows.append(score)
    baselines = {(l["deployer"], view):scores.get((l["route_id"],view)) for l in logical if l["is_baseline"] for view in spec["views"]}
    cells=[]
    for l in logical:
        for view in spec["views"]:
            score = scores.get((l["route_id"],view)); base = baselines[(l["deployer"],view)]
            if score is None or base is None:
                continue
            cells.append({**l, **score, "p0_macro_f1": base["macro_f1"],
                          "macro_f1_gain_pp": 100*(score["macro_f1"]-base["macro_f1"])})
    groups=defaultdict(list)
    for c in cells:
        if not c["is_baseline"]:
            groups[(c["producer"],c["optimizer"],c["deployer"],c["view"])].append(c)
    summary=[]
    for key, group in groups.items():
        values=[c["macro_f1_gain_pp"] for c in group]
        summary.append(dict(zip(["producer","optimizer","deployer","view"],key),
             seeds_complete=len(group), mean_gain_pp=statistics.mean(values), median_gain_pp=statistics.median(values),
             min_gain_pp=min(values), max_gain_pp=max(values), positive_seeds=sum(x>1e-12 for x in values),
             reused_logical_cells=sum(c["prediction_source"]=="rq2_first_vote_projection" for c in group)))
    csv_write(out/"analysis/route_scores.csv",route_rows)
    csv_write(out/"analysis/logical_matrix_results.csv",cells)
    csv_write(out/"analysis/group_summary.csv",summary)
    full=len(route_rows)==2*len(routes)
    save(out/"analysis/analysis_checks.json",{"complete":full,"routes_scored":len(route_rows)//2,
                                               "routes_expected":len(routes),"logical_view_records":len(cells)})
    atomic(out/"analysis/RQ3_运行与复用报告.md",
           f"# RQ3 {'完整描述性结果' if full else '部分结果（尚未完成）'}\n\n"
           f"已评分 {len(route_rows)//2}/{len(routes)} 路线。单票；两固定目标视图。\n\n"
           "复用明细见 ../reused_routes.csv，按部署模型汇总见 ../reuse_summary.csv；"
           "logical_matrix_results.csv 保留全部已完成单元及来源，重复文本并未删除逻辑轨迹。\n\n"
           "分组变化以同部署模型 p0 为基线，五个 seed 等权。当前仅描述性汇总，不将 seed、重复文本或投票当作独立数据集。"
           "Qwen3 独立运行时及 wrapper 差异须与模型差异一并披露。\n")
    emit("RQ3_ANALYSIS",complete=full,routes_scored=len(route_rows)//2)


def orchestrate(spec, config_path):
    out=Path(spec["output_root"]); out.mkdir(parents=True,exist_ok=True)
    with (out/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(out/"controller.json",{"pid":os.getpid(),"gpu":0,"start":now(),"code_root":str(ROOT),
                                    "resume_command":"bash run_target_matrix.sh"})
        plan(spec)
        def child(mode,model):
            python=spec["models"][model].get("python",spec["default_python"])
            cmd=[python,"-u",str(Path(__file__).resolve()),mode,"--config",str(config_path),"--model",model]
            env={**os.environ,"CUDA_VISIBLE_DEVICES":"0"}
            if model_runtime(spec,model).get("vllm_use_v1") is False:
                env["VLLM_USE_V1"]="0"
            emit("RQ3_CHILD",mode=mode,model=model,gpu=0)
            subprocess.run(cmd,env=env,check=True)
        for m in spec["model_order"]:
            child("context",m)
        reuse(spec)
        status(spec)
        for m in spec["model_order"]:
            if (out/"PAUSE").exists():
                emit("RQ3_PAUSED",stage="between_models"); return
            child("infer",m)
            analyze(spec)
        analyze(spec); status(spec)
        if (out/"PAUSE").exists():
            return
        assert read_json(out/"analysis/analysis_checks.json")["complete"]
        save(out/"completion.json",{"status":"complete","time":now(),"gpu":0})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["plan","context","reuse","infer","run","resume","status","analyze"])
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    args=parser.parse_args(); spec=config(args.config)
    if args.mode in ("run","resume"):
        orchestrate(spec,args.config.resolve())
    elif args.mode=="plan": plan(spec)
    elif args.mode=="context": context_audit(spec,args.model)
    elif args.mode=="reuse": reuse(spec)
    elif args.mode=="infer": infer(spec,args.model)
    elif args.mode=="status": status(spec)
    elif args.mode=="analyze": analyze(spec)


if __name__=="__main__":
    main()
