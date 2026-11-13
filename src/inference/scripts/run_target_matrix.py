"""Frozen five-model preflight and final-prompt GitReq deployment.

This is intentionally a two-stage orx run command.  A successful preflight
must be frozen on its own experiment branch before stage=matrix is committed
to a child experiment.  Model jobs use separate processes so CUDA memory is
released between checkpoints; two physical GPUs are never oversubscribed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pe2_btape.backend import VLLMBackend
from pe2_btape.data import load_csv
from pe2_btape.prompts import classification_messages, extract_label, majority

from validate_gitreq_construct import audit as construct_audit


PRODUCERS = ("llama3_8b", "qwen2_7b", "granite3_2_8b")
OPTIMIZERS = ("pe2", "btape")
SEEDS = (40, 41, 42, 43, 44)
GPU_QUEUES = {0: ("llama3_8b", "granite3_2_8b", "falcon3_7b"),
              1: ("qwen2_7b", "ministral_8b")}


def gpu_queues(spec: dict[str, Any]) -> dict[int, tuple[str, ...]]:
    raw = spec.get("gpu_queues")
    if raw is None:
        return GPU_QUEUES
    queues = {int(gpu): tuple(keys) for gpu, keys in raw.items()}
    assigned = [key for keys in queues.values() for key in keys]
    if sorted(assigned) != sorted(spec["models"]):
        raise RuntimeError("GPU queues must assign each configured model exactly once")
    if any(gpu not in (0, 1) for gpu in queues):
        raise RuntimeError("Only the two verified physical V100 GPUs are allowed")
    return queues


def model_runtime(spec: dict[str, Any], key: str) -> dict[str, Any]:
    return {**spec["runtime"], **spec["models"][key].get("runtime", {})}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def interface_contract_sha(spec: dict[str, Any]) -> str:
    """Stage is the sole allowed difference between preflight and matrix specs."""
    contract = {key: value for key, value in spec.items() if key != "stage"}
    return digest(canonical(contract).encode("utf-8"))


def immutable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Refusing to overwrite a different frozen file: {path}")
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Refusing to overwrite a different frozen file: {path}")
    finally:
        temporary.unlink()


def load_inputs(spec: dict[str, Any], *, verify_archive: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    source_archive = Path(spec["source_archive"])
    if verify_archive and sha_file(source_archive) != spec["source_archive_sha256"]:
        raise RuntimeError("Frozen 30-run source archive SHA-256 mismatch")
    matrix = read_json(Path(spec["source_matrix"]) / "matrix-manifest.json")
    cells = matrix["cells"]
    expected = {(p, o, s) for p in PRODUCERS for o in OPTIMIZERS for s in SEEDS}
    observed = {(c["model_key"], c["method"], int(c["seed"])) for c in cells}
    if len(cells) != 30 or observed != expected:
        raise RuntimeError("Source matrix is not the frozen 30-cell design")
    prompts = []
    p0_texts = set()
    seed42_splits = []
    for cell in cells:
        result_path = Path(spec["source_matrix"]) / "cells" / cell["model_key"] / cell["method"] / f"seed-{cell['seed']}" / "result-full.json"
        result = read_json(result_path)
        if result.get("status") != "complete" or len(result.get("states", [])) != 21:
            raise RuntimeError(f"Incomplete source cell: {result_path}")
        initial = result["initialization"]["state"]["definitions"]
        selected = result["best_state"]["definitions"]
        p0_texts.add(initial)
        if int(cell["seed"]) == 42:
            seed42_splits.append(result["split"]["test_ids"])
        prompts.append({
            "prompt_id": f"{cell['model_key']}__{cell['method']}__seed-{cell['seed']}",
            "producer": cell["model_key"], "optimizer": cell["method"],
            "seed": int(cell["seed"]),
            "semantic_text": selected,
            "semantic_sha256": digest(selected.encode("utf-8")),
            "source_best_iteration": result["best_state"]["iteration"],
            "source_result_sha256": sha_file(result_path),
        })
    if len(p0_texts) != 1:
        raise RuntimeError("Source runs disagree on initial p0 definition text")
    if not seed42_splits or any(ids != seed42_splits[0] for ids in seed42_splits):
        raise RuntimeError("Source runs disagree on seed=42 test membership/order")
    p0_text = next(iter(p0_texts))
    p0 = {"prompt_id": "p0", "producer": None, "optimizer": None, "seed": None,
          "semantic_text": p0_text, "semantic_sha256": digest(p0_text.encode("utf-8"))}
    source = load_csv(spec["source_csv"])
    if len(source) != 510 or len(seed42_splits[0]) != 204:
        raise RuntimeError("Wrong SecReq source-test size")
    for row_id in seed42_splits[0]:
        if row_id < 0 or row_id >= len(source):
            raise RuntimeError("Invalid source-test item ID")
    return {"p0": p0, "prompts": prompts, "seed42_test_ids": seed42_splits[0]}, {
        "source": source, "matrix_protocol": matrix["protocol_version"]}


def create_backend(spec: dict[str, Any], key: str, gpu: int) -> VLLMBackend:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    model = spec["models"][key]
    runtime = model_runtime(spec, key)
    if runtime.get("vllm_use_v1") is False:
        os.environ["VLLM_USE_V1"] = "0"
    backend = VLLMBackend(
        model_name=model["repository"], model_dir=model["model_dir"],
        manifest_path=None, expected_revision=model["revision"],
        expected_vllm_version=runtime["vllm_version"], physical_gpu=gpu,
        dtype=runtime["dtype"], gpu_memory_utilization=runtime["gpu_memory_utilization"],
        max_model_len=runtime["max_model_len"],
        classification_max_model_len=runtime["classification_max_model_len"],
        enforce_eager=runtime["enforce_eager"], engine_seed=runtime["engine_seed"],
        attention_backend=runtime["attention_backend"],
        stop_strings=model["stop_strings"], max_batch_requests=1,
        guided_choice=spec["labels"],
        guided_decoding_backend=runtime["guided_decoding_backend"],
        chat_template_kwargs=model.get("chat_template_kwargs"),
        emit_batch_logs=False,
    )
    observed = backend.runtime_metadata()["local_artifact"]["weight_index_sha256"]
    if observed != model["weight_index_sha256"]:
        raise RuntimeError(f"Weight-index hash mismatch for {key}")
    if not backend.runtime_metadata()["tokenizer_chat_template_sha256"]:
        raise RuntimeError(f"Missing chat-template fingerprint for {key}")
    return backend


def vote(backend: VLLMBackend, spec: dict[str, Any], model: dict[str, Any],
         definition: str, text: str) -> dict[str, Any]:
    system, user = classification_messages(
        definition, [], text, spec["labels"], model["interface"]
    )
    raw, record = backend.chat(
        system, user, kind="classification:target_deployment",
        num_predict=spec["runtime"]["classification_num_predict"],
    )
    label = extract_label(raw, spec["labels"])
    if label not in spec["labels"]:
        raise RuntimeError(f"Nonbinary model output: {raw!r}")
    if record["finish_reason"] not in ("stop", "length"):
        raise RuntimeError(f"Unexpected completion reason: {record['finish_reason']}")
    return {"raw": raw, "label": label, "finish_reason": record["finish_reason"],
            "rendered_sha256": record["hf_rendered_prompt_sha256"],
            "input_tokens": record["prompt_eval_count"],
            "output_tokens": record["eval_count"]}


def worker_preflight(spec: dict[str, Any], key: str, gpu: int,
                     prompt_data: dict[str, Any], source: list[Any],
                     output_root: Path, spec_sha: str) -> None:
    model = spec["models"][key]
    backend = create_backend(spec, key, gpu)
    rows = []
    start = time.monotonic()
    for index, row_id in enumerate(prompt_data["seed42_test_ids"]):
        example = source[row_id]
        votes = [vote(backend, spec, model, prompt_data["p0"]["semantic_text"], example.text)
                 for _ in range(spec["preflight_repeats"])]
        rows.append({"source_id": row_id, "gold": example.label, "votes": votes,
                     "majority": majority([v["label"] for v in votes])})
        if (index + 1) % 51 == 0:
            print("PREFLIGHT_PROGRESS " + canonical({"model": key, "rows": index + 1}), flush=True)
            backend._calls.clear()
            backend._render_cache.clear()
    consistency = sum(len({v["label"] for v in row["votes"]}) == 1 for row in rows) / len(rows)
    repeat_count = (1 if consistency >= spec["minimum_repeat_consistency"] else 3)
    result = {
        "status": "passed", "model_key": key, "gpu": gpu, "spec_sha256": spec_sha,
        "model_runtime": backend.runtime_metadata(),
        "interface": model["interface"], "preflight_prompt_sha256": prompt_data["p0"]["semantic_sha256"],
        "source_seed": 42, "source_items": len(rows), "valid_votes": 3 * len(rows),
        "three_repeat_agreement": consistency, "formal_rq3_votes": repeat_count,
        "elapsed_seconds": time.monotonic() - start, "predictions": rows,
    }
    immutable_json(output_root / "preflight" / f"{key}.json", result)
    print("MODEL_PREFLIGHT_PASS " + canonical({"model": key, "agreement": consistency,
                                                "rq3_votes": repeat_count}), flush=True)


def read_target(spec: dict[str, Any]) -> list[dict[str, str]]:
    path = Path(spec["gitreq_enriched"])
    if sha_file(path) != spec["gitreq_enriched_sha256"]:
        raise RuntimeError("GitReq enriched CSV SHA-256 mismatch")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6302 or len({r["gitreq_id"] for r in rows}) != 6302:
        raise RuntimeError("Unexpected GitReq row count or IDs")
    counts = Counter(r["original_label"] for r in rows)
    if counts["Security"] != 1646 or counts["Functional"] != 531:
        raise RuntimeError("GitReq target class counts changed")
    if any(r["binary_label"] != ("Security" if r["original_label"] == "Security" else "Non-security") for r in rows):
        raise RuntimeError("GitReq binary mapping mismatch")
    return rows


def worker_matrix(spec: dict[str, Any], key: str, gpu: int,
                  prompt_data: dict[str, Any], output_root: Path,
                  preflight: dict[str, Any]) -> None:
    target = read_target(spec)
    model = spec["models"][key]
    backend = create_backend(spec, key, gpu)
    for field in ("tokenizer_chat_template_sha256", "tokenizer_config_sha256"):
        before = preflight["model_runtime"].get(field) if field == "tokenizer_chat_template_sha256" else preflight["model_runtime"]["local_artifact"].get(field)
        now = backend.runtime_metadata().get(field) if field == "tokenizer_chat_template_sha256" else backend.runtime_metadata()["local_artifact"].get(field)
        if before != now:
            raise RuntimeError(f"Model interface fingerprint changed after preflight: {key}/{field}")
    for ordinal, prompt in enumerate([prompt_data["p0"]] + prompt_data["prompts"]):
        prompt_id = prompt["prompt_id"]
        path = output_root / "matrix" / "predictions" / key / f"{prompt_id}.jsonl"
        marker = path.with_suffix(".complete.json")
        route_votes = int(preflight["formal_rq3_votes"])
        is_origin = (prompt_id == "p0" and key in PRODUCERS) or prompt.get("producer") == key
        origin_votes = spec["origin_target_repeats"] if is_origin else None
        required_votes = max(route_votes, origin_votes or 0)
        expected_marker = {"model": key, "prompt_id": prompt_id,
                           "semantic_sha256": prompt["semantic_sha256"],
                           "rows": len(target), "votes_per_item": required_votes}
        if marker.exists():
            previous = read_json(marker)
            if any(previous.get(k) != v for k, v in expected_marker.items()):
                raise RuntimeError(f"Completed route marker disagrees: {marker}")
            print("MATRIX_ROUTE_REUSED " + canonical({"model": key, "prompt": prompt_id}), flush=True)
            continue
        if path.exists() or path.with_suffix(".partial.jsonl").exists():
            raise RuntimeError(f"Existing incomplete target evidence needs manual audit: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.jsonl")
        started = time.monotonic()
        with partial.open("x", encoding="utf-8") as handle:
            for index, row in enumerate(target):
                votes = [vote(backend, spec, model, prompt["semantic_text"], row["requirement_text"])
                         for _ in range(required_votes)]
                output = {"gitreq_id": row["gitreq_id"], "original_label": row["original_label"],
                          "gold_binary": row["binary_label"], "repository": row["repository"],
                          "url": row["url"], "prompt_id": prompt_id,
                          "semantic_sha256": prompt["semantic_sha256"], "votes": votes,
                          "rq3_prediction": majority([v["label"] for v in votes[:route_votes]]),
                          "origin_prediction": (majority([v["label"] for v in votes[:origin_votes]])
                                                if origin_votes is not None else None)}
                handle.write(canonical(output) + "\n")
                if (index + 1) % 100 == 0:
                    handle.flush()
                    backend._calls.clear()
                    backend._render_cache.clear()
                if (index + 1) % 1000 == 0:
                    print("MATRIX_PROGRESS " + canonical({"model": key, "prompt": prompt_id,
                                                        "items": index + 1,
                                                        "prompt_ordinal": ordinal + 1}), flush=True)
        partial.rename(path)
        immutable_json(marker, {**expected_marker, "prediction_sha256": sha_file(path),
                                "elapsed_seconds": time.monotonic() - started,
                                "rq3_votes": route_votes, "origin_votes": origin_votes})
        print("MATRIX_ROUTE_COMPLETE " + canonical({"model": key, "prompt": prompt_id,
                                                   "items": len(target)}), flush=True)


def run_workers(spec_path: Path, stage: str, output_root: Path) -> list[dict[str, Any]]:
    spec = read_json(spec_path)
    def queue(gpu: int, model_keys: tuple[str, ...]) -> list[dict[str, Any]]:
        results = []
        for key in model_keys:
            if stage == "preflight":
                result_path = output_root / "preflight" / f"{key}.json"
                if result_path.exists():
                    prior = read_json(result_path)
                    if (prior.get("status") == "passed"
                            and prior.get("spec_sha256") == sha_file(spec_path)
                            and prior.get("source_items") == 204
                            and prior.get("valid_votes") == 612):
                        results.append({"model": key, "gpu": gpu, "exit_code": 0,
                                        "reused": True, "result": str(result_path)})
                        continue
                    raise RuntimeError(f"Conflicting preflight artifact: {result_path}")
            python = spec["models"][key].get("python", sys.executable)
            command = [python, __file__, "--spec", str(spec_path),
                       "--worker", stage, "--model", key, "--gpu", str(gpu)]
            log_path = output_root / stage / "logs" / f"{key}-{time.time_ns()}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("x", encoding="utf-8") as handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    if any(tag in line for tag in ("PREFLIGHT_PROGRESS", "MODEL_PREFLIGHT_PASS",
                                                   "MATRIX_PROGRESS", "MATRIX_ROUTE_COMPLETE",
                                                   "Traceback", "Error")):
                        print(f"[{stage}:GPU{gpu}:{key}] {line}", end="", flush=True)
                exit_code = process.wait()
            results.append({"model": key, "gpu": gpu, "exit_code": exit_code,
                            "log": str(log_path)})
            if exit_code:
                break
        return results
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, gpu, keys) for gpu, keys in gpu_queues(spec).items()]
        return [row for future in futures for row in future.result()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--worker", choices=("preflight", "matrix"))
    parser.add_argument("--model")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = read_json(spec_path)
    output_root = Path(spec["output_root"])
    spec_sha = sha_file(spec_path)
    if (spec.get("version") == "target-deployment-extra-preflight-v1"
            and spec.get("stage") != "preflight"):
        raise RuntimeError("Additional-model preflight configuration cannot run GitReq inference")

    if args.worker:
        if (args.model not in spec["models"]
                or args.model not in gpu_queues(spec).get(args.gpu, ())):
            raise RuntimeError("Invalid worker model/GPU")
        prompt_data, additional = load_inputs(spec, verify_archive=False)
        if args.worker == "preflight":
            worker_preflight(spec, args.model, args.gpu, prompt_data,
                             additional["source"], output_root, spec_sha)
        else:
            preflight = read_json(output_root / "preflight" / f"{args.model}.json")
            if preflight.get("status") != "passed":
                raise RuntimeError("Model has no passing preflight")
            worker_matrix(spec, args.model, args.gpu, prompt_data, output_root, preflight)
        return

    if spec["stage"] not in ("preflight", "matrix"):
        raise RuntimeError("Unknown deployment stage")
    prompt_data, _ = load_inputs(spec, verify_archive=True)
    audit = construct_audit(Path(spec["source_csv"]), Path(spec["gitreq_nfr"]),
                            Path(spec["gitreq_fr"]))
    if audit["status"] != "PASS_WITH_CONSTRUCT_LIMITS":
        raise RuntimeError("GitReq construct gate failed")
    if sha_file(Path(spec["gitreq_enriched"])) != spec["gitreq_enriched_sha256"]:
        raise RuntimeError("Enriched GitReq hash changed")
    immutable_json(output_root / "construct-audit.json", audit)
    print("CONSTRUCT_GATE_PASS " + canonical({"source_rows": audit["source_rows"],
                                              "target_rows": audit["gitreq_rows"]}), flush=True)
    if spec["stage"] == "preflight":
        immutable_json(output_root / "preflight" / "frozen-prompts.json", prompt_data)
        results = run_workers(spec_path, "preflight", output_root)
        passed = len(results) == len(spec["models"]) and all(row["exit_code"] == 0 for row in results)
        summary = {"status": "passed" if passed else "failed", "spec_sha256": spec_sha,
                   "interface_contract_sha256": interface_contract_sha(spec),
                   "model_runs": results, "construct_audit": "PASS_WITH_CONSTRUCT_LIMITS"}
        if passed:
            immutable_json(output_root / "preflight" / "summary.json", summary)
        else:
            immutable_json(output_root / "preflight" / f"failed-attempt-{time.time_ns()}.json", summary)
        print("ADDITIONAL_MODEL_PREFLIGHT_SUMMARY " + canonical(summary), flush=True)
        if not passed:
            raise SystemExit(1)
    else:
        preflight_summary = read_json(output_root / "preflight" / "summary.json")
        if preflight_summary.get("status") != "passed":
            raise RuntimeError("Five-model preflight has not passed")
        if preflight_summary.get("interface_contract_sha256") != interface_contract_sha(spec):
            raise RuntimeError("Preflight and formal deployment interface contracts differ")
        frozen = read_json(output_root / "preflight" / "frozen-prompts.json")
        if frozen != prompt_data:
            raise RuntimeError("Source prompt artifacts changed after preflight")
        target = read_target(spec)
        routes = [{"producer": prompt["producer"], "optimizer": prompt["optimizer"],
                   "seed": prompt["seed"], "prompt_id": prompt["prompt_id"],
                   "deployer": deployer, "semantic_sha256": prompt["semantic_sha256"]}
                  for prompt in prompt_data["prompts"] for deployer in spec["models"]]
        if len(routes) != 150:
            raise RuntimeError("Route manifest is not 3x5x2x5")
        manifest = {"status": "frozen_before_target_inference", "spec_sha256": spec_sha,
                    "source_archive_sha256": spec["source_archive_sha256"],
                    "gitreq_enriched_sha256": spec["gitreq_enriched_sha256"],
                    "target_rows": len(target), "routes": routes,
                    "p0_sha256": prompt_data["p0"]["semantic_sha256"],
                    "prior_gitreq_pilots_disclosed": True}
        immutable_json(output_root / "matrix" / "route-manifest.json", manifest)
        print("ROUTE_MANIFEST_FROZEN " + canonical({"routes": len(routes),
                                                   "target_rows": len(target)}), flush=True)
        results = run_workers(spec_path, "matrix", output_root)
        passed = len(results) == 5 and all(row["exit_code"] == 0 for row in results)
        summary = {"status": "complete" if passed else "failed", "spec_sha256": spec_sha,
                   "model_runs": results, "route_count": 150}
        immutable_json(output_root / "matrix" / "summary.json", summary)
        print("TARGET_MATRIX_SUMMARY " + canonical(summary), flush=True)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
