from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from .backend import VLLMBackend
from .btape_optimizer import BTAPEConfig, run_btape
from .cli import git_revision
from .data import limit_split, load_csv, sha256, three_way_split
from .metrics import INVALID
from .prompts import classification_system, extract_label


def split_hash(ids: list[int]) -> str:
    encoded = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def gpu_metadata() -> dict[str, object]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"nvidia_smi_rows": output.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"warning": "nvidia-smi metadata unavailable"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce corrected BT-APE Algorithm 1")
    parser.add_argument("--config", default="configs/btape_secreq_llama_fp16.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--n-max", type=int)
    parser.add_argument("--max-pool", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    return parser.parse_args()


def build_vllm_backend(raw: dict[str, object]) -> VLLMBackend:
    if raw.get("backend") != "hf_vllm":
        raise ValueError("This rerun is frozen to the hf_vllm backend")
    config = dict(raw["hf_vllm"])
    return VLLMBackend(
        model_name=str(raw["model"]),
        model_dir=str(config["model_dir"]),
        manifest_path=str(config["manifest_path"]),
        expected_revision=str(config["expected_revision"]),
        expected_vllm_version=str(config["expected_vllm_version"]),
        physical_gpu=int(config["physical_gpu"]),
        dtype=str(config["dtype"]),
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        max_model_len=int(config["max_model_len"]),
        classification_max_model_len=int(
            config.get("classification_max_model_len", config["max_model_len"])
        ),
        enforce_eager=bool(config["enforce_eager"]),
        engine_seed=int(config["engine_seed"]),
        attention_backend=str(config["attention_backend"]),
        stop_strings=[str(value) for value in config["stop_strings"]],
        max_batch_requests=int(config["max_batch_requests"]),
    )


def run_synthetic_preflight(backend: VLLMBackend, labels: list[str]) -> None:
    """Validate output formatting without reading a SecReq row."""
    system = classification_system(labels)
    user = (
        "Definitions:\n"
        "Security: requirements that protect against unauthorized access or harm.\n"
        "Non-security: requirements not intended to protect against unauthorized access or harm.\n\n"
        "Requirement: The system shall encrypt stored account passwords.\n\n"
        "Using the Definitions above, classify the requirement and provide the final label in the format:\n"
        '\"Label: {class}\".'
    )
    raw, usage = backend.chat(
        system, user, kind="synthetic_preflight", num_predict=64
    )
    parsed = extract_label(raw, labels)
    evidence = {
        "parsed_label": parsed,
        "raw_output": raw,
        "backend": usage.get("backend"),
        "finish_reason": usage.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_eval_count"),
        "output_tokens": usage.get("eval_count"),
    }
    print("HF_VLLM_PREFLIGHT " + json.dumps(evidence, sort_keys=True), flush=True)
    if parsed == INVALID:
        raise RuntimeError("HF/vLLM synthetic preflight did not yield a valid binary label")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_path = Path(raw["dataset_path"])
    definitions_path = Path(raw["definitions_path"])
    output_path = Path(args.output or raw["output_path"])
    examples = load_csv(dataset_path)
    labels = list(raw["labels"])
    if sorted(labels) != sorted({example.label for example in examples}):
        raise ValueError("Configured labels do not match dataset labels")
    if labels != ["Non-security", "Security"]:
        raise ValueError("The frozen HF/vLLM protocol requires sorted SecReq labels")
    if float(raw["temperature"]) != 0.0:
        raise ValueError("The frozen HF/vLLM protocol requires temperature=0")
    full_split = three_way_split(examples, int(raw["seed"]))
    split = limit_split(full_split, args.max_pool, args.max_val, args.max_test)
    mode = "smoke" if any((args.max_pool, args.max_val, args.max_test)) else "formal"
    n_max = args.n_max if args.n_max is not None else int(raw["n_max"])
    if mode == "formal" and n_max != 20:
        raise ValueError("A formal paper-matched run must use n_max=20")
    config = BTAPEConfig(
        seed=int(raw["seed"]),
        n_max=n_max,
        backtrack_threshold=int(raw["backtrack_threshold"]),
        voting_runs=int(raw["voting_runs"]),
        positive_label=str(raw["positive_label"]),
        classification_num_predict=int(raw["classification_num_predict"]),
        proposal_num_predict=int(raw["proposal_num_predict"]),
    )
    backend = build_vllm_backend(raw)
    run_synthetic_preflight(backend, labels)
    provenance = {
        "protocol_version": raw["protocol_version"],
        "mode": mode,
        "code_git_revision": git_revision(),
        "config_sha256": sha256(config_path),
        "python": platform.python_version(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "definitions_path": str(definitions_path),
        "definitions_sha256": sha256(definitions_path),
        "dataset_rows": len(examples),
        "split_sizes_before_smoke_limits": {
            "pool": len(full_split.pool), "val": len(full_split.val), "test": len(full_split.test),
        },
        "split_sizes_executed": {
            "pool": len(split.pool), "val": len(split.val), "test": len(split.test),
        },
        "model": raw["model"],
        "model_metadata": backend.model_metadata(),
        "hf_vllm_runtime": backend.runtime_metadata(),
        "gpu_runtime": gpu_metadata(),
        "backend": raw["backend"],
        "decoding_temperature": raw["temperature"],
        "classification_num_predict": config.classification_num_predict,
        "proposal_num_predict": config.proposal_num_predict,
        "voting_runs": config.voting_runs,
        "deployment_contract": {
            "classification_context_limit": backend.runtime_metadata()[
                "classification_max_model_len"
            ],
            "request_dispatch_batch_size": backend.runtime_metadata()[
                "request_dispatch_batch_size"
            ],
            "max_num_seqs": backend.runtime_metadata()["max_num_seqs"],
            "proposal_engine_context_limit": backend.runtime_metadata()[
                "max_model_len"
            ],
            "proposal_context_exception": (
                "The engine retains 8192 tokens only for BT-APE definition proposals: "
                "the observed maximum proposal prompt is 1129 tokens and its frozen "
                "1536-token generation budget requires 2665 tokens."
            ),
        },
        "split_id_hashes": {
            "pool": split_hash([example.id for example in split.pool]),
            "val": split_hash([example.id for example in split.val]),
            "test": split_hash([example.id for example in split.test]),
        },
        "runtime_note": (
            "Paper-text reconstruction: verified original HF safetensors through "
            "vLLM 0.6.4 on a V100; hardware remains different from the paper's H200."
        ),
    }
    print("BTAPE_RUN_PROVENANCE " + json.dumps(provenance, sort_keys=True), flush=True)
    print(f"BTAPE_SPLIT pool={len(split.pool)} val={len(split.val)} test={len(split.test)}", flush=True)
    run_btape(
        backend=backend,
        split=split,
        labels=labels,
        initial_definitions=definitions_path.read_text(encoding="utf-8").strip(),
        config=config,
        checkpoint_path=output_path,
        provenance=provenance,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
