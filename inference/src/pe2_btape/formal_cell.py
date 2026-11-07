from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from pathlib import Path

from .backend import VLLMBackend
from .btape_optimizer import BTAPEConfig, run_btape
from .data import load_csv, sha256, stratified_three_way_split
from .metrics import INVALID
from .optimizer import PE2Config, run_pe2
from .prompts import classification_messages, extract_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen SecReq source-optimization matrix cell"
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--method", choices=("pe2", "btape"), required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def split_hash(ids: list[int]) -> str:
    encoded = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def label_counts(examples: list[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(item, "label")) for item in examples).items()))


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


def build_backend(
    spec: dict[str, object], model: dict[str, object], gpu: int
) -> VLLMBackend:
    runtime = dict(spec["runtime"])  # type: ignore[arg-type]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["VLLM_ATTENTION_BACKEND"] = str(runtime["attention_backend"])
    manifest_value = model.get("manifest_path")
    return VLLMBackend(
        model_name=str(model["repository"]),
        model_dir=str(model["model_dir"]),
        manifest_path=str(manifest_value) if manifest_value else None,
        expected_revision=str(model["revision"]),
        expected_vllm_version=str(runtime["vllm_version"]),
        physical_gpu=gpu,
        dtype=str(runtime["dtype"]),
        gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
        max_model_len=int(runtime["max_model_len"]),
        classification_max_model_len=int(runtime["classification_max_model_len"]),
        enforce_eager=bool(runtime["enforce_eager"]),
        engine_seed=int(runtime["engine_seed"]),
        attention_backend=str(runtime["attention_backend"]),
        stop_strings=[str(value) for value in model["stop_strings"]],  # type: ignore[index]
        max_batch_requests=int(runtime["max_batch_requests"]),
        guided_choice=[str(value) for value in spec["labels"]],  # type: ignore[index]
        guided_decoding_backend=str(runtime["guided_decoding_backend"]),
    )


def run_preflight(
    backend: VLLMBackend,
    labels: list[str],
    output_path: Path,
    classification_interface: str,
) -> dict[str, object]:
    definitions = (
        "Security: requirements that protect against unauthorized access or harm.\n"
        "Non-security: requirements not intended to protect against unauthorized access or harm."
    )
    requirement = "The system shall encrypt stored account passwords."
    system, user = classification_messages(
        definitions,
        [],
        requirement,
        labels,
        classification_interface,
    )
    raw_label, label_usage = backend.chat(
        system, user, kind="synthetic_preflight", num_predict=64
    )
    parsed = extract_label(raw_label, labels)
    if parsed == INVALID:
        raise RuntimeError(
            f"Guided binary preflight produced an unparsable response: {raw_label!r}"
        )
    proposal_system = "You are a precise prompt engineer."
    proposal_user = (
        "Rewrite these two definitions while preserving both class names. Return definitions only.\n"
        "Security: protects an asset.\nNon-security: does not protect an asset."
    )
    raw_proposal, proposal_usage = backend.chat(
        proposal_system, proposal_user, kind="proposal", num_predict=128
    )
    if len(raw_proposal.strip()) < 20 or raw_proposal.strip() in labels:
        raise RuntimeError(
            "Unconstrained proposer preflight did not produce a definition-like response"
        )
    payload = {
        "status": "passed",
        "classification": {
            "interface": classification_interface,
            "system": system,
            "user": user,
            "raw_output": raw_label,
            "parsed_label": parsed,
            "usage": label_usage,
        },
        "proposal": {"raw_output": raw_proposal, "usage": proposal_usage},
        "runtime": backend.runtime_metadata(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FORMAL_PREFLIGHT " + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    models = dict(spec["models"])
    if args.model_key not in models:
        raise ValueError(f"Unknown model key: {args.model_key}")
    if args.seed not in [int(value) for value in spec["seeds"]]:
        raise ValueError(f"Seed {args.seed} is not frozen in the matrix specification")
    if args.method not in [str(value) for value in spec["methods"]]:
        raise ValueError(f"Method {args.method} is not frozen in the matrix specification")
    model = dict(models[args.model_key])
    classification_interface = str(
        model.get("classification_interface", "appendix_f")
    )
    backend = build_backend(spec, model, args.gpu)
    output_path = Path(args.output)
    if args.preflight_only:
        run_preflight(
            backend,
            [str(value) for value in spec["labels"]],
            output_path,
            classification_interface,
        )
        return

    dataset_path = Path(str(spec["dataset_path"]))
    definitions_path = Path(str(spec["definitions_path"]))
    observed_dataset_sha = sha256(dataset_path)
    if observed_dataset_sha != str(spec["dataset_sha256"]):
        raise RuntimeError(
            f"Dataset SHA-256 mismatch: {observed_dataset_sha} != {spec['dataset_sha256']}"
        )
    examples = load_csv(dataset_path)
    if len(examples) != int(spec["dataset_rows"]):
        raise RuntimeError("Dataset row count differs from the frozen specification")
    observed_definitions_sha = sha256(definitions_path)
    if observed_definitions_sha != str(spec["definitions_sha256"]):
        raise RuntimeError(
            "Initial definitions SHA-256 differs from the frozen specification"
        )
    labels = [str(value) for value in spec["labels"]]
    if sorted({item.label for item in examples}) != sorted(labels):
        raise RuntimeError("Dataset labels differ from the frozen specification")
    split = stratified_three_way_split(examples, args.seed)
    observed_counts = {
        "pool": label_counts(split.pool),
        "validation": label_counts(split.val),
        "test": label_counts(split.test),
    }
    if observed_counts != spec["expected_split_label_counts"]:
        raise RuntimeError(
            f"Stratified split counts differ from the frozen specification: {observed_counts}"
        )

    optimization = dict(spec["optimization"])
    template_system, template_user = classification_messages(
        "{definitions}",
        [],
        "{requirement}",
        labels,
        classification_interface,
    )
    provenance = {
        "study_id": spec["study_id"],
        "protocol_version": spec["protocol_version"],
        "cell": {
            "method": args.method,
            "model_key": args.model_key,
            "model": model["repository"],
            "seed": args.seed,
        },
        "code_git_revision": git_revision(),
        "python": platform.python_version(),
        "spec_path": str(spec_path),
        "spec_sha256": sha256(spec_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": observed_dataset_sha,
        "definitions_path": str(definitions_path),
        "definitions_sha256": observed_definitions_sha,
        "dataset_rows": len(examples),
        "split_method": "independent label-stratified 30/30/40 repeated holdout",
        "split_sizes": {
            "pool": len(split.pool),
            "validation": len(split.val),
            "test": len(split.test),
        },
        "split_label_counts": observed_counts,
        "split_id_hashes": {
            "pool": split_hash([item.id for item in split.pool]),
            "validation": split_hash([item.id for item in split.val]),
            "test": split_hash([item.id for item in split.test]),
        },
        "model": model["repository"],
        "model_metadata": backend.model_metadata(),
        "hf_vllm_runtime": backend.runtime_metadata(),
        "gpu_runtime": gpu_metadata(),
        "decoding_temperature": 0.0,
        "voting_runs": int(optimization["voting_runs"]),
        "classification_interface": {
            "id": classification_interface,
            "system_template": template_system,
            "user_template": template_user,
            "message_roles": ["system", "user"],
            "definitions_location": "system",
            "requirement_location": "user",
        },
        "output_contract": {
            "type": "guided binary choice",
            "choices": labels,
            "backend": spec["runtime"]["guided_decoding_backend"],  # type: ignore[index]
            "violation_policy": "fail cell and preserve log; never map to a label",
        },
        "candidate_retention": "p0 and every proposed candidate with raw proposal, lineage, optimization evidence, full validation predictions, raw votes, and metrics",
        "source_test_policy": "after validation selection, evaluate p0 and p_star only",
    }
    print("FORMAL_CELL_PROVENANCE " + json.dumps(provenance, sort_keys=True), flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_definitions = definitions_path.read_text(encoding="utf-8").strip()
    if args.method == "pe2":
        config = PE2Config(
            seed=args.seed,
            n_max=int(optimization["n_max"]),
            history_size=int(optimization["pe2_history_size"]),
            error_batch_size=int(optimization["pe2_error_batch_size"]),
            voting_runs=int(optimization["voting_runs"]),
            positive_label=str(spec["positive_label"]),
            classification_num_predict=int(optimization["classification_num_predict"]),
            proposal_num_predict=int(optimization["proposal_num_predict"]),
            classification_interface=classification_interface,
        )
        run_pe2(
            backend=backend,
            split=split,
            labels=labels,
            initial_definitions=initial_definitions,
            config=config,
            checkpoint_path=output_path,
            provenance=provenance,
            resume=args.resume,
        )
    else:
        config = BTAPEConfig(
            seed=args.seed,
            n_max=int(optimization["n_max"]),
            backtrack_threshold=int(optimization["btape_backtrack_threshold"]),
            voting_runs=int(optimization["voting_runs"]),
            positive_label=str(spec["positive_label"]),
            classification_num_predict=int(optimization["classification_num_predict"]),
            proposal_num_predict=int(optimization["proposal_num_predict"]),
            classification_interface=classification_interface,
        )
        run_btape(
            backend=backend,
            split=split,
            labels=labels,
            initial_definitions=initial_definitions,
            config=config,
            checkpoint_path=output_path,
            provenance=provenance,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
