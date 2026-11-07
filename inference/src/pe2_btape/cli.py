from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from .backend import OllamaBackend
from .data import limit_split, load_csv, sha256, three_way_split
from .optimizer import PE2Config, run_pe2


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the PE2 baseline protocol in BT-APE Appendix B"
    )
    parser.add_argument("--config", default="configs/secreq_llama_fp16.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--n-max", type=int)
    parser.add_argument("--max-pool", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_path = Path(raw["dataset_path"])
    definitions_path = Path(raw["definitions_path"])
    output_path = Path(args.output or raw["output_path"])
    examples = load_csv(dataset_path)
    labels = list(raw["labels"])
    observed_labels = sorted({example.label for example in examples})
    if sorted(labels) != observed_labels:
        raise ValueError(
            f"Configured labels {sorted(labels)} do not match dataset {observed_labels}"
        )
    full_split = three_way_split(examples, int(raw["seed"]))
    split = limit_split(
        full_split, args.max_pool, args.max_val, args.max_test
    )
    mode = "smoke" if any((args.max_pool, args.max_val, args.max_test)) else "formal"
    n_max = args.n_max if args.n_max is not None else int(raw["n_max"])
    if mode == "formal" and n_max != 20:
        raise ValueError("A formal paper-matched run must use n_max=20")

    optimizer_config = PE2Config(
        seed=int(raw["seed"]),
        n_max=n_max,
        history_size=int(raw["history_size"]),
        error_batch_size=int(raw["error_batch_size"]),
        voting_runs=int(raw["voting_runs"]),
        positive_label=str(raw["positive_label"]),
        classification_num_predict=int(raw["classification_num_predict"]),
        proposal_num_predict=int(raw["proposal_num_predict"]),
    )
    backend = OllamaBackend(
        model=str(raw["model"]),
        base_url=str(raw["base_url"]),
        timeout=int(raw.get("timeout_seconds", 600)),
    )
    model_metadata = backend.model_metadata()
    provenance = {
        "mode": mode,
        "code_git_revision": git_revision(),
        "python": platform.python_version(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "definitions_path": str(definitions_path),
        "definitions_sha256": sha256(definitions_path),
        "dataset_rows": len(examples),
        "split_sizes_before_smoke_limits": {
            "pool": len(full_split.pool),
            "val": len(full_split.val),
            "test": len(full_split.test),
        },
        "split_sizes_executed": {
            "pool": len(split.pool),
            "val": len(split.val),
            "test": len(split.test),
        },
        "model": raw["model"],
        "model_metadata": model_metadata,
        "base_url": raw["base_url"],
        "decoding_temperature": 0.0,
        "runtime_note": "Ollama differs from the paper's vLLM 0.6.x/H200 runtime",
    }
    print("PE2_RUN_PROVENANCE " + json.dumps(provenance, sort_keys=True), flush=True)
    print(
        f"PE2_SPLIT pool={len(split.pool)} val={len(split.val)} test={len(split.test)}",
        flush=True,
    )
    run_pe2(
        backend=backend,
        split=split,
        labels=labels,
        initial_definitions=definitions_path.read_text(encoding="utf-8").strip(),
        config=optimizer_config,
        checkpoint_path=output_path,
        provenance=provenance,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
