from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight all formal producer models")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    gpu_pool = list(args.gpus)
    gpu_lock = threading.Condition()

    def acquire_gpu() -> int:
        with gpu_lock:
            while not gpu_pool:
                gpu_lock.wait()
            return gpu_pool.pop(0)

    def release_gpu(gpu: int) -> None:
        with gpu_lock:
            gpu_pool.append(gpu)
            gpu_lock.notify()

    def run_model(model_key: str) -> dict[str, object]:
        gpu = acquire_gpu()
        output = output_root / f"{model_key}.json"
        log_path = output_root / f"{model_key}.log"
        command = [
            sys.executable,
            "-m",
            "pe2_btape.formal_cell",
            "--spec",
            str(spec_path),
            "--method",
            "pe2",
            "--model-key",
            model_key,
            "--seed",
            str(spec["seeds"][0]),
            "--gpu",
            str(gpu),
            "--output",
            str(output),
            "--preflight-only",
        ]
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(f"[PREFLIGHT:GPU{gpu}:{model_key}] {line}", end="", flush=True)
                code = process.wait()
            return {
                "model_key": model_key,
                "gpu": gpu,
                "exit_code": code,
                "status": "passed" if code == 0 and output.is_file() else "failed",
                "output": str(output),
                "log": str(log_path),
            }
        finally:
            release_gpu(gpu)

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(run_model, str(model_key)): str(model_key)
            for model_key in spec["models"]
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: str(row["model_key"]))
    summary = {
        "status": "passed" if all(row["status"] == "passed" for row in results) else "failed",
        "models": results,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FORMAL_PREFLIGHT_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
