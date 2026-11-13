from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


EXPECTED_BUNDLE_FILES = {
    "summary.md",
    "trajectory.csv",
    "source-test-predictions.csv",
    "candidates.jsonl",
    "validation-predictions.jsonl",
    "best-prompt.txt",
    "environment.json",
    "split-manifest.json",
    "result-full.json",
    "run.log",
}


@dataclass(frozen=True)
class Cell:
    method: str
    model_key: str
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.model_key}__{self.method}__seed-{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen SecReq optimization matrix")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def read_result_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status"))
    except (OSError, json.JSONDecodeError):
        return "unreadable"


def bundle_complete(path: Path) -> bool:
    return path.is_dir() and EXPECTED_BUNDLE_FILES.issubset(
        {item.name for item in path.iterdir() if item.is_file()}
    )


def stream_process(
    command: list[str], cwd: Path, log_path: Path, prefix: str
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(f"[{prefix}] {line}", end="", flush=True)
        return process.wait()


def main() -> None:
    args = parse_args()
    workspace = Path.cwd()
    spec_path = Path(args.spec).resolve()
    output_root = Path(args.output_root).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cells = [
        Cell(method=str(method), model_key=str(model_key), seed=int(seed))
        for seed in spec["seeds"]
        for model_key in spec["models"]
        for method in spec["methods"]
    ]
    expected_cell_count = (
        len(spec["seeds"]) * len(spec["models"]) * len(spec["methods"])
    )
    if len(cells) != expected_cell_count or not cells:
        raise RuntimeError(
            f"Frozen matrix cell count mismatch: expected {expected_cell_count}, "
            f"found {len(cells)}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "study_id": spec["study_id"],
        "protocol_version": spec["protocol_version"],
        "spec": str(spec_path),
        "gpus": args.gpus,
        "cell_count": len(cells),
        "cells": [asdict(cell) | {"cell_id": cell.cell_id} for cell in cells],
    }
    atomic_json(output_root / "matrix-manifest.json", manifest)

    # Freeze cell-to-GPU placement so a later resume retains identical runtime
    # provenance. BT-APE receives weight 2 and PE2 weight 1 for rough balancing.
    gpu_loads = {gpu: 0 for gpu in args.gpus}
    assignments: dict[str, int] = {}
    queues: dict[int, queue.Queue[Cell]] = {
        gpu: queue.Queue() for gpu in args.gpus
    }
    for cell in sorted(cells, key=lambda item: (item.seed, item.model_key, item.method)):
        gpu = min(args.gpus, key=lambda candidate: (gpu_loads[candidate], candidate))
        assignments[cell.cell_id] = gpu
        gpu_loads[gpu] += 2 if cell.method == "btape" else 1
        queues[gpu].put(cell)
    manifest["assignments"] = assignments
    manifest["estimated_gpu_loads"] = gpu_loads
    atomic_json(output_root / "matrix-manifest.json", manifest)
    status_lock = threading.Lock()
    status: dict[str, dict[str, object]] = {}

    def save_status() -> None:
        with status_lock:
            summary = {
                "pending": sum(row.get("status") == "pending" for row in status.values()),
                "running": sum(row.get("status") == "running" for row in status.values()),
                "complete": sum(row.get("status") == "complete" for row in status.values()),
                "failed": sum(row.get("status") == "failed" for row in status.values()),
                "protocol_violation": sum(
                    row.get("status") == "protocol_violation" for row in status.values()
                ),
            }
            atomic_json(
                output_root / "matrix-status.json",
                {"summary": summary, "cells": status},
            )

    for cell in cells:
        result_path = (
            output_root
            / "cells"
            / cell.model_key
            / cell.method
            / f"seed-{cell.seed}"
            / "result-full.json"
        )
        existing = read_result_status(result_path)
        status[cell.cell_id] = {
            **asdict(cell),
            "status": "complete" if existing == "complete" else "pending",
            "assigned_gpu": assignments[cell.cell_id],
            "result": str(result_path),
            "attempts": 0,
        }
    save_status()

    def worker(gpu: int) -> None:
        cell_queue = queues[gpu]
        while True:
            try:
                cell = cell_queue.get_nowait()
            except queue.Empty:
                return
            cell_dir = (
                output_root
                / "cells"
                / cell.model_key
                / cell.method
                / f"seed-{cell.seed}"
            )
            result_path = cell_dir / "result-full.json"
            log_path = cell_dir / "run.log"
            if read_result_status(result_path) == "complete":
                bundle_dir = cell_dir / "bundle"
                if not bundle_complete(bundle_dir):
                    export_code = stream_process(
                        [
                            sys.executable,
                            "scripts/export_result.py",
                            "--result",
                            str(result_path),
                            "--run-log",
                            str(log_path),
                            "--out",
                            str(bundle_dir),
                        ],
                        workspace,
                        log_path,
                        f"EXPORT:{cell.cell_id}",
                    )
                    if export_code != 0:
                        with status_lock:
                            status[cell.cell_id]["status"] = "failed"
                            status[cell.cell_id]["exit_code"] = export_code
                        save_status()
                        cell_queue.task_done()
                        continue
                with status_lock:
                    status[cell.cell_id]["status"] = "complete"
                    status[cell.cell_id]["reused"] = True
                save_status()
                cell_queue.task_done()
                continue

            final_status = "failed"
            last_code: int | None = None
            for attempt in range(1, args.max_attempts + 1):
                with status_lock:
                    status[cell.cell_id].update({
                        "status": "running",
                        "gpu": gpu,
                        "attempts": attempt,
                        "started_unix": time.time(),
                    })
                save_status()
                command = [
                    sys.executable,
                    "-m",
                    "pe2_btape.formal_cell",
                    "--spec",
                    str(spec_path),
                    "--method",
                    cell.method,
                    "--model-key",
                    cell.model_key,
                    "--seed",
                    str(cell.seed),
                    "--gpu",
                    str(gpu),
                    "--output",
                    str(result_path),
                    "--resume",
                ]
                last_code = stream_process(
                    command, workspace, log_path, f"GPU{gpu}:{cell.cell_id}"
                )
                observed = read_result_status(result_path)
                if last_code == 0 and observed == "complete":
                    export_code = stream_process(
                        [
                            sys.executable,
                            "scripts/export_result.py",
                            "--result",
                            str(result_path),
                            "--run-log",
                            str(log_path),
                            "--out",
                            str(cell_dir / "bundle"),
                        ],
                        workspace,
                        log_path,
                        f"EXPORT:{cell.cell_id}",
                    )
                    if export_code == 0:
                        final_status = "complete"
                        break
                    last_code = export_code
                if observed == "protocol_violation":
                    final_status = "protocol_violation"
                    break
                print(
                    f"CELL_RETRY cell={cell.cell_id} gpu={gpu} attempt={attempt} "
                    f"exit_code={last_code} checkpoint_status={observed}",
                    flush=True,
                )

            if final_status != "complete":
                failure = {
                    "cell": asdict(cell),
                    "status": final_status,
                    "exit_code": last_code,
                    "checkpoint_status": read_result_status(result_path),
                    "traceback": None,
                    "finished_unix": time.time(),
                }
                atomic_json(cell_dir / "failure.json", failure)
            with status_lock:
                status[cell.cell_id].update({
                    "status": final_status,
                    "exit_code": last_code,
                    "finished_unix": time.time(),
                })
            save_status()
            cell_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}", daemon=False)
        for gpu in args.gpus
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except BaseException:
        atomic_json(
            output_root / "scheduler-failure.json",
            {"traceback": traceback.format_exc(), "unix": time.time()},
        )
        raise

    save_status()
    counts = Counter(row["status"] for row in status.values())
    summary = {
        "status": "complete" if counts["complete"] == len(cells) else "complete_with_cell_failures",
        "cell_count": len(cells),
        "counts": dict(counts),
        "output_root": str(output_root),
    }
    atomic_json(output_root / "matrix-summary.json", summary)
    print("FORMAL_MATRIX_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
