from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


STANDARD_FILES = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a readable result bundle")
    parser.add_argument("--result", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = Path(args.result)
    log_path = Path(args.run_log)
    out = Path(args.out)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("Only a complete result may be exported")
    out.mkdir(parents=True, exist_ok=True)

    source_tests = payload.get("source_tests") or {"p_star": payload["final_test"]}
    final_test = source_tests["p_star"]
    initial_test = source_tests.get("p0")
    predictions = final_test["predictions"]
    metrics = final_test["metrics"]
    best = payload["best_state"]
    labels = sorted({str(row["gold"]) for row in predictions})
    confusion = Counter((row["gold"], row["prediction"]) for row in predictions)
    usage = payload.get("usage", {})
    lines = [
        f"# {payload['method']}：运行摘要",
        "",
        "## 最终测试结果",
        "",
        f"- Weighted-F1: **{float(metrics['weighted_f1']):.6f}**",
        f"- Macro-F1: **{float(metrics['macro_f1']):.6f}**",
        f"- Accuracy: **{float(metrics['accuracy']):.6f}**",
        f"- Balanced Accuracy: **{float(metrics.get('balanced_accuracy', 0.0)):.6f}**",
        f"- MCC: **{float(metrics.get('matthews_correlation', 0.0)):.6f}**",
        f"- INVALID: **{int(metrics['invalid_count'])}**",
        f"- 最佳迭代: **{best['iteration']}**",
        f"- 最佳验证集 Weighted-F1: **{float(best['val_weighted_f1']):.6f}**",
        f"- 测试集访问次数: **{payload['test_access_count']}**",
        f"- 动态样例补位次数: **{int(payload.get('summary', {}).get('selection_fallback_count', 0))}**",
        "",
    ]
    if initial_test is not None:
        initial_metrics = initial_test["metrics"]
        lines.extend([
            "## 冻结初始 Prompt（p0）源测试结果",
            "",
            f"- Weighted-F1: **{float(initial_metrics['weighted_f1']):.6f}**",
            f"- Macro-F1: **{float(initial_metrics['macro_f1']):.6f}**",
            f"- Accuracy: **{float(initial_metrics['accuracy']):.6f}**",
            f"- Balanced Accuracy: **{float(initial_metrics.get('balanced_accuracy', 0.0)):.6f}**",
            f"- MCC: **{float(initial_metrics.get('matthews_correlation', 0.0)):.6f}**",
            f"- INVALID: **{int(initial_metrics['invalid_count'])}**",
            "",
        ])
    lines.extend([
        "## 混淆计数",
        "",
    ])
    for gold in labels:
        for predicted in labels + ["__INVALID__"]:
            lines.append(f"- {gold} → {predicted}: {confusion[(gold, predicted)]}")
    lines.extend([
        "", "## 资源与溯源", "",
        f"- 模型调用次数: {usage.get('calls', 'unknown')}",
        f"- 输入 token: {usage.get('input_tokens', 'unknown')}",
        f"- 输出 token: {usage.get('output_tokens', 'unknown')}",
        f"- 模型: `{payload['provenance'].get('model', 'unknown')}`",
        f"- 数据集 SHA-256: `{payload['provenance'].get('dataset_sha256', 'unknown')}`",
        f"- 代码提交: `{payload['provenance'].get('code_git_revision', 'unknown')}`",
        "", "## 文件说明", "",
        "- `result-full.json`：完整逐调用结果与协议状态。",
        "- `trajectory.csv`：逐轮分数与轨迹。",
        "- `candidates.jsonl`：p0 与全部候选 Prompt、父节点、提议原文和优化证据。",
        "- `validation-predictions.jsonl`：每个候选的验证集逐样本投票和原始输出。",
        "- `source-test-predictions.csv`：p0 与 p-star 的源测试逐样本投票和原始输出。",
        "- `best-prompt.txt`：验证集选出的最佳定义。",
        "- `environment.json`：模型、运行时、配置和协议指纹。",
        "- `split-manifest.json`：本 cell 的精确划分 ID。",
        "- `run.log`：完整运行日志。",
    ])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    trajectory_fields = [
        "iteration", "validation_weighted_f1", "validation_macro_f1",
        "validation_accuracy", "validation_balanced_accuracy", "validation_mcc",
        "validation_invalid_count", "candidate_demo_ids",
        "current_before_iteration", "current_after_iteration", "improved_or_tied_best",
        "backtracked", "selection_fallback_count", "proposal_parse_valid",
    ]
    with (out / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_fields)
        writer.writeheader()
        init_state = payload["initialization"]["state"]
        init_metrics = payload["initialization"]["validation"]["metrics"]
        writer.writerow({
            "iteration": 0,
            "validation_weighted_f1": init_metrics["weighted_f1"],
            "validation_macro_f1": init_metrics["macro_f1"],
            "validation_accuracy": init_metrics["accuracy"],
            "validation_balanced_accuracy": init_metrics.get("balanced_accuracy"),
            "validation_mcc": init_metrics.get("matthews_correlation"),
            "validation_invalid_count": init_metrics["invalid_count"],
            "candidate_demo_ids": json.dumps(init_state["demo_ids"]),
            "current_after_iteration": 0,
            "improved_or_tied_best": True,
            "backtracked": False,
            "selection_fallback_count": 0,
            "proposal_parse_valid": True,
        })
        for item in payload["iterations"]:
            candidate = item["candidate"]
            row_metrics = item["validation"]["metrics"]
            writer.writerow({
                "iteration": item["iteration"],
                "validation_weighted_f1": row_metrics["weighted_f1"],
                "validation_macro_f1": row_metrics["macro_f1"],
                "validation_accuracy": row_metrics["accuracy"],
                "validation_balanced_accuracy": row_metrics.get("balanced_accuracy"),
                "validation_mcc": row_metrics.get("matthews_correlation"),
                "validation_invalid_count": row_metrics["invalid_count"],
                "candidate_demo_ids": json.dumps(candidate["demo_ids"]),
                "current_before_iteration": item["current_before"]["iteration"],
                "current_after_iteration": item["current_after"]["iteration"],
                "improved_or_tied_best": item.get("improved_or_tied_best", ""),
                "backtracked": item.get("backtrack") is not None,
                "selection_fallback_count": len(item.get("selection_fallbacks", [])),
                "proposal_parse_valid": item["proposal"]["parse_valid"],
            })

    prediction_fields = [
        "artifact", "id", "gold", "prediction", "vote_1", "vote_2", "vote_3",
        "raw_output_1", "raw_output_2", "raw_output_3",
    ]
    with (out / "source-test-predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for artifact, test_result in source_tests.items():
            for item in test_result["predictions"]:
                votes = list(item.get("votes", [])) + [""] * 3
                outputs = list(item.get("raw_outputs", [])) + [""] * 3
                writer.writerow({
                    "artifact": artifact,
                    "id": item["id"], "gold": item["gold"], "prediction": item["prediction"],
                    "vote_1": votes[0], "vote_2": votes[1], "vote_3": votes[2],
                    "raw_output_1": outputs[0], "raw_output_2": outputs[1],
                    "raw_output_3": outputs[2],
                })

    with (out / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        initialization = payload["initialization"]
        handle.write(json.dumps({
            "iteration": 0,
            "candidate": initialization["state"],
            "parent_iteration": None,
            "proposal": None,
            "optimization_evidence": initialization.get("example_batch"),
            "validation_metrics": initialization["validation"]["metrics"],
        }, ensure_ascii=False) + "\n")
        for item in payload["iterations"]:
            evidence = item.get("example_batch_used", item.get("error_batch"))
            handle.write(json.dumps({
                "iteration": item["iteration"],
                "candidate": item["candidate"],
                "parent_iteration": item["current_before"]["iteration"],
                "current_after_iteration": item["current_after"]["iteration"],
                "proposal": item["proposal"],
                "optimization_evidence": evidence,
                "backtrack": item.get("backtrack"),
                "validation_metrics": item["validation"]["metrics"],
            }, ensure_ascii=False) + "\n")

    with (out / "validation-predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in payload["initialization"]["validation"]["predictions"]:
            handle.write(json.dumps({"iteration": 0, **item}, ensure_ascii=False) + "\n")
        for iteration in payload["iterations"]:
            for item in iteration["validation"]["predictions"]:
                handle.write(json.dumps({
                    "iteration": iteration["iteration"], **item
                }, ensure_ascii=False) + "\n")

    (out / "best-prompt.txt").write_text(
        f"Method: {payload['method']}\n"
        f"Best iteration: {best['iteration']}\n"
        f"Validation weighted-F1: {float(best['val_weighted_f1']):.12f}\n"
        f"Demonstration IDs: {best['demo_ids']}\n\n{best['definitions'].strip()}\n",
        encoding="utf-8",
    )
    (out / "environment.json").write_text(
        json.dumps({
            "provenance": payload["provenance"],
            "config": payload["config"],
            "protocol_interpretation": payload.get("protocol_interpretation"),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "split-manifest.json").write_text(
        json.dumps(payload["split"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(result_path, out / "result-full.json")
    shutil.copy2(log_path, out / "run.log")
    missing = [name for name in STANDARD_FILES if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete export: {missing}")
    print(json.dumps({"result_archive": str(out), "files": list(STANDARD_FILES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
