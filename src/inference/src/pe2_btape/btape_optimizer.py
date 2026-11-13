from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Example, Split
from .optimizer import ChatBackend, classify, request_evidence, summarize_usage
from .prompts import btape_meta_prompt, parse_btape_definitions


@dataclass(frozen=True)
class BTAPEConfig:
    seed: int = 42
    n_max: int = 20
    backtrack_threshold: int = 3
    voting_runs: int = 3
    positive_label: str = "Security"
    classification_num_predict: int = 64
    proposal_num_predict: int = 1536
    classification_interface: str = "appendix_f"


@dataclass(frozen=True)
class BTAPEState:
    iteration: int
    definitions: str
    demo_ids: list[int]
    val_weighted_f1: float
    val_metrics: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def rank_states(states: list[BTAPEState]) -> list[BTAPEState]:
    """Rank by validation weighted-F1; use the latest candidate for exact ties."""
    return sorted(states, key=lambda state: (state.val_weighted_f1, state.iteration), reverse=True)


def select_backtrack_state(
    states: list[BTAPEState], pointer: int
) -> tuple[BTAPEState, int, int]:
    """Apply the published package's zero-based interpretation of ``ptr``.

    Algorithm 1 does not define the pointer base and its increment/decrement
    direction is internally ambiguous.  We retain the replication package's
    zero-based arithmetic, while correcting its stale-definition bug.
    """
    ranked = rank_states(states)
    used_index = min(max(pointer, 0), len(ranked) - 1)
    return ranked[used_index], used_index, pointer - 1


def _state_from_dict(data: dict[str, object]) -> BTAPEState:
    return BTAPEState(
        iteration=int(data["iteration"]),
        definitions=str(data["definitions"]),
        demo_ids=[int(value) for value in data["demo_ids"]],  # type: ignore[arg-type]
        val_weighted_f1=float(data["val_weighted_f1"]),
        val_metrics=dict(data["val_metrics"]),  # type: ignore[arg-type]
    )


def select_initial_feedback_examples(
    pool: list[Example], positive_label: str, seed: int
) -> list[Example]:
    """Package-compatible 2-positive/2-negative selection with explicit semantics."""
    positives = [example for example in pool if example.label == positive_label]
    negatives = [example for example in pool if example.label != positive_label]
    if len(positives) < 2 or len(negatives) < 2:
        raise ValueError("Initial feedback requires at least two examples per class")
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:2] + negatives[:2]


class EmptyQuadrantError(RuntimeError):
    def __init__(self, missing: list[str], bucket_sizes: dict[str, int]) -> None:
        self.missing = missing
        self.bucket_sizes = bucket_sizes
        super().__init__(f"Algorithm 1 feedback quadrant(s) empty: {', '.join(missing)}")


def select_balanced_dynamic_examples(
    pool: list[Example],
    evaluation: dict[str, object],
    positive_label: str,
    seed: int,
) -> list[dict[str, object]]:
    """Select the Algorithm-1 batch with the published package's fallback.

    Algorithm 1 requests one item from each correctness/class quadrant but does
    not define what happens when a quadrant is empty.  The published v2 package
    keeps one item from every non-empty quadrant, then deterministically shuffles
    the remaining pool and tops the batch up to four.  We reproduce that behavior
    while recording which requested quadrant required a backfill.
    """
    prediction_by_id = {
        int(row["id"]): str(row["prediction"])
        for row in evaluation["predictions"]  # type: ignore[index]
    }
    buckets: dict[tuple[bool, bool], list[dict[str, object]]] = {
        (is_positive, is_correct): []
        for is_positive in (True, False)
        for is_correct in (True, False)
    }
    rows_in_pool_order: list[dict[str, object]] = []
    quadrant_names = {
        (True, True): "correct_positive",
        (False, True): "correct_negative",
        (True, False): "misclassified_positive",
        (False, False): "misclassified_negative",
    }
    for example in pool:
        prediction = prediction_by_id[example.id]
        key = (example.label == positive_label, prediction == example.label)
        row: dict[str, object] = {
            "id": example.id,
            "text": example.text,
            "label": example.label,
            "prediction": prediction,
            "actual_quadrant": quadrant_names[key],
        }
        buckets[key].append(row)
        rows_in_pool_order.append(row)

    requested = [
        (True, True, "correct_positive"),
        (False, True, "correct_negative"),
        (True, False, "misclassified_positive"),
        (False, False, "misclassified_negative"),
    ]
    bucket_sizes = {
        name: len(buckets[(is_positive, is_correct)])
        for is_positive, is_correct, name in requested
    }
    rng = random.Random(seed)
    selected: list[dict[str, object]] = []
    for is_positive, is_correct, name in requested:
        candidates = list(buckets[(is_positive, is_correct)])
        rng.shuffle(candidates)
        if candidates:
            chosen = dict(candidates[0])
            chosen["requested_quadrant"] = name
            chosen["actual_quadrant"] = name
            chosen["selection_source"] = "exact_quadrant"
            selected.append(chosen)

    missing = [name for name, size in bucket_sizes.items() if size == 0]
    if missing:
        selected_ids = {int(row["id"]) for row in selected}
        remaining = [
            dict(row)
            for row in rows_in_pool_order
            if int(row["id"]) not in selected_ids
        ]
        rng.shuffle(remaining)
        if len(remaining) < len(missing):
            raise EmptyQuadrantError(missing, bucket_sizes)
        for requested_name, candidate in zip(missing, remaining):
            chosen = dict(candidate)
            chosen["requested_quadrant"] = requested_name
            chosen["selection_source"] = "package_random_backfill"
            selected.append(chosen)

    if len(selected) != 4 or len({int(row["id"]) for row in selected}) != 4:
        raise RuntimeError("BT-APE feedback selection must yield four unique examples")
    return selected


def _aggregate_usage(payload: dict[str, object]) -> dict[str, object]:
    buckets: list[dict[str, object]] = []
    initialization = payload.get("initialization")
    if isinstance(initialization, dict):
        for key in ("validation", "feedback_classification"):
            value = initialization.get(key)
            if isinstance(value, dict) and isinstance(value.get("usage"), dict):
                buckets.append(value["usage"])
    iterations = payload.get("iterations", [])
    if isinstance(iterations, list):
        for iteration in iterations:
            if not isinstance(iteration, dict):
                continue
            for key in ("proposal", "validation", "pool_evaluation"):
                value = iteration.get(key)
                if isinstance(value, dict) and isinstance(value.get("usage"), dict):
                    buckets.append(value["usage"])
    source_tests = payload.get("source_tests", {})
    if isinstance(source_tests, dict):
        for test_result in source_tests.values():
            if isinstance(test_result, dict) and isinstance(test_result.get("usage"), dict):
                buckets.append(test_result["usage"])
    elif isinstance(payload.get("final_test"), dict):
        final_test = payload["final_test"]
        if isinstance(final_test.get("usage"), dict):
            buckets.append(final_test["usage"])
    return {
        "calls": sum(int(bucket.get("calls", 0)) for bucket in buckets),
        "input_tokens": sum(int(bucket.get("input_tokens", 0)) for bucket in buckets),
        "output_tokens": sum(int(bucket.get("output_tokens", 0)) for bucket in buckets),
        "duration_ns": sum(int(bucket.get("duration_ns", 0)) for bucket in buckets),
    }


def run_btape(
    backend: ChatBackend,
    split: Split,
    labels: list[str],
    initial_definitions: str,
    config: BTAPEConfig,
    checkpoint_path: Path,
    provenance: dict[str, object],
    resume: bool = False,
) -> dict[str, object]:
    if config.voting_runs != 3:
        raise ValueError("The paper-matched protocol requires exactly 3 voting runs")
    if config.positive_label not in labels or len(labels) != 2:
        raise ValueError("BT-APE SecReq reproduction requires two labels and explicit positive class")
    started_at = time.time()

    if resume and checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("provenance") != provenance:
            raise ValueError("Checkpoint provenance differs from the requested run")
        if payload.get("status") == "complete":
            summary = payload["summary"]
            print("BTAPE_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
            return payload
        if payload.get("status") == "protocol_violation":
            raise RuntimeError(
                "Checkpoint records an unresolved protocol violation; do not resume it"
            )
        states = [_state_from_dict(item) for item in payload["states"]]
        best = _state_from_dict(payload["best_state"])
        current = _state_from_dict(payload["current_state"])
        feedback = list(payload["next_example_batch"])
        failure_count = int(payload["failure_count"])
        pointer = int(payload["pointer"])
        print(f"[resume] continuing after iteration {payload['completed_iteration']}", flush=True)
    else:
        initial_feedback_examples = select_initial_feedback_examples(
            split.pool, config.positive_label, config.seed
        )
        print("[initialization] evaluating p0 on validation", flush=True)
        validation = classify(
            backend, initial_definitions, [], split.val, labels,
            config.voting_runs, config.classification_num_predict, "val:init",
            config.classification_interface,
        )
        metrics = validation["metrics"]
        initial = BTAPEState(
            iteration=0,
            definitions=initial_definitions,
            demo_ids=[],
            val_weighted_f1=float(metrics["weighted_f1"]),  # type: ignore[index]
            val_metrics=dict(metrics),  # type: ignore[arg-type]
        )
        initial_feedback_eval = classify(
            backend, initial.definitions, [], initial_feedback_examples, labels,
            config.voting_runs, config.classification_num_predict, "initial-feedback",
            config.classification_interface,
        )
        by_id = {
            int(row["id"]): row
            for row in initial_feedback_eval["predictions"]  # type: ignore[index]
        }
        feedback = [
            {
                "id": example.id,
                "text": example.text,
                "label": example.label,
                "prediction": by_id[example.id]["prediction"],
                "requested_quadrant": "initial_2_positive_2_negative",
                "selection_source": "random_initialization",
            }
            for example in initial_feedback_examples
        ]
        states = [initial]
        best = initial
        current = initial
        failure_count = 0
        pointer = 1
        payload = {
            "schema_version": 1,
            "method": "BT-APE formal source-domain optimization",
            "status": "running",
            "provenance": provenance,
            "config": asdict(config),
            "protocol_interpretation": {
                "objective": "validation weighted-F1",
                "positive_class": config.positive_label,
                "classification": (
                    f"{config.classification_interface}; dynamic examples condition proposer only"
                ),
                "pointer_indexing": "zero-based, matching the published replication package; paper is ambiguous",
                "empty_quadrant_policy": (
                    "published-v2-package deterministic random backfill from the "
                    "remaining pool; every fallback is retained in the trajectory"
                ),
                "output_contract": "guided binary choice; any violation fails the cell",
                "source_test_policy": "evaluate frozen p0 and validation-selected p_star only after selection",
            },
            "split": split.manifest(),
            "initialization": {
                "state": initial.as_dict(),
                "validation": validation,
                "feedback_classification": initial_feedback_eval,
                "example_batch": feedback,
            },
            "states": [initial.as_dict()],
            "iterations": [],
            "completed_iteration": 0,
            "best_state": best.as_dict(),
            "current_state": current.as_dict(),
            "next_example_batch": feedback,
            "failure_count": failure_count,
            "pointer": pointer,
            "test_access_count": 0,
            "source_tests": {},
        }
        payload["usage"] = _aggregate_usage(payload)
        atomic_json(checkpoint_path, payload)

    start_iteration = int(payload["completed_iteration"]) + 1
    for iteration in range(start_iteration, config.n_max + 1):
        print(
            f"[iteration {iteration}/{config.n_max}] current=p{current.iteration} "
            f"best=p{best.iteration} best_wF1={best.val_weighted_f1:.6f} "
            f"failures={failure_count} ptr={pointer}", flush=True,
        )
        proposer_system, proposer_user = btape_meta_prompt(
            current.definitions, feedback, labels
        )
        raw_proposal, proposal_usage = backend.chat(
            proposer_system, proposer_user, kind="proposal",
            num_predict=config.proposal_num_predict,
        )
        candidate_definitions = parse_btape_definitions(raw_proposal)
        parse_valid = candidate_definitions is not None
        if candidate_definitions is None:
            candidate_definitions = current.definitions
        validation = classify(
            backend, candidate_definitions, [], split.val, labels,
            config.voting_runs, config.classification_num_predict, f"val:{iteration}",
            config.classification_interface,
        )
        metrics = validation["metrics"]
        candidate = BTAPEState(
            iteration=iteration,
            definitions=candidate_definitions,
            demo_ids=[],
            val_weighted_f1=float(metrics["weighted_f1"]),  # type: ignore[index]
            val_metrics=dict(metrics),  # type: ignore[arg-type]
        )
        states.append(candidate)
        improved = candidate.val_weighted_f1 >= best.val_weighted_f1
        if improved:
            best = candidate
            current = candidate
            failure_count = 0
            pointer += 1
        else:
            failure_count += 1

        backtrack: dict[str, object] | None = None
        if failure_count == config.backtrack_threshold:
            previous = current
            current, position, next_pointer = select_backtrack_state(states, pointer)
            backtrack = {
                "from_iteration": previous.iteration,
                "to_iteration": current.iteration,
                "requested_index_zero_based": pointer,
                "used_index_zero_based": position,
            }
            pointer = next_pointer
            failure_count = 0

        pool_evaluation = classify(
            backend, current.definitions, [], split.pool,
            labels, config.voting_runs, config.classification_num_predict,
            f"pool:{iteration}",
            config.classification_interface,
        )
        example_seed = config.seed + iteration
        next_feedback = select_balanced_dynamic_examples(
            split.pool, pool_evaluation, config.positive_label, example_seed
        )
        selection_fallbacks = [
            {
                "id": int(row["id"]),
                "requested_quadrant": row["requested_quadrant"],
                "actual_quadrant": row["actual_quadrant"],
                "selection_source": row["selection_source"],
            }
            for row in next_feedback
            if row["selection_source"] != "exact_quadrant"
        ]
        record = {
            "iteration": iteration,
            "current_before": payload["current_state"],
            "example_batch_used": feedback,
            "proposal": {
                "meta_prompt_system": proposer_system,
                "meta_prompt_user": proposer_user,
                "request": request_evidence(
                    proposer_system, proposer_user, proposal_usage
                ),
                "raw_response": raw_proposal,
                "parse_valid": parse_valid,
                "parsed_definitions": candidate_definitions,
                "usage": summarize_usage([proposal_usage]),
            },
            "candidate": candidate.as_dict(),
            "validation": validation,
            "improved_or_tied_best": improved,
            "backtrack": backtrack,
            "current_after": current.as_dict(),
            "failure_count_after": failure_count,
            "pointer_after": pointer,
            "pool_evaluation": pool_evaluation,
            "next_example_seed": example_seed,
            "next_example_batch": next_feedback,
            "selection_fallbacks": selection_fallbacks,
            "protocol_violation": None,
        }
        feedback = next_feedback
        payload["iterations"].append(record)  # type: ignore[union-attr]
        payload["states"] = [state.as_dict() for state in states]
        payload["completed_iteration"] = iteration
        payload["best_state"] = best.as_dict()
        payload["current_state"] = current.as_dict()
        payload["next_example_batch"] = feedback
        payload["failure_count"] = failure_count
        payload["pointer"] = pointer
        payload["usage"] = _aggregate_usage(payload)
        atomic_json(checkpoint_path, payload)
        if selection_fallbacks:
            print(
                "BTAPE_SELECTION_FALLBACK "
                + json.dumps(
                    {
                        "iteration": iteration,
                        "seed": example_seed,
                        "fallbacks": selection_fallbacks,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(
            f"[iteration {iteration}/{config.n_max}] candidate_wF1="
            f"{candidate.val_weighted_f1:.6f} improved={improved} "
            f"next=p{current.iteration} backtrack={backtrack is not None}", flush=True,
        )

    source_tests = payload.setdefault("source_tests", {})
    if "p0" not in source_tests:
        print("[final] held-out source test evaluation of frozen p0", flush=True)
        source_tests["p0"] = classify(
            backend, states[0].definitions, [], split.test, labels,
            config.voting_runs, config.classification_num_predict, "test:p0",
            config.classification_interface,
        )
        payload["test_access_count"] = 1
        payload["usage"] = _aggregate_usage(payload)
        atomic_json(checkpoint_path, payload)
    if "p_star" not in source_tests:
        print(
            f"[final] held-out source test evaluation of p_star=p{best.iteration}",
            flush=True,
        )
        source_tests["p_star"] = classify(
            backend, best.definitions, [], split.test, labels,
            config.voting_runs, config.classification_num_predict, "test:p_star",
            config.classification_interface,
        )
        payload["test_access_count"] = 2
        payload["usage"] = _aggregate_usage(payload)
        atomic_json(checkpoint_path, payload)
    final_test = source_tests["p_star"]
    payload["best_state"] = best.as_dict()
    # Retain the reproduction-era key for compatible result readers.
    payload["final_test"] = final_test
    payload["usage"] = _aggregate_usage(payload)
    payload["status"] = "complete"
    payload["elapsed_seconds_this_process"] = time.time() - started_at
    summary = {
        "status": "complete",
        "best_iteration": best.iteration,
        "best_validation_weighted_f1": best.val_weighted_f1,
        "test_weighted_f1": final_test["metrics"]["weighted_f1"],  # type: ignore[index]
        "test_macro_f1": final_test["metrics"]["macro_f1"],  # type: ignore[index]
        "test_accuracy": final_test["metrics"]["accuracy"],  # type: ignore[index]
        "test_invalid_count": final_test["metrics"]["invalid_count"],  # type: ignore[index]
        "p0_test_weighted_f1": source_tests["p0"]["metrics"]["weighted_f1"],  # type: ignore[index]
        "candidate_count": len(states),
        "backtrack_count": sum(
            item["backtrack"] is not None for item in payload["iterations"]  # type: ignore[index]
        ),
        "selection_fallback_count": sum(
            len(item.get("selection_fallbacks", []))
            for item in payload["iterations"]  # type: ignore[index]
        ),
        "selection_fallback_iterations": [
            int(item["iteration"])
            for item in payload["iterations"]  # type: ignore[index]
            if item.get("selection_fallbacks")
        ],
        "protocol_violation_count": 0,
        "test_access_count": 2,
        "usage": payload["usage"],
        "result_file": str(checkpoint_path),
    }
    payload["summary"] = summary
    atomic_json(checkpoint_path, payload)
    print("BTAPE_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return payload
