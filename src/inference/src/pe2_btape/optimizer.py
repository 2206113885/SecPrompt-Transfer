from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .backend import ChatRequest
from .data import Example, Split
from .metrics import INVALID, Metrics, compute_metrics
from .prompts import (
    classification_messages,
    extract_label,
    majority,
    parse_revised_prompt,
    pe2_meta_prompt,
)


class ChatBackend(Protocol):
    def chat(
        self,
        system: str,
        user: str,
        *,
        kind: str,
        num_predict: int,
    ) -> tuple[str, dict[str, object]]: ...


def request_evidence(
    system: str, user: str, usage: dict[str, object]
) -> dict[str, object]:
    """Persist the exact logical request and HF rendering when available."""
    evidence: dict[str, object] = {"system": system, "user": user}
    for key in (
        "backend",
        "hf_rendered_prompt",
        "hf_rendered_prompt_sha256",
        "prompt_eval_count",
    ):
        if key in usage:
            evidence[key] = usage[key]
    return evidence


def vote_evidence(usage: dict[str, object], vote_index: int) -> dict[str, object]:
    """Keep completion-level provenance without duplicating each input prompt."""
    evidence: dict[str, object] = {"vote_index": vote_index}
    for key in (
        "backend",
        "batch_id",
        "batch_size",
        "batch_elapsed_ns",
        "finish_reason",
        "output_token_ids",
        "prompt_eval_count",
        "eval_count",
        "total_duration_ns",
    ):
        if key in usage:
            evidence[key] = usage[key]
    return evidence


@dataclass(frozen=True)
class PE2Config:
    seed: int = 42
    n_max: int = 20
    history_size: int = 3
    error_batch_size: int = 4
    voting_runs: int = 3
    positive_label: str = "Security"
    classification_num_predict: int = 64
    proposal_num_predict: int = 1536
    classification_interface: str = "appendix_f"


@dataclass(frozen=True)
class PromptState:
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
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _lookup(split: Split) -> dict[int, Example]:
    return {example.id: example for part in (split.pool, split.val, split.test) for example in part}


def _demos(state: PromptState, examples_by_id: dict[int, Example]) -> list[Example]:
    return [examples_by_id[example_id] for example_id in state.demo_ids]


def rank_states(states: list[PromptState]) -> list[PromptState]:
    """Highest validation score first; a newer prompt wins an exact tie."""
    return sorted(states, key=lambda state: (state.val_weighted_f1, state.iteration), reverse=True)


def sample_errors(
    pool: list[Example],
    evaluation: dict[str, object],
    k: int,
    seed: int,
) -> list[dict[str, object]]:
    by_id = {int(row["id"]): row for row in evaluation["predictions"]}  # type: ignore[index]
    errors = [
        {
            "id": example.id,
            "text": example.text,
            "label": example.label,
            "prediction": by_id[example.id]["prediction"],
        }
        for example in pool
        if by_id[example.id]["prediction"] != example.label
    ]
    random.Random(seed).shuffle(errors)
    return errors[: min(k, len(errors))]


def classify(
    backend: ChatBackend,
    definitions: str,
    demos: list[Example],
    subset: list[Example],
    labels: list[str],
    voting_runs: int,
    num_predict: int,
    split_name: str,
    classification_interface: str = "appendix_f",
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    usage_records: list[dict[str, object]] = []

    requests_by_example: list[tuple[Example, str, str]] = [
        (
            example,
            *classification_messages(
                definitions,
                demos,
                example.text,
                labels,
                classification_interface,
            ),
        )
        for example in subset
    ]
    batched_chat = getattr(backend, "chat_many", None)
    batched_responses: list[tuple[str, dict[str, object]]] | None = None
    if callable(batched_chat):
        requests = [
            ChatRequest(
                system=system,
                user=user,
                kind=f"classification:{split_name}",
                num_predict=num_predict,
            )
            for _example, system, user in requests_by_example
            for _vote_index in range(voting_runs)
        ]
        batched_responses = batched_chat(requests)
        if len(batched_responses) != len(requests):
            raise RuntimeError(
                f"Batch backend returned {len(batched_responses)} responses for "
                f"{len(requests)} classification requests"
            )

    response_offset = 0
    for offset, (example, system, user) in enumerate(requests_by_example, start=1):
        responses: list[tuple[str, dict[str, object]]] = []
        if batched_responses is not None:
            responses = batched_responses[response_offset : response_offset + voting_runs]
            response_offset += voting_runs
        else:
            for _ in range(voting_runs):
                responses.append(
                    backend.chat(
                        system,
                        user,
                        kind=f"classification:{split_name}",
                        num_predict=num_predict,
                    )
                )
        votes: list[str] = []
        raw_outputs: list[str] = []
        vote_metadata: list[dict[str, object]] = []
        for vote_index, (response, usage) in enumerate(responses, start=1):
            raw_outputs.append(response)
            votes.append(extract_label(response, labels))
            usage_records.append(usage)
            vote_metadata.append(vote_evidence(usage, vote_index))
        prediction = majority(votes)
        if getattr(backend, "enforces_binary_choice", False) and (
            INVALID in votes or prediction == INVALID
        ):
            failure = {
                "split": split_name,
                "id": example.id,
                "gold": example.label,
                "votes": votes,
                "raw_outputs": raw_outputs,
                "request": request_evidence(system, user, responses[0][1]),
                "vote_metadata": vote_metadata,
            }
            print(
                "CLASSIFICATION_CONTRACT_FAILURE "
                + json.dumps(failure, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            raise RuntimeError(
                f"Guided binary classification failed for {split_name} id={example.id}"
            )
        rows.append(
            {
                "id": example.id,
                "gold": example.label,
                "prediction": prediction,
                "votes": votes,
                "raw_outputs": raw_outputs,
                "request": request_evidence(system, user, responses[0][1]),
                "vote_metadata": vote_metadata,
            }
        )
        if offset == 1 or offset % 25 == 0 or offset == len(subset):
            print(f"[{split_name}] {offset}/{len(subset)}", flush=True)

    metrics = compute_metrics(
        [str(row["prediction"]) for row in rows],
        [str(row["gold"]) for row in rows],
        labels,
    )
    return {
        "split": split_name,
        "size": len(subset),
        "metrics": metrics.as_dict(),
        "predictions": rows,
        "usage": summarize_usage(usage_records),
    }


def summarize_usage(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "calls": len(records),
        "input_tokens": sum(int(record.get("prompt_eval_count", 0)) for record in records),
        "output_tokens": sum(int(record.get("eval_count", 0)) for record in records),
        "duration_ns": sum(int(record.get("total_duration_ns", 0)) for record in records),
    }


def aggregate_usage(payload: dict[str, object]) -> dict[str, object]:
    buckets: list[dict[str, object]] = []
    initialization = payload.get("initialization")
    if isinstance(initialization, dict) and isinstance(initialization.get("validation"), dict):
        buckets.append(initialization["validation"]["usage"])  # type: ignore[index]
    for iteration in payload.get("iterations", []):  # type: ignore[union-attr]
        if not isinstance(iteration, dict):
            continue
        for key in ("pool_evaluation", "proposal", "validation"):
            value = iteration.get(key)
            if not isinstance(value, dict):
                continue
            usage = value.get("usage")
            if isinstance(usage, dict):
                buckets.append(usage)
    source_tests = payload.get("source_tests", {})
    if isinstance(source_tests, dict):
        for test_result in source_tests.values():
            if isinstance(test_result, dict) and isinstance(test_result.get("usage"), dict):
                buckets.append(test_result["usage"])
    elif isinstance(payload.get("final_test"), dict):
        # Backward compatibility for the earlier reproduction schema.
        final_test = payload["final_test"]
        if isinstance(final_test.get("usage"), dict):
            buckets.append(final_test["usage"])
    return {
        "calls": sum(int(bucket.get("calls", 0)) for bucket in buckets),
        "input_tokens": sum(int(bucket.get("input_tokens", 0)) for bucket in buckets),
        "output_tokens": sum(int(bucket.get("output_tokens", 0)) for bucket in buckets),
        "duration_ns": sum(int(bucket.get("duration_ns", 0)) for bucket in buckets),
    }


def _state_from_dict(data: dict[str, object]) -> PromptState:
    return PromptState(
        iteration=int(data["iteration"]),
        definitions=str(data["definitions"]),
        demo_ids=[int(value) for value in data["demo_ids"]],  # type: ignore[arg-type]
        val_weighted_f1=float(data["val_weighted_f1"]),
        val_metrics=dict(data["val_metrics"]),  # type: ignore[arg-type]
    )


def run_pe2(
    backend: ChatBackend,
    split: Split,
    labels: list[str],
    initial_definitions: str,
    config: PE2Config,
    checkpoint_path: Path,
    provenance: dict[str, object],
    resume: bool = False,
) -> dict[str, object]:
    if config.voting_runs != 3:
        raise ValueError("The paper-matched protocol requires exactly 3 voting runs")
    started_at = time.time()

    if resume and checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("provenance") != provenance:
            raise ValueError("Checkpoint provenance differs from the requested run")
        if payload.get("status") == "complete":
            final_test = payload["source_tests"]["p_star"]
            initial_test = payload["source_tests"]["p0"]
            best = _state_from_dict(payload["best_state"])
            summary = {
                "status": "complete",
                "best_iteration": best.iteration,
                "best_validation_weighted_f1": best.val_weighted_f1,
                "test_weighted_f1": final_test["metrics"]["weighted_f1"],
                "test_macro_f1": final_test["metrics"]["macro_f1"],
                "test_accuracy": final_test["metrics"]["accuracy"],
                "test_invalid_count": final_test["metrics"]["invalid_count"],
                "p0_test_weighted_f1": initial_test["metrics"]["weighted_f1"],
                "candidate_count": len(payload["states"]),
                "test_access_count": payload["test_access_count"],
                "usage": payload["usage"],
                "result_file": str(checkpoint_path),
                "note": "completed checkpoint reused; no model call was made",
            }
            print("PE2_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
            return payload
        states = [_state_from_dict(item) for item in payload["states"]]
        print(
            f"[resume] continuing after iteration {payload['completed_iteration']}",
            flush=True,
        )
    else:
        print("[initialization] evaluating p0 on validation", flush=True)
        initial_evaluation = classify(
            backend,
            initial_definitions,
            [],
            split.val,
            labels,
            config.voting_runs,
            config.classification_num_predict,
            "val:init",
            config.classification_interface,
        )
        initial_metrics = initial_evaluation["metrics"]
        initial_state = PromptState(
            iteration=0,
            definitions=initial_definitions,
            demo_ids=[],
            val_weighted_f1=float(initial_metrics["weighted_f1"]),  # type: ignore[index]
            val_metrics=dict(initial_metrics),  # type: ignore[arg-type]
        )
        states = [initial_state]
        payload = {
            "schema_version": 1,
            "method": "PE2 baseline protocol-matched to BT-APE Algorithm 2",
            "status": "running",
            "provenance": provenance,
            "config": asdict(config),
            "protocol_interpretation": {
                "classification": (
                    f"{config.classification_interface}; pool errors condition proposer only"
                ),
                "objective": "validation weighted-F1",
                "output_contract": "guided binary choice; any violation fails the cell",
                "source_test_policy": "evaluate frozen p0 and validation-selected p_star only after selection",
            },
            "split": split.manifest(),
            "initialization": {
                "state": initial_state.as_dict(),
                "validation": initial_evaluation,
            },
            "states": [initial_state.as_dict()],
            "iterations": [],
            "completed_iteration": 0,
            "test_access_count": 0,
            "source_tests": {},
        }
        payload["usage"] = aggregate_usage(payload)
        atomic_json(checkpoint_path, payload)

    start_iteration = int(payload["completed_iteration"]) + 1
    for iteration in range(start_iteration, config.n_max + 1):
        current = rank_states(states)[0]
        print(
            f"[iteration {iteration}/{config.n_max}] current=p{current.iteration} "
            f"val_wF1={current.val_weighted_f1:.6f}",
            flush=True,
        )
        pool_evaluation = classify(
            backend,
            current.definitions,
            [],
            split.pool,
            labels,
            config.voting_runs,
            config.classification_num_predict,
            f"pool:{iteration}",
            config.classification_interface,
        )
        error_seed = config.seed + iteration
        errors = sample_errors(
            split.pool,
            pool_evaluation,
            config.error_batch_size,
            error_seed,
        )
        top_history = rank_states(states)[: config.history_size]
        meta_history = [state.as_dict() for state in top_history]
        proposal_system, proposal_user = pe2_meta_prompt(
            current.definitions,
            meta_history,
            errors,
            labels,
            config.classification_interface,
        )
        raw_proposal, proposal_usage = backend.chat(
            proposal_system,
            proposal_user,
            kind="proposal",
            num_predict=config.proposal_num_predict,
        )
        candidate_definitions = parse_revised_prompt(raw_proposal)
        parse_valid = candidate_definitions is not None
        if candidate_definitions is None:
            candidate_definitions = current.definitions

        validation = classify(
            backend,
            candidate_definitions,
            [],
            split.val,
            labels,
            config.voting_runs,
            config.classification_num_predict,
            f"val:{iteration}",
            config.classification_interface,
        )
        metrics = validation["metrics"]
        candidate = PromptState(
            iteration=iteration,
            definitions=candidate_definitions,
            demo_ids=[],
            val_weighted_f1=float(metrics["weighted_f1"]),  # type: ignore[index]
            val_metrics=dict(metrics),  # type: ignore[arg-type]
        )
        states.append(candidate)
        next_current = rank_states(states)[0]
        iteration_record = {
            "iteration": iteration,
            "current_before": current.as_dict(),
            "top_history_shown": meta_history,
            "error_sampling_seed": error_seed,
            "error_batch": errors,
            "pool_evaluation": pool_evaluation,
            "proposal": {
                "meta_prompt_system": proposal_system,
                "meta_prompt_user": proposal_user,
                "request": request_evidence(
                    proposal_system, proposal_user, proposal_usage
                ),
                "raw_response": raw_proposal,
                "parse_valid": parse_valid,
                "parsed_definitions": candidate_definitions,
                "usage": summarize_usage([proposal_usage]),
            },
            "candidate": candidate.as_dict(),
            "validation": validation,
            "current_after": next_current.as_dict(),
        }
        payload["iterations"].append(iteration_record)  # type: ignore[union-attr]
        payload["states"] = [state.as_dict() for state in states]
        payload["completed_iteration"] = iteration
        payload["usage"] = aggregate_usage(payload)
        atomic_json(checkpoint_path, payload)
        print(
            f"[iteration {iteration}/{config.n_max}] candidate val_wF1="
            f"{candidate.val_weighted_f1:.6f}; next=p{next_current.iteration}",
            flush=True,
        )

    best = rank_states(states)[0]
    source_tests = payload.setdefault("source_tests", {})
    if "p0" not in source_tests:
        print("[final] held-out source test evaluation of frozen p0", flush=True)
        source_tests["p0"] = classify(
            backend,
            states[0].definitions,
            [],
            split.test,
            labels,
            config.voting_runs,
            config.classification_num_predict,
            "test:p0",
            config.classification_interface,
        )
        payload["test_access_count"] = 1
        payload["usage"] = aggregate_usage(payload)
        atomic_json(checkpoint_path, payload)
    if "p_star" not in source_tests:
        print(
            f"[final] held-out source test evaluation of p_star=p{best.iteration}",
            flush=True,
        )
        source_tests["p_star"] = classify(
            backend,
            best.definitions,
            [],
            split.test,
            labels,
            config.voting_runs,
            config.classification_num_predict,
            "test:p_star",
            config.classification_interface,
        )
        payload["test_access_count"] = 2
        payload["usage"] = aggregate_usage(payload)
        atomic_json(checkpoint_path, payload)
    final_test = source_tests["p_star"]
    payload["best_state"] = best.as_dict()
    # Retain the old key for readers of the reproduction schema.
    payload["final_test"] = final_test
    payload["usage"] = aggregate_usage(payload)
    payload["status"] = "complete"
    payload["elapsed_seconds_this_process"] = time.time() - started_at
    atomic_json(checkpoint_path, payload)

    summary = {
        "status": payload["status"],
        "best_iteration": best.iteration,
        "best_validation_weighted_f1": best.val_weighted_f1,
        "test_weighted_f1": final_test["metrics"]["weighted_f1"],  # type: ignore[index]
        "test_macro_f1": final_test["metrics"]["macro_f1"],  # type: ignore[index]
        "test_accuracy": final_test["metrics"]["accuracy"],  # type: ignore[index]
        "test_invalid_count": final_test["metrics"]["invalid_count"],  # type: ignore[index]
        "p0_test_weighted_f1": source_tests["p0"]["metrics"]["weighted_f1"],  # type: ignore[index]
        "candidate_count": len(states),
        "test_access_count": payload["test_access_count"],
        "usage": payload["usage"],
        "result_file": str(checkpoint_path),
    }
    print("PE2_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return payload
