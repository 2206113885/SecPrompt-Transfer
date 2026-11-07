from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pe2_btape.btape_optimizer import (
    BTAPEConfig,
    BTAPEState,
    EmptyQuadrantError,
    rank_states,
    run_btape,
    select_backtrack_state,
    select_balanced_dynamic_examples,
)
from pe2_btape.data import Example, Split
from pe2_btape.prompts import btape_meta_prompt


class ScriptedBackend:
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def chat(self, system, user, *, kind, num_predict):
        self.kinds.append(kind)
        if kind == "proposal":
            response = "Security: protects assets.\nNon-security: all other requirements."
        else:
            target = user.rsplit("Requirement:", 1)[-1].casefold()
            response = "Label: Security" if "secure" in target else "Label: Non-security"
        return response, {
            "kind": kind,
            "prompt_eval_count": 10,
            "eval_count": 2,
            "total_duration_ns": 100,
            "load_duration_ns": 0,
        }


class BTAPEProtocolTests(unittest.TestCase):
    def test_rank_uses_weighted_f1_and_latest_tie(self):
        states = [
            BTAPEState(0, "a", [1], 0.8, {"macro_f1": 0.9}),
            BTAPEState(1, "b", [2], 0.8, {"macro_f1": 0.1}),
        ]
        self.assertEqual(rank_states(states)[0].iteration, 1)

    def test_backtrack_pointer_uses_published_package_zero_based_interpretation(self):
        states = [
            BTAPEState(0, "initial", [10, 11], 0.7, {}),
            BTAPEState(1, "best", [20, 21], 0.9, {}),
            BTAPEState(2, "second", [30, 31], 0.8, {}),
        ]
        selected, used_index, next_pointer = select_backtrack_state(states, 1)
        self.assertEqual(used_index, 1)
        self.assertEqual(next_pointer, 0)
        self.assertEqual(selected.definitions, "second")
        self.assertEqual(selected.demo_ids, [30, 31])

    def test_balanced_batch_uses_explicit_security_positive(self):
        pool = [
            Example(0, "secure good", "Security"),
            Example(1, "secure miss", "Security"),
            Example(2, "plain good", "Non-security"),
            Example(3, "plain miss", "Non-security"),
        ]
        evaluation = {
            "predictions": [
                {"id": 0, "prediction": "Security"},
                {"id": 1, "prediction": "Non-security"},
                {"id": 2, "prediction": "Non-security"},
                {"id": 3, "prediction": "Security"},
            ]
        }
        selected = select_balanced_dynamic_examples(
            pool, evaluation, "Security", 42
        )
        self.assertEqual(
            {row["requested_quadrant"] for row in selected},
            {"correct_positive", "correct_negative", "misclassified_positive", "misclassified_negative"},
        )

    def test_empty_feedback_quadrant_uses_package_random_backfill(self):
        pool = [
            Example(0, "secure", "Security"),
            Example(1, "plain", "Non-security"),
            Example(2, "secure two", "Security"),
            Example(3, "plain two", "Non-security"),
        ]
        evaluation = {
            "predictions": [
                {"id": 0, "prediction": "Security"},
                {"id": 1, "prediction": "Non-security"},
                {"id": 2, "prediction": "Security"},
                {"id": 3, "prediction": "Non-security"},
            ]
        }
        selected = select_balanced_dynamic_examples(
            pool, evaluation, "Security", 42
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["id"] for row in selected}), 4)
        fallbacks = [
            row for row in selected
            if row["selection_source"] == "package_random_backfill"
        ]
        self.assertEqual(len(fallbacks), 2)
        self.assertEqual(
            {row["requested_quadrant"] for row in fallbacks},
            {"misclassified_positive", "misclassified_negative"},
        )
        self.assertEqual(
            selected,
            select_balanced_dynamic_examples(pool, evaluation, "Security", 42),
        )

    def test_meta_prompt_excludes_scores_and_history(self):
        examples = [{"text": "r", "label": "Security", "prediction": "Non-security"}]
        _, user = btape_meta_prompt("definitions", examples, ["Security", "Non-security"])
        self.assertNotIn("weighted-F1", user)
        self.assertNotIn("Historical Prompt", user)
        self.assertIn("incorrectly classified", user)

    def test_run_touches_test_once_and_persists_complete_state(self):
        split = Split(
            pool=[
                Example(0, "secure a", "Security"),
                Example(1, "plain a", "Non-security"),
                Example(2, "secure b", "Security"),
                Example(3, "plain b", "Non-security"),
                Example(4, "secure miss", "Non-security"),
                Example(5, "plain miss", "Security"),
            ],
            val=[Example(6, "secure v", "Security"), Example(7, "plain v", "Non-security")],
            test=[Example(8, "secure t", "Security"), Example(9, "plain t", "Non-security")],
        )
        backend = ScriptedBackend()
        with tempfile.TemporaryDirectory() as directory:
            result = run_btape(
                backend=backend,
                split=split,
                labels=["Non-security", "Security"],
                initial_definitions="Security: protects.\nNon-security: other.",
                config=BTAPEConfig(n_max=2),
                checkpoint_path=Path(directory) / "result.json",
                provenance={"test": True},
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["test_access_count"], 2)
        self.assertEqual(len(result["states"]), 3)
        self.assertTrue(all(state["demo_ids"] == [] for state in result["states"]))
        self.assertEqual(
            sum(
                kind in ("classification:test:p0", "classification:test:p_star")
                for kind in backend.kinds
            ),
            12,
        )
        self.assertEqual(set(result["source_tests"]), {"p0", "p_star"})


if __name__ == "__main__":
    unittest.main()
