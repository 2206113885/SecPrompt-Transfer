from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pe2_btape.data import Example, Split, stratified_three_way_split, three_way_split
from pe2_btape.metrics import INVALID, compute_metrics
from pe2_btape.optimizer import (
    PE2Config,
    PromptState,
    rank_states,
    run_pe2,
    sample_errors,
)
from pe2_btape.prompts import (
    classification_messages,
    extract_label,
    majority,
    parse_revised_prompt,
    pe2_meta_prompt,
)


class ScriptedBackend:
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def chat(self, system, user, *, kind, num_predict):
        self.kinds.append(kind)
        if kind == "proposal":
            response = (
                "<revised_prompt>Security: protects assets from harm.\n"
                "Non-security: does not protect assets from harm.</revised_prompt>"
            )
        else:
            response = "Label: Security"
        return response, {
            "kind": kind,
            "prompt_eval_count": 10,
            "eval_count": 2,
            "total_duration_ns": 100,
            "load_duration_ns": 0,
        }


class ProtocolTests(unittest.TestCase):
    def test_package_minimal_interface_is_exact_and_rejects_demos(self):
        definitions = "Security: protects.\nNon-security: other."
        system, user = classification_messages(
            definitions,
            [],
            "Encrypt passwords.",
            ["Non-security", "Security"],
            "package_minimal",
        )
        self.assertEqual(
            system,
            "You are a precise text classifier. Classify each input text into exactly "
            "one of the provided categories. Output only the exact label name.\n\n"
            "Security: protects.\nNon-security: other.\n\n"
            "Output only the exact label name, nothing else.",
        )
        self.assertEqual(user, "Text: Encrypt passwords.")
        with self.assertRaises(ValueError):
            classification_messages(
                definitions,
                [Example(1, "demo", "Security")],
                "target",
                ["Non-security", "Security"],
                "package_minimal",
            )

    def test_split_is_30_30_40_and_disjoint(self):
        examples = [Example(i, f"r{i}", "Security") for i in range(510)]
        split = three_way_split(examples, 42)
        self.assertEqual((len(split.pool), len(split.val), len(split.test)), (153, 153, 204))
        ids = [item.id for part in (split.pool, split.val, split.test) for item in part]
        self.assertEqual(len(set(ids)), 510)
        self.assertEqual(
            split.manifest(), three_way_split(examples, 42).manifest()
        )

    def test_label_parsing_is_conservative(self):
        labels = ["Security", "Non-security"]
        self.assertEqual(extract_label("Label: Security", labels), "Security")
        self.assertEqual(extract_label("Non-security", labels), "Non-security")
        self.assertEqual(extract_label("I cannot decide", labels), INVALID)
        self.assertEqual(
            extract_label("Security or Non-security may apply", labels), INVALID
        )
        self.assertEqual(
            extract_label("Label: Security\nLabel: Non-security", labels), INVALID
        )

    def test_formal_stratified_split_reuses_all_rows_with_expected_counts(self):
        examples = [
            *[Example(i, f"s{i}", "Security") for i in range(187)],
            *[
                Example(187 + i, f"n{i}", "Non-security")
                for i in range(323)
            ],
        ]
        for seed in (40, 41, 42, 43, 44):
            split = stratified_three_way_split(examples, seed)
            self.assertEqual((len(split.pool), len(split.val), len(split.test)), (153, 153, 204))
            self.assertEqual(
                [sum(item.label == "Security" for item in part) for part in (split.pool, split.val, split.test)],
                [56, 56, 75],
            )
            ids = [item.id for part in (split.pool, split.val, split.test) for item in part]
            self.assertEqual(len(ids), len(set(ids)))

    def test_majority_requires_strict_majority(self):
        self.assertEqual(majority(["Security", "Security", "Non-security"]), "Security")
        self.assertEqual(majority(["Security", "Non-security"]), INVALID)

    def test_weighted_f1_keeps_invalid_outputs_as_errors(self):
        metrics = compute_metrics(
            ["Security", INVALID],
            ["Security", "Non-security"],
            ["Security", "Non-security"],
        )
        self.assertEqual(metrics.invalid_count, 1)
        self.assertAlmostEqual(metrics.weighted_f1, 0.5)

    def test_error_sample_uses_only_misclassifications(self):
        pool = [
            Example(0, "a", "Security"),
            Example(1, "b", "Non-security"),
            Example(2, "c", "Non-security"),
        ]
        evaluation = {
            "predictions": [
                {"id": 0, "prediction": "Security"},
                {"id": 1, "prediction": "Security"},
                {"id": 2, "prediction": INVALID},
            ]
        }
        errors = sample_errors(pool, evaluation, 4, 99)
        self.assertEqual({row["id"] for row in errors}, {1, 2})

    def test_newer_prompt_wins_an_exact_tie(self):
        states = [
            PromptState(0, "a", [1], 0.8, {}),
            PromptState(1, "b", [2], 0.8, {}),
            PromptState(2, "c", [3], 0.7, {}),
        ]
        self.assertEqual(rank_states(states)[0].iteration, 1)

    def test_meta_prompt_contains_ranked_history_and_delimiters(self):
        history = [
            {"iteration": 1, "val_weighted_f1": 0.9, "definitions": "best"},
            {"iteration": 0, "val_weighted_f1": 0.8, "definitions": "initial"},
        ]
        errors = [
            {"text": "r", "label": "Security", "prediction": "Non-security"}
        ]
        _, user = pe2_meta_prompt("current", history, errors, ["Security", "Non-security"])
        self.assertIn("Validation weighted-F1: 0.900000000000", user)
        self.assertIn("Ground-truth label: Security", user)
        self.assertIn("<revised_prompt>", user)
        self.assertEqual(parse_revised_prompt("<revised_prompt>long enough definitions text</revised_prompt>"), "long enough definitions text")
        self.assertIsNone(parse_revised_prompt("no delimiters"))

    def test_optimizer_touches_test_only_once_and_preserves_all_candidates(self):
        split = Split(
            pool=[
                Example(0, "p0", "Security"),
                Example(1, "p1", "Security"),
                Example(2, "n0", "Non-security"),
                Example(3, "n1", "Non-security"),
            ],
            val=[
                Example(4, "v0", "Security"),
                Example(5, "v1", "Non-security"),
            ],
            test=[
                Example(6, "t0", "Security"),
                Example(7, "t1", "Non-security"),
            ],
        )
        backend = ScriptedBackend()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = run_pe2(
                backend=backend,
                split=split,
                labels=["Security", "Non-security"],
                initial_definitions="Security: protects.\nNon-security: other.",
                config=PE2Config(n_max=2),
                checkpoint_path=output,
                provenance={"test": True},
            )
            calls_after_completion = len(backend.kinds)
            resumed = run_pe2(
                backend=backend,
                split=split,
                labels=["Security", "Non-security"],
                initial_definitions="Security: protects.\nNon-security: other.",
                config=PE2Config(n_max=2),
                checkpoint_path=output,
                provenance={"test": True},
                resume=True,
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["test_access_count"], 2)
        self.assertEqual(len(result["states"]), 3)
        self.assertEqual(len(result["iterations"]), 2)
        self.assertEqual(
            len(
                [
                    kind
                    for kind in backend.kinds
                    if kind in ("classification:test:p0", "classification:test:p_star")
                ]
            ),
            12,
        )
        self.assertEqual(persisted["test_access_count"], 2)
        self.assertEqual(resumed["test_access_count"], 2)
        self.assertEqual(set(result["source_tests"]), {"p0", "p_star"})
        self.assertTrue(all(state["demo_ids"] == [] for state in result["states"]))
        self.assertEqual(len(backend.kinds), calls_after_completion)
        for item in result["iterations"]:
            self.assertLessEqual(len(item["top_history_shown"]), 3)
            self.assertTrue(item["proposal"]["parse_valid"])


if __name__ == "__main__":
    unittest.main()
