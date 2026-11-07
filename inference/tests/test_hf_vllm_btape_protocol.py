from __future__ import annotations

import json
import unittest
from pathlib import Path

from pe2_btape.data import Example
from pe2_btape.backend import _context_limit_for_request
from pe2_btape.optimizer import classify


class BatchScriptedBackend:
    """No-vLLM test double for the optimizer's batch contract."""

    def __init__(self) -> None:
        self.requests = []

    def chat_many(self, requests):
        self.requests.extend(requests)
        return [
            (
                "Label: Security",
                {
                    "kind": request.kind,
                    "backend": "hf_safetensors_vllm",
                    "prompt_eval_count": 11,
                    "eval_count": 2,
                    "total_duration_ns": 10,
                    "batch_id": 9,
                    "batch_size": len(requests),
                    "batch_elapsed_ns": 60,
                    "finish_reason": "stop",
                    "output_token_ids": [index, 1],
                    "hf_rendered_prompt": f"rendered-{index}",
                    "hf_rendered_prompt_sha256": f"hash-{index}",
                },
            )
            for index, request in enumerate(requests, start=1)
        ]

    def chat(self, system, user, *, kind, num_predict):
        return self.chat_many(
            [type("Request", (), {"system": system, "user": user, "kind": kind, "num_predict": num_predict})()]
        )[0]


class HfVllmBtapeProtocolTests(unittest.TestCase):
    def test_formal_config_freezes_hf_vllm_and_sorted_labels(self):
        config = json.loads(
            Path("configs/frozen/source_optimization_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["protocol_version"],
            "source-optimization-v1",
        )
        self.assertEqual(config["labels"], ["Non-security", "Security"])
        self.assertEqual(config["seeds"], [40, 41, 42, 43, 44])
        self.assertEqual(config["methods"], ["pe2", "btape"])
        self.assertEqual(len(config["models"]), 3)
        self.assertEqual(config["optimization"]["voting_runs"], 3)
        self.assertEqual(config["runtime"]["temperature"], 0.0)
        self.assertEqual(config["runtime"]["max_model_len"], 8192)
        self.assertEqual(config["runtime"]["classification_max_model_len"], 2048)
        self.assertEqual(config["runtime"]["max_batch_requests"], 1)
        self.assertEqual(config["runtime"]["vllm_version"], "0.6.4")
        self.assertEqual(config["runtime"]["guided_decoding_backend"], "lm-format-enforcer")

    def test_only_classification_requests_take_the_2048_context_limit(self):
        limits = {
            kind: _context_limit_for_request(
                kind,
                engine_max_model_len=8192,
                classification_max_model_len=2048,
            )
            for kind in [
                "synthetic_preflight",
                "classification:validation",
                "classification:test:final",
                "proposal:definitions",
            ]
        }
        self.assertEqual(limits["synthetic_preflight"], 2048)
        self.assertEqual(limits["classification:validation"], 2048)
        self.assertEqual(limits["classification:test:final"], 2048)
        self.assertEqual(limits["proposal:definitions"], 8192)

    def test_batch_classifier_preserves_three_logical_votes_and_evidence(self):
        backend = BatchScriptedBackend()
        result = classify(
            backend=backend,
            definitions="Security: protect.\nNon-security: other.",
            demos=[],
            subset=[
                Example(1, "protect passwords", "Security"),
                Example(2, "show dashboard", "Non-security"),
            ],
            labels=["Non-security", "Security"],
            voting_runs=3,
            num_predict=64,
            split_name="validation",
        )
        self.assertEqual(len(backend.requests), 6)
        self.assertTrue(all(request.kind == "classification:validation" for request in backend.requests))
        self.assertTrue(all(request.num_predict == 64 for request in backend.requests))
        self.assertEqual(result["usage"]["calls"], 6)
        row = result["predictions"][0]
        self.assertEqual(row["votes"], ["Security", "Security", "Security"])
        self.assertEqual(len(row["vote_metadata"]), 3)
        self.assertIn("hf_rendered_prompt", row["request"])
        self.assertEqual(row["request"]["backend"], "hf_safetensors_vllm")

    def test_batch_classifier_records_exact_package_minimal_request(self):
        backend = BatchScriptedBackend()
        result = classify(
            backend=backend,
            definitions="Security: protect.\nNon-security: other.",
            demos=[],
            subset=[Example(1, "protect passwords", "Security")],
            labels=["Non-security", "Security"],
            voting_runs=3,
            num_predict=64,
            split_name="validation",
            classification_interface="package_minimal",
        )
        request = result["predictions"][0]["request"]
        self.assertEqual(request["user"], "Text: protect passwords")
        self.assertEqual(
            request["system"],
            "You are a precise text classifier. Classify each input text into exactly "
            "one of the provided categories. Output only the exact label name.\n\n"
            "Security: protect.\nNon-security: other.\n\n"
            "Output only the exact label name, nothing else.",
        )


if __name__ == "__main__":
    unittest.main()
