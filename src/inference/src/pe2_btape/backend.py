from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CallRecord:
    kind: str
    prompt_eval_count: int
    eval_count: int
    total_duration_ns: int
    load_duration_ns: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChatRequest:
    """One logical chat completion request.

    The optimizer deliberately treats an inference vote as a separate request,
    even under greedy decoding.  A batch-capable backend may execute those
    requests together, but it must return them in the same logical order.
    """

    system: str
    user: str
    kind: str
    num_predict: int


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_classification(kind: str) -> bool:
    return kind == "synthetic_preflight" or kind.startswith("classification:")


def _local_hf_artifact_metadata(
    model_dir: str,
    expected_revision: str,
    *,
    revision_verified_externally: bool = False,
) -> dict[str, object]:
    """Validate a local snapshot without hashing multi-gigabyte shards per cell."""
    root = Path(model_dir)
    required = ("config.json", "tokenizer_config.json", "model.safetensors.index.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Local HF artifact is incomplete: missing {missing}")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    missing_shards = [name for name in shard_names if not (root / name).is_file()]
    if missing_shards:
        raise RuntimeError(f"Local HF artifact is missing weight shards: {missing_shards}")

    metadata_dir = root / ".cache" / "huggingface" / "download"
    revision_files = sorted(metadata_dir.glob("*.metadata"))
    observed_revisions: set[str] = set()
    for metadata_path in revision_files:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        if lines:
            observed_revisions.add(lines[0].strip())
    if not observed_revisions and not revision_verified_externally:
        raise RuntimeError("Local HF download metadata does not record a revision")
    if observed_revisions and expected_revision not in observed_revisions:
        raise RuntimeError(
            f"Expected HF revision {expected_revision!r} is absent from local download metadata"
        )
    unexpected = sorted(value for value in observed_revisions if value != expected_revision)
    if unexpected:
        raise RuntimeError(
            f"Local snapshot contains files from revisions other than {expected_revision}: {unexpected}"
        )
    return {
        "expected_revision": expected_revision,
        "observed_revisions": sorted(observed_revisions),
        "revision_metadata_files": len(revision_files),
        "config_sha256": _sha256_file(root / "config.json"),
        "tokenizer_config_sha256": _sha256_file(root / "tokenizer_config.json"),
        "weight_index_sha256": _sha256_file(root / "model.safetensors.index.json"),
        "weight_shards": [
            {"name": name, "bytes": (root / name).stat().st_size}
            for name in shard_names
        ],
        "total_weight_bytes": sum((root / name).stat().st_size for name in shard_names),
    }


def _context_limit_for_request(
    kind: str,
    *,
    engine_max_model_len: int,
    classification_max_model_len: int | None,
) -> int:
    """Return the frozen context budget applicable to one logical request.

    PE2/BT-APE must use a long context only when asking the model to propose a
    revised definition.  Classification calls are separately constrained to
    the 2,048-token capacity used by the matched prompting baselines, so an
    accidental long classification prompt fails loudly instead of silently
    changing the deployment protocol.
    """

    if _is_classification(kind) and classification_max_model_len is not None:
        return classification_max_model_len
    return engine_max_model_len


def _summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    by_kind: dict[str, dict[str, int]] = {}
    for record in records:
        kind = str(record.get("kind", "unknown"))
        bucket = by_kind.setdefault(
            kind,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "duration_ns": 0},
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += int(record.get("prompt_eval_count", 0))
        bucket["output_tokens"] += int(record.get("eval_count", 0))
        bucket["duration_ns"] += int(record.get("total_duration_ns", 0))
    return {
        "calls": len(records),
        "input_tokens": sum(int(record.get("prompt_eval_count", 0)) for record in records),
        "output_tokens": sum(int(record.get("eval_count", 0)) for record in records),
        "duration_ns": sum(int(record.get("total_duration_ns", 0)) for record in records),
        "by_kind": by_kind,
    }


class OllamaBackend:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 600,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.calls: list[CallRecord] = []

    def _request(self, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Ollama request failed after {self.retries} attempts: {last_error}")

    def model_metadata(self) -> dict[str, object]:
        tags = self._request("/api/tags")
        for model in tags.get("models", []):
            if model.get("name") == self.model or model.get("model") == self.model:
                return model
        return {"name": self.model, "warning": "model not found in /api/tags"}

    def runtime_metadata(self) -> dict[str, object]:
        return self._request("/api/version")

    def chat(
        self,
        system: str,
        user: str,
        *,
        kind: str,
        num_predict: int,
    ) -> tuple[str, dict[str, object]]:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0.0, "num_predict": num_predict},
        }
        result = self._request("/api/chat", payload)
        record = CallRecord(
            kind=kind,
            prompt_eval_count=int(result.get("prompt_eval_count", 0)),
            eval_count=int(result.get("eval_count", 0)),
            total_duration_ns=int(result.get("total_duration", 0)),
            load_duration_ns=int(result.get("load_duration", 0)),
        )
        self.calls.append(record)
        return str(result.get("message", {}).get("content", "")).strip(), record.as_dict()

    def usage_since(self, start: int = 0) -> dict[str, object]:
        return _summarize_records([record.as_dict() for record in self.calls[start:]])


class VLLMBackend:
    """Original-HF-weight backend with auditable vLLM inference.

    The completed zero-shot controls use direct vLLM ``LLM.generate`` calls.
    This backend keeps that exact inference family while adding an optimizer
    interface.  The dispatch capacity is explicit in the frozen configuration;
    for the equalized rerun every logical request is singleton.  The engine
    lives for the whole optimization process so no method call silently falls
    back to Ollama.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_dir: str,
        manifest_path: str | None,
        expected_revision: str,
        expected_vllm_version: str,
        physical_gpu: int,
        dtype: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        classification_max_model_len: int | None,
        enforce_eager: bool,
        engine_seed: int,
        attention_backend: str,
        stop_strings: list[str],
        max_batch_requests: int = 512,
        guided_choice: list[str] | None = None,
        guided_decoding_backend: str | None = None,
        emit_batch_logs: bool = True,
        chat_template_kwargs: dict[str, object] | None = None,
    ) -> None:
        expected_gpu = str(physical_gpu)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
            raise RuntimeError(
                "HF/vLLM backend requires CUDA_VISIBLE_DEVICES="
                f"{expected_gpu}, got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend

        try:
            import torch
            import transformers
            import vllm
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import GuidedDecodingParams
        except ImportError as error:
            raise RuntimeError(
                "HF/vLLM dependencies are absent; use the provisioned vLLM environment"
            ) from error

        if vllm.__version__ != expected_vllm_version:
            raise RuntimeError(
                f"Expected vLLM {expected_vllm_version}, found {vllm.__version__}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "HF/vLLM backend expects exactly one CUDA-visible device after GPU pinning"
            )

        self.model_name = str(model_name)
        self.model_dir = os.fspath(model_dir)
        if not os.path.isdir(self.model_dir):
            raise RuntimeError(f"Local HF model directory is missing: {self.model_dir}")
        self.manifest_path = os.fspath(manifest_path) if manifest_path else None
        external_manifest: dict[str, object] | None = None
        if self.manifest_path is not None:
            if not os.path.isfile(self.manifest_path):
                raise RuntimeError(f"Configured model manifest is missing: {self.manifest_path}")
            with open(self.manifest_path, encoding="utf-8") as handle:
                external_manifest = json.load(handle)
            observed_revision = str(
                external_manifest.get("revision")
                or external_manifest.get("model_revision")
                or external_manifest.get("requested_revision")
                or ""
            )
            if observed_revision and observed_revision != expected_revision:
                raise RuntimeError(
                    f"Local model manifest revision {observed_revision!r} differs from "
                    f"{expected_revision!r}"
                )
        artifact_metadata = _local_hf_artifact_metadata(
            self.model_dir,
            expected_revision,
            revision_verified_externally=external_manifest is not None,
        )

        self._SamplingParams = SamplingParams
        self._GuidedDecodingParams = GuidedDecodingParams
        self._max_model_len = int(max_model_len)
        self._classification_max_model_len = (
            int(classification_max_model_len)
            if classification_max_model_len is not None
            else None
        )
        if (
            self._classification_max_model_len is not None
            and (
                self._classification_max_model_len < 1
                or self._classification_max_model_len > self._max_model_len
            )
        ):
            raise ValueError(
                "classification_max_model_len must be positive and no greater than "
                "max_model_len"
            )
        self._engine_seed = int(engine_seed)
        self._stop_strings = [str(value) for value in stop_strings]
        self._max_batch_requests = int(max_batch_requests)
        if self._max_batch_requests < 1:
            raise ValueError("max_batch_requests must be positive")
        self._guided_choice = (
            [str(value) for value in guided_choice]
            if guided_choice is not None
            else None
        )
        if self._guided_choice is not None and (
            len(self._guided_choice) != 2
            or len(set(self._guided_choice)) != len(self._guided_choice)
        ):
            raise ValueError("guided_choice must contain exactly two distinct labels")
        self._guided_decoding_backend = (
            str(guided_decoding_backend)
            if guided_decoding_backend is not None
            else None
        )
        if (self._guided_choice is None) != (self._guided_decoding_backend is None):
            raise ValueError(
                "guided_choice and guided_decoding_backend must be configured together"
            )
        self.enforces_binary_choice = self._guided_choice is not None
        self._calls: list[dict[str, object]] = []
        self._batch_id = 0
        self._emit_batch_logs = bool(emit_batch_logs)
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        self._render_cache: dict[tuple[str, str], tuple[str, int]] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False
        )
        self.runtime = {
            "backend": "hf_safetensors_vllm",
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_gpu_name": torch.cuda.get_device_name(0),
            "logical_gpu_capability": list(torch.cuda.get_device_capability(0)),
            "model_name": self.model_name,
            "model_dir": self.model_dir,
            "model_manifest_path": self.manifest_path,
            "model_manifest_sha256": (
                _sha256_file(self.manifest_path) if self.manifest_path else None
            ),
            "model_manifest": external_manifest,
            "local_artifact": artifact_metadata,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "tokenizer_chat_template_sha256": _text_sha256(
                self.tokenizer.chat_template or ""
            ),
            "max_model_len": self._max_model_len,
            "classification_max_model_len": self._classification_max_model_len,
            "dtype": dtype,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enforce_eager": bool(enforce_eager),
            "engine_seed": self._engine_seed,
            "attention_backend": attention_backend,
            "stop_strings": self._stop_strings,
            "guided_choice": self._guided_choice,
            "guided_decoding_backend": self._guided_decoding_backend,
            "chat_template_kwargs": self._chat_template_kwargs,
            "max_batch_requests": self._max_batch_requests,
            "request_dispatch_batch_size": self._max_batch_requests,
            "max_num_seqs": self._max_batch_requests,
        }
        print("HF_VLLM_RUNTIME " + json.dumps(self.runtime, sort_keys=True), flush=True)
        self.llm = LLM(
            model=self.model_dir,
            tokenizer=self.model_dir,
            tokenizer_mode="auto",
            trust_remote_code=False,
            tensor_parallel_size=1,
            dtype=dtype,
            seed=self._engine_seed,
            gpu_memory_utilization=float(gpu_memory_utilization),
            swap_space=0,
            max_model_len=self._max_model_len,
            max_num_seqs=self._max_batch_requests,
            enforce_eager=bool(enforce_eager),
        )

    def model_metadata(self) -> dict[str, object]:
        return {
            "name": self.model_name,
            "artifact": "verified local safetensors",
            "model_dir": self.model_dir,
            "manifest_sha256": self.runtime["model_manifest_sha256"],
            "revision": self.runtime["local_artifact"]["expected_revision"],
            "weight_index_sha256": self.runtime["local_artifact"]["weight_index_sha256"],
        }

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self.runtime)

    def _render(self, system: str, user: str) -> tuple[str, int]:
        key = (system, user)
        cached = self._render_cache.get(key)
        if cached is not None:
            return cached
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **self._chat_template_kwargs,
        )
        token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            **self._chat_template_kwargs,
        )
        cached = (str(rendered), len(token_ids))
        self._render_cache[key] = cached
        return cached

    def chat(
        self,
        system: str,
        user: str,
        *,
        kind: str,
        num_predict: int,
    ) -> tuple[str, dict[str, object]]:
        return self.chat_many(
            [ChatRequest(system=system, user=user, kind=kind, num_predict=num_predict)]
        )[0]

    def chat_many(
        self, requests: list[ChatRequest]
    ) -> list[tuple[str, dict[str, object]]]:
        if not requests:
            return []
        grouped: dict[tuple[int, bool], list[tuple[int, ChatRequest, str, int]]] = {}
        for index, request in enumerate(requests):
            rendered, prompt_tokens = self._render(request.system, request.user)
            context_limit = _context_limit_for_request(
                request.kind,
                engine_max_model_len=self._max_model_len,
                classification_max_model_len=self._classification_max_model_len,
            )
            if prompt_tokens + request.num_predict > context_limit:
                raise RuntimeError(
                    f"{request.kind} prompt plus generation budget "
                    f"({prompt_tokens + request.num_predict}) exceeds context limit "
                    f"({context_limit})"
                )
            constrained = _is_classification(request.kind)
            grouped.setdefault((request.num_predict, constrained), []).append(
                (index, request, rendered, prompt_tokens)
            )

        responses: list[tuple[str, dict[str, object]] | None] = [None] * len(requests)
        for (num_predict, constrained), members in grouped.items():
            for start in range(0, len(members), self._max_batch_requests):
                batch = members[start : start + self._max_batch_requests]
                self._batch_id += 1
                batch_id = self._batch_id
                sampling = self._SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=num_predict,
                    stop=self._stop_strings,
                    include_stop_str_in_output=False,
                    guided_decoding=(
                        self._GuidedDecodingParams(
                            choice=self._guided_choice,
                            backend=self._guided_decoding_backend,
                        )
                        if constrained and self._guided_choice is not None
                        else None
                    ),
                )
                if self._emit_batch_logs:
                    print(
                        "HF_VLLM_BATCH "
                        + json.dumps(
                            {
                                "batch_id": batch_id,
                                "request_count": len(batch),
                                "request_dispatch_batch_size": self._max_batch_requests,
                                "max_num_seqs": self._max_batch_requests,
                                "num_predict": num_predict,
                                "guided_choice": (
                                    self._guided_choice if constrained else None
                                ),
                                "guided_decoding_backend": (
                                    self._guided_decoding_backend if constrained else None
                                ),
                                "kinds": sorted({member[1].kind for member in batch}),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                started_ns = time.perf_counter_ns()
                outputs = self.llm.generate(
                    [member[2] for member in batch],
                    sampling_params=sampling,
                    use_tqdm=False,
                )
                elapsed_ns = time.perf_counter_ns() - started_ns
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"vLLM returned {len(outputs)} outputs for {len(batch)} requests"
                    )
                elapsed_share_ns = elapsed_ns // len(batch)
                for (index, request, rendered, prompt_tokens), output in zip(batch, outputs):
                    if len(output.outputs) != 1:
                        raise RuntimeError("vLLM backend expected exactly one completion per request")
                    completion = output.outputs[0]
                    raw = str(completion.text)
                    record: dict[str, object] = {
                        "kind": request.kind,
                        "prompt_eval_count": prompt_tokens,
                        "eval_count": len(completion.token_ids),
                        "total_duration_ns": elapsed_share_ns,
                        "load_duration_ns": 0,
                        "backend": "hf_safetensors_vllm",
                        "batch_id": batch_id,
                        "batch_size": len(batch),
                        "batch_elapsed_ns": elapsed_ns,
                        "finish_reason": completion.finish_reason,
                        "output_token_ids": list(completion.token_ids),
                        "hf_rendered_prompt": rendered,
                        "hf_rendered_prompt_sha256": _text_sha256(rendered),
                    }
                    self._calls.append(record)
                    responses[index] = (raw, record)
        if any(response is None for response in responses):
            raise AssertionError("A vLLM request was not assigned an output")
        return [response for response in responses if response is not None]

    def usage_since(self, start: int = 0) -> dict[str, object]:
        return _summarize_records(self._calls[start:])
