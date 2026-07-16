"""Single-GPU Transformers recurrent evaluator for the reproduction contract.

The parent process owns timeouts and atomic result publication.  A persistent
child process owns the model so a wedged CUDA generation can be terminated
without silently skipping the affected sample.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import multiprocessing
import os
import queue
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from recurrent.protocol import (
    CALLBACK_MODES,
    MemoryRecord,
    parse_final_action,
    parse_intermediate_action,
    resolve_callback_query,
    retrieve_top1,
    word_recall,
)
from taskutils.data_synthesis.reproduction_builder import (
    manifest_record_from_dict,
    validate_artifact_bundle,
)
from taskutils.data_synthesis.reproduction_manifest import (
    ManifestRecord,
    canonical_json_bytes,
    token_ids_sha256,
    validate_canonical_jsonl,
    validate_manifest_record,
)
from taskutils.memory_eval.reproduction_metrics import evaluate_output


RUN_SCHEMA_VERSION = 1
RUN_KIND = "rememr1-transformers-recurrent-eval-v1"
COMPLETION_FILENAME = "COMPLETED.json"
RESULTS_FILENAME = "results.jsonl"
SUMMARY_FILENAME = "summary.json"
RUN_CONFIG_FILENAME = "run_config.json"
NO_MEMORY = "No previous memory"
NO_RECALLED_MEMORY = "No memory was recalled."
PROMPT_TEMPLATE_REVISION = "rememr1-template-v1"
_FLOATING_REVISIONS = frozenset({"", "main", "master", "latest", "head"})
_WORD_TOKEN = re.compile(r"\S+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


INTERMEDIATE_PROMPT = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. You should generate response in the following format:
- Output your thinking process in <thinking>your_thinking_process</thinking>.
- Read the provided section carefully and update the memory with the new information that helps to answer the problem in only one <update>the_updated_memory</update> action. Be sure to retain all relevant details from the previous memory while adding any new, useful information.
- If you notice partial key evidence that is not enough to answer the problem, also output only one `<recall>query</recall>` (e.g. `<recall>who's the president of the United States?</recall>`) to retrieve information in previous memories.

<problem>
{question}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

FINAL_PROMPT = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{question}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

Your answer:
"""

PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "final": FINAL_PROMPT,
            "intermediate": INTERMEDIATE_PROMPT,
            "kind": "reproduction-eval-prompts-v1",
            "revision": PROMPT_TEMPLATE_REVISION,
        }
    )
).hexdigest()


class EvaluationRunnerError(RuntimeError):
    """Raised when the runner cannot satisfy a hard evaluation contract."""


class WorkerTimeoutError(TimeoutError):
    pass


class WorkerExitedError(RuntimeError):
    pass


def _require_nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _require_fixed_revision(value: Any, name: str) -> str:
    revision = _require_nonempty(value, name)
    if revision.casefold() in _FLOATING_REVISIONS:
        raise ValueError(f"{name} must be an immutable revision")
    return revision


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BackendGeneration:
    text: str
    generated_tokens: int
    reached_token_limit: bool
    generated_token_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("generated text must be a string")
        if (
            isinstance(self.generated_tokens, bool)
            or not isinstance(self.generated_tokens, int)
            or self.generated_tokens < 0
        ):
            raise ValueError("generated_tokens must be a non-negative integer")
        if not isinstance(self.reached_token_limit, bool):
            raise TypeError("reached_token_limit must be bool")
        if (
            not isinstance(self.generated_token_sha256, str)
            or _SHA256.fullmatch(self.generated_token_sha256) is None
        ):
            raise ValueError("generated_token_sha256 must be a lowercase SHA256 digest")


class RecurrentBackend(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def encode_context(self, text: str) -> Sequence[Any]: ...

    def decode_context_tokens(self, token_ids: Sequence[Any]) -> str: ...

    def generate_greedy(self, prompt: str, *, max_new_tokens: int) -> BackendGeneration: ...


@dataclass(frozen=True, slots=True)
class SampleEvaluationConfig:
    callback_mode: str
    chunk_size: int
    memory_max_tokens: int = 768
    final_max_tokens: int = 512

    def validate(self) -> None:
        if self.callback_mode not in CALLBACK_MODES:
            raise ValueError(
                f"callback_mode must be one of {sorted(CALLBACK_MODES)}"
            )
        _require_positive_int(self.chunk_size, "chunk_size")
        _require_positive_int(self.memory_max_tokens, "memory_max_tokens")
        _require_positive_int(self.final_max_tokens, "final_max_tokens")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelLoadSpec:
    kind: str = "transformers"
    artifact_kind: str = "base"
    base_model_id: str = ""
    revision: str = ""
    artifact_path: str | None = None
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    template_revision: str = ""
    attention_implementation: str = "sdpa"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    seed: int = 42
    local_files_only: bool = True
    allow_test_backend: bool = False
    scripted_outputs: tuple[str, ...] = ()
    scripted_load_delay_s: float = 0.0

    def validate(self) -> None:
        if self.kind not in {"transformers", "scripted"}:
            raise ValueError("backend kind must be transformers or scripted")
        if self.kind == "scripted":
            if not self.allow_test_backend:
                raise ValueError("scripted backend is restricted to explicit tests")
            if (
                isinstance(self.scripted_load_delay_s, bool)
                or not isinstance(self.scripted_load_delay_s, (int, float))
                or not math.isfinite(self.scripted_load_delay_s)
                or self.scripted_load_delay_s < 0
            ):
                raise ValueError("scripted load delay must be finite and non-negative")
            if any(not isinstance(output, str) for output in self.scripted_outputs):
                raise TypeError("scripted_outputs must contain only strings")
            if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
                raise ValueError("seed must be a non-negative integer")
            return
        if self.artifact_kind not in {"base", "adapter", "merged"}:
            raise ValueError("artifact_kind must be base, adapter, or merged")
        _require_nonempty(self.base_model_id, "base_model_id")
        _require_fixed_revision(self.revision, "revision")
        _require_nonempty(self.tokenizer_id, "tokenizer_id")
        _require_fixed_revision(self.tokenizer_revision, "tokenizer_revision")
        _require_fixed_revision(self.template_revision, "template_revision")
        if self.template_revision != PROMPT_TEMPLATE_REVISION:
            raise ValueError(
                "template_revision does not match the compiled eval prompt contract"
            )
        _require_nonempty(self.attention_implementation, "attention_implementation")
        if self.dtype != "bfloat16":
            raise ValueError("formal evaluation dtype is fixed to bfloat16")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("local_files_only must be bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.artifact_kind in {"adapter", "merged"}:
            if self.artifact_path is None:
                raise ValueError(f"{self.artifact_kind} evaluation needs artifact_path")
            artifact = Path(self.artifact_path).expanduser()
            if not artifact.is_dir():
                raise ValueError("artifact_path must be an existing directory")
        elif self.artifact_path is not None:
            raise ValueError("base evaluation must not set artifact_path")

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("scripted_outputs")
        value.pop("allow_test_backend")
        value.pop("scripted_load_delay_s")
        return value


class ScriptedBackend:
    """Spawn-safe deterministic backend used only by CPU contract tests."""

    def __init__(self, spec: ModelLoadSpec):
        spec.validate()
        if spec.scripted_load_delay_s:
            time.sleep(spec.scripted_load_delay_s)
        self._spec = spec
        self._outputs = list(spec.scripted_outputs)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"backend": "scripted", "greedy": True, "seed": self._spec.seed}

    def encode_context(self, text: str) -> Sequence[int]:
        self._context_token_text = tuple(
            match.group(0) for match in _WORD_TOKEN.finditer(text)
        )
        return tuple(range(1, len(self._context_token_text) + 1))

    def decode_context_tokens(self, token_ids: Sequence[int]) -> str:
        return " ".join(
            self._context_token_text[int(token_id) - 1] for token_id in token_ids
        )

    def generate_greedy(self, prompt: str, *, max_new_tokens: int) -> BackendGeneration:
        del prompt
        if not self._outputs:
            raise RuntimeError("scripted backend exhausted its outputs")
        output = self._outputs.pop(0)
        if output.startswith("__sleep__:"):
            _, seconds, output = output.split(":", 2)
            time.sleep(float(seconds))
        if output.startswith("__raise__:"):
            raise RuntimeError(output.partition(":")[2])
        if output == "__exit__":
            os._exit(23)
        token_values = [match.group(0) for match in _WORD_TOKEN.finditer(output)]
        generated_tokens = len(token_values)
        return BackendGeneration(
            text=output,
            generated_tokens=generated_tokens,
            reached_token_limit=generated_tokens >= max_new_tokens,
            generated_token_sha256=_sha256_bytes(
                canonical_json_bytes(token_values)
            ),
        )


class TransformersBackend:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        device: str,
        metadata: Mapping[str, Any],
        chat_template_renderer: Callable[..., Any] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.device = device
        self._metadata = dict(metadata)
        self._chat_template_renderer = chat_template_renderer

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    def encode_context(self, text: str) -> Sequence[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        return tuple(int(value) for value in input_ids)

    def decode_context_tokens(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def generate_greedy(self, prompt: str, *, max_new_tokens: int) -> BackendGeneration:
        renderer = self._chat_template_renderer
        if renderer is None:
            from verl.utils.chat_template import (
                apply_chat_template_without_native_thinking as renderer,
            )

        rendered = renderer(
            self.tokenizer,
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if isinstance(rendered, Mapping):
            input_ids = rendered["input_ids"]
            attention_mask = rendered.get("attention_mask")
        else:
            input_ids = rendered
            attention_mask = None
        if getattr(input_ids, "ndim", 0) == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = self.torch.ones_like(input_ids)
        elif getattr(attention_mask, "ndim", 0) == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(self.device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        generated = sequences[0, input_ids.shape[-1] :]
        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return BackendGeneration(
            text=text,
            generated_tokens=int(generated.numel()),
            reached_token_limit=int(generated.numel()) >= max_new_tokens,
            generated_token_sha256=_sha256_bytes(
                canonical_json_bytes(
                    [int(token_id) for token_id in generated.detach().cpu().tolist()]
                )
            ),
        )


def _configure_deterministic_torch(torch_module: Any, seed: int) -> dict[str, Any]:
    """Apply reproducible greedy-eval settings supported by the installed torch."""

    settings = {
        "cublas_workspace_config": os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        ),
        "deterministic_algorithms_warn_only": False,
        "seed": seed,
    }
    manual_seed = getattr(torch_module, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    cuda_manual_seed_all = getattr(cuda, "manual_seed_all", None)
    if callable(cuda_manual_seed_all):
        cuda_manual_seed_all(seed)
    use_deterministic_algorithms = getattr(
        torch_module, "use_deterministic_algorithms", None
    )
    if callable(use_deterministic_algorithms):
        use_deterministic_algorithms(True, warn_only=True)
        settings["deterministic_algorithms_warn_only"] = True
    backends = getattr(torch_module, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        settings["cudnn_benchmark"] = False
        settings["cudnn_deterministic"] = True
    return settings


def _load_strict_merged_qwen35_model(
    model_path: str | Path,
    *,
    dtype: Any,
    attention_implementation: str,
    local_files_only: bool,
    low_cpu_mem_usage: bool,
) -> Any:
    """Load exported merged weights without misrepresenting them as a base snapshot."""

    from transformers import AutoModelForCausalLM

    from verl.models.qwen35 import (
        inspect_qwen35_config,
        validate_qwen35_loading_info,
    )

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attention_implementation,
        dtype=dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=low_cpu_mem_usage,
        output_loading_info=True,
        trust_remote_code=False,
    )
    validate_qwen35_loading_info(loading_info)
    inspect_qwen35_config(model.config)
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise EvaluationRunnerError(
            f"merged artifact loaded unexpected model class {type(model).__name__}"
        )
    if hasattr(model, "peft_config"):
        raise EvaluationRunnerError("merged artifact unexpectedly contains PEFT state")
    wrong_dtypes = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    ]
    if wrong_dtypes:
        raise EvaluationRunnerError(
            f"merged artifact floating parameters are not BF16: {wrong_dtypes[:8]}"
        )
    return model


def _artifact_metadata_value(metadata: Any, name: str) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return getattr(metadata, name, None)


def load_transformers_backend(
    spec: ModelLoadSpec,
    *,
    torch_module: Any = None,
    qwen_loader: Callable[..., Any] | None = None,
    tokenizer_loader: Any = None,
    adapter_validator: Callable[..., Any] | None = None,
    adapter_loader: Callable[..., Any] | None = None,
    merged_validator: Callable[..., Any] | None = None,
    merged_loader: Callable[..., Any] | None = None,
) -> TransformersBackend:
    """Load base, validated PEFT adapter, or merged Qwen text-only weights."""

    spec.validate()
    if spec.kind != "transformers":
        raise ValueError("load_transformers_backend requires kind=transformers")
    if torch_module is None:
        import torch as torch_module
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        tokenizer_loader = AutoTokenizer.from_pretrained
    deterministic_settings = _configure_deterministic_torch(torch_module, spec.seed)
    dtype = getattr(torch_module, spec.dtype)
    adapter_metadata = None
    merged_metadata = None
    loaded = None
    if spec.artifact_kind == "merged":
        if merged_validator is None:
            from verl.utils.checkpoint.reproduction import (
                validate_merged_model_artifact,
            )

            merged_validator = validate_merged_model_artifact
        if merged_loader is None:
            merged_loader = _load_strict_merged_qwen35_model
        merged_metadata = merged_validator(spec.artifact_path)
        expected = {
            "base_model_id": spec.base_model_id,
            "base_model_revision": spec.revision,
            "tokenizer_id": spec.tokenizer_id,
            "tokenizer_revision": spec.tokenizer_revision,
            "template_revision": spec.template_revision,
            "dtype": spec.dtype,
        }
        for name, value in expected.items():
            if _artifact_metadata_value(merged_metadata, name) != value:
                raise EvaluationRunnerError(
                    f"merged metadata {name} differs from eval load spec"
                )
        model = merged_loader(
            spec.artifact_path,
            dtype=dtype,
            attention_implementation=spec.attention_implementation,
            local_files_only=spec.local_files_only,
            low_cpu_mem_usage=True,
        )
    else:
        if qwen_loader is None:
            from verl.models.qwen35 import load_qwen35_text_model as qwen_loader
        loaded = qwen_loader(
            spec.base_model_id,
            revision=spec.revision,
            attn_implementation=spec.attention_implementation,
            dtype=dtype,
            local_files_only=spec.local_files_only,
            low_cpu_mem_usage=True,
        )
        model = loaded.model
    if spec.artifact_kind == "adapter":
        if adapter_validator is None or adapter_loader is None:
            from verl.utils.checkpoint.reproduction import (
                load_peft_adapter,
                validate_adapter_export,
            )

            adapter_validator = adapter_validator or validate_adapter_export
            adapter_loader = adapter_loader or load_peft_adapter
        adapter_metadata = adapter_validator(spec.artifact_path)
        expected = {
            "base_model_id": spec.base_model_id,
            "base_model_revision": spec.revision,
            "tokenizer_id": spec.tokenizer_id,
            "tokenizer_revision": spec.tokenizer_revision,
            "template_revision": spec.template_revision,
        }
        for name, value in expected.items():
            if getattr(adapter_metadata, name) != value:
                raise EvaluationRunnerError(
                    f"adapter metadata {name} differs from eval load spec"
                )
        model = adapter_loader(model, spec.artifact_path, validate_export=False)
    tokenizer = tokenizer_loader(
        spec.tokenizer_id,
        revision=spec.tokenizer_revision,
        local_files_only=spec.local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise EvaluationRunnerError("evaluation requires the pinned fast tokenizer")
    model = model.to(spec.device)
    model.eval()
    mapping_metadata = getattr(loaded, "metadata", None) if loaded is not None else None
    if hasattr(mapping_metadata, "to_dict"):
        mapping_metadata = mapping_metadata.to_dict()
    metadata = {
        "adapter_metadata": (
            adapter_metadata.to_dict() if hasattr(adapter_metadata, "to_dict") else None
        ),
        "merged_metadata": (
            dict(merged_metadata)
            if isinstance(merged_metadata, Mapping)
            else merged_metadata.to_dict()
            if hasattr(merged_metadata, "to_dict")
            else None
        ),
        "artifact_kind": spec.artifact_kind,
        "backend": "transformers",
        "base_model_id": spec.base_model_id,
        "device": spec.device,
        "determinism": deterministic_settings,
        "dtype": spec.dtype,
        "greedy": True,
        "qwen35_mapping": mapping_metadata,
        "revision": spec.revision,
        "seed": spec.seed,
        "tokenizer_id": spec.tokenizer_id,
        "tokenizer_revision": spec.tokenizer_revision,
    }
    return TransformersBackend(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        device=spec.device,
        metadata=metadata,
    )


def _load_backend(spec: ModelLoadSpec) -> RecurrentBackend:
    spec.validate()
    if spec.kind == "scripted":
        return ScriptedBackend(spec)
    return load_transformers_backend(spec)


def _occurrences_dict(occurrences: Any) -> dict[str, Any]:
    return {
        "closing_count": occurrences.closing_count,
        "empty": occurrences.has_empty,
        "opening_count": occurrences.opening_count,
        "payloads": list(occurrences.payloads),
        "well_formed": occurrences.is_well_formed,
    }


def _query_status(action: Any) -> str:
    occurrences = action.recall_occurrences
    if occurrences.is_absent:
        return "absent"
    if occurrences.has_duplicate:
        return "duplicate"
    if not occurrences.is_well_formed:
        return "malformed"
    if occurrences.has_empty:
        return "empty"
    if occurrences.is_single_non_empty:
        return "valid"
    return "malformed"


def _input_identity(record: ManifestRecord) -> dict[str, Any]:
    return {
        "context_sha256": record.context_sha256,
        "context_token_ids_sha256": record.context_token_ids_sha256,
        "document_count": record.document_count,
        "manifest_record_sha256": record.record_sha256,
        "qa_id": record.qa.qa_id,
        "qa_index": record.qa_index,
    }


def evaluate_manifest_record(
    record: ManifestRecord,
    backend: RecurrentBackend,
    config: SampleEvaluationConfig,
) -> dict[str, Any]:
    """Consume every manifest chunk and return one complete recurrent trajectory."""

    started = time.monotonic()
    config.validate()
    validate_manifest_record(record)
    if record.chunk_size != config.chunk_size:
        raise EvaluationRunnerError(
            f"record chunk_size={record.chunk_size} differs from runner {config.chunk_size}"
        )
    if hashlib.sha256(record.context.encode("utf-8")).hexdigest() != record.context_sha256:
        raise EvaluationRunnerError("context hash changed before evaluation")
    context_tokens = tuple(backend.encode_context(record.context))
    if len(context_tokens) != record.context_token_count:
        raise EvaluationRunnerError(
            "backend tokenizer count differs from manifest: "
            f"{len(context_tokens)} != {record.context_token_count}"
        )
    if token_ids_sha256(context_tokens) != record.context_token_ids_sha256:
        raise EvaluationRunnerError(
            "backend tokenizer IDs differ from the sealed manifest"
        )
    expected_chunks = math.ceil(len(context_tokens) / config.chunk_size)
    if len(record.chunks) != expected_chunks:
        raise EvaluationRunnerError("manifest does not cover the full dynamic chunk count")

    question = record.qa.question
    gold_answers = tuple(answer.text for answer in record.qa.gold_answers)
    supporting_doc_ids = frozenset(
        fact.document_id for fact in record.supporting_facts
    )
    memory = NO_MEMORY
    recalled_memory = NO_RECALLED_MEMORY
    history: list[MemoryRecord] = []
    trajectory: list[dict[str, Any]] = []
    processed_doc_ids: set[str] = set()
    query_statuses: list[str] = []
    effective_queries = 0
    retrieval_count = 0
    retrieval_empty_count = 0
    supporting_doc_hits = 0
    lexical_gold_hits = 0
    lexical_supporting_fact_hits = 0
    lookbacks: list[int] = []
    retrieved_steps: list[int] = []
    intermediate_valid_count = 0
    thinking_single_non_empty_count = 0
    update_single_non_empty_count = 0
    recall_protocol_valid_count = 0
    model_query_count = 0
    memory_truncated_count = 0

    for turn_index, chunk in enumerate(record.chunks):
        if chunk.chunk_index != turn_index:
            raise EvaluationRunnerError("chunk order changed during evaluation")
        token_slice = context_tokens[chunk.token_start : chunk.token_end]
        if len(token_slice) != chunk.token_end - chunk.token_start:
            raise EvaluationRunnerError("chunk token slice is incomplete")
        chunk_text = backend.decode_context_tokens(token_slice)
        processed_doc_ids.update(str(value) for value in chunk.document_ids)
        prompt = INTERMEDIATE_PROMPT.format(
            question=question,
            recalled_memory=recalled_memory,
            memory=memory,
            chunk=chunk_text,
        )
        generated = backend.generate_greedy(
            prompt,
            max_new_tokens=config.memory_max_tokens,
        )
        action = parse_intermediate_action(generated.text)
        intermediate_valid_count += int(action.format_valid)
        thinking_single_non_empty_count += int(
            action.thinking_occurrences.is_single_non_empty
        )
        update_single_non_empty_count += int(
            action.update_occurrences.is_single_non_empty
        )
        recall_protocol_valid_count += int(
            action.recall_occurrences.is_valid_optional_non_empty
        )
        model_query_count += int(action.recall is not None)
        memory_truncated_count += int(generated.reached_token_limit)
        status = _query_status(action)
        query_statuses.append(status)
        effective_query = resolve_callback_query(
            config.callback_mode,
            learned_query=action.recall,
            question=question,
        )
        retrieval = None
        if effective_query is not None:
            effective_queries += 1
            retrieval = retrieve_top1(effective_query, history)
        if retrieval is None:
            retrieval_empty_count += int(effective_query is not None)
            next_recalled_memory = NO_RECALLED_MEMORY
            retrieval_value = None
        else:
            retrieval_count += 1
            retrieved_steps.append(retrieval.record.step_id)
            lookbacks.append(turn_index - retrieval.record.step_id)
            source_doc_ids = tuple(str(value) for value in retrieval.record.source_doc_ids)
            support_hit = bool(supporting_doc_ids.intersection(source_doc_ids))
            supporting_doc_hits += int(support_hit)
            lexical_hit = any(
                word_recall(gold, retrieval.record.update_text) > 0.0
                for gold in gold_answers
            )
            lexical_gold_hits += int(lexical_hit)
            lexical_supporting_fact_hit = any(
                word_recall(fact.text, retrieval.record.update_text) > 0.0
                for fact in record.supporting_facts
            )
            lexical_supporting_fact_hits += int(lexical_supporting_fact_hit)
            next_recalled_memory = retrieval.record.update_text
            retrieval_value = {
                "lexical_gold_hit_proxy": lexical_hit,
                "lexical_supporting_fact_hit_proxy": lexical_supporting_fact_hit,
                "lookback_distance": turn_index - retrieval.record.step_id,
                "score": retrieval.score,
                "source_chunk_ids": [
                    str(value) for value in retrieval.record.source_chunk_ids
                ],
                "source_doc_ids": source_doc_ids,
                "step_id": retrieval.record.step_id,
                "supporting_doc_hit_proxy": support_hit,
                "update_text": retrieval.record.update_text,
            }
        memory_before = memory
        recalled_before = recalled_memory
        memory = action.update if action.update is not None else NO_MEMORY
        if action.update is not None:
            history.append(
                MemoryRecord(
                    step_id=turn_index,
                    update_text=action.update,
                    source_chunk_ids=(chunk.chunk_id,),
                    source_doc_ids=tuple(str(value) for value in chunk.document_ids),
                )
            )
        recalled_memory = next_recalled_memory
        trajectory.append(
            {
                "callback": {
                    "effective_query": effective_query,
                    "mode": config.callback_mode,
                    "model_query_status": status,
                    "retrieval": retrieval_value,
                },
                "chunk": {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "source_doc_ids": [str(value) for value in chunk.document_ids],
                    "supporting_fact_ids": [
                        str(value) for value in chunk.supporting_fact_ids
                    ],
                    "token_end": chunk.token_end,
                    "token_start": chunk.token_start,
                },
                "generation": {
                    "generated_token_sha256": generated.generated_token_sha256,
                    "generated_tokens": generated.generated_tokens,
                    "max_new_tokens": config.memory_max_tokens,
                    "reached_token_limit": generated.reached_token_limit,
                },
                "kind": "memory",
                "memory_after": memory,
                "memory_before": memory_before,
                "parsed": {
                    "format_valid": action.format_valid,
                    "recall": _occurrences_dict(action.recall_occurrences),
                    "thinking": _occurrences_dict(action.thinking_occurrences),
                    "update": _occurrences_dict(action.update_occurrences),
                },
                "prompt": prompt,
                "raw_output": generated.text,
                "recalled_memory_before": recalled_before,
                "recalled_memory_next": recalled_memory,
                "turn_index": turn_index,
            }
        )

    expected_doc_ids = {document.document_id for document in record.documents}
    if processed_doc_ids != expected_doc_ids:
        missing = sorted(expected_doc_ids - processed_doc_ids)
        extra = sorted(processed_doc_ids - expected_doc_ids)
        raise EvaluationRunnerError(
            f"processed document provenance mismatch; missing={missing}, extra={extra}"
        )
    final_prompt = FINAL_PROMPT.format(
        question=question,
        recalled_memory=recalled_memory,
        memory=memory,
    )
    final_generation = backend.generate_greedy(
        final_prompt,
        max_new_tokens=config.final_max_tokens,
    )
    final_action = parse_final_action(final_generation.text)
    evaluated = evaluate_output(final_generation.text, gold_answers)
    trajectory.append(
        {
            "generation": {
                "generated_token_sha256": final_generation.generated_token_sha256,
                "generated_tokens": final_generation.generated_tokens,
                "max_new_tokens": config.final_max_tokens,
                "reached_token_limit": final_generation.reached_token_limit,
            },
            "kind": "final",
            "memory": memory,
            "parsed": {
                "boxed_answer": final_action.boxed_answer,
                "boxed_count": final_action.boxed_count,
                "format_valid": final_action.format_valid,
            },
            "prompt": final_prompt,
            "raw_output": final_generation.text,
            "recalled_memory": recalled_memory,
            "turn_index": len(record.chunks),
        }
    )
    status_counts = Counter(query_statuses)
    duplicate_retrieval_count = len(retrieved_steps) - len(set(retrieved_steps))
    chunk_count = len(record.chunks)
    status_rates = {
        name: status_counts.get(name, 0) / chunk_count
        for name in ("valid", "empty", "malformed", "duplicate", "absent")
    }
    scores = evaluated.scores
    metrics = {
        "answer": evaluated.to_dict(),
        "callback": {
            "average_lookback_distance": (
                sum(lookbacks) / len(lookbacks) if lookbacks else None
            ),
            "duplicate_retrieved_state_count": duplicate_retrieval_count,
            "duplicate_retrieved_state_rate": (
                duplicate_retrieval_count / retrieval_count
                if retrieval_count
                else None
            ),
            "effective_query_count": effective_queries,
            "effective_query_rate": effective_queries / chunk_count,
            "eligible_steps": chunk_count,
            "lexical_gold_hit_count": lexical_gold_hits,
            "lexical_gold_hit_rate": (
                lexical_gold_hits / retrieval_count if retrieval_count else None
            ),
            "lexical_supporting_fact_hit_count": lexical_supporting_fact_hits,
            "lexical_supporting_fact_hit_rate": (
                lexical_supporting_fact_hits / retrieval_count
                if retrieval_count
                else None
            ),
            "model_query_count": model_query_count,
            "model_query_rate": model_query_count / chunk_count,
            "model_query_status_counts": {
                name: status_counts.get(name, 0)
                for name in ("valid", "empty", "malformed", "duplicate", "absent")
            },
            "model_query_status_rates": status_rates,
            "retrieval_count": retrieval_count,
            "retrieval_empty_count": retrieval_empty_count,
            "retrieval_empty_rate": (
                retrieval_empty_count / effective_queries
                if effective_queries
                else None
            ),
            "retrieval_rate": retrieval_count / chunk_count,
            "retrieval_success_rate": (
                retrieval_count / effective_queries if effective_queries else None
            ),
            "supporting_doc_hit_count": supporting_doc_hits,
            "supporting_doc_hit_rate": (
                supporting_doc_hits / retrieval_count if retrieval_count else None
            ),
        },
        "format": {
            "final_valid": final_action.format_valid,
            "intermediate_valid_count": intermediate_valid_count,
            "intermediate_valid_rate": intermediate_valid_count / chunk_count,
            "recall_protocol_valid_count": recall_protocol_valid_count,
            "recall_protocol_valid_rate": recall_protocol_valid_count / chunk_count,
            "thinking_single_non_empty_count": thinking_single_non_empty_count,
            "thinking_single_non_empty_rate": (
                thinking_single_non_empty_count / chunk_count
            ),
            "update_single_non_empty_count": update_single_non_empty_count,
            "update_single_non_empty_rate": update_single_non_empty_count / chunk_count,
        },
        "scores": scores.to_dict(),
        "truncation": {
            "final_reached_token_limit": final_generation.reached_token_limit,
            "memory_reached_token_limit_count": memory_truncated_count,
            "memory_reached_token_limit_rate": memory_truncated_count / chunk_count,
        },
    }
    return {
        **_input_identity(record),
        "callback_mode": config.callback_mode,
        "context_token_count": record.context_token_count,
        "elapsed_seconds": time.monotonic() - started,
        "gold_answers": list(gold_answers),
        "metrics": metrics,
        "parsed_answer": evaluated.extraction.answer,
        "processed_chunk_count": chunk_count,
        "processed_doc_count": len(processed_doc_ids),
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "raw_final_output": final_generation.text,
        "status": "success",
        "trajectory": trajectory,
    }


def _failure_record(
    record: ManifestRecord,
    config: SampleEvaluationConfig,
    *,
    stage: str,
    error_type: str,
    message: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        **_input_identity(record),
        "callback_mode": config.callback_mode,
        "elapsed_seconds": elapsed_seconds,
        "error": {
            "message": str(message),
            "stage": stage,
            "type": error_type,
        },
        "gold_answers": [answer.text for answer in record.qa.gold_answers],
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "status": "failed",
        "trajectory": [],
    }


def evaluate_manifest_record_safely(
    record: ManifestRecord,
    backend: RecurrentBackend,
    config: SampleEvaluationConfig,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        return evaluate_manifest_record(record, backend, config)
    except Exception as exc:
        return _failure_record(
            record,
            config,
            stage="sample_evaluation",
            error_type=type(exc).__name__,
            message=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )


def _worker_main(
    spec: ModelLoadSpec,
    input_queue: Any,
    output_queue: Any,
) -> None:
    try:
        backend = _load_backend(spec)
        output_queue.put({"kind": "ready", "metadata": dict(backend.metadata)})
    except BaseException as exc:
        output_queue.put(
            {
                "error": {"message": str(exc), "type": type(exc).__name__},
                "kind": "load_error",
            }
        )
        return
    while True:
        command = input_queue.get()
        if command.get("kind") == "stop":
            return
        if command.get("kind") != "evaluate":
            output_queue.put(
                {
                    "error": {
                        "message": "unknown worker command",
                        "type": "EvaluationRunnerError",
                    },
                    "kind": "worker_error",
                }
            )
            continue
        record = command["record"]
        config = command["config"]
        try:
            result = evaluate_manifest_record_safely(record, backend, config)
            output_queue.put({"kind": "result", "result": result})
        except BaseException as exc:
            output_queue.put(
                {
                    "error": {"message": str(exc), "type": type(exc).__name__},
                    "kind": "worker_error",
                }
            )


@dataclass(slots=True)
class _WorkerHandle:
    process: Any
    input_queue: Any
    output_queue: Any
    metadata: Mapping[str, Any]


def _terminate_worker(handle: _WorkerHandle | None) -> None:
    if handle is None:
        return
    process = handle.process
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=5)
    for worker_queue in (handle.input_queue, handle.output_queue):
        try:
            worker_queue.close()
            worker_queue.join_thread()
        except (AttributeError, ValueError):
            pass


def _wait_for_message(process: Any, output_queue: Any, deadline: float) -> Any:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerTimeoutError("worker deadline expired")
        try:
            return output_queue.get(timeout=min(0.1, remaining))
        except queue.Empty:
            if not process.is_alive():
                raise WorkerExitedError(
                    f"model worker exited early with code {process.exitcode}"
                )


def _start_worker(
    spec: ModelLoadSpec,
    *,
    load_deadline: float,
    start_method: str,
) -> _WorkerHandle:
    context = multiprocessing.get_context(start_method)
    input_queue = context.Queue()
    output_queue = context.Queue()
    process = context.Process(
        target=_worker_main,
        args=(spec, input_queue, output_queue),
        daemon=True,
    )
    process.start()
    provisional = _WorkerHandle(process, input_queue, output_queue, {})
    try:
        message = _wait_for_message(process, output_queue, load_deadline)
        if message.get("kind") != "ready":
            error = message.get("error", {})
            raise EvaluationRunnerError(
                f"model load failed: {error.get('type')}: {error.get('message')}"
            )
        provisional.metadata = message["metadata"]
        return provisional
    except Exception:
        _terminate_worker(provisional)
        raise


def _canonical_jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(value) + b"\n" for value in values)


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _aggregate_summary(
    results: Sequence[Mapping[str, Any]],
    *,
    callback_mode: str,
) -> dict[str, Any]:
    successes = [result for result in results if result["status"] == "success"]
    failures = [result for result in results if result["status"] == "failed"]

    def success_mean(path: tuple[str, ...]) -> float | None:
        values = []
        for result in successes:
            value: Any = result
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        return sum(values) / len(values) if values else None

    def all_input_mean(path: tuple[str, ...]) -> float:
        # Failed inputs contribute zero so infrastructure errors cannot inflate QA scores.
        total = 0.0
        for result in successes:
            value: Any = result
            for key in path:
                value = value[key]
            total += float(value)
        return total / len(results)

    error_counts = Counter(
        result["error"]["type"] for result in failures
    )
    return {
        "callback_mode": callback_mode,
        "failure_count": len(failures),
        "failure_types": dict(sorted(error_counts.items())),
        "metric_denominators": {
            "answer_all_inputs": len(results),
            "behavior_success_only": len(successes),
        },
        "metrics": {
            "exact_match": all_input_mean(("metrics", "scores", "exact_match")),
            "fallback_extraction_success_rate": all_input_mean(
                ("metrics", "answer", "extraction", "fallback_success")
            ),
            "strict_boxed_success_rate": all_input_mean(
                ("metrics", "answer", "extraction", "strict_boxed_success")
            ),
            "substring_exact_match": all_input_mean(
                ("metrics", "scores", "substring_exact_match")
            ),
            "token_f1": all_input_mean(("metrics", "scores", "token_f1")),
            "success_only": {
                "duplicate_retrieved_state_rate": success_mean(
                    ("metrics", "callback", "duplicate_retrieved_state_rate")
                ),
                "effective_query_rate": success_mean(
                    ("metrics", "callback", "effective_query_rate")
                ),
                "final_format_valid_rate": success_mean(
                    ("metrics", "format", "final_valid")
                ),
                "final_reached_token_limit_rate": success_mean(
                    ("metrics", "truncation", "final_reached_token_limit")
                ),
                "intermediate_format_valid_rate": success_mean(
                    ("metrics", "format", "intermediate_valid_rate")
                ),
                "lexical_gold_hit_rate": success_mean(
                    ("metrics", "callback", "lexical_gold_hit_rate")
                ),
                "lexical_supporting_fact_hit_rate": success_mean(
                    ("metrics", "callback", "lexical_supporting_fact_hit_rate")
                ),
                "memory_reached_token_limit_rate": success_mean(
                    ("metrics", "truncation", "memory_reached_token_limit_rate")
                ),
                "model_query_rate": success_mean(
                    ("metrics", "callback", "model_query_rate")
                ),
                "model_query_absent_rate": success_mean(
                    ("metrics", "callback", "model_query_status_rates", "absent")
                ),
                "model_query_duplicate_rate": success_mean(
                    ("metrics", "callback", "model_query_status_rates", "duplicate")
                ),
                "model_query_empty_rate": success_mean(
                    ("metrics", "callback", "model_query_status_rates", "empty")
                ),
                "model_query_malformed_rate": success_mean(
                    ("metrics", "callback", "model_query_status_rates", "malformed")
                ),
                "model_query_valid_rate": success_mean(
                    ("metrics", "callback", "model_query_status_rates", "valid")
                ),
                "recall_protocol_valid_rate": success_mean(
                    ("metrics", "format", "recall_protocol_valid_rate")
                ),
                "retrieval_empty_rate": success_mean(
                    ("metrics", "callback", "retrieval_empty_rate")
                ),
                "retrieval_rate": success_mean(
                    ("metrics", "callback", "retrieval_rate")
                ),
                "supporting_doc_hit_rate": success_mean(
                    ("metrics", "callback", "supporting_doc_hit_rate")
                ),
                "thinking_single_non_empty_rate": success_mean(
                    ("metrics", "format", "thinking_single_non_empty_rate")
                ),
                "update_single_non_empty_rate": success_mean(
                    ("metrics", "format", "update_single_non_empty_rate")
                ),
            },
        },
        "sample_count": len(results),
        "status": "completed" if not failures else "completed_with_failures",
        "success_count": len(successes),
    }


def _publish_results(
    destination: Path,
    *,
    results: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite eval output {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        results_payload = _canonical_jsonl_bytes(results)
        results_path = staging / RESULTS_FILENAME
        results_path.write_bytes(results_payload)
        config_sha = _write_canonical_json(staging / RUN_CONFIG_FILENAME, run_config)
        summary_sha = _write_canonical_json(staging / SUMMARY_FILENAME, summary)
        marker_payload = {
            "failure_count": summary["failure_count"],
            "kind": RUN_KIND,
            "results_sha256": _sha256_bytes(results_payload),
            "run_config_sha256": config_sha,
            "schema_version": RUN_SCHEMA_VERSION,
            "status": summary["status"],
            "success_count": summary["success_count"],
            "summary_sha256": summary_sha,
        }
        marker = {
            **marker_payload,
            "marker_sha256": _sha256_bytes(canonical_json_bytes(marker_payload)),
        }
        _write_canonical_json(staging / COMPLETION_FILENAME, marker)
        _validate_result_directory(staging, expected_qa_ids=[r["qa_id"] for r in results])
        os.replace(staging, destination)
        _validate_result_directory(
            destination,
            expected_qa_ids=[r["qa_id"] for r in results],
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _read_canonical_json(path: Path) -> Mapping[str, Any]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise EvaluationRunnerError(f"{path.name} is not newline-terminated")
    value = json.loads(payload)
    if canonical_json_bytes(value) + b"\n" != payload:
        raise EvaluationRunnerError(f"{path.name} is not canonical JSON")
    if not isinstance(value, Mapping):
        raise EvaluationRunnerError(f"{path.name} must contain an object")
    return value


def _validate_result_directory(
    directory: Path,
    *,
    expected_qa_ids: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    expected_files = {
        COMPLETION_FILENAME,
        RESULTS_FILENAME,
        RUN_CONFIG_FILENAME,
        SUMMARY_FILENAME,
    }
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise EvaluationRunnerError("eval output file inventory mismatch")
    marker = _read_canonical_json(directory / COMPLETION_FILENAME)
    marker_hash = marker.get("marker_sha256")
    marker_payload = {k: v for k, v in marker.items() if k != "marker_sha256"}
    if marker_hash != _sha256_bytes(canonical_json_bytes(marker_payload)):
        raise EvaluationRunnerError("completion marker self-hash mismatch")
    results_payload = (directory / RESULTS_FILENAME).read_bytes()
    if marker["results_sha256"] != _sha256_bytes(results_payload):
        raise EvaluationRunnerError("results hash mismatch")
    results = [json.loads(line) for line in results_payload.splitlines()]
    if _canonical_jsonl_bytes(results) != results_payload:
        raise EvaluationRunnerError("results JSONL is not canonical")
    config_payload = (directory / RUN_CONFIG_FILENAME).read_bytes()
    summary_payload = (directory / SUMMARY_FILENAME).read_bytes()
    if marker["run_config_sha256"] != _sha256_bytes(config_payload):
        raise EvaluationRunnerError("run config hash mismatch")
    if marker["summary_sha256"] != _sha256_bytes(summary_payload):
        raise EvaluationRunnerError("summary hash mismatch")
    summary = _read_canonical_json(directory / SUMMARY_FILENAME)
    _read_canonical_json(directory / RUN_CONFIG_FILENAME)
    if len(results) != summary["sample_count"]:
        raise EvaluationRunnerError("result count differs from summary")
    if expected_qa_ids is not None:
        if [result["qa_id"] for result in results] != list(expected_qa_ids):
            raise EvaluationRunnerError("result QA order differs from input")
    return summary


def run_evaluation_task(
    records: Sequence[ManifestRecord],
    *,
    output_dir: str | os.PathLike[str],
    backend_spec: ModelLoadSpec,
    sample_config: SampleEvaluationConfig,
    model_load_timeout_s: float,
    sample_timeout_s: float,
    task_timeout_s: float,
    input_metadata: Mapping[str, Any] | None = None,
    start_method: str = "spawn",
) -> Mapping[str, Any]:
    """Evaluate every input exactly once with load/sample/task hard deadlines."""

    records = tuple(records)
    if not records:
        raise ValueError("records must not be empty")
    for record in records:
        validate_manifest_record(record)
    backend_spec.validate()
    sample_config.validate()
    qa_ids = [record.qa.qa_id for record in records]
    if len(set(qa_ids)) != len(qa_ids):
        raise EvaluationRunnerError("evaluation inputs contain duplicate QA IDs")
    if any(record.chunk_size != sample_config.chunk_size for record in records):
        raise EvaluationRunnerError("input chunk_size differs from sample config")
    tokenizer_contracts = {
        (record.metadata.tokenizer_name, record.metadata.tokenizer_revision)
        for record in records
    }
    if len(tokenizer_contracts) != 1:
        raise EvaluationRunnerError("evaluation inputs mix tokenizer identities")
    if backend_spec.kind == "transformers":
        data_tokenizer_id, data_tokenizer_revision = next(iter(tokenizer_contracts))
        if (
            data_tokenizer_id != backend_spec.tokenizer_id
            or data_tokenizer_revision != backend_spec.tokenizer_revision
        ):
            raise EvaluationRunnerError(
                "eval model tokenizer identity differs from the data manifest"
            )
    load_timeout = _require_positive_number(model_load_timeout_s, "model_load_timeout_s")
    sample_timeout = _require_positive_number(sample_timeout_s, "sample_timeout_s")
    task_timeout = _require_positive_number(task_timeout_s, "task_timeout_s")
    destination = Path(output_dir).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"refusing to append or overwrite {destination}")
    task_started = time.monotonic()
    task_deadline = task_started + task_timeout
    results: list[Mapping[str, Any]] = []
    handle: _WorkerHandle | None = None
    model_metadata: Mapping[str, Any] | None = None
    load_failure: tuple[str, str] | None = None
    try:
        try:
            handle = _start_worker(
                backend_spec,
                load_deadline=min(task_deadline, time.monotonic() + load_timeout),
                start_method=start_method,
            )
            model_metadata = handle.metadata
        except Exception as exc:
            load_failure = (type(exc).__name__, str(exc))

        for record in records:
            if load_failure is not None:
                results.append(
                    _failure_record(
                        record,
                        sample_config,
                        stage="model_load",
                        error_type=load_failure[0],
                        message=load_failure[1],
                        elapsed_seconds=0.0,
                    )
                )
                continue
            if time.monotonic() >= task_deadline:
                results.append(
                    _failure_record(
                        record,
                        sample_config,
                        stage="task_timeout",
                        error_type="WorkerTimeoutError",
                        message="full task deadline expired",
                        elapsed_seconds=0.0,
                    )
                )
                continue
            if handle is None:
                try:
                    handle = _start_worker(
                        backend_spec,
                        load_deadline=min(
                            task_deadline,
                            time.monotonic() + load_timeout,
                        ),
                        start_method=start_method,
                    )
                    model_metadata = handle.metadata
                except Exception as exc:
                    results.append(
                        _failure_record(
                            record,
                            sample_config,
                            stage="model_reload",
                            error_type=type(exc).__name__,
                            message=str(exc),
                            elapsed_seconds=0.0,
                        )
                    )
                    continue
            sample_started = time.monotonic()
            handle.input_queue.put(
                {"config": sample_config, "kind": "evaluate", "record": record}
            )
            deadline = min(task_deadline, sample_started + sample_timeout)
            try:
                message = _wait_for_message(
                    handle.process,
                    handle.output_queue,
                    deadline,
                )
                if message.get("kind") == "result":
                    result = message["result"]
                else:
                    error = message.get("error", {})
                    result = _failure_record(
                        record,
                        sample_config,
                        stage="worker_protocol",
                        error_type=error.get("type", "EvaluationRunnerError"),
                        message=error.get("message", "worker protocol error"),
                        elapsed_seconds=time.monotonic() - sample_started,
                    )
                results.append(result)
            except Exception as exc:
                stage = (
                    "task_timeout"
                    if time.monotonic() >= task_deadline
                    else "sample_timeout_or_worker_exit"
                )
                results.append(
                    _failure_record(
                        record,
                        sample_config,
                        stage=stage,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        elapsed_seconds=time.monotonic() - sample_started,
                    )
                )
                _terminate_worker(handle)
                handle = None
    finally:
        if handle is not None:
            try:
                handle.input_queue.put({"kind": "stop"})
                handle.process.join(timeout=5)
            finally:
                _terminate_worker(handle)
    summary = _aggregate_summary(results, callback_mode=sample_config.callback_mode)
    run_config = {
        "backend": backend_spec.public_dict(),
        "callback_mode": sample_config.callback_mode,
        "input": dict(input_metadata or {}),
        "kind": RUN_KIND,
        "model_metadata": dict(model_metadata or {}),
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
        "qa_ids": qa_ids,
        "sample": sample_config.to_dict(),
        "schema_version": RUN_SCHEMA_VERSION,
        "timeouts": {
            "model_load_seconds": load_timeout,
            "sample_seconds": sample_timeout,
            "task_seconds": task_timeout,
        },
    }
    _publish_results(
        destination,
        results=results,
        run_config=run_config,
        summary=summary,
    )
    return _validate_result_directory(
        destination,
        expected_qa_ids=qa_ids,
    )


def load_eval_records(
    bundle_dir: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
    variant: int,
    sample_count: int | None = None,
) -> tuple[tuple[ManifestRecord, ...], Mapping[str, Any]]:
    bundle = Path(bundle_dir).expanduser().resolve(strict=True)
    manifest = validate_artifact_bundle(bundle)
    if (
        not isinstance(expected_manifest_sha256, str)
        or _SHA256.fullmatch(expected_manifest_sha256) is None
    ):
        raise ValueError("expected_manifest_sha256 must be a lowercase SHA-256")
    if not hmac.compare_digest(
        manifest["manifest_sha256"],
        expected_manifest_sha256,
    ):
        raise EvaluationRunnerError(
            "eval bundle manifest SHA-256 differs from the locked run contract"
        )
    if manifest["mode"] != "eval":
        raise EvaluationRunnerError("input data bundle is not an eval bundle")
    if isinstance(variant, bool) or not isinstance(variant, int) or variant <= 0:
        raise ValueError("variant must be a positive document count")
    sidecar_name = f"eval_{variant}.sidecar.jsonl"
    artifacts = manifest["artifacts"]
    if sidecar_name not in artifacts:
        raise EvaluationRunnerError(f"bundle has no {variant}-document sidecar")
    artifact = artifacts[sidecar_name]
    values = validate_canonical_jsonl(
        (bundle / sidecar_name).read_bytes(),
        artifact["sha256"],
    )
    records = tuple(manifest_record_from_dict(value) for value in values)
    if any(record.document_count != variant for record in records):
        raise EvaluationRunnerError("sidecar document count differs from variant")
    if sample_count is not None:
        _require_positive_int(sample_count, "sample_count")
        if sample_count > len(records):
            raise ValueError("sample_count exceeds fixed manifest length")
        if manifest["profile"] == "formal" and sample_count not in {32, 64}:
            raise ValueError("formal eval sample_count must be 32 or 64")
        records = records[:sample_count]
    return records, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-manifest recurrent evaluation on one Transformers GPU",
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--variant", type=int, choices=(200, 800), required=True)
    parser.add_argument("--sample-count", type=int, choices=(32, 64), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--callback-mode", choices=sorted(CALLBACK_MODES), required=True)
    parser.add_argument("--artifact-kind", choices=("base", "adapter", "merged"), required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--template-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-max-tokens", type=int, default=768)
    parser.add_argument("--final-max-tokens", type=int, default=512)
    parser.add_argument("--model-load-timeout", type=float, default=1800.0)
    parser.add_argument("--sample-timeout", type=float, default=1800.0)
    parser.add_argument("--task-timeout", type=float, default=86400.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records, data_manifest = load_eval_records(
        args.bundle_dir,
        expected_manifest_sha256=args.expected_manifest_sha256,
        variant=args.variant,
        sample_count=args.sample_count,
    )
    if data_manifest["profile"] != "formal":
        raise EvaluationRunnerError("CLI refuses non-formal data bundles")
    chunk_size = data_manifest["contract"]["chunk_size"]
    if chunk_size != 5000:
        raise EvaluationRunnerError("formal eval chunk_size must be 5000")
    tokenizer_id = args.tokenizer or args.base_model
    tokenizer_revision = args.tokenizer_revision or args.revision
    backend_spec = ModelLoadSpec(
        artifact_kind=args.artifact_kind,
        base_model_id=args.base_model,
        revision=args.revision,
        artifact_path=args.artifact_path,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        template_revision=args.template_revision,
        attention_implementation=args.attention_implementation,
        device=args.device,
        seed=args.seed,
    )
    sample_config = SampleEvaluationConfig(
        callback_mode=args.callback_mode,
        chunk_size=chunk_size,
        memory_max_tokens=args.memory_max_tokens,
        final_max_tokens=args.final_max_tokens,
    )
    summary = run_evaluation_task(
        records,
        output_dir=args.output_dir,
        backend_spec=backend_spec,
        sample_config=sample_config,
        model_load_timeout_s=args.model_load_timeout,
        sample_timeout_s=args.sample_timeout,
        task_timeout_s=args.task_timeout,
        input_metadata={
            "bundle_manifest_sha256": data_manifest["manifest_sha256"],
            "bundle_path": str(Path(args.bundle_dir).resolve()),
            "dataset": data_manifest["dataset"],
            "sample_count": args.sample_count,
            "variant": args.variant,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BackendGeneration",
    "COMPLETION_FILENAME",
    "EvaluationRunnerError",
    "FINAL_PROMPT",
    "INTERMEDIATE_PROMPT",
    "ModelLoadSpec",
    "PROMPT_TEMPLATE_SHA256",
    "PROMPT_TEMPLATE_REVISION",
    "RESULTS_FILENAME",
    "RUN_CONFIG_FILENAME",
    "SUMMARY_FILENAME",
    "SampleEvaluationConfig",
    "ScriptedBackend",
    "TransformersBackend",
    "evaluate_manifest_record",
    "evaluate_manifest_record_safely",
    "load_eval_records",
    "load_transformers_backend",
    "run_evaluation_task",
    "main",
]
