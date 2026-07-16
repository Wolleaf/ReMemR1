# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail-closed LoRA contracts for the Qwen3.5 text language model.

This module intentionally has no module-level Torch or PEFT import.  Target
resolution and contract tests can therefore run with small duck-typed model
fixtures, while the training process imports PEFT only when it injects LoRA.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Pattern, Sequence, Tuple, Type


LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_BIAS = "none"

# Qwen3.5 text blocks contain Gated DeltaNet, full-attention, and dense MLP
# projections.  Unknown linear leaf names fail closed instead of silently
# changing the adapter surface when the upstream architecture changes.
DEFAULT_LORA_ALLOW_PATTERNS: Tuple[str, ...] = (
    r"(?:^|\.)(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$",
    r"(?:^|\.)(?:q_proj|k_proj|v_proj|o_proj)$",
    r"(?:^|\.)(?:gate_proj|up_proj|down_proj)$",
)

# These patterns apply to complete module paths.  In particular, ``out_proj``
# remains allowed for Gated DeltaNet while output heads are denied explicitly.
DEFAULT_LORA_DENY_PATTERNS: Tuple[str, ...] = (
    r"(?:^|\.)(?:lm_head|output_layer|output_projection|embed_out)(?:\.|$)",
    r"(?:^|\.)(?:embed_tokens|embeddings?|word_embeddings?|wte|token_embeddings?)(?:\.|$)",
    r"(?:^|\.)(?:visual|vision|vision_model|vision_tower|vision_encoder|image_encoder|image_tower|patch_embed)(?:\.|$)",
    (
        r"(?:^|\.)(?:projector|mm_projector|multi_modal_projector|vision_projector|"
        r"visual_projection|modality_projection)(?:\.|$)"
    ),
    r"(?:^|\.)(?:mtp|mtp_head|multi_token_prediction|multi_token_predictor|nextn|next_n)(?:\.|$)",
    r"(?:^|\.)(?:classifier|score|auxiliary_head|draft_model|medusa)(?:\.|$)",
)

PatternLike = str | Pattern[str]
LinearPredicate = Callable[[Any], bool]


class LoraContractError(ValueError):
    """Raised when a model does not satisfy the reproduction LoRA contract."""


@dataclass(frozen=True)
class LoraTargetManifest:
    """Canonical, serializable list of exact modules receiving LoRA adapters."""

    target_modules: Tuple[str, ...]
    sha256: str
    text_model_prefix: Optional[str] = None

    @property
    def module_names(self) -> Tuple[str, ...]:
        """Alias used by logging and checkpoint metadata."""

        return self.target_modules

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "target_modules": list(self.target_modules),
            "sha256": self.sha256,
            "text_model_prefix": self.text_model_prefix,
        }


@dataclass(frozen=True)
class TrainableParameterManifest:
    """Deterministic metadata and state hashes for trainable parameters."""

    names: Tuple[str, ...]
    tensor_count: int
    trainable_numel: int
    total_numel: int
    trainable_ratio: float
    manifest_sha256: str
    state_sha256: str

    @property
    def sha256(self) -> str:
        """The state hash used to compare seeded adapter initialization."""

        return self.state_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "names": list(self.names),
            "tensor_count": self.tensor_count,
            "trainable_numel": self.trainable_numel,
            "total_numel": self.total_numel,
            "trainable_ratio": self.trainable_ratio,
            "manifest_sha256": self.manifest_sha256,
            "state_sha256": self.state_sha256,
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _target_manifest_hash(target_modules: Sequence[str]) -> str:
    payload = {"schema_version": 1, "target_modules": list(target_modules)}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _compile_patterns(patterns: Sequence[PatternLike], label: str) -> Tuple[Pattern[str], ...]:
    if not patterns:
        raise LoraContractError(f"LoRA {label} patterns must not be empty")

    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, flags=re.IGNORECASE) if isinstance(pattern, str) else pattern)
        except re.error as exc:
            raise LoraContractError(f"Invalid LoRA {label} pattern {pattern!r}: {exc}") from exc
    return tuple(compiled)


def _matches_any(name: str, patterns: Sequence[Pattern[str]]) -> bool:
    return any(pattern.search(name) is not None for pattern in patterns)


def _default_linear_predicate(module: Any) -> bool:
    marker = getattr(module, "_lora_contract_is_linear", None)
    if marker is not None:
        return bool(marker)

    try:
        from torch import nn
    except ImportError:
        nn = None

    if nn is not None and isinstance(module, nn.Linear):
        return True

    # Test doubles commonly use Linear/FakeLinear/StubLinear without importing
    # Torch.  The explicit marker above is preferred for ambiguous class names.
    class_name = type(module).__name__.casefold()
    if not class_name.endswith("linear"):
        return False
    stem = class_name[: -len("linear")]
    return not stem.endswith(("non", "not"))


def _normalize_prefix(text_model_prefix: Optional[str]) -> Optional[str]:
    if text_model_prefix is None:
        return None
    prefix = text_model_prefix.strip(".")
    if not prefix:
        raise LoraContractError("text_model_prefix must be a non-empty module path")
    return prefix


def _in_scope(name: str, prefix: Optional[str]) -> bool:
    return prefix is None or name == prefix or name.startswith(f"{prefix}.")


def _canonical_target_names(target_modules: Sequence[str]) -> Tuple[str, ...]:
    names = tuple(str(name).strip(".") for name in target_modules)
    if not names or any(not name for name in names):
        raise LoraContractError("LoRA target manifest must contain at least one named module")
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise LoraContractError(f"Duplicate LoRA target modules: {duplicates}")
    return tuple(sorted(names))


def validate_lora_target_manifest(
    target_modules: Sequence[str],
    *,
    allow_patterns: Sequence[PatternLike] = DEFAULT_LORA_ALLOW_PATTERNS,
    deny_patterns: Sequence[PatternLike] = DEFAULT_LORA_DENY_PATTERNS,
) -> Tuple[str, ...]:
    """Validate and canonicalize exact LoRA target module names.

    Every target must match the allow list and no target may match the deny
    list.  Both lists are regular expressions evaluated against full paths.
    """

    names = _canonical_target_names(target_modules)
    allowed = _compile_patterns(allow_patterns, "allow")
    denied = _compile_patterns(deny_patterns, "deny")

    denied_names = [name for name in names if _matches_any(name, denied)]
    if denied_names:
        raise LoraContractError(f"Denied modules present in LoRA target manifest: {denied_names}")

    unknown_names = [name for name in names if not _matches_any(name, allowed)]
    if unknown_names:
        raise LoraContractError(f"Unknown linear modules in LoRA target manifest: {unknown_names}")
    return names


def resolve_lora_target_manifest(
    model: Any,
    *,
    text_model_prefix: Optional[str] = None,
    allow_patterns: Sequence[PatternLike] = DEFAULT_LORA_ALLOW_PATTERNS,
    deny_patterns: Sequence[PatternLike] = DEFAULT_LORA_DENY_PATTERNS,
    is_linear_module: Optional[LinearPredicate] = None,
) -> LoraTargetManifest:
    """Resolve text-LM ``all-linear`` into a fail-closed exact manifest.

    Denied paths are deliberately ignored, but any other linear module whose
    leaf name is unknown raises.  This catches upstream architecture changes
    before PEFT silently trains a different set of modules.
    """

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError("model must provide named_modules()")

    prefix = _normalize_prefix(text_model_prefix)
    allowed = _compile_patterns(allow_patterns, "allow")
    denied = _compile_patterns(deny_patterns, "deny")
    predicate = is_linear_module or _default_linear_predicate

    seen_prefix = prefix is None
    target_names = []
    unknown_names = []
    seen_names = set()
    for raw_name, module in named_modules():
        name = str(raw_name).strip(".")
        if name in seen_names:
            raise LoraContractError(f"named_modules() returned duplicate module path: {name!r}")
        seen_names.add(name)

        if prefix is not None and name == prefix:
            seen_prefix = True
        if not name or not _in_scope(name, prefix) or not predicate(module):
            continue
        if _matches_any(name, denied):
            continue
        if _matches_any(name, allowed):
            target_names.append(name)
        else:
            unknown_names.append(name)

    if not seen_prefix:
        raise LoraContractError(f"Unknown text_model_prefix: {prefix!r}")
    if unknown_names:
        raise LoraContractError(f"Unknown linear modules in text language model: {sorted(unknown_names)}")

    canonical_names = validate_lora_target_manifest(
        target_names,
        allow_patterns=allow_patterns,
        deny_patterns=deny_patterns,
    )
    return LoraTargetManifest(
        target_modules=canonical_names,
        sha256=_target_manifest_hash(canonical_names),
        text_model_prefix=prefix,
    )


def _coerce_manifest_target_names(manifest: LoraTargetManifest | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(manifest, LoraTargetManifest):
        expected_hash = _target_manifest_hash(manifest.target_modules)
        if manifest.sha256 != expected_hash:
            raise LoraContractError(
                f"LoRA target manifest hash mismatch: expected {expected_hash}, got {manifest.sha256}"
            )
        return validate_lora_target_manifest(manifest.target_modules)
    return validate_lora_target_manifest(manifest)


def build_lora_config(
    manifest: LoraTargetManifest | Sequence[str],
    *,
    r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    bias: str = LORA_BIAS,
    lora_config_cls: Optional[Type[Any]] = None,
    task_type: Any = None,
) -> Any:
    """Build the fixed causal-LM PEFT config, importing PEFT lazily."""

    target_modules = _coerce_manifest_target_names(manifest)
    if r != LORA_R or lora_alpha != LORA_ALPHA or float(lora_dropout) != LORA_DROPOUT or bias != LORA_BIAS:
        raise LoraContractError(
            "Reproduction LoRA config must be r=32, lora_alpha=64, lora_dropout=0.0, bias='none'"
        )

    if lora_config_cls is None or task_type is None:
        try:
            from peft import LoraConfig, TaskType
        except ImportError as exc:
            raise ImportError("PEFT is required only when constructing or injecting a LoRA adapter") from exc
        lora_config_cls = lora_config_cls or LoraConfig
        task_type = TaskType.CAUSAL_LM if task_type is None else task_type

    return lora_config_cls(
        task_type=task_type,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=list(target_modules),
    )


def inject_lora_adapter(
    model: Any,
    manifest: LoraTargetManifest | Sequence[str],
    *,
    get_peft_model_fn: Optional[Callable[[Any, Any], Any]] = None,
    lora_config_cls: Optional[Type[Any]] = None,
    task_type: Any = None,
) -> Any:
    """Inject the validated adapter using lazy, replaceable PEFT callables."""

    config = build_lora_config(
        manifest,
        lora_config_cls=lora_config_cls,
        task_type=task_type,
    )
    if get_peft_model_fn is None:
        try:
            from peft import get_peft_model
        except ImportError as exc:
            raise ImportError("PEFT is required only when constructing or injecting a LoRA adapter") from exc
        get_peft_model_fn = get_peft_model
    return get_peft_model_fn(model, config)


def assert_injected_lora_targets(
    model: Any,
    manifest: LoraTargetManifest | Sequence[str],
) -> Tuple[str, ...]:
    """Verify that PEFT adapted exactly the resolved module manifest."""

    expected = _coerce_manifest_target_names(manifest)
    targeted = getattr(model, "targeted_module_names", None)
    if targeted is None:
        targeted = getattr(getattr(model, "base_model", None), "targeted_module_names", None)
    if targeted is None:
        raise LoraContractError("Injected PEFT model exposes no targeted_module_names")
    actual = tuple(sorted(str(name) for name in targeted))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise LoraContractError(
            "Injected LoRA targets differ from the resolved manifest; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return actual


def _parameter_numel(parameter: Any) -> int:
    numel = getattr(parameter, "numel", None)
    if callable(numel):
        return int(numel())
    if numel is not None:
        return int(numel)

    shape = getattr(parameter, "shape", None)
    if shape is None:
        raise TypeError(f"Parameter {type(parameter).__name__} must provide numel() or shape")
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _parameter_metadata(name: str, parameter: Any) -> dict[str, Any]:
    shape = getattr(parameter, "shape", ())
    return {
        "name": name,
        "shape": [int(dimension) for dimension in shape],
        "dtype": str(getattr(parameter, "dtype", type(parameter).__name__)),
        "numel": _parameter_numel(parameter),
    }


def _tensor_bytes(parameter: Any) -> bytes:
    value = parameter
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    contiguous = getattr(value, "contiguous", None)
    if callable(contiguous):
        value = contiguous()

    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        try:
            return numpy().tobytes(order="C")
        except (RuntimeError, TypeError):
            # NumPy does not expose every Torch dtype (notably bfloat16).
            try:
                import torch

                return value.view(torch.uint8).numpy().tobytes(order="C")
            except (ImportError, RuntimeError, TypeError, AttributeError):
                pass

    tobytes = getattr(value, "tobytes", None)
    if callable(tobytes):
        return tobytes()

    raw_value = getattr(value, "value", None)
    if raw_value is not None and raw_value is not value:
        return _canonical_json_bytes(raw_value)

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _canonical_json_bytes(tolist())

    # Some structural fakes intentionally have no storage.  Metadata is still
    # hashed by the caller, so their hash remains deterministic.
    return b""


def collect_trainable_parameter_manifest(
    model: Any,
    *,
    require_nonempty: bool = True,
) -> TrainableParameterManifest:
    """Collect trainable names/counts plus deterministic manifest/state hashes."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model must provide named_parameters()")

    all_parameters = [(str(name), parameter) for name, parameter in named_parameters()]
    names = [name for name, _ in all_parameters]
    if len(names) != len(set(names)):
        raise LoraContractError("named_parameters() returned duplicate parameter names")

    all_parameters.sort(key=lambda item: item[0])
    trainable = [
        (name, parameter)
        for name, parameter in all_parameters
        if bool(getattr(parameter, "requires_grad", False))
    ]
    if require_nonempty and not trainable:
        raise LoraContractError("Model has no trainable parameters after LoRA injection")

    metadata = [_parameter_metadata(name, parameter) for name, parameter in trainable]
    manifest_hash = hashlib.sha256(
        _canonical_json_bytes({"schema_version": 1, "parameters": metadata})
    ).hexdigest()

    state_hasher = hashlib.sha256()
    state_hasher.update(_canonical_json_bytes({"schema_version": 1, "parameters": metadata}))
    for name, parameter in trainable:
        data = _tensor_bytes(parameter)
        state_hasher.update(len(name).to_bytes(8, "big"))
        state_hasher.update(name.encode("utf-8"))
        state_hasher.update(len(data).to_bytes(8, "big"))
        state_hasher.update(data)

    trainable_numel = sum(item["numel"] for item in metadata)
    total_numel = sum(_parameter_numel(parameter) for _, parameter in all_parameters)
    ratio = trainable_numel / total_numel if total_numel else 0.0
    return TrainableParameterManifest(
        names=tuple(name for name, _ in trainable),
        tensor_count=len(trainable),
        trainable_numel=trainable_numel,
        total_numel=total_numel,
        trainable_ratio=ratio,
        manifest_sha256=manifest_hash,
        state_sha256=state_hasher.hexdigest(),
    )


def assert_only_lora_parameters_trainable(
    model: Any,
    *,
    name_pattern: PatternLike = r"(?:^|\.)(?:lora_[AB]|lora_embedding_[AB])(?:\.|$)",
) -> TrainableParameterManifest:
    """Assert that the base is frozen and only PEFT LoRA tensors are trainable."""

    manifest = collect_trainable_parameter_manifest(model)
    pattern = re.compile(name_pattern, flags=re.IGNORECASE) if isinstance(name_pattern, str) else name_pattern
    unexpected = [name for name in manifest.names if pattern.search(name) is None]
    if unexpected:
        raise LoraContractError(f"Non-LoRA parameters are trainable: {unexpected}")
    return manifest


def _iter_optimizer_parameters(parameters_or_optimizer: Any) -> Iterable[Any]:
    param_groups = getattr(parameters_or_optimizer, "param_groups", None)
    if param_groups is not None:
        source = param_groups
    elif isinstance(parameters_or_optimizer, Mapping):
        source = [parameters_or_optimizer]
    else:
        source = parameters_or_optimizer

    for item in source:
        if isinstance(item, Mapping):
            if "params" not in item:
                raise LoraContractError("Optimizer parameter group is missing 'params'")
            yield from item["params"]
        else:
            yield item


def assert_no_frozen_optimizer_parameters(parameters_or_optimizer: Any) -> Tuple[Any, ...]:
    """Return flattened optimizer parameters after rejecting frozen tensors."""

    parameters = tuple(_iter_optimizer_parameters(parameters_or_optimizer))
    if not parameters:
        raise LoraContractError("Optimizer parameter list must not be empty")
    frozen_indices = [
        index
        for index, parameter in enumerate(parameters)
        if not bool(getattr(parameter, "requires_grad", False))
    ]
    if frozen_indices:
        raise LoraContractError(f"Optimizer contains frozen parameters at flattened indices {frozen_indices}")
    return parameters


def trainable_optimizer_parameters(model: Any) -> Tuple[Any, ...]:
    """Return only trainable model parameters in deterministic name order."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model must provide named_parameters()")
    trainable = sorted(
        (
            (str(name), parameter)
            for name, parameter in named_parameters()
            if bool(getattr(parameter, "requires_grad", False))
        ),
        key=lambda item: item[0],
    )
    return assert_no_frozen_optimizer_parameters(parameter for _, parameter in trainable)


# Readable aliases for call sites that use "get"/"apply" terminology.
get_trainable_optimizer_parameters = trainable_optimizer_parameters
get_peft_model_with_lora_contract = inject_lora_adapter


__all__ = [
    "DEFAULT_LORA_ALLOW_PATTERNS",
    "DEFAULT_LORA_DENY_PATTERNS",
    "LORA_ALPHA",
    "LORA_BIAS",
    "LORA_DROPOUT",
    "LORA_R",
    "LoraContractError",
    "LoraTargetManifest",
    "TrainableParameterManifest",
    "assert_no_frozen_optimizer_parameters",
    "assert_injected_lora_targets",
    "assert_only_lora_parameters_trainable",
    "build_lora_config",
    "collect_trainable_parameter_manifest",
    "get_peft_model_with_lora_contract",
    "get_trainable_optimizer_parameters",
    "inject_lora_adapter",
    "resolve_lora_target_manifest",
    "trainable_optimizer_parameters",
    "validate_lora_target_manifest",
]
