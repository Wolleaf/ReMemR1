"""Build and verify the durable CPU-to-GPU cloud handoff state.

This helper deliberately keeps network access in ``resolve-assets`` only.
Every later command either consumes locally cached files or re-verifies bytes
that were already published on the shared volume.
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.reproduction.prefetch_assets import (  # noqa: E402
    AssetIntegrityError,
    load_asset_manifest,
    seal_manifest,
    validate_asset_manifest,
    verify_cached_file,
)
from scripts.reproduction.verify_environment import (  # noqa: E402
    load_environment_lock,
)
from taskutils.data_synthesis.reproduction_builder import (  # noqa: E402
    TrainManifestContract,
    build_artifact_bundle,
    build_train_records,
    parse_source_record,
    validate_artifact_bundle,
)
from taskutils.data_synthesis.reproduction_manifest import (  # noqa: E402
    EvalManifestContract,
    ManifestMetadata,
    ManifestValidationError,
)


SCHEMA_VERSION = 3
HANDOFF_STATUS = "cpu_ready"
EXPERIMENT_PROFILE_ID = "rtx5090-32g-qwen35-2b-v1"
SEED = 42
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TRACKED_ASSET_MANIFEST = REPOSITORY_ROOT / "environment" / "reproduction-assets.json"
_ENVIRONMENT_LOCK = REPOSITORY_ROOT / "environment" / "reproduction-cu130.lock.json"
_GATE_SOURCE_REVISION_PREFIX = "rememr1-gate-source-v1"
_GATE_REPEAT_CANDIDATES = tuple(range(32, 113, 4))
_LENGTH_STRESS_SOURCE_REVISION_PREFIX = "rememr1-length-stress-source-v1"
_LENGTH_STRESS_REPEAT_CANDIDATES = tuple(range(80, 181, 4))
_GATE_CONTRACT = TrainManifestContract(
    qa_count=20,
    document_count=16,
    chunk_size=1024,
    max_chunks=2,
    min_context_tokens=1025,
    max_context_tokens=2048,
)
_G1_EVAL_CONTRACT = EvalManifestContract(
    qa_count=2,
    prefix_document_count=8,
    pool_document_count=16,
    chunk_size=1024,
)
_LENGTH_STRESS_CONTRACT = TrainManifestContract(
    qa_count=2,
    document_count=200,
    chunk_size=5000,
    max_chunks=6,
    min_context_tokens=25_001,
    max_context_tokens=30_000,
)
_ACTIVE_ASSET_IDS = {
    "2wikimultihopqa-source",
    "byted-hotpotqa-formal",
    "hotpotqa-source",
    "qwen35-08b-model-tokenizer",
    "qwen35-2b-model-tokenizer",
}
_GATE_CONFIG_IDS = {
    "g0_qwen35_08b",
    "g1_qwen35_2b_resume2",
    "g1_qwen35_2b_step1",
}
_TRAINING_SOURCE_IDS = {
    "b20_qwen35_2b_5090",
    "b40_qwen35_2b_5090",
    "b60_qwen35_2b_5090",
    "b80_qwen35_2b_5090",
    "b_pilot_qwen35_2b_5090",
    "c20_qwen35_2b_5090",
    "c40_qwen35_2b_5090",
    "c60_qwen35_2b_5090",
    "c80_qwen35_2b_5090",
    "c_pilot_qwen35_2b_5090",
    "g2_length_stress_qwen35_2b_5090",
    "g2a_qwen35_2b_5090",
    "g2b_qwen35_2b_5090_resume5",
    "g2b_qwen35_2b_5090_step1",
}
_EVAL_CONFIG_IDS = {
    "eval40_qwen35_2b_5090",
    "eval80_qwen35_2b_5090",
}
_CONFIG_IDS = (
    _GATE_CONFIG_IDS
    | _EVAL_CONFIG_IDS
    | {
        f"{source}_{profile}"
        for source in _TRAINING_SOURCE_IDS
        for profile in ("r0", "r1")
    }
)
_CONFIG_NAMES = {f"{config_id}.yaml" for config_id in _CONFIG_IDS}
_CONFIG_TREE_NAMES = _CONFIG_NAMES | {"index.json"}
_PLACEHOLDER_DIGESTS = {character * 64 for character in "0123"}
_SHA256_CACHE: contextvars.ContextVar[
    dict[tuple[str, int, int, int, int, int], str] | None
] = contextvars.ContextVar("rememr1_cloud_sha256_cache", default=None)


class CloudStateError(RuntimeError):
    """Raised when durable handoff state cannot be proven complete."""


@dataclass(frozen=True, slots=True)
class BundleSpec:
    keys: tuple[str, ...]
    relative_path: str
    mode: str
    profile: str
    dataset: str
    source_asset: str | None
    source_file: str | None
    source_split: str | None
    tokenizer_asset: str
    gate_split: str | None = None
    length_stress_split: str | None = None


_BUNDLE_SPECS = (
    BundleSpec(
        ("gates", "g0", "train"),
        "gates/g0/train",
        "train",
        "fixture",
        "hotpotqa",
        None,
        None,
        None,
        "qwen35-08b-model-tokenizer",
        "train",
    ),
    BundleSpec(
        ("gates", "g0", "validation"),
        "gates/g0/validation",
        "train",
        "fixture",
        "hotpotqa",
        None,
        None,
        None,
        "qwen35-08b-model-tokenizer",
        "validation",
    ),
    BundleSpec(
        ("gates", "g1", "train"),
        "gates/g1/train",
        "train",
        "fixture",
        "hotpotqa",
        None,
        None,
        None,
        "qwen35-2b-model-tokenizer",
        "train",
    ),
    BundleSpec(
        ("gates", "g1", "validation"),
        "gates/g1/validation",
        "train",
        "fixture",
        "hotpotqa",
        None,
        None,
        None,
        "qwen35-2b-model-tokenizer",
        "validation",
    ),
    BundleSpec(
        ("gates", "g1", "eval"),
        "gates/g1/eval",
        "eval",
        "fixture",
        "hotpotqa",
        None,
        None,
        None,
        "qwen35-2b-model-tokenizer",
        "validation",
    ),
    BundleSpec(
        ("formal", "train"),
        "formal/train",
        "train",
        "formal",
        "hotpotqa",
        "byted-hotpotqa-formal",
        "hotpotqa_train_32k.parquet",
        None,
        "qwen35-2b-model-tokenizer",
    ),
    BundleSpec(
        ("formal", "validation"),
        "formal/validation",
        "train",
        "formal",
        "hotpotqa",
        "byted-hotpotqa-formal",
        "hotpotqa_dev.parquet",
        None,
        "qwen35-2b-model-tokenizer",
    ),
    BundleSpec(
        ("formal", "eval", "hotpotqa"),
        "formal/eval/hotpotqa",
        "eval",
        "formal",
        "hotpotqa",
        "hotpotqa-source",
        "hotpotqa/dev.jsonl",
        None,
        "qwen35-2b-model-tokenizer",
    ),
    BundleSpec(
        ("formal", "eval", "2wikimultihopqa"),
        "formal/eval/2wikimultihopqa",
        "eval",
        "formal",
        "2wikimultihopqa",
        "2wikimultihopqa-source",
        "2wikimultihopqa/dev.jsonl",
        None,
        "qwen35-2b-model-tokenizer",
    ),
    BundleSpec(
        ("capacity", "length_stress", "train"),
        "capacity/length-stress/train",
        "train",
        "fixture",
        "hotpotqa",
        None,
        None,
        None,
        "qwen35-2b-model-tokenizer",
        length_stress_split="train",
    ),
    BundleSpec(
        ("capacity", "length_stress", "validation"),
        "capacity/length-stress/validation",
        "train",
        "fixture",
        "hotpotqa",
        None,
        None,
        None,
        "qwen35-2b-model-tokenizer",
        length_stress_split="validation",
    ),
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CloudStateError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("ascii")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    path = path.resolve(strict=True)
    before = path.stat()
    key = (
        str(path),
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    cache = _SHA256_CACHE.get()
    if cache is not None and key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != key[1:]:
        raise CloudStateError(f"file changed while hashing: {path}")
    value = digest.hexdigest()
    if cache is not None:
        cache[key] = value
    return value


def _absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(Path(value).expanduser()))


def _resolve_without_symlinks(
    value: str | Path,
    label: str,
    *,
    strict: bool = True,
) -> Path:
    candidate = _absolute_path(value)
    for component in [*reversed(candidate.parents), candidate]:
        if component.is_symlink():
            raise CloudStateError(f"{label} contains a symlink component: {component}")
    try:
        return candidate.resolve(strict=strict)
    except OSError as exc:
        raise CloudStateError(f"cannot resolve {label}: {candidate}: {exc}") from exc


def _require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CloudStateError(f"{label} escaped {root}: {path}") from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CloudStateError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CloudStateError(f"non-finite JSON number: {value}")
            ),
        )
    except CloudStateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudStateError(f"cannot read JSON {source}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    output = _resolve_without_symlinks(path, "atomic output", strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = _resolve_without_symlinks(output, "atomic output", strict=False)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _canonical_bytes(value) + b"\n")


def _require_commit(value: str, path: str = "commit") -> str:
    if not isinstance(value, str) or not _HEX40.fullmatch(value) or value == "0" * 40:
        raise CloudStateError(f"{path} must be a non-zero lowercase 40-character commit SHA")
    return value


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise CloudStateError(f"{path} must be a lowercase SHA-256")
    return value


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CloudStateError(f"{path} must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CloudStateError(
            f"{path} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _member(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _repo_file_path(value: Any) -> str | None:
    path = _member(value, "path")
    if path is None:
        path = _member(value, "rfilename")
    return path if isinstance(path, str) else None


def _resolved_hf_file(value: Any, expected_path: str) -> dict[str, Any]:
    observed_path = _repo_file_path(value)
    if observed_path != expected_path:
        raise CloudStateError(
            f"Hugging Face metadata path mismatch: expected {expected_path!r}, "
            f"observed {observed_path!r}"
        )
    size = _member(value, "size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise CloudStateError(f"Hugging Face metadata lacks a valid size for {expected_path!r}")
    lfs = _member(value, "lfs")
    if lfs is None:
        raise CloudStateError(f"{expected_path!r} is not backed by verifiable LFS metadata")
    sha256 = _member(lfs, "sha256")
    _require_sha256(sha256, f"Hugging Face LFS metadata for {expected_path!r}")
    lfs_size = _member(lfs, "size")
    if lfs_size is not None and lfs_size != size:
        raise CloudStateError(
            f"Hugging Face file/LFS size mismatch for {expected_path!r}: "
            f"{size} != {lfs_size}"
        )
    return {"path": expected_path, "sha256": sha256, "size": size}


def _new_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise CloudStateError("huggingface-hub is required to resolve asset metadata") from exc
    return HfApi()


def _query_hf_paths(
    api: Any,
    asset: Mapping[str, Any],
    paths: Sequence[str],
    *,
    retries: int,
    sleep: Callable[[float], None],
) -> dict[str, dict[str, Any]]:
    if retries < 1:
        raise CloudStateError("metadata retries must be at least one")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = api.get_paths_info(
                repo_id=asset["repo_id"],
                paths=list(paths),
                revision=asset["revision"],
                repo_type=asset["repo_type"],
                expand=True,
            )
            by_path: dict[str, Any] = {}
            for item in response:
                item_path = _repo_file_path(item)
                if item_path is None or item_path in by_path:
                    raise CloudStateError("Hugging Face returned missing or duplicate path metadata")
                by_path[item_path] = item
            if set(by_path) != set(paths):
                raise CloudStateError(
                    "Hugging Face metadata response does not exactly match requested paths"
                )
            return {path: _resolved_hf_file(by_path[path], path) for path in paths}
        except Exception as exc:  # Hub exception classes vary across pinned releases.
            last_error = exc
            if attempt < retries:
                sleep(min(2 ** (attempt - 1), 8))
    assert last_error is not None
    raise CloudStateError(
        f"cannot resolve Hugging Face metadata for {asset['asset_id']!r} "
        f"after {retries} attempts: {type(last_error).__name__}: {last_error}"
    ) from last_error


def resolve_assets(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    api: Any | None = None,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Resolve blocked LFS metadata and atomically publish a runtime manifest."""

    source = _resolve_without_symlinks(manifest_path, "tracked asset manifest")
    output = _resolve_without_symlinks(
        output_path,
        "runtime asset manifest output",
        strict=False,
    )
    if output == source:
        raise CloudStateError("runtime asset manifest must not overwrite the tracked manifest")
    manifest = load_asset_manifest(source)
    resolved = json.loads(json.dumps(manifest))
    unresolved_count = 0
    for asset in resolved["assets"]:
        paths = [
            file_spec["path"]
            for file_spec in asset["files"]
            if file_spec.get("metadata_status") == "METADATA_UNAVAILABLE"
        ]
        if not paths:
            continue
        unresolved_count += len(paths)
        if api is None:
            api = _new_hf_api()
        replacements = _query_hf_paths(
            api,
            asset,
            paths,
            retries=retries,
            sleep=sleep,
        )
        asset["files"] = [
            replacements.get(file_spec["path"], file_spec)
            for file_spec in asset["files"]
        ]
        asset.pop("metadata_resolution", None)
        asset.pop("training_gate", None)
    sealed = seal_manifest(resolved)
    validate_asset_manifest(sealed)
    _require_resolved_manifest(sealed)
    _atomic_write_json(output, sealed)
    return {
        "manifest_sha256": sealed["manifest_sha256"],
        "output": str(output),
        "resolved_file_count": unresolved_count,
        "status": "resolved",
    }


def _require_resolved_manifest(manifest: Mapping[str, Any]) -> None:
    validate_asset_manifest(manifest)
    unresolved = [
        f"{asset['asset_id']}:{file_spec['path']}"
        for asset in manifest["assets"]
        for file_spec in asset["files"]
        if file_spec.get("metadata_status") == "METADATA_UNAVAILABLE"
    ]
    gated = [
        asset["asset_id"]
        for asset in manifest["assets"]
        if "metadata_resolution" in asset or "training_gate" in asset
    ]
    if unresolved or gated:
        raise CloudStateError(
            f"asset manifest remains unresolved; files={unresolved}, gated_assets={gated}"
        )


def _validate_runtime_against_tracked(
    runtime: Mapping[str, Any],
    tracked: Mapping[str, Any],
) -> None:
    _require_resolved_manifest(runtime)
    validate_asset_manifest(tracked)
    runtime_assets = {asset["asset_id"]: asset for asset in runtime["assets"]}
    tracked_assets = {asset["asset_id"]: asset for asset in tracked["assets"]}
    if set(tracked_assets) != _ACTIVE_ASSET_IDS:
        raise CloudStateError(
            "tracked asset IDs do not match the active 5090/2B experiment profile"
        )
    if set(runtime_assets) != set(tracked_assets):
        raise CloudStateError("runtime asset IDs differ from the tracked manifest")
    for asset_id, baseline in tracked_assets.items():
        observed = runtime_assets[asset_id]
        for key in ("asset_id", "kind", "repo_id", "repo_type", "revision"):
            if observed[key] != baseline[key]:
                raise CloudStateError(f"runtime asset {asset_id!r} changed {key}")
        baseline_files = {item["path"]: item for item in baseline["files"]}
        observed_files = {item["path"]: item for item in observed["files"]}
        if set(observed_files) != set(baseline_files):
            raise CloudStateError(f"runtime asset {asset_id!r} changed its file set")
        for path, baseline_file in baseline_files.items():
            observed_file = observed_files[path]
            if baseline_file.get("metadata_status") == "METADATA_UNAVAILABLE":
                if set(observed_file) != {"path", "sha256", "size"}:
                    raise CloudStateError(
                        f"runtime metadata for {asset_id}:{path} is not fully resolved"
                    )
            elif observed_file != baseline_file:
                raise CloudStateError(f"runtime file pin changed for {asset_id}:{path}")


def _asset_by_id(manifest: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    matches = [asset for asset in manifest["assets"] if asset["asset_id"] == asset_id]
    if len(matches) != 1:
        raise CloudStateError(f"required asset {asset_id!r} is missing or duplicated")
    return matches[0]


def _file_spec(asset: Mapping[str, Any], filename: str) -> Mapping[str, Any]:
    matches = [item for item in asset["files"] if item["path"] == filename]
    if len(matches) != 1:
        raise CloudStateError(
            f"required file {filename!r} is missing or duplicated in {asset['asset_id']!r}"
        )
    if matches[0].get("metadata_status") == "METADATA_UNAVAILABLE":
        raise CloudStateError(f"required file metadata remains unresolved: {filename}")
    return matches[0]


def _hf_downloader() -> Callable[..., str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise CloudStateError("huggingface-hub is required to locate cached sources") from exc
    return hf_hub_download


def _local_tokenizer_loader(name: str, revision: str) -> Callable[[str], Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise CloudStateError("transformers is required to load cached tokenizers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise CloudStateError("gate/formal data requires a cached fast tokenizer")

    def encode(text: str) -> Any:
        return tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

    return encode


def _synthetic_gate_records(split: str, repeat_count: int) -> list[dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise CloudStateError(f"unsupported synthetic gate split: {split}")
    records: list[dict[str, Any]] = []
    for qa_index in range(_GATE_CONTRACT.qa_count):
        contexts = []
        for document_index in range(2):
            title = f"Gate {split} document {qa_index:02d}-{document_index}"
            marker = f"{split}-{qa_index:02d}-{document_index}"
            sentence = f"Evidence {marker}. " + ("memory " * repeat_count) + "End."
            contexts.append(
                {
                    "document_id": f"gate-{split}-{qa_index:02d}-{document_index}",
                    "sentences": [sentence],
                    "title": title,
                }
            )
        records.append(
            {
                "_id": f"gate-{split}-qa-{qa_index:02d}",
                "answers": [f"answer-{split}-{qa_index:02d}"],
                "context": contexts,
                "question": f"What is the gate {split} answer {qa_index:02d}?",
                "supporting_facts": [[contexts[0]["title"], 0]],
            }
        )
    return records


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(record) + b"\n" for record in records)


def _probe_gate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    repeat_count: int,
    tokenizer_name: str,
    tokenizer_revision: str,
    encode: Callable[[str], Any],
) -> tuple[int, int]:
    payload = _jsonl_bytes(records)
    source_sha256 = hashlib.sha256(payload).hexdigest()
    examples = tuple(
        parse_source_record(record, dataset="hotpotqa", source_index=index)
        for index, record in enumerate(records)
    )
    metadata = ManifestMetadata(
        source_name="hotpotqa",
        source_revision=f"{_GATE_SOURCE_REVISION_PREFIX}-r{repeat_count:03d}",
        source_sha256=source_sha256,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        seed=SEED,
    )
    built = build_train_records(
        examples,
        metadata=metadata,
        encode=encode,
        contract=_GATE_CONTRACT,
    )
    counts = [record.context_token_count for record in built]
    return min(counts), max(counts)


def _select_gate_sources(
    tokenizers: Mapping[str, tuple[str, str, Callable[[str], Any]]],
) -> tuple[int, dict[str, bytes], dict[str, dict[str, dict[str, int]]]]:
    last_errors: list[str] = []
    for repeat_count in _GATE_REPEAT_CANDIDATES:
        payloads: dict[str, bytes] = {}
        observations: dict[str, dict[str, dict[str, int]]] = {}
        try:
            for split in ("train", "validation"):
                records = _synthetic_gate_records(split, repeat_count)
                payloads[split] = _jsonl_bytes(records)
                for gate, (name, revision, encode) in tokenizers.items():
                    minimum, maximum = _probe_gate_records(
                        records,
                        repeat_count=repeat_count,
                        tokenizer_name=name,
                        tokenizer_revision=revision,
                        encode=encode,
                    )
                    observations.setdefault(gate, {})[split] = {
                        "max_context_tokens": maximum,
                        "min_context_tokens": minimum,
                    }
            return repeat_count, payloads, observations
        except (ManifestValidationError, ValueError) as exc:
            last_errors.append(f"r{repeat_count}: {exc}")
    detail = "; ".join(last_errors[-3:])
    raise CloudStateError(
        "no finite synthetic gate source candidate satisfies both pinned tokenizers; "
        f"last failures: {detail}"
    )


def _publish_gate_sources(
    data_root: Path,
    payloads: Mapping[str, bytes],
) -> dict[str, Path]:
    source_root = data_root / "gates" / "source"
    result: dict[str, Path] = {}
    for split in ("train", "validation"):
        path = source_root / f"{split}.jsonl"
        payload = payloads[split]
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise CloudStateError(
                    f"existing synthetic gate source differs from deterministic output: {path}"
                )
        else:
            _atomic_write_bytes(path, payload)
        result[split] = path.resolve(strict=True)
    return result


def _synthetic_length_stress_records(
    split: str,
    repeat_count: int,
) -> list[dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise CloudStateError(f"unsupported length-stress split: {split}")
    records: list[dict[str, Any]] = []
    for qa_index in range(100):
        contexts = []
        for document_index in range(2):
            marker = f"{split}-{qa_index:03d}-{document_index}"
            title = f"Length stress {split} document {qa_index:03d}-{document_index}"
            contexts.append(
                {
                    "document_id": f"length-stress-{marker}",
                    "sentences": [
                        f"Capacity evidence {marker}. "
                        + ("memory " * repeat_count)
                        + "End."
                    ],
                    "title": title,
                }
            )
        records.append(
            {
                "_id": f"length-stress-{split}-qa-{qa_index:03d}",
                "answers": [f"answer-{split}-{qa_index:03d}"],
                "context": contexts,
                "question": f"What is the fixed {split} capacity answer {qa_index:03d}?",
                "supporting_facts": [[contexts[0]["title"], 0]],
            }
        )
    return records


def _probe_length_stress_records(
    records: Sequence[Mapping[str, Any]],
    *,
    repeat_count: int,
    tokenizer_name: str,
    tokenizer_revision: str,
    encode: Callable[[str], Any],
) -> tuple[int, int]:
    payload = _jsonl_bytes(records)
    examples = tuple(
        parse_source_record(record, dataset="hotpotqa", source_index=index)
        for index, record in enumerate(records)
    )
    metadata = ManifestMetadata(
        source_name="hotpotqa",
        source_revision=(
            f"{_LENGTH_STRESS_SOURCE_REVISION_PREFIX}-r{repeat_count:03d}"
        ),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        seed=SEED,
    )
    built = build_train_records(
        examples,
        metadata=metadata,
        encode=encode,
        contract=_LENGTH_STRESS_CONTRACT,
    )
    counts = [record.context_token_count for record in built]
    return min(counts), max(counts)


def _select_length_stress_sources(
    tokenizer: tuple[str, str, Callable[[str], Any]],
) -> tuple[int, dict[str, bytes], dict[str, dict[str, int]]]:
    tokenizer_name, tokenizer_revision, encode = tokenizer
    last_errors: list[str] = []
    for repeat_count in _LENGTH_STRESS_REPEAT_CANDIDATES:
        payloads: dict[str, bytes] = {}
        observations: dict[str, dict[str, int]] = {}
        try:
            for split in ("train", "validation"):
                records = _synthetic_length_stress_records(split, repeat_count)
                payloads[split] = _jsonl_bytes(records)
                minimum, maximum = _probe_length_stress_records(
                    records,
                    repeat_count=repeat_count,
                    tokenizer_name=tokenizer_name,
                    tokenizer_revision=tokenizer_revision,
                    encode=encode,
                )
                observations[split] = {
                    "max_context_tokens": maximum,
                    "min_context_tokens": minimum,
                }
            return repeat_count, payloads, observations
        except (ManifestValidationError, ValueError) as exc:
            last_errors.append(f"r{repeat_count}: {exc}")
    detail = "; ".join(last_errors[-3:])
    raise CloudStateError(
        "no finite non-scientific length-stress candidate satisfies the pinned "
        f"2B tokenizer; last failures: {detail}"
    )


def _publish_length_stress_sources(
    data_root: Path,
    payloads: Mapping[str, bytes],
) -> dict[str, Path]:
    source_root = data_root / "capacity" / "length-stress" / "source"
    result: dict[str, Path] = {}
    for split in ("train", "validation"):
        path = source_root / f"{split}.jsonl"
        payload = payloads[split]
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise CloudStateError(
                    "existing length-stress source differs from deterministic output: "
                    f"{path}"
                )
        else:
            _atomic_write_bytes(path, payload)
        result[split] = path.resolve(strict=True)
    return result


def _insert_nested(root: dict[str, Any], keys: Sequence[str], value: Any) -> None:
    cursor = root
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise CloudStateError(f"cannot insert nested bundle key {'.'.join(keys)}")
        cursor = child
    cursor[keys[-1]] = value


def _download_local_source(
    asset: Mapping[str, Any],
    filename: str,
    *,
    cache_dir: Path,
    downloader: Callable[..., str],
) -> Path:
    spec = _file_spec(asset, filename)
    try:
        cached = downloader(
            repo_id=asset["repo_id"],
            filename=filename,
            repo_type=asset["repo_type"],
            revision=asset["revision"],
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    except Exception as exc:
        raise CloudStateError(
            f"cached source is unavailable for {asset['asset_id']}:{filename}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    path = Path(cached).expanduser().resolve(strict=True)
    try:
        verify_cached_file(path, spec)
    except (AssetIntegrityError, OSError, KeyError) as exc:
        raise CloudStateError(
            f"cached source failed integrity verification for {asset['asset_id']}:{filename}: {exc}"
        ) from exc
    return path


def _probe_formal_parquet(path: Path, *, dataset: str) -> None:
    """Fail before a full read when the pinned formal source lost structure."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CloudStateError("pyarrow is required for the formal source probe") from exc
    try:
        parquet = pq.ParquetFile(path)
        if parquet.num_row_groups < 1:
            raise CloudStateError(f"formal parquet has no row groups: {path}")
        rows = parquet.read_row_group(0).slice(0, 3).to_pylist()
        if not rows:
            raise CloudStateError(f"formal parquet has no records: {path}")
        for index, row in enumerate(rows):
            parse_source_record(row, dataset=dataset, source_index=index)
    except CloudStateError:
        raise
    except Exception as exc:
        raise CloudStateError(
            f"formal parquet lacks structured context/supporting facts: {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _expected_bundle_contract(spec: BundleSpec) -> Mapping[str, Any]:
    if spec.length_stress_split is not None:
        return _LENGTH_STRESS_CONTRACT.to_dict()
    if spec.profile == "fixture":
        if spec.mode == "train":
            return _GATE_CONTRACT.to_dict()
        return {
            "chunk_size": _G1_EVAL_CONTRACT.chunk_size,
            "pool_document_count": _G1_EVAL_CONTRACT.pool_document_count,
            "prefix_document_count": _G1_EVAL_CONTRACT.prefix_document_count,
            "qa_count": _G1_EVAL_CONTRACT.qa_count,
        }
    if spec.mode == "train":
        return TrainManifestContract().to_dict()
    contract = EvalManifestContract()
    return {
        "chunk_size": contract.chunk_size,
        "pool_document_count": contract.pool_document_count,
        "prefix_document_count": contract.prefix_document_count,
        "qa_count": contract.qa_count,
    }


def _validate_bundle_identity(
    manifest: Mapping[str, Any],
    spec: BundleSpec,
    *,
    source_path: Path,
    source_revision: str,
    tokenizer_name: str,
    tokenizer_revision: str,
) -> None:
    expected = {
        "contract": _expected_bundle_contract(spec),
        "dataset": spec.dataset,
        "mode": spec.mode,
        "profile": spec.profile,
        "seed": SEED,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CloudStateError(
                f"bundle {spec.relative_path} has unexpected {key}: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    source = _require_mapping(manifest.get("source"), "bundle.source")
    tokenizer = _require_mapping(manifest.get("tokenizer"), "bundle.tokenizer")
    if source.get("revision") != source_revision:
        raise CloudStateError(f"bundle {spec.relative_path} source revision mismatch")
    if source.get("sha256") != _sha256_file(source_path):
        raise CloudStateError(f"bundle {spec.relative_path} source hash mismatch")
    recorded_source = Path(str(source.get("path", ""))).expanduser().resolve(strict=False)
    if recorded_source != source_path.resolve():
        raise CloudStateError(f"bundle {spec.relative_path} source path mismatch")
    if source.get("split") != spec.source_split:
        raise CloudStateError(f"bundle {spec.relative_path} source split mismatch")
    if tokenizer.get("name") != tokenizer_name or tokenizer.get("revision") != tokenizer_revision:
        raise CloudStateError(f"bundle {spec.relative_path} tokenizer identity mismatch")


def _bundle_summary(path: Path, manifest: Mapping[str, Any], action: str) -> dict[str, Any]:
    return {
        "action": action,
        "dataset": manifest["dataset"],
        "manifest_sha256": manifest["manifest_sha256"],
        "mode": manifest["mode"],
        "path": str(path),
        "profile": manifest["profile"],
        "source_sha256": manifest["source"]["sha256"],
        "source_revision": manifest["source"]["revision"],
        "tokenizer_name": manifest["tokenizer"]["name"],
        "tokenizer_revision": manifest["tokenizer"]["revision"],
    }


def build_data(
    manifest_path: str | Path,
    cache_dir: str | Path,
    data_root: str | Path,
    *,
    downloader: Callable[..., str] | None = None,
    tokenizer_loader: Callable[[str, str], Callable[[str], Any]] = _local_tokenizer_loader,
    builder: Callable[..., Mapping[str, Any]] = build_artifact_bundle,
    validator: Callable[[str | os.PathLike[str]], Mapping[str, Any]] = validate_artifact_bundle,
    tracked_manifest_path: str | Path | None = _TRACKED_ASSET_MANIFEST,
) -> dict[str, Any]:
    """Build or strictly validate all gate, capacity, and formal bundles offline."""

    runtime = load_asset_manifest(manifest_path)
    _require_resolved_manifest(runtime)
    if tracked_manifest_path is not None:
        tracked = load_asset_manifest(tracked_manifest_path)
        _validate_runtime_against_tracked(runtime, tracked)
    cache = _resolve_without_symlinks(cache_dir, "Hugging Face cache")
    root_candidate = _resolve_without_symlinks(data_root, "data root", strict=False)
    root_candidate.mkdir(parents=True, exist_ok=True)
    root = _resolve_without_symlinks(root_candidate, "data root")
    if not root.is_dir():
        raise CloudStateError(f"data root is not a directory: {root}")
    downloader = downloader or _hf_downloader()
    tokenizer_cache: dict[str, tuple[str, str, Callable[[str], Any]]] = {}
    for gate, asset_id in (
        ("g0", "qwen35-08b-model-tokenizer"),
        ("g1", "qwen35-2b-model-tokenizer"),
    ):
        asset = _asset_by_id(runtime, asset_id)
        tokenizer_cache[gate] = (
            asset["repo_id"],
            asset["revision"],
            tokenizer_loader(asset["repo_id"], asset["revision"]),
        )
    repeat_count, gate_payloads, gate_observations = _select_gate_sources(tokenizer_cache)
    gate_sources = _publish_gate_sources(root, gate_payloads)
    (
        length_repeat_count,
        length_payloads,
        length_observations,
    ) = _select_length_stress_sources(tokenizer_cache["g1"])
    length_sources = _publish_length_stress_sources(root, length_payloads)

    local_sources: dict[tuple[str, str], Path] = {}
    bundles: dict[str, Any] = {}
    encoder_cache: dict[tuple[str, str], Callable[[str], Any]] = {
        (name, revision): encode
        for name, revision, encode in tokenizer_cache.values()
    }
    for spec in _BUNDLE_SPECS:
        tokenizer_asset = _asset_by_id(runtime, spec.tokenizer_asset)
        tokenizer_key = (tokenizer_asset["repo_id"], tokenizer_asset["revision"])
        if spec.gate_split is not None:
            source_path = gate_sources[spec.gate_split]
            source_revision = f"{_GATE_SOURCE_REVISION_PREFIX}-r{repeat_count:03d}"
        elif spec.length_stress_split is not None:
            source_path = length_sources[spec.length_stress_split]
            source_revision = (
                f"{_LENGTH_STRESS_SOURCE_REVISION_PREFIX}-r{length_repeat_count:03d}"
            )
        else:
            assert spec.source_asset is not None and spec.source_file is not None
            source_asset = _asset_by_id(runtime, spec.source_asset)
            source_key = (spec.source_asset, spec.source_file)
            if source_key not in local_sources:
                local_sources[source_key] = _download_local_source(
                    source_asset,
                    spec.source_file,
                    cache_dir=cache,
                    downloader=downloader,
                )
            source_path = local_sources[source_key]
            source_revision = source_asset["revision"]
            if spec.source_asset == "byted-hotpotqa-formal":
                _probe_formal_parquet(source_path, dataset=spec.dataset)
        destination = _resolve_without_symlinks(
            root / Path(*PurePosixPath(spec.relative_path).parts),
            f"bundle {spec.relative_path}",
            strict=False,
        )
        _require_within(destination, root, f"bundle {spec.relative_path}")
        if destination.exists():
            try:
                bundle_manifest = validator(destination)
            except Exception as exc:
                raise CloudStateError(
                    f"existing bundle is invalid and will not be overwritten: {destination}: {exc}"
                ) from exc
            action = "verified"
        else:
            encode = encoder_cache.get(tokenizer_key)
            if encode is None:
                encode = tokenizer_loader(*tokenizer_key)
                encoder_cache[tokenizer_key] = encode
            kwargs: dict[str, Any] = {
                "dataset": spec.dataset,
                "encode": encode,
                "input_path": source_path,
                "mode": spec.mode,
                "output_dir": destination,
                "profile": spec.profile,
                "seed": SEED,
                "source_revision": source_revision,
                "split": spec.source_split,
                "tokenizer_name": tokenizer_key[0],
                "tokenizer_revision": tokenizer_key[1],
            }
            if spec.length_stress_split is not None:
                kwargs["train_contract"] = _LENGTH_STRESS_CONTRACT
            elif spec.profile == "fixture":
                if spec.mode == "train":
                    kwargs["train_contract"] = _GATE_CONTRACT
                else:
                    kwargs["eval_contract"] = _G1_EVAL_CONTRACT
            try:
                builder(**kwargs)
                destination = _resolve_without_symlinks(
                    destination,
                    f"bundle {spec.relative_path}",
                )
                _require_within(destination, root, f"bundle {spec.relative_path}")
                bundle_manifest = validator(destination)
            except Exception as exc:
                raise CloudStateError(
                    f"failed to build bundle {spec.relative_path}: {type(exc).__name__}: {exc}"
                ) from exc
            action = "built"
        _validate_bundle_identity(
            bundle_manifest,
            spec,
            source_path=source_path,
            source_revision=source_revision,
            tokenizer_name=tokenizer_key[0],
            tokenizer_revision=tokenizer_key[1],
        )
        _insert_nested(bundles, spec.keys, _bundle_summary(destination, bundle_manifest, action))

    gate_source_summary = {
        "generator": _GATE_SOURCE_REVISION_PREFIX,
        "repeat_count": repeat_count,
        "sources": {
            split: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
            for split, path in sorted(gate_sources.items())
        },
        "token_observations": gate_observations,
    }
    length_source_summary = {
        "generator": _LENGTH_STRESS_SOURCE_REVISION_PREFIX,
        "non_scientific": True,
        "ordered_source_ids_sha256": hashlib.sha256(
            "\n".join(
                record["_id"]
                for split in ("train", "validation")
                for record in _synthetic_length_stress_records(split, length_repeat_count)
            ).encode("ascii")
        ).hexdigest(),
        "repeat_count": length_repeat_count,
        "sources": {
            split: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
            for split, path in sorted(length_sources.items())
        },
        "token_observations": length_observations,
    }
    return {
        "bundles": bundles,
        "gate_source": gate_source_summary,
        "length_stress_source": length_source_summary,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
    }


def _file_record(path: str | Path) -> dict[str, Any]:
    file_path = _resolve_without_symlinks(path, "recorded file")
    if not file_path.is_file():
        raise CloudStateError(f"recorded path must be a regular non-symlink file: {file_path}")
    size = file_path.stat().st_size
    return {"path": str(file_path), "sha256": _sha256_file(file_path), "size": size}


def _verify_file_record(record: Any, path: str) -> Path:
    value = _require_mapping(record, path)
    _require_exact_keys(value, {"path", "sha256", "size"}, path)
    _require_sha256(value["sha256"], f"{path}.sha256")
    if not isinstance(value["size"], int) or isinstance(value["size"], bool) or value["size"] < 0:
        raise CloudStateError(f"{path}.size must be a non-negative integer")
    candidate = Path(str(value["path"])).expanduser()
    if candidate.is_symlink():
        raise CloudStateError(f"{path}.path must not be a symlink")
    file_path = candidate.resolve(strict=True)
    if not file_path.is_file():
        raise CloudStateError(f"{path}.path is not a regular non-symlink file")
    if file_path.stat().st_size != value["size"]:
        raise CloudStateError(f"{path} size changed")
    if _sha256_file(file_path) != value["sha256"]:
        raise CloudStateError(f"{path} hash changed")
    return file_path


def _tree_records(root: str | Path, *, expected_names: set[str] | None = None) -> dict[str, Any]:
    directory = _resolve_without_symlinks(root, "tree root")
    if not directory.is_dir():
        raise CloudStateError(f"tree root must be a non-symlink directory: {directory}")
    records: dict[str, Any] = {}
    for candidate in sorted(directory.rglob("*")):
        if candidate.is_symlink():
            raise CloudStateError(f"tree contains a symlink: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(directory).as_posix()
            record = _file_record(candidate)
            record.pop("path")
            records[relative] = record
    if not records:
        raise CloudStateError(f"tree contains no files: {directory}")
    if expected_names is not None and set(records) != expected_names:
        raise CloudStateError(
            f"config tree file set mismatch; missing={sorted(expected_names - set(records))}, "
            f"extra={sorted(set(records) - expected_names)}"
        )
    return records


def _tree_sha256(records: Mapping[str, Any]) -> str:
    return _canonical_sha256(records)


def _validate_resolved_config_tree(
    root: Path,
    records: Mapping[str, Any],
    data_root: Path,
) -> None:
    index = _require_mapping(_load_json(root / "index.json"), "config_index")
    _require_exact_keys(
        index,
        {"configs", "data_root", "schema_version", "status"},
        "config_index",
    )
    if index["schema_version"] != 1 or index["status"] != "resolved":
        raise CloudStateError("resolved config index has an invalid schema/status")
    indexed_data_root = Path(str(index["data_root"])).expanduser().resolve(strict=True)
    if indexed_data_root != data_root:
        raise CloudStateError("resolved config index points at a different data root")
    configs = _require_mapping(index["configs"], "config_index.configs")
    expected_config_ids = {Path(name).stem for name in _CONFIG_NAMES}
    _require_exact_keys(configs, expected_config_ids, "config_index.configs")
    for config_id, raw_entry in configs.items():
        entry = _require_mapping(raw_entry, f"config_index.configs.{config_id}")
        _require_exact_keys(
            entry,
            {"offload_profile", "overrides", "path", "sha256", "source_config"},
            f"config_index.configs.{config_id}",
        )
        filename = f"{config_id}.yaml"
        expected_path = (root / filename).resolve(strict=True)
        if Path(str(entry["path"])).expanduser().resolve(strict=True) != expected_path:
            raise CloudStateError(f"resolved config index path mismatch for {config_id}")
        digest = _require_sha256(entry["sha256"], f"config_index.configs.{config_id}.sha256")
        if digest != records[filename]["sha256"]:
            raise CloudStateError(f"resolved config index hash mismatch for {config_id}")
        overrides = entry["overrides"]
        if not isinstance(overrides, list) or not overrides or not all(
            isinstance(item, str) and item for item in overrides
        ):
            raise CloudStateError(f"resolved config overrides are malformed for {config_id}")
        source_config = entry["source_config"]
        offload_profile = entry["offload_profile"]
        if config_id in _GATE_CONFIG_IDS | _EVAL_CONFIG_IDS:
            if source_config != config_id or offload_profile is not None:
                raise CloudStateError(
                    f"resolved config composition identity is invalid for {config_id}"
                )
        else:
            profile = config_id.rsplit("_", 1)[-1]
            source = config_id.removesuffix(f"_{profile}")
            if (
                profile not in {"r0", "r1"}
                or source not in _TRAINING_SOURCE_IDS
                or source_config != source
                or offload_profile != profile
            ):
                raise CloudStateError(
                    f"resolved offload composition identity is invalid for {config_id}"
                )
        text = expected_path.read_text(encoding="utf-8")
        if any(placeholder in text for placeholder in _PLACEHOLDER_DIGESTS):
            raise CloudStateError(f"resolved config retains a placeholder digest: {config_id}")


def _verify_tree_records(
    root: str | Path,
    records: Any,
    expected_sha256: str,
    path: str,
    *,
    expected_names: set[str] | None = None,
) -> None:
    value = _require_mapping(records, f"{path}.files")
    _require_sha256(expected_sha256, f"{path}.sha256")
    observed = _tree_records(root, expected_names=expected_names)
    if observed != value:
        raise CloudStateError(f"{path} files changed")
    if _tree_sha256(value) != expected_sha256:
        raise CloudStateError(f"{path} tree hash mismatch")


def _validate_asset_report(
    report: Any,
    manifest: Mapping[str, Any],
) -> None:
    value = _require_mapping(report, "asset_report")
    _require_exact_keys(
        value,
        {"assets", "manifest_sha256", "mode", "schema_version", "status"},
        "asset_report",
    )
    if value["schema_version"] != 1 or value["mode"] != "download_and_verify":
        raise CloudStateError("asset report schema/mode is not a download verification")
    if value.get("status") != "complete":
        raise CloudStateError("asset report is not complete")
    if value.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise CloudStateError("asset report is not bound to the runtime manifest")
    results = value.get("assets")
    if not isinstance(results, list):
        raise CloudStateError("asset report assets must be a list")
    result_by_id = {
        item.get("asset_id"): item
        for item in results
        if isinstance(item, Mapping) and isinstance(item.get("asset_id"), str)
    }
    expected_by_id = {asset["asset_id"]: asset for asset in manifest["assets"]}
    if len(result_by_id) != len(results) or set(result_by_id) != set(expected_by_id):
        raise CloudStateError("asset report does not contain every manifest asset exactly once")
    for asset_id, asset in expected_by_id.items():
        result = result_by_id[asset_id]
        _require_exact_keys(
            result,
            {"asset_id", "files", "repo_id", "revision", "status"},
            f"asset_report.{asset_id}",
        )
        if result.get("status") != "complete":
            raise CloudStateError(f"asset report remains incomplete for {asset_id}")
        if result.get("repo_id") != asset["repo_id"] or result.get("revision") != asset["revision"]:
            raise CloudStateError(f"asset report identity mismatch for {asset_id}")
        files = result.get("files")
        if not isinstance(files, list):
            raise CloudStateError(f"asset report files are malformed for {asset_id}")
        files_by_path = {
            item.get("path"): item
            for item in files
            if isinstance(item, Mapping) and isinstance(item.get("path"), str)
        }
        expected_files = {item["path"]: item for item in asset["files"]}
        if len(files_by_path) != len(files) or set(files_by_path) != set(expected_files):
            raise CloudStateError(f"asset report file set mismatch for {asset_id}")
        for filename, file_spec in expected_files.items():
            file_result = files_by_path[filename]
            _require_exact_keys(
                file_result,
                {"attempts", "cached_path", "integrity", "path", "status"},
                f"asset_report.{asset_id}.{filename}",
            )
            if file_result.get("status") != "verified":
                raise CloudStateError(f"asset file is not verified: {asset_id}:{filename}")
            cached_path_value = file_result.get("cached_path")
            if not isinstance(cached_path_value, str) or not cached_path_value:
                raise CloudStateError(
                    f"asset report lacks the verified cache path: {asset_id}:{filename}"
                )
            cached_path = _resolve_without_symlinks(
                cached_path_value,
                f"asset cache path {asset_id}:{filename}",
            )
            if not cached_path.is_file():
                raise CloudStateError(
                    f"asset cache path is not a regular file: {asset_id}:{filename}"
                )
            integrity = _require_mapping(
                file_result.get("integrity"),
                f"asset_report.{asset_id}.{filename}.integrity",
            )
            digest_key = "sha256" if "sha256" in file_spec else "git_blob_sha1"
            if (
                integrity.get("algorithm") != digest_key
                or integrity.get("digest") != file_spec[digest_key]
                or integrity.get("size") != file_spec["size"]
            ):
                raise CloudStateError(f"asset integrity evidence mismatch: {asset_id}:{filename}")
            try:
                observed = verify_cached_file(cached_path, file_spec)
            except (AssetIntegrityError, OSError, KeyError) as exc:
                raise CloudStateError(
                    f"asset cache bytes changed: {asset_id}:{filename}: {exc}"
                ) from exc
            if observed != dict(integrity):
                raise CloudStateError(
                    f"asset cache evidence changed: {asset_id}:{filename}"
                )
def _run_git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise CloudStateError(f"git {' '.join(args)} failed for {repository}: {exc}") from exc
    return result.stdout.strip()


def _normalize_git_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    ssh_match = re.fullmatch(r"git@([^:]+):(.+)", normalized)
    if ssh_match:
        normalized = f"https://{ssh_match.group(1)}/{ssh_match.group(2)}"
    normalized = re.sub(r"^ssh://git@", "https://", normalized)
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def _git_repository_record(
    path: str | Path,
    *,
    expected_commit: str,
    expected_repository: str | None,
    require_clean: bool = True,
) -> dict[str, Any]:
    repository = _resolve_without_symlinks(path, "Git repository")
    if not repository.is_dir():
        raise CloudStateError(f"Git repository must be a non-symlink directory: {repository}")
    head = _run_git(repository, "rev-parse", "HEAD")
    _require_commit(head, f"Git HEAD for {repository}")
    if head != expected_commit:
        raise CloudStateError(f"Git HEAD mismatch for {repository}: {head} != {expected_commit}")
    if require_clean and _run_git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise CloudStateError(f"Git repository has tracked or untracked changes: {repository}")
    remote = _run_git(repository, "remote", "get-url", "origin") if expected_repository else None
    if expected_repository and _normalize_git_url(remote or "") != _normalize_git_url(expected_repository):
        raise CloudStateError(f"Git origin mismatch for {repository}")
    tree = _run_git(repository, "rev-parse", "HEAD^{tree}")
    _require_commit(tree, f"Git tree for {repository}")
    result = {"commit": head, "path": str(repository), "tree": tree}
    if expected_repository is not None:
        result["repository"] = expected_repository
    return result


def _bundle_record(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = _tree_records(path)
    source_path = _resolve_without_symlinks(
        str(manifest["source"]["path"]),
        "bundle source file",
    )
    source_file = _file_record(source_path)
    return {
        "contract": manifest["contract"],
        "dataset": manifest["dataset"],
        "files": files,
        "manifest_sha256": manifest["manifest_sha256"],
        "mode": manifest["mode"],
        "path": str(path),
        "profile": manifest["profile"],
        "source_file": source_file,
        "source_revision": manifest["source"]["revision"],
        "tokenizer_name": manifest["tokenizer"]["name"],
        "tokenizer_revision": manifest["tokenizer"]["revision"],
        "tree_sha256": _tree_sha256(files),
    }


def _expected_bundle_sources(
    spec: BundleSpec,
    manifest: Mapping[str, Any],
    source_path: Path,
) -> tuple[str, str, str]:
    tokenizer_asset = _asset_by_id(manifest, spec.tokenizer_asset)
    if spec.gate_split is not None:
        raise CloudStateError("formal source resolver received a gate bundle")
    assert spec.source_asset is not None and spec.source_file is not None
    source_asset = _asset_by_id(manifest, spec.source_asset)
    source_spec = _file_spec(source_asset, spec.source_file)
    if source_path.stat().st_size != source_spec["size"]:
        raise CloudStateError(f"formal source size mismatch for {spec.relative_path}")
    digest_key = "sha256" if "sha256" in source_spec else "git_blob_sha1"
    if digest_key != "sha256" or _sha256_file(source_path) != source_spec["sha256"]:
        raise CloudStateError(f"formal source digest mismatch for {spec.relative_path}")
    return source_asset["revision"], tokenizer_asset["repo_id"], tokenizer_asset["revision"]


def _validate_gate_source_file(spec: BundleSpec, source_path: Path, revision: str) -> None:
    match = re.fullmatch(
        rf"{re.escape(_GATE_SOURCE_REVISION_PREFIX)}-r(\d{{3}})",
        revision,
    )
    if match is None:
        raise CloudStateError(f"gate bundle has an invalid synthetic source revision: {revision!r}")
    repeat_count = int(match.group(1))
    if repeat_count not in _GATE_REPEAT_CANDIDATES:
        raise CloudStateError(f"gate bundle uses an unsupported repeat count: {repeat_count}")
    assert spec.gate_split is not None
    expected = _jsonl_bytes(_synthetic_gate_records(spec.gate_split, repeat_count))
    if source_path.read_bytes() != expected:
        raise CloudStateError(f"gate source bytes are not deterministic for {spec.relative_path}")


def _validate_length_stress_source_file(
    spec: BundleSpec,
    source_path: Path,
    revision: str,
) -> None:
    match = re.fullmatch(
        rf"{re.escape(_LENGTH_STRESS_SOURCE_REVISION_PREFIX)}-r(\d{{3}})",
        revision,
    )
    if match is None:
        raise CloudStateError(
            f"length-stress bundle has an invalid source revision: {revision!r}"
        )
    repeat_count = int(match.group(1))
    if repeat_count not in _LENGTH_STRESS_REPEAT_CANDIDATES:
        raise CloudStateError(
            f"length-stress bundle uses an unsupported repeat count: {repeat_count}"
        )
    assert spec.length_stress_split is not None
    expected = _jsonl_bytes(
        _synthetic_length_stress_records(spec.length_stress_split, repeat_count)
    )
    if source_path.read_bytes() != expected:
        raise CloudStateError(
            f"length-stress source bytes are not deterministic for {spec.relative_path}"
        )


def _load_and_record_bundles(
    data_root: Path,
    asset_manifest: Mapping[str, Any],
    persistent_root: Path,
) -> dict[str, Any]:
    bundles: dict[str, Any] = {}
    for spec in _BUNDLE_SPECS:
        path = _resolve_without_symlinks(
            data_root / Path(*PurePosixPath(spec.relative_path).parts),
            f"bundle {spec.relative_path}",
        )
        _require_within(path, data_root, f"bundle {spec.relative_path}")
        try:
            manifest = validate_artifact_bundle(path)
        except Exception as exc:
            raise CloudStateError(f"bundle validation failed for {path}: {exc}") from exc
        source_path = _resolve_without_symlinks(
            str(manifest["source"]["path"]),
            f"bundle source {spec.relative_path}",
        )
        _require_within(
            source_path,
            persistent_root,
            f"bundle source {spec.relative_path}",
        )
        if spec.gate_split is not None:
            _validate_gate_source_file(spec, source_path, manifest["source"]["revision"])
            expected_source_revision = manifest["source"]["revision"]
            tokenizer_asset = _asset_by_id(asset_manifest, spec.tokenizer_asset)
            tokenizer_name = tokenizer_asset["repo_id"]
            tokenizer_revision = tokenizer_asset["revision"]
        elif spec.length_stress_split is not None:
            _validate_length_stress_source_file(
                spec,
                source_path,
                manifest["source"]["revision"],
            )
            expected_source_revision = manifest["source"]["revision"]
            tokenizer_asset = _asset_by_id(asset_manifest, spec.tokenizer_asset)
            tokenizer_name = tokenizer_asset["repo_id"]
            tokenizer_revision = tokenizer_asset["revision"]
        else:
            expected_source_revision, tokenizer_name, tokenizer_revision = _expected_bundle_sources(
                spec,
                asset_manifest,
                source_path,
            )
        _validate_bundle_identity(
            manifest,
            spec,
            source_path=source_path,
            source_revision=expected_source_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
        )
        _insert_nested(bundles, spec.keys, _bundle_record(path, manifest))
    return bundles


def _kernel_records(kernel_source_root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for kernel in lock["kernels"]:
        records[kernel["name"]] = _git_repository_record(
            kernel_source_root / kernel["name"],
            expected_commit=kernel["commit"],
            expected_repository=kernel["repository"],
        )
    return records


def _publish_handoff_impl(
    output: str | Path,
    commit: str,
    asset_manifest_path: str | Path,
    asset_report_path: str | Path,
    pip_freeze_path: str | Path,
    data_root: str | Path,
    kernel_source_root: str | Path,
    config_root: str | Path,
    persist_root: str | Path,
    *,
    experiment_profile_id: str = EXPERIMENT_PROFILE_ID,
    repository_root: str | Path = REPOSITORY_ROOT,
    tracked_manifest_path: str | Path = _TRACKED_ASSET_MANIFEST,
    environment_lock_path: str | Path = _ENVIRONMENT_LOCK,
) -> dict[str, Any]:
    """Validate all CPU outputs and atomically publish a self-hashed handoff."""

    commit = _require_commit(commit)
    if experiment_profile_id != EXPERIMENT_PROFILE_ID:
        raise CloudStateError(
            f"unsupported experiment profile: {experiment_profile_id!r}"
        )
    _git_repository_record(
        repository_root,
        expected_commit=commit,
        expected_repository=None,
    )
    runtime_manifest = load_asset_manifest(asset_manifest_path)
    tracked_manifest = load_asset_manifest(tracked_manifest_path)
    _validate_runtime_against_tracked(runtime_manifest, tracked_manifest)
    asset_report = _load_json(asset_report_path)
    _validate_asset_report(asset_report, runtime_manifest)
    environment_lock = load_environment_lock(
        environment_lock_path,
        repository_root=repository_root,
    )
    pip_record = _file_record(pip_freeze_path)
    if pip_record["size"] == 0:
        raise CloudStateError("pip freeze artifact is empty")
    persistent_directory = _resolve_without_symlinks(persist_root, "persistent root")
    if not persistent_directory.is_dir():
        raise CloudStateError("persistent root must be a directory")
    handoff_output = _resolve_without_symlinks(
        output,
        "CPU handoff output",
        strict=False,
    )
    _require_within(handoff_output, persistent_directory, "CPU handoff output")
    root = _resolve_without_symlinks(data_root, "data root")
    if not root.is_dir():
        raise CloudStateError("data root must be a non-symlink directory")
    _require_within(root, persistent_directory, "data root")
    bundles = _load_and_record_bundles(root, runtime_manifest, persistent_directory)
    kernel_root = _resolve_without_symlinks(kernel_source_root, "kernel source root")
    _require_within(kernel_root, persistent_directory, "kernel source root")
    kernels = _kernel_records(kernel_root, environment_lock)
    config_directory = _resolve_without_symlinks(config_root, "resolved config root")
    _require_within(config_directory, persistent_directory, "resolved config root")
    config_files = _tree_records(config_directory, expected_names=_CONFIG_TREE_NAMES)
    config_sha256 = _tree_sha256(config_files)
    _validate_resolved_config_tree(config_directory, config_files, root)
    payload: dict[str, Any] = {
        "asset_manifest": {
            "file": _file_record(asset_manifest_path),
            "manifest_sha256": runtime_manifest["manifest_sha256"],
        },
        "asset_report": {"file": _file_record(asset_report_path)},
        "bundles": bundles,
        "config_files": config_files,
        "config_root": str(config_directory),
        "config_tree_sha256": config_sha256,
        "data_root": str(root),
        "environment_lock": {"file": _file_record(environment_lock_path)},
        "environment_lock_sha256": environment_lock["lock_sha256"],
        "experiment_profile_id": experiment_profile_id,
        "git_commit": commit,
        "kernel_source_root": str(kernel_root),
        "kernel_sources": kernels,
        "pip_freeze": pip_record,
        "persist_root": str(persistent_directory),
        "schema_version": SCHEMA_VERSION,
        "status": HANDOFF_STATUS,
    }
    sealed = dict(payload)
    sealed["handoff_sha256"] = _canonical_sha256(payload)
    _atomic_write_json(handoff_output, sealed)
    return sealed


def publish_handoff(
    output: str | Path,
    commit: str,
    asset_manifest_path: str | Path,
    asset_report_path: str | Path,
    pip_freeze_path: str | Path,
    data_root: str | Path,
    kernel_source_root: str | Path,
    config_root: str | Path,
    persist_root: str | Path,
    *,
    experiment_profile_id: str = EXPERIMENT_PROFILE_ID,
    repository_root: str | Path = REPOSITORY_ROOT,
    tracked_manifest_path: str | Path = _TRACKED_ASSET_MANIFEST,
    environment_lock_path: str | Path = _ENVIRONMENT_LOCK,
) -> dict[str, Any]:
    token = _SHA256_CACHE.set({})
    try:
        return _publish_handoff_impl(
            output,
            commit,
            asset_manifest_path,
            asset_report_path,
            pip_freeze_path,
            data_root,
            kernel_source_root,
            config_root,
            persist_root,
            experiment_profile_id=experiment_profile_id,
            repository_root=repository_root,
            tracked_manifest_path=tracked_manifest_path,
            environment_lock_path=environment_lock_path,
        )
    finally:
        _SHA256_CACHE.reset(token)


def _iter_bundle_records(bundles: Mapping[str, Any]) -> list[tuple[BundleSpec, Mapping[str, Any]]]:
    _require_exact_keys(bundles, {"capacity", "formal", "gates"}, "bundles")
    capacity = _require_mapping(bundles["capacity"], "bundles.capacity")
    _require_exact_keys(capacity, {"length_stress"}, "bundles.capacity")
    length_stress = _require_mapping(
        capacity["length_stress"],
        "bundles.capacity.length_stress",
    )
    _require_exact_keys(
        length_stress,
        {"train", "validation"},
        "bundles.capacity.length_stress",
    )
    gates = _require_mapping(bundles["gates"], "bundles.gates")
    _require_exact_keys(gates, {"g0", "g1"}, "bundles.gates")
    g0 = _require_mapping(gates["g0"], "bundles.gates.g0")
    _require_exact_keys(g0, {"train", "validation"}, "bundles.gates.g0")
    g1 = _require_mapping(gates["g1"], "bundles.gates.g1")
    _require_exact_keys(g1, {"eval", "train", "validation"}, "bundles.gates.g1")
    formal = _require_mapping(bundles["formal"], "bundles.formal")
    _require_exact_keys(formal, {"eval", "train", "validation"}, "bundles.formal")
    formal_eval = _require_mapping(formal["eval"], "bundles.formal.eval")
    _require_exact_keys(
        formal_eval,
        {"2wikimultihopqa", "hotpotqa"},
        "bundles.formal.eval",
    )
    result: list[tuple[BundleSpec, Mapping[str, Any]]] = []
    for spec in _BUNDLE_SPECS:
        value: Any = bundles
        for key in spec.keys:
            value = _require_mapping(value, f"bundles.{'.'.join(spec.keys)}").get(key)
        result.append((spec, _require_mapping(value, f"bundles.{'.'.join(spec.keys)}")))
    return result


def _verify_bundle_record(
    spec: BundleSpec,
    record: Mapping[str, Any],
    asset_manifest: Mapping[str, Any],
) -> None:
    expected_keys = {
        "contract",
        "dataset",
        "files",
        "manifest_sha256",
        "mode",
        "path",
        "profile",
        "source_file",
        "source_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "tree_sha256",
    }
    _require_exact_keys(record, expected_keys, f"bundles.{'.'.join(spec.keys)}")
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    _verify_tree_records(
        path,
        record["files"],
        record["tree_sha256"],
        f"bundles.{'.'.join(spec.keys)}",
    )
    source_path = _verify_file_record(
        record["source_file"],
        f"bundles.{'.'.join(spec.keys)}.source_file",
    )
    try:
        manifest = validate_artifact_bundle(path)
    except Exception as exc:
        raise CloudStateError(f"bundle validation failed for {path}: {exc}") from exc
    if spec.gate_split is not None:
        _validate_gate_source_file(spec, source_path, manifest["source"]["revision"])
        expected_source_revision = manifest["source"]["revision"]
        tokenizer_asset = _asset_by_id(asset_manifest, spec.tokenizer_asset)
        tokenizer_name = tokenizer_asset["repo_id"]
        tokenizer_revision = tokenizer_asset["revision"]
    elif spec.length_stress_split is not None:
        _validate_length_stress_source_file(
            spec,
            source_path,
            manifest["source"]["revision"],
        )
        expected_source_revision = manifest["source"]["revision"]
        tokenizer_asset = _asset_by_id(asset_manifest, spec.tokenizer_asset)
        tokenizer_name = tokenizer_asset["repo_id"]
        tokenizer_revision = tokenizer_asset["revision"]
    else:
        expected_source_revision, tokenizer_name, tokenizer_revision = _expected_bundle_sources(
            spec,
            asset_manifest,
            source_path,
        )
    if (
        record["source_revision"] != expected_source_revision
        or record["tokenizer_name"] != tokenizer_name
        or record["tokenizer_revision"] != tokenizer_revision
    ):
        raise CloudStateError(f"recorded bundle provenance mismatch for {spec.relative_path}")
    _validate_bundle_identity(
        manifest,
        spec,
        source_path=source_path,
        source_revision=expected_source_revision,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
    )
    comparisons = {
        "contract": manifest["contract"],
        "dataset": manifest["dataset"],
        "manifest_sha256": manifest["manifest_sha256"],
        "mode": manifest["mode"],
        "profile": manifest["profile"],
    }
    for key, expected in comparisons.items():
        if record[key] != expected:
            raise CloudStateError(f"recorded bundle {key} differs from validated bundle")


def _field_value(handoff: Mapping[str, Any], dotted_path: str) -> Any:
    if not dotted_path or any(not component for component in dotted_path.split(".")):
        raise CloudStateError("field must be a non-empty dotted path")
    value: Any = handoff
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise CloudStateError(f"handoff field does not exist: {dotted_path}")
        value = value[component]
    if isinstance(value, (Mapping, list)):
        raise CloudStateError(f"handoff field is not a scalar: {dotted_path}")
    return value


def _verify_handoff_impl(
    handoff_path: str | Path,
    expected_commit: str,
    *,
    field: str | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[Mapping[str, Any], Any | None]:
    """Re-validate a handoff and every byte/repository identity it records."""

    expected_commit = _require_commit(expected_commit, "expected_commit")
    path = _resolve_without_symlinks(handoff_path, "CPU handoff")
    handoff = _require_mapping(_load_json(path), "handoff")
    expected_keys = {
        "asset_manifest",
        "asset_report",
        "bundles",
        "config_files",
        "config_root",
        "config_tree_sha256",
        "data_root",
        "environment_lock",
        "environment_lock_sha256",
        "experiment_profile_id",
        "git_commit",
        "handoff_sha256",
        "kernel_source_root",
        "kernel_sources",
        "pip_freeze",
        "persist_root",
        "schema_version",
        "status",
    }
    _require_exact_keys(handoff, expected_keys, "handoff")
    if handoff["schema_version"] != SCHEMA_VERSION or handoff["status"] != HANDOFF_STATUS:
        raise CloudStateError("handoff schema/status is not CPU-ready")
    if handoff["experiment_profile_id"] != EXPERIMENT_PROFILE_ID:
        raise CloudStateError("handoff experiment profile is not the active 5090/2B profile")
    digest = _require_sha256(handoff["handoff_sha256"], "handoff.handoff_sha256")
    unsigned = dict(handoff)
    unsigned.pop("handoff_sha256")
    if digest != _canonical_sha256(unsigned):
        raise CloudStateError("handoff self-hash mismatch")
    canonical = _canonical_bytes(handoff) + b"\n"
    if path.read_bytes() != canonical:
        raise CloudStateError("handoff JSON is not canonical")
    if handoff["git_commit"] != expected_commit:
        raise CloudStateError("handoff Git commit differs from the expected commit")
    persistent_root = _resolve_without_symlinks(
        str(handoff["persist_root"]),
        "handoff persistent root",
    )
    if str(persistent_root) != handoff["persist_root"]:
        raise CloudStateError("handoff persistent root is not canonical")
    _require_within(path, persistent_root, "CPU handoff")
    _git_repository_record(
        repository_root,
        expected_commit=expected_commit,
        expected_repository=None,
    )

    asset_manifest_container = _require_mapping(handoff["asset_manifest"], "asset_manifest")
    _require_exact_keys(asset_manifest_container, {"file", "manifest_sha256"}, "asset_manifest")
    manifest_path = _verify_file_record(asset_manifest_container["file"], "asset_manifest.file")
    runtime_manifest = load_asset_manifest(manifest_path)
    _require_resolved_manifest(runtime_manifest)
    _validate_runtime_against_tracked(
        runtime_manifest,
        load_asset_manifest(_TRACKED_ASSET_MANIFEST),
    )
    if runtime_manifest["manifest_sha256"] != asset_manifest_container["manifest_sha256"]:
        raise CloudStateError("runtime asset manifest digest changed")
    asset_report_container = _require_mapping(handoff["asset_report"], "asset_report")
    _require_exact_keys(asset_report_container, {"file"}, "asset_report")
    report_path = _verify_file_record(asset_report_container["file"], "asset_report.file")
    _validate_asset_report(_load_json(report_path), runtime_manifest)

    environment_container = _require_mapping(handoff["environment_lock"], "environment_lock")
    _require_exact_keys(environment_container, {"file"}, "environment_lock")
    environment_path = _verify_file_record(environment_container["file"], "environment_lock.file")
    environment_lock = load_environment_lock(environment_path, repository_root=repository_root)
    if environment_lock["lock_sha256"] != handoff["environment_lock_sha256"]:
        raise CloudStateError("environment lock digest changed")
    _verify_file_record(handoff["pip_freeze"], "pip_freeze")
    _verify_tree_records(
        handoff["config_root"],
        handoff["config_files"],
        handoff["config_tree_sha256"],
        "config",
        expected_names=_CONFIG_TREE_NAMES,
    )

    data_root = _resolve_without_symlinks(str(handoff["data_root"]), "handoff data root")
    if not data_root.is_dir():
        raise CloudStateError("handoff data root must be a non-symlink directory")
    _require_within(data_root, persistent_root, "handoff data root")
    config_root = _resolve_without_symlinks(
        str(handoff["config_root"]),
        "handoff resolved config root",
    )
    _require_within(config_root, persistent_root, "handoff resolved config root")
    _validate_resolved_config_tree(
        config_root,
        _require_mapping(handoff["config_files"], "config_files"),
        data_root,
    )

    bundles = _require_mapping(handoff["bundles"], "bundles")
    bundle_records = _iter_bundle_records(bundles)
    for spec, record in bundle_records:
        expected_bundle_path = _resolve_without_symlinks(
            data_root / Path(*PurePosixPath(spec.relative_path).parts),
            f"handoff bundle {spec.relative_path}",
        )
        _require_within(
            expected_bundle_path,
            data_root,
            f"handoff bundle {spec.relative_path}",
        )
        recorded_bundle_path = _resolve_without_symlinks(
            str(record["path"]),
            f"recorded bundle {spec.relative_path}",
        )
        if recorded_bundle_path != expected_bundle_path:
            raise CloudStateError(f"bundle escaped the recorded data root: {spec.relative_path}")
        _verify_bundle_record(spec, record, runtime_manifest)

    kernel_records = _require_mapping(handoff["kernel_sources"], "kernel_sources")
    expected_kernels = {kernel["name"]: kernel for kernel in environment_lock["kernels"]}
    if set(kernel_records) != set(expected_kernels):
        raise CloudStateError("handoff kernel source set differs from the environment lock")
    kernel_root = _resolve_without_symlinks(
        str(handoff["kernel_source_root"]),
        "handoff kernel source root",
    )
    _require_within(kernel_root, persistent_root, "handoff kernel source root")
    for name, expected in expected_kernels.items():
        observed = _git_repository_record(
            kernel_root / name,
            expected_commit=expected["commit"],
            expected_repository=expected["repository"],
        )
        if observed != kernel_records[name]:
            raise CloudStateError(f"kernel repository state changed for {name}")
    value = _field_value(handoff, field) if field is not None else None
    return handoff, value


def verify_handoff(
    handoff_path: str | Path,
    expected_commit: str,
    *,
    field: str | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[Mapping[str, Any], Any | None]:
    token = _SHA256_CACHE.set({})
    try:
        return _verify_handoff_impl(
            handoff_path,
            expected_commit,
            field=field,
            repository_root=repository_root,
        )
    finally:
        _SHA256_CACHE.reset(token)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    resolve = commands.add_parser("resolve-assets", help="resolve blocked HF LFS metadata")
    resolve.add_argument("--manifest", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)

    data = commands.add_parser("build-data", help="build all data bundles from local assets")
    data.add_argument("--manifest", type=Path, required=True)
    data.add_argument("--cache-dir", type=Path, required=True)
    data.add_argument("--data-root", type=Path, required=True)
    data.add_argument("--summary", type=Path)

    publish = commands.add_parser("publish-handoff", help="seal the CPU-ready handoff")
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--commit", required=True)
    publish.add_argument("--asset-manifest", type=Path, required=True)
    publish.add_argument("--asset-report", type=Path, required=True)
    publish.add_argument("--pip-freeze", type=Path, required=True)
    publish.add_argument("--data-root", type=Path, required=True)
    publish.add_argument("--kernel-source-root", type=Path, required=True)
    publish.add_argument("--config-root", type=Path, required=True)
    publish.add_argument("--persist-root", type=Path, required=True)
    publish.add_argument(
        "--experiment-profile",
        default=EXPERIMENT_PROFILE_ID,
    )

    verify = commands.add_parser("verify-handoff", help="strictly re-verify a handoff")
    verify.add_argument("--handoff", type=Path, required=True)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument(
        "--field",
        action="append",
        help="print one verified scalar field; repeat to print several fields once verified",
    )
    return parser


def _print_json(value: Mapping[str, Any]) -> None:
    print(_canonical_bytes(value).decode("ascii"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "resolve-assets":
            result = resolve_assets(args.manifest, args.output)
            _print_json(result)
        elif args.command == "build-data":
            result = build_data(args.manifest, args.cache_dir, args.data_root)
            if args.summary is not None:
                _atomic_write_json(args.summary, result)
            _print_json(result)
        elif args.command == "publish-handoff":
            result = publish_handoff(
                args.output,
                args.commit,
                args.asset_manifest,
                args.asset_report,
                args.pip_freeze,
                args.data_root,
                args.kernel_source_root,
                args.config_root,
                args.persist_root,
                experiment_profile_id=args.experiment_profile,
            )
            _print_json(
                {
                    "handoff_sha256": result["handoff_sha256"],
                    "output": str(Path(args.output).expanduser().resolve()),
                    "status": result["status"],
                }
            )
        elif args.command == "verify-handoff":
            fields = args.field or []
            result, _ = verify_handoff(
                args.handoff,
                args.expected_commit,
            )
            if not fields:
                _print_json(
                    {
                        "git_commit": result["git_commit"],
                        "handoff_sha256": result["handoff_sha256"],
                        "status": "verified",
                    }
                )
            else:
                for field in fields:
                    field_value = _field_value(result, field)
                    if isinstance(field_value, str):
                        print(field_value)
                    else:
                        print(
                            json.dumps(
                                field_value,
                                ensure_ascii=True,
                                allow_nan=False,
                            )
                        )
        else:  # pragma: no cover - argparse enforces the command set.
            raise CloudStateError(f"unsupported command: {args.command}")
    except Exception as exc:
        _print_json({"error": f"{type(exc).__name__}: {exc}", "status": "blocked"})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CloudStateError",
    "build_data",
    "main",
    "publish_handoff",
    "resolve_assets",
    "verify_handoff",
]
