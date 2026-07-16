"""Prefetch pinned Hugging Face assets and verify an offline-complete cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 8 * 1024 * 1024


class AssetManifestError(ValueError):
    """Raised when the pinned asset manifest is malformed or has been edited."""


class AssetIntegrityError(RuntimeError):
    """Raised when a cached file does not match its pinned content identity."""


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return payload


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    sealed = json.loads(json.dumps(manifest))
    sealed["manifest_sha256"] = canonical_json_sha256(manifest_payload(sealed))
    return sealed


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssetManifestError(f"{path} keys mismatch; missing={missing}, extra={extra}")


def _require_revision(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _HEX40.fullmatch(value):
        raise AssetManifestError(f"{path} must be a lowercase 40-character commit SHA")
    return value


def _require_safe_repo_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssetManifestError(f"{path} must be a non-empty repository path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or "\\" in value or any(part in {"", ".", ".."} for part in candidate.parts):
        raise AssetManifestError(f"{path} must be a safe POSIX-relative path")
    return value


def validate_asset_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise AssetManifestError("manifest must be a JSON object")
    _require_exact_keys(manifest, {"assets", "manifest_sha256", "schema_version"}, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise AssetManifestError(f"unsupported schema_version={manifest['schema_version']!r}")
    digest = manifest["manifest_sha256"]
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise AssetManifestError("manifest_sha256 must be a lowercase SHA-256")
    expected_digest = canonical_json_sha256(manifest_payload(manifest))
    if digest != expected_digest:
        raise AssetManifestError("manifest_sha256 does not match the canonical manifest payload")

    assets = manifest["assets"]
    if not isinstance(assets, list) or not assets:
        raise AssetManifestError("assets must be a non-empty list")
    seen_asset_ids: set[str] = set()
    for asset_index, asset in enumerate(assets):
        asset_path = f"assets[{asset_index}]"
        if not isinstance(asset, Mapping):
            raise AssetManifestError(f"{asset_path} must be an object")
        base_asset_keys = {"asset_id", "files", "kind", "repo_id", "repo_type", "revision"}
        unresolved_asset_keys = base_asset_keys | {"metadata_resolution", "training_gate"}
        if set(asset) not in (base_asset_keys, unresolved_asset_keys):
            raise AssetManifestError(f"{asset_path} has unsupported schema keys")
        asset_id = asset["asset_id"]
        if not isinstance(asset_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", asset_id):
            raise AssetManifestError(f"{asset_path}.asset_id is invalid")
        if asset_id in seen_asset_ids:
            raise AssetManifestError(f"duplicate asset_id {asset_id!r}")
        seen_asset_ids.add(asset_id)
        if asset["kind"] not in {
            "dataset_source",
            "formal_training_source",
            "model_and_tokenizer",
        }:
            raise AssetManifestError(f"{asset_path}.kind is unsupported")
        expected_repo_type = "model" if asset["kind"] == "model_and_tokenizer" else "dataset"
        if asset["repo_type"] != expected_repo_type:
            raise AssetManifestError(
                f"{asset_path}.repo_type must be {expected_repo_type!r} for {asset['kind']!r}"
            )
        repo_id = asset["repo_id"]
        if not isinstance(repo_id, str) or repo_id.count("/") != 1 or any(not part for part in repo_id.split("/")):
            raise AssetManifestError(f"{asset_path}.repo_id must be an owner/name identifier")
        _require_revision(asset["revision"], f"{asset_path}.revision")
        has_unresolved_metadata = "metadata_resolution" in asset
        if has_unresolved_metadata:
            if asset["kind"] != "formal_training_source" or asset["training_gate"] != "BLOCKED":
                raise AssetManifestError(
                    f"{asset_path} unresolved metadata must be a BLOCKED formal training source"
                )
            resolution = asset["metadata_resolution"]
            if not isinstance(resolution, Mapping):
                raise AssetManifestError(f"{asset_path}.metadata_resolution must be an object")
            _require_exact_keys(
                resolution,
                {"command", "required_fields", "status"},
                f"{asset_path}.metadata_resolution",
            )
            if resolution["status"] != "METADATA_UNAVAILABLE":
                raise AssetManifestError(f"{asset_path}.metadata_resolution.status is unsupported")
            if not isinstance(resolution["command"], str) or not resolution["command"].strip():
                raise AssetManifestError(f"{asset_path}.metadata_resolution.command is required")
            if resolution["required_fields"] != ["size", "sha256"]:
                raise AssetManifestError(
                    f"{asset_path}.metadata_resolution.required_fields must be ['size', 'sha256']"
                )

        files = asset["files"]
        if not isinstance(files, list) or not files:
            raise AssetManifestError(f"{asset_path}.files must be a non-empty list")
        seen_paths: set[str] = set()
        for file_index, file_spec in enumerate(files):
            file_path = f"{asset_path}.files[{file_index}]"
            if not isinstance(file_spec, Mapping):
                raise AssetManifestError(f"{file_path} must be an object")
            keys = set(file_spec)
            if keys == {"metadata_status", "path", "required"}:
                relative_path = _require_safe_repo_path(file_spec["path"], f"{file_path}.path")
                if not has_unresolved_metadata:
                    raise AssetManifestError(f"{file_path} unresolved metadata lacks an asset-level gate")
                if file_spec["metadata_status"] != "METADATA_UNAVAILABLE" or file_spec["required"] is not True:
                    raise AssetManifestError(f"{file_path} must remain required and METADATA_UNAVAILABLE")
                if relative_path in seen_paths:
                    raise AssetManifestError(f"duplicate file path {relative_path!r} in {asset_id!r}")
                seen_paths.add(relative_path)
                continue
            if keys not in (
                {"git_blob_sha1", "path", "size"},
                {"path", "sha256", "size"},
            ):
                raise AssetManifestError(
                    f"{file_path} must contain exactly one of git_blob_sha1 or sha256"
                )
            relative_path = _require_safe_repo_path(file_spec["path"], f"{file_path}.path")
            if relative_path in seen_paths:
                raise AssetManifestError(f"duplicate file path {relative_path!r} in {asset_id!r}")
            seen_paths.add(relative_path)
            size = file_spec["size"]
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise AssetManifestError(f"{file_path}.size must be a non-negative integer")
            digest_key = "sha256" if "sha256" in file_spec else "git_blob_sha1"
            digest_pattern = _HEX64 if digest_key == "sha256" else _HEX40
            if not isinstance(file_spec[digest_key], str) or not digest_pattern.fullmatch(file_spec[digest_key]):
                raise AssetManifestError(f"{file_path}.{digest_key} is malformed")


def load_asset_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetManifestError(f"cannot read asset manifest {manifest_path}: {exc}") from exc
    validate_asset_manifest(manifest)
    return manifest


def _stream_digest(path: Path, algorithm: str, *, git_blob_size: int | None = None) -> str:
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        if git_blob_size is None:
            raise ValueError("git_blob_size is required for Git blob hashing")
        digest.update(f"blob {git_blob_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    identity = (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    # Windows reports st_ctime_ns inconsistently between stat() and fstat().
    return identity if os.name == "nt" else (*identity, value.st_ctime_ns)


def verify_cached_file(path: str | Path, file_spec: Mapping[str, Any]) -> dict[str, Any]:
    cache_path = Path(path)
    try:
        path_before = cache_path.stat()
        handle = cache_path.open("rb")
    except OSError as exc:
        raise AssetIntegrityError(f"cached file is unavailable: {exc}") from exc
    observed_size = path_before.st_size
    expected_size = file_spec["size"]
    if observed_size != expected_size:
        handle.close()
        raise AssetIntegrityError(f"size mismatch: expected {expected_size}, observed {observed_size}")

    try:
        descriptor_before = os.fstat(handle.fileno())
        if _stat_identity(descriptor_before) != _stat_identity(path_before):
            raise AssetIntegrityError("cached file changed before hashing")
        if "sha256" in file_spec:
            algorithm = "sha256"
            digest = hashlib.sha256()
            expected_digest = file_spec["sha256"]
        else:
            algorithm = "git_blob_sha1"
            digest = hashlib.sha1()
            digest.update(f"blob {expected_size}\0".encode("ascii"))
            expected_digest = file_spec["git_blob_sha1"]
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
        descriptor_after = os.fstat(handle.fileno())
        if _stat_identity(descriptor_after) != _stat_identity(descriptor_before):
            raise AssetIntegrityError("cached file changed while hashing")
        observed_digest = digest.hexdigest()
    finally:
        handle.close()
    try:
        path_after = cache_path.stat()
    except OSError as exc:
        raise AssetIntegrityError(f"cached file disappeared after hashing: {exc}") from exc
    if _stat_identity(path_after) != _stat_identity(path_before):
        raise AssetIntegrityError("cached file path changed while hashing")
    if observed_digest != expected_digest:
        raise AssetIntegrityError(
            f"{algorithm} mismatch: expected {expected_digest}, observed {observed_digest}"
        )
    return {"algorithm": algorithm, "digest": observed_digest, "size": observed_size}


def _huggingface_downloader(timeout: float) -> Callable[..., str]:
    timeout_text = str(timeout)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = timeout_text
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = timeout_text
    try:
        import huggingface_hub.constants as hub_constants
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required; install the pinned reproduction environment first"
        ) from exc
    # The constants are initialized at import time, so update them for callers that imported the hub earlier.
    hub_constants.HF_HUB_ETAG_TIMEOUT = timeout
    hub_constants.HF_HUB_DOWNLOAD_TIMEOUT = timeout
    return hf_hub_download


def _fetch_one_file(
    asset: Mapping[str, Any],
    file_spec: Mapping[str, Any],
    *,
    cache_dir: str | Path | None,
    download: bool,
    retries: int,
    timeout: float,
    downloader: Callable[..., str],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    if file_spec.get("metadata_status") == "METADATA_UNAVAILABLE":
        return {
            "attempts": 0,
            "error": "required size and LFS SHA-256 metadata are unavailable; training gate remains BLOCKED",
            "path": file_spec["path"],
            "status": "metadata_unavailable",
        }
    total_attempts = retries if download else 1
    last_error: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            cached_path = downloader(
                repo_id=asset["repo_id"],
                filename=file_spec["path"],
                repo_type=asset["repo_type"],
                revision=asset["revision"],
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=not download,
                etag_timeout=timeout,
                force_download=download and attempt > 1,
            )
            integrity = verify_cached_file(cached_path, file_spec)
            return {
                "_cached_path": str(cached_path),
                "attempts": attempt,
                "integrity": integrity,
                "path": file_spec["path"],
                "status": "verified",
            }
        except Exception as exc:  # Hub exceptions vary between huggingface-hub releases.
            last_error = exc
            if attempt < total_attempts:
                sleep(min(2 ** (attempt - 1), 8))
    assert last_error is not None
    return {
        "attempts": total_attempts,
        "error": f"{type(last_error).__name__}: {last_error}",
        "path": file_spec["path"],
        "status": "missing_or_invalid",
    }


def _materialize_qwen35_snapshot(
    asset: Mapping[str, Any],
    file_results: Sequence[Mapping[str, Any]],
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / asset["asset_id"]
    if destination.exists():
        raise AssetIntegrityError(
            f"materialized snapshot destination already exists: {destination}; verify or remove it explicitly"
        )
    staging = root / f".{asset['asset_id']}.staging-{os.getpid()}"
    if staging.exists():
        raise AssetIntegrityError(f"stale materialization staging directory exists: {staging}")
    staging.mkdir()

    manifest_files: dict[str, dict[str, Any]] = {}
    try:
        for file_spec, result in zip(asset["files"], file_results, strict=True):
            if result["status"] != "verified" or "_cached_path" not in result:
                raise AssetIntegrityError(
                    f"cannot materialize incomplete cached file {file_spec['path']!r}"
                )
            source = Path(result["_cached_path"])
            verify_cached_file(source, file_spec)
            target = staging.joinpath(*PurePosixPath(file_spec["path"]).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source.resolve(strict=True), target)
            except OSError:
                shutil.copy2(source, target)
            verify_cached_file(target, file_spec)
            sha256 = _stream_digest(target, "sha256")
            manifest_files[file_spec["path"]] = {
                "sha256": sha256,
                "size": file_spec["size"],
            }

        unsigned_manifest = {
            "files": dict(sorted(manifest_files.items())),
            "model_id": asset["repo_id"],
            "revision": asset["revision"],
            "schema_version": 1,
        }
        manifest = {
            **unsigned_manifest,
            "manifest_sha256": canonical_json_sha256(unsigned_manifest),
        }
        manifest_path = staging / "qwen35_snapshot_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "path": str(destination),
        "status": "complete",
    }


def prefetch_assets(
    manifest: Mapping[str, Any],
    *,
    cache_dir: str | Path | None = None,
    download: bool = False,
    retries: int = 3,
    timeout: float = 30.0,
    asset_ids: Sequence[str] | None = None,
    materialize_dir: str | Path | None = None,
    downloader: Callable[..., str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    validate_asset_manifest(manifest)
    if retries < 1:
        raise ValueError("retries must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    available = {asset["asset_id"]: asset for asset in manifest["assets"]}
    selected_ids = list(asset_ids) if asset_ids else list(available)
    unknown_ids = sorted(set(selected_ids) - set(available))
    if unknown_ids:
        raise AssetManifestError(f"unknown asset IDs: {unknown_ids}")
    if len(selected_ids) != len(set(selected_ids)):
        raise AssetManifestError("asset selection contains duplicate IDs")
    if downloader is None:
        downloader = _huggingface_downloader(timeout)

    asset_results: list[dict[str, Any]] = []
    complete = True
    for asset_id in selected_ids:
        asset = available[asset_id]
        file_results = [
            _fetch_one_file(
                asset,
                file_spec,
                cache_dir=cache_dir,
                download=download,
                retries=retries,
                timeout=timeout,
                downloader=downloader,
                sleep=sleep,
            )
            for file_spec in asset["files"]
        ]
        asset_complete = all(item["status"] == "verified" for item in file_results)
        materialized_snapshot = None
        if asset_complete and materialize_dir is not None and asset["kind"] == "model_and_tokenizer":
            try:
                materialized_snapshot = _materialize_qwen35_snapshot(
                    asset,
                    file_results,
                    materialize_dir,
                )
            except Exception as exc:
                asset_complete = False
                materialized_snapshot = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": "incomplete",
                }
        for result in file_results:
            cached_path = result.pop("_cached_path", None)
            if cached_path is not None:
                result["cached_path"] = str(Path(cached_path).resolve(strict=True))
        complete = complete and asset_complete
        asset_result = {
            "asset_id": asset_id,
            "files": file_results,
            "repo_id": asset["repo_id"],
            "revision": asset["revision"],
            "status": "complete" if asset_complete else "incomplete",
        }
        if materialized_snapshot is not None:
            asset_result["materialized_snapshot"] = materialized_snapshot
        asset_results.append(asset_result)

    return {
        "assets": asset_results,
        "manifest_sha256": manifest["manifest_sha256"],
        "mode": "download_and_verify" if download else "offline_verify",
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "incomplete",
    }


def write_json_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    serialized = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    try:
        temporary_path.write_text(serialized, encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "environment" / "reproduction-assets.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--asset", action="append", dest="asset_ids")
    parser.add_argument("--download", action="store_true", help="explicitly allow network downloads")
    parser.add_argument(
        "--materialize-dir",
        type=Path,
        help="create self-contained Qwen snapshots with loader-compatible manifests",
    )
    parser.add_argument("--retries", type=int, default=3, help="total attempts per file in download mode")
    parser.add_argument("--timeout", type=float, default=30.0, help="Hub metadata and transfer timeout in seconds")
    parser.add_argument("--report", type=Path, help="atomically write the JSON verification report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_asset_manifest(args.manifest)
        report = prefetch_assets(
            manifest,
            cache_dir=args.cache_dir,
            download=args.download,
            retries=args.retries,
            timeout=args.timeout,
            asset_ids=args.asset_ids,
            materialize_dir=args.materialize_dir,
        )
    except Exception as exc:
        report = {
            "error": f"{type(exc).__name__}: {exc}",
            "mode": "download_and_verify" if args.download else "offline_verify",
            "schema_version": SCHEMA_VERSION,
            "status": "invalid",
        }
    if args.report is not None:
        write_json_report(args.report, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
