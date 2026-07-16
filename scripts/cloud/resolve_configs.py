"""Compose every reproduction config with sealed runtime data identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from taskutils.data_synthesis.reproduction_builder import validate_artifact_bundle


PLACEHOLDERS = {character * 64 for character in "0123"}
CONFIG_NAMES = (
    "g0_qwen35_08b",
    "g1_qwen35_2b_step1",
    "g1_qwen35_2b_resume2",
    "g2a_qwen35_4b",
    "g2b_qwen35_4b_step1",
    "g2b_qwen35_4b_resume2",
    "b_pilot_qwen35_4b",
    "c_pilot_qwen35_4b",
    "b40_qwen35_4b",
    "c40_qwen35_4b",
    "b80_qwen35_4b",
    "c80_qwen35_4b",
    "eval_qwen35_4b",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_without_symlinks(value: Path, label: str) -> Path:
    candidate = Path(os.path.abspath(value.expanduser()))
    for component in [*reversed(candidate.parents), candidate]:
        if component.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {component}")
    return candidate.resolve(strict=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bundle(data_root: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    path = _resolve_without_symlinks(data_root / relative, f"bundle {relative}")
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"bundle escaped data root: {relative}") from exc
    manifest = dict(validate_artifact_bundle(path))
    digest = manifest["manifest_sha256"]
    if digest in PLACEHOLDERS:
        raise ValueError(f"bundle retained a placeholder digest: {path}")
    return path, manifest


def _validate_existing_output(data_root: Path, output: Path) -> dict[str, Any]:
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"resolved config output is not a regular directory: {output}")
    verification = output.with_name(f".{output.name}.verification-{os.getpid()}")
    if verification.exists():
        raise FileExistsError(f"stale config verification directory exists: {verification}")
    try:
        generated = compose_all(data_root, verification)
        expected_names = {"index.json", *(f"{name}.yaml" for name in CONFIG_NAMES)}
        observed_names = set()
        for path in output.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"resolved config output contains an unsafe entry: {path}")
            observed_names.add(path.name)
        if observed_names != expected_names:
            raise ValueError("resolved config output file set changed")

        expected_configs: dict[str, Any] = {}
        for name, record in generated["configs"].items():
            filename = f"{name}.yaml"
            if (output / filename).read_bytes() != (verification / filename).read_bytes():
                raise ValueError(f"resolved config bytes changed: {name}")
            expected_configs[name] = {
                **record,
                "path": str((output / filename).resolve(strict=True)),
            }
        expected_index = {
            "configs": expected_configs,
            "data_root": str(data_root),
            "schema_version": 1,
            "status": "resolved",
        }
        observed_index = json.loads((output / "index.json").read_text(encoding="utf-8"))
        if observed_index != expected_index:
            raise ValueError("resolved config index changed")
        return {**expected_index, "index_sha256": _sha256(output / "index.json")}
    finally:
        shutil.rmtree(verification, ignore_errors=True)


def compose_all(data_root: Path, output: Path) -> dict[str, Any]:
    data_root = _resolve_without_symlinks(data_root, "data root")
    final_output = Path(os.path.abspath(output.expanduser()))
    if final_output.is_symlink():
        raise ValueError(f"resolved config output must not be a symlink: {final_output}")
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError("hydra-core and omegaconf are required") from exc

    root = Path(__file__).resolve().parents[2]
    config_root = root / "verl/trainer/config"
    g0_train, g0_train_manifest = _bundle(data_root, "gates/g0/train")
    g0_val, g0_val_manifest = _bundle(data_root, "gates/g0/validation")
    g1_train, g1_train_manifest = _bundle(data_root, "gates/g1/train")
    g1_val, g1_val_manifest = _bundle(data_root, "gates/g1/validation")
    formal_train, formal_train_manifest = _bundle(data_root, "formal/train")
    formal_val, formal_val_manifest = _bundle(data_root, "formal/validation")
    hotpot_eval, hotpot_eval_manifest = _bundle(data_root, "formal/eval/hotpotqa")
    wiki_eval, wiki_eval_manifest = _bundle(
        data_root, "formal/eval/2wikimultihopqa"
    )

    gate_values = {
        "g0": (g0_train, g0_train_manifest, g0_val, g0_val_manifest),
        "g1": (g1_train, g1_train_manifest, g1_val, g1_val_manifest),
        "formal": (
            formal_train,
            formal_train_manifest,
            formal_val,
            formal_val_manifest,
        ),
    }
    final_output.parent.mkdir(parents=True, exist_ok=True)
    if final_output.exists():
        return _validate_existing_output(data_root, final_output)
    output = final_output.with_name(f".{final_output.name}.staging-{os.getpid()}")
    if output.exists():
        raise FileExistsError(f"stale resolved config staging directory exists: {output}")
    output.mkdir()

    records: dict[str, Any] = {}
    try:
        with initialize_config_dir(config_dir=str(config_root), version_base=None):
            for name in CONFIG_NAMES:
                profile = "g0" if name.startswith("g0_") else "g1" if name.startswith("g1_") else "formal"
                train_path, train_manifest, val_path, val_manifest = gate_values[profile]
                overrides = [
                    f"data.train_files={train_path / 'train.parquet'}",
                    f"data.val_files={val_path / 'train.parquet'}",
                    f"reproduction.data_manifest_sha256={train_manifest['manifest_sha256']}",
                    f"reproduction.val_data_manifest_sha256={val_manifest['manifest_sha256']}",
                ]
                if name == "eval_qwen35_4b":
                    overrides.extend(
                        [
                            f"reproduction_evaluation.datasets.hotpotqa.bundle_dir={hotpot_eval}",
                            "reproduction_evaluation.datasets.hotpotqa.manifest_sha256="
                            f"{hotpot_eval_manifest['manifest_sha256']}",
                            "reproduction_evaluation.datasets.2wikimultihopqa.bundle_dir="
                            f"{wiki_eval}",
                            "reproduction_evaluation.datasets.2wikimultihopqa.manifest_sha256="
                            f"{wiki_eval_manifest['manifest_sha256']}",
                        ]
                    )
                config = compose(
                    config_name=f"reproduction/{name}",
                    overrides=overrides,
                    return_hydra_config=False,
                )
                OmegaConf.resolve(config)
                serialized = OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
                if any(value in serialized for value in PLACEHOLDERS):
                    raise ValueError(f"resolved config {name} still contains a placeholder")
                destination = output / f"{name}.yaml"
                _atomic_text(destination, serialized)
                records[name] = {
                    "overrides": overrides,
                    "path": str((final_output / destination.name).resolve()),
                    "sha256": _sha256(destination),
                }

        index = {
            "configs": records,
            "data_root": str(data_root),
            "schema_version": 1,
            "status": "resolved",
        }
        index_path = output / "index.json"
        _atomic_text(
            index_path,
            json.dumps(index, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        )
        index_sha256 = _sha256(index_path)
        os.replace(output, final_output)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return {**index, "index_sha256": index_sha256}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = compose_all(args.data_root, args.output)
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "blocked"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
