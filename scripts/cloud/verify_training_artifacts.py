"""Verify one gate checkpoint and adapter against its sealed runtime identity."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


class ArtifactGateError(RuntimeError):
    """Raised when training artifacts are valid in isolation but have the wrong identity."""


def _field(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ArtifactGateError(f"resolved checkpoint config lacks {dotted_path}")
        current = current[component]
    return current


def _resolved_file(value: Any, dotted_path: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactGateError(f"checkpoint resolved config lacks a path at {dotted_path}")
    candidate = Path(os.path.abspath(Path(value).expanduser()))
    for component in [*reversed(candidate.parents), candidate]:
        if component.is_symlink():
            raise ArtifactGateError(
                f"checkpoint path contains a symlink at {dotted_path}: {component}"
            )
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise ArtifactGateError(
            f"checkpoint resolved path is unavailable at {dotted_path}: {exc}"
        ) from exc
    if not path.is_file():
        raise ArtifactGateError(f"checkpoint resolved path is not a file at {dotted_path}")
    return path


def verify_training_artifacts(
    checkpoint_dir: Path,
    adapter_dir: Path,
    *,
    expected_step: int,
    expected_train_file: Path,
    expected_validation_file: Path,
    expected_resolved_config: Path,
    expected_train_manifest: str,
    expected_validation_manifest: str,
    expected_base_model: str,
    expected_revision: str,
    expected_resume_from: Path | None,
) -> dict[str, Any]:
    from verl.utils.checkpoint.reproduction import (
        validate_adapter_export,
        validate_resolved_config_compatibility,
        verify_reproduction_checkpoint_directory,
    )

    _, state = verify_reproduction_checkpoint_directory(checkpoint_dir)
    adapter = validate_adapter_export(adapter_dir)
    expected_state = {
        "base_model_id": expected_base_model,
        "base_model_revision": expected_revision,
        "data_manifest_sha256": expected_train_manifest,
        "global_step": expected_step,
    }
    for name, expected in expected_state.items():
        if getattr(state, name) != expected:
            raise ArtifactGateError(f"checkpoint {name} differs from the gate identity")
    expected_config = {
        "actor_rollout_ref.model.path": expected_base_model,
        "actor_rollout_ref.model.revision": expected_revision,
        "actor_rollout_ref.actor.optim.total_training_steps": expected_step,
        "critic.optim.total_training_steps": expected_step,
        "reproduction.data_manifest_sha256": expected_train_manifest,
        "reproduction.template_revision": "rememr1-template-v1",
        "reproduction.val_data_manifest_sha256": expected_validation_manifest,
        "trainer.total_training_steps": expected_step,
    }
    for path, expected in expected_config.items():
        if _field(state.resolved_config, path) != expected:
            raise ArtifactGateError(f"checkpoint resolved config differs at {path}")

    expected_files = {
        "data.train_files": _resolved_file(
            str(expected_train_file), "expected_train_file"
        ),
        "data.val_files": _resolved_file(
            str(expected_validation_file), "expected_validation_file"
        ),
    }
    for path, expected in expected_files.items():
        if _resolved_file(_field(state.resolved_config, path), path) != expected:
            raise ArtifactGateError(f"checkpoint resolved config differs at {path}")

    config_path = _resolved_file(
        str(expected_resolved_config),
        "expected_resolved_config",
    )
    try:
        sealed_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ArtifactGateError(f"cannot load sealed resolved config: {exc}") from exc
    if not isinstance(sealed_config, Mapping) or not sealed_config:
        raise ArtifactGateError("sealed resolved config must contain a mapping")
    try:
        validate_resolved_config_compatibility(
            sealed_config,
            state.resolved_config,
            allowed_drift_paths=(
                "reproduction.adapter_export_dir",
                "reproduction.pilot_evidence_path",
                "reproduction.runtime_attempt_id",
                "reproduction.runtime_binding_sha256",
                "reproduction.runtime_bound_evidence_path",
                "reproduction.runtime_telemetry_path",
                "reproduction.sealed_config_id",
                "reproduction.sealed_config_sha256",
                "reproduction.step_zero_fingerprint_path",
                "reproduction.step_zero_reference_path",
                "trainer.default_local_dir",
                "trainer.rollout_data_dir",
                "trainer.resume_from_path",
                "trainer.validation_data_dir",
            ),
        )
    except Exception as exc:
        raise ArtifactGateError(f"checkpoint differs from the sealed CPU config: {exc}") from exc

    resume_mode = _field(state.resolved_config, "trainer.resume_mode")
    resume_path = _field(state.resolved_config, "trainer.resume_from_path")
    if expected_resume_from is None:
        if resume_mode != "disable" or resume_path is not None:
            raise ArtifactGateError("fresh gate checkpoint unexpectedly records resume state")
    else:
        expected_resume = expected_resume_from.expanduser().resolve(strict=True)
        if resume_mode != "resume_path" or not isinstance(resume_path, str):
            raise ArtifactGateError("resumed gate checkpoint lacks resume_path mode")
        if Path(resume_path).expanduser().resolve(strict=True) != expected_resume:
            raise ArtifactGateError("checkpoint resume predecessor differs from the gate")

    if (
        adapter.global_step != expected_step
        or adapter.base_model_id != expected_base_model
        or adapter.base_model_revision != expected_revision
        or adapter.tokenizer_id != expected_base_model
        or adapter.tokenizer_revision != expected_revision
        or adapter.template_revision != "rememr1-template-v1"
        or adapter.source_extra_state_sha256 != state.sha256
    ):
        raise ArtifactGateError("adapter is not the export of the verified checkpoint")
    for field in (
        "adapter_state_sha256",
        "adapter_tensor_keys",
        "lora_config_sha256",
        "lora_target_sha256",
        "text_mapping_sha256",
    ):
        if getattr(adapter, field) != getattr(state, field):
            raise ArtifactGateError(
                f"adapter semantic state differs from the checkpoint at {field}"
            )
    return {
        "adapter_metadata_sha256": adapter.sha256,
        "checkpoint_extra_state_sha256": state.sha256,
        "global_step": state.global_step,
        "status": "verified",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-train-file", type=Path, required=True)
    parser.add_argument("--expected-validation-file", type=Path, required=True)
    parser.add_argument("--expected-resolved-config", type=Path, required=True)
    parser.add_argument("--expected-train-manifest", required=True)
    parser.add_argument("--expected-validation-manifest", required=True)
    parser.add_argument("--expected-base-model", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-resume-from", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_training_artifacts(
            args.checkpoint_dir,
            args.adapter_dir,
            expected_step=args.expected_step,
            expected_train_file=args.expected_train_file,
            expected_validation_file=args.expected_validation_file,
            expected_resolved_config=args.expected_resolved_config,
            expected_train_manifest=args.expected_train_manifest,
            expected_validation_manifest=args.expected_validation_manifest,
            expected_base_model=args.expected_base_model,
            expected_revision=args.expected_revision,
            expected_resume_from=args.expected_resume_from,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "blocked"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
