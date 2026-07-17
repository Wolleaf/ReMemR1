import hashlib
import json
import os
from contextlib import nullcontext

import pytest

from scripts.cloud import resolve_configs


def _patch_composer(monkeypatch, data_root):
    import hydra
    from omegaconf import OmegaConf

    calls = []

    def bundle(root, relative):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(relative.encode("ascii")).hexdigest()
        return path, {"manifest_sha256": digest}

    original_to_container = OmegaConf.to_container

    def compose(*, config_name, overrides, return_hydra_config):
        call = {
            "config_name": config_name,
            "overrides": list(overrides),
        }
        calls.append(call)
        return OmegaConf.create(call)

    monkeypatch.setattr(resolve_configs, "_bundle", bundle)
    monkeypatch.setattr(hydra, "initialize_config_dir", lambda **kwargs: nullcontext())
    monkeypatch.setattr(hydra, "compose", compose)
    monkeypatch.setattr(OmegaConf, "resolve", lambda config: None)
    monkeypatch.setattr(
        OmegaConf,
        "to_yaml",
        lambda config, **kwargs: json.dumps(
            original_to_container(config, resolve=True), sort_keys=True
        )
        + "\n",
    )
    return calls


def test_active_inventory_is_exactly_three_gates_twenty_eight_profiles_and_two_evals():
    expected = {
        *resolve_configs.GATE_CONFIG_NAMES,
        *(
            f"{source}_{profile}"
            for source in resolve_configs.TRAINING_SOURCE_NAMES
            for profile in resolve_configs.OFFLOAD_PROFILES
        ),
        *resolve_configs.EVAL_CONFIG_NAMES,
    }
    assert len(resolve_configs.GATE_CONFIG_NAMES) == 3
    assert len(resolve_configs.TRAINING_SOURCE_NAMES) == 14
    assert resolve_configs.OFFLOAD_PROFILES == ("r0", "r1")
    assert len(resolve_configs.EVAL_CONFIG_NAMES) == 2
    assert len(resolve_configs.CONFIG_NAMES) == 33
    assert len(set(resolve_configs.CONFIG_NAMES)) == 33
    assert set(resolve_configs.CONFIG_NAMES) == expected
    assert all("4b" not in name.casefold() for name in resolve_configs.CONFIG_NAMES)


def test_compose_all_seals_each_registered_config_and_profile_once(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    calls = _patch_composer(monkeypatch, data_root)
    output = tmp_path / "resolved"

    result = resolve_configs.compose_all(data_root, output)

    assert tuple(result["configs"]) == resolve_configs.CONFIG_NAMES
    assert len(calls) == 33
    assert {path.name for path in output.iterdir()} == {
        "index.json",
        *(f"{name}.yaml" for name in resolve_configs.CONFIG_NAMES),
    }
    for name, source_name, profile in resolve_configs.CONFIG_COMPOSITIONS:
        record = result["configs"][name]
        assert set(record) == {
            "offload_profile",
            "overrides",
            "path",
            "sha256",
            "source_config",
        }
        assert record["source_config"] == source_name
        assert record["offload_profile"] == profile
        payload = json.loads((output / f"{name}.yaml").read_text(encoding="utf-8"))
        assert payload["config_name"] == f"reproduction/{source_name}"
        offload_overrides = [
            value
            for value in payload["overrides"]
            if value.startswith("reproduction/offload@_global_=")
        ]
        assert offload_overrides == (
            []
            if profile is None
            else [f"reproduction/offload@_global_={profile}"]
        )
        assert any(
            value.startswith("data.train_files=") and value.endswith("train.parquet")
            for value in record["overrides"]
        )


def test_length_stress_and_evaluation_use_their_sealed_bundle_namespaces(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _patch_composer(monkeypatch, data_root)
    output = tmp_path / "resolved"

    result = resolve_configs.compose_all(data_root, output)

    for profile in resolve_configs.OFFLOAD_PROFILES:
        record = result["configs"][
            f"g2_length_stress_qwen35_2b_5090_{profile}"
        ]
        overrides = record["overrides"]
        assert any(
            "capacity/length-stress/train/train.parquet"
            in value.replace("\\", "/")
            for value in overrides
        )
        assert any(
            "capacity/length-stress/validation/train.parquet"
            in value.replace("\\", "/")
            for value in overrides
        )
    for name in resolve_configs.EVAL_CONFIG_NAMES:
        overrides = result["configs"][name]["overrides"]
        assert any(
            value.startswith(
                "reproduction_evaluation.datasets.hotpotqa.bundle_dir="
            )
            for value in overrides
        )
        assert any(
            value.startswith(
                "reproduction_evaluation.datasets.2wikimultihopqa.bundle_dir="
            )
            for value in overrides
        )


def test_real_hydra_composes_all_33_configs_with_runtime_data_bindings(
    tmp_path, monkeypatch
):
    pytest.importorskip("hydra", reason="hydra-core is required to resolve configs")
    data_root = tmp_path / "data"
    data_root.mkdir()

    def bundle(root, relative):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        return path, {
            "manifest_sha256": hashlib.sha256(relative.encode("ascii")).hexdigest()
        }

    monkeypatch.setattr(resolve_configs, "_bundle", bundle)
    output = tmp_path / "resolved"

    result = resolve_configs.compose_all(data_root, output)

    assert tuple(result["configs"]) == resolve_configs.CONFIG_NAMES
    assert len(list(output.glob("*.yaml"))) == 33
    assert all(
        placeholder not in path.read_text(encoding="utf-8")
        for path in output.glob("*.yaml")
        for placeholder in resolve_configs.PLACEHOLDERS
    )


def test_existing_resolved_configs_are_reverified_and_tampering_is_rejected(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _patch_composer(monkeypatch, data_root)
    output = tmp_path / "resolved"

    first = resolve_configs.compose_all(data_root, output)
    second = resolve_configs.compose_all(data_root, output)

    assert second == first
    assert len(second["configs"]) == 33
    (output / "g0_qwen35_08b.yaml").write_text("tampered\n", encoding="ascii")
    with pytest.raises(ValueError, match="resolved config bytes changed"):
        resolve_configs.compose_all(data_root, output)


def test_cli_rejects_the_original_output_path_when_it_is_a_symlink(
    tmp_path, monkeypatch, capsys
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _patch_composer(monkeypatch, data_root)
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "resolved-link"
    try:
        os.symlink(target, output, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    result = resolve_configs.main(
        ["--data-root", str(data_root), "--output", str(output)]
    )

    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert "must not be a symlink" in error["error"]
