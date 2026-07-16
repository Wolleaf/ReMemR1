import json
import os
from contextlib import nullcontext

import pytest

from scripts.cloud import resolve_configs


def _patch_composer(monkeypatch, data_root):
    import hydra
    from omegaconf import OmegaConf

    def bundle(root, relative):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        return path, {"manifest_sha256": "a" * 64}

    monkeypatch.setattr(resolve_configs, "_bundle", bundle)
    monkeypatch.setattr(hydra, "initialize_config_dir", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        hydra,
        "compose",
        lambda *, config_name, overrides, return_hydra_config: {
            "config_name": config_name,
            "overrides": list(overrides),
        },
    )
    monkeypatch.setattr(OmegaConf, "resolve", lambda config: None)
    monkeypatch.setattr(
        OmegaConf,
        "to_yaml",
        lambda config, **kwargs: json.dumps(config, sort_keys=True) + "\n",
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
    assert len(second["configs"]) == len(resolve_configs.CONFIG_NAMES)
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
