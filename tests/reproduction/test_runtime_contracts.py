import ast
import builtins
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DP_ACTOR_PATH = REPO_ROOT / "verl" / "workers" / "actor" / "dp_actor.py"
MAIN_PPO_PATH = REPO_ROOT / "verl" / "trainer" / "main_ppo.py"
RAY_TRAINER_PATH = REPO_ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _load_isolated_function(path, name, namespace=None):
    function = _find_function(_parse(path), name)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {} if namespace is None else namespace
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_flash_attn_padding_import_is_not_module_level():
    tree = _parse(DP_ACTOR_PATH)

    module_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("flash_attn")
        )
        for node in module_imports
    )

    loader = _find_function(tree, "_load_flash_attn_padding")
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "flash_attn.bert_padding"
        for node in ast.walk(loader)
    )


def test_actor_without_remove_padding_does_not_load_flash_attn():
    tree = _parse(DP_ACTOR_PATH)
    actor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor"
    )
    init = next(
        node
        for node in actor.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    isolated_actor = ast.ClassDef(
        name="IsolatedActor",
        bases=[ast.Name(id="BasePPOActor", ctx=ast.Load())],
        keywords=[],
        body=[init],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[isolated_actor], type_ignores=[]))

    class BasePPOActor:
        def __init__(self, config):
            self.config = config

    class Config(dict):
        ulysses_sequence_parallel_size = 1

    def fail_if_loaded():
        raise AssertionError("FlashAttention loader must stay lazy")

    namespace = {
        "BasePPOActor": BasePPOActor,
        "nn": SimpleNamespace(Module=object),
        "torch": SimpleNamespace(optim=SimpleNamespace(Optimizer=object)),
        "verl_F": SimpleNamespace(entropy_from_logits=object()),
        "_load_flash_attn_padding": fail_if_loaded,
    }
    exec(compile(module, str(DP_ACTOR_PATH), "exec"), namespace)

    instance = namespace["IsolatedActor"](
        Config(use_remove_padding=False, use_torch_compile=False),
        actor_module=object(),
    )

    assert instance._flash_attn_padding is None


def test_missing_flash_attn_has_actionable_remove_padding_error(monkeypatch):
    loader = _load_isolated_function(DP_ACTOR_PATH, "_load_flash_attn_padding")
    original_import = builtins.__import__

    def import_without_flash_attn(name, *args, **kwargs):
        if name.startswith("flash_attn"):
            raise ModuleNotFoundError("flash_attn is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_flash_attn)

    with pytest.raises(ImportError, match=r"flash-attn.*use_remove_padding=True"):
        loader()


class _FakeRunMethod:
    def remote(self, config):
        return ("run", config)


class _FakeTaskRunner:
    @staticmethod
    def remote():
        return SimpleNamespace(run=_FakeRunMethod())


class _FakeRay:
    def __init__(self):
        self.init_kwargs = None

    def is_initialized(self):
        return False

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def get(self, value):
        return value


def _run_ppo_with_fake_ray():
    fake_ray = _FakeRay()
    run_ppo = _load_isolated_function(
        MAIN_PPO_PATH,
        "run_ppo",
        {
            "os": os,
            "ray": fake_ray,
            "TaskRunner": _FakeTaskRunner,
            "apply_reproduction_seed_contract": lambda config: None,
        },
    )
    config = SimpleNamespace(ray_init=SimpleNamespace(num_cpus=3))
    run_ppo(config)
    return fake_ray.init_kwargs


def test_ray_tmpdir_is_propagated_to_head_and_workers(monkeypatch, tmp_path):
    ray_tmpdir = str(tmp_path / "ray")
    monkeypatch.setenv("RAY_TMPDIR", ray_tmpdir)

    init_kwargs = _run_ppo_with_fake_ray()

    assert init_kwargs["_temp_dir"] == ray_tmpdir
    assert init_kwargs["runtime_env"]["env_vars"]["RAY_TMPDIR"] == ray_tmpdir
    assert init_kwargs["num_cpus"] == 3


def test_ray_tmpdir_uses_ray_default_when_environment_is_unset(monkeypatch):
    monkeypatch.delenv("RAY_TMPDIR", raising=False)

    init_kwargs = _run_ppo_with_fake_ray()

    assert "_temp_dir" not in init_kwargs
    assert "RAY_TMPDIR" not in init_kwargs["runtime_env"]["env_vars"]
    assert "/tmp/ray" not in MAIN_PPO_PATH.read_text(encoding="utf-8")


def test_training_started_marker_binds_the_first_rollout(tmp_path):
    writes = []

    def atomic_write(path, value):
        writes.append((Path(path), value))

    publish = _load_isolated_function(
        RAY_TRAINER_PATH,
        "publish_training_started",
        {"Path": Path, "atomic_write_text": atomic_write, "json": json},
    )

    class ReproductionConfig(dict):
        __getattr__ = dict.__getitem__

    config = ReproductionConfig(
        runtime_telemetry_path=str(tmp_path / "telemetry.json"),
        runtime_attempt_id="attempt-1",
        runtime_binding_sha256="a" * 64,
        sealed_config_id="g0_qwen35_08b",
        sealed_config_sha256="b" * 64,
    )
    marker = publish(config, 1)

    assert marker == tmp_path / "training-started.json"
    assert writes[0][0] == marker
    assert json.loads(writes[0][1]) == {
        "event": "first-rollout-started",
        "global_step": 1,
        "runtime_attempt_id": "attempt-1",
        "runtime_binding_sha256": "a" * 64,
        "schema_version": 1,
        "sealed_config_id": "g0_qwen35_08b",
        "sealed_config_sha256": "b" * 64,
    }


def test_training_started_marker_is_published_before_generation():
    tree = _parse(RAY_TRAINER_PATH)
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer"
    )
    fit = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "fit"
    )
    publish_call = next(
        node
        for node in ast.walk(fit)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_training_started"
    )
    rollout_call = next(
        node
        for node in ast.walk(fit)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_llm_loop_revisit"
    )
    coordinate_call = next(
        node
        for node in ast.walk(fit)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_attach_recurrent_rollout_coordinates"
    )
    assert coordinate_call.lineno < publish_call.lineno < rollout_call.lineno
    assert rollout_call.lineno - publish_call.lineno <= 6
