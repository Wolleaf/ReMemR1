import importlib.util
import ast
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "verl" / "utils" / "reproducibility.py"
)
MAIN_PPO_PATH = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "main_ppo.py"


def _load_module():
    name = "_reproduction_seed_contract_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_seed_namespaces_are_stable_and_explicit():
    module = _load_module()

    seeds = module.ReproductionSeeds.from_run_seed(42)

    assert seeds.to_dict() == {
        "run": 42,
        "data": 1938833696,
        "model_init": 1767863184,
        "rollout": 2924589348,
    }
    assert seeds.rollout_for(0, 0, 0) == 3571890595
    assert seeds.rollout_for(1, 2, 3) == 1705356498


def test_rollout_coordinates_do_not_share_a_seed():
    module = _load_module()
    seeds = module.ReproductionSeeds.from_run_seed(42)

    derived = {
        seeds.rollout_for(step, sample, trajectory)
        for step in range(2)
        for sample in range(2)
        for trajectory in range(2)
    }

    assert len(derived) == 8


def test_recurrent_turns_have_independent_rollout_streams():
    module = _load_module()
    seeds = module.ReproductionSeeds.from_run_seed(42)

    derived = {
        seeds.rollout_generation_for(3, 2, 1, turn)
        for turn in range(4)
    }

    assert len(derived) == 4
    assert seeds.rollout_generation_for(3, 2, 1, 0) == 4040670340


def test_interleaved_trajectory_coordinates_match_dataproto_repeat_order():
    torch = pytest.importorskip("torch")
    module = _load_module()

    coordinates = module.make_rollout_coordinate_tensors(
        8,
        4,
        sample_offset=10,
    )

    assert coordinates[module.ROLLOUT_SAMPLE_INDEX_KEY].tolist() == [
        10,
        10,
        10,
        10,
        11,
        11,
        11,
        11,
    ]
    assert coordinates[module.ROLLOUT_TRAJECTORY_INDEX_KEY].tolist() == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]
    assert all(value.dtype == torch.long for value in coordinates.values())


@pytest.mark.parametrize("bad_seed", [-1, 2**32, True, 1.5])
def test_invalid_run_seed_fails_closed(bad_seed):
    module = _load_module()

    with pytest.raises((TypeError, ValueError)):
        module.ReproductionSeeds.from_run_seed(bad_seed)


def test_seed_process_covers_python_numpy_torch_and_available_cuda(monkeypatch):
    module = _load_module()
    calls = []

    numpy_module = ModuleType("numpy")
    numpy_module.random = SimpleNamespace(seed=lambda value: calls.append(("numpy", value)))
    torch_module = ModuleType("torch")
    torch_module.manual_seed = lambda value: calls.append(("torch", value))
    torch_module.cuda = SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda value: calls.append(("cuda", value)),
    )
    monkeypatch.setitem(sys.modules, "numpy", numpy_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    module.seed_process(123)
    first = random.random()
    module.seed_process(123)
    second = random.random()

    assert first == second
    assert calls == [
        ("numpy", 123),
        ("torch", 123),
        ("cuda", 123),
        ("numpy", 123),
        ("torch", 123),
        ("cuda", 123),
    ]


def _load_seed_wiring_function(monkeypatch):
    omegaconf = pytest.importorskip("omegaconf")
    seed_module = _load_module()
    fake_verl = ModuleType("verl")
    fake_verl.__path__ = []
    fake_utils = ModuleType("verl.utils")
    fake_utils.__path__ = []
    fake_seed_module = ModuleType("verl.utils.reproducibility")
    fake_seed_module.ReproductionSeeds = seed_module.ReproductionSeeds
    fake_seed_module.validate_seed = seed_module.validate_seed
    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "verl.utils.reproducibility", fake_seed_module)

    tree = ast.parse(MAIN_PPO_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_reproduction_seed_contract"
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(MAIN_PPO_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["apply_reproduction_seed_contract"], omegaconf


def test_run_seed_wiring_populates_data_model_and_rollout(monkeypatch):
    apply_contract, omegaconf = _load_seed_wiring_function(monkeypatch)
    config = omegaconf.OmegaConf.create(
        {
            "reproduction": {
                "run_seed": 42,
                "data_seed": None,
                "model_init_seed": None,
                "rollout_seed": None,
            },
            "data": {"seed": 1},
            "actor_rollout_ref": {
                "model": {"model_init_seed": None},
                "rollout": {"seed": None},
            },
        }
    )

    seeds = apply_contract(config)

    assert config.data.seed == seeds.data == 1938833696
    assert config.actor_rollout_ref.model.model_init_seed == seeds.model_init == 1767863184
    assert config.actor_rollout_ref.rollout.seed == seeds.rollout == 2924589348


def test_formal_data_cannot_disable_all_reproduction_guards_by_omitting_seed(
    monkeypatch,
):
    apply_contract, omegaconf = _load_seed_wiring_function(monkeypatch)
    config = omegaconf.OmegaConf.create(
        {
            "reproduction": {"formal_data": True, "run_seed": None},
        }
    )

    with pytest.raises(ValueError, match="formal_data=true requires"):
        apply_contract(config)


def test_run_seed_wiring_rejects_an_inconsistent_explicit_derivation(monkeypatch):
    apply_contract, omegaconf = _load_seed_wiring_function(monkeypatch)
    config = omegaconf.OmegaConf.create(
        {
            "reproduction": {
                "run_seed": 42,
                "data_seed": 7,
                "model_init_seed": None,
                "rollout_seed": None,
            },
            "data": {},
            "actor_rollout_ref": {"model": {}, "rollout": {}},
        }
    )

    with pytest.raises(ValueError, match="data_seed"):
        apply_contract(config)


@pytest.mark.parametrize("bad_seed", [True, 42.0, "42"])
def test_run_seed_wiring_does_not_coerce_invalid_types(monkeypatch, bad_seed):
    apply_contract, omegaconf = _load_seed_wiring_function(monkeypatch)
    config = omegaconf.OmegaConf.create(
        {
            "reproduction": {
                "run_seed": bad_seed,
                "data_seed": None,
                "model_init_seed": None,
                "rollout_seed": None,
            },
            "data": {},
            "actor_rollout_ref": {"model": {}, "rollout": {}},
        }
    )

    with pytest.raises(TypeError, match="run_seed must be an int"):
        apply_contract(config)


@pytest.mark.parametrize("bad_seed", [True, 1938833696.0, "1938833696"])
def test_explicit_derived_seed_does_not_accept_coercible_types(monkeypatch, bad_seed):
    apply_contract, omegaconf = _load_seed_wiring_function(monkeypatch)
    config = omegaconf.OmegaConf.create(
        {
            "reproduction": {
                "run_seed": 42,
                "data_seed": bad_seed,
                "model_init_seed": None,
                "rollout_seed": None,
            },
            "data": {},
            "actor_rollout_ref": {"model": {}, "rollout": {}},
        }
    )

    with pytest.raises(TypeError, match=r"reproduction\.data_seed must be an int"):
        apply_contract(config)
