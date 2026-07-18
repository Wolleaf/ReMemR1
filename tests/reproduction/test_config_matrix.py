from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path, PurePosixPath
import re
import sys

import pytest
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "verl" / "trainer" / "config"
REPRODUCTION_CONFIG_ROOT = CONFIG_ROOT / "reproduction"
BASE_CONFIG = CONFIG_ROOT / "ppo_trainer.yaml"
PROFILE_ID = "rtx5090-32g-qwen35-2b-v1"
PROFILE_ROOT = f"/root/autodl-tmp/rememr1/profiles/{PROFILE_ID}"
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ZERO_SHA256 = "0" * 64
ONE_SHA256 = "1" * 64

CHECKPOINT_MODULE_PATH = (
    REPO_ROOT / "verl" / "utils" / "checkpoint" / "reproduction.py"
)
CHECKPOINT_SPEC = importlib.util.spec_from_file_location(
    "config_checkpoint_contract_under_test",
    CHECKPOINT_MODULE_PATH,
)
checkpoint_contract = importlib.util.module_from_spec(CHECKPOINT_SPEC)
sys.modules[CHECKPOINT_SPEC.name] = checkpoint_contract
CHECKPOINT_SPEC.loader.exec_module(checkpoint_contract)

GATE_CONFIG_NAMES = (
    "g0_qwen35_08b",
    "g1_qwen35_2b_step1",
    "g1_qwen35_2b_resume2",
)
TRAINING_CONTRACTS = {
    "g2a_qwen35_2b_5090": {
        "batch": 1,
        "mini": 1,
        "steps": 1,
        "save_freq": 1,
        "resume_step": None,
        "alpha": 0.8,
    },
    "g2b_qwen35_2b_5090_step1": {
        "batch": 2,
        "mini": 2,
        "steps": 1,
        "save_freq": 1,
        "resume_step": None,
        "alpha": 0.8,
    },
    "g2b_qwen35_2b_5090_resume5": {
        "batch": 2,
        "mini": 2,
        "steps": 5,
        "save_freq": 1,
        "resume_step": 1,
        "alpha": 0.8,
    },
    "g2_length_stress_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 1,
        "save_freq": -1,
        "resume_step": None,
        "alpha": 0.8,
    },
    "b_pilot_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 3,
        "save_freq": 3,
        "resume_step": None,
        "alpha": 1.0,
    },
    "c_pilot_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 3,
        "save_freq": 3,
        "resume_step": None,
        "alpha": 0.8,
    },
    "b20_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 20,
        "save_freq": 20,
        "resume_step": None,
        "alpha": 1.0,
    },
    "c20_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 20,
        "save_freq": 20,
        "resume_step": None,
        "alpha": 0.8,
    },
    "b40_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 40,
        "save_freq": 20,
        "resume_step": 20,
        "alpha": 1.0,
    },
    "c40_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 40,
        "save_freq": 20,
        "resume_step": 20,
        "alpha": 0.8,
    },
    "b60_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 60,
        "save_freq": 20,
        "resume_step": 40,
        "alpha": 1.0,
    },
    "c60_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 60,
        "save_freq": 20,
        "resume_step": 40,
        "alpha": 0.8,
    },
    "b80_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 80,
        "save_freq": 20,
        "resume_step": 60,
        "alpha": 1.0,
    },
    "c80_qwen35_2b_5090": {
        "batch": 2,
        "mini": 2,
        "steps": 80,
        "save_freq": 20,
        "resume_step": 60,
        "alpha": 0.8,
    },
}
EVAL_CONFIG_NAMES = (
    "eval40_qwen35_2b_5090",
    "eval80_qwen35_2b_5090",
)
OFFLOAD_PROFILES = ("r0", "r1")
INACTIVE_4B_CONFIG_NAMES = {
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
}


def _load_child(name: str, stack: tuple[str, ...] = ()) -> DictConfig:
    assert name not in stack, f"cyclic reproduction config defaults: {stack + (name,)}"
    path = REPRODUCTION_CONFIG_ROOT / f"{name}.yaml"
    child = OmegaConf.load(path)
    defaults = OmegaConf.to_container(child.defaults, resolve=False)
    assert isinstance(defaults, list)
    assert defaults[-1] == "_self_"
    del child["defaults"]
    parents = []
    for default in defaults[:-1]:
        if default == "/ppo_trainer":
            continue
        if not isinstance(default, str) and "offload@_global_" in default:
            continue
        prefix = "/reproduction/"
        assert isinstance(default, str) and default.startswith(prefix)
        parents.append(_load_child(default.removeprefix(prefix), stack + (name,)))
    return OmegaConf.merge(*parents, child)


def _compose_source(name: str, profile: str | None = None) -> DictConfig:
    configs = [OmegaConf.load(BASE_CONFIG), _load_child(name)]
    if profile is not None:
        configs.append(
            OmegaConf.load(REPRODUCTION_CONFIG_ROOT / "offload" / f"{profile}.yaml")
        )
    config = OmegaConf.merge(*configs)
    OmegaConf.resolve(config)
    return config


@pytest.fixture(scope="module")
def training_configs() -> dict[tuple[str, str], DictConfig]:
    return {
        (name, profile): _compose_source(name, profile)
        for name in TRAINING_CONTRACTS
        for profile in OFFLOAD_PROFILES
    }


@pytest.fixture(scope="module")
def eval_configs() -> dict[str, DictConfig]:
    return {name: _compose_source(name) for name in EVAL_CONFIG_NAMES}


def _assert_absolute_persistent_path(value: str) -> None:
    path = PurePosixPath(value)
    assert path.is_absolute()
    assert path.parts[:4] == ("/", "root", "autodl-tmp", "rememr1")


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        flattened = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, name))
        return flattened
    if isinstance(value, list):
        return {prefix: tuple(value)}
    return {prefix: value}


def _different_leaf_paths(left: DictConfig, right: DictConfig) -> set[str]:
    left_values = _flatten(OmegaConf.to_container(left, resolve=True))
    right_values = _flatten(OmegaConf.to_container(right, resolve=True))
    assert left_values.keys() == right_values.keys()
    return {key for key in left_values if left_values[key] != right_values[key]}


def test_source_inventory_retains_inactive_4b_and_adds_exact_2b_sources():
    actual = {path.stem for path in REPRODUCTION_CONFIG_ROOT.glob("*.yaml")}
    expected = {
        *GATE_CONFIG_NAMES,
        *INACTIVE_4B_CONFIG_NAMES,
        *TRAINING_CONTRACTS,
        *EVAL_CONFIG_NAMES,
    }
    assert actual == expected
    assert {
        path.stem for path in (REPRODUCTION_CONFIG_ROOT / "offload").glob("*.yaml")
    } == set(OFFLOAD_PROFILES)


def test_training_sources_fail_closed_without_an_offload_overlay():
    for name in TRAINING_CONTRACTS:
        config = OmegaConf.merge(OmegaConf.load(BASE_CONFIG), _load_child(name))
        assert OmegaConf.is_missing(config.reproduction, "offload_profile")


def test_active_gate_configs_expose_the_pilot_evidence_schema():
    for name in GATE_CONFIG_NAMES:
        reproduction = _compose_source(name).reproduction
        assert reproduction.experiment_profile_id == PROFILE_ID
        assert reproduction.offload_profile == "r0"
        assert reproduction.pilot_evidence_path is None


@pytest.mark.parametrize("name", TRAINING_CONTRACTS)
@pytest.mark.parametrize("profile", OFFLOAD_PROFILES)
def test_training_config_resolves_to_registered_2b_shape(
    name, profile, training_configs
):
    config = training_configs[name, profile]
    expected = TRAINING_CONTRACTS[name]
    actor = config.actor_rollout_ref.actor
    ref = config.actor_rollout_ref.ref
    rollout = config.actor_rollout_ref.rollout
    memory = config.recurrent.memory.config

    assert config.reproduction.experiment_profile_id == PROFILE_ID
    assert config.reproduction.offload_profile == profile
    assert config.actor_rollout_ref.model.path == MODEL_ID
    assert config.actor_rollout_ref.model.revision == MODEL_REVISION
    assert memory.tokenizer_name == MODEL_ID
    assert memory.tokenizer_revision == MODEL_REVISION
    assert config.data.train_batch_size == expected["batch"]
    assert actor.ppo_mini_batch_size == expected["mini"]
    assert config.critic.ppo_mini_batch_size == expected["mini"]
    assert rollout.n == 4
    assert memory.chunk_size == 5000
    assert memory.max_chunks == 6
    assert memory.max_memorization_length == 768
    assert memory.max_final_response_length == 512
    assert actor.ppo_max_token_len_per_gpu == 12288
    assert ref.log_prob_max_token_len_per_gpu == 12288
    assert rollout.log_prob_max_token_len_per_gpu == 12288
    assert config.trainer.total_training_steps == expected["steps"]
    assert actor.optim.total_training_steps == expected["steps"]
    assert config.critic.optim.total_training_steps == expected["steps"]
    assert config.trainer.save_freq == expected["save_freq"]
    assert config.algorithm.alpha == expected["alpha"]

    resume_step = expected["resume_step"]
    if resume_step is None:
        assert config.trainer.resume_mode == "disable"
        assert config.trainer.resume_from_path is None
    else:
        assert config.trainer.resume_mode == "resume_path"
        assert config.trainer.resume_from_path.endswith(
            f"/global_step_{resume_step}"
        )


@pytest.mark.parametrize("name", TRAINING_CONTRACTS)
@pytest.mark.parametrize("profile", OFFLOAD_PROFILES)
def test_training_config_enforces_precision_lora_and_runtime_contract(
    name, profile, training_configs
):
    config = training_configs[name, profile]
    model = config.actor_rollout_ref.model
    actor = config.actor_rollout_ref.actor
    ref = config.actor_rollout_ref.ref
    rollout = config.actor_rollout_ref.rollout

    assert model.text_only is True
    assert model.model_init_seed == 1767863184
    assert model.attn_implementation == "sdpa"
    assert model.enable_gradient_checkpointing is True
    assert model.use_remove_padding is False
    assert model.use_liger is False
    assert actor.strategy == "fsdp"
    assert actor.lora.enabled is True
    assert (actor.lora.rank, actor.lora.alpha) == (32, 64)
    assert actor.lora.dropout == 0.0
    assert actor.lora.bias == "none"
    assert list(actor.checkpoint.contents) == ["model", "optimizer", "extra"]
    assert actor.fsdp_config.model_dtype == "fp32"
    assert actor.fsdp_config.mixed_precision.param_dtype == "bf16"
    assert actor.fsdp_config.mixed_precision.reduce_dtype == "fp32"
    assert actor.fsdp_config.mixed_precision.buffer_dtype == "fp32"
    assert actor.fsdp_config.param_offload is False
    assert actor.fsdp_config.optimizer_offload is False
    assert ref.fsdp_config.model_dtype == "bf16"
    assert ref.fsdp_config.mixed_precision.param_dtype == "bf16"
    assert ref.fsdp_config.param_offload is (profile == "r1")
    assert rollout.dtype == "bfloat16"
    assert rollout.name == "hf"
    assert rollout.mode == "sync"
    assert rollout.tensor_model_parallel_size == 1
    assert rollout.micro_batch_size == 1
    assert rollout.temperature == 1.0
    assert rollout.top_p == 1.0
    assert rollout.top_k == 0
    assert rollout.do_sample is True
    assert actor.ppo_micro_batch_size_per_gpu == 1
    assert ref.log_prob_micro_batch_size_per_gpu == 1
    assert rollout.log_prob_micro_batch_size_per_gpu == 1
    assert actor.use_dynamic_bsz is False
    assert actor.use_torch_compile is False
    assert actor.ppo_epochs == 1
    assert actor.optim.lr == pytest.approx(5e-6)
    assert actor.optim.lr_warmup_steps == 8
    assert actor.optim.weight_decay == pytest.approx(0.01)
    assert config.algorithm.adv_estimator == "grpo"
    assert config.algorithm.grpo_use_adv is False
    assert config.algorithm.norm_adv_by_std_in_grpo is False
    assert config.reward_model.enable is False
    assert config.reproduction.run_seed == 42
    assert config.reproduction.data_seed == 1938833696
    assert config.reproduction.model_init_seed == 1767863184
    assert config.reproduction.rollout_seed == 2924589348
    assert config.data.shuffle is False
    assert config.data.truncation == "center"
    assert config.trainer.val_before_train is False
    assert config.trainer.test_freq == -1
    assert config.trainer.n_gpus_per_node == 1


def test_length_stress_is_the_only_forced_length_non_scientific_job(
    training_configs,
):
    stress_name = "g2_length_stress_qwen35_2b_5090"
    for (name, _), config in training_configs.items():
        is_stress = name == stress_name
        assert config.reproduction.length_stress is is_stress
        assert config.actor_rollout_ref.rollout.ignore_eos is is_stress
        assert config.reproduction.formal_data is (not is_stress)
        assert config.reproduction.data_manifest_profile == (
            "fixture" if is_stress else "formal"
        )
        assert config.reproduction.val_data_manifest_profile == (
            "fixture" if is_stress else "formal"
        )
        assert config.reproduction.export_adapter_on_save is (not is_stress)
        if is_stress:
            assert "/capacity/length-stress/train/" in str(config.data.train_files)
            assert "/capacity/length-stress/validation/" in str(config.data.val_files)
            assert config.reproduction.adapter_export_dir is None


def test_all_training_output_namespaces_are_profile_isolated(training_configs):
    observed = set()
    for (name, profile), config in training_configs.items():
        values = (
            config.trainer.default_local_dir,
            config.trainer.rollout_data_dir,
            config.trainer.validation_data_dir,
        )
        for value in values:
            _assert_absolute_persistent_path(value)
            assert str(value).startswith(f"{PROFILE_ROOT}/outputs/{profile}/")
        identity = (name, profile, *values, config.trainer.experiment_name)
        assert identity not in observed
        observed.add(identity)


def test_r0_r1_change_only_offload_and_profile_identity_paths(training_configs):
    scientific_paths = {
        "algorithm.alpha",
        "data.train_batch_size",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.model.revision",
        "recurrent.memory.config.chunk_size",
        "recurrent.memory.config.max_chunks",
    }
    for name in TRAINING_CONTRACTS:
        r0 = training_configs[name, "r0"]
        r1 = training_configs[name, "r1"]
        differences = _different_leaf_paths(r0, r1)
        assert "actor_rollout_ref.ref.fsdp_config.param_offload" in differences
        assert "reproduction.offload_profile" in differences
        assert differences.isdisjoint(scientific_paths)
        for path in differences - {
            "actor_rollout_ref.ref.fsdp_config.param_offload",
            "reproduction.offload_profile",
        }:
            assert path.endswith("_path") or path.endswith("_dir") or path in {
                "trainer.experiment_name",
                "trainer.rollout_data_dir",
                "trainer.validation_data_dir",
            }


@pytest.mark.parametrize(
    ("b_name", "c_name", "has_pilot_path", "has_resume_path"),
    [
        ("b_pilot_qwen35_2b_5090", "c_pilot_qwen35_2b_5090", True, False),
        ("b20_qwen35_2b_5090", "c20_qwen35_2b_5090", False, False),
        ("b40_qwen35_2b_5090", "c40_qwen35_2b_5090", False, True),
        ("b60_qwen35_2b_5090", "c60_qwen35_2b_5090", False, True),
        ("b80_qwen35_2b_5090", "c80_qwen35_2b_5090", False, True),
    ],
)
@pytest.mark.parametrize("profile", OFFLOAD_PROFILES)
def test_bc_pairs_differ_only_in_alpha_and_registered_identity_paths(
    b_name,
    c_name,
    has_pilot_path,
    has_resume_path,
    profile,
    training_configs,
):
    expected = {
        "algorithm.alpha",
        "reproduction.adapter_export_dir",
        "reproduction.step_zero_fingerprint_path",
        "reproduction.step_zero_reference_path",
        "trainer.default_local_dir",
        "trainer.experiment_name",
        "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
    }
    if has_pilot_path:
        expected.add("reproduction.pilot_evidence_path")
    if has_resume_path:
        expected.add("trainer.resume_from_path")
    assert (
        _different_leaf_paths(
            training_configs[b_name, profile], training_configs[c_name, profile]
        )
        == expected
    )


def test_segmented_resume_paths_and_real_compatibility_allowlist(training_configs):
    predecessors = {
        "g2b_qwen35_2b_5090_resume5": (
            "g2b_qwen35_2b_5090_step1",
            1,
        ),
        "b40_qwen35_2b_5090": ("b20_qwen35_2b_5090", 20),
        "c40_qwen35_2b_5090": ("c20_qwen35_2b_5090", 20),
        "b60_qwen35_2b_5090": ("b40_qwen35_2b_5090", 40),
        "c60_qwen35_2b_5090": ("c40_qwen35_2b_5090", 40),
        "b80_qwen35_2b_5090": ("b60_qwen35_2b_5090", 60),
        "c80_qwen35_2b_5090": ("c60_qwen35_2b_5090", 60),
    }
    for profile in OFFLOAD_PROFILES:
        for resume_name, (predecessor_name, step) in predecessors.items():
            resume = training_configs[resume_name, profile]
            predecessor = training_configs[predecessor_name, profile]
            assert resume.trainer.resume_from_path == (
                f"{predecessor.trainer.default_local_dir}/global_step_{step}"
            )
            checkpoint_contract.validate_resolved_config_compatibility(
                OmegaConf.to_container(predecessor, resolve=True),
                OmegaConf.to_container(resume, resolve=True),
            )


def test_step_zero_fingerprints_are_paired_and_stable_across_segments(
    training_configs,
):
    for profile in OFFLOAD_PROFILES:
        b20 = training_configs["b20_qwen35_2b_5090", profile]
        c20 = training_configs["c20_qwen35_2b_5090", profile]
        assert b20.reproduction.step_zero_reference_path is None
        assert c20.reproduction.step_zero_reference_path == (
            b20.reproduction.step_zero_fingerprint_path
        )
        for step in (40, 60, 80):
            b = training_configs[f"b{step}_qwen35_2b_5090", profile]
            c = training_configs[f"c{step}_qwen35_2b_5090", profile]
            assert b.reproduction.step_zero_fingerprint_path == (
                b20.reproduction.step_zero_fingerprint_path
            )
            assert c.reproduction.step_zero_fingerprint_path == (
                c20.reproduction.step_zero_fingerprint_path
            )
            assert c.reproduction.step_zero_reference_path == (
                b20.reproduction.step_zero_fingerprint_path
            )


@pytest.mark.parametrize(
    ("name", "level", "step", "count"),
    [
        ("eval40_qwen35_2b_5090", "l1", 40, 32),
        ("eval80_qwen35_2b_5090", "l2", 80, 64),
    ],
)
def test_eval_configs_are_level_specific_fail_closed_matrices(
    name, level, step, count, eval_configs
):
    config = eval_configs[name]
    evaluation = config.reproduction_evaluation
    arm_names = {f"b{step}", f"c{step}"}

    assert config.trainer.total_training_steps == 0
    assert config.actor_rollout_ref.model.path == MODEL_ID
    assert config.actor_rollout_ref.model.revision == MODEL_REVISION
    assert config.actor_rollout_ref.rollout.do_sample is False
    assert config.actor_rollout_ref.rollout.n == 1
    assert config.actor_rollout_ref.rollout.temperature == 0.0
    assert evaluation.schema_version == 2
    assert evaluation.delivery_level == level
    assert evaluation.evaluator_version == "rememr1-eval-matrix-v1"
    assert evaluation.runner_module == "taskutils.memory_eval.reproduction_runner"
    assert evaluation.backend == "transformers"
    assert evaluation.base_model_id == MODEL_ID
    assert evaluation.revision == MODEL_REVISION
    assert evaluation.tokenizer_id == MODEL_ID
    assert evaluation.tokenizer_revision == MODEL_REVISION
    assert evaluation.template_revision == "rememr1-template-v1"
    assert evaluation.dtype == "bfloat16"
    assert evaluation.device == "cuda:0"
    assert evaluation.local_files_only is True
    assert set(evaluation.datasets) == {"hotpotqa", "2wikimultihopqa"}
    assert list(evaluation.document_variants) == [200, 800]
    assert evaluation.sample_count_per_stratum == count
    assert evaluation.required_cell_count == 20
    assert evaluation.primary_callback_mode == "learned"
    assert evaluation.callback_ablation.artifact == f"c{step}"
    assert list(evaluation.callback_ablation.modes) == [
        "learned",
        "none",
        "fixed_question",
    ]
    assert set(evaluation.artifacts) == {"base", *arm_names}
    expected_artifact_keys = {
        "artifact_kind",
        "artifact_path",
        "training_step",
        "training_config_source_id",
        "training_config_id",
        "training_config_path",
        "training_config_sha256",
        "adapter_metadata_sha256",
        "checkpoint_extra_state_sha256",
    }
    for artifact_name, artifact in evaluation.artifacts.items():
        assert set(artifact) == expected_artifact_keys
        assert artifact.training_step == (0 if artifact_name == "base" else step)
        assert artifact.artifact_kind == (
            "base" if artifact_name == "base" else "adapter"
        )
        expected_source = (
            None
            if artifact_name == "base"
            else f"{artifact_name}_qwen35_2b_5090"
        )
        assert artifact.training_config_source_id == expected_source
        for field in (
            "artifact_path",
            "training_config_id",
            "training_config_path",
            "training_config_sha256",
            "adapter_metadata_sha256",
            "checkpoint_extra_state_sha256",
        ):
            assert artifact[field] is None
    assert evaluation.decode.greedy is True
    assert evaluation.decode.do_sample is False
    assert evaluation.decode.n == 1
    assert evaluation.decode.temperature == 0.0
    assert evaluation.decode.chunk_size == 5000
    assert evaluation.decode.memory_max_tokens == 768
    assert evaluation.decode.final_max_tokens == 512
    _assert_absolute_persistent_path(evaluation.output_root)
    for dataset in evaluation.datasets.values():
        _assert_absolute_persistent_path(dataset.bundle_dir)
        assert re.fullmatch(r"[0-9a-f]{64}", dataset.manifest_sha256)


def test_new_sources_do_not_add_a_compress_or_tool_action():
    for name in (*TRAINING_CONTRACTS, *EVAL_CONFIG_NAMES):
        source = (REPRODUCTION_CONFIG_ROOT / f"{name}.yaml").read_text(
            encoding="utf-8"
        )
        assert "compress_context" not in source
        assert "tool_actions" not in source


def test_all_active_sources_compose_with_hydra_when_installed():
    pytest.importorskip("hydra", reason="hydra-core is not installed in this CPU test env")
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        for name in GATE_CONFIG_NAMES:
            config = compose(config_name=f"reproduction/{name}")
            OmegaConf.resolve(config)
        for name in TRAINING_CONTRACTS:
            for profile in OFFLOAD_PROFILES:
                config = compose(
                    config_name=f"reproduction/{name}",
                    overrides=[f"reproduction/offload@_global_={profile}"],
                )
                OmegaConf.resolve(config)
                assert config.reproduction.offload_profile == profile
        for name in EVAL_CONFIG_NAMES:
            config = compose(config_name=f"reproduction/{name}")
            OmegaConf.resolve(config)
