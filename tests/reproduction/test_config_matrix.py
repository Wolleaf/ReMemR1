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

PINNED_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}
SEEDS = {
    "run": 42,
    "data": 1938833696,
    "model_init": 1767863184,
    "rollout": 2924589348,
}

TRAINING_CONTRACTS = {
    "g0_qwen35_08b": {
        "model": "Qwen/Qwen3.5-0.8B",
        "batch": 1,
        "mini": 1,
        "group": 4,
        "chunk_size": 1024,
        "chunks": 2,
        "memory_tokens": 256,
        "final_tokens": 256,
        "token_budget": 4096,
        "steps": 20,
        "save_freq": 20,
        "resume_step": None,
        "workers": 1,
        "alpha": 0.8,
    },
    "g1_qwen35_2b_step1": {
        "model": "Qwen/Qwen3.5-2B",
        "batch": 1,
        "mini": 1,
        "group": 4,
        "chunk_size": 1024,
        "chunks": 2,
        "memory_tokens": 256,
        "final_tokens": 256,
        "token_budget": 4096,
        "steps": 1,
        "save_freq": 1,
        "resume_step": None,
        "workers": 1,
        "alpha": 0.8,
    },
    "g1_qwen35_2b_resume2": {
        "model": "Qwen/Qwen3.5-2B",
        "batch": 1,
        "mini": 1,
        "group": 4,
        "chunk_size": 1024,
        "chunks": 2,
        "memory_tokens": 256,
        "final_tokens": 256,
        "token_budget": 4096,
        "steps": 2,
        "save_freq": 1,
        "resume_step": 1,
        "workers": 1,
        "alpha": 0.8,
    },
    "g2a_qwen35_4b": {
        "model": "Qwen/Qwen3.5-4B",
        "batch": 1,
        "mini": 1,
        "group": 4,
        "chunk_size": 5000,
        "chunks": 6,
        "memory_tokens": 768,
        "final_tokens": 512,
        "token_budget": 12288,
        "steps": 1,
        "save_freq": 1,
        "resume_step": None,
        "workers": 2,
        "alpha": 0.8,
    },
    "g2b_qwen35_4b_step1": {
        "model": "Qwen/Qwen3.5-4B",
        "batch": 4,
        "mini": 4,
        "group": 8,
        "chunk_size": 5000,
        "chunks": 6,
        "memory_tokens": 768,
        "final_tokens": 512,
        "token_budget": 12288,
        "steps": 1,
        "save_freq": 1,
        "resume_step": None,
        "workers": 2,
        "alpha": 0.8,
    },
    "g2b_qwen35_4b_resume2": {
        "model": "Qwen/Qwen3.5-4B",
        "batch": 4,
        "mini": 4,
        "group": 8,
        "chunk_size": 5000,
        "chunks": 6,
        "memory_tokens": 768,
        "final_tokens": 512,
        "token_budget": 12288,
        "steps": 2,
        "save_freq": 1,
        "resume_step": 1,
        "workers": 2,
        "alpha": 0.8,
    },
    "b_pilot_qwen35_4b": {
        "model": "Qwen/Qwen3.5-4B",
        "batch": 4,
        "mini": 4,
        "group": 8,
        "chunk_size": 5000,
        "chunks": 6,
        "memory_tokens": 768,
        "final_tokens": 512,
        "token_budget": 12288,
        "steps": 1,
        "save_freq": 1,
        "resume_step": None,
        "workers": 2,
        "alpha": 1.0,
    },
    "c_pilot_qwen35_4b": {
        "model": "Qwen/Qwen3.5-4B",
        "batch": 4,
        "mini": 4,
        "group": 8,
        "chunk_size": 5000,
        "chunks": 6,
        "memory_tokens": 768,
        "final_tokens": 512,
        "token_budget": 12288,
        "steps": 1,
        "save_freq": 1,
        "resume_step": None,
        "workers": 2,
        "alpha": 0.8,
    },
    "b40_qwen35_4b": {
        "model": "Qwen/Qwen3.5-4B",
        "batch": 4,
        "mini": 4,
        "group": 8,
        "chunk_size": 5000,
        "chunks": 6,
        "memory_tokens": 768,
        "final_tokens": 512,
        "token_budget": 12288,
        "steps": 40,
        "save_freq": 20,
        "resume_step": None,
        "workers": 2,
        "alpha": 1.0,
    },
    "c40_qwen35_4b": {
        "model": "Qwen/Qwen3.5-4B",
        "batch": 4,
        "mini": 4,
        "group": 8,
        "chunk_size": 5000,
        "chunks": 6,
        "memory_tokens": 768,
        "final_tokens": 512,
        "token_budget": 12288,
        "steps": 40,
        "save_freq": 20,
        "resume_step": None,
        "workers": 2,
        "alpha": 0.8,
    },
    "b80_qwen35_4b": {
        "model": "Qwen/Qwen3.5-4B",
        "batch": 4,
        "mini": 4,
        "group": 8,
        "chunk_size": 5000,
        "chunks": 6,
        "memory_tokens": 768,
        "final_tokens": 512,
        "token_budget": 12288,
        "steps": 80,
        "save_freq": 20,
        "resume_step": 40,
        "workers": 2,
        "alpha": 1.0,
    },
    "c80_qwen35_4b": {
        "model": "Qwen/Qwen3.5-4B",
        "batch": 4,
        "mini": 4,
        "group": 8,
        "chunk_size": 5000,
        "chunks": 6,
        "memory_tokens": 768,
        "final_tokens": 512,
        "token_budget": 12288,
        "steps": 80,
        "save_freq": 20,
        "resume_step": 40,
        "workers": 2,
        "alpha": 0.8,
    },
}
ALL_CONFIG_NAMES = (*TRAINING_CONTRACTS, "eval_qwen35_4b")


def _load_child(name: str, stack: tuple[str, ...] = ()) -> DictConfig:
    assert name not in stack, f"cyclic reproduction config defaults: {stack + (name,)}"
    path = REPRODUCTION_CONFIG_ROOT / f"{name}.yaml"
    child = OmegaConf.load(path)
    defaults = list(child.defaults)
    assert defaults[-1] == "_self_"
    del child["defaults"]
    parents = []
    for default in defaults[:-1]:
        if default == "/ppo_trainer":
            continue
        prefix = "/reproduction/"
        assert isinstance(default, str) and default.startswith(prefix)
        parents.append(_load_child(default.removeprefix(prefix), stack + (name,)))
    return OmegaConf.merge(*parents, child)


def _compose_with_omegaconf(name: str) -> DictConfig:
    config = OmegaConf.merge(OmegaConf.load(BASE_CONFIG), _load_child(name))
    OmegaConf.resolve(config)
    return config


@pytest.fixture(scope="module")
def configs() -> dict[str, DictConfig]:
    return {name: _compose_with_omegaconf(name) for name in ALL_CONFIG_NAMES}


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
    return {
        key for key in left_values if left_values[key] != right_values[key]
    }


def test_matrix_has_exactly_the_thirteen_planned_configs():
    actual = {path.stem for path in REPRODUCTION_CONFIG_ROOT.glob("*.yaml")}
    assert actual == set(ALL_CONFIG_NAMES)


@pytest.mark.parametrize("name", TRAINING_CONTRACTS)
def test_training_config_resolves_to_its_gate_or_formal_shape(name, configs):
    config = configs[name]
    expected = TRAINING_CONTRACTS[name]
    model = config.actor_rollout_ref.model
    actor = config.actor_rollout_ref.actor
    ref = config.actor_rollout_ref.ref
    rollout = config.actor_rollout_ref.rollout
    memory = config.recurrent.memory.config

    assert model.path == expected["model"]
    assert model.revision == PINNED_REVISIONS[model.path]
    assert model.text_only is True
    assert model.model_init_seed == SEEDS["model_init"]
    assert model.attn_implementation == "sdpa"
    assert model.enable_gradient_checkpointing is True
    assert model.use_remove_padding is False
    assert model.use_liger is False

    assert config.data.train_batch_size == expected["batch"]
    assert actor.ppo_mini_batch_size == expected["mini"]
    assert rollout.n == expected["group"]
    assert memory.chunk_size == expected["chunk_size"]
    assert memory.max_chunks == expected["chunks"]
    assert memory.max_memorization_length == expected["memory_tokens"]
    assert memory.max_final_response_length == expected["final_tokens"]
    assert actor.ppo_max_token_len_per_gpu == expected["token_budget"]
    assert ref.log_prob_max_token_len_per_gpu == expected["token_budget"]
    assert rollout.log_prob_max_token_len_per_gpu == expected["token_budget"]
    assert config.trainer.total_training_steps == expected["steps"]
    assert actor.optim.total_training_steps == expected["steps"]
    assert config.critic.optim.total_training_steps == expected["steps"]
    assert config.trainer.save_freq == expected["save_freq"]
    assert config.data.dataloader_num_workers == expected["workers"]
    assert config.algorithm.alpha == expected["alpha"]

    resume_step = expected["resume_step"]
    if resume_step is None:
        assert config.trainer.resume_mode == "disable"
        assert config.trainer.resume_from_path is None
    else:
        assert config.trainer.resume_mode == "resume_path"
        _assert_absolute_persistent_path(config.trainer.resume_from_path)
        assert config.trainer.resume_from_path.endswith(
            f"/global_step_{resume_step}"
        )


@pytest.mark.parametrize("name", TRAINING_CONTRACTS)
def test_training_config_enforces_common_qwen35_lora_grpo_contract(name, configs):
    config = configs[name]
    model = config.actor_rollout_ref.model
    actor = config.actor_rollout_ref.actor
    ref = config.actor_rollout_ref.ref
    rollout = config.actor_rollout_ref.rollout
    actor_fsdp = actor.fsdp_config
    ref_fsdp = ref.fsdp_config

    assert config.reproduction.run_seed == SEEDS["run"]
    assert config.reproduction.data_seed == SEEDS["data"]
    assert config.reproduction.model_init_seed == SEEDS["model_init"]
    assert config.reproduction.rollout_seed == SEEDS["rollout"]
    assert config.data.seed == SEEDS["data"]
    assert rollout.seed == SEEDS["rollout"]
    expected_profile = (
        "fixture" if name.startswith(("g0_", "g1_")) else "formal"
    )
    assert config.reproduction.formal_data is (expected_profile == "formal")
    assert config.reproduction.data_manifest_sha256 == ZERO_SHA256
    assert config.reproduction.val_data_manifest_sha256 == ONE_SHA256
    assert (
        config.reproduction.data_manifest_sha256
        != config.reproduction.val_data_manifest_sha256
    )
    assert config.reproduction.data_manifest_mode == "train"
    assert config.reproduction.val_data_manifest_mode == "train"
    assert config.reproduction.data_manifest_profile == expected_profile
    assert config.reproduction.val_data_manifest_profile == expected_profile
    assert config.reproduction.export_adapter_on_save is True
    assert config.reproduction.template_revision == "rememr1-template-v1"

    assert actor.strategy == "fsdp"
    assert actor.lora.enabled is True
    assert actor.lora.rank == 32
    assert actor.lora.alpha == 64
    assert actor.lora.dropout == 0.0
    assert actor.lora.bias == "none"
    assert actor.lora.text_model_prefix == "model"
    assert list(actor.checkpoint.contents) == ["model", "optimizer", "extra"]
    assert actor_fsdp.root_only is True
    assert actor_fsdp.use_orig_params is True
    assert actor_fsdp.model_dtype == "fp32"
    assert actor_fsdp.mixed_precision.param_dtype == "bf16"
    assert actor_fsdp.mixed_precision.reduce_dtype == "fp32"
    assert actor_fsdp.mixed_precision.buffer_dtype == "fp32"
    assert actor_fsdp.param_offload is False
    assert actor_fsdp.optimizer_offload is False
    assert ref_fsdp.model_dtype == "bf16"
    assert ref_fsdp.param_offload is False

    assert actor.optim.lr == pytest.approx(5e-6)
    assert actor.optim.weight_decay == pytest.approx(0.01)
    assert actor.optim.lr_warmup_steps == 8
    assert actor.optim.warmup_style == "constant"
    assert actor.grad_clip == 1.0
    assert actor.ppo_epochs == 1
    assert actor.use_dynamic_bsz is False
    assert actor.use_torch_compile is False
    assert actor.use_kl_loss is True
    assert actor.kl_loss_coef == pytest.approx(0.001)
    assert actor.kl_loss_type == "low_var_kl"

    assert actor.ppo_micro_batch_size is None
    assert actor.ppo_micro_batch_size_per_gpu == 1
    assert ref.log_prob_micro_batch_size is None
    assert ref.log_prob_micro_batch_size_per_gpu == 1
    assert rollout.log_prob_micro_batch_size is None
    assert rollout.log_prob_micro_batch_size_per_gpu == 1
    assert config.critic.ppo_micro_batch_size is None
    assert config.critic.ppo_micro_batch_size_per_gpu == 1
    assert config.reward_model.micro_batch_size is None
    assert config.reward_model.micro_batch_size_per_gpu == 1

    assert rollout.name == "hf"
    assert rollout.mode == "sync"
    assert rollout.tensor_model_parallel_size == 1
    assert rollout.micro_batch_size == 1
    assert rollout.temperature == 1.0
    assert rollout.top_p == 1.0
    assert rollout.top_k == 0
    assert rollout.do_sample is True
    assert rollout.val_kwargs.do_sample is False
    assert rollout.val_kwargs.temperature == 0.0

    assert config.algorithm.adv_estimator == "grpo"
    assert config.algorithm.grpo_use_adv is False
    assert config.algorithm.norm_adv_by_std_in_grpo is False
    assert config.algorithm.action_reweight is False
    assert config.algorithm.use_kl_in_reward is False
    assert config.reward_model.enable is False
    assert config.reward_model.reward_metric == "em"

    assert config.recurrent.enable == "memory"
    assert config.recurrent.memory.path == "recurrent/impls/memory_revisit.py"
    assert config.recurrent.memory.config.max_prompt_length == 1024
    assert config.recurrent.memory.config.callback_mode == "learned"
    assert config.recurrent.memory.config.require_manifest is True
    assert config.recurrent.memory.config.tokenizer_name == model.path
    assert config.recurrent.memory.config.tokenizer_revision == model.revision
    assert config.data.context_key == "context"
    assert str(config.data.train_files).endswith("/train/train.parquet")
    assert str(config.data.val_files).endswith("/validation/train.parquet")
    assert config.data.max_response_length == 1024
    assert config.data.shuffle is False
    assert config.data.truncation == "center"
    assert config.data.filter_overlong_prompts is True
    assert config.data.filter_overlong_prompts_workers == 1

    assert config.trainer.total_epochs == 1
    assert config.trainer.nnodes == 1
    assert config.trainer.n_gpus_per_node == 1
    assert list(config.trainer.logger) == ["console"]
    assert config.trainer.val_before_train is False
    assert config.trainer.test_freq == -1
    assert config.trainer.save_best_val is False
    assert config.trainer.default_hdfs_dir is None
    assert config.trainer.del_local_ckpt_after_load is False
    assert config.trainer.max_actor_ckpt_to_keep is None
    assert config.trainer.max_critic_ckpt_to_keep is None


@pytest.mark.parametrize("name", TRAINING_CONTRACTS)
def test_training_config_has_no_legacy_backend_scale_or_dataset_defaults(name, configs):
    config = configs[name]
    assert "gsm8k" not in str(config.data.train_files).casefold()
    assert "gsm8k" not in str(config.data.val_files).casefold()
    assert config.actor_rollout_ref.rollout.name not in {"vllm", "sglang"}
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size != 2
    assert config.trainer.n_gpus_per_node != 8
    assert config.trainer.total_epochs != 30
    assert config.algorithm.adv_estimator != "gae"
    assert config.trainer.resume_mode != "auto"
    assert config.data.train_batch_size <= 4
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size <= 4


def test_every_training_output_namespace_is_absolute_and_isolated(configs):
    local_dirs = []
    adapter_dirs = []
    rollout_dirs = []
    validation_dirs = []
    experiment_names = []
    for name in TRAINING_CONTRACTS:
        config = configs[name]
        for value in (
            config.trainer.default_local_dir,
            config.trainer.rollout_data_dir,
            config.trainer.validation_data_dir,
            config.reproduction.adapter_export_dir,
        ):
            _assert_absolute_persistent_path(value)
        local_dirs.append(config.trainer.default_local_dir)
        adapter_dirs.append(config.reproduction.adapter_export_dir)
        rollout_dirs.append(config.trainer.rollout_data_dir)
        validation_dirs.append(config.trainer.validation_data_dir)
        experiment_names.append(config.trainer.experiment_name)

    assert len(local_dirs) == len(set(local_dirs))
    assert len(adapter_dirs) == len(set(adapter_dirs))
    assert len(rollout_dirs) == len(set(rollout_dirs))
    assert len(validation_dirs) == len(set(validation_dirs))
    assert len(experiment_names) == len(set(experiment_names))


def test_resume_paths_point_to_the_explicit_predecessor_checkpoint(configs):
    predecessors = {
        "g1_qwen35_2b_resume2": ("g1_qwen35_2b_step1", 1),
        "g2b_qwen35_4b_resume2": ("g2b_qwen35_4b_step1", 1),
        "b80_qwen35_4b": ("b40_qwen35_4b", 40),
        "c80_qwen35_4b": ("c40_qwen35_4b", 40),
    }
    for resume_name, (predecessor_name, step) in predecessors.items():
        assert configs[resume_name].trainer.resume_from_path == (
            f"{configs[predecessor_name].trainer.default_local_dir}"
            f"/global_step_{step}"
        )


def test_segmented_resume_configs_pass_the_real_resolved_config_allowlist(configs):
    predecessors = {
        "g1_qwen35_2b_resume2": "g1_qwen35_2b_step1",
        "g2b_qwen35_4b_resume2": "g2b_qwen35_4b_step1",
        "b80_qwen35_4b": "b40_qwen35_4b",
        "c80_qwen35_4b": "c40_qwen35_4b",
    }
    for resume_name, predecessor_name in predecessors.items():
        checkpoint_contract.validate_resolved_config_compatibility(
            OmegaConf.to_container(configs[predecessor_name], resolve=True),
            OmegaConf.to_container(configs[resume_name], resolve=True),
        )


def test_bc_step_zero_evidence_is_absolute_paired_and_stable_across_resume(configs):
    b_path = configs["b40_qwen35_4b"].reproduction.step_zero_fingerprint_path
    c_path = configs["c40_qwen35_4b"].reproduction.step_zero_fingerprint_path
    _assert_absolute_persistent_path(b_path)
    _assert_absolute_persistent_path(c_path)
    assert b_path != c_path
    assert configs["b40_qwen35_4b"].reproduction.step_zero_reference_path is None
    assert configs["c40_qwen35_4b"].reproduction.step_zero_reference_path == b_path
    assert configs["b80_qwen35_4b"].reproduction.step_zero_fingerprint_path == b_path
    assert configs["b80_qwen35_4b"].reproduction.step_zero_reference_path is None
    assert configs["c80_qwen35_4b"].reproduction.step_zero_fingerprint_path == c_path
    assert configs["c80_qwen35_4b"].reproduction.step_zero_reference_path == b_path

    pilot_b_path = configs[
        "b_pilot_qwen35_4b"
    ].reproduction.step_zero_fingerprint_path
    pilot_c_path = configs[
        "c_pilot_qwen35_4b"
    ].reproduction.step_zero_fingerprint_path
    _assert_absolute_persistent_path(pilot_b_path)
    _assert_absolute_persistent_path(pilot_c_path)
    assert pilot_b_path not in {b_path, c_path, pilot_c_path}
    assert configs[
        "b_pilot_qwen35_4b"
    ].reproduction.step_zero_reference_path is None
    assert configs[
        "c_pilot_qwen35_4b"
    ].reproduction.step_zero_reference_path == pilot_b_path

    for name in set(TRAINING_CONTRACTS) - {
        "b_pilot_qwen35_4b",
        "c_pilot_qwen35_4b",
        "b40_qwen35_4b",
        "c40_qwen35_4b",
        "b80_qwen35_4b",
        "c80_qwen35_4b",
    }:
        assert configs[name].reproduction.step_zero_fingerprint_path is None
        assert configs[name].reproduction.step_zero_reference_path is None


@pytest.mark.parametrize(
    ("b_name", "c_name", "extra_paths"),
    [
        ("b_pilot_qwen35_4b", "c_pilot_qwen35_4b", set()),
        ("b40_qwen35_4b", "c40_qwen35_4b", set()),
        (
            "b80_qwen35_4b",
            "c80_qwen35_4b",
            {"trainer.resume_from_path"},
        ),
    ],
)
def test_bc_pairs_differ_only_in_alpha_and_identity_paths(
    b_name, c_name, extra_paths, configs
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
    } | extra_paths
    assert _different_leaf_paths(configs[b_name], configs[c_name]) == expected


def test_b80_c80_keep_warmup_and_restore_scheduler_bearing_checkpoint(configs):
    for name in ("b80_qwen35_4b", "c80_qwen35_4b"):
        config = configs[name]
        assert config.actor_rollout_ref.actor.optim.lr_warmup_steps == 8
        assert config.actor_rollout_ref.actor.optim.total_training_steps == 80
        assert config.critic.optim.total_training_steps == 80
        assert config.trainer.resume_mode == "resume_path"
        assert set(config.actor_rollout_ref.actor.checkpoint.contents) == {
            "model",
            "optimizer",
            "extra",
        }


def test_external_eval_config_fixes_runner_artifacts_and_greedy_decode(configs):
    config = configs["eval_qwen35_4b"]
    evaluation = config.reproduction_evaluation

    assert config.trainer.total_training_steps == 0
    assert config.trainer.total_epochs == 1
    assert config.trainer.n_gpus_per_node == 1
    assert config.actor_rollout_ref.rollout.name == "hf"
    assert config.data.dataloader_num_workers == 2
    assert evaluation.runner_module == "taskutils.memory_eval.reproduction_runner"
    assert evaluation.backend == "transformers"
    assert evaluation.base_model_id == "Qwen/Qwen3.5-4B"
    assert evaluation.revision == PINNED_REVISIONS["Qwen/Qwen3.5-4B"]
    assert evaluation.tokenizer_id == evaluation.base_model_id
    assert evaluation.tokenizer_revision == evaluation.revision
    assert evaluation.template_revision == "rememr1-template-v1"
    assert evaluation.dtype == "bfloat16"
    assert evaluation.device == "cuda:0"
    assert evaluation.local_files_only is True
    assert evaluation.seed == 42

    assert set(evaluation.datasets) == {"hotpotqa", "2wikimultihopqa"}
    eval_manifest_hashes = set()
    for dataset in evaluation.datasets.values():
        _assert_absolute_persistent_path(dataset.bundle_dir)
        assert re.fullmatch(r"[0-9a-f]{64}", dataset.manifest_sha256)
        eval_manifest_hashes.add(dataset.manifest_sha256)
    assert len(eval_manifest_hashes) == len(evaluation.datasets)
    assert list(evaluation.document_variants) == [200, 800]
    assert evaluation.sample_counts.step_40 == 32
    assert evaluation.sample_counts.step_80 == 64
    assert evaluation.primary_callback_mode == "learned"
    assert evaluation.callback_ablation.artifact == "c80"
    assert list(evaluation.callback_ablation.modes) == [
        "learned",
        "none",
        "fixed_question",
    ]

    assert set(evaluation.artifacts) == {"base", "b40", "c40", "b80", "c80"}
    assert evaluation.artifacts.base.artifact_kind == "base"
    assert evaluation.artifacts.base.artifact_path is None
    for artifact_name in ("b40", "c40", "b80", "c80"):
        artifact = evaluation.artifacts[artifact_name]
        assert artifact.artifact_kind == "adapter"
        _assert_absolute_persistent_path(artifact.artifact_path)
        assert artifact.artifact_path.endswith(
            f"/global_step_{artifact.training_step}/adapter"
        )
        training_config = configs[f"{artifact_name}_qwen35_4b"]
        assert artifact.artifact_path == (
            f"{training_config.reproduction.adapter_export_dir}"
            f"/global_step_{artifact.training_step}/adapter"
        )

    assert evaluation.decode.greedy is True
    assert evaluation.decode.do_sample is False
    assert evaluation.decode.temperature == 0.0
    assert evaluation.decode.top_p == 1.0
    assert evaluation.decode.top_k == 0
    assert evaluation.decode.chunk_size == 5000
    assert evaluation.decode.memory_max_tokens == 768
    assert evaluation.decode.final_max_tokens == 512
    assert evaluation.timeouts_seconds.model_load > 0
    assert evaluation.timeouts_seconds.sample > 0
    assert evaluation.timeouts_seconds.task > 0
    _assert_absolute_persistent_path(evaluation.output_root)


def test_all_configs_compose_and_resolve_with_hydra_when_installed():
    pytest.importorskip("hydra", reason="hydra-core is not installed in this CPU test env")
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        for name in ALL_CONFIG_NAMES:
            config = compose(config_name=f"reproduction/{name}")
            OmegaConf.resolve(config)
            assert config.actor_rollout_ref.model.revision
