# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
import json
import importlib.metadata
import subprocess
import threading
import time
from typing import Union

import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
import datetime
def _now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_REPRODUCTION_TELEMETRY_PHASES = (
    "rollout",
    "reward",
    "actor_log_prob",
    "reference_log_prob",
    "update",
    "save",
)


def _query_single_gpu_nvml_memory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi telemetry failed: {result.stderr.strip()}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"telemetry requires exactly one physical GPU, got {lines}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 3 or not fields[0].startswith("GPU-"):
        raise RuntimeError(f"malformed nvidia-smi telemetry: {lines[0]!r}")
    try:
        used_mib, total_mib = (int(value) for value in fields[1:])
    except ValueError as exc:
        raise RuntimeError(f"non-integer nvidia-smi telemetry: {lines[0]!r}") from exc
    return {
        "gpu_uuid": fields[0],
        "used_bytes": used_mib * 1024**2,
        "total_bytes": total_mib * 1024**2,
    }


def _create_reproduction_host_memory_probe():
    # Keep cloud-only probing lazy so ordinary verl deployments still import this worker.
    from scripts.cloud.host_resource_probe import create_memory_telemetry_probe

    host = psutil.virtual_memory()
    return create_memory_telemetry_probe(host_total_memory_bytes=host.total)


def _sample_reproduction_host_memory(probe):
    host = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return probe.sample(
        fallback_used_memory_bytes=host.used,
        fallback_swap_used_bytes=swap.used,
    )


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh("cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group()

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.model_build_metadata = {}

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            #######
            # ADD: actor must aware how many steps are there in a batch, since we may have variant of batch sizes
            #######
            update_steps_per_batch = self.config.actor.train_batch_size // self.config.actor.ppo_mini_batch_size
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.actor.train_batch_size = update_steps_per_batch * self.config.actor.ppo_mini_batch_size
            assert self.config.actor.ppo_mini_batch_size > 0, f'ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization'
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        model_revision=None,
        attention_implementation="sdpa",
        lora_config=None,
        model_init_seed=None,
        role="actor",
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForCausalLM

        from verl.models.lora_contract import (
            LORA_ALPHA,
            LORA_BIAS,
            LORA_DROPOUT,
            LORA_R,
            assert_injected_lora_targets,
            assert_only_lora_parameters_trainable,
            inject_lora_adapter,
            resolve_lora_target_manifest,
            trainable_optimizer_parameters,
        )
        from verl.models.qwen35 import (
            QWEN35_CONDITIONAL_MODEL_TYPE,
            QWEN35_TEXT_MODEL_TYPE,
            inspect_qwen35_config,
            load_qwen35_text_model,
        )
        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.reproducibility import seed_process
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = copy_to_local(model_path)

        if model_init_seed is not None:
            seed_process(int(model_init_seed))

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if role == "actor" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        lora_enabled = role == "actor" and bool(
            lora_config is not None and lora_config.get("enabled", False)
        )
        if lora_enabled and torch_dtype != torch.float32:
            raise ValueError(
                "LoRA actor original/master parameters must remain FP32; "
                f"got {torch_dtype}"
            )

        config_load_kwargs = {"trust_remote_code": trust_remote_code}
        if model_revision is not None:
            config_load_kwargs["revision"] = model_revision
        source_model_config = AutoConfig.from_pretrained(local_path, **config_load_kwargs)
        is_qwen35 = getattr(source_model_config, "model_type", None) in {
            QWEN35_CONDITIONAL_MODEL_TYPE,
            QWEN35_TEXT_MODEL_TYPE,
        }
        if is_qwen35:
            inspect_qwen35_config(source_model_config)
        if is_qwen35 and role == "ref" and torch_dtype != torch.bfloat16:
            raise ValueError(f"Qwen3.5 reference model must load in BF16, got {torch_dtype}")

        if is_qwen35 and trust_remote_code:
            raise ValueError("Qwen3.5 reproduction requires trust_remote_code=false")
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        if model_revision is not None:
            tokenizer_kwargs["revision"] = model_revision
        self.tokenizer = hf_tokenizer(local_path, **tokenizer_kwargs)
        # The reproduction is text-only; loading a unified vision processor can
        # accidentally route text data through multimodal dataset code.
        self.processor = None if is_qwen35 else hf_processor(local_path, **tokenizer_kwargs)

        self.generation_config = get_generation_config(
            local_path,
            trust_remote_code=trust_remote_code,
            revision=model_revision,
        )

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(source_model_config, override_config_kwargs=override_config_kwargs)
        text_config = getattr(source_model_config, "text_config", None)
        if text_config is not None:
            update_model_config(text_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {source_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not source_model_config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        build_metadata = {
            "role": role,
            "base_model": str(model_path),
            "revision": model_revision,
            "model_init_seed": model_init_seed,
            "attention_implementation": attention_implementation,
            "model_dtype": str(torch_dtype),
            "qwen35_text_only": is_qwen35,
            "torch_version": str(torch.__version__),
        }

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if is_qwen35:
                qwen35_result = load_qwen35_text_model(
                    local_path,
                    revision=model_revision,
                    attn_implementation=attention_implementation,
                    dtype=torch_dtype,
                    config=source_model_config,
                )
                actor_module = qwen35_result.model
                qwen35_mapping = qwen35_result.metadata.to_dict()
                from verl.utils.checkpoint.reproduction import canonical_json_sha256

                build_metadata["qwen35_mapping"] = qwen35_mapping
                build_metadata["qwen35_mapping_sha256"] = canonical_json_sha256(
                    qwen35_mapping
                )
            else:
                model_load_kwargs = {
                    "pretrained_model_name_or_path": local_path,
                    "dtype": torch_dtype,
                    "config": source_model_config,
                    "attn_implementation": attention_implementation,
                    "trust_remote_code": trust_remote_code,
                }
                if model_revision is not None:
                    model_load_kwargs["revision"] = model_revision
                actor_module = AutoModelForCausalLM.from_pretrained(**model_load_kwargs)

            actor_model_config = actor_module.config

            if use_remove_padding or self.ulysses_sequence_parallel_size > 1:
                if is_qwen35:
                    raise ValueError(
                        "Qwen3.5 reproduction does not support the Qwen2 remove-padding/"
                        "Ulysses monkey patch; use_remove_padding=false and SP=1 are required"
                    )
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=actor_module, ulysses_sp_size=self.ulysses_sequence_parallel_size)

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            if lora_enabled:
                configured_lora = {
                    "rank": int(lora_config.get("rank", LORA_R)),
                    "alpha": int(lora_config.get("alpha", LORA_ALPHA)),
                    "dropout": float(lora_config.get("dropout", LORA_DROPOUT)),
                    "bias": str(lora_config.get("bias", LORA_BIAS)),
                }
                expected_lora = {
                    "rank": LORA_R,
                    "alpha": LORA_ALPHA,
                    "dropout": LORA_DROPOUT,
                    "bias": LORA_BIAS,
                }
                if configured_lora != expected_lora:
                    raise ValueError(
                        f"LoRA config must match the reproduction contract: {expected_lora}, "
                        f"got {configured_lora}"
                    )
                text_model_prefix = lora_config.get("text_model_prefix", "model")
                if is_qwen35 and text_model_prefix != "model":
                    raise ValueError(
                        "Qwen3.5 text all-linear LoRA requires text_model_prefix='model'"
                    )
                target_manifest = resolve_lora_target_manifest(
                    actor_module,
                    text_model_prefix=text_model_prefix,
                )
                actor_module = inject_lora_adapter(actor_module, target_manifest)
                assert_injected_lora_targets(actor_module, target_manifest)
                trainable_manifest = assert_only_lora_parameters_trainable(actor_module)
                build_metadata["lora_target_manifest"] = target_manifest.to_dict()
                build_metadata["trainable_parameters"] = trainable_manifest.to_dict()
                build_metadata["peft_version"] = importlib.metadata.version("peft")
            else:
                build_metadata["lora_target_manifest"] = None

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if lora_enabled and param_dtype != torch.bfloat16:
            raise ValueError(
                "LoRA actor mixed-precision forward/backward must use BF16 param_dtype; "
                f"got {param_dtype}"
            )

        root_only = bool(fsdp_config.get("root_only", False))
        use_orig_params = bool(fsdp_config.get("use_orig_params", False))
        if lora_enabled and not root_only:
            raise ValueError("LoRA PPO requires root-only FSDP for the HF rollout contract")
        if lora_enabled and not use_orig_params:
            raise ValueError("LoRA PPO requires FSDP use_orig_params=true")

        auto_wrap_policy = None if root_only else get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        param_offload = bool(fsdp_config.get("param_offload", False))
        cpu_offload = CPUOffload(offload_params=True) if param_offload else None
        build_metadata["fsdp"] = {
            "root_only": root_only,
            "use_orig_params": use_orig_params,
            "param_offload": param_offload,
            "optimizer_offload": bool(fsdp_config.get("optimizer_offload", False)),
            "param_dtype": str(param_dtype),
            "reduce_dtype": str(reduce_dtype),
            "buffer_dtype": str(buffer_dtype),
        }
        self.model_build_metadata[role] = build_metadata

        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=use_orig_params,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False,
        )

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            optimizer_parameters = trainable_optimizer_parameters(actor_module_fsdp)
            build_metadata["optimizer_trainable_numel"] = sum(
                parameter.numel() for parameter in optimizer_parameters
            )
            actor_optimizer = optim.AdamW(
                optimizer_parameters,
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        if self.rank == 0:
            print("Model build metadata:\n" + json.dumps(build_metadata, indent=2, sort_keys=True))

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            from verl.models.qwen35 import QWEN35_TEXT_MODEL_TYPE

            if (
                getattr(self.actor_model_config, "model_type", None) == QWEN35_TEXT_MODEL_TYPE
                and int(self.config.rollout.get("micro_batch_size", 1)) != 1
            ):
                raise ValueError("Qwen3.5 recurrent HF rollout requires micro_batch_size=1")
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMAsyncRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path)
            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                )
            elif vllm_mode == "spmd":
                vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                )
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang":
            from verl.workers.rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to veRL's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                model_revision=self.config.model.get("revision"),
                attention_implementation=self.config.model.get("attn_implementation", "sdpa"),
                lora_config=self.config.actor.get("lora") if self._is_actor else None,
                model_init_seed=self.config.model.get("model_init_seed"),
                role="actor",
            )

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                model_revision=self.config.model.get("revision"),
                attention_implementation=self.config.model.get("attn_implementation", "sdpa"),
                lora_config=None,
                model_init_seed=self.config.model.get("model_init_seed"),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            if estimated_flops > 0 and promised_flops not in (0, float("inf")):
                metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        if torch.distributed.get_rank() == 0:
            print(f"{_now()} fsdp_workers generate_sequences")
        prompts = prompts.to(torch.cuda.current_device())

        assert self._is_rollout

        generation_eos = getattr(self.generation_config, "eos_token_id", None)
        generation_pad = getattr(self.generation_config, "pad_token_id", None)
        meta_info = {
            "eos_token_id": generation_eos if generation_eos is not None else self.tokenizer.eos_token_id,
            "pad_token_id": generation_pad if generation_pad is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            ######
            # MODIFY: apply pad_to and generation_kwargs
            ######
            output = self.rollout.generate_sequences(prompts=prompts, 
                                                     pad_to=prompts.meta_info.get("pad_to", None),
                                                     **prompts.meta_info.get("generation_kwargs", {}))
            output = self.rollout_sharding_manager.postprocess_data(output)
            if torch.distributed.get_rank() == 0:
                print(f"{_now()} data postprocessed")
        if torch.distributed.get_rank() == 0:
            print(f"{_now()} left rollout sharding manager")

        output = output.to("cpu")

        # clear kv cache
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    def _current_adapter_state_metadata(self):
        from peft import get_peft_model_state_dict

        from verl.utils.checkpoint.reproduction import canonical_tensor_state_sha256

        adapter_state = get_peft_model_state_dict(self.actor_module)
        tensor_keys = tuple(sorted(adapter_state))
        return {
            "adapter_tensor_keys": list(tensor_keys),
            "adapter_state_sha256": canonical_tensor_state_sha256(adapter_state),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_reproduction_step_telemetry(self, sample_nvml=True):
        """Reset CUDA peaks and start whole-card sampling for one optimizer loop."""

        if not self._is_actor or self._is_ref:
            raise RuntimeError(
                "whole-process resource telemetry requires an actor-only worker"
            )
        if type(sample_nvml) is not bool:
            raise TypeError("sample_nvml must be boolean")
        active = getattr(self, "_reproduction_telemetry_sampler", None)
        if active is not None:
            raise RuntimeError("a reproduction telemetry sampler is already active")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.actor._reproduction_logits_evidence = None
        initial = _query_single_gpu_nvml_memory()
        host_memory_probe = _create_reproduction_host_memory_probe()
        host = _sample_reproduction_host_memory(host_memory_probe)
        stop_event = threading.Event()
        started = time.monotonic()
        state = {
            "allocator_retry_start": int(
                torch.cuda.memory_stats().get("num_alloc_retries", 0)
            ),
            "error": None,
            "gpu_uuid": initial["gpu_uuid"],
            "host_peak_used_bytes": host.used,
            "host_memory_probe": host_memory_probe,
            "lock": threading.Lock(),
            "nvml_peak_used_bytes": initial["used_bytes"],
            "nvml_total_bytes": initial["total_bytes"],
            "phase_host_peak_used_bytes": host.used,
            "phase_name": _REPRODUCTION_TELEMETRY_PHASES[0],
            "phase_nvml_peak_used_bytes": initial["used_bytes"],
            "phase_started_monotonic": started,
            "phase_swap_peak_used_bytes": host.swap_used,
            "phases": [],
            "sample_nvml": sample_nvml,
            "started_monotonic": started,
            "stop_event": stop_event,
            "swap_peak_used_bytes": host.swap_used,
        }

        def sample():
            while not stop_event.wait(0.1):
                try:
                    observed = _query_single_gpu_nvml_memory()
                    if (
                        observed["gpu_uuid"] != state["gpu_uuid"]
                        or observed["total_bytes"] != state["nvml_total_bytes"]
                    ):
                        raise RuntimeError("GPU identity changed during telemetry sampling")
                    host = _sample_reproduction_host_memory(
                        state["host_memory_probe"]
                    )
                    host_used = host.used
                    swap_used = host.swap_used
                    with state["lock"]:
                        state["nvml_peak_used_bytes"] = max(
                            state["nvml_peak_used_bytes"], observed["used_bytes"]
                        )
                        state["host_peak_used_bytes"] = max(
                            state["host_peak_used_bytes"], host_used
                        )
                        state["swap_peak_used_bytes"] = max(
                            state["swap_peak_used_bytes"], swap_used
                        )
                        state["phase_nvml_peak_used_bytes"] = max(
                            state["phase_nvml_peak_used_bytes"], observed["used_bytes"]
                        )
                        state["phase_host_peak_used_bytes"] = max(
                            state["phase_host_peak_used_bytes"], host_used
                        )
                        state["phase_swap_peak_used_bytes"] = max(
                            state["phase_swap_peak_used_bytes"], swap_used
                        )
                except Exception as exc:
                    state["error"] = f"{type(exc).__name__}: {exc}"
                    stop_event.set()

        thread = threading.Thread(
            target=sample,
            name="rememr1-nvml-telemetry",
            daemon=True,
        )
        state["thread"] = thread if sample_nvml else None
        self._reproduction_telemetry_sampler = state
        if sample_nvml:
            thread.start()
        return {"rank": self.rank, "role": "actor", "status": "reset"}

    def _close_reproduction_telemetry_phase(self, state):
        torch.cuda.synchronize()
        observed = _query_single_gpu_nvml_memory()
        if (
            observed["gpu_uuid"] != state["gpu_uuid"]
            or observed["total_bytes"] != state["nvml_total_bytes"]
        ):
            raise RuntimeError("GPU identity changed during telemetry sampling")
        host = _sample_reproduction_host_memory(state["host_memory_probe"])
        ended = time.monotonic()
        with state["lock"]:
            state["nvml_peak_used_bytes"] = max(
                state["nvml_peak_used_bytes"], observed["used_bytes"]
            )
            state["host_peak_used_bytes"] = max(
                state["host_peak_used_bytes"], host.used
            )
            state["swap_peak_used_bytes"] = max(
                state["swap_peak_used_bytes"], host.swap_used
            )
            phase = {
                "duration_seconds": ended - state["phase_started_monotonic"],
                "host_peak_used_bytes": max(
                    state["phase_host_peak_used_bytes"], host.used
                ),
                "name": state["phase_name"],
                "nvml_peak_used_bytes": max(
                    state["phase_nvml_peak_used_bytes"], observed["used_bytes"]
                ),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "post_allocated_bytes": torch.cuda.memory_allocated(),
                "post_reserved_bytes": torch.cuda.memory_reserved(),
                "swap_peak_used_bytes": max(
                    state["phase_swap_peak_used_bytes"], host.swap_used
                ),
            }
            state["phases"].append(phase)
        return observed, host

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def advance_reproduction_step_telemetry(self, next_phase):
        """Seal one phase window and reset CUDA peaks for the next phase."""

        state = getattr(self, "_reproduction_telemetry_sampler", None)
        if state is None:
            raise RuntimeError("reproduction telemetry was not reset for this step")
        completed_count = len(state["phases"])
        if completed_count >= len(_REPRODUCTION_TELEMETRY_PHASES) - 1:
            raise RuntimeError("reproduction telemetry has no next phase")
        expected_current = _REPRODUCTION_TELEMETRY_PHASES[completed_count]
        expected_next = _REPRODUCTION_TELEMETRY_PHASES[completed_count + 1]
        if state["phase_name"] != expected_current or next_phase != expected_next:
            raise RuntimeError(
                f"telemetry phase transition must be {expected_current}->{expected_next}"
            )
        observed, host = self._close_reproduction_telemetry_phase(state)
        torch.cuda.reset_peak_memory_stats()
        with state["lock"]:
            state["phase_name"] = next_phase
            state["phase_started_monotonic"] = time.monotonic()
            state["phase_nvml_peak_used_bytes"] = observed["used_bytes"]
            state["phase_host_peak_used_bytes"] = host.used
            state["phase_swap_peak_used_bytes"] = host.swap_used
        return {"phase": expected_current, "rank": self.rank, "status": "sealed"}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def collect_reproduction_step_telemetry(self):
        """Stop sampling and return JSON-safe CUDA/NVML evidence for this rank."""

        state = getattr(self, "_reproduction_telemetry_sampler", None)
        if state is None:
            raise RuntimeError("reproduction telemetry was not reset for this step")
        if state["phase_name"] != _REPRODUCTION_TELEMETRY_PHASES[-1]:
            raise RuntimeError("reproduction telemetry did not reach the save phase")
        state["stop_event"].set()
        if state["thread"] is not None:
            state["thread"].join(timeout=10)
            if state["thread"].is_alive():
                raise RuntimeError("NVML telemetry sampler did not stop")
        final, host = self._close_reproduction_telemetry_phase(state)
        error = state["error"]
        self._reproduction_telemetry_sampler = None
        if error is not None:
            raise RuntimeError(f"NVML telemetry sampler failed: {error}")
        memory_stats = torch.cuda.memory_stats()
        phases = state["phases"]
        if [phase["name"] for phase in phases] != list(
            _REPRODUCTION_TELEMETRY_PHASES
        ):
            raise RuntimeError("reproduction telemetry phase inventory is incomplete")
        return {
            "actor_logits": self.actor._reproduction_logits_evidence,
            "allocator_retry_count": max(
                0,
                int(memory_stats.get("num_alloc_retries", 0))
                - state["allocator_retry_start"],
            ),
            "gpu_uuid": state["gpu_uuid"],
            "host_peak_used_bytes": max(state["host_peak_used_bytes"], host.used),
            "host_total_memory_bytes": host.total,
            "nvml_peak_used_bytes": state["nvml_peak_used_bytes"],
            "nvml_total_bytes": state["nvml_total_bytes"],
            "peak_allocated_bytes": max(
                phase["peak_allocated_bytes"] for phase in phases
            ),
            "peak_reserved_bytes": max(
                phase["peak_reserved_bytes"] for phase in phases
            ),
            "phase_records": phases,
            "post_step_allocated_bytes": torch.cuda.memory_allocated(),
            "post_step_nvml_used_bytes": final["used_bytes"],
            "post_step_reserved_bytes": torch.cuda.memory_reserved(),
            "rank": self.rank,
            "reference_logits": None,
            "role": "actor",
            "step_wall_seconds": time.monotonic() - state["started_monotonic"],
            "swap_used_bytes": max(
                state["swap_peak_used_bytes"], host.swap_used
            ),
            "world_size": self.world_size,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_reproduction_reference_logits(self):
        """Clear reference-logits evidence without touching process-wide CUDA peaks."""

        if not self._is_ref or self._is_actor:
            raise RuntimeError(
                "reference logits telemetry requires a reference-only worker"
            )
        if getattr(self, "_reproduction_reference_logits_pending", False):
            raise RuntimeError("reference logits telemetry is already active")
        self.ref_policy._reproduction_logits_evidence = None
        self._reproduction_reference_logits_pending = True
        return {"rank": self.rank, "role": "reference", "status": "reset"}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def collect_reproduction_reference_logits(self):
        """Return logits plus zero resource placeholders for the colocated reference."""

        if not self._is_ref or self._is_actor:
            raise RuntimeError(
                "reference logits telemetry requires a reference-only worker"
            )
        if not getattr(self, "_reproduction_reference_logits_pending", False):
            raise RuntimeError("reference logits telemetry was not reset for this step")
        logits = self.ref_policy._reproduction_logits_evidence
        if logits is None:
            raise RuntimeError("reference worker produced no logits evidence")
        gpu = _query_single_gpu_nvml_memory()
        host = _sample_reproduction_host_memory(
            _create_reproduction_host_memory_probe()
        )
        self._reproduction_reference_logits_pending = False
        zero_phases = [
            {
                "duration_seconds": 0.0,
                "host_peak_used_bytes": 0,
                "name": name,
                "nvml_peak_used_bytes": 0,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
                "post_allocated_bytes": 0,
                "post_reserved_bytes": 0,
                "swap_peak_used_bytes": 0,
            }
            for name in _REPRODUCTION_TELEMETRY_PHASES
        ]
        return {
            "actor_logits": None,
            "allocator_retry_count": 0,
            "gpu_uuid": gpu["gpu_uuid"],
            "host_peak_used_bytes": 0,
            "host_total_memory_bytes": host.total,
            "nvml_peak_used_bytes": 0,
            "nvml_total_bytes": gpu["total_bytes"],
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "phase_records": zero_phases,
            "post_step_allocated_bytes": 0,
            "post_step_nvml_used_bytes": 0,
            "post_step_reserved_bytes": 0,
            "rank": self.rank,
            "reference_logits": logits,
            "role": "reference",
            "step_wall_seconds": 0.0,
            "swap_used_bytes": 0,
            "world_size": self.world_size,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_reproduction_build_metadata(self):
        """Return immutable JSON-safe actor identity used by strict checkpoints."""

        assert self._is_actor
        from verl.utils.checkpoint.reproduction import to_json_safe_state

        build_metadata = self.model_build_metadata.get("actor")
        if not isinstance(build_metadata, dict):
            raise RuntimeError("actor build metadata is unavailable")
        trainable = build_metadata.get("trainable_parameters")
        if not isinstance(trainable, dict) or not trainable.get("state_sha256"):
            raise RuntimeError(
                "initial trainable adapter state hash is unavailable"
            )
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "build_metadata": to_json_safe_state(build_metadata),
            "initial_adapter_state_sha256": trainable["state_sha256"],
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_initial_adapter_state_sha256(self):
        """Expose the seeded step-0 adapter fingerprint to the driver."""

        metadata = self.get_reproduction_build_metadata()
        return metadata["initial_adapter_state_sha256"]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_reproduction_rng_state(self):
        """Return a JSON-safe snapshot without mutating any RNG stream."""

        assert self._is_actor
        from verl.utils.checkpoint.reproduction import (
            capture_process_rng_state,
            json_safe_state_sha256,
            to_json_safe_state,
        )

        rng_state = capture_process_rng_state()
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "rng_state": to_json_safe_state(rng_state),
            "rng_state_sha256": json_safe_state_sha256(rng_state),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_reproduction_adapter(
        self,
        local_path,
        checkpoint_extra_state,
        tokenizer_id,
        tokenizer_revision,
        template_revision,
    ):
        """Export only PEFT tensors while full FSDP parameters are summoned."""

        assert self._is_actor
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from verl.utils.checkpoint.reproduction import (
            AdapterExportMetadata,
            CheckpointContractError,
            CheckpointExtraState,
            canonical_tensor_state_sha256,
            export_peft_adapter,
        )
        from peft import get_peft_model_state_dict

        if not os.path.isabs(local_path):
            raise CheckpointContractError(
                "reproduction adapter export path must be absolute"
            )
        state = CheckpointExtraState.from_dict(checkpoint_extra_state)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        exported_metadata = None
        with FSDP.summon_full_params(
            self.actor_module_fsdp,
            recurse=True,
            writeback=False,
            rank0_only=True,
            # PyTorch converts single-rank FULL_SHARD to NO_SHARD, where
            # summon_full_params does not support CPU offload.
            offload_to_cpu=self.world_size > 1,
        ):
            if self.rank == 0:
                if not hasattr(self.actor_module, "peft_config"):
                    raise CheckpointContractError(
                        "adapter export requires a PEFT actor model"
                    )
                adapter_state = get_peft_model_state_dict(self.actor_module)
                adapter_state_sha256 = canonical_tensor_state_sha256(adapter_state)
                if tuple(sorted(adapter_state)) != state.adapter_tensor_keys:
                    raise CheckpointContractError(
                        "live adapter tensor keys differ from checkpoint extra-state"
                    )
                if adapter_state_sha256 != state.adapter_state_sha256:
                    raise CheckpointContractError(
                        "live adapter tensor-state hash differs from checkpoint extra-state"
                    )
                metadata = AdapterExportMetadata.from_checkpoint(
                    state,
                    tokenizer_id=tokenizer_id,
                    tokenizer_revision=tokenizer_revision,
                    template_revision=template_revision,
                )
                export_peft_adapter(self.actor_module, local_path, metadata)
                exported_metadata = metadata.to_dict()

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return exported_metadata

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
        reproduction=False,
    ):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        save_metadata = self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            reproduction=reproduction,
        )

        torch.distributed.barrier()
        if reproduction:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(
                self.actor_module_fsdp,
                recurse=True,
                writeback=False,
                rank0_only=True,
                offload_to_cpu=self.world_size > 1,
            ):
                if self.rank == 0:
                    save_metadata.update(self._current_adapter_state_metadata())
            torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return save_metadata

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        del_local_after_load=False,
        reproduction=False,
        expected_global_step=None,
    ):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
            reproduction=reproduction,
            expected_global_step=expected_global_step,
        )

        loaded_metadata = None
        if reproduction:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(
                self.actor_module_fsdp,
                recurse=True,
                writeback=False,
                rank0_only=True,
                offload_to_cpu=self.world_size > 1,
            ):
                if self.rank == 0:
                    loaded_metadata = self._current_adapter_state_metadata()
            torch.distributed.barrier()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)
        return loaded_metadata


class CriticWorker(Worker):
    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.forward_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        local_path = copy_to_local(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        init_context = get_init_weight_context_manager(use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=critic_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            use_remove_padding = config.model.get("use_remove_padding", False)
            if use_remove_padding or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=critic_module, ulysses_sp_size=self.ulysses_sequence_parallel_size)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        critic_module = FSDP(
            critic_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        warmup_style = config.optim.get("warmup_style", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if warmup_style == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps)
        elif warmup_style == "cosine":
            critic_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForTokenClassification

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if config.model.get("use_remove_padding", False) or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=reward_module, ulysses_sp_size=self.ulysses_sequence_parallel_size)

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=True),
            forward_prefetch=False,
            device_mesh=self.device_mesh,
        )

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

        from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False)  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}")
        return self.rollout.execute_method(method, *args, **kwargs)
