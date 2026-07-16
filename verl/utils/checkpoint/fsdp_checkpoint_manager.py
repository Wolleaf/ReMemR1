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

import os
from pathlib import Path
import warnings
from typing import Optional, Union

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.fs import copy_to_local, is_non_local

from .checkpoint_manager import BaseCheckpointManager
from .reproduction import (
    REPRODUCTION_RANK_EXTRA_SCHEMA_VERSION,
    CheckpointContractError,
    capture_process_rng_state,
    json_safe_state_sha256,
    restore_process_rng_state,
    to_json_safe_state,
    validate_reproduction_rank_extra_state,
    validate_scheduler_optimizer_alignment,
    verify_reproduction_checkpoint_directory,
)


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer/processor and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        **kwargs,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra", 'hf_model']
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")
        assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'optimizer', 'extra'], got {checkpoint_contents}"

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        del_local_after_load=False,
        reproduction=False,
        expected_global_step=None,
    ):
        if local_path is None:
            return

        reproduction_state = None
        if reproduction:
            if hdfs_path is not None:
                raise CheckpointContractError(
                    "reproduction checkpoint load does not support HDFS"
                )
            if del_local_after_load:
                raise CheckpointContractError(
                    "reproduction checkpoint load cannot delete verified shards"
                )
            if not os.path.isabs(local_path):
                raise CheckpointContractError(
                    "reproduction actor checkpoint path must be absolute"
                )
            checkpoint_root = Path(local_path).parent
            _, reproduction_state = verify_reproduction_checkpoint_directory(
                checkpoint_root
            )
            if expected_global_step is None:
                expected_global_step = reproduction_state.global_step
            if reproduction_state.global_step != expected_global_step:
                raise CheckpointContractError(
                    "root extra-state global_step does not match requested resume step"
                )
            if self.optimizer is None:
                raise CheckpointContractError(
                    "reproduction actor resume requires an optimizer"
                )
            if self.lr_scheduler is None:
                raise CheckpointContractError(
                    "reproduction actor resume requires an lr_scheduler"
                )

        # every rank download its own checkpoint
        remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}")
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        model_state_dict = torch.load(local_model_path, weights_only=False)
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

        if reproduction:
            if optimizer_state_dict is None:
                raise CheckpointContractError(
                    "reproduction actor checkpoint is missing optimizer state"
                )
            validate_reproduction_rank_extra_state(
                extra_state_dict,
                expected_global_step=expected_global_step,
            )
            validate_scheduler_optimizer_alignment(
                extra_state_dict["lr_scheduler"],
                optimizer_state_dict,
                expected_global_step=expected_global_step,
            )

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if reproduction:
                optimizer_state_dict = FSDP.optim_state_dict_to_load(
                    self.model,
                    self.optimizer,
                    optimizer_state_dict,
                )
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if reproduction:
            restore_process_rng_state(extra_state_dict["rng"])
        elif "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict["rng"])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        global_step: int = 0,
        max_ckpt_to_keep=None,
        reproduction=False,
    ):
        if local_path is None:
            return

        if reproduction:
            if hdfs_path is not None:
                raise CheckpointContractError(
                    "reproduction checkpoint save does not support HDFS"
                )
            if not os.path.isabs(local_path):
                raise CheckpointContractError(
                    "reproduction actor checkpoint path must be absolute"
                )
            if max_ckpt_to_keep is not None:
                raise CheckpointContractError(
                    "reproduction checkpoints disable in-save retention; "
                    "prune only after explicit verification"
                )
            if self.optimizer is None:
                raise CheckpointContractError(
                    "reproduction actor checkpoint requires optimizer state"
                )
            if self.lr_scheduler is None:
                raise CheckpointContractError(
                    "reproduction actor checkpoint requires lr_scheduler state"
                )

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        if not reproduction and max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                if self.optimizer is None:
                    optimizer_state_dict = None
                elif reproduction:
                    optimizer_state_dict = FSDP.optim_state_dict(
                        self.model,
                        self.optimizer,
                    )
                else:
                    optimizer_state_dict = self.optimizer.state_dict()
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None

                if reproduction:
                    rng_state = capture_process_rng_state()
                    extra_state_dict = {
                        "schema_version": REPRODUCTION_RANK_EXTRA_SCHEMA_VERSION,
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": rng_state,
                    }
                    validate_reproduction_rank_extra_state(
                        extra_state_dict,
                        expected_global_step=global_step,
                    )
                    validate_scheduler_optimizer_alignment(
                        lr_scheduler_state_dict,
                        optimizer_state_dict,
                        expected_global_step=global_step,
                    )
                else:
                    rng_state = self.get_rng_state()
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": rng_state,
                    }
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}")
                print(f"[rank-{self.rank}]: Saving optim to {os.path.abspath(optim_path)}")
                print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}")
                torch.save(model_state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)

        if "hf_model" in self.checkpoint_contents:
            # wait for everyone to dump to local
            torch.distributed.barrier()

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)
                self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
                self.processing_class.save_pretrained(hf_local_path)

        torch.distributed.barrier()

        if max_ckpt_to_keep is not None and max_ckpt_to_keep > 0:
            self.previous_saved_paths.append(local_path)

        if reproduction:
            return {
                "schema_version": REPRODUCTION_RANK_EXTRA_SCHEMA_VERSION,
                "rank": self.rank,
                "world_size": self.world_size,
                "global_step": global_step,
                "rng_state": to_json_safe_state(rng_state),
                "rng_state_sha256": json_safe_state_sha256(rng_state),
                "lr_scheduler_sha256": json_safe_state_sha256(
                    lr_scheduler_state_dict
                ),
            }
