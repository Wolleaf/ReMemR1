import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_checkpoint_modules_without_verl_runtime(monkeypatch):
    for name in (
        "verl.utils.checkpoint.fsdp_checkpoint_manager",
        "verl.utils.checkpoint.checkpoint_manager",
        "verl.utils.checkpoint.reproduction",
        "verl.utils.checkpoint",
        "verl.utils.fs",
        "verl.utils",
        "verl",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name, path in (
        ("verl", REPO_ROOT / "verl"),
        ("verl.utils", REPO_ROOT / "verl" / "utils"),
        ("verl.utils.checkpoint", REPO_ROOT / "verl" / "utils" / "checkpoint"),
    ):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    fs_module = ModuleType("verl.utils.fs")
    fs_module.copy_to_local = lambda src, cache_dir=None: src
    fs_module.is_non_local = lambda unused: False
    monkeypatch.setitem(sys.modules, "verl.utils.fs", fs_module)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    checkpoint_dir = REPO_ROOT / "verl" / "utils" / "checkpoint"
    reproduction = load(
        "verl.utils.checkpoint.reproduction",
        checkpoint_dir / "reproduction.py",
    )
    load(
        "verl.utils.checkpoint.checkpoint_manager",
        checkpoint_dir / "checkpoint_manager.py",
    )
    manager = load(
        "verl.utils.checkpoint.fsdp_checkpoint_manager",
        checkpoint_dir / "fsdp_checkpoint_manager.py",
    )
    return manager, reproduction


def test_real_tiny_peft_fsdp_optimizer_scheduler_roundtrip(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    manager_module, reproduction = _load_checkpoint_modules_without_verl_runtime(
        monkeypatch
    )
    FSDPCheckpointManager = manager_module.FSDPCheckpointManager
    CheckpointExtraState = reproduction.CheckpointExtraState
    DataloaderProgress = reproduction.DataloaderProgress
    atomic_publish_directory = reproduction.atomic_publish_directory
    canonical_tensor_state_sha256 = reproduction.canonical_tensor_state_sha256

    rendezvous = (tmp_path / "gloo-init").resolve().as_posix()
    initialized_here = not torch.distributed.is_initialized()
    if initialized_here:
        try:
            torch.distributed.init_process_group(
                "gloo",
                rank=0,
                world_size=1,
                init_method=f"file:///{rendezvous}",
            )
        except RuntimeError as exc:
            pytest.skip(
                "Windows Torch/Gloo cannot create the CPU process group needed "
                f"for a real FSDP roundtrip: {exc}"
            )

    class FakeProcessingClass:
        def save_pretrained(self, directory):
            Path(directory, "tokenizer_config.json").write_text(
                "{}",
                encoding="utf-8",
            )

    def build_stack(seed):
        torch.manual_seed(seed)
        config = transformers.GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
        )
        model = transformers.GPT2LMHeadModel(config)
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["c_attn"],
            ),
        )
        try:
            fsdp = FSDP(
                model,
                device_id=torch.device("cpu"),
                use_orig_params=True,
            )
        except RuntimeError as exc:
            if "accelerator" in str(exc).casefold() or "cpu" in str(exc).casefold():
                pytest.skip(f"Torch FSDP CPU is unsupported on this platform: {exc}")
            raise
        trainable = [parameter for parameter in fsdp.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0,
        )
        manager = FSDPCheckpointManager(
            model=fsdp,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            processing_class=FakeProcessingClass(),
            checkpoint_contents=["model", "optimizer", "extra"],
        )
        return model, fsdp, optimizer, scheduler, manager

    try:
        model, fsdp, optimizer, scheduler, manager = build_stack(7)
        input_ids = torch.tensor([[1, 2, 3]])
        loss = fsdp(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        assert scheduler.last_epoch == 1
        fsdp.eval()
        with torch.no_grad():
            expected_logits = fsdp(input_ids=input_ids).logits.detach().clone()
        with FSDP.summon_full_params(fsdp, writeback=False):
            adapter_state = peft.get_peft_model_state_dict(model)
            adapter_keys = tuple(sorted(adapter_state))
            adapter_hash = canonical_tensor_state_sha256(adapter_state)

        text_mapping = {"mapping_strategy": "tiny_fsdp_test"}
        lora_config = {
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "bias": "none",
        }
        target_manifest = {"schema_version": 1, "target_modules": ["c_attn"]}
        build_metadata = {"model_type": "tiny-gpt2", "use_orig_params": True}
        resolved_config = {
            "algorithm": {"alpha": 0.5},
            "actor_rollout_ref": {
                "actor": {"optim": {"total_training_steps": 1}}
            },
            "trainer": {"resume_mode": "disable", "total_training_steps": 1},
        }
        extra_state = CheckpointExtraState.create(
            global_step=1,
            rng_state={"rank": 0, "state": "covered-by-raw-extra"},
            dataloader_state=DataloaderProgress(state_sha256="a" * 64),
            data_manifest_sha256="b" * 64,
            base_model_id="tiny-local-gpt2",
            base_model_revision="tiny-v1",
            text_mapping=text_mapping,
            lora_config=lora_config,
            lora_target_manifest=target_manifest,
            adapter_tensor_keys=adapter_keys,
            adapter_state_sha256=adapter_hash,
            model_build_metadata=build_metadata,
            resolved_config=resolved_config,
        )
        destination = tmp_path / "global_step_1"

        def writer(staging):
            manager.save_checkpoint(
                str(staging / "actor"),
                global_step=1,
                max_ckpt_to_keep=None,
                reproduction=True,
            )
            extra_state.save(staging / "reproduction_extra_state.json")

        atomic_publish_directory(
            destination,
            writer,
            marker_metadata={
                "artifact_type": "reproduction_training_checkpoint",
                "global_step": 1,
            },
        )

        fresh_model, fresh_fsdp, fresh_optimizer, fresh_scheduler, fresh_manager = (
            build_stack(999)
        )
        fresh_manager.load_checkpoint(
            str(destination / "actor"),
            reproduction=True,
            expected_global_step=1,
        )
        fresh_fsdp.eval()
        with torch.no_grad():
            actual_logits = fresh_fsdp(input_ids=input_ids).logits
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-5)
        assert fresh_scheduler.last_epoch == scheduler.last_epoch == 1
        assert fresh_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        with FSDP.summon_full_params(fresh_fsdp, writeback=False):
            loaded_adapter = peft.get_peft_model_state_dict(fresh_model)
            assert tuple(sorted(loaded_adapter)) == adapter_keys
            assert canonical_tensor_state_sha256(loaded_adapter) == adapter_hash
    finally:
        if initialized_here and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
