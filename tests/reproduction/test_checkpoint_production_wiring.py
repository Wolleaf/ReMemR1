import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGER_PATH = REPO_ROOT / "verl" / "utils" / "checkpoint" / "fsdp_checkpoint_manager.py"
WORKER_PATH = REPO_ROOT / "verl" / "workers" / "fsdp_workers.py"
TRAINER_PATH = REPO_ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
CONFIG_PATH = REPO_ROOT / "verl" / "trainer" / "config" / "ppo_trainer.yaml"
EXPORTER_PATH = REPO_ROOT / "scripts" / "reproduction" / "export_adapter.py"


def _class_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _calls(method, name):
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        )
    ]


def test_manager_strict_save_captures_step_scheduler_and_raw_rng_without_pruning():
    method = _class_method(
        MANAGER_PATH,
        "FSDPCheckpointManager",
        "save_checkpoint",
    )
    source = ast.unparse(method)

    assert "if reproduction" in source
    assert "max_ckpt_to_keep is not None" in source
    assert "in-save retention" in source
    assert "capture_process_rng_state()" in source
    assert "'global_step': global_step" in source
    assert "'lr_scheduler': lr_scheduler_state_dict" in source
    assert "validate_reproduction_rank_extra_state" in source
    assert "if not reproduction and max_ckpt_to_keep" in source


def test_manager_verifies_root_before_loading_and_never_tolerates_missing_strict_state():
    method = _class_method(
        MANAGER_PATH,
        "FSDPCheckpointManager",
        "load_checkpoint",
    )
    verify_call = _calls(method, "verify_reproduction_checkpoint_directory")[0]
    first_torch_load = min(_calls(method, "load"), key=lambda node: node.lineno)
    source = ast.unparse(method)

    assert verify_call.lineno < first_torch_load.lineno
    assert "_, reproduction_state = verify_reproduction_checkpoint_directory" in source
    assert "validate_reproduction_rank_extra_state" in source
    assert "restore_process_rng_state" in source
    assert "missing optimizer state" in source
    assert "requires an lr_scheduler" in source


def test_manager_uses_fsdp_optimizer_translation_and_cross_checks_scheduler():
    save_method = _class_method(
        MANAGER_PATH,
        "FSDPCheckpointManager",
        "save_checkpoint",
    )
    load_method = _class_method(
        MANAGER_PATH,
        "FSDPCheckpointManager",
        "load_checkpoint",
    )
    save_source = ast.unparse(save_method)
    load_source = ast.unparse(load_method)

    assert "FSDP.optim_state_dict(self.model, self.optimizer)" in save_source
    assert "FSDP.optim_state_dict_to_load" in load_source
    assert "validate_scheduler_optimizer_alignment" in save_source
    assert "validate_scheduler_optimizer_alignment" in load_source


def test_worker_exposes_json_safe_build_rng_and_initial_adapter_fingerprint():
    build_getter = _class_method(
        WORKER_PATH,
        "ActorRolloutRefWorker",
        "get_reproduction_build_metadata",
    )
    rng_getter = _class_method(
        WORKER_PATH,
        "ActorRolloutRefWorker",
        "get_reproduction_rng_state",
    )
    hash_getter = _class_method(
        WORKER_PATH,
        "ActorRolloutRefWorker",
        "get_initial_adapter_state_sha256",
    )

    assert "to_json_safe_state(build_metadata)" in ast.unparse(build_getter)
    assert "trainable['state_sha256']" in ast.unparse(build_getter)
    assert "capture_process_rng_state()" in ast.unparse(rng_getter)
    assert "json_safe_state_sha256(rng_state)" in ast.unparse(rng_getter)
    assert "initial_adapter_state_sha256" in ast.unparse(hash_getter)


def test_worker_adapter_export_uses_full_param_context_and_adapter_only_helper():
    method = _class_method(
        WORKER_PATH,
        "ActorRolloutRefWorker",
        "export_reproduction_adapter",
    )
    source = ast.unparse(method)

    assert "FSDP.summon_full_params" in source
    assert "rank0_only=True" in source
    assert "get_peft_model_state_dict(self.actor_module)" in source
    assert "canonical_tensor_state_sha256(adapter_state)" in source
    assert "state.adapter_tensor_keys" in source
    assert "state.adapter_state_sha256" in source
    assert "AdapterExportMetadata.from_checkpoint" in source
    assert "export_peft_adapter(self.actor_module" in source


def test_worker_full_param_context_avoids_unsupported_single_rank_cpu_offload():
    for method_name in (
        "export_reproduction_adapter",
        "save_checkpoint",
        "load_checkpoint",
    ):
        method = _class_method(
            WORKER_PATH,
            "ActorRolloutRefWorker",
            method_name,
        )
        summon_calls = _calls(method, "summon_full_params")

        assert len(summon_calls) == 1
        keywords = {keyword.arg: keyword.value for keyword in summon_calls[0].keywords}
        assert ast.unparse(keywords["offload_to_cpu"]) == "self.world_size > 1"
        assert ast.unparse(keywords["rank0_only"]) == "True"


def test_trainer_publishes_whole_global_step_before_updating_tracker():
    method = _class_method(
        TRAINER_PATH,
        "RayPPOTrainer",
        "_save_reproduction_checkpoint",
    )
    source = ast.unparse(method)
    publish_call = _calls(method, "atomic_publish_directory")[0]
    tracker_call = _calls(method, "atomic_write_text")[0]

    assert "staging / 'actor'" in source
    assert "staging / 'data.pt'" in source
    assert "staging / 'driver_rng.pt'" in source
    assert "staging / EXTRA_STATE_FILENAME" in source
    assert "reproduction=True" in source
    assert publish_call.lineno < tracker_call.lineno


def test_trainer_strict_resume_verifies_schema_identity_and_all_raw_artifacts_first():
    method = _class_method(
        TRAINER_PATH,
        "RayPPOTrainer",
        "_load_reproduction_checkpoint",
    )
    source = ast.unparse(method)
    verify_call = _calls(method, "verify_reproduction_checkpoint_directory")[0]
    actor_load = _calls(method, "load_checkpoint")[0]

    assert verify_call.lineno < actor_load.lineno
    assert "_, state = verify_reproduction_checkpoint_directory(root)" in source
    assert "validate_checkpoint_compatibility" in source
    assert "_validate_reproduction_checkpoint_artifacts" in source
    assert "reproduction=True" in source
    assert "restore_process_rng_state" in source
    assert "Warning: No dataloader state" not in source


def test_trainer_formal_mode_is_explicit_and_binds_dataset_manifest():
    init_method = _class_method(TRAINER_PATH, "RayPPOTrainer", "__init__")
    validate_method = _class_method(TRAINER_PATH, "RayPPOTrainer", "_validate_config")
    dataset_method = _class_method(
        TRAINER_PATH,
        "RayPPOTrainer",
        "_validate_reproduction_dataset_manifest",
    )
    init_source = ast.unparse(init_method)
    validate_source = ast.unparse(validate_method)
    dataset_source = ast.unparse(dataset_method)

    assert "reproduction.get('run_seed') is not None" in init_source
    assert "formal_data') is True" in init_source
    assert "requires reproduction.run_seed" in init_source
    assert "{'disable', 'resume_path'}" in validate_source
    assert "auto resume is forbidden" in validate_source
    assert "config.recurrent.enable != 'memory'" in validate_source
    assert "require_manifest" in validate_source
    assert "lr_warmup_steps=8" in validate_source
    assert "validate_reproduction_manifest_configuration" in validate_source
    assert dataset_source.count("validate_bound_dataset_manifest") == 2
    assert "self.train_dataset" in dataset_source
    assert "self.val_dataset" in dataset_source
    assert "reproduction.val_data_manifest_sha256" in dataset_source


def test_step_zero_is_paired_no_clobber_and_finalized_before_reward_or_update():
    validate_method = _class_method(
        TRAINER_PATH,
        "RayPPOTrainer",
        "_validate_step_zero_configuration",
    )
    record_method = _class_method(
        TRAINER_PATH,
        "RayPPOTrainer",
        "_record_step_zero_fingerprint",
    )
    fit_method = _class_method(TRAINER_PATH, "RayPPOTrainer", "fit")
    validate_source = ast.unparse(validate_method)
    record_source = ast.unparse(record_method)
    fit_source = ast.unparse(fit_method)

    assert "fresh run refuses to overwrite step-zero evidence" in validate_source
    assert "load_and_verify_step_zero_fingerprint" in validate_source
    assert "hash_ordered_sample_ids(ordered_sample_ids)" in record_source
    assert "hash_sampled_token_rows(sampled_token_rows)" in record_source
    assert "publish_step_zero_fingerprint" in record_source
    assert "manifest_qa_id" in fit_source
    assert ".detach().cpu().tolist()" in fit_source

    fingerprint_call = _calls(fit_method, "_record_step_zero_fingerprint")[0]
    reward_call = min(_calls(fit_method, "compute_reward"), key=lambda node: node.lineno)
    update_call = _calls(fit_method, "update_actor")[0]
    assert fingerprint_call.lineno < reward_call.lineno < update_call.lineno


def test_trainer_constructs_each_dataloader_once_with_controlled_workers():
    create_method = _class_method(TRAINER_PATH, "RayPPOTrainer", "_create_dataloader")
    validate_method = _class_method(TRAINER_PATH, "RayPPOTrainer", "_validate_config")
    create_source = ast.unparse(create_method)
    validate_source = ast.unparse(validate_method)

    assert create_source.count("self.train_dataloader = StatefulDataLoader") == 1
    assert create_source.count("num_workers=dataloader_num_workers") == 2
    assert "self.config.data.get('dataloader_num_workers', 8)" in create_source
    assert "dataloader_num_workers < 0" in validate_source
    assert "non-negative integer" in validate_source


def test_trainer_deserializes_every_actor_shard_before_publish():
    artifact_method = _class_method(
        TRAINER_PATH,
        "RayPPOTrainer",
        "_validate_reproduction_checkpoint_artifacts",
    )
    save_method = _class_method(
        TRAINER_PATH,
        "RayPPOTrainer",
        "_save_reproduction_checkpoint",
    )
    artifact_source = ast.unparse(artifact_method)
    save_source = ast.unparse(save_method)

    assert "torch.load(extra_path, weights_only=False)" in artifact_source
    assert "torch.load(model_path, weights_only=False)" in artifact_source
    assert "torch.load(optimizer_path, weights_only=False)" in artifact_source
    assert "validate_scheduler_optimizer_alignment" in artifact_source
    assert "allow_atomic_staging_name=allow_atomic_staging_name" in artifact_source
    assert "allow_atomic_staging_name=True" in save_source
    assert "validator=validator" in save_source
    assert "'artifact_type': 'reproduction_training_checkpoint'" in save_source


def test_formal_config_exposes_manifest_and_export_contract_and_disables_retention():
    source = CONFIG_PATH.read_text(encoding="utf-8")

    for fragment in (
        "data_manifest_sha256: null",
        "val_data_manifest_sha256: null",
        "step_zero_fingerprint_path: null",
        "step_zero_reference_path: null",
        "export_adapter_on_save: False",
        "adapter_export_dir: null",
        "template_revision: null",
        "max_actor_ckpt_to_keep: null",
        "max_critic_ckpt_to_keep: null",
    ):
        assert fragment in source


def test_exporter_lazily_merges_saves_reloads_and_compares_logits_and_generation():
    tree = ast.parse(EXPORTER_PATH.read_text(encoding="utf-8"), filename=str(EXPORTER_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "merge_and_verify_adapter"
    )
    source = ast.unparse(function)
    validator = next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "validator"
    )
    validator_source = ast.unparse(validator)

    assert "validate_adapter_export(adapter_path)" in source
    assert "revision=metadata.tokenizer_revision" in source
    assert "_load_pinned_base_model" in source
    assert "load_peft_adapter(base_model, adapter_path)" in source
    assert "adapter_model.to(device=device, dtype=dtype)" in source
    assert "merge_and_unload(safe_merge=True)" in source
    assert "build_merged_model_metadata" in source
    assert "atomic_publish_directory" in source
    assert "validator=validator" in source
    assert "validate_merged_model_artifact" in validator_source
    assert "AutoModelForCausalLM.from_pretrained(staging" in validator_source
    assert validator_source.count("torch.testing.assert_close") == 1
    assert validator_source.count("torch.equal") == 1


def test_exporter_strict_qwen_loader_and_bf16_model_contract_are_fail_closed():
    tree = ast.parse(EXPORTER_PATH.read_text(encoding="utf-8"), filename=str(EXPORTER_PATH))
    strict_loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_pinned_base_model"
    )
    merged_contract = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_assert_merged_model_contract"
    )
    loader_source = ast.unparse(strict_loader)
    contract_source = ast.unparse(merged_contract)

    assert "load_qwen35_text_model" in loader_source
    assert "revision=metadata.base_model_revision" in loader_source
    assert "strict=True" in loader_source
    assert "metadata.text_mapping_sha256" in loader_source
    assert "hasattr(model, 'peft_config')" in contract_source
    assert "merged model still contains PEFT/LoRA modules" in contract_source
    assert "parameter.dtype != torch_module.bfloat16" in contract_source
    assert "canonical_tensor_state_sha256(model.state_dict())" in contract_source
