import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = REPO_ROOT / "verl" / "workers" / "fsdp_workers.py"
BASE_CONFIG_PATH = REPO_ROOT / "verl" / "trainer" / "config" / "ppo_trainer.yaml"
MAIN_PPO_PATH = REPO_ROOT / "verl" / "trainer" / "main_ppo.py"


def _worker_method(name):
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"), filename=str(WORKER_PATH))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker"
    )
    return next(
        node
        for node in worker.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _calls(method, function_name):
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name) and node.func.id == function_name
            or isinstance(node.func, ast.Attribute) and node.func.attr == function_name
        )
    ]


def test_worker_uses_transformers5_text_loader_without_removed_vision_auto_class():
    method = _worker_method("_build_model_optimizer")
    source = ast.unparse(method)

    assert "AutoModelForVision2Seq" not in source
    loader_calls = _calls(method, "load_qwen35_text_model")
    assert len(loader_calls) == 1
    keywords = {keyword.arg for keyword in loader_calls[0].keywords}
    assert {"revision", "attn_implementation", "dtype", "config"} <= keywords


def test_model_seed_is_set_before_load_and_lora_injection():
    method = _worker_method("_build_model_optimizer")
    seed_call = _calls(method, "seed_process")[0]
    load_call = _calls(method, "load_qwen35_text_model")[0]
    inject_call = _calls(method, "inject_lora_adapter")[0]

    assert seed_call.lineno < load_call.lineno < inject_call.lineno


def test_actor_only_lora_and_trainable_only_optimizer_are_wired():
    method = _worker_method("_build_model_optimizer")
    source = ast.unparse(method)

    assert "role == 'actor'" in source
    assert "inject_lora_adapter(actor_module, target_manifest)" in source
    assert "assert_only_lora_parameters_trainable(actor_module)" in source
    assert "optimizer_parameters = trainable_optimizer_parameters(actor_module_fsdp)" in source
    assert "optim.AdamW(optimizer_parameters" in source
    assert "optim.AdamW(actor_module_fsdp.parameters()" not in source
    assert "LoRA actor original/master parameters must remain FP32" in source
    assert "LoRA actor mixed-precision forward/backward must use BF16" in source
    assert "Qwen3.5 reference model must load in BF16" in source


def test_fsdp_policy_and_offload_are_config_driven():
    method = _worker_method("_build_model_optimizer")
    source = ast.unparse(method)
    fsdp_call = _calls(method, "FSDP")[0]
    fsdp_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in fsdp_call.keywords}

    assert fsdp_keywords["use_orig_params"] == "use_orig_params"
    assert fsdp_keywords["auto_wrap_policy"] == "auto_wrap_policy"
    assert "auto_wrap_policy = None if root_only else" in source
    assert "param_offload = bool(fsdp_config.get('param_offload', False))" in source
    assert "None if role == 'actor' else CPUOffload" not in source


def test_qwen35_monkey_patch_is_rejected_and_reference_has_no_adapter_config():
    build_method = _worker_method("_build_model_optimizer")
    init_method = _worker_method("init_model")
    build_source = ast.unparse(build_method)
    init_source = ast.unparse(init_method)

    assert "Qwen3.5 reproduction does not support the Qwen2 remove-padding" in build_source
    assert "lora_config=None" in init_source
    assert "model_revision=self.config.model.get('revision')" in init_source
    assert "attention_implementation=self.config.model.get('attn_implementation', 'sdpa')" in init_source
    rollout_source = ast.unparse(_worker_method("_build_rollout"))
    assert "Qwen3.5 recurrent HF rollout requires micro_batch_size=1" in rollout_source


def test_rollout_falls_back_when_generation_config_has_no_pad_token():
    method = _worker_method("generate_sequences")
    source = ast.unparse(method)

    assert "generation_pad if generation_pad is not None else self.tokenizer.pad_token_id" in source
    assert "generation_eos if generation_eos is not None else self.tokenizer.eos_token_id" in source


def test_actor_does_not_report_zero_or_unknown_mfu_as_a_real_metric():
    method = _worker_method("update_actor")
    source = ast.unparse(method)

    assert "estimated_flops > 0" in source
    assert "promised_flops not in (0, float('inf'))" in source


def test_base_hydra_schema_exposes_reproduction_model_and_fsdp_contracts():
    source = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    for fragment in (
        "revision: null",
        "text_only: False",
        "model_init_seed: null",
        "attn_implementation: sdpa",
        "micro_batch_size: 1",
        "lora:",
        "rank: 32",
        "alpha: 64",
        "root_only: False",
        "use_orig_params: False",
    ):
        assert fragment in source


def test_driver_tokenizer_uses_revision_and_skips_processor_for_text_only_model():
    source = MAIN_PPO_PATH.read_text(encoding="utf-8")

    assert 'tokenizer_kwargs["revision"] = model_revision' in source
    assert 'model.get("text_only", False)' in source
    assert "processor = None" in source
