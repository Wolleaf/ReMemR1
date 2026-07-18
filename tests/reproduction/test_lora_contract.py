import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "verl" / "models" / "lora_contract.py"
SPEC = importlib.util.spec_from_file_location("lora_contract_under_test", MODULE_PATH)
lc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lc
SPEC.loader.exec_module(lc)


class FakeModule:
    def __init__(self, modules=(), parameters=()):
        self._modules = list(modules)
        self._parameters = list(parameters)

    def named_modules(self):
        yield "", self
        yield from self._modules

    def named_parameters(self):
        yield from self._parameters


class FakeLinear:
    pass


class FakeNonLinear:
    pass


class FakeParameter:
    def __init__(self, values, *, requires_grad):
        self.value = list(values)
        self.requires_grad = requires_grad
        self.shape = (len(self.value),)
        self.dtype = "fake-fp32"

    def numel(self):
        return len(self.value)


def _qwen35_text_modules():
    prefix = "model.language_model.layers"
    leaf_names = (
        "0.linear_attn.in_proj_qkv",
        "0.linear_attn.in_proj_z",
        "0.linear_attn.in_proj_b",
        "0.linear_attn.in_proj_a",
        "0.linear_attn.out_proj",
        "1.self_attn.q_proj",
        "1.self_attn.k_proj",
        "1.self_attn.v_proj",
        "1.self_attn.o_proj",
        "0.mlp.gate_proj",
        "0.mlp.up_proj",
        "0.mlp.down_proj",
    )
    return [(f"{prefix}.{leaf_name}", FakeLinear()) for leaf_name in leaf_names]


def test_resolves_gdn_full_attention_and_mlp_linear_targets():
    modules = [
        ("model", FakeNonLinear()),
        ("model.language_model", FakeNonLinear()),
        *_qwen35_text_modules(),
    ]

    manifest = lc.resolve_lora_target_manifest(
        FakeModule(modules),
        text_model_prefix="model.language_model",
    )

    assert set(manifest.target_modules) == {name for name, _ in _qwen35_text_modules()}
    assert any(name.endswith("in_proj_qkv") for name in manifest.target_modules)
    assert any(name.endswith("q_proj") for name in manifest.target_modules)
    assert any(name.endswith("gate_proj") for name in manifest.target_modules)
    assert manifest.target_modules == tuple(sorted(manifest.target_modules))


def test_excludes_embeddings_heads_vision_projector_and_mtp():
    modules = [
        ("model.language_model.layers.0.self_attn.q_proj", FakeLinear()),
        ("model.language_model.embed_tokens", FakeLinear()),
        ("lm_head", FakeLinear()),
        ("model.visual.blocks.0.self_attn.q_proj", FakeLinear()),
        ("model.vision_tower.blocks.0.mlp.up_proj", FakeLinear()),
        ("model.multi_modal_projector.q_proj", FakeLinear()),
        ("model.mtp.layers.0.self_attn.q_proj", FakeLinear()),
    ]

    manifest = lc.resolve_lora_target_manifest(FakeModule(modules))

    assert manifest.target_modules == (
        "model.language_model.layers.0.self_attn.q_proj",
    )
    assert all("visual" not in name for name in manifest.target_modules)
    assert all("mtp" not in name for name in manifest.target_modules)


def test_unknown_text_linear_fails_closed():
    model = FakeModule(
        [("model.language_model.layers.0.linear_attn.mystery_proj", FakeLinear())]
    )

    with pytest.raises(lc.LoraContractError, match="Unknown linear modules"):
        lc.resolve_lora_target_manifest(model)


def test_empty_target_match_fails():
    model = FakeModule(
        [
            ("model.language_model.embed_tokens", FakeLinear()),
            ("lm_head", FakeLinear()),
        ]
    )

    with pytest.raises(lc.LoraContractError, match="at least one"):
        lc.resolve_lora_target_manifest(model)


def test_unknown_text_model_prefix_fails():
    model = FakeModule(
        [("model.language_model.layers.0.self_attn.q_proj", FakeLinear())]
    )

    with pytest.raises(lc.LoraContractError, match="Unknown text_model_prefix"):
        lc.resolve_lora_target_manifest(model, text_model_prefix="model.text_model")


def test_explicit_manifest_rejects_denied_and_unknown_names():
    with pytest.raises(lc.LoraContractError, match="Denied modules"):
        lc.validate_lora_target_manifest(["model.visual.self_attn.q_proj"])

    with pytest.raises(lc.LoraContractError, match="Unknown linear modules"):
        lc.validate_lora_target_manifest(["model.layers.0.mystery_proj"])


def test_manifest_names_and_hash_are_deterministic():
    modules = _qwen35_text_modules()

    forward = lc.resolve_lora_target_manifest(FakeModule(modules))
    reverse = lc.resolve_lora_target_manifest(FakeModule(reversed(modules)))

    assert forward.target_modules == reverse.target_modules
    assert forward.sha256 == reverse.sha256
    assert forward.to_dict()["sha256"] == forward.sha256


class FakeLoraConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_lora_config_uses_fixed_reproduction_values_and_exact_manifest():
    manifest = lc.resolve_lora_target_manifest(FakeModule(_qwen35_text_modules()))

    config = lc.build_lora_config(
        manifest,
        lora_config_cls=FakeLoraConfig,
        task_type="CAUSAL_LM",
    )

    assert config.r == 32
    assert config.lora_alpha == 64
    assert config.lora_dropout == 0.0
    assert config.bias == "none"
    assert config.inference_mode is False
    assert config.task_type == "CAUSAL_LM"
    assert config.target_modules == list(manifest.target_modules)


def test_non_reproduction_lora_config_is_rejected():
    manifest = ["model.layers.0.self_attn.q_proj"]

    with pytest.raises(lc.LoraContractError, match="r=32"):
        lc.build_lora_config(
            manifest,
            r=16,
            lora_config_cls=FakeLoraConfig,
            task_type="CAUSAL_LM",
        )


def test_peft_injection_helper_uses_lazy_replaceable_callables():
    model = FakeModule()
    calls = []

    def fake_get_peft_model(received_model, config):
        calls.append((received_model, config))
        return "peft-model"

    result = lc.inject_lora_adapter(
        model,
        ["model.layers.0.self_attn.q_proj"],
        get_peft_model_fn=fake_get_peft_model,
        lora_config_cls=FakeLoraConfig,
        task_type="CAUSAL_LM",
    )

    assert result == "peft-model"
    assert calls[0][0] is model
    assert calls[0][1].target_modules == ["model.layers.0.self_attn.q_proj"]


def _parameter_model(reverse=False):
    parameters = [
        ("base.weight", FakeParameter([1, 2, 3, 4], requires_grad=False)),
        ("base.layer.lora_B.default.weight", FakeParameter([7, 8], requires_grad=True)),
        ("base.layer.lora_A.default.weight", FakeParameter([5, 6], requires_grad=True)),
    ]
    return FakeModule(parameters=reversed(parameters) if reverse else parameters)


def test_collects_deterministic_trainable_names_counts_and_hashes():
    forward = lc.collect_trainable_parameter_manifest(_parameter_model())
    reverse = lc.collect_trainable_parameter_manifest(_parameter_model(reverse=True))

    assert forward.names == (
        "base.layer.lora_A.default.weight",
        "base.layer.lora_B.default.weight",
    )
    assert forward.tensor_count == 2
    assert forward.trainable_numel == 4
    assert forward.total_numel == 8
    assert forward.trainable_ratio == 0.5
    assert forward.manifest_sha256 == reverse.manifest_sha256
    assert forward.state_sha256 == reverse.state_sha256
    assert forward.sha256 == forward.state_sha256


def test_only_lora_trainable_assertion_rejects_unfrozen_base():
    model = FakeModule(
        parameters=[("base.weight", FakeParameter([1], requires_grad=True))]
    )

    with pytest.raises(lc.LoraContractError, match="Non-LoRA"):
        lc.assert_only_lora_parameters_trainable(model)


def test_optimizer_parameters_include_only_trainable_tensors():
    model = _parameter_model()
    by_name = dict(model.named_parameters())

    parameters = lc.trainable_optimizer_parameters(model)

    assert parameters == (
        by_name["base.layer.lora_A.default.weight"],
        by_name["base.layer.lora_B.default.weight"],
    )
    assert all(parameter.requires_grad for parameter in parameters)
    assert by_name["base.weight"] not in parameters


def test_optimizer_contract_rejects_any_frozen_parameter():
    trainable = FakeParameter([1], requires_grad=True)
    frozen = FakeParameter([2], requires_grad=False)
    optimizer = SimpleNamespace(param_groups=[{"params": [trainable, frozen]}])

    with pytest.raises(lc.LoraContractError, match="frozen parameters"):
        lc.assert_no_frozen_optimizer_parameters(optimizer)


def test_injected_target_names_must_exactly_match_resolved_manifest():
    manifest = lc.LoraTargetManifest(
        target_modules=("model.layers.0.self_attn.q_proj",),
        sha256=lc._target_manifest_hash(("model.layers.0.self_attn.q_proj",)),
        text_model_prefix="model",
    )
    injected = SimpleNamespace(
        base_model=SimpleNamespace(
            targeted_module_names=["model.layers.1.self_attn.q_proj"]
        )
    )

    with pytest.raises(lc.LoraContractError, match="differ"):
        lc.assert_injected_lora_targets(injected, manifest)


def test_trainable_optimizer_parameters_rejects_empty_input():
    model = FakeModule(parameters=[("base.weight", FakeParameter([1], requires_grad=False))])

    with pytest.raises(lc.LoraContractError, match="must not be empty"):
        lc.trainable_optimizer_parameters(model)


def test_real_torch_peft_tiny_model_contract_without_downloads():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    nn = torch.nn

    class TinyConfig:
        model_type = "tiny-lora-contract"
        tie_word_embeddings = False

        def to_dict(self):
            return {
                "model_type": self.model_type,
                "tie_word_embeddings": self.tie_word_embeddings,
            }

    class TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)

        def forward(self, hidden_states):
            return self.q_proj(hidden_states)

    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = TinyAttention()

        def forward(self, hidden_states):
            return self.self_attn(hidden_states)

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer()])

        def forward(self, hidden_states):
            return self.layers[0](hidden_states)

    class TinyCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyBackbone()
            self.config = TinyConfig()
            self.generation_config = SimpleNamespace()
            self.is_gradient_checkpointing = False

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            del input_ids, kwargs
            return self.model(inputs_embeds)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            del args
            return kwargs

        def generate(self, *args, inputs_embeds=None, **kwargs):
            del args, kwargs
            return self.forward(inputs_embeds=inputs_embeds)

    base_model = TinyCausalLM()
    manifest = lc.resolve_lora_target_manifest(base_model)
    assert manifest.target_modules == ("model.layers.0.self_attn.q_proj",)

    peft_model = lc.inject_lora_adapter(base_model, manifest)
    assert lc.assert_injected_lora_targets(peft_model, manifest) == manifest.target_modules
    trainable = lc.assert_only_lora_parameters_trainable(peft_model)
    optimizer_parameters = lc.trainable_optimizer_parameters(peft_model)

    assert trainable.tensor_count == 2
    assert optimizer_parameters
    assert all(parameter.requires_grad for parameter in optimizer_parameters)
    assert not peft_model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight.requires_grad

    adapter_calls = []
    hooks = [
        module.register_forward_hook(lambda *unused: adapter_calls.append(1))
        for name, module in peft_model.named_modules()
        if name.endswith("lora_A.default")
    ]
    assert len(hooks) == 1

    inputs = torch.ones(1, 2, 4)
    peft_model(inputs_embeds=inputs)
    calls_after_forward = len(adapter_calls)
    peft_model.generate(inputs_embeds=inputs)

    assert calls_after_forward == 1
    assert len(adapter_calls) == 2
    for hook in hooks:
        hook.remove()


def test_real_peft_large_manifest_minimization_can_serialize_exact_targets(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")

    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=24,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    base_model = transformers.LlamaForCausalLM(config)
    manifest = lc.resolve_lora_target_manifest(base_model)
    peft_model = lc.inject_lora_adapter(base_model, manifest)
    peft_config = peft_model.peft_config["default"]

    # PEFT condenses a sufficiently large exact manifest into suffix selectors.
    assert len(manifest.target_modules) > len(peft_config.target_modules)
    assert lc.assert_injected_lora_targets(peft_model, manifest) == manifest.target_modules

    original_targets = peft_config.target_modules
    peft_config.target_modules = list(manifest.target_modules)
    try:
        peft_model.save_pretrained(tmp_path, safe_serialization=True)
    finally:
        peft_config.target_modules = original_targets

    adapter_config = json.loads(
        (tmp_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert adapter_config["target_modules"] == list(manifest.target_modules)
    assert peft_config.target_modules is original_targets

    reloaded_base = transformers.LlamaForCausalLM(config)
    reloaded = peft.PeftModel.from_pretrained(reloaded_base, tmp_path)
    assert lc.assert_injected_lora_targets(reloaded, manifest) == manifest.target_modules


def test_pinned_qwen35_meta_models_match_target_manifest_snapshots_when_cached():
    transformers = pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    pinned = {
        "Qwen/Qwen3.5-0.8B": (
            "2fc06364715b967f1860aea9cf38778875588b17",
            186,
            "58bf2b0885ddda49428a28ff775d8f41526134611c6ef569585cd27f83b76ede",
        ),
        "Qwen/Qwen3.5-2B": (
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
            186,
            "58bf2b0885ddda49428a28ff775d8f41526134611c6ef569585cd27f83b76ede",
        ),
        "Qwen/Qwen3.5-4B": (
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            248,
            "9527268b7bc9d67372be3f76e28c29f000129dfdae814b9ee3e13544c40e6e4b",
        ),
    }

    for model_id, (revision, expected_count, expected_hash) in pinned.items():
        try:
            config = transformers.AutoConfig.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=True,
            )
        except OSError:
            pytest.skip("pinned Qwen3.5 configs are not all present in the local HF cache")
        with accelerate.init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_config(
                config,
                attn_implementation="sdpa",
            )
        manifest = lc.resolve_lora_target_manifest(
            model,
            text_model_prefix="model",
        )

        assert len(manifest.target_modules) == expected_count
        assert manifest.sha256 == expected_hash
