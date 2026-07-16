import contextlib
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class FakeTensorDict(dict):
    def __init__(self, source=None, batch_size=None):
        super().__init__(source or {})
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        self.batch_size = torch.Size(batch_size)


class FakeDataProto:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __getitem__(self, item):
        tensors = {key: value[item] for key, value in self.batch.items()}
        batch_size = next(iter(tensors.values())).shape[0]
        batch = FakeTensorDict(tensors, batch_size=batch_size)
        return FakeDataProto(batch=batch, meta_info=self.meta_info)

    @staticmethod
    def concat(items):
        tensors = {
            key: torch.cat([item.batch[key] for item in items], dim=0)
            for key in items[0].batch
        }
        batch_size = next(iter(tensors.values())).shape[0]
        return FakeDataProto(
            batch=FakeTensorDict(tensors, batch_size=batch_size),
            meta_info=items[0].meta_info,
        )


class FakeGenerationConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.adapter_enabled = True
        self.inside_summon = False
        self.emit_eos = True
        self.sample_from_rng = False

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        eos_token_id,
        pad_token_id,
        generation_config,
        do_sample,
        **kwargs,
    ):
        del attention_mask, pad_token_id, kwargs
        n = generation_config.num_return_sequences
        self.calls.append(
            {
                "batch_size": input_ids.shape[0],
                "max_new_tokens": max_new_tokens,
                "n": n,
                "temperature": getattr(generation_config, "temperature", None),
                "training": self.training,
                "adapter_enabled": self.adapter_enabled,
                "inside_summon": self.inside_summon,
            }
        )

        prompts = input_ids.repeat_interleave(n, dim=0)
        generated_length = min(max_new_tokens, 2)
        if self.sample_from_rng and do_sample:
            generated_length = max_new_tokens
            weights = torch.ones(
                (prompts.shape[0] * generated_length, 997),
                device=input_ids.device,
            )
            generated = torch.multinomial(weights, num_samples=1).reshape(
                prompts.shape[0],
                generated_length,
            ) + 10
            generated = generated.to(input_ids.dtype)
        else:
            generated = torch.full(
                (prompts.shape[0], generated_length),
                5,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        if self.emit_eos:
            generated[:, -1] = eos_token_id
        return types.SimpleNamespace(sequences=torch.cat([prompts, generated], dim=1))


@pytest.fixture
def hf_module(monkeypatch):
    tensordict_module = types.ModuleType("tensordict")
    tensordict_module.TensorDict = FakeTensorDict
    transformers_module = types.ModuleType("transformers")
    transformers_module.GenerationConfig = FakeGenerationConfig

    verl_module = types.ModuleType("verl")
    verl_module.__path__ = []
    verl_module.DataProto = FakeDataProto
    workers_module = types.ModuleType("verl.workers")
    workers_module.__path__ = []
    rollout_package = types.ModuleType("verl.workers.rollout")
    rollout_package.__path__ = []
    utils_package = types.ModuleType("verl.utils")
    utils_package.__path__ = []

    base_module = types.ModuleType("verl.workers.rollout.base")
    base_module.BaseRollout = object
    torch_functional_module = types.ModuleType("verl.utils.torch_functional")

    def get_response_mask(response_id, eos_token, dtype):
        eos = torch.as_tensor(eos_token, device=response_id.device)
        eos_mask = torch.isin(response_id, eos).int()
        return (eos_mask.cumsum(dim=1) - eos_mask).eq(0).to(dtype)

    torch_functional_module.get_response_mask = get_response_mask

    reproducibility_name = "_hf_rollout_reproducibility_contract_test"
    reproducibility_source = (
        Path(__file__).parents[2] / "verl" / "utils" / "reproducibility.py"
    )
    reproducibility_spec = importlib.util.spec_from_file_location(
        reproducibility_name,
        reproducibility_source,
    )
    reproducibility_module = importlib.util.module_from_spec(reproducibility_spec)
    monkeypatch.setitem(sys.modules, reproducibility_name, reproducibility_module)
    reproducibility_spec.loader.exec_module(reproducibility_module)

    stubs = {
        "tensordict": tensordict_module,
        "transformers": transformers_module,
        "verl": verl_module,
        "verl.workers": workers_module,
        "verl.workers.rollout": rollout_package,
        "verl.workers.rollout.base": base_module,
        "verl.utils": utils_package,
        "verl.utils.reproducibility": reproducibility_module,
        "verl.utils.torch_functional": torch_functional_module,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "verl.workers.rollout._hf_rollout_contract_test"
    source = Path(__file__).parents[2] / "verl" / "workers" / "rollout" / "hf_rollout.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def make_config(**overrides):
    config = AttrDict(
        micro_batch_size=2,
        do_sample=True,
        temperature=1.0,
        response_length=7,
        top_p=1.0,
        top_k=0,
        n=8,
        val_kwargs=AttrDict(top_k=0, top_p=1.0, temperature=1.0),
    )
    config.update(overrides)
    return config


def make_prompts(module, batch_size, prompt_length=3):
    input_ids = torch.arange(
        10,
        10 + batch_size * prompt_length,
        dtype=torch.long,
    ).reshape(batch_size, prompt_length)
    tensors = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(prompt_length).repeat(batch_size, 1),
    }
    return module.DataProto(
        batch=FakeTensorDict(tensors, batch_size=batch_size),
        meta_info={"eos_token_id": 9, "pad_token_id": 0},
    )


def add_seed_coordinates(
    module,
    prompts,
    *,
    global_step=1,
    sample_index=0,
    trajectory_index=0,
    turn_index=0,
):
    prompts.meta_info[module.ROLLOUT_GLOBAL_STEP_KEY] = global_step
    prompts.batch[module.ROLLOUT_SAMPLE_INDEX_KEY] = torch.full(
        (len(prompts.batch["input_ids"]),),
        sample_index,
        dtype=torch.long,
    )
    prompts.batch[module.ROLLOUT_TRAJECTORY_INDEX_KEY] = torch.full(
        (len(prompts.batch["input_ids"]),),
        trajectory_index,
        dtype=torch.long,
    )
    prompts.batch[module.ROLLOUT_TURN_INDEX_KEY] = torch.full(
        (len(prompts.batch["input_ids"]),),
        turn_index,
        dtype=torch.long,
    )
    return prompts


def test_call_overrides_and_cpu_shapes(hf_module, monkeypatch):
    model = FakeModel()
    rollout = hf_module.HFRollout(model, make_config(micro_batch_size=4))
    prompts = make_prompts(hf_module, batch_size=1)

    def unexpected_cuda_call(*args, **kwargs):
        raise AssertionError("CPU rollout must not touch CUDA autocast/cache")

    monkeypatch.setattr(hf_module.torch, "autocast", unexpected_cuda_call)
    monkeypatch.setattr(hf_module.torch.cuda, "empty_cache", unexpected_cuda_call)

    result = rollout.generate_sequences(
        prompts,
        pad_to=5,
        max_tokens=3,
        n=1,
        temperature=0.25,
    )

    assert model.calls == [
        {
            "batch_size": 1,
            "max_new_tokens": 3,
            "n": 1,
            "temperature": 0.25,
            "training": False,
            "adapter_enabled": True,
            "inside_summon": False,
        }
    ]
    assert model.training
    assert result.batch["responses"].shape == (1, 5)
    assert result.batch["input_ids"].shape == (1, 8)
    assert result.batch["attention_mask"].shape == (1, 8)
    assert result.batch["position_ids"].shape == (1, 8)
    assert result.batch["responses"].tolist() == [[5, 9, 0, 0, 0]]
    assert result.batch["attention_mask"][0, -5:].tolist() == [1, 1, 0, 0, 0]


def test_non_divisible_batch_never_exceeds_micro_batch(hf_module):
    model = FakeModel()
    rollout = hf_module.HFRollout(model, make_config(micro_batch_size=2))

    result = rollout.generate_sequences(
        make_prompts(hf_module, batch_size=5),
        pad_to=2,
        max_tokens=2,
        n=1,
    )

    assert [call["batch_size"] for call in model.calls] == [2, 2, 1]
    assert max(call["batch_size"] for call in model.calls) <= 2
    assert result.batch.batch_size == torch.Size([5])
    assert result.batch["responses"].shape == (5, 2)


def test_padding_after_max_tokens_is_masked_without_eos(hf_module):
    model = FakeModel()
    model.emit_eos = False
    rollout = hf_module.HFRollout(model, make_config())

    result = rollout.generate_sequences(
        make_prompts(hf_module, batch_size=1),
        pad_to=4,
        max_tokens=2,
        n=1,
    )

    assert result.batch["responses"].tolist() == [[5, 5, 0, 0]]
    assert result.batch["attention_mask"][0, -4:].tolist() == [1, 1, 0, 0]


def test_call_level_n_expands_once_across_micro_batches(hf_module):
    model = FakeModel()
    rollout = hf_module.HFRollout(model, make_config(micro_batch_size=2, n=1))
    prompts = make_prompts(hf_module, batch_size=3)

    result = rollout.generate_sequences(prompts, pad_to=4, max_tokens=2, n=8)

    assert [(call["batch_size"], call["n"]) for call in model.calls] == [(2, 8), (1, 8)]
    assert result.batch.batch_size == torch.Size([24])
    assert result.batch["responses"].shape == (24, 4)
    expected_first_prompt = prompts.batch["input_ids"][0].repeat(8, 1)
    torch.testing.assert_close(result.batch["prompts"][:8], expected_first_prompt)


def test_pad_to_cannot_be_shorter_than_max_tokens(hf_module):
    model = FakeModel()
    rollout = hf_module.HFRollout(model, make_config())

    with pytest.raises(ValueError, match="pad_to .* max_tokens"):
        rollout.generate_sequences(
            make_prompts(hf_module, batch_size=1),
            pad_to=3,
            max_tokens=4,
            n=1,
        )

    assert model.calls == []


def test_fsdp_summon_context_keeps_adapter_enabled(hf_module, monkeypatch):
    class FakeFSDP(FakeModel):
        @staticmethod
        @contextlib.contextmanager
        def summon_full_params(module, writeback, recurse):
            assert writeback is False
            assert recurse is False
            assert module.adapter_enabled
            module.inside_summon = True
            try:
                yield
            finally:
                module.inside_summon = False

    monkeypatch.setattr(hf_module, "FSDP", FakeFSDP)
    model = FakeFSDP()
    rollout = hf_module.HFRollout(model, make_config())

    rollout.generate_sequences(
        make_prompts(hf_module, batch_size=1),
        pad_to=2,
        max_tokens=2,
        n=1,
    )

    assert model.calls[0]["inside_summon"]
    assert model.calls[0]["adapter_enabled"]
    assert model.adapter_enabled


def test_seeded_generate_uses_private_generator_and_matches_bc_token_hash(hf_module):
    config = make_config(micro_batch_size=1, seed=2924589348)
    model_b = FakeModel()
    model_b.sample_from_rng = True
    model_b.emit_eos = False
    prompts_b = add_seed_coordinates(
        hf_module,
        make_prompts(hf_module, batch_size=1),
        global_step=7,
        sample_index=2,
        trajectory_index=5,
        turn_index=1,
    )

    torch.manual_seed(1234)
    global_state = torch.random.get_rng_state().clone()
    output_b = hf_module.HFRollout(model_b, config).generate_sequences(
        prompts_b,
        pad_to=16,
        max_tokens=16,
        n=1,
    )
    assert torch.equal(torch.random.get_rng_state(), global_state)

    model_c = FakeModel()
    model_c.sample_from_rng = True
    model_c.emit_eos = False
    prompts_c = add_seed_coordinates(
        hf_module,
        make_prompts(hf_module, batch_size=1),
        global_step=7,
        sample_index=2,
        trajectory_index=5,
        turn_index=1,
    )
    torch.manual_seed(9999)
    output_c = hf_module.HFRollout(model_c, config).generate_sequences(
        prompts_c,
        pad_to=16,
        max_tokens=16,
        n=1,
    )

    token_hash_b = hashlib.sha256(
        output_b.batch["responses"].numpy().tobytes()
    ).hexdigest()
    token_hash_c = hashlib.sha256(
        output_c.batch["responses"].numpy().tobytes()
    ).hexdigest()
    assert token_hash_b == token_hash_c


def test_different_rollout_coordinates_use_different_rng_streams(hf_module):
    config = make_config(micro_batch_size=1, seed=2924589348)

    def generate_for(*, trajectory_index, turn_index):
        model = FakeModel()
        model.sample_from_rng = True
        model.emit_eos = False
        prompts = add_seed_coordinates(
            hf_module,
            make_prompts(hf_module, batch_size=1),
            global_step=7,
            sample_index=2,
            trajectory_index=trajectory_index,
            turn_index=turn_index,
        )
        return hf_module.HFRollout(model, config).generate_sequences(
            prompts,
            pad_to=16,
            max_tokens=16,
            n=1,
        ).batch["responses"]

    trajectory_tokens = generate_for(trajectory_index=6, turn_index=1)
    next_turn_tokens = generate_for(trajectory_index=5, turn_index=2)
    baseline_tokens = generate_for(trajectory_index=5, turn_index=1)

    assert not torch.equal(trajectory_tokens, baseline_tokens)
    assert not torch.equal(next_turn_tokens, baseline_tokens)
    assert hf_module.derive_rollout_generation_seed(
        config.seed, 7, 2, 6, 1
    ) != hf_module.derive_rollout_generation_seed(config.seed, 7, 2, 5, 1)


def test_seed_contract_fails_closed_without_single_prompt_microbatch(hf_module):
    model = FakeModel()
    prompts = add_seed_coordinates(
        hf_module,
        make_prompts(hf_module, batch_size=2),
    )
    rollout = hf_module.HFRollout(
        model,
        make_config(micro_batch_size=2, seed=2924589348),
    )

    with pytest.raises(ValueError, match="micro_batch_size=1"):
        rollout.generate_sequences(prompts, max_tokens=2, n=1)

    assert model.calls == []


def test_seeded_greedy_generation_does_not_consume_rng(hf_module):
    model = FakeModel()
    model.sample_from_rng = True
    prompts = add_seed_coordinates(
        hf_module,
        make_prompts(hf_module, batch_size=1),
    )
    rollout = hf_module.HFRollout(
        model,
        make_config(micro_batch_size=1, seed=2924589348),
    )

    torch.manual_seed(314)
    global_state = torch.random.get_rng_state().clone()
    result = rollout.generate_sequences(
        prompts,
        max_tokens=2,
        n=8,
        do_sample=False,
    )

    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert result.batch.batch_size == torch.Size([1])
