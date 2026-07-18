import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGER_PATH = REPO_ROOT / "recurrent" / "generation_manager.py"
RECURRENT_UTILS_PATH = REPO_ROOT / "recurrent" / "utils.py"
TRAINER_PATH = REPO_ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
REPRODUCIBILITY_PATH = REPO_ROOT / "verl" / "utils" / "reproducibility.py"


def _load_reproducibility():
    name = "_recurrent_rollout_seed_wiring_reproducibility"
    spec = importlib.util.spec_from_file_location(name, REPRODUCIBILITY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _compile_function(path, node, namespace):
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[node.name]


def _compile_isolated_manager(method, namespace):
    isolated = ast.ClassDef(
        name="IsolatedManager",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[]))
    exec(compile(module, str(MANAGER_PATH), "exec"), namespace)
    return namespace["IsolatedManager"]


class FakeProto:
    def __init__(self, batch, meta_info=None):
        self.batch = batch
        self.meta_info = meta_info or {}

    def __len__(self):
        return len(next(iter(self.batch.values())))

    @classmethod
    def from_dict(cls, tensors, meta_info=None, **kwargs):
        del kwargs
        return cls(tensors, meta_info)

    @staticmethod
    def concat(items):
        assert len(items) == 1
        return items[0]


def test_trainer_coordinates_follow_interleaved_repeat_and_resume_step():
    reproducibility = _load_reproducibility()
    node = _function(TRAINER_PATH, "_attach_recurrent_rollout_coordinates")
    attach = _compile_function(
        TRAINER_PATH,
        node,
        {
            "DataProto": FakeProto,
            "ROLLOUT_GLOBAL_STEP_KEY": reproducibility.ROLLOUT_GLOBAL_STEP_KEY,
            "make_rollout_coordinate_tensors": reproducibility.make_rollout_coordinate_tensors,
        },
    )
    proto = FakeProto({"input_ids": torch.zeros((8, 2), dtype=torch.long)})

    original_samples = attach(
        proto,
        global_step=41,
        trajectories_per_sample=4,
        sample_offset=10,
    )

    assert original_samples == 2
    assert proto.meta_info[reproducibility.ROLLOUT_GLOBAL_STEP_KEY] == 41
    assert proto.batch[reproducibility.ROLLOUT_SAMPLE_INDEX_KEY].tolist() == [
        10,
        10,
        10,
        10,
        11,
        11,
        11,
        11,
    ]
    assert proto.batch[reproducibility.ROLLOUT_TRAJECTORY_INDEX_KEY].tolist() == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]


def test_td_split_preserves_tensordict_batch_metadata():
    td_split = _compile_function(
        RECURRENT_UTILS_PATH,
        _function(RECURRENT_UTILS_PATH, "td_split"),
        {"TensorDict": TensorDict},
    )
    batch = TensorDict.from_dict(
        {
            "tokens": torch.arange(10).reshape(5, 2),
            "mask": torch.ones((5, 2), dtype=torch.bool),
        },
        batch_size=[5],
    )

    single_split = td_split(batch, 1)
    assert [split.batch_size for split in single_split] == [torch.Size([5])]
    assert [len(split) for split in single_split] == [5]
    assert torch.equal(single_split[0]["tokens"], batch["tokens"])

    splits = td_split(batch, 2)
    assert [split.batch_size for split in splits] == [torch.Size([3]), torch.Size([2])]
    assert [len(split) for split in splits] == [3, 2]
    assert torch.equal(splits[0]["tokens"], batch["tokens"][:3])
    assert torch.equal(splits[1]["tokens"], batch["tokens"][3:])

    with pytest.raises(ValueError, match="positive integer"):
        td_split(batch, 0)
    with pytest.raises(ValueError, match=r"len\(proto\)=5 < sections=6"):
        td_split(batch, 6)


def test_graceful_padding_keeps_seed_coordinates_aligned():
    reproducibility = _load_reproducibility()
    graceful_padding = _compile_function(
        RECURRENT_UTILS_PATH,
        _function(RECURRENT_UTILS_PATH, "graceful_padding"),
        {"torch": torch},
    )

    captured = {}

    class WorkerGroup:
        def generate_sequences(self, batch):
            captured.update(batch.batch)
            return batch

    def indexing_proto(proto, indices):
        return FakeProto(
            {key: value[indices] for key, value in proto.batch.items()},
            proto.meta_info,
        )

    manager_cls = _compile_isolated_manager(
        _method(
            MANAGER_PATH,
            "LLMGenerationManager",
            "generate_with_graceful_padding",
        ),
        {
            "DataProto": FakeProto,
            "indexing_proto": indexing_proto,
            "graceful_padding": graceful_padding,
            "torch": torch,
        },
    )
    manager = manager_cls()
    manager.world_size = 3
    manager.actor_rollout_wg = WorkerGroup()
    manager.get_paddings = lambda shape: (
        torch.full(shape[1:], 8, dtype=torch.long),
        torch.zeros(shape[1:], dtype=torch.long),
        torch.zeros(shape[1:], dtype=torch.long),
    )
    input_ids = torch.arange(14, dtype=torch.long).reshape(7, 2)
    coordinates = {
        reproducibility.ROLLOUT_SAMPLE_INDEX_KEY: torch.arange(10, 17),
        reproducibility.ROLLOUT_TRAJECTORY_INDEX_KEY: torch.arange(7),
        reproducibility.ROLLOUT_TURN_INDEX_KEY: torch.full((7,), 2),
    }

    output = manager.generate_with_graceful_padding(
        input_ids,
        torch.ones_like(input_ids),
        torch.zeros_like(input_ids),
        {reproducibility.ROLLOUT_GLOBAL_STEP_KEY: 4},
        rollout_coordinates=coordinates,
    )

    assert captured[reproducibility.ROLLOUT_SAMPLE_INDEX_KEY].tolist() == [
        10,
        11,
        12,
        0,
        13,
        14,
        0,
        15,
        16,
    ]
    assert captured[reproducibility.ROLLOUT_TURN_INDEX_KEY].tolist() == [
        2,
        2,
        2,
        0,
        2,
        2,
        0,
        2,
        2,
    ]
    assert output.batch[reproducibility.ROLLOUT_SAMPLE_INDEX_KEY].tolist() == list(
        range(10, 17)
    )


def test_manager_selects_active_coordinates_and_adds_recurrent_turn():
    reproducibility = _load_reproducibility()
    captured = {}

    @contextmanager
    def timer(name, timing_raw):
        del name, timing_raw
        yield

    class Agent:
        step = 2
        finished = False

        def start(self, gen_batch, timing_raw):
            del gen_batch, timing_raw
            self.sample_index_list = []

        def done(self):
            return self.finished

        def action(self):
            self.sample_index_list.append(torch.tensor([2, 0, 1]))
            return [torch.tensor([1, 2])] * 3, {"input_pad_to": 2}

        def update(self, output):
            output.batch["recalled_memories"] = np.asarray([None] * 3, dtype=object)
            self.finished = True
            return output

        def end(self):
            return torch.ones(3, dtype=torch.bool), torch.tensor([2, 0, 1])

    manager_cls = _compile_isolated_manager(
        _method(MANAGER_PATH, "LLMGenerationManager", "run_llm_loop_revisit"),
        {
            "DataProto": FakeProto,
            "Dict": Dict,
            "Tuple": Tuple,
            "ROLLOUT_GLOBAL_STEP_KEY": reproducibility.ROLLOUT_GLOBAL_STEP_KEY,
            "ROLLOUT_SAMPLE_INDEX_KEY": reproducibility.ROLLOUT_SAMPLE_INDEX_KEY,
            "ROLLOUT_TRAJECTORY_INDEX_KEY": reproducibility.ROLLOUT_TRAJECTORY_INDEX_KEY,
            "ROLLOUT_TURN_INDEX_KEY": reproducibility.ROLLOUT_TURN_INDEX_KEY,
            "_timer": timer,
            "create_attention_mask": lambda ids, pad_token_id: (ids != pad_token_id).long(),
            "create_position_ids": lambda mask: torch.cumsum(mask, dim=1) - 1,
            "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
            "np": np,
            "pad_tensor_list_to_length": lambda messages, **kwargs: torch.stack(messages),
            "torch": torch,
        },
    )
    manager = manager_cls()
    manager.tokenizer = SimpleNamespace(pad_token_id=0)
    manager.agent = Agent()

    def generate(*args, rollout_coordinates, **kwargs):
        del args, kwargs
        captured.update(rollout_coordinates)
        return FakeProto({"responses": torch.ones((3, 2), dtype=torch.long)})

    manager.generate_with_graceful_padding = generate
    gen_batch = FakeProto(
        {
            reproducibility.ROLLOUT_SAMPLE_INDEX_KEY: torch.tensor([10, 11, 12]),
            reproducibility.ROLLOUT_TRAJECTORY_INDEX_KEY: torch.tensor([0, 1, 2]),
        },
        {reproducibility.ROLLOUT_GLOBAL_STEP_KEY: 5},
    )

    manager.run_llm_loop_revisit(gen_batch, {})

    assert captured[reproducibility.ROLLOUT_SAMPLE_INDEX_KEY].tolist() == [12, 10, 11]
    assert captured[reproducibility.ROLLOUT_TRAJECTORY_INDEX_KEY].tolist() == [2, 0, 1]
    assert captured[reproducibility.ROLLOUT_TURN_INDEX_KEY].tolist() == [2, 2, 2]
