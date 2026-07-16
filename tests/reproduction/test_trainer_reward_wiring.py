import ast
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch

from recurrent.protocol import parse_final_action, parse_intermediate_action
from recurrent.rewards import compute_state_reward


REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_UTILS_PATH = REPO_ROOT / "verl" / "trainer" / "ppo" / "metric_utils.py"
RAY_TRAINER_PATH = REPO_ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
MEMORY_AGENT_PATH = REPO_ROOT / "recurrent" / "impls" / "memory_revisit.py"


def _load_functions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {
        "DataProto": object,
        "List": List,
        "compute_state_reward": compute_state_reward,
        "parse_final_action": parse_final_action,
        "parse_intermediate_action": parse_intermediate_action,
        "torch": torch,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return tuple(namespace[name] for name in names)


class _Batch:
    def __init__(self, **tensors):
        self.batch = tensors


class _RewardBatch:
    def __init__(self, answers):
        self._items = [
            SimpleNamespace(
                non_tensor_batch={"reward_model": {"ground_truth": item_answers}}
            )
            for item_answers in answers
        ]

    def __getitem__(self, indices):
        return [self._items[int(index)] for index in indices]


def test_trainer_format_wrapper_uses_the_shared_multiline_parser():
    compute_format_rewards, _ = _load_functions(
        METRIC_UTILS_PATH,
        ("compute_format_rewards", "compute_action_rewards"),
    )
    responses = [
        "<update>line one\nline two</update><recall>query</recall>",
        "<update>ok</update><recall>one</recall><recall>two</recall>",
        r"answer \boxed{alpha}",
    ]
    batch = _Batch(
        action_type=torch.tensor([2, 2, 0]),
        responses=torch.zeros(3, 1, dtype=torch.long),
    )

    rewards = compute_format_rewards(responses, batch)

    torch.testing.assert_close(rewards.cpu(), torch.tensor([1.0, 0.0, 1.0]))


def test_trainer_action_wrapper_uses_equations_and_ignores_no_retrieval_placeholder():
    _, compute_action_rewards = _load_functions(
        METRIC_UTILS_PATH,
        ("compute_format_rewards", "compute_action_rewards"),
    )
    batch = _Batch(
        action_type=torch.tensor([2, 2]),
        responses=torch.zeros(2, 1, dtype=torch.long),
        recalled_step_ids=torch.tensor([-1, 0]),
    )
    reward_batch = _RewardBatch([["alpha", "beta"], ["alpha beta"]])

    rewards = compute_action_rewards(
        all_prompt_str=[
            "<memory>alpha</memory><section>unrelated</section>",
            "<memory>alpha</memory><section>unrelated</section>",
        ],
        all_responses_str=[
            "<update>alpha beta</update>",
            "<update>alpha</update><recall>beta</recall>",
        ],
        all_recalled_memories_str=["No memory was recalled.", "beta"],
        batch=batch,
        reward_batch=reward_batch,
        sample_index=torch.tensor([0, 1]),
        rewarded_scalar=torch.tensor([0.0, 0.0]),
    )

    torch.testing.assert_close(rewards.cpu(), torch.tensor([-0.5, 0.5]))


def test_state_advantage_does_not_duplicate_the_scalar_reward_column():
    source = RAY_TRAINER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RAY_TRAINER_PATH))
    matching_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_1D_grpo_advantage"
        and any(
            keyword.arg == "index"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "step_uid_indexed"
            for keyword in node.keywords
        )
    ]

    assert len(matching_calls) == 1
    reward_argument = next(
        keyword.value
        for keyword in matching_calls[0].keywords
        if keyword.arg == "token_level_rewards"
    )
    assert ".unsqueeze(-1)" in ast.unparse(reward_argument)
    assert ".tile(" not in ast.unparse(reward_argument)


def test_memory_agent_wires_ordered_history_and_single_index_flatten():
    source = MEMORY_AGENT_PATH.read_text(encoding="utf-8")

    assert "List[List[MemoryRecord]]" in source
    assert "MemoryRecord(" in source
    assert ".append(" in source
    assert ".nonzero().flatten()" in source
    assert ".nonzero().squeeze()" not in source
    assert "resolve_callback_query(" in source
    assert "parse_intermediate_action(response)" in source
