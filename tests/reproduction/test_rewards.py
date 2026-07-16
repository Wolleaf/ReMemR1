import pytest

from recurrent.rewards import (
    callback_gain_reward,
    compute_state_reward,
    extract_prompt_payload,
    memory_gain_reward,
)


def test_memory_gain_uses_directional_recall_and_max_per_state():
    # max-over-golds gives 0.5 - 1.0; averaging the two golds would give 0.
    reward = memory_gain_reward(
        previous_memory="alpha",
        current_memory="alpha beta",
        answers=["alpha", "beta"],
    )

    assert reward == pytest.approx(-0.5)


def test_callback_gain_uses_gold_as_recall_denominator():
    reward = callback_gain_reward(
        current_memory="alpha",
        current_context="unrelated",
        recalled_memory="beta",
        answers=["alpha beta"],
    )

    assert reward == pytest.approx(0.5)


def test_prompt_payload_supports_multiline_and_rejects_duplicates():
    prompt = "<memory>first\nsecond</memory><section>chunk</section>"

    assert extract_prompt_payload(prompt, "memory") == "first\nsecond"
    assert extract_prompt_payload(prompt + "<memory>other</memory>", "memory") == ""


def test_state_reward_uses_only_parsed_update_payload():
    prompt = """<memory>alpha</memory>
<section>unrelated</section>"""
    response = """<thinking>beta should be retained</thinking>
<update>alpha beta</update>
<recall>find beta</recall>"""

    reward = compute_state_reward(
        prompt=prompt,
        response=response,
        recalled_memory="beta",
        answers=["alpha beta"],
        is_final=False,
    )

    assert reward.memory == pytest.approx(0.0)
    assert reward.callback == pytest.approx(0.0)
    assert reward.format == 1.0
    assert reward.total == 1.0


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("<update>ok</update>", 1.0),
        ("<update>ok</update><recall>query</recall>", 1.0),
        ("<update>ok</update><recall></recall>", 0.0),
        ("<update>one</update><update>two</update>", 0.0),
    ],
)
def test_intermediate_format_reward_checks_optional_recall(response, expected):
    reward = compute_state_reward(
        prompt="<memory></memory><section></section>",
        response=response,
        recalled_memory=None,
        answers=["answer"],
        is_final=False,
    )

    assert reward.format == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"The answer is \boxed{alpha}", 1.0),
        (r"\boxed{alpha} and \boxed{beta}", 0.0),
        (r"The answer is \boxed{}", 0.0),
        ("alpha", 0.0),
    ],
)
def test_final_format_reward_requires_one_nonempty_box(response, expected):
    reward = compute_state_reward(
        prompt="",
        response=response,
        recalled_memory=None,
        answers=["alpha"],
        is_final=True,
    )

    assert reward.memory == 0.0
    assert reward.callback == 0.0
    assert reward.format == expected


def test_rewards_require_at_least_one_valid_gold():
    with pytest.raises(ValueError, match="at least one"):
        memory_gain_reward("old", "new", [])

    with pytest.raises(ValueError, match="empty strings"):
        memory_gain_reward("old", "new", [""])

    with pytest.raises(TypeError, match="every answer"):
        memory_gain_reward("old", "new", [42])
