import pytest

from recurrent.protocol import (
    MemoryRecord,
    parse_intermediate_action,
    resolve_callback_query,
    retrieve_top1,
)


@pytest.mark.parametrize(
    ("mode", "learned", "question", "expected"),
    [
        ("learned", "  beta evidence  ", "where is alpha?", "beta evidence"),
        ("learned", None, "where is alpha?", None),
        ("none", "beta evidence", "where is alpha?", None),
        ("fixed_question", "beta evidence", "  where is alpha?  ", "where is alpha?"),
    ],
)
def test_callback_mode_resolves_the_actual_retrieval_query(
    mode, learned, question, expected
):
    assert (
        resolve_callback_query(
            mode,
            learned_query=learned,
            question=question,
        )
        == expected
    )


def test_none_mode_disables_retrieval_instead_of_only_hiding_output():
    records = [MemoryRecord(step_id=0, update_text="alpha evidence")]
    query = resolve_callback_query(
        "none",
        learned_query="alpha",
        question="alpha question",
    )

    result = retrieve_top1(query, records) if query is not None else None

    assert result is None


def test_fixed_question_and_learned_modes_can_select_different_memories():
    records = [
        MemoryRecord(step_id=0, update_text="alpha evidence"),
        MemoryRecord(step_id=1, update_text="beta evidence"),
    ]
    learned = resolve_callback_query(
        "learned",
        learned_query="beta",
        question="alpha",
    )
    fixed = resolve_callback_query(
        "fixed_question",
        learned_query="beta",
        question="alpha",
    )

    assert retrieve_top1(learned, records).record.step_id == 1
    assert retrieve_top1(fixed, records).record.step_id == 0


def test_duplicate_learned_recall_is_not_silently_selected():
    action = parse_intermediate_action(
        "<update>memory</update><recall>one</recall><recall>two</recall>"
    )

    assert action.recall is None
    assert (
        resolve_callback_query(
            "learned",
            learned_query=action.recall,
            question="question",
        )
        is None
    )


def test_unknown_callback_mode_fails_closed():
    with pytest.raises(ValueError, match="callback mode"):
        resolve_callback_query("typo", learned_query="query", question="question")
