import pytest

from recurrent.protocol import (
    MemoryRecord,
    parse_final_action,
    parse_intermediate_action,
    retrieve_top1,
    top1_retrieve,
    word_recall,
)


def _record(step_id, text, chunks=(), docs=()):
    return MemoryRecord(
        step_id=step_id,
        update_text=text,
        source_chunk_ids=chunks,
        source_doc_ids=docs,
    )


def test_intermediate_parser_extracts_multiline_payloads_only():
    response = """
    <thinking>
    First line.
    Second line.
    </thinking>
    <update>
    Alice founded Acme.
    It happened in 1998.
    </update>
    <recall>
    Who acquired Acme
    after 1998?
    </recall>
    """

    action = parse_intermediate_action(response)

    assert action.is_valid
    assert action.format_valid
    assert action.thinking == "First line.\n    Second line."
    assert action.update == "Alice founded Acme.\n    It happened in 1998."
    assert action.recall == "Who acquired Acme\n    after 1998?"
    assert action.update_count == 1
    assert action.recall_count == 1


def test_intermediate_without_recall_is_valid():
    action = parse_intermediate_action("<update>useful memory</update>")

    assert action.is_valid
    assert action.update == "useful memory"
    assert action.recall is None
    assert action.recall_occurrences.is_absent


@pytest.mark.parametrize(
    "response",
    [
        "plain text with no update",
        "<update></update>",
        "<update>   \n </update>",
        "<update>memory",
        "memory</update>",
    ],
)
def test_intermediate_requires_one_complete_non_empty_update(response):
    action = parse_intermediate_action(response)

    assert not action.is_valid
    assert action.update is None


def test_empty_recall_is_invalid_and_observable():
    action = parse_intermediate_action(
        "<update>memory</update><recall> \n </recall>"
    )

    assert not action.is_valid
    assert action.recall is None
    assert action.recall_count == 1
    assert action.recall_occurrences.has_empty
    assert action.recall_occurrences.payloads == ("",)


def test_duplicate_recall_is_invalid_and_all_queries_are_observable():
    action = parse_intermediate_action(
        "<update>memory</update>"
        "<recall>first query</recall>"
        "<recall>second query</recall>"
    )

    assert not action.is_valid
    assert action.recall is None
    assert action.recall_count == 2
    assert action.has_duplicate_recall
    assert action.recall_occurrences.payloads == ("first query", "second query")


def test_duplicate_update_is_invalid_and_all_updates_are_observable():
    action = parse_intermediate_action(
        "<update>first memory</update><update>second memory</update>"
    )

    assert not action.is_valid
    assert action.update is None
    assert action.update_count == 2
    assert action.has_duplicate_update
    assert action.update_occurrences.payloads == (
        "first memory",
        "second memory",
    )


def test_unmatched_extra_tag_is_not_silently_accepted():
    action = parse_intermediate_action(
        "<update>valid-looking memory</update><update>"
    )

    assert not action.is_valid
    assert action.update is None
    assert action.update_occurrences.opening_count == 2
    assert action.update_occurrences.closing_count == 1
    assert not action.update_occurrences.is_well_formed


def test_thinking_is_recorded_but_not_required_for_format_validity():
    without_thinking = parse_intermediate_action("<update>memory</update>")
    with_thinking = parse_intermediate_action(
        "<thinking>reasoning</thinking><update>memory</update>"
    )

    assert without_thinking.thinking is None
    assert without_thinking.is_valid
    assert with_thinking.thinking == "reasoning"
    assert with_thinking.is_valid


def test_final_parser_supports_multiline_and_nested_braces():
    action = parse_final_action(
        """Reasoning first.
\\boxed{
  \\frac{Alice}{Acme}
}"""
    )

    assert action.is_valid
    assert action.boxed_answer == "\\frac{Alice}{Acme}"
    assert action.boxed_count == 1


@pytest.mark.parametrize(
    "response",
    [
        "answer without a box",
        r"\boxed{}",
        "\\boxed{  \n }",
        r"\boxed missing-braces",
        r"\boxed{missing-close",
    ],
)
def test_final_requires_one_complete_non_empty_boxed_answer(response):
    action = parse_final_action(response)

    assert not action.is_valid
    assert action.boxed_answer is None


def test_duplicate_boxed_answers_are_invalid_and_observable():
    action = parse_final_action(r"first \boxed{Alice}; second \boxed{Bob}")

    assert not action.is_valid
    assert action.boxed_answer is None
    assert action.boxed_count == 2
    assert action.has_duplicate_boxed
    assert action.boxed_occurrences.payloads == ("Alice", "Bob")


def test_valid_box_plus_malformed_box_is_still_invalid():
    action = parse_final_action(r"\boxed{Alice} and \boxed missing")

    assert not action.is_valid
    assert action.boxed_count == 2
    assert action.boxed_occurrences.malformed_count == 1


def test_word_recall_direction_uses_first_argument_as_denominator():
    assert word_recall("alpha beta beta", "beta") == pytest.approx(2 / 3)
    assert word_recall("beta", "alpha beta beta") == 1.0


def test_word_recall_normalizes_case_and_punctuation():
    assert word_recall("Alpha, BETA!", "alpha beta gamma") == 1.0
    assert word_recall("", "anything") == 0.0


def test_retrieval_calls_word_recall_with_query_first():
    records = [
        _record(1, "alpha"),
        _record(9, "alpha beta"),
    ]

    result = retrieve_top1("alpha beta gamma", records)

    assert result is not None
    assert result.record is records[1]
    assert result.score == pytest.approx(2 / 3)


def test_top1_tie_break_uses_smallest_step_id_not_input_order():
    later = _record(12, "alpha one")
    earlier = _record(3, "alpha two")

    result = retrieve_top1("alpha", [later, earlier])

    assert result is not None
    assert result.record is earlier
    assert result.score == 1.0


def test_zero_score_tie_is_also_deterministic():
    result = retrieve_top1(
        "missing",
        [_record(8, "alpha"), _record(2, "beta")],
    )

    assert result is not None
    assert result.record.step_id == 2
    assert result.score == 0.0


def test_ordered_duplicate_memories_are_not_deduplicated():
    later_duplicate = _record(7, "same memory", chunks=("c7",), docs=("d7",))
    earlier_duplicate = _record(2, "same memory", chunks=("c2",), docs=("d2",))
    history = [later_duplicate, earlier_duplicate]

    result = top1_retrieve("same memory", history)

    assert len(history) == 2
    assert history == [later_duplicate, earlier_duplicate]
    assert result is not None
    assert result.record is earlier_duplicate


def test_retrieval_returns_full_record_with_provenance():
    selected = _record(
        4,
        "the supporting fact",
        chunks=["chunk-10", "chunk-11"],
        docs=["doc-a", "doc-b"],
    )

    result = retrieve_top1("supporting fact", [selected])

    assert result is not None
    assert result.record is selected
    assert result.record.step_id == 4
    assert result.record.update_text == "the supporting fact"
    assert result.record.source_chunk_ids == ("chunk-10", "chunk-11")
    assert result.record.source_doc_ids == ("doc-a", "doc-b")


def test_empty_query_or_history_returns_no_retrieval():
    assert retrieve_top1("", [_record(1, "memory")]) is None
    assert retrieve_top1("   !!!", [_record(1, "memory")]) is None
    assert retrieve_top1("query", []) is None
