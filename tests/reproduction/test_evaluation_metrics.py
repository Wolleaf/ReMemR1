import pytest

from taskutils.memory_eval.reproduction_metrics import (
    EvaluationContractError,
    evaluate_output,
    extract_answer,
    normalize_answer,
    paired_bootstrap_delta,
    score_all_gold,
    token_f1,
)


def test_normalization_and_all_gold_max_do_not_use_only_first_answer():
    assert normalize_answer("  The, Eiffel-Tower! ") == "eiffeltower"

    scores = score_all_gold("Paris", ["City of Light", "Paris"])

    assert scores.exact_match == 1.0
    assert scores.exact_match_gold == "Paris"
    assert scores.token_f1 == 1.0
    assert scores.substring_exact_match == 1.0


def test_each_metric_takes_its_own_maximum_over_all_gold_answers():
    scores = score_all_gold("alpha beta", ["alpha", "alpha beta gamma"])

    assert scores.substring_exact_match == 1.0
    assert scores.token_f1 == pytest.approx(0.8)
    assert scores.token_f1_gold == "alpha beta gamma"


def test_strict_boxed_extraction_supports_nested_braces():
    evaluated = evaluate_output(
        "reasoning\n\\boxed{New {York}}",
        ["New {York}", "NYC"],
    )

    assert evaluated.extraction.answer == "New {York}"
    assert evaluated.extraction.mode == "boxed"
    assert evaluated.extraction.strict_boxed_success
    assert not evaluated.extraction.fallback_success
    assert evaluated.scores.exact_match == 1.0


def test_fallback_is_used_only_when_no_boxed_command_exists():
    explicit = extract_answer("work\nFinal answer: Mercury")
    last_line = extract_answer("work\nVenus")
    malformed = extract_answer("Answer: Mercury\n\\boxed Mercury")
    duplicate = extract_answer("\\boxed{A} and \\boxed{B}")

    assert explicit.answer == "Mercury"
    assert explicit.fallback_success
    assert last_line.answer == "Venus"
    assert last_line.fallback_success
    assert malformed.answer is None
    assert malformed.mode == "invalid_boxed"
    assert duplicate.answer is None
    assert duplicate.mode == "invalid_boxed"


def test_special_yes_no_answers_do_not_receive_partial_token_credit():
    assert token_f1("no answer", "no") == 0.0
    assert token_f1("yes", "yes") == 1.0


def test_invalid_gold_schema_fails_closed():
    with pytest.raises(EvaluationContractError, match="must not be empty"):
        score_all_gold("answer", [])
    with pytest.raises(EvaluationContractError, match=r"gold_answers\[1\]"):
        score_all_gold("answer", ["valid", ""])


def test_paired_bootstrap_is_deterministic_and_uses_paired_deltas():
    baseline = [0.0, 1.0, 0.0, 1.0]
    candidate = [1.0, 1.0, 0.0, 1.0]

    first = paired_bootstrap_delta(
        baseline,
        candidate,
        resamples=2_000,
        seed=7,
    )
    second = paired_bootstrap_delta(
        baseline,
        candidate,
        resamples=2_000,
        seed=7,
    )

    assert first == second
    assert first.observed_delta == pytest.approx(0.25)
    assert first.sample_count == 4
    assert first.ci_low <= first.observed_delta <= first.ci_high


def test_paired_bootstrap_rejects_unpaired_or_nonfinite_values():
    with pytest.raises(EvaluationContractError, match="same length"):
        paired_bootstrap_delta([0.0], [0.0, 1.0])
    with pytest.raises(EvaluationContractError, match="finite"):
        paired_bootstrap_delta([0.0], [float("nan")])
