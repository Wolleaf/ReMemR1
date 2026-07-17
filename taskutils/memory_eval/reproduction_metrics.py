"""Deterministic answer extraction and scoring for reproduction evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import random
import re
import string
from statistics import fmean
from typing import Iterable, Sequence

from recurrent.protocol import parse_final_action


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_FALLBACK_ANSWER = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*(?P<answer>[^\n]+)",
    flags=re.IGNORECASE,
)
_SPECIAL_ANSWERS = frozenset({"yes", "no", "noanswer"})


class EvaluationContractError(ValueError):
    """Raised when evaluation input violates the fixed scoring contract."""


@dataclass(frozen=True, slots=True)
class ExtractedAnswer:
    answer: str | None
    mode: str
    strict_boxed_success: bool
    fallback_success: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnswerScores:
    exact_match: float
    token_f1: float
    substring_exact_match: float
    exact_match_gold: str | None
    token_f1_gold: str | None
    substring_exact_match_gold: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluatedAnswer:
    extraction: ExtractedAnswer
    scores: AnswerScores

    def to_dict(self) -> dict[str, object]:
        return {
            "extraction": self.extraction.to_dict(),
            "scores": self.scores.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    sample_count: int
    resamples: int
    seed: int
    observed_delta: float
    confidence_level: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StratifiedPairedBootstrapResult:
    stratum_count: int
    sample_counts: tuple[int, ...]
    resamples: int
    seed: int
    observed_delta: float
    confidence_level: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["sample_counts"] = list(self.sample_counts)
        return value


def normalize_answer(text: str) -> str:
    """Apply the SQuAD-style normalization used by EM and token F1."""

    if not isinstance(text, str):
        raise TypeError(f"answer must be str, got {type(text).__name__}")
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return _WHITESPACE.sub(" ", without_articles).strip()


def extract_answer(raw_output: str) -> ExtractedAnswer:
    """Extract one answer while keeping strict boxed and fallback rates separate."""

    if not isinstance(raw_output, str):
        raise TypeError(f"raw_output must be str, got {type(raw_output).__name__}")
    parsed = parse_final_action(raw_output)
    if parsed.boxed_answer is not None:
        return ExtractedAnswer(
            answer=parsed.boxed_answer,
            mode="boxed",
            strict_boxed_success=True,
            fallback_success=False,
        )

    # A malformed or duplicate boxed command is an observable format failure,
    # not permission to recover a convenient answer from the surrounding text.
    if not parsed.boxed_occurrences.is_absent:
        return ExtractedAnswer(
            answer=None,
            mode="invalid_boxed",
            strict_boxed_success=False,
            fallback_success=False,
        )

    matches = tuple(_FALLBACK_ANSWER.finditer(raw_output))
    if matches:
        fallback = matches[-1].group("answer").strip()
    else:
        nonempty_lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        fallback = nonempty_lines[-1] if nonempty_lines else ""
    if not fallback:
        return ExtractedAnswer(
            answer=None,
            mode="missing",
            strict_boxed_success=False,
            fallback_success=False,
        )
    return ExtractedAnswer(
        answer=fallback,
        mode="fallback",
        strict_boxed_success=False,
        fallback_success=True,
    )


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def substring_exact_match(prediction: str, gold: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    if not normalized_prediction or not normalized_gold:
        return 0.0
    return float(
        normalized_gold in normalized_prediction
        or normalized_prediction in normalized_gold
    )


def token_f1(prediction: str, gold: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    if (
        normalized_prediction in _SPECIAL_ANSWERS
        or normalized_gold in _SPECIAL_ANSWERS
    ) and normalized_prediction != normalized_gold:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    if not prediction_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _validate_gold_answers(gold_answers: Sequence[str]) -> tuple[str, ...]:
    if isinstance(gold_answers, (str, bytes)) or not isinstance(gold_answers, Sequence):
        raise TypeError("gold_answers must be a sequence of strings")
    normalized: list[str] = []
    for index, answer in enumerate(gold_answers):
        if not isinstance(answer, str) or not answer.strip():
            raise EvaluationContractError(
                f"gold_answers[{index}] must be a non-empty string"
            )
        normalized.append(answer)
    if not normalized:
        raise EvaluationContractError("gold_answers must not be empty")
    return tuple(normalized)


def score_all_gold(prediction: str | None, gold_answers: Sequence[str]) -> AnswerScores:
    """Score every legal gold and take a separate maximum for each metric."""

    golds = _validate_gold_answers(gold_answers)
    if prediction is None:
        return AnswerScores(0.0, 0.0, 0.0, None, None, None)
    if not isinstance(prediction, str):
        raise TypeError(f"prediction must be str or None, got {type(prediction).__name__}")

    exact_values = tuple((exact_match(prediction, gold), gold) for gold in golds)
    f1_values = tuple((token_f1(prediction, gold), gold) for gold in golds)
    substring_values = tuple(
        (substring_exact_match(prediction, gold), gold) for gold in golds
    )
    best_exact = max(exact_values, key=lambda item: item[0])
    best_f1 = max(f1_values, key=lambda item: item[0])
    best_substring = max(substring_values, key=lambda item: item[0])
    return AnswerScores(
        exact_match=best_exact[0],
        token_f1=best_f1[0],
        substring_exact_match=best_substring[0],
        exact_match_gold=best_exact[1],
        token_f1_gold=best_f1[1],
        substring_exact_match_gold=best_substring[1],
    )


def evaluate_output(raw_output: str, gold_answers: Sequence[str]) -> EvaluatedAnswer:
    extraction = extract_answer(raw_output)
    return EvaluatedAnswer(
        extraction=extraction,
        scores=score_all_gold(extraction.answer, gold_answers),
    )


def _validate_metric_values(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(values)
    if not result:
        raise EvaluationContractError(f"{name} must not be empty")
    for index, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{index}] must be numeric")
        if not math.isfinite(value):
            raise EvaluationContractError(f"{name}[{index}] must be finite")
    return tuple(float(value) for value in result)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_delta(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Return a deterministic percentile CI for candidate minus baseline."""

    baseline_values = _validate_metric_values(baseline, name="baseline")
    candidate_values = _validate_metric_values(candidate, name="candidate")
    if len(baseline_values) != len(candidate_values):
        raise EvaluationContractError("paired metrics must have the same length")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise EvaluationContractError("resamples must be a positive int")
    if not isinstance(confidence_level, (int, float)) or not 0 < confidence_level < 1:
        raise EvaluationContractError("confidence_level must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")

    deltas = tuple(
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(baseline_values, candidate_values)
    )
    rng = random.Random(seed)
    count = len(deltas)
    bootstrap = sorted(
        fmean(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    tail = (1.0 - float(confidence_level)) / 2.0
    return PairedBootstrapResult(
        sample_count=count,
        resamples=resamples,
        seed=seed,
        observed_delta=fmean(deltas),
        confidence_level=float(confidence_level),
        ci_low=_quantile(bootstrap, tail),
        ci_high=_quantile(bootstrap, 1.0 - tail),
    )


def stratified_paired_bootstrap_delta(
    strata: Sequence[tuple[Iterable[float], Iterable[float]]],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> StratifiedPairedBootstrapResult:
    """Bootstrap paired deltas within strata, then macro-average strata equally."""

    if isinstance(strata, (str, bytes)) or not isinstance(strata, Sequence):
        raise TypeError("strata must be a sequence of baseline/candidate pairs")
    validated: list[tuple[float, ...]] = []
    sample_counts: list[int] = []
    for index, pair in enumerate(strata):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise EvaluationContractError(
                f"strata[{index}] must contain baseline and candidate metrics"
            )
        baseline = _validate_metric_values(pair[0], name=f"strata[{index}].baseline")
        candidate = _validate_metric_values(pair[1], name=f"strata[{index}].candidate")
        if len(baseline) != len(candidate):
            raise EvaluationContractError(
                f"strata[{index}] paired metrics must have the same length"
            )
        validated.append(
            tuple(
                candidate_value - baseline_value
                for baseline_value, candidate_value in zip(baseline, candidate)
            )
        )
        sample_counts.append(len(baseline))
    if not validated:
        raise EvaluationContractError("strata must not be empty")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise EvaluationContractError("resamples must be a positive int")
    if not isinstance(confidence_level, (int, float)) or not 0 < confidence_level < 1:
        raise EvaluationContractError("confidence_level must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")

    observed = fmean(fmean(deltas) for deltas in validated)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(resamples):
        stratum_means = []
        for deltas in validated:
            count = len(deltas)
            stratum_means.append(
                fmean(deltas[rng.randrange(count)] for _ in range(count))
            )
        bootstrap.append(fmean(stratum_means))
    bootstrap.sort()
    tail = (1.0 - float(confidence_level)) / 2.0
    return StratifiedPairedBootstrapResult(
        stratum_count=len(validated),
        sample_counts=tuple(sample_counts),
        resamples=resamples,
        seed=seed,
        observed_delta=observed,
        confidence_level=float(confidence_level),
        ci_low=_quantile(bootstrap, tail),
        ci_high=_quantile(bootstrap, 1.0 - tail),
    )


__all__ = [
    "AnswerScores",
    "EvaluatedAnswer",
    "EvaluationContractError",
    "ExtractedAnswer",
    "PairedBootstrapResult",
    "StratifiedPairedBootstrapResult",
    "evaluate_output",
    "exact_match",
    "extract_answer",
    "normalize_answer",
    "paired_bootstrap_delta",
    "score_all_gold",
    "stratified_paired_bootstrap_delta",
    "substring_exact_match",
    "token_f1",
]
