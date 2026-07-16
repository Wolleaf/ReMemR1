import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl"
    / "utils"
    / "reproduction_fingerprint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reproduction_fingerprint_under_test",
    MODULE_PATH,
)
fingerprint_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fingerprint_module
SPEC.loader.exec_module(fingerprint_module)

FingerprintContractError = fingerprint_module.FingerprintContractError
StepZeroFingerprint = fingerprint_module.StepZeroFingerprint
hash_ordered_sample_ids = fingerprint_module.hash_ordered_sample_ids
hash_sampled_token_rows = fingerprint_module.hash_sampled_token_rows
publish_step_zero_fingerprint = fingerprint_module.publish_step_zero_fingerprint


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def make_fingerprint(**overrides):
    values = {
        "run_seed": 42,
        "rollout_global_step": 1,
        "base_model_id": "Qwen/Qwen3.5-4B",
        "base_model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "data_manifest_sha256": SHA_A,
        "initial_adapter_sha256": SHA_B,
        "first_batch_sha256": SHA_C,
        "first_sampled_tokens_sha256": SHA_D,
    }
    values.update(overrides)
    return StepZeroFingerprint(**values)


def test_ordered_batch_and_sampled_token_hashes_are_stable_and_sensitive():
    first_batch = hash_ordered_sample_ids(["qa-1", "qa-2", 3])
    same_batch = hash_ordered_sample_ids(["qa-1", "qa-2", 3])
    reordered_batch = hash_ordered_sample_ids(["qa-2", "qa-1", 3])

    assert first_batch == same_batch
    assert first_batch != reordered_batch
    assert hash_sampled_token_rows([[1, 2, 0], [3, 4, 5]]) != hash_sampled_token_rows(
        [[1, 2, 0], [3, 5, 4]]
    )


def test_fingerprint_round_trip_is_self_hashed_and_atomic(tmp_path):
    fingerprint = make_fingerprint()
    path = fingerprint.save(tmp_path / "step0_fingerprint.json")

    assert StepZeroFingerprint.load(path) == fingerprint
    assert json.loads(path.read_text(encoding="utf-8"))["fingerprint_sha256"] == fingerprint.sha256
    assert not tuple(tmp_path.glob(".step0_fingerprint.json.tmp-*"))


def test_fingerprint_paths_are_absolute_and_existing_evidence_is_never_replaced(
    tmp_path,
):
    with pytest.raises(FingerprintContractError, match="absolute"):
        make_fingerprint().save(Path("relative-step-zero.json"))

    path = make_fingerprint().save(tmp_path / "step0_fingerprint.json")
    original = path.read_bytes()
    with pytest.raises(FingerprintContractError, match="already exists"):
        make_fingerprint(run_seed=7).save(path)

    assert path.read_bytes() == original


def test_fingerprint_tampering_and_unknown_fields_fail_closed(tmp_path):
    path = make_fingerprint().save(tmp_path / "step0_fingerprint.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_seed"] = 7
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FingerprintContractError, match="fingerprint_sha256"):
        StepZeroFingerprint.load(path)

    payload = make_fingerprint().to_dict()
    payload["unexpected"] = True
    with pytest.raises(FingerprintContractError, match="keys"):
        StepZeroFingerprint.from_dict(payload)

    payload = make_fingerprint().to_dict()
    payload["schema_version"] = True
    with pytest.raises(FingerprintContractError, match="schema"):
        StepZeroFingerprint.from_dict(payload)


def test_duplicate_json_keys_and_floating_revision_fail_closed(tmp_path):
    path = tmp_path / "duplicate.json"
    payload = make_fingerprint().to_dict()
    serialized = json.dumps(payload)
    path.write_text(serialized.replace('"run_seed": 42', '"run_seed": 7, "run_seed": 42'), encoding="utf-8")

    with pytest.raises(FingerprintContractError, match="duplicate JSON key"):
        StepZeroFingerprint.load(path)
    with pytest.raises(FingerprintContractError, match="40-character"):
        make_fingerprint(base_model_revision="main")


def test_bc_fingerprint_comparison_reports_every_divergent_contract_field():
    reference = make_fingerprint()
    candidate = make_fingerprint(
        initial_adapter_sha256="e" * 64,
        first_sampled_tokens_sha256="f" * 64,
    )

    with pytest.raises(FingerprintContractError) as error:
        candidate.assert_matches(reference)

    assert "initial_adapter_sha256" in str(error.value)
    assert "first_sampled_tokens_sha256" in str(error.value)


def test_c_mismatch_is_rejected_before_its_evidence_is_published(tmp_path):
    reference_path = make_fingerprint().save(tmp_path / "b-step-zero.json")
    candidate_path = tmp_path / "c-step-zero.json"

    with pytest.raises(FingerprintContractError, match="mismatch"):
        publish_step_zero_fingerprint(
            make_fingerprint(first_sampled_tokens_sha256="e" * 64),
            candidate_path,
            reference_path=reference_path,
        )

    assert not candidate_path.exists()


@pytest.mark.parametrize(
    "call",
    [
        lambda: hash_ordered_sample_ids([]),
        lambda: hash_ordered_sample_ids([True]),
        lambda: hash_sampled_token_rows([]),
        lambda: hash_sampled_token_rows([[1], [1, 2]]),
        lambda: hash_sampled_token_rows([[True]]),
        lambda: make_fingerprint(rollout_global_step=0),
        lambda: make_fingerprint(rollout_global_step=2),
    ],
)
def test_hash_inputs_reject_ambiguous_or_empty_values(call):
    with pytest.raises((FingerprintContractError, TypeError)):
        call()
