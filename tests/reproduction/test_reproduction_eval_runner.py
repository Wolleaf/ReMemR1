import ast
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from taskutils.data_synthesis.reproduction_builder import build_artifact_bundle
from taskutils.data_synthesis.reproduction_manifest import (
    DocumentInput,
    EvalManifestContract,
    ManifestMetadata,
    QARecord,
    SupportingFactInput,
    build_manifest_record,
    ordered_values_sha256,
)
from taskutils.memory_eval.reproduction_runner import (
    COMPLETION_FILENAME,
    FINAL_PROMPT,
    INTERMEDIATE_PROMPT,
    ModelLoadSpec,
    RUN_CONFIG_FILENAME,
    RESULTS_FILENAME,
    SUMMARY_FILENAME,
    SampleEvaluationConfig,
    ScriptedBackend,
    TransformersBackend,
    EvaluationRunnerError,
    evaluate_manifest_record,
    load_eval_records,
    load_transformers_backend,
    run_evaluation_task,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _tokenize_with_offsets(text):
    spans = [match.span() for match in re.finditer(r"\S+", text)]
    return {
        "input_ids": list(range(1, len(spans) + 1)),
        "offset_mapping": spans,
    }


def _make_record(
    *,
    qa_index=0,
    chunk_size=2,
    gold_answers=("wrong answer", "right answer"),
):
    documents = tuple(
        DocumentInput(
            title=f"Title {qa_index}-{index}",
            text=(
                f"alpha evidence gives right answer for item {qa_index}"
                if index == 0
                else f"distractor text number {index} for item {qa_index}"
            ),
            source_document_id=f"source-{qa_index}-{index}",
        )
        for index in range(3)
    )
    return build_manifest_record(
        metadata=ManifestMetadata(
            source_name="fixture",
            source_revision="source-revision-1",
            source_sha256="a" * 64,
            tokenizer_name="whitespace-fixture",
            tokenizer_revision="tokenizer-revision-1",
            seed=42,
        ),
        qa_index=qa_index,
        qa_order_sha256=hashlib.sha256(b"fixed-qa-order").hexdigest(),
        qa=QARecord.create(
            f"fixture:qa-{qa_index}",
            "Where is alpha evidence?",
            gold_answers,
        ),
        documents=documents,
        supporting_facts=(
            SupportingFactInput(
                document_index=0,
                sentence_index=0,
                text=f"alpha evidence gives right answer for item {qa_index}",
            ),
        ),
        encode=_tokenize_with_offsets,
        chunk_size=chunk_size,
        pool_document_count=len(documents),
        document_pool_sha256=ordered_values_sha256(
            [document.document_id for document in documents]
        ),
    )


def _intermediate(update, recall=None):
    recall_text = "" if recall is None else f"<recall>{recall}</recall>"
    return (
        f"<thinking>inspect evidence</thinking>"
        f"<update>{update}</update>{recall_text}"
    )


def _valid_outputs(record, *, final_answer="right answer", recall=None):
    memory_outputs = tuple(
        _intermediate(f"state value {index}", recall=recall)
        for index in range(len(record.chunks))
    )
    return (*memory_outputs, rf"\boxed{{{final_answer}}}")


def _scripted_backend(outputs, *, load_delay=0.0):
    return ScriptedBackend(
        ModelLoadSpec(
            kind="scripted",
            allow_test_backend=True,
            scripted_outputs=tuple(outputs),
            scripted_load_delay_s=load_delay,
        )
    )


def _sample_config(mode="learned", *, chunk_size=2, memory_tokens=128, final_tokens=64):
    return SampleEvaluationConfig(
        callback_mode=mode,
        chunk_size=chunk_size,
        memory_max_tokens=memory_tokens,
        final_max_tokens=final_tokens,
    )


def _scripted_spec(outputs, *, load_delay=0.0):
    return ModelLoadSpec(
        kind="scripted",
        allow_test_backend=True,
        scripted_outputs=tuple(outputs),
        scripted_load_delay_s=load_delay,
    )


def _run_task(records, output_dir, spec, *, sample_timeout=10.0, load_timeout=20.0):
    return run_evaluation_task(
        records,
        output_dir=output_dir,
        backend_spec=spec,
        sample_config=_sample_config(),
        model_load_timeout_s=load_timeout,
        sample_timeout_s=sample_timeout,
        task_timeout_s=30.0,
        input_metadata={"fixture": True},
        start_method="spawn",
    )


def _read_results(output_dir):
    return [
        json.loads(line)
        for line in (output_dir / RESULTS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]


def _training_prompt_constants():
    source = REPO_ROOT / "recurrent" / "impls" / "memory_revisit.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"TEMPLATE", "TEMPLATE_FINAL_BOXED"}:
            values[target.id] = ast.literal_eval(node.value)
    return values


def test_eval_prompts_are_byte_identical_to_training_task_prompts():
    training = _training_prompt_constants()

    assert INTERMEDIATE_PROMPT == training["TEMPLATE"].replace("{prompt}", "{question}")
    assert FINAL_PROMPT == training["TEMPLATE_FINAL_BOXED"].replace(
        "{prompt}", "{question}"
    )


def test_dynamic_chunk_count_consumes_every_chunk_doc_and_all_gold_answers():
    record = _make_record(chunk_size=2)
    assert len(record.chunks) > 6
    outputs = _valid_outputs(record, final_answer="right answer", recall="alpha evidence")

    result = evaluate_manifest_record(
        record,
        _scripted_backend(outputs),
        _sample_config(),
    )

    assert result["status"] == "success"
    assert result["processed_chunk_count"] == len(record.chunks)
    assert result["processed_doc_count"] == len(record.documents)
    memory_turns = result["trajectory"][:-1]
    assert [turn["chunk"]["chunk_id"] for turn in memory_turns] == [
        chunk.chunk_id for chunk in record.chunks
    ]
    assert {
        doc_id
        for turn in memory_turns
        for doc_id in turn["chunk"]["source_doc_ids"]
    } == {document.document_id for document in record.documents}
    assert result["metrics"]["scores"]["exact_match"] == 1.0
    assert result["metrics"]["scores"]["token_f1"] == 1.0
    assert result["metrics"]["scores"]["substring_exact_match"] == 1.0
    assert result["metrics"]["scores"]["exact_match_gold"] == "right answer"
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", turn["generation"]["generated_token_sha256"])
        for turn in result["trajectory"]
    )


def test_eval_rejects_same_length_but_different_context_token_ids():
    record = _make_record(chunk_size=2)
    backend = _scripted_backend(_valid_outputs(record))
    original_encode = backend.encode_context
    backend.encode_context = lambda text: tuple(
        token_id + 100 for token_id in original_encode(text)
    )

    with pytest.raises(EvaluationRunnerError, match="tokenizer IDs"):
        evaluate_manifest_record(record, backend, _sample_config())


@pytest.mark.parametrize("mode", ["learned", "none", "fixed_question"])
def test_callback_modes_change_effective_query_retrieval_and_injection(mode):
    record = _make_record()
    outputs = []
    for index in range(len(record.chunks)):
        update = (
            "where is alpha evidence right answer"
            if index == 0
            else f"state value {index}"
        )
        recall = "alpha evidence" if mode != "fixed_question" else None
        outputs.append(_intermediate(update, recall=recall))
    outputs.append(r"\boxed{right answer}")

    result = evaluate_manifest_record(
        record,
        _scripted_backend(outputs),
        _sample_config(mode),
    )
    turns = result["trajectory"][:-1]
    callback = result["metrics"]["callback"]

    if mode == "none":
        assert callback["effective_query_count"] == 0
        assert callback["retrieval_count"] == 0
        assert all(turn["callback"]["effective_query"] is None for turn in turns)
        assert all(turn["callback"]["retrieval"] is None for turn in turns)
        assert all(
            turn["recalled_memory_next"] == "No memory was recalled."
            for turn in turns
        )
    else:
        expected_query = (
            "alpha evidence" if mode == "learned" else record.qa.question
        )
        assert callback["effective_query_count"] == len(record.chunks)
        assert callback["retrieval_count"] == len(record.chunks) - 1
        assert all(turn["callback"]["effective_query"] == expected_query for turn in turns)
        assert turns[0]["callback"]["retrieval"] is None
        assert all(
            turn["callback"]["retrieval"]["step_id"] == 0
            for turn in turns[1:]
        )
        assert all(
            turn["callback"]["retrieval"]["source_chunk_ids"]
            == [record.chunks[0].chunk_id]
            for turn in turns[1:]
        )
        assert callback["supporting_doc_hit_count"] == len(record.chunks) - 1
        assert callback["lexical_gold_hit_count"] == len(record.chunks) - 1

    expected_status = "absent" if mode == "fixed_question" else "valid"
    assert all(
        turn["callback"]["model_query_status"] == expected_status for turn in turns
    )


def test_query_format_and_generation_limit_metrics_keep_all_failure_shapes():
    record = _make_record()
    assert len(record.chunks) >= 5
    outputs = [
        "<thinking>ok</thinking><update>state value</update><recall>alpha</recall>",
        "<thinking>ok</thinking><update>state value</update><recall></recall>",
        "<thinking>ok</thinking><update>state value</update><recall>broken",
        (
            "<thinking>ok</thinking><update>state value</update>"
            "<recall>alpha</recall><recall>beta</recall>"
        ),
    ]
    outputs.extend(
        "<thinking>ok</thinking><update>state value</update>"
        for _ in range(len(record.chunks) - len(outputs))
    )
    outputs.append(r"\boxed{right answer}")

    result = evaluate_manifest_record(
        record,
        _scripted_backend(outputs),
        _sample_config(memory_tokens=2, final_tokens=1),
    )

    assert result["metrics"]["callback"]["model_query_status_counts"] == {
        "valid": 1,
        "empty": 1,
        "malformed": 1,
        "duplicate": 1,
        "absent": len(record.chunks) - 4,
    }
    assert result["metrics"]["format"]["intermediate_valid_count"] == (
        len(record.chunks) - 3
    )
    assert result["metrics"]["format"]["final_valid"] is True
    assert result["metrics"]["format"]["thinking_single_non_empty_rate"] == 1.0
    assert result["metrics"]["format"]["update_single_non_empty_rate"] == 1.0
    assert result["metrics"]["callback"]["model_query_status_rates"] == {
        name: count / len(record.chunks)
        for name, count in result["metrics"]["callback"][
            "model_query_status_counts"
        ].items()
    }
    assert result["metrics"]["truncation"] == {
        "final_reached_token_limit": True,
        "memory_reached_token_limit_count": len(record.chunks),
        "memory_reached_token_limit_rate": 1.0,
    }


def test_task_records_sample_exception_then_continues_and_refuses_overwrite(tmp_path):
    failed_record = _make_record(qa_index=0)
    successful_record = _make_record(qa_index=1)
    output = tmp_path / "eval-output"
    spec = _scripted_spec(
        ("__raise__:boom", *_valid_outputs(successful_record))
    )

    summary = _run_task((failed_record, successful_record), output, spec)

    assert summary["sample_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["success_count"] == 1
    assert summary["metric_denominators"] == {
        "answer_all_inputs": 2,
        "behavior_success_only": 1,
    }
    assert summary["metrics"]["exact_match"] == pytest.approx(0.5)
    assert summary["metrics"]["token_f1"] == pytest.approx(0.5)
    assert summary["metrics"]["substring_exact_match"] == pytest.approx(0.5)
    assert {path.name for path in output.iterdir()} == {
        COMPLETION_FILENAME,
        RESULTS_FILENAME,
        RUN_CONFIG_FILENAME,
        SUMMARY_FILENAME,
    }
    results = _read_results(output)
    assert [result["qa_id"] for result in results] == [
        failed_record.qa.qa_id,
        successful_record.qa.qa_id,
    ]
    assert results[0]["error"] == {
        "message": "boom",
        "stage": "sample_evaluation",
        "type": "RuntimeError",
    }
    assert results[1]["status"] == "success"
    marker = json.loads((output / COMPLETION_FILENAME).read_text(encoding="utf-8"))
    assert marker["status"] == "completed_with_failures"

    with pytest.raises(FileExistsError, match="overwrite"):
        _run_task((failed_record,), output, _scripted_spec(("unused",)))


def test_sample_timeout_is_structured_and_still_publishes_completion(tmp_path):
    record = _make_record()
    output = tmp_path / "timeout-output"
    first_valid = _intermediate("state value")

    summary = _run_task(
        (record,),
        output,
        _scripted_spec((f"__sleep__:2:{first_valid}",)),
        sample_timeout=0.2,
    )

    assert summary["failure_count"] == 1
    result = _read_results(output)[0]
    assert result["error"]["stage"] == "sample_timeout_or_worker_exit"
    assert result["error"]["type"] == "WorkerTimeoutError"
    assert (output / COMPLETION_FILENAME).is_file()


def test_worker_exit_is_detected_without_waiting_for_sample_timeout(tmp_path):
    record = _make_record()
    output = tmp_path / "exit-output"
    started = time.monotonic()

    summary = _run_task(
        (record,),
        output,
        _scripted_spec(("__exit__",)),
        sample_timeout=10.0,
    )

    assert time.monotonic() - started < 8.0
    assert summary["failure_count"] == 1
    result = _read_results(output)[0]
    assert result["error"]["type"] == "WorkerExitedError"
    assert "code 23" in result["error"]["message"]


def test_model_load_timeout_marks_every_input_failed_and_publishes(tmp_path):
    records = (_make_record(qa_index=0), _make_record(qa_index=1))
    output = tmp_path / "load-timeout-output"

    summary = _run_task(
        records,
        output,
        _scripted_spec((), load_delay=2.0),
        load_timeout=0.2,
    )

    assert summary["failure_count"] == len(records)
    assert summary["metrics"]["exact_match"] == 0.0
    assert summary["metrics"]["token_f1"] == 0.0
    assert summary["metrics"]["substring_exact_match"] == 0.0
    assert summary["metrics"]["success_only"]["retrieval_rate"] is None
    results = _read_results(output)
    assert all(result["error"]["stage"] == "model_load" for result in results)
    assert all(result["error"]["type"] == "WorkerTimeoutError" for result in results)
    assert [result["qa_id"] for result in results] == [
        record.qa.qa_id for record in records
    ]


class _FakeModel:
    def __init__(self):
        self.to_device = None
        self.eval_called = False

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class _FastTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token_id = 1


def _transformers_spec(tmp_path, artifact_kind="base"):
    artifact_path = None
    if artifact_kind != "base":
        artifact = tmp_path / artifact_kind
        artifact.mkdir()
        artifact_path = str(artifact)
    return ModelLoadSpec(
        artifact_kind=artifact_kind,
        base_model_id="Qwen/Qwen3.5-4B",
        revision=FIXED_REVISION,
        artifact_path=artifact_path,
        tokenizer_id="Qwen/Qwen3.5-4B",
        tokenizer_revision=FIXED_REVISION,
        template_revision="rememr1-template-v1",
        device="cpu",
    )


def test_transformers_loader_pins_text_model_tokenizer_bf16_sdpa_and_adapter(tmp_path):
    spec = _transformers_spec(tmp_path, "adapter")
    base_model = _FakeModel()
    adapter_model = _FakeModel()
    calls = {}

    def qwen_loader(source, **kwargs):
        calls["qwen"] = (source, kwargs)
        return SimpleNamespace(
            model=base_model,
            metadata=SimpleNamespace(to_dict=lambda: {"text_only": True}),
        )

    def tokenizer_loader(source, **kwargs):
        calls["tokenizer"] = (source, kwargs)
        return _FastTokenizer()

    metadata = SimpleNamespace(
        base_model_id=spec.base_model_id,
        base_model_revision=spec.revision,
        tokenizer_id=spec.tokenizer_id,
        tokenizer_revision=spec.tokenizer_revision,
        template_revision=spec.template_revision,
        to_dict=lambda: {"validated": True},
    )

    def adapter_loader(model, path, **kwargs):
        calls["adapter"] = (model, path, kwargs)
        return adapter_model

    deterministic_calls = []
    cudnn = SimpleNamespace(benchmark=True, deterministic=False)
    torch_module = SimpleNamespace(
        bfloat16=object(),
        manual_seed=lambda seed: deterministic_calls.append(("cpu", seed)),
        cuda=SimpleNamespace(
            manual_seed_all=lambda seed: deterministic_calls.append(("cuda", seed))
        ),
        use_deterministic_algorithms=lambda enabled, warn_only: deterministic_calls.append(
            ("algorithms", enabled, warn_only)
        ),
        backends=SimpleNamespace(cudnn=cudnn),
    )
    backend = load_transformers_backend(
        spec,
        torch_module=torch_module,
        qwen_loader=qwen_loader,
        tokenizer_loader=tokenizer_loader,
        adapter_validator=lambda path: metadata,
        adapter_loader=adapter_loader,
    )

    source, qwen_kwargs = calls["qwen"]
    assert source == spec.base_model_id
    assert qwen_kwargs == {
        "revision": FIXED_REVISION,
        "attn_implementation": "sdpa",
        "dtype": torch_module.bfloat16,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    assert calls["tokenizer"] == (
        spec.tokenizer_id,
        {
            "revision": FIXED_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_fast": True,
        },
    )
    assert calls["adapter"] == (
        base_model,
        spec.artifact_path,
        {"validate_export": False},
    )
    assert backend.model is adapter_model
    assert adapter_model.to_device == "cpu"
    assert adapter_model.eval_called is True
    assert backend.metadata["adapter_metadata"] == {"validated": True}
    assert deterministic_calls == [
        ("cpu", 42),
        ("cuda", 42),
        ("algorithms", True, True),
    ]
    assert cudnn.benchmark is False
    assert cudnn.deterministic is True


def test_transformers_loader_rejects_adapter_identity_drift(tmp_path):
    spec = _transformers_spec(tmp_path, "adapter")
    metadata = SimpleNamespace(
        base_model_id=spec.base_model_id,
        base_model_revision=spec.revision,
        tokenizer_id=spec.tokenizer_id,
        tokenizer_revision="different-tokenizer-revision",
        template_revision=spec.template_revision,
    )

    with pytest.raises(EvaluationRunnerError, match="tokenizer_revision"):
        load_transformers_backend(
            spec,
            torch_module=SimpleNamespace(bfloat16=object()),
            qwen_loader=lambda *args, **kwargs: SimpleNamespace(
                model=_FakeModel(), metadata=None
            ),
            tokenizer_loader=lambda *args, **kwargs: _FastTokenizer(),
            adapter_validator=lambda path: metadata,
            adapter_loader=lambda model, path, **kwargs: model,
        )


def test_merged_artifact_uses_its_own_validator_and_never_impersonates_base_snapshot(
    tmp_path,
):
    spec = _transformers_spec(tmp_path, "merged")
    calls = {}
    merged_model = _FakeModel()
    metadata = {
        "base_model_id": spec.base_model_id,
        "base_model_revision": spec.revision,
        "tokenizer_id": spec.tokenizer_id,
        "tokenizer_revision": spec.tokenizer_revision,
        "template_revision": spec.template_revision,
        "dtype": spec.dtype,
        "metadata_sha256": "a" * 64,
    }

    def merged_loader(source, **kwargs):
        calls["merged"] = (source, kwargs)
        return merged_model

    def merged_validator(path):
        calls["validated"] = path
        return metadata

    backend = load_transformers_backend(
        spec,
        torch_module=SimpleNamespace(bfloat16=object()),
        qwen_loader=lambda *args, **kwargs: pytest.fail(
            "merged weights must not use the pinned base snapshot loader"
        ),
        tokenizer_loader=lambda *args, **kwargs: _FastTokenizer(),
        merged_validator=merged_validator,
        merged_loader=merged_loader,
    )

    assert calls["validated"] == spec.artifact_path
    assert calls["merged"][0] == spec.artifact_path
    assert "revision" not in calls["merged"][1]
    assert calls["merged"][1]["local_files_only"] is True
    assert backend.model is merged_model
    assert backend.metadata["merged_metadata"] == metadata


def test_transformers_generation_disables_sampling_and_native_thinking():
    torch = pytest.importorskip("torch")

    class Tokenizer(_FastTokenizer):
        name_or_path = "Qwen/Qwen3.5-4B"

        def __init__(self):
            self.template_calls = []

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            return_tensors,
            return_dict,
            enable_thinking,
        ):
            self.template_calls.append(
                {
                    "messages": messages,
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "return_tensors": return_tensors,
                    "return_dict": return_dict,
                    "enable_thinking": enable_thinking,
                }
            )
            return {
                "input_ids": torch.tensor([[2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

        def decode(self, token_ids, **kwargs):
            del token_ids, kwargs
            return "decoded answer"

    class Model:
        def __init__(self):
            self.generate_kwargs = None

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            suffix = torch.tensor([[7, 8]], dtype=torch.long)
            return torch.cat((kwargs["input_ids"], suffix), dim=1)

    model = Model()
    tokenizer = Tokenizer()

    def render_without_thinking(owner, messages, **kwargs):
        return owner.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )

    backend = TransformersBackend(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        device="cpu",
        metadata={"backend": "fixture"},
        chat_template_renderer=render_without_thinking,
    )

    generated = backend.generate_greedy("hello", max_new_tokens=2)

    assert generated.text == "decoded answer"
    assert generated.generated_tokens == 2
    assert generated.reached_token_limit is True
    assert generated.generated_token_sha256 == hashlib.sha256(b"[7,8]").hexdigest()
    assert tokenizer.template_calls[0]["enable_thinking"] is False
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["num_beams"] == 1
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs


def test_transformers_contract_rejects_prompt_or_data_tokenizer_drift(tmp_path):
    wrong_template = _transformers_spec(tmp_path)
    wrong_template = ModelLoadSpec(
        **{
            **wrong_template.public_dict(),
            "template_revision": "different-template-v2",
        }
    )
    with pytest.raises(ValueError, match="compiled eval prompt"):
        wrong_template.validate()

    record = _make_record()
    spec = _transformers_spec(tmp_path)
    with pytest.raises(EvaluationRunnerError, match="data manifest"):
        run_evaluation_task(
            (record,),
            output_dir=tmp_path / "must-not-exist",
            backend_spec=spec,
            sample_config=_sample_config(),
            model_load_timeout_s=1.0,
            sample_timeout_s=1.0,
            task_timeout_s=1.0,
        )
    assert not (tmp_path / "must-not-exist").exists()


def _hotpot_record(index):
    evidence_title = f"Evidence {index}"
    return {
        "_id": f"hotpot-{index}",
        "question": f"Question {index}?",
        "answers": [f"answer {index}", f"alias {index}"],
        "context": [
            [evidence_title, [f"Evidence gives answer {index}."]],
            [f"Distractor {index}", [f"Unrelated text {index}."]],
        ],
        "supporting_facts": [[evidence_title, 0]],
    }


def test_eval_sidecar_loader_validates_bundle_and_preserves_variant_prefix(tmp_path):
    pytest.importorskip("pyarrow")
    source = tmp_path / "hotpot.json"
    source.write_text(
        json.dumps([_hotpot_record(index) for index in range(5)]),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    build_artifact_bundle(
        input_path=source,
        output_dir=bundle,
        mode="eval",
        dataset="hotpotqa",
        source_revision="fixture-source-revision",
        tokenizer_name="fixture-tokenizer",
        tokenizer_revision="fixture-tokenizer-revision",
        seed=42,
        encode=_tokenize_with_offsets,
        profile="fixture",
        eval_contract=EvalManifestContract(
            qa_count=2,
            prefix_document_count=2,
            pool_document_count=4,
            chunk_size=3,
        ),
    )

    expected_sha256 = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )["manifest_sha256"]
    prefix, prefix_manifest = load_eval_records(
        bundle,
        expected_manifest_sha256=expected_sha256,
        variant=2,
        sample_count=1,
    )
    pool, pool_manifest = load_eval_records(
        bundle,
        expected_manifest_sha256=expected_sha256,
        variant=4,
        sample_count=1,
    )

    assert prefix_manifest["manifest_sha256"] == pool_manifest["manifest_sha256"]
    assert len(prefix) == len(pool) == 1
    assert prefix[0].qa == pool[0].qa
    assert prefix[0].document_count == 2
    assert pool[0].document_count == 4
    assert prefix[0].documents == pool[0].documents[:2]
    assert prefix[0].document_pool_sha256 == pool[0].document_pool_sha256

    with pytest.raises(EvaluationRunnerError, match="locked run contract"):
        load_eval_records(
            bundle,
            expected_manifest_sha256="f" * 64,
            variant=2,
            sample_count=1,
        )
