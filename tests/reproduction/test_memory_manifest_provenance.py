import json
import re
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

try:
    import ray  # noqa: F401
except ModuleNotFoundError:
    ray_stub = types.ModuleType("ray")
    ray_stub.ObjectRef = object
    ray_stub.get = lambda values: values
    sys.modules["ray"] = ray_stub

async_utils_stub = types.ModuleType("recurrent.async_utils")
async_utils_stub.ChatCompletionProxy = object
sys.modules.setdefault("recurrent.async_utils", async_utils_stub)
recurrent_utils_stub = types.ModuleType("recurrent.utils")
recurrent_utils_stub.TokenTemplate = object
recurrent_utils_stub.chat_template = lambda tokenizer: "{message}"
sys.modules.setdefault("recurrent.utils", recurrent_utils_stub)

from recurrent.impls.memory_revisit import (
    ManifestProvenanceError,
    MemoryAgent,
    MemoryConfig,
    MemoryDataset,
    TEMPLATE,
)
from taskutils.data_synthesis.reproduction_builder import (
    TrainManifestContract,
    build_artifact_bundle,
)


def _tokenize_with_offsets(text):
    spans = [match.span() for match in re.finditer(r"\S+", text)]
    return {"input_ids": list(range(1, len(spans) + 1)), "offset_mapping": spans}


def _source_record(index):
    evidence = f"Evidence {index}"
    return {
        "_id": f"qa-{index}",
        "question": f"Who is entity {index}?",
        "answers": [f"answer {index}", f"alias {index}"],
        "context": [
            [
                evidence,
                [
                    f"Entity {index} has answer {index}.",
                    f"A second supporting sentence for {index}.",
                ],
            ],
            [f"Distractor {index}", [f"Unrelated local text {index}."]],
        ],
        "supporting_facts": [[evidence, 0], [evidence, 1]],
        "level": "hard",
        "type": "bridge",
    }


class _WhitespaceTokenizer:
    pad_token_id = 0

    def __init__(self, extra_context_token=False, token_id_offset=0):
        self.extra_context_token = extra_context_token
        self.token_id_offset = token_id_offset

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        count = len(re.findall(r"\S+", text)) + int(self.extra_context_token)
        return {
            "input_ids": torch.arange(
                1 + self.token_id_offset,
                count + 1 + self.token_id_offset,
                dtype=torch.long,
            ).unsqueeze(0),
            "attention_mask": torch.ones((1, count), dtype=torch.long),
        }

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(1, len(re.findall(r"\S+", text)) + 1))

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(value) for value in token_ids)


def _build_bundle(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("datasets")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([_source_record(index) for index in range(5)]),
        encoding="utf-8",
    )
    output = tmp_path / "bundle"
    manifest = build_artifact_bundle(
        input_path=source,
        output_dir=output,
        mode="train",
        dataset="hotpotqa",
        source_revision="fixture-source-revision",
        tokenizer_name="fixture-tokenizer",
        tokenizer_revision="fixture-tokenizer-revision",
        seed=42,
        encode=_tokenize_with_offsets,
        profile="fixture",
        train_contract=TrainManifestContract(
            qa_count=2,
            document_count=3,
            chunk_size=1000,
            max_chunks=1,
            min_context_tokens=1,
            max_context_tokens=1000,
        ),
    )
    return output, manifest


def _config(tmp_path, require_manifest=True):
    recurrent = MemoryConfig(
        context_key="context",
        max_prompt_length=64,
        chunk_size=1000,
        max_memorization_length=32,
        max_chunks=1,
        max_final_response_length=16,
        require_manifest=require_manifest,
        tokenizer_name="fixture-tokenizer",
        tokenizer_revision="fixture-tokenizer-revision",
    )
    data = OmegaConf.create(
        {
            "cache_dir": str(tmp_path / "dataset-cache"),
            "filter_overlong_prompts": False,
            "max_prompt_length": 64,
            "prompt_key": "prompt",
            "truncation": "center",
        }
    )
    return recurrent, data


def test_memory_dataset_binds_sealed_bundle_and_emits_chunk_provenance(tmp_path):
    output, manifest = _build_bundle(tmp_path)
    recurrent, data = _config(tmp_path)

    dataset = MemoryDataset(
        recurrent_config=recurrent,
        data_files=str(output / "train.parquet"),
        tokenizer=_WhitespaceTokenizer(),
        data_config=data,
    )
    item = dataset[0]

    assert dataset.bundle_manifest_sha256s == (manifest["manifest_sha256"],)
    assert item["bundle_manifest_sha256"] == manifest["manifest_sha256"]
    assert item["manifest_record_sha256"]
    assert item["manifest_qa_id"].startswith("hotpotqa:")
    assert item["context_length"].item() > 0
    assert len(item["chunk_provenance"]["chunks"]) == 1
    assert item["chunk_provenance"]["chunks"][0]["chunk_id"] == "chunk-000000"
    assert item["chunk_provenance"]["manifest_record_sha256"] == item[
        "manifest_record_sha256"
    ]
    assert "chunk_provenance" in dataset.get_bactch_keys()[1]


def test_memory_dataset_rejects_runtime_tokenizer_drift(tmp_path):
    output, _ = _build_bundle(tmp_path)
    recurrent, data = _config(tmp_path)
    dataset = MemoryDataset(
        recurrent_config=recurrent,
        data_files=str(output / "train.parquet"),
        tokenizer=_WhitespaceTokenizer(extra_context_token=True),
        data_config=data,
    )

    with pytest.raises(ManifestProvenanceError, match="runtime tokenizer count"):
        dataset[0]

    _, data = _config(tmp_path)
    same_count_different_ids = MemoryDataset(
        recurrent_config=recurrent,
        data_files=str(output / "train.parquet"),
        tokenizer=_WhitespaceTokenizer(token_id_offset=100),
        data_config=data,
    )
    with pytest.raises(ManifestProvenanceError, match="runtime tokenizer IDs"):
        same_count_different_ids[0]


def test_chunk_provenance_stays_one_object_per_sample_through_repeat(tmp_path):
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import collate_fn

    output, _ = _build_bundle(tmp_path)
    recurrent, data = _config(tmp_path)
    tokenizer = _WhitespaceTokenizer()
    dataset = MemoryDataset(
        recurrent_config=recurrent,
        data_files=str(output / "train.parquet"),
        tokenizer=tokenizer,
        data_config=data,
    )
    batch = DataProto.from_single_dict(collate_fn([dataset[0], dataset[1]]))
    tensor_keys, non_tensor_keys = dataset.get_bactch_keys()
    generation_batch = batch.pop(
        batch_keys=tensor_keys,
        non_tensor_batch_keys=non_tensor_keys,
    ).repeat(repeat_times=2, interleave=True)

    provenance = generation_batch.non_tensor_batch["chunk_provenance"]
    assert provenance.shape == (4,)
    assert all(isinstance(value, dict) for value in provenance)

    agent = object.__new__(MemoryAgent)
    agent.config = recurrent
    agent.tokenizer = tokenizer
    agent.start(generation_batch, {})
    assert len(agent.chunk_provenance) == 4
    assert all(chunks[0]["chunk_id"] == "chunk-000000" for chunks in agent.chunk_provenance)

    tampered = generation_batch.non_tensor_batch["chunk_provenance"].copy()
    tampered[0] = dict(tampered[0])
    tampered[0]["manifest_record_sha256"] = "f" * 64
    generation_batch.non_tensor_batch["chunk_provenance"] = tampered
    rejected = object.__new__(MemoryAgent)
    rejected.config = recurrent
    rejected.tokenizer = tokenizer
    with pytest.raises(ManifestProvenanceError, match="differs from batched identity"):
        rejected.start(generation_batch, {})


def test_memory_dataset_requires_adjacent_manifest_in_formal_mode(tmp_path):
    output, _ = _build_bundle(tmp_path)
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "train.parquet").write_bytes((output / "train.parquet").read_bytes())
    recurrent, data = _config(tmp_path)

    with pytest.raises(ManifestProvenanceError, match="require_manifest=true"):
        MemoryDataset(
            recurrent_config=recurrent,
            data_files=str(orphan / "train.parquet"),
            tokenizer=_WhitespaceTokenizer(),
            data_config=data,
        )


def test_memory_agent_records_real_chunk_and_document_ids():
    agent = object.__new__(MemoryAgent)
    agent.config = SimpleNamespace(require_manifest=True)
    agent.step = 0
    agent.history_memory = [[]]
    agent.chunk_provenance = [
        (
            {
                "chunk_id": "chunk-000000",
                "document_ids": ("doc-a", "doc-b"),
            },
        )
    ]
    agent.manifest_identities = [
        {
            "bundle_manifest_sha256": "a" * 64,
            "manifest_record_sha256": "b" * 64,
            "manifest_qa_id": "qa-1",
        }
    ]
    agent._provenance_validated = True

    agent.update_memory(["retained evidence"], [0])

    assert agent.history_memory[0][0].source_chunk_ids == ("chunk-000000",)
    assert agent.history_memory[0][0].source_doc_ids == ("doc-a", "doc-b")


def test_memory_agent_fails_closed_without_formal_provenance():
    agent = object.__new__(MemoryAgent)
    agent.config = SimpleNamespace(require_manifest=True)
    agent.step = 0
    agent.history_memory = [[]]
    agent.chunk_provenance = None
    agent.manifest_identities = None
    agent._provenance_validated = False

    with pytest.raises(ManifestProvenanceError, match="no manifest provenance"):
        agent.update_memory(["memory"], [0])


def test_training_and_eval_render_identical_whole_prompt_token_ids():
    from taskutils.memory_eval.reproduction_runner import TransformersBackend

    class BoundaryTokenizer:
        name_or_path = "Qwen/Qwen3.5-4B"
        pad_token_id = 0
        eos_token_id = 9

        def __init__(self):
            self.thinking_flags = []

        def apply_chat_template(
            self,
            messages,
            *,
            add_generation_prompt,
            tokenize,
            enable_thinking,
            return_tensors=None,
            return_dict=False,
        ):
            assert add_generation_prompt is True
            assert tokenize is True
            self.thinking_flags.append(enable_thinking)
            text = messages[0]["content"]
            token_ids = [
                len(text),
                sum(text.encode("utf-8")) % 997,
                text.count("alpha beta"),
            ]
            if return_tensors == "pt":
                values = torch.tensor([token_ids], dtype=torch.long)
                return {
                    "attention_mask": torch.ones_like(values),
                    "input_ids": values,
                }
            assert return_dict is False
            return token_ids

        def decode(self, token_ids, **kwargs):
            del token_ids, kwargs
            return "decoded"

    class Model:
        def generate(self, **kwargs):
            self.input_ids = kwargs["input_ids"].detach().clone()
            suffix = torch.tensor([[7]], dtype=torch.long)
            return torch.cat((kwargs["input_ids"], suffix), dim=1)

    values = {
        "prompt": "alpha beta question",
        "memory": "memory alpha",
        "recalled_memory": "recalled beta",
        "chunk": "alpha beta evidence",
    }
    tokenizer = BoundaryTokenizer()
    agent = object.__new__(MemoryAgent)
    agent.tokenizer = tokenizer
    training_ids = agent._render_message_tokens(TEMPLATE, **values)
    model = Model()
    backend = TransformersBackend(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        device="cpu",
        metadata={"backend": "fixture"},
    )

    backend.generate_greedy(TEMPLATE.format(**values), max_new_tokens=1)

    assert training_ids.tolist() == model.input_ids[0].tolist()
    assert tokenizer.thinking_flags == [False, False]
