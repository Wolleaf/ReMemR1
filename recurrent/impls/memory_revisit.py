import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.protocol import (
    CALLBACK_MODES,
    MemoryRecord,
    parse_intermediate_action,
    retrieve_top1,
    resolve_callback_query,
)
from recurrent.utils import TokenTemplate, chat_template
from verl.protocol import DataProto
from verl.utils.chat_template import apply_chat_template_without_native_thinking

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    callback_mode: str = "learned"
    require_manifest: bool = False
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    # max_output_length = max_final_response_length if final else max_memorization_length

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.chunk_size + self.max_memorization_length + self.max_memorization_length

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)


class ManifestProvenanceError(ValueError):
    """Raised when a training row cannot be bound to its sealed sidecar."""


def _right_pad_recalled_memory_tokens(
    token_rows: Sequence[torch.Tensor],
    *,
    max_length: int,
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max recalled-memory length must be a positive integer")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("recalled-memory pad token ID must be a non-negative integer")
    if not isinstance(device, torch.device):
        raise TypeError("recalled-memory output device must be a torch.device")

    padded = torch.full(
        (len(token_rows), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    for row_index, token_ids in enumerate(token_rows):
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError(f"recalled-memory row {row_index} must be a tensor")
        if token_ids.ndim != 1:
            raise ValueError(f"recalled-memory row {row_index} must be one-dimensional")
        if token_ids.dtype != torch.long:
            raise TypeError(f"recalled-memory row {row_index} must use torch.long")
        if token_ids.numel() > max_length:
            raise ValueError(
                f"recalled-memory row {row_index} has {token_ids.numel()} tokens; "
                f"maximum is {max_length}"
            )
        padded[row_index, : token_ids.numel()] = token_ids.to(device=device)

    if padded.device != device:
        raise RuntimeError("recalled-memory tensor was created on the wrong device")
    return padded


_REMOTE_PATH = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _local_data_path(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)):
        raise ManifestProvenanceError("data file paths must be strings or Path objects")
    text = str(value)
    if _REMOTE_PATH.match(text):
        return None
    return Path(text).expanduser().resolve(strict=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bundle_records(parquet_path: Path):
    """Validate a complete bundle and return the sidecar paired to one Parquet."""

    from taskutils.data_synthesis.reproduction_builder import (
        manifest_record_from_dict,
        validate_artifact_bundle,
    )
    from taskutils.data_synthesis.reproduction_manifest import (
        validate_canonical_jsonl,
    )

    manifest = validate_artifact_bundle(parquet_path.parent)
    parquet_entry = manifest["artifacts"].get(parquet_path.name)
    if (
        not isinstance(parquet_entry, Mapping)
        or parquet_entry.get("kind") != "memory-dataset-parquet"
    ):
        raise ManifestProvenanceError(
            f"{parquet_path.name} is not a listed memory-dataset Parquet"
        )
    variant = parquet_entry.get("variant")
    sidecars = [
        (name, entry)
        for name, entry in manifest["artifacts"].items()
        if entry.get("kind") == "canonical-sidecar-jsonl"
        and entry.get("variant") == variant
    ]
    if len(sidecars) != 1:
        raise ManifestProvenanceError(
            f"variant {variant!r} must have exactly one canonical sidecar"
        )
    sidecar_name, sidecar_entry = sidecars[0]
    values = validate_canonical_jsonl(
        (parquet_path.parent / sidecar_name).read_bytes(),
        sidecar_entry["sha256"],
    )
    records = tuple(manifest_record_from_dict(value) for value in values)
    if len(records) != parquet_entry["row_count"]:
        raise ManifestProvenanceError("sidecar row count differs from Parquet metadata")
    return manifest, records


def _identifier_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestProvenanceError(f"{name} must be a sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ManifestProvenanceError(f"{name} must contain non-empty strings")
    return result


def _normalize_sample_provenance(
    value: Any,
    *,
    context_length: int,
    chunk_size: int,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        expected_wrapper_keys = {
            "bundle_manifest_sha256",
            "chunks",
            "manifest_qa_id",
            "manifest_record_sha256",
        }
        if set(value) != expected_wrapper_keys:
            raise ManifestProvenanceError("chunk_provenance wrapper is invalid")
        identity = {
            key: value[key]
            for key in expected_wrapper_keys
            if key != "chunks"
        }
        if any(
            not isinstance(item, str) or not item
            for item in identity.values()
        ):
            raise ManifestProvenanceError(
                "chunk_provenance identity fields must be non-empty strings"
            )
        value = value["chunks"]
    else:
        identity = None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestProvenanceError("chunk_provenance must be a sequence")
    expected_count = (context_length + chunk_size - 1) // chunk_size
    if len(value) != expected_count:
        raise ManifestProvenanceError(
            "chunk provenance does not cover the complete runtime context"
        )
    normalized = []
    expected_keys = {
        "chunk_id",
        "chunk_index",
        "document_ids",
        "supporting_fact_ids",
        "token_end",
        "token_start",
    }
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            raise ManifestProvenanceError(
                f"chunk_provenance[{index}] has an invalid schema"
            )
        expected_start = index * chunk_size
        expected_end = min(expected_start + chunk_size, context_length)
        if (
            item["chunk_index"] != index
            or item["token_start"] != expected_start
            or item["token_end"] != expected_end
        ):
            raise ManifestProvenanceError(
                f"chunk_provenance[{index}] has non-contiguous token bounds"
            )
        chunk_id = item["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ManifestProvenanceError(
                f"chunk_provenance[{index}].chunk_id must be non-empty"
            )
        normalized.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": index,
                "document_ids": _identifier_tuple(
                    item["document_ids"],
                    name=f"chunk_provenance[{index}].document_ids",
                ),
                "supporting_fact_ids": _identifier_tuple(
                    item["supporting_fact_ids"],
                    name=f"chunk_provenance[{index}].supporting_fact_ids",
                ),
                "token_end": expected_end,
                "token_start": expected_start,
            }
        )
    return {"chunks": tuple(normalized), "identity": identity}

class MemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'center':
            raise ValueError('MemoryDataset only support center truncation')
        data_config.max_prompt_length=recurrent_config.max_chunks * recurrent_config.chunk_size
        self.context_key = recurrent_config.context_key
        self.require_manifest = recurrent_config.require_manifest
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )
        self._bind_manifest_provenance(recurrent_config)

    def _bind_manifest_provenance(self, recurrent_config: MemoryConfig) -> None:
        row_claims = []
        for index in range(len(self.dataframe)):
            extra = self.dataframe[index].get("extra_info")
            row_claims.append(
                isinstance(extra, Mapping)
                and isinstance(extra.get("manifest_record_sha256"), str)
            )
        if any(row_claims) and not all(row_claims):
            raise ManifestProvenanceError(
                "dataset cannot mix manifest-identified and legacy rows"
            )

        expected_records = []
        record_bundle_hashes: dict[str, str] = {}
        bundle_hashes: list[str] = []
        bundle_metadata: dict[str, dict[str, Any]] = {}
        unbound_files: list[str] = []

        if not self.require_manifest and not any(row_claims):
            self.bundle_manifest_sha256s = ()
            self.bundle_manifest_metadata = ()
            self._manifest_records = {}
            self._record_bundle_hashes = {}
            return

        for raw_path, cached_path_value in zip(
            self.original_data_files,
            self.data_files,
            strict=True,
        ):
            parquet_path = _local_data_path(raw_path)
            if parquet_path is None or not (parquet_path.parent / "manifest.json").is_file():
                unbound_files.append(str(raw_path))
                continue
            manifest, records = _load_bundle_records(parquet_path)
            parquet_entry = manifest["artifacts"][parquet_path.name]
            cached_path = _local_data_path(cached_path_value)
            if cached_path is None or not cached_path.is_file():
                raise ManifestProvenanceError(
                    "downloaded Parquet cache is not a local readable file"
                )
            if (
                cached_path.stat().st_size != parquet_entry["size_bytes"]
                or _sha256_file(cached_path) != parquet_entry["sha256"]
            ):
                raise ManifestProvenanceError(
                    "downloaded Parquet cache differs from the sealed bundle"
                )
            manifest_hash = manifest["manifest_sha256"]
            if manifest_hash not in bundle_hashes:
                bundle_hashes.append(manifest_hash)
                bundle_metadata[manifest_hash] = {
                    "manifest_sha256": manifest_hash,
                    "mode": manifest["mode"],
                    "profile": manifest["profile"],
                    "tokenizer_name": manifest["tokenizer"]["name"],
                    "tokenizer_revision": manifest["tokenizer"]["revision"],
                }
            if self.require_manifest:
                if not recurrent_config.tokenizer_name or not recurrent_config.tokenizer_revision:
                    raise ManifestProvenanceError(
                        "manifest-bound data requires tokenizer_name and tokenizer_revision"
                    )
                if (
                    manifest["tokenizer"]["name"] != recurrent_config.tokenizer_name
                    or manifest["tokenizer"]["revision"]
                    != recurrent_config.tokenizer_revision
                ):
                    raise ManifestProvenanceError(
                        "runtime tokenizer identity differs from the sealed bundle"
                    )
            for record in records:
                if record.record_sha256 in record_bundle_hashes:
                    raise ManifestProvenanceError(
                        "manifest record hashes must be unique across data files"
                    )
                record_bundle_hashes[record.record_sha256] = manifest_hash
                expected_records.append(record)

        if expected_records and unbound_files:
            raise ManifestProvenanceError(
                "cannot mix manifest-bound and legacy data files: "
                f"{unbound_files}"
            )
        if self.require_manifest and not expected_records:
            raise ManifestProvenanceError(
                "require_manifest=true but no adjacent sealed manifest.json was found"
            )
        if any(row_claims) and not expected_records:
            raise ManifestProvenanceError(
                "Parquet rows claim manifest identity but the sealed bundle is unavailable"
            )

        self.bundle_manifest_sha256s = tuple(bundle_hashes)
        self.bundle_manifest_metadata = tuple(
            bundle_metadata[digest] for digest in bundle_hashes
        )
        self._manifest_records = {
            record.record_sha256: record for record in expected_records
        }
        self._record_bundle_hashes = record_bundle_hashes
        if not expected_records:
            return

        observed_hashes = []
        for index in range(len(self.dataframe)):
            row = self.dataframe[index]
            extra = row.get("extra_info")
            if not isinstance(extra, Mapping):
                raise ManifestProvenanceError(
                    f"Parquet row {index} is missing extra_info"
                )
            observed_hashes.append(extra.get("manifest_record_sha256"))
        expected_hashes = [record.record_sha256 for record in expected_records]
        if observed_hashes != expected_hashes:
            raise ManifestProvenanceError(
                "loaded Parquet row order/filtering differs from the sealed sidecar"
            )

        for record in expected_records:
            if record.chunk_size != recurrent_config.chunk_size:
                raise ManifestProvenanceError(
                    "manifest chunk_size differs from recurrent chunk_size"
                )
            if len(record.chunks) > recurrent_config.max_chunks:
                raise ManifestProvenanceError(
                    "recurrent max_chunks would truncate a manifest record"
                )

    def _record_for_row(self, row: Mapping[str, Any], item: int):
        if not self._manifest_records:
            return None
        extra = row.get("extra_info")
        if not isinstance(extra, Mapping):
            raise ManifestProvenanceError(f"Parquet row {item} is missing extra_info")
        record_hash = extra.get("manifest_record_sha256")
        record = self._manifest_records.get(record_hash)
        if record is None:
            raise ManifestProvenanceError(
                f"Parquet row {item} references an unknown manifest record"
            )
        expected_extra = {
            "context_sha256": record.context_sha256,
            "context_token_ids_sha256": record.context_token_ids_sha256,
            "index": record.qa_index,
            "qa_id": record.qa.qa_id,
            "qa_order_sha256": record.qa_order_sha256,
        }
        for key, expected in expected_extra.items():
            if extra.get(key) != expected:
                raise ManifestProvenanceError(
                    f"Parquet row {item} {key} differs from its sidecar"
                )
        context = row.get(self.context_key)
        if not isinstance(context, str):
            raise ManifestProvenanceError(
                f"Parquet row {item} context must be a string"
            )
        if hashlib.sha256(context.encode("utf-8")).hexdigest() != record.context_sha256:
            raise ManifestProvenanceError(
                f"Parquet row {item} context hash differs from its sidecar"
            )
        return record

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        manifest_record = self._record_for_row(row_dict, item)

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key)

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if manifest_record is not None:
            observed_length = int(attention_mask.sum().item())
            if observed_length != manifest_record.context_token_count:
                raise ManifestProvenanceError(
                    "runtime tokenizer count differs from the sealed context token count"
                )
            from taskutils.data_synthesis.reproduction_manifest import (
                token_ids_sha256,
            )

            observed_token_hash = token_ids_sha256(
                context_ids[0].detach().cpu().tolist()
            )
            if observed_token_hash != manifest_record.context_token_ids_sha256:
                raise ManifestProvenanceError(
                    "runtime tokenizer IDs differ from the sealed context tokens"
                )

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt_ids"] = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False
        )
        row_dict["prompt_text"] = chat[0]["content"]
        if manifest_record is not None:
            row_dict["chunk_provenance"] = {
                "bundle_manifest_sha256": self._record_bundle_hashes[
                    manifest_record.record_sha256
                ],
                "chunks": [chunk.to_dict() for chunk in manifest_record.chunks],
                "manifest_qa_id": manifest_record.qa.qa_id,
                "manifest_record_sha256": manifest_record.record_sha256,
            }
            row_dict["bundle_manifest_sha256"] = self._record_bundle_hashes[
                manifest_record.record_sha256
            ]
            row_dict["manifest_record_sha256"] = manifest_record.record_sha256
            row_dict["manifest_qa_id"] = manifest_record.qa.qa_id
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        non_tensor_keys = ["prompt_ids", "prompt_text"]
        if self._manifest_records:
            non_tensor_keys.extend(
                [
                    "chunk_provenance",
                    "bundle_manifest_sha256",
                    "manifest_record_sha256",
                    "manifest_qa_id",
                ]
            )
        return ["context_ids", "context_length"], non_tensor_keys

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. You should generate response in the following format:
- Output your thinking process in <thinking>your_thinking_process</thinking>.
- Read the provided section carefully and update the memory with the new information that helps to answer the problem in only one <update>the_updated_memory</update> action. Be sure to retain all relevant details from the previous memory while adding any new, useful information.
- If you notice partial key evidence that is not enough to answer the problem, also output only one `<recall>query</recall>` (e.g. `<recall>who's the president of the United States?</recall>`) to retrieve information in previous memories.

<problem>
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

Your answer:
"""


class MemoryAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        if self.config.callback_mode not in CALLBACK_MODES:
            raise ValueError(
                "callback_mode must be one of learned, none, fixed_question; "
                f"got {self.config.callback_mode!r}"
            )
        # A trick to get a simple chat_template for any tokenizer
        # the output text looks like:
        # '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n'
        # This is a format string itself, '{message}' will be replaced by the actual message.
        self.chat_template = chat_template(tokenizer)
        self.token_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE), tokenizer)
        self.token_final_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length + self.token_message_template.length  # self.config.max_raw_input_length should not be counted twice as only memory blocks are recalled not full chunk
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
              f'+ {self.token_message_template.length}(message_template) = {self.max_input_length}\n')
        self.NO_MEMORY_STRING = "No previous memory"
        self.NO_MEMORY_TOKENS = torch.tensor(tokenizer.encode(self.NO_MEMORY_STRING, add_special_tokens=False), dtype=torch.long)
        self.NO_MEMORY_RECALLED_STRING = "No memory was recalled."
        self.NO_MEMORY_RECALLED_TOKENS = torch.tensor(tokenizer.encode(self.NO_MEMORY_RECALLED_STRING, add_special_tokens=False), dtype=torch.long)

    def _render_message_tokens(self, template: str, **values: str) -> torch.Tensor:
        message = template.format(**values)
        rendered = apply_chat_template_without_native_thinking(
            self.tokenizer,
            [{"role": "user", "content": message}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if isinstance(rendered, Mapping):
            if "input_ids" not in rendered:
                raise RuntimeError("chat template tokenization returned no input_ids")
            rendered = rendered["input_ids"]
        if isinstance(rendered, torch.Tensor):
            if rendered.ndim == 2:
                if rendered.shape[0] != 1:
                    raise RuntimeError("chat template returned more than one prompt")
                rendered = rendered[0]
            return rendered.to(dtype=torch.long, device="cpu")
        if (
            isinstance(rendered, Sequence)
            and rendered
            and isinstance(rendered[0], Sequence)
        ):
            if len(rendered) != 1:
                raise RuntimeError("chat template returned more than one prompt")
            rendered = rendered[0]
        return torch.tensor(rendered, dtype=torch.long)

    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        
        self.ctx_length = gen_batch.batch['context_length'] # if all context is used, then the sample will no more be active
        self.bsz = len(self.ctx_length)
        self.history_memory: List[List[MemoryRecord]] = [[] for _ in range(self.bsz)]
        prompt_texts = gen_batch.non_tensor_batch.get("prompt_text")
        if prompt_texts is None or len(prompt_texts) != self.bsz:
            raise ValueError("memory rollout requires one prompt_text per sample")
        self.questions = []
        for index, prompt_text in enumerate(prompt_texts):
            if not isinstance(prompt_text, str) or not prompt_text:
                raise ValueError(f"prompt_text[{index}] must be a non-empty string")
            self.questions.append(prompt_text)
        raw_provenance = gen_batch.non_tensor_batch.get("chunk_provenance")
        self._provenance_validated = False
        if raw_provenance is None:
            if self.config.require_manifest:
                raise ManifestProvenanceError(
                    "require_manifest=true but rollout batch has no chunk provenance"
                )
            self.chunk_provenance = None
            self.manifest_identities = None
        else:
            if len(raw_provenance) != self.bsz:
                raise ManifestProvenanceError(
                    "chunk provenance batch size differs from context batch size"
                )
            normalized_provenance = [
                _normalize_sample_provenance(
                    sample,
                    context_length=int(self.ctx_length[index].item()),
                    chunk_size=self.config.chunk_size,
                )
                for index, sample in enumerate(raw_provenance)
            ]
            identity_arrays = {
                "bundle_manifest_sha256": gen_batch.non_tensor_batch.get(
                    "bundle_manifest_sha256"
                ),
                "manifest_record_sha256": gen_batch.non_tensor_batch.get(
                    "manifest_record_sha256"
                ),
                "manifest_qa_id": gen_batch.non_tensor_batch.get("manifest_qa_id"),
            }
            if any(value is None for value in identity_arrays.values()):
                raise ManifestProvenanceError(
                    "chunk provenance is missing its batched identity arrays"
                )
            for name, values in identity_arrays.items():
                if len(values) != self.bsz:
                    raise ManifestProvenanceError(
                        f"{name} batch size differs from context batch size"
                    )
            for index, normalized in enumerate(normalized_provenance):
                identity = normalized["identity"]
                if identity is None:
                    raise ManifestProvenanceError(
                        "chunk provenance wrapper is missing sealed identity"
                    )
                for name, values in identity_arrays.items():
                    if identity[name] != values[index]:
                        raise ManifestProvenanceError(
                            f"chunk provenance {name} differs from batched identity"
                        )
            self.chunk_provenance = [
                normalized["chunks"] for normalized in normalized_provenance
            ]
            self.manifest_identities = [
                normalized["identity"] for normalized in normalized_provenance
            ]
            self._provenance_validated = True
        self.memory = np.empty(self.bsz, dtype=object)
        self.recall_memories = np.empty(self.bsz, dtype=object)
        self.memory_text = np.empty(self.bsz, dtype=object)
        self.recalled_memory_text = np.empty(self.bsz, dtype=object)
        for values in (
            self.memory,
            self.recall_memories,
            self.memory_text,
            self.recalled_memory_text,
        ):
            values.fill(None)
        self.is_final = False
    
    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        # suppose 0 is pad_token_id
        # max_chunks = 3, chunk_size = 2
        # pi is token in prompt, ti is token in chat template, 
        # [1,2] [3,4] [5,0] | p0 string
        # [1,2] [3,0] [0,0] | p1,p1 string
        # [1,0] [0,0] [0,0] | p2,p2,p2 string
        # -------- round 1 ---------
        # [1,2]            [t0,p0,t1, m,t2, 1, 2,t3]                           [ 0, 0, 0,t0,p0,t1, m,t2, 1, 2,t3]
        # [1,2]  -format-> [t0,p1,p1,t1, m,t2, 1, 2,t3] -pad2Dlist2Tendors->   [ 0, 0,t0,p1,p1,t1, m,t2, 1, 2,t3]
        # [1,0]            [t0,p2,p2,p3,t1, m,t2, 1,t3]                        [ 0, 0,t0,p2,p2,p3,t1, m,t2, 1,t3]
        # get mask & positionids
        active_mask = self.ctx_length > self.step * self.config.chunk_size
        self.active_mask = active_mask
        gen_batch = self.gen_batch
        # if all context is used, and its not done, then it will be the final turn for this batch
        if active_mask.sum().item() == 0:
            self.is_final = True
            self.messages = [
                self._render_message_tokens(
                    TEMPLATE_FINAL_BOXED,
                    prompt=prompt,
                    memory=memory if memory is not None else self.NO_MEMORY_STRING,
                    recalled_memory=(
                        recalled_memory
                        if recalled_memory is not None
                        else self.NO_MEMORY_RECALLED_STRING
                    ),
                )
                for prompt, memory, recalled_memory in zip(
                    self.questions,
                    self.memory_text,
                    self.recalled_memory_text,
                )
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.max_final_response_length,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'FINAL TURN: MemoryAgent.next() done')
        else:
            # 1. no need to pad prompt
            # 2. context padded for 2D indexing, elegant engineering
            # 3. no need to pad memory
            active_indices = active_mask.nonzero().flatten().cpu().tolist()
            prompt_i = [self.questions[int(index)] for index in active_indices]
            chunk_i = gen_batch.batch['context_ids'][active_mask, self.config.chunk_size * self.step: self.config.chunk_size * (self.step+1)] # bs * chunk_size
            memory_i = self.memory_text[active_mask]
            recalled_memory_i = self.recalled_memory_text[active_mask]
            
            # Render the complete prompt exactly as the external evaluator does.
            self.messages = [
                self._render_message_tokens(
                        TEMPLATE,
                        prompt=prompt,
                        memory=memory if memory is not None else self.NO_MEMORY_STRING,
                        recalled_memory=(
                            recalled_memory
                            if recalled_memory is not None
                            else self.NO_MEMORY_RECALLED_STRING
                        ),
                        chunk=self.tokenizer.decode(
                            chunk[chunk != self.tokenizer.pad_token_id],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                )
                for prompt, memory, recalled_memory, chunk in zip(prompt_i, memory_i, recalled_memory_i, chunk_i)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] # map active sample to original batch
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'MemoryAgent.action() done')
        if any(message.numel() > self.max_input_length for message in self.messages):
            raise RuntimeError(
                "fully rendered recurrent prompt exceeds the configured input bound"
            )
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        all_decoded_responses = self.tokenizer.batch_decode(gen_output.batch['responses'], skip_special_tokens=True) # List[str], length: [recalled_bsz]
        if not self.is_final:
            active_indices = self.active_mask.nonzero().flatten().cpu().tolist()
            parsed_actions = [parse_intermediate_action(response) for response in all_decoded_responses]
        else:
            active_indices = list(range(len(all_decoded_responses)))
            parsed_actions = []

        recalled_queries = self._resolve_callback_queries(parsed_actions, active_indices)
        retrievals = [
            retrieve_top1(query, self.history_memory[idx]) if query is not None else None
            for query, idx in zip(recalled_queries, active_indices)
        ]
        recalled_memories = [
            result.record.update_text if result is not None else None
            for result in retrievals
        ]
        recalled_memories_values = [
            torch.tensor(self.tokenizer.encode(memory_str, add_special_tokens=False), dtype=torch.long) if memory_str is not None else self.NO_MEMORY_RECALLED_TOKENS
            for memory_str in recalled_memories
        ]
        recalled_memories_arr = np.empty(len(recalled_memories_values), dtype=object)
        recalled_memories_arr[:] = recalled_memories_values
        recalled_memories_tensor = _right_pad_recalled_memory_tokens(
            recalled_memories_values,
            max_length=self.config.max_memorization_length,
            pad_token_id=self.tokenizer.pad_token_id,
            device=gen_output.batch['responses'].device,
        )
        gen_output.batch['recalled_memories'] = recalled_memories_tensor
        gen_output.batch['recalled_step_ids'] = torch.tensor(
            [result.record.step_id if result is not None else -1 for result in retrievals],
            dtype=torch.long,
            device=gen_output.batch['responses'].device,
        )
        gen_output.batch['recall_scores'] = torch.tensor(
            [result.score if result is not None else 0.0 for result in retrievals],
            dtype=torch.float32,
            device=gen_output.batch['responses'].device,
        )
        gen_output.non_tensor_batch['recall_queries'] = np.asarray(recalled_queries, dtype=object)
        source_chunk_ids = np.empty(len(retrievals), dtype=object)
        source_chunk_ids[:] = [
            result.record.source_chunk_ids if result is not None else ()
            for result in retrievals
        ]
        gen_output.non_tensor_batch['recalled_source_chunk_ids'] = source_chunk_ids
        source_doc_ids = np.empty(len(retrievals), dtype=object)
        source_doc_ids[:] = [
            result.record.source_doc_ids if result is not None else ()
            for result in retrievals
        ]
        gen_output.non_tensor_batch['recalled_source_doc_ids'] = source_doc_ids

        current_sources = [
            self._source_for_step(idx) if not self.is_final else ((), ())
            for idx in active_indices
        ]
        current_chunk_ids = np.empty(len(current_sources), dtype=object)
        current_chunk_ids[:] = [source[0] for source in current_sources]
        gen_output.non_tensor_batch['current_source_chunk_ids'] = current_chunk_ids
        current_doc_ids = np.empty(len(current_sources), dtype=object)
        current_doc_ids[:] = [source[1] for source in current_sources]
        gen_output.non_tensor_batch['current_source_doc_ids'] = current_doc_ids
        current_identities = (
            [self.manifest_identities[int(idx)] for idx in active_indices]
            if self.manifest_identities is not None
            else [None] * len(active_indices)
        )
        for name in (
            "bundle_manifest_sha256",
            "manifest_record_sha256",
            "manifest_qa_id",
        ):
            gen_output.non_tensor_batch[name] = np.asarray(
                [identity[name] if identity is not None else None for identity in current_identities],
                dtype=object,
            )

        if not self.is_final:
            all_update_memories = [action.update for action in parsed_actions]

            # update memory
            update_values = [
                torch.tensor(self.tokenizer.encode(memory_str, add_special_tokens=False), dtype=torch.long) if memory_str is not None else self.NO_MEMORY_TOKENS
                for memory_str in all_update_memories
            ]
            updates_arr = np.empty(len(update_values), dtype=object)
            updates_arr[:] = update_values
            self.memory[self.active_mask] = updates_arr # List[torch.Tensor], shape: [recalled_bsz]
            self.recall_memories[self.active_mask] = recalled_memories_arr
            memory_text_arr = np.empty(len(all_update_memories), dtype=object)
            memory_text_arr[:] = all_update_memories
            recalled_text_arr = np.empty(len(recalled_memories), dtype=object)
            recalled_text_arr[:] = recalled_memories
            self.memory_text[self.active_mask] = memory_text_arr
            self.recalled_memory_text[self.active_mask] = recalled_text_arr

            # update history memory
            self.update_memory(all_update_memories, active_indices)

        self.log_step(gen_output)
        self.step += 1
        return gen_output

    def _resolve_callback_queries(self, parsed_actions, active_indices):
        if self.is_final:
            return [None] * len(active_indices)
        return [
            resolve_callback_query(
                self.config.callback_mode,
                learned_query=action.recall,
                question=self.questions[int(idx)],
            )
            for action, idx in zip(parsed_actions, active_indices)
        ]

    def update_memory(self, memory_strings: List[Optional[str]], active_indices: List[int]):
        assert len(active_indices) == len(memory_strings)
        for idx, memory in zip(active_indices, memory_strings):
            if memory is None:
                continue
            source_chunk_ids, source_doc_ids = self._source_for_step(int(idx))
            self.history_memory[int(idx)].append(
                MemoryRecord(
                    step_id=self.step,
                    update_text=memory,
                    source_chunk_ids=source_chunk_ids,
                    source_doc_ids=source_doc_ids,
                )
            )

    def _source_for_step(self, sample_index: int):
        if self.chunk_provenance is None:
            if self.config.require_manifest:
                raise ManifestProvenanceError(
                    "formal memory update has no manifest provenance"
                )
            return (self.step,), ()
        if self.config.require_manifest and not self._provenance_validated:
            raise ManifestProvenanceError(
                "formal chunk provenance was not validated at rollout start"
            )
        try:
            chunk = self.chunk_provenance[sample_index][self.step]
        except IndexError as exc:
            raise ManifestProvenanceError(
                "memory update step is outside manifest chunk provenance"
            ) from exc
        return (chunk["chunk_id"],), chunk["document_ids"]

    ## MODIFIED: Helper function to parse the callback ID from the LLM's text output
    def _parse_recall_query(self, text_response: str) -> str:
        return parse_intermediate_action(text_response).recall

    def _parse_update_memory(self, text_response: str) -> str:
        return parse_intermediate_action(text_response).update

    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.memory_text
        del self.messages
        del self.history_memory
        del self.chunk_provenance
        del self.manifest_identities
        del self._provenance_validated
        del self.questions
        del self.recall_memories
        del self.recalled_memory_text
        del self.active_mask
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index
        

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=3000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.step if not self.is_final else "FINAL"
        active_count = self.active_mask.sum().item() if not self.is_final else self.bsz
        logger.info(f"\n{' '*10}{'='*30}[RECURRENT] STEP{step} [active: {active_count}/{self.bsz}] {'='*30}{' '*10}")

        # Message and Response section
        if self.active_mask[0]:
            decoded_message = self.tokenizer.decode(self.messages[0])
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
