"""Verify a reproduction PEFT adapter or merge it into its pinned base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verl.utils.checkpoint.reproduction import (
    MERGED_MODEL_METADATA_FILENAME,
    AdapterExportMetadata,
    atomic_publish_directory,
    build_merged_model_metadata,
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_tensor_state_sha256,
    load_peft_adapter,
    validate_adapter_export,
    validate_merged_model_artifact,
)


MERGE_METADATA_FILENAME = MERGED_MODEL_METADATA_FILENAME


def _torch_dtype(torch_module: Any, name: str) -> Any:
    mapping = {"bfloat16": torch_module.bfloat16}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported merge dtype {name!r}") from exc


def _load_pinned_base_model(metadata, torch_module, *, local_files_only):
    mapping = metadata.text_mapping
    if mapping.get("mapping_strategy") == "transformers_native_qwen35_causal_lm":
        from verl.models.qwen35 import load_qwen35_text_model

        dtype_name = str(mapping.get("dtype", "")).removeprefix("torch.")
        source_dtype = getattr(torch_module, dtype_name, None)
        if source_dtype is None:
            raise RuntimeError(
                f"unsupported dtype in strict Qwen3.5 mapping: {dtype_name!r}"
            )
        result = load_qwen35_text_model(
            metadata.base_model_id,
            revision=metadata.base_model_revision,
            attn_implementation=mapping["attention_implementation"],
            dtype=source_dtype,
            strict=True,
            local_files_only=local_files_only,
        )
        loaded_mapping = result.metadata.to_dict()
        if canonical_json_sha256(loaded_mapping) != metadata.text_mapping_sha256:
            raise RuntimeError("strict Qwen3.5 text mapping hash mismatch")
        return result.model

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        metadata.base_model_id,
        revision=metadata.base_model_revision,
        local_files_only=local_files_only,
        dtype=torch_module.float32,
    )


def _assert_merged_model_contract(model, torch_module) -> str:
    if hasattr(model, "peft_config"):
        raise RuntimeError("merged model still exposes PEFT adapter configuration")
    residual_modules = [
        name
        for name, module in model.named_modules()
        if "lora" in name.casefold()
        or "peft" in type(module).__module__.casefold()
        or "lora" in type(module).__name__.casefold()
    ]
    if residual_modules:
        raise RuntimeError(
            f"merged model still contains PEFT/LoRA modules: {residual_modules[:8]}"
        )
    wrong_dtypes = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch_module.bfloat16
    ]
    if wrong_dtypes:
        raise RuntimeError(
            f"merged model floating parameters are not all BF16: {wrong_dtypes[:8]}"
        )
    return canonical_tensor_state_sha256(model.state_dict())


def merge_and_verify_adapter(
    adapter_directory: str | Path,
    merged_directory: str | Path,
    *,
    prompt: str,
    dtype_name: str = "bfloat16",
    device: str = "cuda",
    local_files_only: bool = True,
    max_new_tokens: int = 4,
    rtol: float = 2e-3,
    atol: float = 2e-3,
) -> dict[str, Any]:
    """Merge, save, reload, and compare logits plus greedy generation."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Torch, Transformers, and PEFT are required for merge export") from exc

    adapter_path = Path(adapter_directory).expanduser().resolve(strict=True)
    merged_path = Path(merged_directory).expanduser().resolve()
    metadata = validate_adapter_export(adapter_path)
    dtype = _torch_dtype(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        metadata.tokenizer_id,
        revision=metadata.tokenizer_revision,
        local_files_only=local_files_only,
    )
    base_model = _load_pinned_base_model(
        metadata,
        torch,
        local_files_only=local_files_only,
    )
    adapter_model = load_peft_adapter(base_model, adapter_path)
    adapter_model.to(device=device, dtype=dtype).eval()
    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        adapter_logits = adapter_model(**encoded).logits.detach().float().cpu()
        adapter_tokens = adapter_model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        ).detach().cpu()
    merged_model = adapter_model.merge_and_unload(safe_merge=True)
    merged_model.to(dtype=dtype).eval()
    model_state_sha256 = _assert_merged_model_contract(merged_model, torch)

    merge_metadata = build_merged_model_metadata(
        global_step=metadata.global_step,
        source_adapter_metadata_sha256=metadata.sha256,
        base_model_id=metadata.base_model_id,
        base_model_revision=metadata.base_model_revision,
        tokenizer_id=metadata.tokenizer_id,
        tokenizer_revision=metadata.tokenizer_revision,
        template_revision=metadata.template_revision,
        text_mapping_sha256=metadata.text_mapping_sha256,
        dtype=dtype_name,
        model_state_sha256=model_state_sha256,
        verification_prompt=prompt,
        max_new_tokens=max_new_tokens,
        rtol=rtol,
        atol=atol,
    )

    def writer(staging: Path) -> None:
        merged_model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        (staging / MERGE_METADATA_FILENAME).write_bytes(
            canonical_json_bytes(merge_metadata) + b"\n"
        )

    validation_result = {}

    def validator(staging: Path) -> None:
        validate_merged_model_artifact(
            staging,
            expected_metadata=merge_metadata,
        )
        reloaded = AutoModelForCausalLM.from_pretrained(
            staging,
            local_files_only=True,
            dtype=dtype,
        )
        reloaded.to(device).eval()
        reloaded_state_sha256 = _assert_merged_model_contract(reloaded, torch)
        if reloaded_state_sha256 != model_state_sha256:
            raise RuntimeError("reloaded merged model-state hash mismatch")
        with torch.no_grad():
            reloaded_logits = reloaded(**encoded).logits.detach().float().cpu()
            reloaded_tokens = reloaded.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            ).detach().cpu()
        torch.testing.assert_close(
            reloaded_logits,
            adapter_logits,
            rtol=rtol,
            atol=atol,
        )
        if not torch.equal(reloaded_tokens, adapter_tokens):
            raise RuntimeError(
                "reloaded merged model generation differs from the adapter"
            )
        validation_result["verified"] = True

    atomic_publish_directory(
        merged_path,
        writer,
        validator=validator,
        marker_metadata={
            "artifact_type": "merged_causal_lm",
            "metadata_sha256": merge_metadata["metadata_sha256"],
            "model_state_sha256": model_state_sha256,
        },
    )
    del merged_model, adapter_model, base_model
    if validation_result.get("verified") is not True:
        raise RuntimeError("merged model staging validation did not complete")

    return {
        **merge_metadata,
        "adapter_directory": str(adapter_path),
        "merged_directory": str(merged_path),
        "verified": validation_result["verified"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="verify adapter-only files and hashes")
    verify_parser.add_argument("--adapter-dir", type=Path, required=True)

    merge_parser = subparsers.add_parser("merge", help="merge, save, reload, and compare")
    merge_parser.add_argument("--adapter-dir", type=Path, required=True)
    merge_parser.add_argument("--merged-dir", type=Path, required=True)
    merge_parser.add_argument(
        "--prompt",
        default="ReMemR1 merge verification.",
    )
    merge_parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    merge_parser.add_argument("--device", default="cuda")
    merge_parser.add_argument("--max-new-tokens", type=int, default=4)
    merge_parser.add_argument("--rtol", type=float, default=2e-3)
    merge_parser.add_argument("--atol", type=float, default=2e-3)
    merge_parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow hub access instead of requiring the pinned local cache",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        metadata: AdapterExportMetadata = validate_adapter_export(args.adapter_dir)
        result = {**metadata.to_dict(), "adapter_directory": str(args.adapter_dir.resolve())}
    else:
        result = merge_and_verify_adapter(
            args.adapter_dir,
            args.merged_dir,
            prompt=args.prompt,
            dtype_name=args.dtype,
            device=args.device,
            local_files_only=not args.allow_download,
            max_new_tokens=args.max_new_tokens,
            rtol=args.rtol,
            atol=args.atol,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
