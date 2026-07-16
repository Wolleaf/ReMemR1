# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import re
from collections.abc import Mapping
from typing import Any


def _template_owner_identifiers(template_owner: Any) -> list[str]:
    identifiers = [type(template_owner).__name__, type(template_owner).__module__]
    for attribute in ("name_or_path", "_name_or_path"):
        try:
            value = getattr(template_owner, attribute, None)
        except Exception:
            continue
        if isinstance(value, str):
            identifiers.append(value)

    try:
        init_kwargs = getattr(template_owner, "init_kwargs", None)
    except Exception:
        init_kwargs = None
    if isinstance(init_kwargs, dict):
        identifiers.extend(
            value
            for key in ("name_or_path", "_name_or_path", "tokenizer_name", "model_id")
            if isinstance((value := init_kwargs.get(key)), str)
        )
    return identifiers


def _is_qwen35_template_owner(template_owner: Any) -> bool:
    """Identify Qwen3.5 tokenizers and processors using stable metadata."""
    identifiers = _template_owner_identifiers(template_owner)
    try:
        nested_tokenizer = getattr(template_owner, "tokenizer", None)
    except Exception:
        nested_tokenizer = None
    if nested_tokenizer is not None and nested_tokenizer is not template_owner:
        identifiers.extend(_template_owner_identifiers(nested_tokenizer))

    qwen35_pattern = re.compile(r"qwen[-_./]?3[._-]?5(?!\d)", re.IGNORECASE)
    return any(qwen35_pattern.search(identifier) for identifier in identifiers)


def _supports_enable_thinking(template_owner: Any) -> bool | None:
    try:
        parameters = inspect.signature(template_owner.apply_chat_template).parameters.values()
    except (TypeError, ValueError):
        return None

    return any(
        parameter.name == "enable_thinking"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _is_unsupported_enable_thinking_error(error: Exception) -> bool:
    message = str(error).lower()
    return "enable_thinking" in message and any(
        marker in message
        for marker in (
            "unexpected",
            "unsupported",
            "not supported",
            "invalid",
            "unrecognized",
            "unknown",
        )
    )


def _validate_qwen35_native_thinking(rendered: str) -> None:
    """Accept the official empty sentinel, but reject active native thinking."""
    opening_tag_end = None
    for tag in re.finditer(r"</?think>", rendered, flags=re.IGNORECASE):
        is_closing_tag = tag.group().startswith("</")
        if not is_closing_tag:
            if opening_tag_end is not None:
                raise RuntimeError(
                    "Qwen3.5 rendered nested or unclosed native <think> tags "
                    "despite enable_thinking=False."
                )
            opening_tag_end = tag.end()
            continue

        if opening_tag_end is None:
            raise RuntimeError(
                "Qwen3.5 rendered an unmatched native </think> tag "
                "despite enable_thinking=False."
            )
        if rendered[opening_tag_end : tag.start()].strip():
            raise RuntimeError(
                "Qwen3.5 rendered a non-empty native <think> payload "
                "despite enable_thinking=False."
            )
        opening_tag_end = None

    if opening_tag_end is not None:
        raise RuntimeError(
            "Qwen3.5 rendered an unclosed native <think> tag "
            "despite enable_thinking=False."
        )


def apply_chat_template_without_native_thinking(
    template_owner: Any,
    conversation: Any,
    **kwargs: Any,
) -> Any:
    """Apply a tokenizer or processor chat template with Qwen thinking disabled.

    Older non-Qwen template owners may not accept the template variable. They retain
    their historical behavior, while Qwen3.5 fails closed instead of silently using
    its native thinking mode.
    """
    supports_thinking = _supports_enable_thinking(template_owner)
    is_qwen35 = _is_qwen35_template_owner(template_owner)

    if supports_thinking is False:
        if is_qwen35:
            raise RuntimeError(
                "Qwen3.5 apply_chat_template does not accept enable_thinking; "
                "refusing to render a prompt that may enable native thinking."
            )
        return template_owner.apply_chat_template(conversation, **kwargs)

    template_kwargs = dict(kwargs)
    template_kwargs["enable_thinking"] = False
    try:
        rendered = template_owner.apply_chat_template(conversation, **template_kwargs)
    except (TypeError, ValueError) as error:
        if not _is_unsupported_enable_thinking_error(error):
            raise
        if is_qwen35:
            raise RuntimeError(
                "Qwen3.5 apply_chat_template rejected enable_thinking=False; "
                "refusing to render a prompt that may enable native thinking."
            ) from error
        return template_owner.apply_chat_template(conversation, **kwargs)

    if is_qwen35 and isinstance(rendered, str):
        _validate_qwen35_native_thinking(rendered)
    return rendered


def chat_template_token_length_without_native_thinking(
    template_owner: Any,
    conversation: Any,
    **kwargs: Any,
) -> int:
    """Render token IDs and return their sequence length across HF API variants."""

    template_kwargs = dict(kwargs)
    template_kwargs["tokenize"] = True
    encoded = apply_chat_template_without_native_thinking(
        template_owner,
        conversation,
        **template_kwargs,
    )
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise RuntimeError("chat template tokenization returned no input_ids")
        encoded = encoded["input_ids"]

    shape = getattr(encoded, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if isinstance(encoded, (list, tuple)) and encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise RuntimeError("chat template length expects exactly one conversation")
        encoded = encoded[0]
    try:
        return len(encoded)
    except TypeError as error:
        raise RuntimeError("chat template tokenization returned an unsupported value") from error
