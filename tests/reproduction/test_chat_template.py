import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RECURRENT_UTILS_PATH = REPO_ROOT / "recurrent" / "utils.py"
SHARED_UTILS_PATH = REPO_ROOT / "verl" / "utils" / "chat_template.py"
RL_DATASET_PATH = REPO_ROOT / "verl" / "utils" / "dataset" / "rl_dataset.py"


def _load_shared_module():
    spec = importlib.util.spec_from_file_location("shared_chat_template", SHARED_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared_chat_template = _load_shared_module()
apply_without_native_thinking = (
    shared_chat_template.apply_chat_template_without_native_thinking
)
template_token_length = (
    shared_chat_template.chat_template_token_length_without_native_thinking
)


def _load_recurrent_chat_template():
    tree = ast.parse(
        RECURRENT_UTILS_PATH.read_text(encoding="utf-8"),
        filename=str(RECURRENT_UTILS_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "chat_template"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "apply_chat_template_without_native_thinking": apply_without_native_thinking
    }
    exec(compile(module, str(RECURRENT_UTILS_PATH), "exec"), namespace)
    return namespace["chat_template"]


chat_template = _load_recurrent_chat_template()


class FakeQwen35Tokenizer:
    name_or_path = "Qwen/Qwen3.5-4B"
    native_think_prefix = "<think>\n\n</think>\n\n"

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        enable_thinking,
    ):
        self.calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "enable_thinking": enable_thinking,
            }
        )
        rendered = "\n".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return rendered + "\n<assistant>" + self.native_think_prefix


@pytest.mark.parametrize("system", [False, True])
def test_qwen35_explicitly_disables_native_thinking_for_all_prompt_shapes(system):
    tokenizer = FakeQwen35Tokenizer()

    rendered = chat_template(tokenizer, system=system)

    call = tokenizer.calls[0]
    assert call["enable_thinking"] is False
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False
    assert [message["role"] for message in call["messages"]] == (
        ["system", "user"] if system else ["user"]
    )
    assert "{message}" in rendered
    assert ("{system}" in rendered) is system


def test_official_empty_native_think_sentinel_is_allowed_and_preserved():
    tokenizer = FakeQwen35Tokenizer()

    rendered = chat_template(tokenizer)

    assert rendered.endswith("<assistant><think>\n\n</think>\n\n")
    assert tokenizer.calls[0]["enable_thinking"] is False


@pytest.mark.parametrize(
    "native_think_prefix",
    [
        "<think>native reasoning</think>",
        "<think>unfinished native reasoning",
        "orphan native close</think>",
    ],
)
def test_qwen35_nonempty_or_unclosed_native_thinking_fails_closed(
    native_think_prefix,
):
    tokenizer = FakeQwen35Tokenizer()
    tokenizer.native_think_prefix = native_think_prefix

    with pytest.raises(RuntimeError, match=r"Qwen3\.5.*native.*think"):
        chat_template(tokenizer)

    assert tokenizer.calls[0]["enable_thinking"] is False


def test_task_level_protocol_tags_are_preserved_when_native_thinking_is_disabled():
    tokenizer = FakeQwen35Tokenizer()
    task_template = (
        "Return <thinking>task reasoning</thinking> followed by "
        "<update>memory text</update> and <recall>search terms</recall>."
    )

    rendered = chat_template(tokenizer).format(message=task_template)

    assert "<thinking>task reasoning</thinking>" in rendered
    assert "<update>memory text</update>" in rendered
    assert "<recall>search terms</recall>" in rendered
    assert tokenizer.calls[0]["enable_thinking"] is False


class LegacyTokenizer:
    name_or_path = "example/legacy-tokenizer"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        self.calls.append(messages)
        return messages[-1]["content"] + "<assistant>"


def test_legacy_non_qwen_tokenizer_falls_back_without_enable_thinking():
    tokenizer = LegacyTokenizer()

    rendered = chat_template(tokenizer)

    assert rendered == "{message}<assistant>"
    assert len(tokenizer.calls) == 1


class LegacyProcessor(LegacyTokenizer):
    name_or_path = "example/legacy-processor"


def test_legacy_non_qwen_processor_falls_back_without_enable_thinking():
    processor = LegacyProcessor()
    messages = [{"role": "user", "content": "hello"}]

    rendered = apply_without_native_thinking(
        processor,
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    assert rendered == "hello<assistant>"
    assert processor.calls == [messages]


class FakeQwen35Processor(FakeQwen35Tokenizer):
    pass


def test_modern_qwen35_processor_receives_explicit_false():
    processor = FakeQwen35Processor()
    messages = [{"role": "user", "content": "hello"}]

    apply_without_native_thinking(
        processor,
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    assert processor.calls[0]["enable_thinking"] is False


class LegacyQwen35Tokenizer(LegacyTokenizer):
    name_or_path = "Qwen/Qwen3.5-0.8B"


def test_qwen35_never_silently_falls_back_to_native_thinking_default():
    tokenizer = LegacyQwen35Tokenizer()

    with pytest.raises(RuntimeError, match=r"Qwen3\.5.*enable_thinking"):
        chat_template(tokenizer)

    assert tokenizer.calls == []


class Qwen35TokenizerMetadata:
    name_or_path = "Qwen/Qwen3.5-4B"


def test_legacy_processor_with_qwen35_tokenizer_fails_closed():
    processor = LegacyProcessor()
    processor.tokenizer = Qwen35TokenizerMetadata()
    messages = [{"role": "user", "content": "hello"}]

    with pytest.raises(RuntimeError, match=r"Qwen3\.5.*enable_thinking"):
        apply_without_native_thinking(
            processor,
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    assert processor.calls == []


def test_rl_dataset_routes_every_chat_template_path_through_shared_contract():
    tree = ast.parse(
        RL_DATASET_PATH.read_text(encoding="utf-8"),
        filename=str(RL_DATASET_PATH),
    )
    direct_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_chat_template"
    ]
    shared_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_chat_template_without_native_thinking"
    ]
    length_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "chat_template_token_length_without_native_thinking"
    ]

    assert direct_calls == []
    assert len(shared_calls) == 2
    assert sorted(ast.unparse(call.args[0]) for call in shared_calls) == [
        "self.processor",
        "self.tokenizer",
    ]
    assert len(length_calls) == 1
    assert ast.unparse(length_calls[0].args[0]) == "tokenizer"


def test_filter_length_uses_input_ids_not_batch_encoding_field_count():
    class ModernTokenizer:
        name_or_path = "Qwen/Qwen3.5-4B"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["tokenize"] is True
            return {"input_ids": list(range(37)), "attention_mask": [1] * 37}

    assert template_token_length(ModernTokenizer(), [], add_generation_prompt=True) == 37


def test_raw_prompt_ids_use_the_same_center_truncation_name_as_tensor_inputs():
    source = RL_DATASET_PATH.read_text(encoding="utf-8")

    assert 'self.truncation == "center"' in source
    assert 'self.truncation == "middle"' not in source
    assert "raw_prompt_ids diverged from the tensor prompt after truncation" in source


def test_pinned_qwen35_tokenizer_snapshots_when_cached():
    transformers = pytest.importorskip("transformers")
    pinned = {
        "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
        "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
    messages = [
        {
            "role": "user",
            "content": "Keep <thinking>x</thinking><update>y</update><recall>z</recall>",
        }
    ]

    rendered_by_model = {}
    for model_id, revision in pinned.items():
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=True,
            )
        except OSError:
            pytest.skip("pinned Qwen3.5 tokenizers are not all present in the local HF cache")
        rendered_by_model[model_id] = apply_without_native_thinking(
            tokenizer,
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        assert template_token_length(
            tokenizer,
            messages,
            add_generation_prompt=True,
        ) == 30

    assert len(set(rendered_by_model.values())) == 1
    rendered = next(iter(rendered_by_model.values()))
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == (
        "ff253123e276dc3f4df21c87032c25aa874ea14f28bb3fb8ba03c9c8f194d907"
    )
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
