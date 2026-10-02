# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the native Peacebell ChatML renderer.

The hermetic tests use a fake tokenizer (one id per character) and pin the
exact id stream layout: ``<|im_start|>`` + enc(role + "\\n") + enc(content)
+ ``<|im_end|>`` per message, no newline after ``<|im_end|>``, and the
``<|im_start|>assistant\\n`` generation prompt. The parity test loads the real
hub tokenizer and checks the renderer against its authoritative
``build_chatml_ids``, which is what the model was trained on.
"""

import pytest

from vllm.renderers.peacebell import _HfBackedPeacebellTokenizer
from vllm.renderers.peacebell_encoding import (
    IM_END,
    IM_START,
    render_peacebell_messages,
)

PEACEBELL_MODEL = "wayneworkman2012/peacebell-v1-148M"

_IM_START_ID = 100_000
_IM_END_ID = 100_001


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    return False


class FakeSentencePiece:
    """Stands in for ``SentencePieceProcessor``: one id per character."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


class FakeHfTokenizer:
    def __init__(self, with_sp: bool = True) -> None:
        if with_sp:
            self.sp = FakeSentencePiece()

    def get_vocab(self) -> dict[str, int]:
        return {IM_START: _IM_START_ID, IM_END: _IM_END_ID}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        return [ord(ch) for ch in text]


def _decode(ids: list[int]) -> str:
    names = {_IM_START_ID: IM_START, _IM_END_ID: IM_END}
    return "".join(names.get(i, chr(i)) for i in ids)


def _enc(text: str) -> list[int]:
    return [ord(ch) for ch in text]


@pytest.fixture(params=[True, False], ids=["sp", "hf_encode"])
def fake_tokenizer(request):
    return _HfBackedPeacebellTokenizer(FakeHfTokenizer(with_sp=request.param))


def test_control_ids_come_from_vocab(fake_tokenizer):
    assert fake_tokenizer.im_start_id == _IM_START_ID
    assert fake_tokenizer.im_end_id == _IM_END_ID


def test_missing_control_tokens_rejected():
    class NoControl(FakeHfTokenizer):
        def get_vocab(self):
            return {}

    with pytest.raises(ValueError, match="missing control tokens"):
        _HfBackedPeacebellTokenizer(NoControl())


def test_single_user_turn_layout(fake_tokenizer):
    ids = render_peacebell_messages([{"role": "user", "content": "Hi"}], fake_tokenizer)
    assert ids == (
        [_IM_START_ID]
        + _enc("user\n")
        + _enc("Hi")
        + [_IM_END_ID]
        + [_IM_START_ID]
        + _enc("assistant\n")
    )
    assert _decode(ids) == "<|im_start|>user\nHi<|im_end|><|im_start|>assistant\n"


def test_role_and_content_are_encoded_separately():
    """Each segment is its own encode call; nothing is joined into one string."""
    calls: list[str] = []

    class RecordingSentencePiece(FakeSentencePiece):
        def encode(self, text: str) -> list[int]:
            calls.append(text)
            return _enc(text)

    hf_tokenizer = FakeHfTokenizer()
    hf_tokenizer.sp = RecordingSentencePiece()
    tokenizer = _HfBackedPeacebellTokenizer(hf_tokenizer)
    render_peacebell_messages(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Who are you?"},
        ],
        tokenizer,
    )
    assert calls == ["system\n", "Be brief.", "user\n", "Who are you?", "assistant\n"]


def test_multi_turn_has_no_newline_after_im_end(fake_tokenizer):
    ids = render_peacebell_messages(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ],
        fake_tokenizer,
    )
    assert _decode(ids) == (
        "<|im_start|>user\nA<|im_end|>"
        "<|im_start|>assistant\nB<|im_end|>"
        "<|im_start|>user\nC<|im_end|>"
        "<|im_start|>assistant\n"
    )


def test_no_generation_prompt(fake_tokenizer):
    ids = render_peacebell_messages(
        [{"role": "user", "content": "A"}],
        fake_tokenizer,
        add_generation_prompt=False,
    )
    assert _decode(ids) == "<|im_start|>user\nA<|im_end|>"


def test_continue_final_message(fake_tokenizer):
    ids = render_peacebell_messages(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "The answer is"},
        ],
        fake_tokenizer,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    assert _decode(ids) == (
        "<|im_start|>user\nA<|im_end|><|im_start|>assistant\nThe answer is"
    )


def test_continue_final_message_conflicts_with_generation_prompt(fake_tokenizer):
    with pytest.raises(ValueError, match="cannot both be set"):
        render_peacebell_messages(
            [{"role": "user", "content": "A"}],
            fake_tokenizer,
            add_generation_prompt=True,
            continue_final_message=True,
        )


def test_text_parts_and_empty_content(fake_tokenizer):
    ids = render_peacebell_messages(
        [
            {"role": "user", "content": [{"type": "text", "text": "A"}]},
            {"role": "assistant", "content": None},
        ],
        fake_tokenizer,
        add_generation_prompt=False,
    )
    assert (
        _decode(ids) == "<|im_start|>user\nA<|im_end|><|im_start|>assistant\n<|im_end|>"
    )


def test_non_text_parts_rejected(fake_tokenizer):
    with pytest.raises(ValueError, match="text-only"):
        render_peacebell_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
            fake_tokenizer,
        )


def test_empty_conversation_rejected(fake_tokenizer):
    with pytest.raises(ValueError, match="at least one message"):
        render_peacebell_messages([], fake_tokenizer)


_PARITY_CONVERSATIONS = [
    [{"role": "user", "content": "When did World War II end?"}],
    [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Who was Erwin Rommel?"},
    ],
    [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am Peacebell."},
        {"role": "user", "content": "  Explain that again,\nwith two lines.  "},
    ],
    [{"role": "user", "content": "Dates like 6 June 1944 and <s> or </s> inside text"}],
    [{"role": "user", "content": ""}],
]


@pytest.mark.parametrize("conversation", _PARITY_CONVERSATIONS)
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_parity_with_hub_build_chatml_ids(conversation, add_generation_prompt):
    """The renderer must reproduce the training-time encoder id for id."""
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(PEACEBELL_MODEL, trust_remote_code=True)
    assert hasattr(tokenizer, "build_chatml_ids")

    expected = tokenizer.build_chatml_ids(
        conversation, add_generation_prompt=add_generation_prompt
    )
    actual = render_peacebell_messages(
        conversation,
        _HfBackedPeacebellTokenizer(tokenizer),
        add_generation_prompt=add_generation_prompt,
    )
    assert actual == expected
