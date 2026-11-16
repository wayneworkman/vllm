# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Boris2 renderer's default stop string.

Boris2 writes ChatML as plain text, so a chat reply ends at the string
`<|im_end|>` rather than at a token. These tests pin when the renderer hands
that string to the engine: only for a tokenizer whose chat template closes
turns with `<|im_end|>` and whose vocabulary cannot express it as one token.
"""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend

from vllm.renderers.boris2 import IM_END, plain_text_end_of_turn

EOS = "<|endoftext|>"

# The template shipped in the Boris2 chat model's tokenizer_config.json.
CHATML_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] | trim }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    return False


def _tokenizer(
    extra_vocab: tuple[str, ...] = (), chat_template: str | None = None
) -> TokenizersBackend:
    vocab = {token: i for i, token in enumerate((EOS, "<", "|", ">", *extra_vocab))}
    return TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token=EOS)),
        eos_token=EOS,
        chat_template=chat_template,
    )


def test_chat_model_stops_at_the_im_end_string():
    tokenizer = _tokenizer(chat_template=CHATML_TEMPLATE)

    assert plain_text_end_of_turn(tokenizer) == [IM_END] == ["<|im_end|>"]


def test_base_model_without_chat_template_has_no_default_stop():
    assert plain_text_end_of_turn(_tokenizer()) == []


def test_template_without_im_end_has_no_default_stop():
    tokenizer = _tokenizer(chat_template="{{ messages[0]['content'] }}")

    assert plain_text_end_of_turn(tokenizer) == []


def test_im_end_token_in_the_vocabulary_needs_no_stop_string():
    tokenizer = _tokenizer(extra_vocab=(IM_END,), chat_template=CHATML_TEMPLATE)

    assert plain_text_end_of_turn(tokenizer) == []


def test_missing_tokenizer_has_no_default_stop():
    assert plain_text_end_of_turn(None) == []
