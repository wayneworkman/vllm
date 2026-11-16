# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boris2 chat renderer.

Boris2 (`wayneworkman2012/boris2-peacebell-base` and `-think`) is the
Peacebell architecture with a byte-level BPE tokenizer whose only special
token is `<|endoftext|>`. Its chat format is ChatML written as plain text:
`<|im_start|>` and `<|im_end|>` are ordinary characters spanning several
tokens, so the Jinja chat template is rendered and tokenized in one call like
any other HF model.

What the stock HF renderer cannot express is how a reply ends: at the
*string* `<|im_end|>`. The model is not trained to emit EOS after it and
would run on into the next turn, so this renderer hands that string to the
engine as a default stop. A model without a chat template (the base,
text-continuation model) gets no default stop.
"""

from vllm.config import VllmConfig
from vllm.tokenizers.hf import HfTokenizer

from .hf import HfRenderer

IM_END = "<|im_end|>"


def plain_text_end_of_turn(tokenizer: HfTokenizer | None) -> list[str]:
    """The stop strings a tokenizer's ChatML template needs, if any.

    Returns `[IM_END]` when the chat template closes turns with `<|im_end|>`
    and the vocabulary has no such token, so nothing but a stop string can
    end the reply.
    """
    if tokenizer is None:
        return []
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or IM_END not in chat_template:
        return []
    if IM_END in tokenizer.get_vocab():
        return []
    return [IM_END]


class Boris2Renderer(HfRenderer):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: HfTokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._default_stop = plain_text_end_of_turn(self.tokenizer)

    def get_default_stop(self) -> list[str]:
        return self._default_stop
