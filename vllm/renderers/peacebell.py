# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Peacebell chat renderer.

Peacebell ships no Jinja chat template: its training-time ChatML encoding
cannot be reproduced from text (see ``peacebell_encoding.py``), so chat
messages are rendered directly to token ids, the way the Inkling renderer
does for that model.
"""

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.logger import init_logger
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams
from .peacebell_encoding import IM_END, IM_START, render_peacebell_messages

logger = init_logger(__name__)


class _HfBackedPeacebellTokenizer:
    """Adapts the hub's HF tokenizer to the encoding core's protocol.

    The hub tokenizer (``tokenization_gpt.GPTTokenizer``) wraps a
    ``sentencepiece.SentencePieceProcessor`` as ``.sp``. Text is encoded with
    that processor directly when it is exposed, because that is exactly what
    training did; the HF ``encode`` path differs only for text containing the
    literal ``<s>`` / ``</s>`` strings, which HF splits as added tokens.
    """

    def __init__(self, tokenizer: HfTokenizer) -> None:
        vocab = tokenizer.get_vocab()
        missing = [tok for tok in (IM_START, IM_END) if tok not in vocab]
        if missing:
            raise ValueError(
                f"Peacebell tokenizer is missing control tokens: {missing}"
            )
        self.im_start_id: int = vocab[IM_START]
        self.im_end_id: int = vocab[IM_END]

        sp = getattr(tokenizer, "sp", None)
        if sp is not None and hasattr(sp, "encode"):
            self._encode = sp.encode
        else:
            logger.warning_once(
                "Peacebell tokenizer does not expose its SentencePiece processor; "
                "falling back to HF encode(add_special_tokens=False)."
            )
            self._encode = lambda text: tokenizer.encode(text, add_special_tokens=False)

    def encode_text(self, text: str) -> list[int]:
        return list(self._encode(text))


class PeacebellRenderer(BaseRenderer[HfTokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: HfTokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._peacebell_tokenizer = _HfBackedPeacebellTokenizer(self.get_tokenizer())
        self._render_async = make_async(self._render, executor=self._executor)

    def _render(
        self,
        conversation: list[ConversationMessage],
        params: ChatParams,
    ) -> list[int]:
        kwargs = params.chat_template_kwargs or {}
        continue_final_message = bool(kwargs.get("continue_final_message", False))
        add_generation_prompt = bool(
            kwargs.get("add_generation_prompt", not continue_final_message)
        )
        try:
            return render_peacebell_messages(
                conversation,
                self._peacebell_tokenizer,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )
        except ValueError:
            raise
        except (TypeError, KeyError) as e:
            raise ValueError(str(e)) from e

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        if mm_data is not None or mm_uuids is not None:
            raise ValueError("Peacebell is a text-only model")

        token_ids = self._render(conversation, params)

        return conversation, parse_dec_only_prompt(token_ids)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        if mm_data is not None or mm_uuids is not None:
            raise ValueError("Peacebell is a text-only model")

        token_ids = await self._render_async(conversation, params)

        return conversation, parse_dec_only_prompt(token_ids)
