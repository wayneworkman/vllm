# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peacebell ChatML chat-encoding core.

Pure implementation of the Peacebell chat rendering, kept free of vLLM
imports: it depends only on the :class:`PeacebellTextTokenizer` protocol and
speaks OpenAI-style message dicts.

Peacebell was trained on ChatML rendered by ``build_chatml_ids`` in the
checkpoint repo's ``tokenization_gpt.py``. That encoder does NOT build one big
string and tokenize it. Per message it emits the ``<|im_start|>`` id, then
``encode(role + "\\n")``, then ``encode(content)`` as a *separate* SentencePiece
call, then the ``<|im_end|>`` id; no newline follows ``<|im_end|>``. Because the
SentencePiece model adds a dummy ``▁`` prefix to every ``encode`` call, a Jinja
chat template applied to the joined text would tokenize the role/content
boundary differently and would also turn the literal ``<|im_start|>`` into
``▁`` + ``<|im_start|>``. This module reproduces the training-time id stream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
GENERATION_PROMPT_ROLE = "assistant"


class PeacebellTextTokenizer(Protocol):
    """Minimal tokenizer surface needed to render Peacebell chats."""

    @property
    def im_start_id(self) -> int: ...

    @property
    def im_end_id(self) -> int: ...

    def encode_text(self, text: str) -> list[int]:
        """Raw SentencePiece encoding of ``text`` with no special tokens."""
        ...


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "text":
                raise ValueError(
                    "Peacebell is text-only; unsupported message content part: "
                    f"{part!r}"
                )
            parts.append(str(part.get("text", "")))
        return "".join(parts)
    raise ValueError(f"Unsupported message content: {content!r}")


def render_peacebell_messages(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: PeacebellTextTokenizer,
    *,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> list[int]:
    """Render OpenAI-style chat messages to Peacebell prompt token ids.

    Mirrors ``GPTTokenizer.build_chatml_ids`` from the checkpoint repo exactly.

    Args:
        messages: ``{"role": str, "content": str | text parts}`` dicts.
        tokenizer: Provides the ChatML control ids and raw text encoding.
        add_generation_prompt: Append ``<|im_start|>assistant\\n`` so the
            model continues as the assistant.
        continue_final_message: Leave the final message open (no
            ``<|im_end|>``, no generation prompt) so the model continues it.

    Returns:
        The prompt token ids.

    """
    if not messages:
        raise ValueError("Peacebell chat rendering needs at least one message")
    if continue_final_message and add_generation_prompt:
        raise ValueError(
            "continue_final_message and add_generation_prompt cannot both be set"
        )

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    last = len(messages) - 1

    ids: list[int] = []
    for i, message in enumerate(messages):
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"Message {i} has no role")
        ids.append(im_start)
        ids.extend(tokenizer.encode_text(role + "\n"))
        ids.extend(tokenizer.encode_text(_message_text(message)))
        if continue_final_message and i == last:
            break
        ids.append(im_end)

    if add_generation_prompt:
        ids.append(im_start)
        ids.extend(tokenizer.encode_text(GENERATION_PROMPT_ROLE + "\n"))

    return ids
