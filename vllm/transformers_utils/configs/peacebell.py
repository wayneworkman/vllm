# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peacebell model configuration.

Mirrors the `configuration_gpt.GPTConfig` (`model_type = "gpt"`) shipped
inside the `wayneworkman2012/peacebell-v1-*` and `boris2-peacebell-*`
checkpoint repos, so vLLM can load them without `trust_remote_code`. The
project's own field names (`n_embd` / `n_head` / `n_layer` / `block_size`)
are authoritative; `attribute_map` exposes them under the generic names
vLLM reads.

The two families share the architecture and differ in vocabulary:

- Peacebell: SentencePiece with ChatML control tokens. Its `config.json`
  carries `im_start_id` / `im_end_id` / `unk_pad_id`.
- Boris2: byte-level BPE whose only special token is `<|endoftext|>`.
  Its `config.json` carries `train_pad_id` instead.
"""

from transformers import PreTrainedConfig


class PeacebellConfig(PreTrainedConfig):
    model_type = "gpt"

    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "max_position_embeddings": "block_size",
    }

    def __init__(
        self,
        vocab_size: int = 28000,
        n_embd: int = 1664,
        n_head: int = 26,
        n_layer: int = 27,
        block_size: int = 32768,
        dropout: float = 0.0,
        tie_weights: bool = True,
        rezero_scale_init: float = 0.01,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1_000_000.0,
        # Peacebell only: ids of its ChatML control tokens and of <unk>/<pad>.
        im_start_id: int | None = None,
        im_end_id: int | None = None,
        unk_pad_id: int | None = None,
        # Boris2 only, and what tells the two families apart: the id training
        # padded with. It was never a prediction target, so its output row is
        # untrained and must never be sampled.
        train_pad_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        pad_token_id: int | None = 0,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.tie_weights = tie_weights
        self.rezero_scale_init = rezero_scale_init
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.unk_pad_id = unk_pad_id
        self.train_pad_id = train_pad_id

        kwargs.setdefault("tie_word_embeddings", tie_weights)

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
