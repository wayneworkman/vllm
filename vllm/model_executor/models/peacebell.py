# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Peacebell model compatible with the HuggingFace weights.

Peacebell (``wayneworkman2012/peacebell-v1-148M`` and ``-291M``) is a GPT-style
decoder trained from scratch: pre-norm RMSNorm, ReZero residual gates (one
learned scalar per sublayer), NeoX-style rotary position embeddings, bias-free
fused-QKV attention, an exact-GELU MLP and tied token / LM-head weights.
Ported from the ``modeling_gpt.py`` shipped inside the checkpoint repos; the
submodule names match that checkpoint's state_dict keys 1:1.
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import PeacebellConfig

from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class PeacebellAttention(nn.Module):
    def __init__(
        self,
        config: PeacebellConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.n_embd
        total_num_heads = config.n_head
        tp_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tp_size == 0
        self.num_heads = total_num_heads // tp_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5

        # Fused q/k/v projection, stored as one (3 * n_embd, n_embd) weight.
        self.c_attn = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.c_attn",
        )
        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.block_size,
            is_neox_style=True,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": float(config.rope_theta),
            },
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scale,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.c_proj(attn_output)
        return output


class PeacebellMLP(nn.Module):
    def __init__(
        self,
        config: PeacebellConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.n_embd
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            4 * hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.c_fc",
        )
        self.c_proj = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        # Exact (erf) GELU, as ``F.gelu`` in the reference implementation.
        self.act = get_act_fn("gelu")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class PeacebellBlock(nn.Module):
    def __init__(
        self,
        config: PeacebellConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = PeacebellAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.attn"
        )
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = PeacebellMLP(config, quant_config, prefix=f"{prefix}.mlp")
        # ReZero residual gates: x = x + attn_scale * attn(ln_1(x)), then
        # x = x + mlp_scale * mlp(ln_2(x)). The gated sublayer output is what
        # gets folded into the residual stream by the next fused norm.
        self.attn_scale = nn.Parameter(torch.zeros(1))
        self.mlp_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
        else:
            hidden_states, residual = self.ln_1(hidden_states, residual)
        hidden_states = self.attn(positions, hidden_states) * self.attn_scale
        hidden_states, residual = self.ln_2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states) * self.mlp_scale
        return hidden_states, residual


@support_torch_compile
class PeacebellModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.n_embd,
            quant_config=quant_config,
            prefix=f"{prefix}.wte",
        )
        self.start_layer, self.end_layer, self.h = make_layers(
            config.n_layer,
            lambda prefix: PeacebellBlock(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.h",
        )
        self.ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.n_embd
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.h, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states


class PeacebellForCausalLM(nn.Module, SupportsPP):
    # The checkpoint stores the decoder at the top level (``wte``, ``h.<i>``,
    # ``ln_f``); vLLM keeps it under ``model`` like every other decoder here.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "wte.": "model.wte.",
            "h.": "model.h.",
            "ln_f.": "model.ln_f.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = PeacebellModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.n_embd,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.wte)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ``lm_head.weight`` is tied to ``wte.weight`` and absent from the
        # checkpoint; the loader treats a tied alias as already loaded.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
