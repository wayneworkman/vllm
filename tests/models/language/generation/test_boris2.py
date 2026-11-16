# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check of Boris2 in vLLM against the model's own code.

Boris2 (`boris2-peacebell-base` and `-think`) is the Peacebell architecture
with a byte-level BPE tokenizer. Each test takes its model, a hub id or a
local directory, from the environment and is skipped without it:

- `BORIS2_BASE_MODEL`: the base, text-continuation model.
- `BORIS2_CHAT_MODEL`: the chat model.

The reference is the model's own `modeling_gpt.py`, in float32 on the CPU.
It has no KV cache and no chat loop, so the checks are: identical prompt ids
(no BOS; a chat is rendered whole, then tokenized once), identical greedy
tokens, matching next-token logprobs, the untrained training-pad id never
offered, and a chat reply that ends at the plain-text `<|im_end|>`.
"""

import os

import pytest
import torch
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vllm import SamplingParams

BASE_MODEL = os.environ.get("BORIS2_BASE_MODEL")
CHAT_MODEL = os.environ.get("BORIS2_CHAT_MODEL")

IM_END = "<|im_end|>"
MAX_TOKENS = 32
NUM_LOGPROBS = 5
# The float32 Triton attention kernel takes its dot products in TF32.
LOGPROB_ATOL = 0.05

PROMPTS = [
    "The water cycle begins when",
    "Sam had 43 rubber bands. A friend gave Sam 19 more. Afterward, Sam had",
    "Once upon a time, in a small village by the sea,",
    "The three primary colors are red,",
    "def fibonacci(n):\n    ",
]

CONVERSATIONS = [
    [{"role": "user", "content": "What is the water cycle?"}],
    [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "  What is 2 + 2?  "},
        {"role": "assistant", "content": "4.\n"},
        {"role": "user", "content": "And 3 + 3?"},
    ],
    [
        {"role": "user", "content": "Write a haiku.\nAbout rain."},
        {"role": "assistant", "content": "Soft rain on the roof"},
        {"role": "user", "content": "Another."},
    ],
]


def _hf_reference(model_name: str, prompt_ids: list[list[int]]):
    """Greedy continuations and their top logprobs from the model's own code.

    Loaded through the repo's dynamic classes rather than `AutoModel`:
    vLLM registers its in-tree `PeacebellConfig` for `model_type="gpt"`, which
    transformers>=5 prefers over remote code, so `AutoModelForCausalLM` would
    trip its config-class consistency check in the same process.
    """
    config_cls = get_class_from_dynamic_module(
        "configuration_gpt.GPTConfig", model_name
    )
    model_cls = get_class_from_dynamic_module("modeling_gpt.GPTForCausalLM", model_name)
    config = config_cls.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name, config=config, dtype=torch.float32)
    model.eval()
    suppressed = list(model.generation_config.suppress_tokens or [])
    assert config.train_pad_id in suppressed

    outputs = []
    with torch.no_grad():
        for ids in prompt_ids:
            input_ids = torch.tensor([ids])
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                repetition_penalty=1.0,
            )[0, len(ids) :].tolist()
            logits = model(input_ids=torch.tensor([ids + generated])).logits[0]
            logits[:, suppressed] = -torch.inf
            start = len(ids) - 1
            logprobs = torch.log_softmax(logits[start : start + len(generated)], -1)
            top = logprobs.topk(NUM_LOGPROBS, dim=-1)
            outputs.append(
                (
                    generated,
                    [
                        dict(zip(indices.tolist(), values.tolist()))
                        for indices, values in zip(top.indices, top.values)
                    ],
                )
            )
    return config.train_pad_id, outputs


def _assert_matches_reference(request_output, prompt_ids, reference, train_pad_id):
    generated, reference_logprobs = reference
    output = request_output.outputs[0]

    assert list(request_output.prompt_token_ids) == prompt_ids
    assert list(output.token_ids) == generated
    for position, expected in zip(output.logprobs, reference_logprobs):
        assert train_pad_id not in position
        actual = {
            token_id: position[token_id].logprob
            for token_id in expected
            if token_id in position
        }
        assert len(actual) >= NUM_LOGPROBS - 1
        assert actual == pytest.approx(
            {token_id: expected[token_id] for token_id in actual}, abs=LOGPROB_ATOL
        )


@pytest.mark.skipif(BASE_MODEL is None, reason="BORIS2_BASE_MODEL is not set")
def test_base_matches_reference(vllm_runner) -> None:
    assert BASE_MODEL is not None
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    prompt_ids = [tokenizer(prompt).input_ids for prompt in PROMPTS]
    assert tokenizer.bos_token_id is None
    train_pad_id, references = _hf_reference(BASE_MODEL, prompt_ids)

    with vllm_runner(
        BASE_MODEL, max_model_len=2048, gpu_memory_utilization=0.1
    ) as vllm_model:
        llm = vllm_model.llm
        model_config = llm.llm_engine.model_config
        assert model_config.tokenizer_mode == "boris2"
        assert model_config.dtype == torch.float32
        # A text-continuation model has no chat template, hence no stop.
        assert llm.llm_engine.renderer.get_default_stop() == []
        request_outputs = llm.generate(
            PROMPTS,
            SamplingParams(
                temperature=0.0, max_tokens=MAX_TOKENS, logprobs=NUM_LOGPROBS
            ),
        )
        # The largest bias a request may ask for would make the untrained
        # training-pad id the greedy choice at every step if it were not masked.
        biased = llm.generate(
            PROMPTS[0],
            SamplingParams(
                temperature=0.0, max_tokens=4, logit_bias={train_pad_id: 100.0}
            ),
        )

    for request_output, ids, reference in zip(request_outputs, prompt_ids, references):
        _assert_matches_reference(request_output, ids, reference, train_pad_id)
    assert train_pad_id not in biased[0].outputs[0].token_ids


@pytest.mark.skipif(CHAT_MODEL is None, reason="BORIS2_CHAT_MODEL is not set")
def test_chat_renders_and_stops_like_reference(vllm_runner) -> None:
    assert CHAT_MODEL is not None
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
    prompt_ids = [
        tokenizer(
            tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            ),
            add_special_tokens=False,
        ).input_ids
        for conversation in CONVERSATIONS
    ]
    train_pad_id, references = _hf_reference(CHAT_MODEL, prompt_ids)

    with vllm_runner(
        CHAT_MODEL, max_model_len=2048, gpu_memory_utilization=0.1
    ) as vllm_model:
        llm = vllm_model.llm
        assert llm.llm_engine.model_config.tokenizer_mode == "boris2"
        assert llm.llm_engine.renderer.get_default_stop() == [IM_END]
        # ignore_eos also turns the default stop string off, which lets the
        # whole continuation be compared with the reference's.
        unstopped = llm.chat(
            CONVERSATIONS,
            SamplingParams(
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                logprobs=NUM_LOGPROBS,
                ignore_eos=True,
            ),
        )
        replies = llm.chat(
            CONVERSATIONS, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        )

    for request_output, ids, reference in zip(unstopped, prompt_ids, references):
        _assert_matches_reference(request_output, ids, reference, train_pad_id)

    # No stop was requested: the reply still ends where the model wrote the
    # marker, and the marker is not part of the text.
    for request_output, (generated, _) in zip(replies, references):
        output = request_output.outputs[0]
        reference_text = tokenizer.decode(generated)
        assert IM_END not in output.text
        if IM_END in reference_text:
            assert output.text == reference_text.split(IM_END)[0]
            assert output.finish_reason == "stop"
            assert output.stop_reason == IM_END
        else:
            assert list(output.token_ids) == generated
            assert output.finish_reason == "length"
