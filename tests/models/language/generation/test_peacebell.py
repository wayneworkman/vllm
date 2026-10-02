# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check of Peacebell in vLLM against the checkpoint's own code.

The hub repos ship ``modeling_gpt.py`` / ``tokenization_gpt.py`` with a
KV-cached greedy ``chat_generate`` loop and the authoritative ChatML encoder
``build_chatml_ids``. That model class is not a ``GenerationMixin``, so the
generic HF-runner comparison cannot be used. This test instead:

* renders chats through ``LLM.chat`` (exercising the native
  ``PeacebellRenderer``) and asserts the prompt ids equal ``build_chatml_ids``;
* compares vLLM's greedy continuation and logprobs with the repo's
  ``chat_generate`` plus teacher-forced logits, via ``check_logprobs_close``.
"""

import pytest
import torch
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vllm import SamplingParams
from vllm.platforms import current_platform

from ...utils import check_logprobs_close

MODELS = [
    "wayneworkman2012/peacebell-v1-148M",
    "wayneworkman2012/peacebell-v1-291M",
]

CONVERSATIONS = [
    [{"role": "user", "content": "When did World War II end?"}],
    [{"role": "user", "content": "Who are you?"}],
    [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Who was Erwin Rommel?"},
    ],
    [
        {"role": "user", "content": "Tell me about the Battle of Midway."},
        {
            "role": "assistant",
            "content": "The Battle of Midway was fought in June 1942.",
        },
        {"role": "user", "content": "Explain that again, in more detail."},
    ],
]


def _hf_reference(
    model_name: str,
    conversations: list[list[dict[str, str]]],
    max_tokens: int,
    num_logprobs: int,
):
    """Greedy continuations and top-k logprobs from the repo's own code.

    Loaded through the repo's dynamic classes rather than ``AutoModel``:
    vLLM registers its in-tree ``PeacebellConfig`` for ``model_type="gpt"``,
    which transformers>=5 prefers over remote code, so ``AutoModelForCausalLM``
    would trip its config-class consistency check in the same process.
    """
    device = current_platform.device_type
    dtype = torch.bfloat16 if current_platform.is_cuda_alike() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config_cls = get_class_from_dynamic_module(
        "configuration_gpt.GPTConfig", model_name
    )
    model_cls = get_class_from_dynamic_module("modeling_gpt.GPTForCausalLM", model_name)
    model = (
        model_cls.from_pretrained(
            model_name, config=config_cls.from_pretrained(model_name), dtype=dtype
        )
        .to(device)
        .eval()
    )

    prompts: list[list[int]] = []
    outputs = []
    with torch.no_grad():
        for conversation in conversations:
            prompt_ids = tokenizer.build_chatml_ids(conversation)
            gen = model.chat_generate(
                torch.tensor(prompt_ids), max_new_tokens=max_tokens, temperature=0.0
            )
            logits = model(torch.tensor([prompt_ids + gen], device=device)).logits
            start = len(prompt_ids) - 1
            logprobs = torch.log_softmax(
                logits[0, start : start + len(gen)].float(), dim=-1
            )
            top = logprobs.topk(num_logprobs, dim=-1)
            per_position: list[dict[int, float]] = []
            for i, token_id in enumerate(gen):
                entry = {
                    int(t): float(lp) for lp, t in zip(top.values[i], top.indices[i])
                }
                entry[token_id] = float(logprobs[i, token_id])
                per_position.append(entry)
            prompts.append(prompt_ids)
            outputs.append((gen, tokenizer.decode_response(gen), per_position))

    del model
    current_platform.empty_cache()
    return prompts, outputs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_chat_matches_reference(
    vllm_runner,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    hf_prompts, hf_outputs = _hf_reference(
        model, CONVERSATIONS, max_tokens, num_logprobs
    )

    # The weights are at most 1.2 GB; a small fraction keeps the test runnable
    # on a shared GPU.
    with vllm_runner(
        model,
        max_model_len=2048,
        trust_remote_code=True,
        gpu_memory_utilization=0.3,
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.tokenizer_mode == "peacebell"
        params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, logprobs=num_logprobs
        )
        request_outputs = vllm_model.llm.chat(CONVERSATIONS, params)

    vllm_outputs = []
    for prompt_ids, request_output in zip(hf_prompts, request_outputs):
        assert list(request_output.prompt_token_ids) == prompt_ids
        output = request_output.outputs[0]
        token_ids = list(output.token_ids)
        logprobs = list(output.logprobs or [])
        # vLLM keeps the <|im_end|> stop token id in the output ids; the
        # reference loop stops before emitting it.
        if output.finish_reason == "stop" and token_ids[-1:] == [output.stop_reason]:
            token_ids, logprobs = token_ids[:-1], logprobs[:-1]
        vllm_outputs.append((token_ids, output.text, logprobs))

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
