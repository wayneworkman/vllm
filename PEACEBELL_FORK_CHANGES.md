# Peacebell support in this vLLM fork

This fork adds native support for the two released Peacebell models:

- `wayneworkman2012/peacebell-v1-148M`
- `wayneworkman2012/peacebell-v1-291M`

Both are GPT-style decoders trained from scratch, published with
`trust_remote_code` model, config and tokenizer files. Upstream vLLM knows
nothing about them. This document records every change made to the fork so
the same support can be re-applied to a future vLLM version, explains why each
change is shaped the way it is, and lists the checks that prove it works.

Base commit of the fork when this was done: `721d0e5c11` (upstream `main`,
September 2026, transformers 5.17, torch 2.13).

## 1. What the model is (what vLLM has to reproduce)

Source of truth: the files shipped inside each hub repo
(`configuration_gpt.py`, `modeling_gpt.py`, `tokenization_gpt.py`), which are
byte-identical copies of `make_your_own_llm/model_packaging/model_assets/`.

| Property | 148M | 291M |
| --- | --- | --- |
| `architectures` in `config.json` | `["GPTForCausalLM"]` | same |
| `model_type` | `gpt` | same |
| `n_embd` / `n_head` / head dim | 960 / 15 / 64 | 1280 / 20 / 64 |
| `n_layer` | 11 | 13 |
| `block_size` (context) | 16384 | 32768 |
| `vocab_size` | 28000 | 28000 |
| weights on disk | fp32 safetensors | fp32 safetensors |

Architecture, per block, all Linears bias-free:

```text
x = x + attn_scale * Attn(RMSNorm_1(x))      # attn_scale: learned scalar (ReZero)
x = x + mlp_scale  * MLP(RMSNorm_2(x))       # mlp_scale:  learned scalar (ReZero)
```

- RMSNorm, eps `1e-5`, reduction in fp32.
- Attention: fused `c_attn` weight of shape `(3*n_embd, n_embd)` split q, k, v
  in that order; `c_proj` output projection; NeoX-style rotary embeddings
  (`rotate_half` over `cat(freqs, freqs)`), theta `1e6`, full head dim.
- MLP: `c_fc` (`n_embd -> 4*n_embd`), exact erf GELU (`F.gelu` default),
  `c_proj` back down.
- `lm_head.weight` is tied to `wte.weight` and is not stored in the checkpoint.
- Final `ln_f` RMSNorm.
- State-dict keys are top-level: `wte.weight`, `h.<i>.ln_1.weight`,
  `h.<i>.attn.c_attn.weight`, `h.<i>.attn.c_proj.weight`, `h.<i>.ln_2.weight`,
  `h.<i>.mlp.c_fc.weight`, `h.<i>.mlp.c_proj.weight`, `h.<i>.attn_scale`,
  `h.<i>.mlp_scale`, `ln_f.weight`. No `transformer.` or `model.` prefix.

Tokenizer: a SentencePiece unigram model (`sentencepiece.model`) wrapped by the
repo's `GPTTokenizer` (a slow, Python `PreTrainedTokenizer`). Special ids:
`0 <unk>/<pad>`, `1 <s>`, `2 </s>`, `3 <|im_start|>`, `4 <|im_end|>`. The
ChatML control tokens are SentencePiece *user-defined symbols*, not HF added
tokens. `generation_config.json` sets `eos_token_id: 4` (`<|im_end|>`), while
`config.json` keeps `eos_token_id: 2`.

Chat format, produced by `GPTTokenizer.build_chatml_ids` and nothing else:

```text
for each message:  [3] + encode(role + "\n") + encode(content) + [4]
generation prompt: [3] + encode("assistant\n")
```

There is **no newline after `<|im_end|>`**, and `role + "\n"` and `content`
are encoded by **separate** `SentencePieceProcessor.encode` calls. That detail
is why vLLM needs a native renderer instead of a Jinja chat template (see
section 3.3).

## 2. Design summary

| Concern | vLLM mechanism used | Precedent in upstream |
| --- | --- | --- |
| Model weights and forward pass | new model file, registered under the hub's architecture name | every in-tree model |
| `model_type = "gpt"` config | in-tree `PretrainedConfig` subclass registered in vLLM's config registry, so no remote config code is needed | `NemotronConfig`, `RWConfig`, `ChatGLMConfig` |
| Tokenizer | the hub's own `GPTTokenizer` through vLLM's standard cached HF tokenizer (needs `--trust-remote-code`) | Inkling (`"inkling": ("hf", "CachedHfTokenizer")`) |
| Chat rendering | a native renderer that emits token ids directly, selected by a new tokenizer mode `peacebell` that is chosen automatically from the architecture | Inkling (`InklingRenderer`, `tokenizer_mode = "inkling"`) |
| Stop token | `generation_config.json` `eos_token_id: 4` is picked up by vLLM's default sampling params as a stop token id | standard vLLM behaviour |

`--trust-remote-code` is required, for the tokenizer only. The model and the
config are in-tree.

## 3. Every change, file by file

### 3.1 New: `vllm/transformers_utils/configs/peacebell.py`

`PeacebellConfig(PretrainedConfig)` with `model_type = "gpt"`. It mirrors the
hub's `GPTConfig` field for field (`vocab_size`, `n_embd`, `n_head`, `n_layer`,
`block_size`, `dropout`, `tie_weights`, `rezero_scale_init`, `rms_norm_eps`,
`rope_theta`, `im_start_id`, `im_end_id`, `unk_pad_id`, bos/eos/pad ids) and
carries the same `attribute_map` so vLLM's generic readers find
`hidden_size`, `num_attention_heads`, `num_hidden_layers` and
`max_position_embeddings`. `tie_word_embeddings` defaults to `tie_weights`.

Registered in two places (both edits):

- `vllm/transformers_utils/configs/__init__.py`: `"PeacebellConfig":
  "vllm.transformers_utils.configs.peacebell"` in `_CLASS_TO_MODULE`, and
  `"PeacebellConfig"` appended to `__all__`.
- `vllm/transformers_utils/config.py`: `gpt="PeacebellConfig"` added to
  `_CONFIG_REGISTRY`.

Why in-tree rather than remote code: vLLM's `get_config` checks
`_CONFIG_REGISTRY` first and loads a registered class with
`trust_remote_code=False`, so the config path is deterministic and does not
depend on the hub file or on transformers' dynamic-module machinery. This is
the documented reason for `vllm/transformers_utils/configs/` to exist.

Consequence to know about: vLLM registers the class with
`AutoConfig.register("gpt", PeacebellConfig)`. transformers 5 prefers an
externally registered config class over a repo's remote code even when
`trust_remote_code=True`, so **in a process where vLLM has already loaded a
Peacebell config, `AutoModelForCausalLM.from_pretrained(repo,
trust_remote_code=True)` fails** with a config-class consistency error. Load
the reference model through the repo's own classes instead
(`transformers.dynamic_module_utils.get_class_from_dynamic_module`), as the
tests do. Running vLLM alone is unaffected.

### 3.2 New: `vllm/model_executor/models/peacebell.py`

`PeacebellForCausalLM` (with `SupportsPP`), `PeacebellModel`,
`PeacebellBlock`, `PeacebellAttention`, `PeacebellMLP`. Ported from the hub's
`modeling_gpt.py` onto vLLM layers:

| Hub module | vLLM layer | Notes |
| --- | --- | --- |
| `wte` (`nn.Embedding`) | `VocabParallelEmbedding` | |
| `attn.c_attn` | `QKVParallelLinear(n_embd, head_dim, n_head, bias=False)` | the fused `(3*n_embd, n_embd)` weight loads as-is; vLLM splits it per TP rank |
| `attn.c_proj` | `RowParallelLinear(bias=False)` | |
| RoPE | `get_rope(head_dim, max_position=block_size, is_neox_style=True, rope_parameters={"rope_type": "default", "rope_theta": rope_theta})` | |
| SDPA | `Attention(num_heads, head_dim, head_dim**-0.5)` | paged KV cache, any attention backend |
| `mlp.c_fc` | `ColumnParallelLinear(n_embd, 4*n_embd, bias=False)` | |
| `mlp.c_proj` | `RowParallelLinear(4*n_embd, n_embd, bias=False)` | |
| `F.gelu` | `get_act_fn("gelu")` | exact erf GELU, not `gelu_new` |
| `ln_1`, `ln_2`, `ln_f` | `RMSNorm(n_embd, eps=rms_norm_eps)` | fused add-and-norm path |
| `attn_scale`, `mlp_scale` | `nn.Parameter(torch.zeros(1))` | loaded by name from the checkpoint |
| `lm_head` | `ParallelLMHead` tied via `lm_head.tie_weights(model.wte)` | |

The block uses vLLM's fused residual convention: each `RMSNorm` call takes
`(hidden_states, residual)` and returns `(normed, residual + hidden_states)`.
The ReZero gate is applied to the sublayer output before it is handed to the
next norm, which is algebraically the reference computation.

Weight loading: `AutoWeightsLoader` with a `WeightsMapper` that prefixes the
checkpoint's top-level keys (`wte.`, `h.`, `ln_f.`) with `model.`. The tied
`lm_head.weight` is absent from the checkpoint; the loader treats a tied alias
as already loaded, and would skip it if a future export wrote it.

Registered in `vllm/model_executor/models/registry.py` (edit), twice:

```python
"GPTForCausalLM": ("peacebell", "PeacebellForCausalLM"),        # what the hub config says
"PeacebellForCausalLM": ("peacebell", "PeacebellForCausalLM"),  # alias for a future re-export
```

`GPTForCausalLM` is a generic name. If upstream ever claims it for another
model, the first line will conflict on rebase; the alias line exists so the
hub `config.json` can be switched to `"architectures": ["PeacebellForCausalLM"]`
at that point.

### 3.3 New: `vllm/renderers/peacebell_encoding.py` and `vllm/renderers/peacebell.py`

Why a text chat template cannot work. The SentencePiece model was trained with
`add_dummy_prefix=True`, so every `encode` call starts with a `▁`. Measured on
the real tokenizer:

```text
encode("user\n") + encode("When did World War II end?")
  -> [27888, 506, 209, 1492, 621, 982, 775, 1010, 959, 27925]     # training-time ids
encode("user\nWhen did World War II end?")
  -> [27888, 506, 209, 27922, 1103, 621, 982, 775, 1010, 959, 27925]  # different
encode("<|im_start|>") -> [27888, 3]                              # stray ▁ before the control id
```

A Jinja template renders one string and tokenizes it, which gives the second
form, not the first. The model only ever saw the first form.

`peacebell_encoding.py` is a pure module (no vLLM imports, mirrors
`inkling_encoding.py`) with `render_peacebell_messages(messages, tokenizer,
add_generation_prompt=True, continue_final_message=False)`. It implements the
exact `build_chatml_ids` algorithm against a three-method protocol
(`im_start_id`, `im_end_id`, `encode_text`). Text content parts are joined;
non-text parts are rejected (the model is text only).
`continue_final_message` leaves the last message open (no `<|im_end|>`, no
generation prompt).

`peacebell.py` has `PeacebellRenderer(BaseRenderer)`, modeled on
`InklingRenderer`. It resolves the control ids from the tokenizer vocab and
encodes text through the hub tokenizer's `.sp` (its
`SentencePieceProcessor`) when present, because that is exactly what training
did; it falls back to `tokenizer.encode(text, add_special_tokens=False)`, which
differs only for text containing the literal strings `<s>` or `</s>` (HF
splits those as added tokens). Messages go through vLLM's standard
`parse_chat_messages(content_format="string")` first, so multi-part text
content is flattened the same way as for every other model.

Registered in `vllm/renderers/registry.py` (edit):

```python
"peacebell": ("peacebell", "PeacebellRenderer"),
```

### 3.4 Tokenizer mode `peacebell`

- `vllm/tokenizers/registry.py` (edit): `"peacebell": ("hf", "CachedHfTokenizer")`
  in `_VLLM_TOKENIZERS`. Token operations (encode for `/v1/completions`,
  decode, incremental detokenization) use the hub's `GPTTokenizer` unchanged.
  The mode exists to select the renderer.
- `vllm/config/model.py` (two edits): `"peacebell"` added to the
  `TokenizerMode` literal, and in `ModelConfig.__post_init__` the automatic
  mode selection gains

  ```python
  elif arch in ("GPTForCausalLM", "PeacebellForCausalLM"):
      self.tokenizer_mode = "peacebell"
  ```

  so `vllm serve wayneworkman2012/peacebell-v1-291M --trust-remote-code` needs
  no further flags. `--tokenizer-mode peacebell` sets it explicitly.

The hub tokenizer is a slow (Python) tokenizer, so vLLM logs its usual
"Using a slow tokenizer" warning and uses `SlowIncrementalDetokenizer`. That
path decodes through `convert_ids_to_tokens` / `convert_tokens_to_string`,
which the hub class implements over SentencePiece; byte-fallback pieces
(`<0x0A>` for newline) decode correctly.

### 3.5 Tests

- New `tests/renderers/test_peacebell.py`: hermetic layout tests with a
  character-level fake tokenizer (id stream shape, no newline after
  `<|im_end|>`, separate encode calls per segment, generation prompt,
  `continue_final_message`, error cases), plus a parity test that loads the
  real hub tokenizer and asserts the renderer equals `build_chatml_ids` on
  several conversations, with and without the generation prompt.
- New `tests/models/language/generation/test_peacebell.py`: end-to-end.
  Renders chats through `LLM.chat` (so the real renderer runs), asserts the
  prompt ids equal `build_chatml_ids`, and compares vLLM's greedy tokens and
  top-5 logprobs against the repo's own `chat_generate` loop plus
  teacher-forced logits with `check_logprobs_close`, for both models. The
  hub model class is not a `GenerationMixin` (no `generate()`), which is why
  the generic HF-runner comparison in `test_common.py` is not used. The test
  asks for `gpu_memory_utilization=0.3` because the weights are tiny and the
  GPU on the development machine is shared.
- `tests/models/registry.py` (edit): `_HfExamplesInfo` entries for
  `GPTForCausalLM` (default 148M, extra `291m`) and for the
  `PeacebellForCausalLM` alias (via `hf_overrides={"architectures": [...]}`),
  both with `tokenizer_mode="peacebell"` and `trust_remote_code=True`. These
  feed `tests/models/test_registry.py` and
  `tests/models/test_initialization.py`.

### 3.6 Docs

- `docs/models/supported_models.md` (edit): one row in the text-generation
  table for `GPTForCausalLM`, `PeacebellForCausalLM` (Peacebell), PP supported,
  LoRA not wired.
- This file.

### 3.7 Complete list of touched paths

```text
new      PEACEBELL_FORK_CHANGES.md
new      tests/models/language/generation/test_peacebell.py
new      tests/renderers/test_peacebell.py
new      vllm/model_executor/models/peacebell.py
new      vllm/renderers/peacebell.py
new      vllm/renderers/peacebell_encoding.py
new      vllm/transformers_utils/configs/peacebell.py
edited   docs/models/supported_models.md
edited   tests/models/registry.py
edited   vllm/config/model.py
edited   vllm/model_executor/models/registry.py
edited   vllm/renderers/registry.py
edited   vllm/tokenizers/registry.py
edited   vllm/transformers_utils/config.py
edited   vllm/transformers_utils/configs/__init__.py
```

Every edit to an existing file is a one-to-six line insertion next to an
existing entry; `git diff` on those files shows them all.

## 4. Running it

```bash
vllm serve wayneworkman2012/peacebell-v1-291M --trust-remote-code \
    --port 8002 --gpu-memory-utilization 0.2 --max-model-len 8192
```

```bash
curl -s localhost:8002/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role":"user","content":"When did World War II end?"}],
  "temperature": 0.0
}'
```

- `--dtype auto` downcasts the fp32 checkpoint to bfloat16 (vLLM's rule for
  fp32 checkpoints on CUDA), matching what `serve.py` in the packaging repo
  does. Pass `--dtype float32` to run the master weights as stored.
- `--max-model-len` defaults to `block_size` (16384 or 32768). Lower it to
  save KV-cache memory.
- The OpenAI server's default `temperature` is 1.0. To make greedy the
  server-side default the way `serve.py` does, pass
  `--override-generation-config '{"temperature": 0.0}'`.
- Offline: `LLM("wayneworkman2012/peacebell-v1-148M", trust_remote_code=True)`
  then `llm.chat(messages, SamplingParams(temperature=0))`.

## 5. Verification performed (2026-09-24, RTX Pro 6000 Blackwell, shared GPU)

Reference for every comparison: the hub's own code, loaded via
`get_class_from_dynamic_module`, in bfloat16, `chat_generate(temperature=0)`.

Compiled mode (torch.compile plus CUDA graphs), six conversations each
(single turn, system prompt, three-turn history, off-topic question):

| Model | Prompt ids equal `build_chatml_ids` | Greedy outputs identical to HF | Largest teacher-forced logprob gap |
| --- | --- | --- | --- |
| 148M | 6 / 6 | 4 / 6 (the other two diverge at tokens 13 and 61 into an equally good phrasing) | 0.124 |
| 291M | 6 / 6 | 5 / 6 (the other diverges at token 10) | 0.065 |

The divergences are bfloat16 kernel differences (paged attention and fused
RMSNorm versus SDPA and a plain norm); they are within the tolerance vLLM's own
model tests use.

Test suite, all passing in the fork's dev venv:

```bash
pytest tests/renderers/test_peacebell.py                                   # 31 passed
pytest tests/models/language/generation/test_peacebell.py                  # 2 passed
pytest tests/models/test_registry.py tests/models/test_initialization.py \
    -k "GPTForCausalLM or PeacebellForCausalLM"                            # 4 passed
pytest tests/models/test_registry.py -k "test_hf_registry_coverage or test_registry_imports"
    # 376 passed; the single failure is upstream's KananaV module importing
    # `timm`, which the dev venv does not have. Unrelated to Peacebell.
pre-commit run --files <all touched files>                                 # clean
pre-commit run mypy-3.12 --hook-stage manual --files <touched vllm/ files> # clean
```

`vllm serve` with the 291M model, checked by hand: `/v1/models`,
`/v1/chat/completions` greedy and with a system prompt and history, streaming
(31 chunks, `finish_reason: stop`), sampling with `top_p`/`top_k`,
`/tokenize` on messages (15 ids, identical to `build_chatml_ids`),
`/v1/completions` on a raw string, and 16 concurrent chat requests (all
succeeded through vLLM's continuous batching; the GPU was shared with
another serving job, so no throughput figure is quoted).

## 6. Feature coverage

| Feature | Status |
| --- | --- |
| Continuous batching / concurrent requests | Verified (16 parallel chat requests on GPU and on CPU) |
| Prefix caching, chunked prefill | On by default; verified running |
| torch.compile + CUDA graphs | On by default on GPU; verified |
| Streaming, sampling params (`temperature`, `top_p`, `top_k`, `seed`), `logprobs`, `stop`, `max_tokens` | Verified on GPU; `logprobs` hangs on the CPU backend (see section 7) |
| `/v1/chat/completions`, `/v1/completions`, `/tokenize`, `/v1/models` | Verified |
| `--dtype float32` | Supported (fp32 master weights); not benchmarked |
| Tensor parallelism, pipeline parallelism | Layers are the parallel ones and `SupportsPP` is declared; untested (one GPU here) |
| Quantization (e.g. FP8 on the fly) | Layers accept `quant_config`; untested |
| Speculative decoding, structured outputs | Generic vLLM machinery; untested |
| LoRA | Not wired (`SupportsLoRA` not implemented) |
| Tool calling, multimodal input | Not supported; the model was not trained for tools and is text-only |
| Rust frontend (`VLLM_USE_RUST_FRONTEND=1`) | Not supported: its renderer selection has no Peacebell entry and falls back to the Jinja path. Use the default Python frontend. |

## 7. CPU backend

Peacebell also runs on vLLM's CPU backend. Verified 2026-09-24 on an
i5-12600K (16 threads, AVX2 only, no AVX-512), bfloat16, with the same four
conversations as section 5:

| Model | Prompt ids equal `build_chatml_ids` | Greedy outputs identical to HF (CPU, bf16) | Offline throughput |
| --- | --- | --- | --- |
| 148M | 4 / 4 | 1 / 4; the others diverge at tokens 37, 13 and 61 | 102 tok/s single stream; 178 tok/s aggregate over 8 batched prompts |
| 291M | 4 / 4 | 4 / 4 | 60 tok/s single stream; 110 tok/s aggregate over 8 batched prompts |

The 148M divergences are ties in the reference's own bf16 logits (gap 0.25
nats between `,` and `.` at token 37, and exactly 0.000 nats between
`World` and `the` at token 13), so they are numeric noise, not a modelling
difference. Sampling (`temperature=0.7`, `top_p`, `top_k`, `seed`) works and is
what the batched figures above measure.

`vllm serve` on CPU with the 291M model (startup about 90 s): a 9-token greedy
chat answer in 0.24 s, multi-turn with a system prompt, streaming (31 chunks),
8 concurrent 64-token requests at 118 tok/s aggregate and 16 concurrent at
126 tok/s aggregate, with the scheduler running all 16 together. These
numbers were taken with the other GPU job's CPU load present.

Known limitation: any request with `logprobs` hangs the CPU engine until
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (300 s) expires. This reproduces with
stock `facebook/opt-125m`, so it is the CPU backend's triton-based logprob
kernels (`vllm/v1/worker/gpu/sample/logprob.py`) on this machine, not
Peacebell. Do not request `logprobs` on CPU. The hang also leaves the
`VLLM::Worker` process spinning after the engine gives up; kill it, or every
later CPU run on the machine is several times slower.

CPU setup (all of these were needed; see section 10 for the venv layout):

```bash
# separate checkout for the CPU build: a git worktree inside the CPU venv
uv venv --python 3.12 /faster/python_virtual_environments/vllm_peacebell_cpu
git worktree add --detach /faster/python_virtual_environments/vllm_peacebell_cpu/src HEAD
# copy the uncommitted Peacebell changes into it (git diff | git apply, plus the new files)
cd /faster/python_virtual_environments/vllm_peacebell_cpu/src
VIRTUAL_ENV=/faster/python_virtual_environments/vllm_peacebell_cpu \
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu \
VLLM_PRECOMPILED_WHEEL_COMMIT=7f1a5398e9610d96c473931a26c0e12bbe0d0423 \
    uv pip install --editable . --torch-backend cpu
```

Then three fixes the precompiled CPU path does not do for you:

1. `setup.py` extracts a CUDA-centric list of `.so` files from the wheel and
   skips `vllm/_C_AVX2.abi3.so`, `vllm/_C_AVX512.abi3.so` and
   `vllm/libs/libtcmalloc_minimal.so.4`. Without `_C_AVX2` the engine dies
   with `'_C' object has no attribute 'init_cpu_memory_env'`. Download the
   wheel (its URL is in `https://wheels.vllm.ai/<commit>/cpu/vllm/metadata.json`)
   and `unzip` those three members into the checkout's `vllm/`.
2. The resolver installs a CUDA-linked `torchcodec` that fails on
   `libnvrtc.so.13`. Replace it:
   `uv pip install --reinstall --no-deps torchcodec --index-url https://download.pytorch.org/whl/cpu`.
3. On a machine that has an NVIDIA driver, `triton-cpu` selects its NVIDIA
   backend even under CPU-only torch, and Inductor then asserts
   "Torch not compiled with CUDA enabled". Run with
   `TRITON_DEFAULT_BACKEND=cpu`.

The CPU wheel commit (`7f1a5398e`) is older than the CUDA one because the
nightly CPU build lags; the only native change between it and HEAD is a
compile-time AVX10.2 gate, irrelevant on AVX2 hardware.

Serving on CPU:

```bash
export TRITON_DEFAULT_BACKEND=cpu VLLM_CPU_KVCACHE_SPACE=4   # GiB of KV cache
/faster/python_virtual_environments/vllm_peacebell_cpu/bin/vllm serve \
    wayneworkman2012/peacebell-v1-291M --trust-remote-code --dtype bfloat16 \
    --port 8002 --max-model-len 4096 --max-num-seqs 16
```

## 8. Behaviour differences from `model_packaging/serve.py`

- **Leading space.** vLLM's output text begins with a space
  (`' World War II ended in 1945.'`). The first generated token carries the
  SentencePiece `▁` dummy prefix, and vLLM's incremental detokenizer renders it
  as a space, exactly as it does for Llama-family models. `serve.py` strips it
  via `decode_response(...).strip()`. Clients should `strip()`.
- **`<unk>` ban.** `stream_generate` sets the logit of id 0 to `-inf` on every
  step. vLLM does not have a per-model logit mask. With greedy decoding the
  model does not emit `<unk>`; a client that samples can add
  `"logit_bias": {"0": -100}` to a request.
- **Stop tokens.** vLLM stops on `<|im_end|>` (from `generation_config.json`)
  and on `</s>` (the tokenizer's `eos_token`). The stop token is not part of
  the returned text. In offline `RequestOutput` objects the `<|im_end|>` id is
  the last element of `token_ids` with `stop_reason == 4`, which is vLLM's
  convention for stop ids that are not the tokenizer's EOS.
- **Raw `/v1/completions`.** Works mechanically but the model was trained on
  ChatML only, so raw continuations are poor. Use chat completions.
- **Context limit.** `stream_generate` silently truncates prompts to the last
  `block_size - 1` tokens; vLLM rejects prompts longer than `max_model_len`.
- **Multi-part content.** OpenAI `content: [{"type": "text", ...}]` arrays are
  accepted and concatenated; image or audio parts are rejected.

## 9. Porting checklist for a future vLLM version

Re-create the seven new files (they are self-contained) and re-apply the seven
small edits. Then check each of these, because they are the places upstream
refactors most often:

1. `vllm/model_executor/models/utils.py`: `AutoWeightsLoader` constructor
   arguments and `WeightsMapper` field names (`orig_to_new_prefix`).
   The loader's automatic handling of tied parameters is what lets
   `load_weights` stay one line.
2. `vllm/model_executor/layers/rotary_embedding/get_rope`: the
   `rope_parameters` dict format (`rope_type`, `rope_theta`).
3. `vllm/model_executor/layers/layernorm.RMSNorm`: the `(x, residual)`
   fused-add calling convention.
4. `vllm/renderers/base.BaseRenderer`: the abstract `render_messages` /
   `render_messages_async` signatures and `parse_dec_only_prompt`. Diff the
   Inkling renderer between the two versions; whatever changed there changes
   here the same way.
5. `vllm/renderers/registry.py`, `vllm/tokenizers/registry.py`: dict shape.
6. `vllm/config/model.py`: the `TokenizerMode` literal and the automatic
   tokenizer-mode block (search for `"inkling"`).
7. `vllm/transformers_utils/config.py` `_CONFIG_REGISTRY` and
   `vllm/transformers_utils/configs/__init__.py` `_CLASS_TO_MODULE` /
   `__all__`.
8. `tests/models/registry.py` `_HfExamplesInfo` fields, and
   `tests/conftest.py` `VllmRunner` (the e2e test uses `vllm_runner(...)` and
   `vllm_model.llm.chat`).
9. transformers: the hub `GPTTokenizer` subclasses `PreTrainedTokenizer`
   (`PythonBackend` in transformers 5). If a transformers bump breaks it, fix
   the tokenizer in `make_your_own_llm/model_packaging/model_assets/` and
   re-export; vLLM only calls `get_vocab`, `.sp.encode`, `encode`, `decode`,
   `convert_ids_to_tokens`, `convert_tokens_to_string`, `get_added_vocab`.
10. The registry name `GPTForCausalLM`: if upstream takes it, switch the hub
    `config.json` to `PeacebellForCausalLM` and drop the first registry line.

Then rerun sections 5 and 7.

## 10. Development environment used

- CUDA venv: `/faster/python_virtual_environments/vllm_peacebell` (Python 3.12,
  torch 2.13.0+cu130, transformers 5.17.0, editable install of this checkout).
- CPU venv: `/faster/python_virtual_environments/vllm_peacebell_cpu`, whose
  editable install points at the git worktree
  `/faster/python_virtual_environments/vllm_peacebell_cpu/src` (section 7).
  It carries a copy of the uncommitted Peacebell changes; after committing,
  re-sync it with `git -C <worktree> checkout <commit>` or remove it with
  `git worktree remove`.
- **Never point two editable installs at one checkout.** Each precompiled
  install writes its compiled `.so` files into the source tree, and the CPU
  wheel's `vllm/_C.abi3.so` shadowed the CUDA build (the CUDA build has no
  `_C.abi3.so`; its ops live in `_C_stable_libtorch.abi3.so`), which crashed
  the CUDA venv with an illegal instruction until the file was deleted.
  Re-running `uv pip install -e .` does not re-extract: uv reuses its cached
  build. To repair, delete the stray files and unzip the right members from
  the wheel.
- `VLLM_USE_PRECOMPILED=1` needs a published wheel for the checkout's commit.
  Commits that touch only ROCm (like the fork's base) are not built, so pin
  the nearest built commit:

  ```bash
  VIRTUAL_ENV=/faster/python_virtual_environments/vllm_peacebell \
  VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=dcfc17e0b1cb4a4bb4eeca9a75c82b82d776f25e \
      uv pip install -e . --torch-backend=auto
  ```

  Find a built commit at `https://wheels.vllm.ai/nightly/cu130/vllm/metadata.json`
  (or `.../nightly/cpu/...` for the CPU wheel).
- Extra packages for the tests and hooks: `requirements/lint.txt`, `pytest`,
  `pytest-asyncio`, `tblib`, `protobuf` (only for inspecting the SentencePiece
  proto).
