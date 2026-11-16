# Peacebell and Boris2 support in this vLLM fork

This fork of vLLM adds native support for two model families that upstream
vLLM does not know:

- Peacebell: `wayneworkman2012/peacebell-v1-148M` and
  `wayneworkman2012/peacebell-v1-291M`
- Boris2: `wayneworkman2012/boris2-peacebell-base` (text continuation) and
  `wayneworkman2012/boris2-peacebell-think` (chat)

All four are GPT-style decoders trained from scratch and published with
`trust_remote_code` model and config files. Boris2 is the Peacebell
architecture with a different tokenizer.

This document explains how to install and run the fork, what each model
family needs from an inference engine, every change the fork makes to vLLM
and why, how the result was verified, and how to carry the changes to a newer
vLLM.

The fork is upstream vLLM `main` at `4e3463c874` (vLLM 0.34.0rc1.dev156,
torch 2.14, transformers 5.22) plus the changes described here.

## 1. Installing

Install from a checkout of this fork, the same way as upstream vLLM:

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
VLLM_PRECOMPILED_WHEEL_COMMIT=4e3463c87493ebd2c5d5fbe4be6469911e23e2bf \
    uv pip install -e . --torch-backend=cu130
```

The fork's own commits have no published wheels, so the compiled extensions
come from the upstream commit the fork is based on. The fork changes Python
files only, so they match. `cu130` is the CUDA 13.0 build; section 10.1
explains why the fork is pinned to it.

To run the tests, add `uv pip install -r requirements/lint.txt pytest
pytest-asyncio tblib`.

## 2. Running the models

### 2.1 Peacebell

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

- `--trust-remote-code` is required, for the tokenizer only. The model and
  the config are in-tree.
- `--dtype auto` downcasts the fp32 checkpoint to bfloat16 (vLLM's rule for
  fp32 checkpoints on CUDA), which is also what the checkpoint's own
  `run_model.py` uses on CUDA. Pass `--dtype float32` to run the weights as
  stored.
- `--max-model-len` defaults to `block_size` (16384 or 32768). Lower it to
  save KV-cache memory.
- The OpenAI server's default `temperature` is 1.0, while `run_model.py`
  defaults to greedy. To make greedy the server-side default, pass
  `--override-generation-config '{"temperature": 0.0}'`.
- Offline: `LLM("wayneworkman2012/peacebell-v1-148M", trust_remote_code=True)`
  then `llm.chat(messages, SamplingParams(temperature=0))`.

### 2.2 Boris2

```bash
vllm serve wayneworkman2012/boris2-peacebell-think \
    --port 8002 --gpu-memory-utilization 0.1 --max-model-len 8192
```

```bash
curl -s localhost:8002/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role":"user","content":"What is the water cycle?"}],
  "temperature": 0.0
}'
```

- No `--trust-remote-code`, `--dtype` or `--tokenizer-mode` is needed, and
  clients do not have to send `stop`. A local model directory works in place
  of the hub id.
- The base model is a text-continuation model: use `/v1/completions` (or
  `LLM.generate`). It has no chat template, so a chat request fails with
  vLLM's usual `ChatTemplateResolutionError`.
- The fork bakes in no sampling defaults. vLLM reads only what the model's
  `generation_config.json` carries; with nothing there, the OpenAI server's
  own default applies (`temperature` 1.0). Set a server-side default with
  `--override-generation-config '{"temperature": 0.0}'`.
- `--max-model-len` defaults to the full 32,768-token context. Lower it to
  save KV-cache memory.

## 3. The models (what vLLM has to reproduce)

### 3.1 Peacebell

Source of truth: the files shipped inside each hub repo
(`configuration_gpt.py`, `modeling_gpt.py`, `tokenization_gpt.py`).

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
section 5.3).

### 3.2 Boris2

Source of truth: `configuration_gpt.py` and `modeling_gpt.py` inside each
model repo.

| Property | Boris2 (both models) |
| --- | --- |
| `architectures` / `model_type` | `["GPTForCausalLM"]` / `gpt`, the same as Peacebell |
| `n_embd` / `n_head` / head dim / `n_layer` | 960 / 15 / 64 / 11 |
| `block_size` (context) | 32768 |
| `vocab_size` | 32768 |
| parameters | 153,130,582 (tied head counted once) |
| weights on disk | fp32 safetensors, the same key names as Peacebell |

The blocks are Peacebell's, unchanged. What differs is everything around the
weights:

- **Tokenizer.** A byte-level BPE as a stock `PreTrainedTokenizerFast` over
  `tokenizer.json`. The only special token is `<|endoftext|>` (id 0), which is
  EOS and the tokenizer's pad. **There is no BOS**: a prompt starts at
  position 0 with its first text token.
- **Id 188 must never be emitted.** It is the token of the raw byte `0xFF`,
  which training used as padding. It was never a prediction target, so its
  output row is untrained. The model records it as `train_pad_id` in
  `config.json` and under `suppress_tokens` in `generation_config.json`.
- **Chat is ChatML as plain text.** `<|im_start|>` and `<|im_end|>` are not
  tokens; `<|im_end|>` is the seven ordinary tokens `<` `|` `im` `_` `end`
  `|` `>`. The Jinja template is in the chat model's `tokenizer_config.json`.
  The whole conversation is rendered and then tokenized in one call. A reply
  ends at the **string** `<|im_end|>` (or at EOS); the model is not trained to
  emit EOS after it, so without a stop string it runs on into an invented
  next turn.
- **Float32.** The weights are float32 and the model is meant to run in
  float32.
- The reference `modeling_gpt.py` has no KV cache; vLLM's paged attention
  replaces it, as it does for Peacebell.

## 4. Design summary

| Concern | vLLM mechanism used | Precedent in upstream |
| --- | --- | --- |
| Model weights and forward pass | new model file, registered under the hub's architecture name | every in-tree model |
| `model_type = "gpt"` config | in-tree `PreTrainedConfig` subclass registered in vLLM's config registry, so no remote config code is needed | `RWConfig`, `ChatGLMConfig` |
| Peacebell tokenizer | the hub's own `GPTTokenizer` through vLLM's standard cached HF tokenizer (needs `--trust-remote-code`) | Inkling (`"inkling": ("hf", "CachedHfTokenizer")`) |
| Peacebell chat rendering | a native renderer that emits token ids directly, selected by a tokenizer mode chosen automatically from the architecture | Inkling (`InklingRenderer`, `tokenizer_mode = "inkling"`) |
| Peacebell stop token | `generation_config.json` `eos_token_id: 4` is picked up by vLLM's default sampling params as a stop token id | standard vLLM behaviour |
| Boris2 chat rendering | the stock HF renderer and the model's Jinja template | every HF chat model |
| Boris2 stop string | a renderer hook that hands the engine default stop strings | new in this fork |
| Boris2 untrained token | masked in the model's `compute_logits` | new in this fork |

Both families arrive as `GPTForCausalLM` with `model_type = "gpt"`, so they
share the model class, the registry entries and the in-tree `PeacebellConfig`.
The fork tells them apart by one config field, in `vllm/config/model.py`:

```python
def _is_boris2(config):
    return config.model_type == "gpt" and getattr(config, "train_pad_id", None) is not None
```

A Peacebell `config.json` has no `train_pad_id`. What follows from that:

| Concern | Peacebell | Boris2 |
| --- | --- | --- |
| Tokenizer mode / renderer | `peacebell` / `PeacebellRenderer` (native token ids) | `boris2` / `Boris2Renderer` (the HF renderer and the Jinja template) |
| `--dtype auto` | bfloat16 (vLLM's rule for fp32 checkpoints) | float32 |
| `--trust-remote-code` | required, for the hub's `GPTTokenizer` | not needed: config in-tree, stock tokenizer |
| Ids masked in the logits | none | `[188]` |
| Default stop string | none (the end-of-turn marker is a token, listed in `generation_config.json`) | the end-of-turn marker as a string for a chat model, none for a base model |

## 5. Every change, file by file

### 5.1 New: `vllm/transformers_utils/configs/peacebell.py`

`PeacebellConfig(PreTrainedConfig)` with `model_type = "gpt"`. It mirrors the
hub's `GPTConfig` field for field (`vocab_size`, `n_embd`, `n_head`, `n_layer`,
`block_size`, `dropout`, `tie_weights`, `rezero_scale_init`, `rms_norm_eps`,
`rope_theta`, bos/eos/pad ids) and carries the same `attribute_map` so vLLM's
generic readers find `hidden_size`, `num_attention_heads`,
`num_hidden_layers` and `max_position_embeddings`. `tie_word_embeddings`
defaults to `tie_weights`.

The fields only one family has (`im_start_id` / `im_end_id` / `unk_pad_id`
for Peacebell, `train_pad_id` for Boris2) default to `None`. Each
`config.json` always states its own, because `save_pretrained` writes every
model-specific key.

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
`trust_remote_code=True`, so **in a process where vLLM has already loaded one
of these configs, `AutoModelForCausalLM.from_pretrained(repo,
trust_remote_code=True)` fails** with a config-class consistency error. Load
the reference model through the repo's own classes instead
(`transformers.dynamic_module_utils.get_class_from_dynamic_module`), as the
tests do. Running vLLM alone is unaffected.

### 5.2 New: `vllm/model_executor/models/peacebell.py`

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

Logit mask: `_suppress_token_ids(vllm_config)` returns the generation
config's `suppress_tokens` plus the config's `train_pad_id`, and
`compute_logits` sets those logits to `-inf`. The list is empty for Peacebell
and `[188]` for Boris2. vLLM does not honour `suppress_tokens` by itself:
with the mask taken out, a request with `logit_bias: {"188": 100}` gets id 188
at every step. Both sources are read so the mask survives
`--generation-config <other dir>`.

Registered in `vllm/model_executor/models/registry.py` (edit), twice:

```python
"GPTForCausalLM": ("peacebell", "PeacebellForCausalLM"),        # what the hub config says
"PeacebellForCausalLM": ("peacebell", "PeacebellForCausalLM"),  # alias for a future re-export
```

`GPTForCausalLM` is a generic name. If upstream ever claims it for another
model, the first line will conflict on a merge; the alias line exists so the
hub `config.json` can be switched to `"architectures": ["PeacebellForCausalLM"]`
at that point.

### 5.3 New: `vllm/renderers/peacebell_encoding.py` and `vllm/renderers/peacebell.py`

Why a text chat template cannot work for Peacebell. The SentencePiece model
was trained with `add_dummy_prefix=True`, so every `encode` call starts with
a `▁`. Measured on the real tokenizer:

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

### 5.4 New: `vllm/renderers/boris2.py`, and the default stop string

`Boris2Renderer` subclasses `HfRenderer` and changes nothing about rendering:
the Jinja template is applied and the result tokenized in one call with no
special tokens added, which is exactly what training did. Its one addition is
`get_default_stop()`, computed once by `plain_text_end_of_turn(tokenizer)`:
`["<|im_end|>"]` when the tokenizer's own chat template contains `<|im_end|>`
and the vocabulary has no such token, else `[]`. The stop string is derived
from the model's files, not hardcoded per model name, and a base model (no
template) gets none.

Getting that string to the engine takes three small edits to upstream files:

- `vllm/renderers/base.py`: `BaseRenderer.get_default_stop()`, returning `[]`.
- `vllm/sampling_params.py`: `SamplingParams.add_default_stop(stop)`. It
  appends to the request's own `stop` (no duplicates) and updates
  `output_text_buffer_length`. It does nothing under `ignore_eos` or
  `detokenize=False`, mirroring how a generation-config EOS id is treated.
- `vllm/v1/engine/input_processor.py`: one call,
  `sampling_params.add_default_stop(self.renderer.get_default_stop())`, right
  after `update_from_generation_config`. Every request passes through here
  (`/v1/chat/completions`, `/v1/completions`, `LLM.chat`, `LLM.generate`), so
  none of the protocol classes needed a change.

Why there and not in `generation_config.json`: a `stop_strings` entry there
makes transformers refuse every `generate()` call that passes no tokenizer,
so the model does not ship one, and vLLM reads no stop strings from a
generation config anyway.

### 5.5 Tokenizer modes `peacebell` and `boris2`

- `vllm/tokenizers/registry.py` (edit): `"peacebell": ("hf",
  "CachedHfTokenizer")` and `"boris2": ("hf", "CachedHfTokenizer")` in
  `_VLLM_TOKENIZERS`. Token operations (encode for `/v1/completions`, decode,
  incremental detokenization) use each model's own HF tokenizer unchanged.
  The modes exist to select the renderer.
- `vllm/renderers/registry.py` (edit): `"peacebell": ("peacebell",
  "PeacebellRenderer")` and `"boris2": ("boris2", "Boris2Renderer")`.
- `vllm/config/model.py` (edits): both names added to the `TokenizerMode`
  literal, and in `ModelConfig.__post_init__` the automatic mode selection
  gains

  ```python
  elif arch in ("GPTForCausalLM", "PeacebellForCausalLM"):
      self.tokenizer_mode = "boris2" if _is_boris2(hf_config) else "peacebell"
  ```

  so neither family needs `--tokenizer-mode`.

The Peacebell hub tokenizer is a slow (Python) tokenizer, so vLLM logs its
usual "Using a slow tokenizer" warning and uses `SlowIncrementalDetokenizer`.
That path decodes through `convert_ids_to_tokens` /
`convert_tokens_to_string`, which the hub class implements over SentencePiece;
byte-fallback pieces (`<0x0A>` for newline) decode correctly.

### 5.6 Dtype

`vllm/config/model.py`, `_get_and_verify_dtype` (edit): `dtype="auto"` on a
float32 Boris2 checkpoint resolves to float32 instead of being downcast to
bfloat16. An explicit `--dtype bfloat16` is still obeyed. The measurement
behind this is in section 6.2. Peacebell keeps vLLM's default.

### 5.7 Tests

- New `tests/renderers/test_peacebell.py`: hermetic layout tests with a
  character-level fake tokenizer (id stream shape, no newline after
  `<|im_end|>`, separate encode calls per segment, generation prompt,
  `continue_final_message`, error cases), plus a parity test that loads the
  real hub tokenizer and asserts the renderer equals `build_chatml_ids` on
  several conversations, with and without the generation prompt.
- New `tests/models/language/generation/test_peacebell.py`: end to end.
  Renders chats through `LLM.chat` (so the real renderer runs), asserts the
  prompt ids equal `build_chatml_ids`, and compares vLLM's greedy tokens and
  top-5 logprobs against the repo's own `chat_generate` loop plus
  teacher-forced logits with `check_logprobs_close`, for both models. The
  hub model class is not a `GenerationMixin` (no `generate()`), which is why
  the generic HF-runner comparison in `test_common.py` is not used.
- New `tests/renderers/test_boris2.py`: when `plain_text_end_of_turn` yields
  the stop string, on real `tokenizers`-backed tokenizers built in the test.
- `tests/test_sampling_params.py` (edit): `add_default_stop` merges with the
  request's stops, does not duplicate, and is skipped under `ignore_eos` and
  `detokenize=False`.
- New `tests/models/language/generation/test_boris2.py`: end to end against
  the model's own code in float32 on the CPU. It takes the models from
  `BORIS2_BASE_MODEL` and `BORIS2_CHAT_MODEL` (a hub id or a local directory)
  and is skipped without them. It asserts: prompt ids equal the tokenizer's
  (no BOS; a chat equals `apply_chat_template` tokenized once); 32 greedy
  tokens identical; top-5 logprobs within 0.05; id 188 absent from the
  logprobs and not produced even under a +100 `logit_bias`; float32 and
  tokenizer mode `boris2` chosen automatically; and that a chat reply
  requested with no `stop` ends where the reference wrote `<|im_end|>`, with
  `stop_reason == "<|im_end|>"` and the marker not in the text.
- `tests/models/registry.py` (edit): `_HfExamplesInfo` entries for
  `GPTForCausalLM` (default 148M, extra `291m`) and for the
  `PeacebellForCausalLM` alias (via `hf_overrides={"architectures": [...]}`),
  both with `tokenizer_mode="peacebell"` and `trust_remote_code=True`. These
  feed `tests/models/test_registry.py` and
  `tests/models/test_initialization.py`.

### 5.8 Docs

- `docs/models/supported_models.md` (edit): one row in the text-generation
  table for `GPTForCausalLM`, `PeacebellForCausalLM` (Peacebell, Boris2), PP
  supported, LoRA not wired.
- This file.

### 5.9 Complete list of touched paths

```text
new      PEACEBELL_FORK_CHANGES.md
new      tests/models/language/generation/test_boris2.py
new      tests/models/language/generation/test_peacebell.py
new      tests/renderers/test_boris2.py
new      tests/renderers/test_peacebell.py
new      vllm/model_executor/models/peacebell.py
new      vllm/renderers/boris2.py
new      vllm/renderers/peacebell.py
new      vllm/renderers/peacebell_encoding.py
new      vllm/transformers_utils/configs/peacebell.py
edited   docs/models/supported_models.md
edited   tests/models/registry.py
edited   tests/test_sampling_params.py
edited   vllm/config/model.py
edited   vllm/model_executor/models/registry.py
edited   vllm/renderers/base.py
edited   vllm/renderers/registry.py
edited   vllm/sampling_params.py
edited   vllm/tokenizers/registry.py
edited   vllm/transformers_utils/config.py
edited   vllm/transformers_utils/configs/__init__.py
edited   vllm/v1/engine/input_processor.py
```

Every edit to an existing file is a short insertion next to an existing
entry. To see the fork's whole footprint against upstream:

```bash
git diff 4e3463c87493ebd2c5d5fbe4be6469911e23e2bf HEAD --stat
```

## 6. Verification

All results below are on the current base: one NVIDIA GPU, torch 2.14.0+cu130,
transformers 5.22.2, vLLM defaults (torch.compile and CUDA graphs on).

### 6.1 Peacebell

Reference: the hub's own code, loaded via `get_class_from_dynamic_module`,
`chat_generate(temperature=0)`, in the same dtype as vLLM. The four
conversations of `tests/models/language/generation/test_peacebell.py`
(single turn, identity question, system prompt, three-turn history), 64
greedy tokens each:

| Model | dtype (both sides) | Prompt ids equal `build_chatml_ids` | Greedy outputs identical to the reference | Largest logprob gap (top 5) |
| --- | --- | --- | --- | --- |
| 148M | bfloat16 (the default) | 4 / 4 | 1 / 4; the others diverge at tokens 37, 13 and 2 | 0.250 |
| 291M | bfloat16 (the default) | 4 / 4 | 4 / 4 | 0.373 |
| 148M | float32 (`--dtype float32`) | 4 / 4 | 4 / 4 | 0.028 |
| 291M | float32 (`--dtype float32`) | 4 / 4 | 4 / 4 | 0.026 |

In float32 the port reproduces the reference token for token. The 148M
divergences in bfloat16 are at positions where the reference's own top-1 to
top-2 margin is one or two bfloat16 steps (0.0625 to 0.125 nats), so they are
ties broken by kernel differences (paged attention and a fused RMSNorm versus
SDPA and a plain norm), not a modelling difference; the bfloat16 gaps are
multiples of the bfloat16 logit spacing. The float32 gap is not zero because
upstream's float32 attention kernel takes its dot products in TF32
(section 6.2).

`vllm serve wayneworkman2012/peacebell-v1-291M --trust-remote-code`, checked
by script: `/v1/models`; `/v1/chat/completions` greedy
(`' World War II ended in 1945.'`, `finish_reason: stop`), with a system
prompt and history, streaming (same text), sampling with
`top_p`/`top_k`/`seed`; `/tokenize` on messages (15 ids, identical to
`build_chatml_ids`); `/v1/completions`; 16 concurrent chat requests, all
finished with `stop`.

### 6.2 Boris2

Reference: the model's own `modeling_gpt.py`, loaded through its dynamic
classes, float32, on the CPU, `generate(do_sample=False)` for 32 new tokens,
plus teacher-forced logits with id 188 masked.

Model numerics, base model, six text prompts:

| Check | vLLM float32 (the default) | vLLM `--dtype bfloat16` |
| --- | --- | --- |
| Prompt ids equal the tokenizer's, no BOS | 6 / 6 | 6 / 6 |
| 32 greedy tokens identical to the reference | 6 / 6 | 4 / 6 (diverges at token 21, and at token 0) |
| Top-1 next token identical | 6 / 6 | 5 / 6 |
| Largest logprob gap, top 20, all positions | 0.010 | 0.100 |
| Id 188 in any returned logprob | never | never |

bfloat16 does not reproduce the reference's greedy tokens, which is why
float32 is the automatic dtype. The think model is the same architecture and
runs the same code path.

Where the remaining 0.010 comes from: with `--attention-backend
FLEX_ATTENTION` the same float32 run agrees with the reference to **0.00003**
on every prompt. The default float32 backend, `TRITON_ATTN`, calls `tl.dot`
with Triton's default input precision, which is TF32 on GPUs with tensor
cores. That is upstream behaviour for every float32 model and was left alone;
the port itself (weights, RoPE, norms, gates, GELU, head, mask) is exact. Use
`--attention-backend FLEX_ATTENTION` for float32 that is closer to bit-exact;
it was not benchmarked.

Chat rendering and the default stop, which depend on the tokenizer files and
the renderer rather than on the weights, on three conversations (single
turn, system prompt with history, multi-line content):

- Prompt ids equal `apply_chat_template` followed by one tokenize call: 3 / 3.
- With no `stop` in the request, a reply the reference continued as
  `4.<|im_end|>\n<|im_start|>user\n...` came back from vLLM as `4.`, with
  `finish_reason: stop` and `stop_reason: '<|im_end|>'`.
- A reply that never completes the marker runs to `max_tokens`, as the
  reference does.

`vllm serve` on a chat model without `--trust-remote-code`, checked by
script: `/v1/models`; `/v1/chat/completions` with no `stop` (ends at the
marker); the same streamed (same text, no partial marker); with a request
`stop` of its own (that one fires); `/tokenize` on messages (identical to
`apply_chat_template`); `/v1/completions` with `logit_bias: {"188": 100}`
(normal text, no BOS in the prompt); 16 concurrent sampled chats (no marker in
any text).

Both end-to-end Boris2 tests were also run once with the logit mask and the
default-stop call removed, and failed for exactly those reasons.

### 6.3 Test commands

```bash
pytest tests/renderers/test_peacebell.py tests/renderers/test_boris2.py \
    tests/test_sampling_params.py                                      # 100 passed
pytest tests/models/language/generation/test_peacebell.py              # 2 passed
BORIS2_BASE_MODEL=<base model> BORIS2_CHAT_MODEL=<chat model> \
    pytest tests/models/language/generation/test_boris2.py             # 2 passed
pytest tests/models/test_registry.py tests/models/test_initialization.py \
    -k "GPTForCausalLM or PeacebellForCausalLM"                        # 4 passed
pre-commit run --files <all touched files>                             # clean
```

## 7. Feature coverage

| Feature | Status |
| --- | --- |
| Continuous batching / concurrent requests | Verified (16 parallel chat requests) |
| Prefix caching, chunked prefill | On by default; verified running |
| torch.compile + CUDA graphs | On by default on GPU; verified |
| Streaming, sampling params (`temperature`, `top_p`, `top_k`, `seed`), `logprobs`, `stop`, `max_tokens` | Verified on GPU |
| `/v1/chat/completions`, `/v1/completions`, `/tokenize`, `/v1/models` | Verified |
| `--dtype float32` | Verified for both families; the default for Boris2 |
| Tensor parallelism, pipeline parallelism | Layers are the parallel ones and `SupportsPP` is declared; untested |
| Quantization (e.g. FP8 on the fly) | Layers accept `quant_config`; untested |
| Speculative decoding, structured outputs | Generic vLLM machinery; untested |
| LoRA | Not wired (`SupportsLoRA` not implemented) |
| Tool calling, multimodal input | Not supported; the models were not trained for tools and are text-only |
| Rust frontend (`VLLM_USE_RUST_FRONTEND=1`) | Not supported: it has no Peacebell renderer and no default-stop hook. Use the default Python frontend. |
| Scale-out `/render` and `/derender` endpoints | Boris2's default stop is not applied there; they evaluate stop strings from the request alone |
| CPU backend | See section 9 |

## 8. Behaviour to know about

### 8.1 Peacebell, compared with the checkpoint's own code

- **Leading space.** vLLM's output text begins with a space
  (`' World War II ended in 1945.'`). The first generated token carries the
  SentencePiece `▁` dummy prefix, and vLLM's incremental detokenizer renders it
  as a space, exactly as it does for Llama-family models. The checkpoint's
  `decode_response` strips it. Clients should `strip()`.
- **`<unk>` ban.** The checkpoint's `stream_generate` sets the logit of id 0
  to `-inf` on every step. vLLM does not do that for Peacebell. With greedy
  decoding the model does not emit `<unk>`; a client that samples can add
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

### 8.2 Boris2

- **The default stop.** A reply that reaches `<|im_end|>` comes back with
  `finish_reason: "stop"` and `stop_reason: "<|im_end|>"`; the marker is not
  in the text and, when streaming, never appears half-written (vLLM holds
  back the last nine characters until it knows). A request's own `stop`
  entries apply in addition. `ignore_eos: true` turns the default stop off
  together with EOS.
- It applies to `/v1/completions` on a chat model as well, which is what a
  client that renders ChatML itself wants.
- It is derived from the tokenizer's own chat template. A **base** model
  served with an external `--chat-template` gets no default stop: pass
  `stop: ["<|im_end|>"]` in that case.
- vLLM rejects prompts longer than `max_model_len`.

## 9. CPU backend

Peacebell was verified on vLLM's CPU backend on the fork's previous base
(upstream `721d0e5c11`). **It has not been re-verified on the current base,
and Boris2 has not been run on the CPU backend.** The numbers below are from
that earlier base: a 16-thread x86 CPU with AVX2 and no AVX-512, bfloat16,
four conversations.

| Model | Prompt ids equal `build_chatml_ids` | Greedy outputs identical to the reference (CPU, bf16) | Offline throughput |
| --- | --- | --- | --- |
| 148M | 4 / 4 | 1 / 4; the others diverge at tokens 37, 13 and 61 | 102 tok/s single stream; 178 tok/s aggregate over 8 batched prompts |
| 291M | 4 / 4 | 4 / 4 | 60 tok/s single stream; 110 tok/s aggregate over 8 batched prompts |

The 148M divergences are ties in the reference's own bf16 logits, so they are
numeric noise, not a modelling difference. Sampling (`temperature=0.7`,
`top_p`, `top_k`, `seed`) works and is what the batched figures measure.

Known limitation on that base: a request with `logprobs` hung the CPU engine
until `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` expired. It reproduced with stock
`facebook/opt-125m`, so it is the CPU backend's logprob path, not these
models.

Installing for CPU (use a separate checkout from any CUDA install, see
section 10.4):

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu \
VLLM_PRECOMPILED_WHEEL_COMMIT=<upstream commit with a CPU wheel> \
    uv pip install --editable . --torch-backend cpu
```

Three things the precompiled CPU path did not do by itself on that base:

1. `setup.py` extracted a CUDA-centric list of `.so` files from the wheel and
   skipped `vllm/_C_AVX2.abi3.so`, `vllm/_C_AVX512.abi3.so` and
   `vllm/libs/libtcmalloc_minimal.so.4`. Without `_C_AVX2` the engine dies
   with `'_C' object has no attribute 'init_cpu_memory_env'`. Unzip those
   members from the wheel into the checkout's `vllm/`.
2. The resolver installed a CUDA-linked `torchcodec`. Replace it:
   `uv pip install --reinstall --no-deps torchcodec --index-url https://download.pytorch.org/whl/cpu`.
3. On a machine that has an NVIDIA driver, `triton-cpu` selects its NVIDIA
   backend even under CPU-only torch. Run with `TRITON_DEFAULT_BACKEND=cpu`.

Serving on CPU:

```bash
export TRITON_DEFAULT_BACKEND=cpu VLLM_CPU_KVCACHE_SPACE=4   # GiB of KV cache
vllm serve wayneworkman2012/peacebell-v1-291M --trust-remote-code --dtype bfloat16 \
    --port 8002 --max-model-len 4096 --max-num-seqs 16
```

## 10. Maintaining the fork

### 10.1 The upstream base and CUDA wheels

The fork is merged with upstream at `4e3463c874`, the last upstream commit
that publishes CUDA 13.0 (`cu130`) wheels. The next upstream commit,
`3996083d37` (*Move the default CUDA build from 13.0 to 13.2*), switched the
published CUDA 13 wheels to `cu132`.

With an NVIDIA driver that supports CUDA 13.0 but not 13.2, a `cu132` wheel
loads the model and then fails at the first attention call with
`CUDA error: the provided PTX was compiled with an unsupported toolchain`
(`cudaErrorUnsupportedPtxVersion`), from the precompiled `_vllm_fa2_C`
kernel. After the switch, upstream publishes no `cu130` wheel, and torch 2.14
has no `cu129` build. A `cu130` wheel from before the switch must not be
paired with later source: `csrc/` and the pinned kernel projects changed.

Staying at `4e3463c874` keeps the fork installable on CUDA 13.0 drivers with
source and binaries that match exactly.

### 10.2 Merging a newer upstream

1. Decide the CUDA variant. Anything after `4e3463c874` needs `cu132` wheels,
   hence a driver with CUDA 13.2 support, or a source build.
2. Merge:

   ```bash
   git remote add upstream https://github.com/vllm-project/vllm.git
   git fetch upstream main --no-tags
   git merge upstream/main
   ```

   Fetch the upstream release tag the new base descends from as well, or the
   version string is computed from a stale tag.
3. Install into a fresh virtual environment (section 1) with the new commit
   and variant, then go through section 10.4.
4. Walk the checklist in section 10.3 and rerun section 6.

As a guide to the size of a merge: bringing the fork from `721d0e5c11` to
`4e3463c874` (3,168 upstream commits) needed two textual conflicts, both in
dicts that upstream had re-sorted (`vllm/renderers/registry.py` and
`vllm/transformers_utils/configs/__init__.py`), and one source change,
`vllm.entrypoints.chat_utils` having moved to `vllm.renderers.chat_utils`.
Upstream's docstring hook had also started to require single backticks.
Everything else merged and ran unchanged.

### 10.3 Porting checklist

The new files are self-contained; the edits are small. These are the places
upstream refactors most often:

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
   in the Peacebell renderer the same way.
5. `vllm/renderers/registry.py`, `vllm/tokenizers/registry.py`: dict shape.
6. `vllm/config/model.py`: the `TokenizerMode` literal, the automatic
   tokenizer-mode block (search for `"inkling"`), and
   `_get_and_verify_dtype`, which must still resolve `"auto"` in one place.
7. `vllm/transformers_utils/config.py` `_CONFIG_REGISTRY` and
   `vllm/transformers_utils/configs/__init__.py` `_CLASS_TO_MODULE` /
   `__all__`.
8. `tests/models/registry.py` `_HfExamplesInfo` fields, and
   `tests/conftest.py` `VllmRunner` (the end-to-end tests use
   `vllm_runner(...)` and `vllm_model.llm.chat`).
9. transformers: the Peacebell hub `GPTTokenizer` subclasses
   `PreTrainedTokenizer` (`PythonBackend` in transformers 5). If a
   transformers bump breaks it, the fix belongs in the checkpoint repo; vLLM
   only calls `get_vocab`, `.sp.encode`, `encode`, `decode`,
   `convert_ids_to_tokens`, `convert_tokens_to_string`, `get_added_vocab`.
10. The registry name `GPTForCausalLM`: if upstream takes it, switch the hub
    `config.json` to `PeacebellForCausalLM` and drop the first registry line.
11. The default stop: `vllm/v1/engine/input_processor.py` must still call
    `sampling_params.update_from_generation_config(...)` once per request
    (the default-stop call sits right after it); `vllm/sampling_params.py`
    must still keep `stop` as a list and `output_text_buffer_length`;
    `vllm/renderers/hf.HfRenderer` must still take `(config, tokenizer)`.
12. The logit mask: the model's `compute_logits` must still be what the
    sampler reads.

### 10.4 Install pitfalls

- **One editable install per checkout.** A precompiled install writes its
  compiled `.so` files into the source tree, so a second install into the
  same checkout (a CPU build beside a CUDA build, for example) overwrites or
  shadows the first one's binaries. Use a separate checkout or git worktree
  for each.
- **A newer wheel does not remove what an older one extracted.** After
  moving a checkout to a newer base, files from the old wheel can remain and
  shadow new modules; stale `vllm/third_party/deep_gemm/_C.cpython-3*.so`
  files, for instance, shadow a newer wheel's `deep_gemm/_C.py` and break
  that import with an undefined symbol. List git-ignored files that the last
  install did not rewrite, and delete them:

  ```bash
  git ls-files --others --ignored --exclude-standard \
      | grep -v '__pycache__\|_cache/\|\.egg-info/' \
      | while read -r f; do [ "$(date -r "$f" +%F)" != "$(date +%F)" ] && echo "$f"; done
  ```

- **uv caches the build.** Re-running `uv pip install -e .` with a different
  wheel does not re-extract the `.so` files unless you add
  `--reinstall-package vllm --refresh-package vllm`.
- **`VLLM_USE_PRECOMPILED=1` needs a published wheel** for the pinned commit
  and for a CUDA variant the driver can run. The wheel index for one commit
  is `https://wheels.vllm.ai/<commit>/<variant>/vllm/metadata.json`.
