# Local brain: IFM/K2-Horizon-3.7B

Runs [K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B) on this
machine and exposes it as an **OpenAI-compatible API**, so the whole agent —
chat, task agent, memory, and the multi-agent swarm — can use it instead of the
openai-mocker. The mocker is untouched; switching is one setting.

```
  agent  ──OpenAI wire──►  serve_k2.py  ──►  k2_runtime.py  ──►  K2 weights
           :8100/v1          (FastAPI)        (transformers)     (E:\hf-models)
```

## Why not Ollama / LM Studio / llama.cpp?

They can't load it. K2-Horizon is a new architecture (`K2HorizonForCausalLM`,
"Mixture of Values" attention) and **upstream llama.cpp does not support it** —
support exists only in an [unmerged draft fork](https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon),
and the maintainers specifically flag the 3.7B as mismatched against llama.cpp's
pinned transformers version. So we run IFM's own reference implementation via
`trust_remote_code`, which is the only path that actually works today.

That also forced a separate venv: the reference code needs a modern
PyTorch + transformers 5.x, and the main agent venv is Python **3.14**, which
PyTorch does not support. `.venv-k2` is Python **3.12**.

## Setup (one time)

```powershell
py -3.12 -m venv .venv-k2
.venv-k2\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv-k2\Scripts\python.exe -m pip install "transformers>=5.13.0" accelerate bitsandbytes fastapi "uvicorn[standard]"
.venv-k2\Scripts\python.exe download_k2.py      # ~7.6 GB -> E:\hf-models
```

## Run

```powershell
agent\start_k2.bat          # serves http://127.0.0.1:8100/v1
```

Then pick the brain:

| How | What to do |
|---|---|
| Web chat | choose **brain: K2-Horizon (local)** in the dropdown |
| Whole process | set `LLM_PROFILE=local` before `start.bat` |
| One API call | `POST /chat` with `"llm": "local"` |
| **Back to the mocker** | drop `LLM_PROFILE` (default is `mock`) or pick **brain: mock** |

Nothing is pinned to K2 — `mock` remains the default everywhere.

## Fitting a 5B model on a 4 GB card

The "3.7B" counts non-embedding parameters; with the 250k-token vocabulary the
real total is ~5B, i.e. **10.1 GB at bf16**. On the RTX 2050 (4 GB) we therefore
run **NF4** (~2.8 GB).

The other budget is the KV cache, which costs **144 KiB/token** at bf16
(36 layers × 8 kv-heads × 128 head_dim × 2 for K+V × 2 bytes). That — not the
weights — is what limits how long the model may reason:

| Setting | Weights | KV cache | Total | Fits 4 GB? |
|---|---|---|---|---|
| 32k ceiling, bf16 KV | 2.8 GB | 4.5 GB | 7.3 GB | no |
| 16k ceiling, bf16 KV | 2.8 GB | 2.25 GB | 5.05 GB | no |
| **16k ceiling, int4 KV** | 2.8 GB | **0.6 GB** | **3.4 GB** | **yes** |

So we run a **16k ceiling with an int4 KV cache**, which is what makes
`reasoning_effort=high` affordable here. IFM recommends allowing 32k output
tokens so reasoning is never truncated; that does not fit on this GPU, but a
typical turn reasons for 1–4k tokens, well inside 16k.

If quantised KV turns out to be unavailable at runtime, the server logs a
warning and falls back to an unquantised cache rather than failing the request —
at which point lower `K2_MAX_CONTEXT` or `K2_REASONING_EFFORT` to avoid OOM.

Anything that doesn't fit spills to CPU via `accelerate`, which works but is
slow — keep `K2_GPU_MEM` just under your real VRAM.

### Why not stream layers from RAM instead?

Tempting (it's how people run 1.8T MoE models on 8 GB), but it's the wrong
lever here. Those demos rely on **MoE sparsity** — only ~3% of parameters are
active per token, so you stream just the live experts. K2-Horizon-3.7B is
**dense** (`num_experts: 0`, `mlp_only_layers` covers all 36), so every
parameter is touched for every token and there are no "important" layers to
prioritise. Worse, streaming targets the weights, which *already fit*; the
binding constraint is the KV cache, which streaming does not shrink. Crossing
PCIe once per token would cost seconds per token — hours for a single long
answer. Layer offloading becomes the right answer only for the 7B/32B variants,
where the weights genuinely exceed VRAM.

## Knobs (env vars)

| Var | Default | Meaning |
|---|---|---|
| `K2_MODEL_PATH` | `E:\hf-models\K2-Horizon-3.7B` | weights directory |
| `K2_QUANT` | `nf4` | `nf4` \| `int8` \| `bf16` |
| `K2_MAX_CONTEXT` | `16384` | prompt+output ceiling (sizes the KV cache) |
| `K2_MAX_NEW_TOKENS` | `12288` | default output budget per call |
| `K2_GPU_MEM` | `3.2GiB` | VRAM accelerate may use |
| `K2_REASONING_EFFORT` | `high` | `low` \| `medium` \| `high`; also per-request |
| `K2_KV_QUANT` | `int4` | `int4` \| `int8` \| `none` — what makes `high` fit |
| `K2_TOOL_CALL_FMT` | `xml` | `xml` \| `xml_typed` \| `json` |
| `K2_PORT` | `8100` | listen port |

## How much context each agent spends before it starts

Measured with the real `Toolkit.get_tool_schemas()` output rendered through the
template, because tool schemas are charged to the context whether or not the
model uses them:

| Agent | Tools | Baseline prompt | % of 16k | Left for work |
|---|---|---|---|---|
| task_manager | 29 | 2,930 | 17.9% | 13,454 |
| orchestrator | 4 | 2,490 | 15.2% | 13,894 |
| comms_agent | 36 | 4,123 | 25.2% | 12,261 |
| browser_agent | 33 | 4,623 | 28.2% | 11,761 |
| scripter | 35 | 6,522 | 39.8% | 9,862 |
| **Hermes-PM (single agent)** | **62** | **8,375** | **51.1%** | **8,009** |

Consequence worth knowing: **swarm mode suits the local model better than
single-agent mode.** Each specialist carries only its own tool subset, while
single-agent Hermes loads all 62 tools and spends half the window before the
conversation begins. "Left for work" still has to cover the dialogue, tool
results (a browser snapshot is ~1,500-2,000 tokens at the default
`BROWSER_MAX_OUTPUT`), and the model's own reasoning at `high` effort.

If single-agent mode runs out of room, in order of preference: use swarm mode,
raise `K2_MAX_CONTEXT` (int4 KV is cheap — 16k costs only ~0.6 GB, so 24k is
likely affordable on this card), or trim the toolkit.

## Tool calling

The template emits calls as markup, not OpenAI JSON, and `k2_runtime.parse_output`
converts them back:

```
<ifm|tool_calls><ifm|tool_call>ls
<ifm|arg_key>path</ifm|arg_key><ifm|arg_value>.</ifm|arg_value>
</ifm|tool_call></ifm|tool_calls>
```

One subtlety worth knowing if you touch the parser: the generation prompt ends
with an **open** `<ifm|think_faster>` tag, so the model's output *starts inside*
the reasoning block and only ever emits the closing tag. That's why
`parse_output` takes `starts_in_think`. Reasoning is returned separately as
`reasoning_content` and never leaks into the answer.

Run the parser tests (no model load, no GPU needed):

```powershell
.venv\Scripts\python.exe -m local_model.test_k2_parse
```

## Notes / limits

- **Requests are serialised.** One model, one GPU, one lock. The swarm's
  orchestrator + 4 children therefore queue rather than run in parallel —
  correct, but slower than the mocker on wide fan-out.
- **Streaming is fake.** The response is generated fully, then replayed as SSE,
  because tool calls can only be parsed once the markup block is complete. The
  agent runs `stream=False`, so this only affects curl.
- A 3.7B model is markedly weaker than ChatGPT at long multi-tool chains.
  Expect to keep `mock` for hard browser tasks and use `local` for fast,
  private, offline turns.
