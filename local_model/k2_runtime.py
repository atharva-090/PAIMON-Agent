"""Load and drive IFM/K2-Horizon-3.7B locally (transformers + bitsandbytes).

Why this exists: K2-Horizon is a novel architecture (`K2HorizonForCausalLM`,
"Mixture of Values" attention) that upstream llama.cpp does NOT support yet, so
Ollama/LM Studio/llama-server cannot serve it. The only working runtime is IFM's
own reference code, loaded via `trust_remote_code`. This module wraps that into
something `serve_k2.py` can expose as an OpenAI-compatible endpoint.

Hardware reality on this box (RTX 2050, 4 GB VRAM): the weights are ~5B params
counting the 250k-token embedding table (10.1 GB at bf16), and the KV cache
costs ~144 KB/token. So we quantise to NF4 (~2.8 GB) and keep the context
modest; anything left over spills to CPU via accelerate. IFM's recommended
"reasoning_effort=high + 32k output tokens" needs ~4.6 GB of KV cache on its
own and simply does not fit here -- hence K2_REASONING_EFFORT defaults to low.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

# --- Configuration (all overridable by env) --------------------------------
MODEL_PATH = os.environ.get("K2_MODEL_PATH", r"E:\hf-models\K2-Horizon-3.7B")
QUANT = os.environ.get("K2_QUANT", "nf4").lower()          # nf4 | int8 | bf16
MAX_CONTEXT = int(os.environ.get("K2_MAX_CONTEXT", "16384"))
# Worst-case output budget per call. This is a LATENCY guard, not a quality
# one: generation is ~4 tok/s here and requests serialise on one GPU, so a
# 12k budget let a single runaway reply monopolise the server for ~50 minutes
# and stall every other agent behind it. 2048 caps that at ~8 min. Raise it if
# you genuinely need very long reasoning and can wait.
MAX_NEW_TOKENS = int(os.environ.get("K2_MAX_NEW_TOKENS", "2048"))
GPU_MEM = os.environ.get("K2_GPU_MEM", "3.2GiB")
CPU_MEM = os.environ.get("K2_CPU_MEM", "10GiB")
# Where the (unquantisable, 1.28 GB) embedding table lives: "cpu" keeps it off
# the GPU at the cost of one host->device copy per forward. Set 0 for a big GPU.
EMBED_DEVICE = os.environ.get("K2_EMBED_DEVICE", "cpu")
if EMBED_DEVICE.isdigit():
    EMBED_DEVICE = int(EMBED_DEVICE)
# Quantise lm_head too (1.28 GB -> 0.33 GB). Needed to fit 4 GB; costs a little
# output quality since it is the final projection over the 250k vocab.
QUANT_LM_HEAD = os.environ.get("K2_QUANT_LM_HEAD", "1") not in ("0", "", "false")
# high -> <ifm|think>, medium -> <ifm|think_fast>, low -> <ifm|think_faster>
REASONING_EFFORT = os.environ.get("K2_REASONING_EFFORT", "high")

# KV cache quantisation: none | int8 | int4.
# This is what makes reasoning_effort=high affordable. The cache costs
# 36 layers * 8 kv-heads * 128 head_dim * 2 (K+V) * 2 bytes = 144 KiB/token at
# bf16, i.e. 2.25 GiB at our 16k ceiling -- which does not fit alongside 2.8 GB
# of NF4 weights on a 4 GB card. int4 cuts it to ~0.6 GiB and it does.
KV_QUANT = os.environ.get("K2_KV_QUANT", "int4").lower()
KV_QUANT_BACKEND = os.environ.get("K2_KV_QUANT_BACKEND", "quanto")
# Tool encodings the template understands. NOTE: the kwargs are spelled
# `tool_call_format` / `tool_presentation_format` -- the `_fmt` names are the
# template's INTERNAL variables, and passing those is silently ignored. We keep
# the template's own defaults (xml / markdown), which is what the model was
# trained on; json is available if you prefer stricter parsing.
TOOL_CALL_FMT = os.environ.get("K2_TOOL_CALL_FMT", "xml")
TOOL_PRESENTATION_FMT = os.environ.get("K2_TOOL_PRESENTATION_FMT", "markdown")
MODEL_ID = os.environ.get("K2_MODEL_ID", "k2-horizon-3.7b")

# --- Literal markers emitted by IFM's chat template -------------------------
THINK_BLOCKS = ("think", "think_fast", "think_faster")
_THINK_RE = re.compile(
    r"<ifm\|(" + "|".join(THINK_BLOCKS) + r")>(.*?)</ifm\|\1>", re.DOTALL
)
_THINK_CLOSE_RE = re.compile(r"</ifm\|(?:" + "|".join(THINK_BLOCKS) + r")>")
_TOOL_CALLS_RE = re.compile(r"<ifm\|tool_calls>(.*?)</ifm\|tool_calls>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.DOTALL)
_ARG_PAIR_RE = re.compile(
    r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
    r"(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
    r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
    re.DOTALL,
)


@dataclass
class Generation:
    """One decoded assistant turn, already split into its parts."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


# ===========================================================================
#  Model loading
# ===========================================================================
class K2Runtime:
    """Holds the loaded model+tokenizer. One instance per process."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()   # generate() is not re-entrant
        self.device = "cpu"
        self.dtype_name = "float32"
        # Flipped off permanently if the installed transformers/quanto cannot
        # do quantised KV, so we degrade instead of failing every request.
        self._kv_quant_ok = KV_QUANT not in ("none", "", "off")

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.path.isdir(MODEL_PATH):
            raise RuntimeError(
                f"model not found at {MODEL_PATH} -- run download_k2.py first, "
                f"or set K2_MODEL_PATH"
            )

        has_cuda = torch.cuda.is_available()
        # RTX 2050 is Ampere (GA107, sm_86) despite the 20-series name, so bf16
        # is native -- but detect rather than assume, for other machines.
        if has_cuda and torch.cuda.is_bf16_supported():
            compute_dtype, self.dtype_name = torch.bfloat16, "bfloat16"
        elif has_cuda:
            compute_dtype, self.dtype_name = torch.float16, "float16"
        else:
            compute_dtype, self.dtype_name = torch.float32, "float32"

        print(f"[k2] loading {MODEL_PATH}", flush=True)
        print(
            f"[k2] cuda={has_cuda} quant={QUANT} compute_dtype={self.dtype_name}",
            flush=True,
        )
        if has_cuda:
            props = torch.cuda.get_device_properties(0)
            print(
                f"[k2] gpu={props.name} vram={props.total_memory / 2**30:.1f}GiB",
                flush=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True,
        )

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if not has_cuda:
            kwargs["device_map"] = {"": "cpu"}
            kwargs["dtype"] = compute_dtype
        else:
            # Explicit placement rather than device_map="auto", which fills the
            # GPU in module order and strands whatever comes last on the CPU.
            #
            # The 250k-token vocabulary makes the two vocab-sized tensors
            # dominate: embed_tokens and lm_head are 1.28 GB EACH at bf16,
            # against 1.95 GB for all 36 decoder layers in NF4. bitsandbytes
            # cannot quantise an nn.Embedding, so embed_tokens stays bf16 --
            # and it is far cheaper to keep on the CPU (one gather per forward)
            # than lm_head would be (a 2560x250624 matmul on EVERY token).
            # "" is the fallback; longer keys win, so this means
            # "everything on the GPU except the embedding table".
            kwargs["device_map"] = {"": 0, "model.embed_tokens": EMBED_DEVICE}

            if QUANT in ("nf4", "int8"):
                from transformers import BitsAndBytesConfig

                common = {
                    # Required as soon as ANY module lives on the CPU, else
                    # bnb refuses the mixed placement outright.
                    "llm_int8_enable_fp32_cpu_offload": True,
                    # By default transformers skips lm_head when quantising.
                    # We cannot afford its 1.28 GB, so quantise it too ([] =
                    # skip nothing) unless asked otherwise.
                    "llm_int8_skip_modules": [] if QUANT_LM_HEAD else None,
                }
                if QUANT == "nf4":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        **common,
                    )
                else:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True, **common,
                    )
            else:
                kwargs["dtype"] = compute_dtype

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
        self.model.eval()
        self.device = "cuda" if has_cuda else "cpu"

        # accelerate keeps CPU-offloaded params as META placeholders and streams
        # the real data in through forward hooks. PreTrainedModel.device reports
        # the FIRST parameter's device -- which here is the offloaded
        # embed_tokens -- so model.device comes back as "meta". generate() then
        # builds its eos/pad tensors on meta and dies with "Tensor.item() cannot
        # be called on meta tensors". Report the actual compute device instead:
        # every module that runs kernels is on the GPU, and the offload hook
        # moves the embedding's data by itself.
        if has_cuda and any(p.device.type == "meta"
                            for p in self.model.parameters()):
            real_device = torch.device("cuda:0")
            base = type(self.model)
            self.model.__class__ = type(
                base.__name__ + "_K2FixedDevice", (base,),
                {"device": property(lambda _self: real_device)},
            )
            print(f"[k2] patched model.device -> {real_device} "
                  f"(embedding is cpu-offloaded via meta)", flush=True)
        print(
            f"[k2] ready. max_context={MAX_CONTEXT} "
            f"reasoning_effort={REASONING_EFFORT}",
            flush=True,
        )

    # -------------------------------------------------------------------
    #  Prompt building
    # -------------------------------------------------------------------
    def build_prompt(self, messages: list[dict], tools: list[dict] | None,
                     reasoning_effort: str | None = None) -> str:
        """Render OpenAI-shaped messages through IFM's chat template."""
        effort = reasoning_effort or REASONING_EFFORT
        norm = [_normalise_message(m, effort) for m in messages]
        tmpl_tools = _normalise_tools(tools)

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "reasoning_effort": effort,
            "tool_call_format": TOOL_CALL_FMT,
            "tool_presentation_format": TOOL_PRESENTATION_FMT,
        }
        if tmpl_tools:
            kwargs["tools"] = tmpl_tools

        try:
            return self.tokenizer.apply_chat_template(norm, **kwargs)
        except TypeError:
            # A template that rejects one of our extra kwargs: retry with only
            # the universally-supported ones.
            for key in ("tool_call_format", "tool_presentation_format",
                        "reasoning_effort"):
                kwargs.pop(key, None)
            return self.tokenizer.apply_chat_template(norm, **kwargs)

    # -------------------------------------------------------------------
    #  Generation
    # -------------------------------------------------------------------
    def _cache_kwargs(self) -> dict[str, Any]:
        """generate() kwargs that quantise the KV cache, or {} if disabled.

        Without this, reasoning_effort=high does not fit on a 4 GB card at our
        16k ceiling -- see the KV_QUANT note at the top of this module.
        """
        if not self._kv_quant_ok:
            return {}
        return {
            "cache_implementation": "quantized",
            "cache_config": {
                "backend": KV_QUANT_BACKEND,
                "nbits": 4 if KV_QUANT == "int4" else 8,
            },
        }

    def generate(self, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 1.0, top_p: float = 0.95,
                 max_new_tokens: int | None = None,
                 stop: list[str] | None = None,
                 reasoning_effort: str | None = None) -> Generation:
        import torch

        prompt = self.build_prompt(messages, tools, reasoning_effort)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=MAX_CONTEXT)
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.model.device)
        prompt_tokens = int(input_ids.shape[-1])

        budget = max_new_tokens or MAX_NEW_TOKENS
        # Never let prompt+output exceed the context we sized the KV cache for.
        budget = max(16, min(budget, MAX_CONTEXT - prompt_tokens))

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": budget,
            "pad_token_id": (self.tokenizer.pad_token_id
                             or self.tokenizer.eos_token_id),
            # CRITICAL on this model. logits_to_keep defaults to 0, and the
            # modeling code does `slice(-0, None)` -- which is slice(0, None),
            # i.e. EVERY position. lm_head then runs over the whole prompt
            # against a 250,624-token vocabulary: 2.5k tokens costs 1.2 GiB of
            # logits, and a full 16k prompt would need 7.6 GiB. We only ever
            # need the last position to sample the next token.
            "logits_to_keep": 1,
        }
        # temperature==0 means greedy in OpenAI-land; transformers wants do_sample
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        cache_kwargs = self._cache_kwargs()
        with self._lock, torch.inference_mode():
            try:
                out = self.model.generate(input_ids=input_ids,
                                          attention_mask=attn,
                                          **gen_kwargs, **cache_kwargs)
            except Exception as exc:
                if not cache_kwargs:
                    raise
                # Quantised KV is an optimisation, never a reason to 500. Drop
                # it for the rest of the process and retry once.
                print(f"[k2] quantised KV cache unavailable ({type(exc).__name__}: "
                      f"{exc}); falling back to an unquantised cache. Long "
                      f"reasoning may not fit -- lower K2_MAX_CONTEXT or "
                      f"K2_REASONING_EFFORT if you hit OOM.", flush=True)
                self._kv_quant_ok = False
                out = self.model.generate(input_ids=input_ids,
                                          attention_mask=attn, **gen_kwargs)

        new_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        completion_tokens = int(new_ids.shape[-1])

        # With only ~1 GiB of headroom, allocator fragmentation across requests
        # is enough to turn a fitting request into an OOM. Releasing the cached
        # blocks between calls costs little at 4 tok/s and buys back that margin.
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # The generation prompt ends inside an OPEN think block (see the
        # template's add_generation_prompt branch), so tell the parser that.
        gen = parse_output(text, starts_in_think=True)
        gen.prompt_tokens = prompt_tokens
        gen.completion_tokens = completion_tokens
        if gen.tool_calls:
            gen.finish_reason = "tool_calls"
        elif completion_tokens >= budget:
            gen.finish_reason = "length"
        if stop:
            gen.content = _apply_stop(gen.content, stop)
        return gen


# ===========================================================================
#  Output parsing  (IFM markup -> OpenAI shape)
# ===========================================================================
def parse_output(text: str, starts_in_think: bool = False) -> Generation:
    """Split raw decoded text into reasoning / tool calls / visible content.

    `starts_in_think` matters: the chat template's generation prompt ends with
    an OPEN think tag (``...assistant\\n<ifm|think_faster>\\n``), so the model's
    continuation begins inside the reasoning block and only ever emits the
    CLOSING tag. Without this flag the reasoning would be mistaken for the
    answer.
    """
    gen = Generation()
    reasoning_parts: list[str] = []

    # 1. The prompt opened a think block for the model -- everything up to the
    #    first closing tag is reasoning. No closing tag means we ran out of
    #    budget mid-thought, so the whole turn is reasoning and there is no
    #    answer yet.
    if starts_in_think:
        m = _THINK_CLOSE_RE.search(text)
        if m:
            reasoning_parts.append(text[:m.start()].strip())
            text = text[m.end():]
        else:
            reasoning_parts.append(text.strip())
            text = ""

    # 2. Pull out any fully-paired reasoning blocks.
    def _take_think(m: re.Match) -> str:
        reasoning_parts.append(m.group(2).strip())
        return ""

    text = _THINK_RE.sub(_take_think, text)
    # An unterminated block opened by the model itself.
    for tag in THINK_BLOCKS:
        open_tag = "<ifm|" + tag + ">"
        close_tag = "</ifm|" + tag + ">"
        if open_tag in text and close_tag not in text:
            head, _, tail = text.partition(open_tag)
            reasoning_parts.append(tail.strip())
            text = head
    gen.reasoning = "\n\n".join(p for p in reasoning_parts if p)

    # 2. Pull out tool calls.
    def _take_calls(m: re.Match) -> str:
        for raw in _TOOL_CALL_RE.findall(m.group(1)):
            call = _parse_one_call(raw.strip())
            if call:
                gen.tool_calls.append(call)
        return ""

    text = _TOOL_CALLS_RE.sub(_take_calls, text)

    # 3. Whatever remains is the visible answer; drop template/special tokens.
    text = re.sub(r"<\|ifm\|[^>]*\|>", "", text)
    text = re.sub(r"</?ifm\|[^>]*>", "", text)
    gen.content = text.strip()
    return gen


def _parse_one_call(raw: str) -> dict | None:
    """Parse one <ifm|tool_call> body in either the json or xml encoding."""
    name, args = None, {}

    if raw.startswith("{"):                       # json fmt
        try:
            obj = json.loads(raw)
            name = obj.get("name")
            args = obj.get("arguments", {})
        except json.JSONDecodeError:
            return None
    else:                                          # xml / xml_typed fmt
        lines = raw.splitlines()
        name = lines[0].strip() if lines else None
        for key, typ, val in _ARG_PAIR_RE.findall(raw):
            args[key.strip()] = _coerce(val.strip(), (typ or "").strip())

    if not name:
        return None
    return {
        "id": "call_" + uuid.uuid4().hex[:20],
        "type": "function",
        "function": {
            "name": name,
            # OpenAI wants arguments as a JSON *string*.
            "arguments": args if isinstance(args, str) else json.dumps(args),
        },
    }


def _coerce(value: str, typ: str) -> Any:
    """Best-effort typing for xml-format args (xml_typed gives us a hint)."""
    t = typ.lower()
    try:
        if t in ("int", "integer"):
            return int(value)
        if t in ("float", "number"):
            return float(value)
        if t in ("bool", "boolean"):
            return value.strip().lower() in ("true", "1", "yes")
        if t in ("object", "dict", "array", "list"):
            return json.loads(value)
    except (ValueError, json.JSONDecodeError):
        return value
    if not typ:                     # untyped: try JSON, else keep the string
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _apply_stop(text: str, stop: list[str]) -> str:
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            text = text[:idx]
    return text


# ===========================================================================
#  Message / tool normalisation  (OpenAI shape -> what the template expects)
# ===========================================================================
def _flatten_content(content: Any) -> str:
    """AgentScope/OpenAI may send content as a list of typed blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block["text"])
        return "\n".join(p for p in parts if p)
    return str(content)


def _normalise_message(msg: dict, effort: str = "high") -> dict:
    """Coerce one OpenAI message into the plain dict the jinja template reads."""
    out: dict[str, Any] = {
        "role": msg.get("role", "user"),
        "content": _flatten_content(msg.get("content")),
    }

    # The template REFUSES to render an assistant turn that has no thinking
    # field ("Assistant message is missing a thinking field"), and the agent's
    # ReAct loop replays assistant+tool_calls turns on every step -- so without
    # this, every multi-turn tool conversation dies on the second step.
    # Reuse real reasoning when the caller echoed it back, otherwise supply an
    # empty one (the template renders that as an empty block, which is legal).
    # Note we only ever copy STRING values: a key that is present but null
    # trips a different raise ("thinking fields must be strings").
    if out["role"] == "assistant":
        for key in _THINK_KEYS:
            if isinstance(msg.get(key), str):
                out[key] = msg[key]
                break
        else:
            # Key choice picks the tag, so match the effort we are running at.
            out[_EFFORT_THINK_KEY.get(effort, "think")] = ""
    if msg.get("name"):
        out["name"] = msg["name"]
    if msg.get("tool_call_id"):
        out["tool_call_id"] = msg["tool_call_id"]

    calls = msg.get("tool_calls")
    if calls:
        norm_calls = []
        for c in calls:
            fn = c.get("function", c) or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)     # template wants a real object
                except json.JSONDecodeError:
                    args = {"_raw": args}
            norm_calls.append({
                "id": c.get("id"),
                "type": "function",
                "function": {"name": fn.get("name"), "arguments": args},
            })
        out["tool_calls"] = norm_calls
    return out


# Thinking fields the template accepts on a replayed assistant turn, and which
# key to synthesise per effort (the key name selects the rendered tag).
_THINK_KEYS = ("think", "think_fast", "think_faster", "reasoning_content",
               "reasoning")
_EFFORT_THINK_KEY = {
    "high": "think",
    "medium": "think_fast",
    "low": "think_faster",
}

_SCHEMA_CHILD_KEYS = ("properties", "$defs", "definitions")
_SCHEMA_VARIANT_KEYS = ("anyOf", "oneOf", "allOf")


def _sanitise_schema(node: Any) -> Any:
    """Make a JSON schema safe for IFM's chat template.

    The template *hard-raises* (killing the whole request) when a schema says
    `required: [x]` but never defines `x` in `properties` -- see the
    validate_schema block in chat_template.jinja. The agent registers ~50 tools
    from assorted generators, so one sloppy schema would take down every call.
    We quietly drop the offending names instead.
    """
    if isinstance(node, list):
        return [_sanitise_schema(n) for n in node]
    if not isinstance(node, dict):
        return node

    out = {k: v for k, v in node.items()}
    props = out.get("properties")

    req = out.get("required")
    if req is not None:
        if not isinstance(req, list):
            out.pop("required")                 # template demands a list
        elif isinstance(props, dict):
            kept = [r for r in req if r in props]
            if kept:
                out["required"] = kept
            else:
                out.pop("required")
        elif not out.get("anyOf") and not out.get("oneOf"):
            # required with no properties object at all -> also a hard raise
            out.pop("required")

    for key in _SCHEMA_CHILD_KEYS:
        if isinstance(out.get(key), dict):
            out[key] = {k: _sanitise_schema(v) for k, v in out[key].items()}
    for key in _SCHEMA_VARIANT_KEYS:
        if isinstance(out.get(key), list):
            out[key] = [_sanitise_schema(v) for v in out[key]]
    if isinstance(out.get("items"), (dict, list)):
        out["items"] = _sanitise_schema(out["items"])
    return out


def _normalise_tools(tools: list[dict] | None) -> list[dict]:
    """Accept OpenAI's {'type':'function','function':{...}} and bare schemas."""
    if not tools:
        return []
    out = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            fn = t["function"]
        elif isinstance(t, dict) and "name" in t:
            fn = t
        else:
            continue
        fn = dict(fn)
        if isinstance(fn.get("parameters"), dict):
            fn["parameters"] = _sanitise_schema(fn["parameters"])
        out.append({"type": "function", "function": fn})
    return out


# Process-wide singleton.
RUNTIME = K2Runtime()
