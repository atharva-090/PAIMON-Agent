"""Load K2-Horizon for real and exercise the paths the agent depends on.

Unlike test_k2_parse.py (pure string handling, no GPU), this actually loads the
weights and generates, so it is slow and needs the GPU. It reports VRAM and
tokens/sec because on a 4 GB card those numbers decide whether the config is
usable at all.

Run:  agent/.venv-k2/Scripts/python.exe -m local_model.smoke_test
"""

from __future__ import annotations

import time
import traceback

import torch

from local_model import k2_runtime
from local_model.k2_runtime import RUNTIME


def vram(label: str) -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"    [vram] {label}: allocated {alloc:.2f} / reserved "
              f"{reserved:.2f} / total {total:.2f} GiB", flush=True)


def show(gen, elapsed: float) -> None:
    tps = gen.completion_tokens / elapsed if elapsed > 0 else 0
    print(f"    prompt={gen.prompt_tokens} completion={gen.completion_tokens} "
          f"in {elapsed:.1f}s ({tps:.1f} tok/s) finish={gen.finish_reason}")
    if gen.reasoning:
        r = gen.reasoning.replace("\n", " ")
        print(f"    reasoning[{len(gen.reasoning)} chars]: {r[:200]}")
    print(f"    content: {gen.content[:400]!r}")
    if gen.tool_calls:
        for c in gen.tool_calls:
            print(f"    TOOL CALL -> {c['function']['name']}"
                  f"({c['function']['arguments']})")


def main() -> int:
    print(f"config: quant={k2_runtime.QUANT} ctx={k2_runtime.MAX_CONTEXT} "
          f"kv={k2_runtime.KV_QUANT} effort={k2_runtime.REASONING_EFFORT}")

    print("\n[1/4] loading weights (slow: ~10 GB read + NF4 quantise)...")
    t0 = time.time()
    try:
        RUNTIME.load()
    except Exception:
        traceback.print_exc()
        print("\nLOAD FAILED")
        return 1
    print(f"    loaded in {time.time() - t0:.0f}s")
    vram("after load")

    # Where did accelerate actually put things?
    dev_map = getattr(RUNTIME.model, "hf_device_map", None)
    if dev_map:
        counts: dict[str, int] = {}
        for dev in dev_map.values():
            counts[str(dev)] = counts.get(str(dev), 0) + 1
        print(f"    device map: {counts}")
        offloaded = [k for k, v in dev_map.items() if str(v) in ("cpu", "disk")]
        if offloaded:
            print(f"    OFFLOADED to cpu/disk ({len(offloaded)}): "
                  f"{offloaded[:6]}{' ...' if len(offloaded) > 6 else ''}")

    # --- 2. plain turn ---------------------------------------------------
    print("\n[2/4] plain generation...")
    t0 = time.time()
    try:
        gen = RUNTIME.generate(
            [{"role": "user", "content": "Reply with exactly: HELLO FROM K2"}],
            temperature=0.0, max_new_tokens=256,
        )
        show(gen, time.time() - t0)
        vram("after plain gen")
    except Exception:
        traceback.print_exc()
        print("PLAIN GENERATION FAILED")
        return 1

    # --- 3. tool calling -------------------------------------------------
    print("\n[3/4] tool-calling generation...")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }]
    t0 = time.time()
    try:
        gen = RUNTIME.generate(
            [{"role": "user", "content": "What is the weather in Paris? "
                                         "Use the tool."}],
            tools=tools, temperature=0.0, max_new_tokens=512,
        )
        show(gen, time.time() - t0)
        if not gen.tool_calls:
            print("    WARNING: expected a tool call, got none")
    except Exception:
        traceback.print_exc()
        print("TOOL GENERATION FAILED")
        return 1

    # --- 4. tool result round-trip ---------------------------------------
    print("\n[4/4] feeding a tool result back...")
    t0 = time.time()
    try:
        gen = RUNTIME.generate([
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"city": "Paris"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "18C, light rain"},
        ], tools=tools, temperature=0.0, max_new_tokens=512)
        show(gen, time.time() - t0)
    except Exception:
        traceback.print_exc()
        print("TOOL ROUND-TRIP FAILED")
        return 1

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
