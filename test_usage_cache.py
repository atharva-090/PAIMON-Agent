"""Item H: prompt-cache accounting + metering the two blind spots.

Background. A real swarm run was estimated at $1.71 while the provider
dashboard charged $1.18 -- a 45% overstatement. Two separate causes:

  1. Cached input was billed at the full input rate. Providers report a
     prompt-cache hit as a SUBSET of input_tokens, charged at a discount;
     we ignored the field entirely.
  2. ~32 model calls were never recorded at all, because only the swarm
     wrapped its models in usage_log.metered -- task_agent.build_model and
     memory_learn._llm_complete built theirs raw.

These pull the estimate in OPPOSITE directions, which is why the total
looked merely "a bit off" rather than obviously broken.

Run:  .venv/Scripts/python.exe test_usage_cache.py
"""
from __future__ import annotations

import json
import os
import tempfile

import usage_log

FAILS: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILS.append(label)


def close(label: str, got: float, want: float, tol: float = 1e-9) -> None:
    ok = abs(got - want) < tol
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want ~{want!r})"))
    if not ok:
        FAILS.append(label)


class FakeUsage:
    """Stands in for AgentScope's ChatUsage."""

    def __init__(self, inp, out, cached=None):
        self.input_tokens = inp
        self.output_tokens = out
        if cached is not None:
            self.cache_input_tokens = cached


# ---------------------------------------------------------------- pricing
print("\npricing")

# gpt-4o: in 2.50, out 10.00, cached 1.25 per 1M.
close("no cache hit = old behaviour",
      usage_log.price_of("gpt-4o", 1_000_000, 0), 2.50)
close("all input cached = cached rate",
      usage_log.price_of("gpt-4o", 1_000_000, 0, 1_000_000), 1.25)
close("half cached = midpoint",
      usage_log.price_of("gpt-4o", 1_000_000, 0, 500_000), 1.875)
close("output unaffected by caching",
      usage_log.price_of("gpt-4o", 0, 1_000_000, 0), 10.00)

# The bug this whole item exists to kill: cached must be SUBTRACTED from the
# full-rate input, never added on top. If it were additive, this would be
# 2.50 + 1.25 = 3.75 instead of 1.25.
close("cached is a subset, not a surcharge",
      usage_log.price_of("gpt-4o", 1_000_000, 0, 1_000_000), 1.25)

# Defensive: a provider reporting cached > input must not produce a credit.
close("cached clamped to input",
      usage_log.price_of("gpt-4o", 1000, 0, 999_999),
      usage_log.price_of("gpt-4o", 1000, 0, 1000))
ok = usage_log.price_of("gpt-4o", 1000, 0, 999_999) > 0
print(f"  {'ok  ' if ok else 'FAIL'} over-reported cache never goes negative")
if not ok:
    FAILS.append("negative cost")

check("unknown model still returns None",
      usage_log.price_of("llama-99b", 1000, 1000, 500), None)

# A model with no explicit cached column falls back to DEFAULT_CACHE_RATIO.
usage_log.PRICES["testmodel"] = (10.0, 20.0)          # 2-tuple, legacy shape
close("legacy 2-tuple infers cached rate",
      usage_log.price_of("testmodel", 1_000_000, 0, 1_000_000),
      10.0 * usage_log.DEFAULT_CACHE_RATIO)
del usage_log.PRICES["testmodel"]

# ---------------------------------------------------------------- recording
print("\nrecording (redirected to a temp file -- never touches data/)")

real_log = usage_log.LOG_PATH
tmp = tempfile.mkdtemp(prefix="usage-test-")
usage_log.LOG_PATH = os.path.join(tmp, "usage.jsonl")
try:
    usage_log.record("orchestrator", "gpt-4o", FakeUsage(1000, 100, 800))
    usage_log.record("browser", "gpt-4o", FakeUsage(2000, 200))  # no attr
    usage_log.record("ghost", "gpt-4o", FakeUsage(0, 0, 0))      # dropped

    rows = usage_log.read()
    check("rows written (empty usage skipped)", len(rows), 2)
    check("cache_input_tokens recorded", rows[0]["cache_input_tokens"], 800)
    check("missing attr defaults to 0", rows[1]["cache_input_tokens"], 0)
    # total_tokens must stay in+out; cached is already inside input.
    check("total excludes cached", rows[0]["total_tokens"], 1100)

    close("cost uses the cached rate",
          rows[0]["est_usd"],
          round(200 / 1e6 * 2.50 + 800 / 1e6 * 1.25 + 100 / 1e6 * 10.00, 6))

    # A row from BEFORE this change has no cache_input_tokens key at all.
    with open(usage_log.LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": "2025-01-01T00:00:00+00:00",
                             "agent": "legacy", "model": "gpt-4o",
                             "input_tokens": 500, "output_tokens": 50,
                             "total_tokens": 550, "est_usd": 0.00175}) + "\n")

    s = usage_log.summary()
    check("legacy row survives summary", s["total"]["calls"], 3)
    check("legacy row counts as 0 cache hits",
          s["total"]["cache_input_tokens"], 800)
    check("per-agent carries cached",
          s["per_agent"]["orchestrator"]["cache_input_tokens"], 800)

    rep = usage_log.report()
    ok = "cached" in rep and "cache hits:" in rep
    print(f"  {'ok  ' if ok else 'FAIL'} report shows cached column + hit rate")
    if not ok:
        FAILS.append("report")
finally:
    usage_log.LOG_PATH = real_log

# ---------------------------------------------------------------- metering
print("\nmetering (the two builders that were invisible)")

os.environ.setdefault("LLM_PROFILE", "mock")

import task_agent          # noqa: E402
import memory_learn        # noqa: E402

m = task_agent.build_model("mock")
ok = type(m).__name__.startswith("Metered")
print(f"  {'ok  ' if ok else 'FAIL'} task_agent.build_model -> {type(m).__name__}")
if not ok:
    FAILS.append("task_agent metered")
check("task_agent tags its spend", getattr(m, "_usage_agent", None), "task_agent")

# memory_learn builds its model inside _llm_complete, so assert on the source
# rather than run a completion (that would make a real call).
src = open(memory_learn.__file__, encoding="utf-8").read()
for frag, label in (("usage_log.metered(", "memory_learn wraps its model"),
                    ('usage_log.attach(model, "memory_learn"',
                     "memory_learn tags its spend")):
    ok = frag in src
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        FAILS.append(label)

# ---------------------------------------------------------------- verdict
print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    raise SystemExit(1)
print("all usage-accounting checks passed")
