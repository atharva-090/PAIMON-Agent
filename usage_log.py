"""Per-API-call token accounting.

AgentScope already hands us the numbers -- every ChatResponse carries a
ChatUsage with input_tokens / output_tokens -- but nothing in this project ever
read them, so a run's cost was invisible and unrecoverable after the fact. This
records one line per model call to data/usage.jsonl and can summarise it.

Cheap and best-effort: a failure here must never break a model call.

Pricing is USD per 1M tokens and is only a local table for estimating -- check
your provider's current rates. Override any of it with USAGE_PRICES in the
environment, e.g. USAGE_PRICES="gpt-4o:2.5/10,gpt-4o-mini:0.15/0.6".
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import config

LOG_PATH = os.path.join(config.DATA_DIR, "usage.jsonl")

# Fraction of the input price charged for a prompt-cache HIT, used when a
# model has no explicit cached rate below. 0.5 is OpenAI's common case; some
# models are cheaper (gpt-4.1 is 0.25x, Anthropic cache reads are 0.1x).
DEFAULT_CACHE_RATIO = 0.5

# model -> (input $/1M, output $/1M, cached-input $/1M). Estimates only --
# the cached column especially, since providers revise it more often than the
# headline rates. Check your provider and override with USAGE_PRICES.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (2.50, 10.00, 1.25),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-4.1": (2.00, 8.00, 0.50),
    "gpt-4.1-mini": (0.40, 1.60, 0.10),
    "o4-mini": (1.10, 4.40, 0.275),
    "claude-3-5-sonnet-latest": (3.00, 15.00, 0.30),
    "gemini-2.0-flash": (0.10, 0.40, 0.025),
}


def _load_price_overrides() -> None:
    raw = os.environ.get("USAGE_PRICES", "").strip()
    for part in filter(None, (p.strip() for p in raw.split(","))):
        try:
            name, rates = part.split(":", 1)
            bits = [float(b) for b in rates.split("/")]
            if len(bits) == 2:          # "in/out" -- cached rate inferred
                bits.append(bits[0] * DEFAULT_CACHE_RATIO)
            if len(bits) != 3:
                continue
            PRICES[name.strip()] = (bits[0], bits[1], bits[2])
        except Exception:
            continue


_load_price_overrides()


def price_of(model: str, inp: int, out: int, cached: int = 0) -> Optional[float]:
    """USD estimate for one call, or None when the model is not in the table.

    `cached` is the part of `inp` that was a prompt-cache HIT. Providers report
    it as a SUBSET of the input tokens (OpenAI's
    prompt_tokens_details.cached_tokens), NOT as an extra charge on top, so it
    has to be discounted out of `inp` rather than added to it.

    Ignoring this is why our estimate read $1.71 against a real dashboard
    charge of $1.18. A swarm run is close to the worst case for it: the
    orchestrator resends a near-identical prompt prefix every iteration, so
    most of its input is a cache hit and was never billed at full rate.
    """
    for name, rates in PRICES.items():
        if model == name or model.startswith(name):
            pi, po = rates[0], rates[1]
            pc = rates[2] if len(rates) > 2 else pi * DEFAULT_CACHE_RATIO
            hit = max(0, min(int(cached), int(inp)))   # never exceed the input
            return (inp - hit) / 1e6 * pi + hit / 1e6 * pc + out / 1e6 * po
    return None


def record(agent: str, model: str, usage, latency: float = 0.0,
           profile: str = "", tools: int = 0, call_id: str = "") -> None:
    """Append one call's usage. Never raises.

    `call_id` links this row to the full request/response that call_log wrote
    for the same call, so the viewer can go from "9,621 tokens" to the actual
    prompt without either file having to carry the other's data.
    """
    try:
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        # Subset of `inp` served from the provider's prompt cache. Stored
        # separately only so the cost estimate can bill it cheaper -- it is
        # already counted inside input_tokens, so never add it to a total.
        cached = int(getattr(usage, "cache_input_tokens", 0) or 0)
        if not (inp or out):
            return
        cached = max(0, min(cached, inp))
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "profile": profile or config.LLM_PROFILE,
            "model": model,
            "input_tokens": inp,
            "cache_input_tokens": cached,
            "output_tokens": out,
            "total_tokens": inp + out,
            "latency_s": round(float(latency), 2),
            "tools_offered": tools,
        }
        if call_id:
            rec["call_id"] = call_id
        # Which conversation this call belongs to. Without it a usage row can
        # only be tied back to a run by timestamp, which the viewer still does
        # for the rows written before this line existed.
        try:
            import action_log
            sid = action_log.current_session()
            if sid:
                rec["session"] = sid
        except Exception:
            pass
        cost = price_of(model, inp, out, cached)
        if cost is not None:
            rec["est_usd"] = round(cost, 6)
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read(since: str = "") -> list[dict]:
    """Every recorded call, optionally only those at/after an ISO timestamp."""
    out = []
    try:
        with open(LOG_PATH, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not since or r.get("ts", "") >= since:
                    out.append(r)
    except FileNotFoundError:
        pass
    return out


def summary(since: str = "") -> dict:
    """Totals overall and per agent."""
    rows = read(since)
    per: dict[str, dict] = {}
    tot = {"calls": 0, "input_tokens": 0, "cache_input_tokens": 0,
           "output_tokens": 0, "est_usd": 0.0}
    for r in rows:
        a = per.setdefault(r.get("agent", "?"),
                           {"calls": 0, "input_tokens": 0,
                            "cache_input_tokens": 0,
                            "output_tokens": 0, "est_usd": 0.0})
        for d in (a, tot):
            d["calls"] += 1
            d["input_tokens"] += r.get("input_tokens", 0)
            # .get default keeps rows written before caching was recorded
            # readable -- they simply count as zero cache hits.
            d["cache_input_tokens"] += r.get("cache_input_tokens", 0)
            d["output_tokens"] += r.get("output_tokens", 0)
            d["est_usd"] += r.get("est_usd", 0.0)
    return {"rows": rows, "per_agent": per, "total": tot}


def report(since: str = "", detail: bool = True) -> str:
    """Human-readable table: one line per call, then per-agent, then totals."""
    s = summary(since)
    rows, lines = s["rows"], []
    if not rows:
        return ("No API usage recorded yet (data/usage.jsonl). Only real "
                "provider calls are counted -- the mocker reports no usage.")
    if detail:
        lines.append("%-4s %-8s %-16s %9s %9s %8s %9s %8s"
                     % ("#", "time", "agent", "in", "cached", "out", "total",
                        "usd"))
        for i, r in enumerate(rows, 1):
            lines.append("%-4d %-8s %-16s %9d %9d %8d %9d %8s"
                         % (i, r.get("ts", "")[11:19], r.get("agent", "?")[:16],
                            r.get("input_tokens", 0),
                            r.get("cache_input_tokens", 0),
                            r.get("output_tokens", 0),
                            r.get("total_tokens", 0),
                            ("%.4f" % r["est_usd"]) if "est_usd" in r else "-"))
        lines.append("")
    lines.append("%-16s %6s %10s %10s %9s %10s"
                 % ("agent", "calls", "in", "cached", "out", "usd"))
    for a, d in sorted(s["per_agent"].items(),
                       key=lambda kv: -kv[1]["input_tokens"]):
        lines.append("%-16s %6d %10d %10d %9d %10.4f"
                     % (a, d["calls"], d["input_tokens"],
                        d["cache_input_tokens"], d["output_tokens"],
                        d["est_usd"]))
    t = s["total"]
    lines.append("%-16s %6d %10d %10d %9d %10.4f"
                 % ("TOTAL", t["calls"], t["input_tokens"],
                    t["cache_input_tokens"], t["output_tokens"], t["est_usd"]))
    # The hit rate is the single number that explains a gap between this
    # estimate and the provider's bill, so print it rather than make someone
    # divide two columns by hand.
    if t["input_tokens"]:
        pct = 100.0 * t["cache_input_tokens"] / t["input_tokens"]
        lines.append("")
        lines.append("cache hits: %d of %d input tokens (%.1f%%) billed at the "
                     "cached rate" % (t["cache_input_tokens"],
                                      t["input_tokens"], pct))
    return "\n".join(lines)


def metered(model_cls):
    """Wrap a model class so every call records its token usage.

    Applied at build time to whatever model the profile selected. The numbers
    are already on the ChatResponse; this is only the part that writes them
    down, tagged with WHICH agent spent them -- otherwise a run's cost is one
    opaque total instead of 'the browser child burned 80% of it'.
    """
    class Metered(model_cls):                       # type: ignore[misc,valid-type]
        _usage_agent = "?"
        _usage_profile = ""

        async def __call__(self, messages, tools=None, tool_choice=None,
                           **kwargs):
            t0 = time.time()
            model_name = (getattr(self, "model_name", None)
                          or getattr(self, "model", "?"))
            try:
                res = await super().__call__(messages, tools, tool_choice,
                                             **kwargs)
            except Exception as exc:
                # A call that FAILED is the one you most want to read back, and
                # it was the one case that logged nothing -- the payload write
                # sat after the await, so a 400 skipped it entirely.
                try:
                    import call_log
                    if call_log.enabled():
                        call_log.write_llm(
                            call_log.new_id(), self._usage_agent, model_name,
                            messages,
                            {"error": f"{type(exc).__name__}: {exc}"}, tools)
                except Exception:
                    pass
                raise
            try:
                # stream=False everywhere here, so res is a single response;
                # an async generator would be consumed by the caller, not us.
                usage = getattr(res, "usage", None)
                # This is the one place holding BOTH the full request and the
                # full response. Nothing downstream keeps them: usage.jsonl has
                # token counts, actions.jsonl has a 400-char result excerpt.
                cid = ""
                try:
                    import call_log
                    if call_log.enabled():
                        cid = call_log.new_id()
                        call_log.write_llm(cid, self._usage_agent, model_name,
                                           messages, res, tools)
                except Exception:
                    cid = ""
                if usage is not None:
                    record(self._usage_agent, model_name,
                           usage, time.time() - t0,
                           profile=self._usage_profile,
                           tools=len(tools or []), call_id=cid)
            except Exception:
                pass
            return res

    Metered.__name__ = "Metered" + model_cls.__name__
    return Metered


def attach(model, agent: str, profile: str = ""):
    """Tag an already-built metered model with the agent spending the tokens."""
    try:
        model._usage_agent = agent
        model._usage_profile = profile
    except Exception:
        pass
    return model


if __name__ == "__main__":
    import sys
    since = sys.argv[1] if len(sys.argv) > 1 else ""
    print(report(since))
