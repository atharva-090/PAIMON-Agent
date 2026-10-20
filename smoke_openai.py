"""Smoke-test the swarm against a REAL provider API (no mocker involved).

Run it once OPENAI_API_KEY is set:

    cd agent
    .venv/Scripts/python.exe smoke_openai.py            # cheap, no browser
    .venv/Scripts/python.exe smoke_openai.py --browser  # adds a web task

It checks the things that actually differ from the mocker, in order of cost, and
stops at the first failure so a broken setup does not burn tokens:

  1. the profile resolves to a real provider (not a silent mock fallback)
  2. the agents get their NATIVE personas, with no JSON-protocol text
  3. one bare API call works and returns a real tool_call (not JSON as text)
  4. a task_manager delegation runs end to end and tools_used is populated
  5. optionally, a browser delegation

Step 3 is the one that matters: it is the difference between native function
calling working and the model writing {"tool_calls":...} into the message body.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                    # noqa: E402
import multi_agent.roster as roster              # noqa: E402

PROFILE = os.environ.get("SMOKE_PROFILE", "openai")
WANT_BROWSER = "--browser" in sys.argv
failures: list[str] = []


def head(n: str) -> None:
    print("\n" + "=" * 68)
    print(n)
    print("=" * 68)


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(("  PASS  " if ok else "  FAIL  ") + label + (" — " + detail if detail else ""))
    if not ok:
        failures.append(label)
    return ok


# ---------------------------------------------------------------- 1. profile
head("1. profile resolution")
prof = config.LLM_PROFILES.get(PROFILE)
if not prof:
    print("  no such profile: %r (have: %s)"
          % (PROFILE, ", ".join(config.LLM_PROFILES)))
    sys.exit(1)
effective = config.resolve_profile(PROFILE)
print("  requested : %s" % PROFILE)
print("  effective : %s" % effective)
print("  provider  : %s" % prof["provider"])
print("  model     : %s" % prof["model"])
print("  base_url  : %s" % prof.get("base_url", "(sdk default)"))
print("  api key   : %s" % ("set" if prof.get("api_key") else "NOT SET"))
if not check(effective == PROFILE, "profile is not silently falling back to mock",
             "set the provider's key in agent/.env" if effective != PROFILE else ""):
    sys.exit(1)
check(not config.is_mock_backend(PROFILE), "is_mock_backend() is False")

# ------------------------------------------------------------- 2. personas
head("2. agents get the native personas")
mock = config.is_mock_backend(PROFILE)
BAD = ['{"tool_calls"', '{"content"', "one JSON object"]
orch = (roster.ORCHESTRATOR_PERSONA if mock
        else roster.ORCHESTRATOR_PERSONA_NATIVE)
check(not any(b in orch for b in BAD), "orchestrator persona has no JSON protocol")
for s in roster.ROSTER:
    p = s.persona_for(mock)
    check(not any(b in p for b in BAD), "%s persona has no JSON protocol" % s.name)
if failures:
    sys.exit(1)

# ------------------------------------------------- 3. one bare API tool call
head("3. one API call — does it return a REAL tool_call?")
print("  (this is the whole question: native function calling vs JSON-as-text)")
try:
    from openai import OpenAI
    client = OpenAI(api_key=prof["api_key"], base_url=prof.get("base_url"))
    t0 = time.time()
    r = client.chat.completions.create(
        model=prof["model"],
        messages=[
            {"role": "system",
             "content": "You are a task-board assistant. Use your tools."},
            {"role": "user", "content": "Create a group called Smoke Test."},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "create_group",
                "description": "Create a task-board group.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }],
    )
    dt = time.time() - t0
    msg = r.choices[0].message
    calls = msg.tool_calls or []
    print("  latency   : %.1fs" % dt)
    print("  tokens    : %s in / %s out"
          % (r.usage.prompt_tokens, r.usage.completion_tokens))
    print("  content   : %r" % (msg.content or "")[:120])
    print("  tool_calls: %s" % [c.function.name for c in calls])
    check(bool(calls), "returned a structured tool_call")
    if msg.content and '{"tool_calls"' in msg.content:
        check(False, "model wrote JSON as message content",
              "the native persona split is not taking effect")
except Exception as exc:
    check(False, "bare API call", "%s: %s" % (type(exc).__name__, exc))

if failures:
    print("\nStopping before the swarm runs — fix the above first.")
    sys.exit(1)

# ------------------------------------------------------ 4. swarm end to end
head("4. swarm end to end (task_manager — no browser, cheap)")
os.environ["LLM_PROFILE"] = PROFILE
from multi_agent.swarm import get_swarm, reset_swarm   # noqa: E402


async def run(prompt: str) -> str:
    sw = get_swarm(PROFILE)
    t0 = time.time()
    reply = await sw.swarm_chat(prompt)
    print("  took %.1fs" % (time.time() - t0))
    for row in sw.board.current_snapshot():
        print("    %-14s %-11s tools_used=%s"
              % (row.get("child"), row.get("status"), row.get("tools_used")))
    return reply


try:
    reply = asyncio.run(run(
        "Create a task board group called 'Smoke Test' and add one task in it "
        "called 'check the openai path'. Report the ids."))
    print("\n  reply: %s" % (reply or "")[:400])
    sw = get_swarm(PROFILE)
    rows = sw.board.current_snapshot()
    check(bool(rows), "the orchestrator actually delegated")
    check(any(r.get("tools_used") for r in rows),
          "a child actually called tools (tools_used populated)")
except Exception as exc:
    check(False, "swarm run", "%s: %s" % (type(exc).__name__, exc))

# ------------------------------------------------------------- 5. browser
if WANT_BROWSER and not failures:
    head("5. browser delegation (slow, many steps, re-sends context each step)")
    try:
        reset_swarm()
        reply = asyncio.run(run(
            "Open https://example.com and tell me the exact heading text."))
        print("\n  reply: %s" % (reply or "")[:400])
        check("Example Domain" in (reply or ""),
              "browser child read the real page")
    except Exception as exc:
        check(False, "browser run", "%s: %s" % (type(exc).__name__, exc))
elif not WANT_BROWSER:
    print("\n(skipping the browser test — pass --browser to include it)")

head("RESULT")
if failures:
    print("FAILED:")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("ALL PASS — the swarm runs on %s (%s)." % (PROFILE, prof["model"]))
