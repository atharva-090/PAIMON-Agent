"""Step 3: injecting the action trail back into a pruned context.

`BrowserSnapshotPruner` collapses old page snapshots on the real-API path. That
is a large, correct saving, but it deletes the only record of what the agent DID
on those pages -- so it can re-try something that already failed, or forget that
a click failed and another route worked.

This puts a few factual lines back at the tail of the context in place of the
kilobytes that were removed.

Run:  .venv/Scripts/python.exe test_action_trail.py
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

import action_log

FAILS: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILS.append(label)


def truthy(label: str, got) -> None:
    ok = bool(got)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        FAILS.append(label)


class FakeState:
    def __init__(self):
        self.context = []


class FakeAgent:
    def __init__(self):
        self.state = FakeState()


def text_of(msg) -> str:
    return "".join(getattr(b, "text", "") or "" for b in msg.content)


tmp = tempfile.mkdtemp(prefix="trail-test-")
real_log = action_log.LOG_PATH
action_log.LOG_PATH = os.path.join(tmp, "actions.jsonl")

try:
    # ------------------------------------------------- middleware contract
    # The test that was MISSING when this shipped: the hooks were exercised
    # directly, which works fine on a plain class, so nothing caught that
    # neither middleware subclassed MiddlewareBase. A real run died at once on
    #   AttributeError: 'ActionTrailInjector' object has no attribute
    #   'is_implemented'
    # because AgentScope asks each middleware which hooks it defines before
    # dispatching. Calling a method is not the same as being installable.
    print()
    print("middleware contract (AgentScope dispatch)")
    from agentscope.middleware import MiddlewareBase
    import agent_events

    for cls, hook in ((action_log.ActionRecorder, "on_acting"),
                      (action_log.ActionTrailInjector, "on_reasoning"),
                      (agent_events.StreamEmitter, "on_acting")):
        obj = cls("x")
        truthy(f"{cls.__name__} is a MiddlewareBase",
               isinstance(obj, MiddlewareBase))
        truthy(f"{cls.__name__} reports {hook}", obj.is_implemented(hook))

    # A middleware must not claim hooks it does not define, or AgentScope
    # routes those calls into the no-op base and silently drops them.
    check("recorder does not claim on_reasoning",
          action_log.ActionRecorder("x").is_implemented("on_reasoning"), False)
    check("injector does not claim on_acting",
          action_log.ActionTrailInjector("x").is_implemented("on_acting"), False)

    # ----------------------------------------------------- message assembly
    print("\ntrail message")
    action_log.set_session("t1")
    check("no actions -> no block",
          action_log.trail_message_text("browser"), "")

    action_log.record("browser", "browse_click", {"text": "Attach files"},
                      result="Error: element not found")
    action_log.record("browser", "browse_download", {"url": "http://x/f.docx"},
                      result="ok, saved to project/reports/f.docx")
    action_log.record("comms", "send_email", {"to": "u@x.com"}, result="ok")

    block = action_log.trail_message_text("browser")
    truthy("block carries the agent's own failure", "FAILED" in block)
    truthy("block carries the agent's own success", "browse_download" in block)
    check("block excludes OTHER agents' actions", "send_email" in block, False)
    truthy("block states it is ground truth", "not from anything you reported" in block)

    # ------------------------------------------------------------ injection
    print("\ninjection into context")

    async def step(inj, agent):
        async def nxt(**kw):
            if False:
                yield None
        async for _ in inj.on_reasoning(agent, {}, nxt):
            pass

    inj = action_log.ActionTrailInjector("browser")
    ag = FakeAgent()
    asyncio.run(step(inj, ag))
    check("one message appended", len(ag.state.context), 1)
    truthy("it is the trail", action_log._is_trail_msg(ag.state.context[0]))
    truthy("it sits at the tail", action_log._is_trail_msg(ag.state.context[-1]))

    # The important property: repeated steps must REPLACE, never accumulate.
    for _ in range(4):
        asyncio.run(step(inj, ag))
    trails = [m for m in ag.state.context if action_log._is_trail_msg(m)]
    check("still exactly one trail after 5 steps", len(trails), 1)
    check("context did not grow", len(ag.state.context), 1)

    # Real context has other messages; those must survive untouched, and the
    # trail must end up after them.
    class Other:
        content = "a real conversation message"
    ag.state.context.insert(0, Other())
    ag.state.context.insert(1, Other())
    action_log.record("browser", "browse_navigate", {"url": "http://y"}, result="ok")
    asyncio.run(step(inj, ag))
    check("other messages preserved",
          sum(1 for m in ag.state.context if not action_log._is_trail_msg(m)), 2)
    truthy("trail re-appended at the tail",
           action_log._is_trail_msg(ag.state.context[-1]))
    truthy("trail content refreshed",
           "browse_navigate" in text_of(ag.state.context[-1]))

    # A cleared trail must not leave a stale block behind.
    action_log.clear("t1")
    asyncio.run(step(inj, ag))
    check("stale block removed when trail empties",
          sum(1 for m in ag.state.context if action_log._is_trail_msg(m)), 0)
    check("real messages still there", len(ag.state.context), 2)

    # ---------------------------------------------------------- line budget
    print("\nbudget")
    action_log.set_session("t2")
    for i in range(20):
        action_log.record("browser", "browse_snapshot", {"i": i}, result="ok")
    body = action_log.trail_message_text("browser")
    n = len([l for l in body.splitlines()
             if l.startswith("browse_")])
    check("capped at TRAIL_LINES", n, action_log.TRAIL_LINES)
    truthy("keeps the NEWEST actions", "i=19" in body)
    check("drops the oldest", "i=0," in body, False)

    # ----------------------------------------------- recorder (single-agent)
    print("\nActionRecorder — the single-agent path")

    class FakeCall:
        id = "c1"
        name = "send_email"
        input = {"to": "umesh@x.com"}

    class FakeChunk:
        content = "Error: SMTP auth failed"

    async def fake_next(**kw):
        yield FakeChunk()

    async def drive():
        rec = action_log.ActionRecorder("hermes")
        got = []
        async for item in rec.on_acting(None, {"tool_call": FakeCall()}, fake_next):
            got.append(item)
        return got

    action_log.set_session("solo")
    out = asyncio.run(drive())
    check("tool output re-yielded unchanged", len(out), 1)
    t = action_log.trail("solo")
    check("action recorded", len(t), 1)
    check("failure detected from the result", t[0]["ok"], False)
    check("agent labelled", t[0]["agent"], "hermes")

    # Recorder and StreamEmitter must never both be installed, or every call
    # would be logged twice. Assert they are wired to different paths.
    src_swarm = open("multi_agent/swarm.py", encoding="utf-8").read()
    src_task = open("task_agent.py", encoding="utf-8").read()
    check("swarm uses StreamEmitter, not ActionRecorder",
          "action_log.ActionRecorder" in src_swarm, False)
    check("task_agent uses ActionRecorder",
          "action_log.ActionRecorder" in src_task, True)
    truthy("injector paired with the pruner in swarm",
           "ActionTrailInjector" in src_swarm)
finally:
    action_log.LOG_PATH = real_log
    action_log.set_session("")
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    raise SystemExit(1)
print("all action-trail checks passed")
