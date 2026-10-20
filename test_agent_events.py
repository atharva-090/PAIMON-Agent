"""The event bus and StreamEmitter: does it publish, and is it really a no-op?

The emitter sits in the middleware chain of every agent in the swarm, so the
bar is higher than "it works": it must be impossible for it to change what a
turn does. These tests pin both halves -- the events it publishes, and the fact
that what flows through it comes out identical and in order.
"""
from __future__ import annotations

import asyncio
import sys
import types

import agent_events as E

# StreamEmitter now writes to the durable action ledger as well as the UI bus,
# so redirect that ledger at a temp file. Without this the suite appends fake
# tool calls to the real data/actions.jsonl -- the same way an earlier test
# quietly wrote fake rows into data/audit.jsonl.
import os
import tempfile

import action_log

action_log.LOG_PATH = os.path.join(
    tempfile.mkdtemp(prefix="agentevents-test-"), "actions.jsonl")

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        fails.append(msg)
    print(("   ok   " if cond else "   FAIL ") + msg)


class _TextDelta:
    """Stands in for agentscope's TextBlockDeltaEvent (matched by class name)."""

    def __init__(self, delta: str) -> None:
        self.delta = delta


TextBlockDeltaEvent = _TextDelta  # the emitter matches on type(...).__name__
_TextDelta.__name__ = "TextBlockDeltaEvent"


def tool_call(name: str, args: dict, cid: str = "c1"):
    return types.SimpleNamespace(type="tool_use", id=cid, name=name, input=args)


def tool_result(text: str):
    return types.SimpleNamespace(
        output=[types.SimpleNamespace(text=text)])


async def main() -> None:
    # ---------------------------------------------------------------- publish
    print("--- the bus fans out to subscribers ---")
    with E.subscribe() as q:
        E.publish("delegation", "orchestrator", child="browser_agent",
                  description="research term loans")
        evt = q.get_nowait()
    check(evt["kind"] == "delegation" and evt["child"] == "browser_agent",
          "an event reaches a subscriber with its fields intact")
    check("t" in evt and "seq" in evt, "events carry a timestamp and sequence")

    print("\n--- publishing with nobody listening is a no-op ---")
    before = E.has_listeners()
    E.publish("text", "orchestrator", text="nobody is watching")
    check(before is False, "no subscribers -> has_listeners() is False")

    # ------------------------------------------------------------ pass-through
    print("\n--- on_reasoning re-yields everything, unchanged and in order ---")
    emitted = [_TextDelta("Delegating "), _TextDelta("to the browser."),
               tool_call("delegate_task", {"child": "browser_agent"})]

    async def reasoning_handler(**_kwargs):
        for item in emitted:
            yield item

    em = E.StreamEmitter("orchestrator")
    with E.subscribe() as q:
        got = [item async for item in em.on_reasoning(
            agent=None, input_kwargs={"tool_choice": None},
            next_handler=reasoning_handler)]
        published = []
        while not q.empty():
            published.append(q.get_nowait())

    check(got == emitted, "every reasoning item is re-yielded, same objects, same order")

    kinds = [p["kind"] for p in published]
    check(kinds[0] == "step_start" and kinds[-1] == "step_end",
          "the step is bracketed by step_start/step_end")
    texts = [p["text"] for p in published if p["kind"] == "text"]
    check(texts == ["Delegating ", "to the browser."],
          "text deltas are published verbatim")
    calls = [p for p in published if p["kind"] == "tool_call"]
    check(len(calls) == 1 and calls[0]["name"] == "delegate_task",
          "a ToolCallBlock is published as a tool_call")

    # ------------------------------------------------------------- on_acting
    print("\n--- on_acting brackets the tool and reports its result ---")
    chunks = [tool_result("Category name already exists")]

    async def acting_handler(**_kwargs):
        for c in chunks:
            yield c

    with E.subscribe() as q:
        got = [item async for item in em.on_acting(
            agent=None,
            input_kwargs={"tool_call": tool_call("browse_click", {"ref": "@e1"})},
            next_handler=acting_handler)]
        published = [q.get_nowait() for _ in range(q.qsize())]

    check(got == chunks, "tool-result chunks are re-yielded untouched")
    start = next(p for p in published if p["kind"] == "tool_start")
    end = next(p for p in published if p["kind"] == "tool_end")
    check(start["name"] == "browse_click", "tool_start names the tool")
    check("@e1" in start["args"], "tool_start carries a preview of the arguments")
    check(end["ok"] is True and "already exists" in end["preview"],
          "tool_end carries the result preview (the UI decides if it reads as failure)")

    # ------------------------------------------------- a raising tool is honest
    print("\n--- a tool that raises is reported as failed, and the error still propagates ---")

    async def boom_handler(**_kwargs):
        raise RuntimeError("browser died")
        yield  # pragma: no cover -- makes this an async generator

    raised = False
    with E.subscribe() as q:
        try:
            async for _ in em.on_acting(
                agent=None,
                input_kwargs={"tool_call": tool_call("browse_open", {})},
                next_handler=boom_handler):
                pass
        except RuntimeError:
            raised = True
        published = [q.get_nowait() for _ in range(q.qsize())]

    check(raised, "the exception is NOT swallowed by the emitter")
    end = next(p for p in published if p["kind"] == "tool_end")
    check(end["ok"] is False and "browser died" in end["preview"],
          "the failure is published with its reason")

    # ------------------------------------------------------- backpressure
    print("\n--- a stalled subscriber drops events instead of blocking the agent ---")
    with E.subscribe() as q:
        for i in range(E.QUEUE_MAX + 40):
            E.publish("text", "browser_agent", text=str(i))
        check(q.qsize() <= E.QUEUE_MAX,
              f"queue stayed bounded at {q.qsize()} <= {E.QUEUE_MAX}")
        drained = [q.get_nowait() for _ in range(q.qsize())]
        check(drained[-1]["text"] == str(E.QUEUE_MAX + 39),
              "the NEWEST events survive (the oldest are dropped)")

    # --------------------------------------------------- publish never raises
    print("\n--- publish can never break a turn ---")

    class Hostile:
        def __repr__(self):
            raise ValueError("even repr explodes")

    with E.subscribe() as q:
        try:
            E.publish("tool_start", "scripter", args=E._preview(Hostile()))
            ok = True
        except Exception:
            ok = False
    check(ok, "an un-renderable argument is swallowed, not raised")

    print("\n" + "=" * 66)
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    asyncio.run(main())
