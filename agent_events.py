"""In-process event bus so the UI can watch a turn happen instead of waiting.

Until now a swarm turn was a black box: the browser POSTed to /api/agent/chat
and got a finished string back a minute later, with only the coarse status board
(`/api/agent/swarm/status`) to poll in the meantime. That is fine for a machine
and terrible for a person -- the interesting part of a multi-agent run IS the
middle, and it was invisible.

This publishes that middle. AgentScope already routes every reasoning step and
every tool call through the middleware chain (the same mechanism the snapshot
pruner hooks), and `_reasoning_impl` yields typed events -- TextBlockDeltaEvent,
ThinkingBlockDeltaEvent, ToolCallBlock -- as they happen. `StreamEmitter` sits in
that chain, copies what flows past onto a pub/sub bus, and re-yields it
untouched. The SSE endpoint in server.py drains the bus.

Design constraints, in order of importance:

  * NEVER break a turn. Every publish is wrapped and every failure is swallowed.
    A UI feature must not be able to kill a run. If the bus jams, events are
    dropped, not awaited -- `put_nowait` on a bounded queue, and a full queue
    loses the event rather than blocking the agent.
  * Tag by agent. Five brains run concurrently; an untagged stream is noise.
    Every event carries the name of the agent that produced it.
  * Cheap when nobody is watching. With no subscribers, publish() is a set
    emptiness check and returns.

The bus is deliberately global and process-local: one server, one event loop,
one swarm. Nothing here is designed to survive a restart or fan out to another
process -- it is a window onto the current turn, not a log. `usage_log` and
`data/audit.jsonl` remain the durable records.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, AsyncGenerator, Callable

from agentscope.middleware import MiddlewareBase

import action_log

# How many events a single slow subscriber may fall behind before we start
# dropping its oldest. A browser tab that backgrounds itself must not be able to
# apply backpressure to the agents.
QUEUE_MAX = 512

# Tool-result previews are for glanceability, not fidelity -- the full result is
# in the agent's context and, for browser work, in data/browser_runs/.
PREVIEW_CHARS = 240

_subscribers: set[asyncio.Queue] = set()

# Bumped on every publish so a subscriber can tell "nothing happened" from
# "I missed it".
_seq = 0


def _now_ms() -> int:
    return int(time.time() * 1000)


def has_listeners() -> bool:
    return bool(_subscribers)


def publish(kind: str, agent: str = "", **fields: Any) -> None:
    """Fan one event out to every subscriber. Never raises, never blocks."""
    global _seq
    if not _subscribers:
        return
    try:
        _seq += 1
        evt = {"seq": _seq, "t": _now_ms(), "kind": kind, "agent": agent}
        evt.update(fields)
        for q in list(_subscribers):
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                # Drop the oldest so a stalled tab degrades to "missed the
                # middle" instead of stalling the run.
                with contextlib.suppress(Exception):
                    q.get_nowait()
                    q.put_nowait(evt)
            except Exception:
                continue
    except Exception:
        pass


@contextlib.contextmanager
def subscribe():
    """Yield a queue receiving every event published while it is open."""
    q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
    _subscribers.add(q)
    try:
        yield q
    finally:
        _subscribers.discard(q)


def _preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    """A short, single-line, safe-to-render rendition of anything."""
    try:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            import json
            text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            text = str(value)
        text = " ".join(text.split())
        return text if len(text) <= limit else text[:limit] + "…"
    except Exception:
        return ""


# One implementation, in action_log, so the UI preview and the durable ledger
# can never disagree about what a tool returned.
_result_text = action_log.result_text


class StreamEmitter(MiddlewareBase):
    """Copies an agent's reasoning and tool activity onto the bus.

    Added to every agent in the swarm. It is a pure observer: each hook
    re-yields exactly what it received, in order, so removing it changes
    nothing about how a turn executes.
    """

    def __init__(self, agent_label: str) -> None:
        self.label = agent_label

    # -- reasoning: model text, thinking, and the decision to call a tool ----
    async def on_reasoning(
        self,
        agent: "Any",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        if not _subscribers:
            async for event in next_handler(**input_kwargs):
                yield event
            return

        publish("step_start", self.label)
        try:
            async for event in next_handler(**input_kwargs):
                try:
                    self._note(event)
                except Exception:
                    pass
                yield event
        finally:
            publish("step_end", self.label)

    def _note(self, event: Any) -> None:
        """Translate one AgentScope event into a bus event, if it is one we
        show. Unknown events are ignored rather than guessed at."""
        name = type(event).__name__
        if name == "TextBlockDeltaEvent":
            delta = getattr(event, "delta", "") or ""
            if delta:
                publish("text", self.label, text=delta)
        elif name == "ThinkingBlockDeltaEvent":
            delta = getattr(event, "delta", "") or ""
            if delta:
                publish("thinking", self.label, text=delta)
        elif getattr(event, "type", None) == "tool_use":
            # A ToolCallBlock: the model has decided on a call but it has not
            # run yet. on_acting reports the execution and its result.
            publish("tool_call", self.label,
                    id=getattr(event, "id", ""),
                    name=getattr(event, "name", ""),
                    args=_preview(getattr(event, "input", None)))

    # -- acting: the tool actually runs -------------------------------------
    async def on_acting(
        self,
        agent: "Any",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        # NOTE: unlike on_reasoning there is no "nobody is watching" shortcut
        # here. The UI bus is ephemeral, but action_log is a DURABLE record that
        # learn_from_session reads afterwards -- skipping it when no browser tab
        # happens to be open would mean a run learns nothing, which is both a
        # silent failure and an almost impossible one to reproduce.
        call = input_kwargs.get("tool_call")
        call_id = getattr(call, "id", "") or ""
        tool_name = getattr(call, "name", "") or "tool"
        raw_args = getattr(call, "input", None)
        started = time.time()
        if _subscribers:
            publish("tool_start", self.label, id=call_id, name=tool_name,
                    args=_preview(raw_args))

        last = ""
        ok = True
        try:
            async for item in next_handler(**input_kwargs):
                text = _result_text(item)
                if text:
                    last = text
                yield item
        except Exception as exc:
            ok = False
            last = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ms = int((time.time() - started) * 1000)
            # Ground truth first: record what the tool was asked to do and what
            # it actually returned. action_log decides ok/FAILED from the real
            # result, not from the absence of an exception.
            try:
                action_log.record(self.label, tool_name, raw_args,
                                  ok=ok, result=last, ms=ms)
            except Exception:
                pass
            # A tool that "succeeds" while returning an error string is the
            # failure mode this project keeps hitting, so the preview -- not
            # the absence of an exception -- is what the UI should show.
            if _subscribers:
                publish("tool_end", self.label, id=call_id, name=tool_name,
                        ok=ok, ms=ms, preview=_preview(last))
