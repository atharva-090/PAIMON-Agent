"""Context rotation for the Hermes swarm.

The single agent (task_agent.py) keeps ONE ChatGPT chat warm and rotates it --
archives + opens a fresh chat -- on a natural boundary: the user went idle, or
the thread grew long enough that the ChatGPT-web brain risks drifting. The
swarm has FIVE warm brains (orchestrator + 4 children), each in its own ChatGPT
chat, and none of them had that rotation -- so over a long conversation every
one of those contexts grew unbounded. The orchestrator's is the worst case: its
context balloons by a few tool-call messages every single turn, and when it
finally breaks the whole swarm goes down with it.

This module ports that idle/length rotation to the swarm. The crucial
difference from the single agent: the swarm's DURABLE memory is the StatusBoard
ledger, not the live agent contexts. Every delegation and every child report
already lives on the board and survives across turns. So a swarm "rotation" is
simpler than the single agent's -- there is nothing to archive, we just CLEAR
the agent's live context (``state.context = []``). After that:

  * the orchestrator recovers earlier work via ``check_status()`` -- its persona
    already tells it to read the 'earlier' ledger for recall -- so dropping its
    in-context memory loses nothing it can't re-read, and
  * a cleared child opens a fresh ChatGPT chat on its next delegation (the mock
    starts a new chat when the incoming context isn't a prefix-extension of what
    it last saw), which is exactly what we want after a boundary.

We rotate on natural boundaries ONLY -- never mid-burst -- so the comms
confirm-then-send handshake (which plays out within seconds/minutes) is never
severed:

  * 'idle'   -- this brain hasn't run for longer than SESSION_IDLE_SECONDS, or
  * 'length' -- its live context grew past SESSION_MAX_CONTEXT_MSGS.

Even if a boundary lands mid-handshake on a very long thread, the draft is still
on the board: the orchestrator re-reads it from the 'earlier' ledger and the
(separately-tracked) comms child usually hasn't rotated, so the send still
completes. That's the same accepted trade-off the single agent makes.
"""

from __future__ import annotations

import os
import time
from typing import Optional


# Same defaults as the single agent (task_agent.py) so the swarm and the solo
# agent age out their chats on the same schedule; override via env. We honour
# the single-agent env names too, so one setting tunes both.
SESSION_IDLE_SECONDS = int(
    os.environ.get("SWARM_SESSION_IDLE_MINUTES",
                   os.environ.get("SESSION_IDLE_MINUTES", "30"))) * 60
SESSION_MAX_CONTEXT_MSGS = int(
    os.environ.get("SWARM_MAX_CONTEXT_MSGS",
                   os.environ.get("SESSION_MAX_CONTEXT_MSGS", "80")))


class SessionRotator:
    """Per-brain idle/length rotation tracker for the swarm.

    One instance is owned by the Swarm. It remembers when each brain (keyed by
    a stable string -- ``"orchestrator"`` or a child's name) last ran, and
    decides, right BEFORE that brain is about to run, whether to clear its live
    context first. Clearing is safe because the board holds the durable history.
    """

    def __init__(
        self,
        idle_seconds: int = SESSION_IDLE_SECONDS,
        max_msgs: int = SESSION_MAX_CONTEXT_MSGS,
    ) -> None:
        self.idle_seconds = idle_seconds
        self.max_msgs = max_msgs
        self._last_active: dict[str, float] = {}

    def reason(self, key: str, agent) -> Optional[str]:
        """Why ``key``'s context should rotate before its next run, or None.

        'idle'   -- it's been quiet longer than the idle window (a fresh ask is
                    coming; the warm thread is cold/stale anyway), or
        'length' -- its context has grown past the message cap (drift risk on
                    the ChatGPT-web brain now outweighs continuity).
        Returns None for a brain's very first run (no recorded activity yet) so
        we never rotate a context that was just built.
        """
        last = self._last_active.get(key)
        if last is not None and time.time() - last > self.idle_seconds:
            return "idle"
        try:
            if len(agent.state.context) >= self.max_msgs:
                return "length"
        except Exception:
            pass
        return None

    def maybe_rotate(self, key: str, agent) -> Optional[str]:
        """Clear ``agent``'s live context if it hit an idle/length boundary.

        The board is the durable memory, so there is nothing to archive -- we
        just drop the context so the next run opens a fresh ChatGPT chat. Returns
        the rotation reason (for logging/tests), or None if no rotation happened.
        """
        reason = self.reason(key, agent)
        if reason:
            try:
                agent.state.context = []
            except Exception as exc:  # never let rotation break a turn
                print(f"[swarm] rotate {key} skipped ({type(exc).__name__}: {exc})")
                return None
            print(f"[swarm] rotated {key} context ({reason})")
        return reason

    def touch(self, key: str) -> None:
        """Record that ``key`` just ran -- resets its idle clock."""
        self._last_active[key] = time.time()

    def forget(self, key: Optional[str] = None) -> None:
        """Drop activity timestamps -- all of them on a full swarm reset, or
        just one brain's when given a key."""
        if key is None:
            self._last_active.clear()
        else:
            self._last_active.pop(key, None)
