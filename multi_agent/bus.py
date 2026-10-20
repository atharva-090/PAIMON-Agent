"""In-process status board + message bus for the Hermes swarm.

Two collaborating pieces:

* ``StatusBoard`` -- the shared "who's doing what" view. The orchestrator
  writes a row when it delegates work (``new_task``), the swarm runner
  flips its status as the child agent runs (in_progress -> done|failed) and
  stores the child's final report on the same row. The orchestrator reads
  ``snapshot()`` to know the state of EVERY delegated task at once, which is
  exactly the user's spec: when one child reports back, the parent should
  also see the status of every other agent it allocated work to.

* ``MessageBus`` -- per-agent asyncio inboxes. v1 only uses it
  orchestrator -> child (the child's runner picks up the assignment), but
  it's the same primitive ruflo uses for child <-> child gossip later.

Concurrency model: everything lives in ONE asyncio loop (the FastAPI server
loop). Children run as background tasks on that loop; status board mutations
are synchronous and thus race-free without locking. Waiters get woken via a
counter + Event so a transition that lands before they start waiting still
returns immediately.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


TaskStatus = str  # "pending" | "in_progress" | "done" | "failed"


@dataclass
class TaskRecord:
    """One delegated task on the shared status board.

    ``report`` is the child agent's final text reply (its "what I did" message
    back to the orchestrator). ``tool_calls`` is the count of tools the child
    actually used -- useful when judging whether work really happened or the
    child just talked.
    """

    id: str
    child: str
    description: str
    status: TaskStatus = "pending"
    report: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)  # actual tool NAMES the child called
    turn: int = 0   # which orchestrator turn delegated this (1-based)
    # Structured sub-answers the child recorded via record_answer (its external
    # scratchpad). list of {"q": ..., "a": ..., "note": ...}. The final report is
    # built from THESE deterministically, so a multi-part answer doesn't depend
    # on the weak brain re-aggregating everything in one last message.
    findings: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        end = self.ended_at if self.ended_at is not None else time.time()
        d["elapsed_s"] = round(end - self.started_at, 1)
        return d


class StatusBoard:
    """The single shared view of every delegated task.

    Designed for one-line orchestrator queries:
      * ``snapshot()`` -> all rows (the orchestrator's "give me everyone's
        status" call)
      * ``wait_for_change(since_version, timeout)`` -> wakes when any task
        transitions, so the orchestrator doesn't have to poll on a timer.

    The ``version`` counter monotonically increments on every state change.
    A waiter passes the version it last saw; if the board has moved on since
    then it returns immediately, otherwise it parks on an Event until the next
    change. That's the standard "no lost wake-ups" trick: state-then-event,
    waiter checks state first, parks only if nothing new.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskRecord] = {}
        self._version: int = 0
        self._event = asyncio.Event()
        self._turn: int = 0   # current orchestrator turn (incremented per user msg)

    # -- turn tracking ------------------------------------------------------
    def begin_turn(self) -> int:
        """Start a new orchestrator turn. Does NOT clear past tasks -- the
        board is a running ledger of the whole conversation -- it just bumps
        the turn counter so new delegations are tagged as 'current' and
        earlier ones become 'history' the orchestrator can still recall."""
        self._turn += 1
        return self._turn

    @property
    def current_turn(self) -> int:
        return self._turn

    # -- mutation -----------------------------------------------------------
    def new_task(self, child: str, description: str) -> TaskRecord:
        tid = "tsk_" + uuid.uuid4().hex[:10]
        rec = TaskRecord(id=tid, child=child, description=description,
                         turn=self._turn)
        self._tasks[tid] = rec
        self._bump()
        return rec

    def mark(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        report: str = "",
        error: str = "",
        tool_calls: Optional[int] = None,
        tools_used: Optional[List[str]] = None,
    ) -> None:
        rec = self._tasks.get(task_id)
        if rec is None:
            return
        rec.status = status
        if report:
            rec.report = report
        if error:
            rec.error = error
        if tool_calls is not None:
            rec.tool_calls = tool_calls
        if tools_used is not None:
            rec.tools_used = tools_used
        if status in ("done", "failed"):
            rec.ended_at = time.time()
        self._bump()

    def _bump(self) -> None:
        self._version += 1
        self._event.set()

    # -- read ---------------------------------------------------------------
    def snapshot(self) -> List[dict]:
        return [r.to_dict() for r in self._tasks.values()]

    def current_snapshot(self) -> List[dict]:
        """Only tasks delegated in the CURRENT turn."""
        return [r.to_dict() for r in self._tasks.values()
                if r.turn == self._turn]

    def history_snapshot(self) -> List[dict]:
        """Tasks from EARLIER turns (the conversation's running ledger), newest
        last. This is what lets the orchestrator answer 'what did you do
        before / what email did you prepare' without re-running anyone."""
        return [r.to_dict() for r in self._tasks.values()
                if r.turn < self._turn]

    def by_status(self, status: TaskStatus) -> List[TaskRecord]:
        return [r for r in self._tasks.values() if r.status == status]

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def version(self) -> int:
        return self._version

    def clear(self) -> None:
        self._tasks.clear()
        self._version = 0

    # -- waits --------------------------------------------------------------
    async def wait_for_change(
        self, since_version: int, timeout: Optional[float] = None
    ) -> int:
        """Return the new version once it exceeds ``since_version`` (or after
        ``timeout`` seconds, whichever comes first; -1 on timeout)."""
        if self._version > since_version:
            return self._version
        # Park until SOMETHING changes; loop because Event.wait() only tells
        # us "set was called", not "version increased past mine".
        self._event.clear()
        try:
            while self._version <= since_version:
                if timeout is None:
                    await self._event.wait()
                else:
                    try:
                        await asyncio.wait_for(self._event.wait(), timeout)
                    except asyncio.TimeoutError:
                        return -1
                self._event.clear()
            return self._version
        finally:
            # Leave the event set so concurrent waiters can also progress.
            self._event.set()


class MessageBus:
    """Per-agent asyncio inboxes. The orchestrator uses ``send(child, ...)``
    to push a 'do this' payload; the child's runner ``await``s the queue.

    Kept intentionally minimal -- v1 only uses orchestrator -> child, but
    the primitive is the same one ruflo's MessageBus uses for child <-> child
    gossip and broadcasts in their swarm. We can add filters/priorities/acks
    later without changing the call sites."""

    def __init__(self) -> None:
        self._inboxes: Dict[str, asyncio.Queue] = {}

    def subscribe(self, agent_name: str) -> asyncio.Queue:
        return self._inboxes.setdefault(agent_name, asyncio.Queue())

    async def send(self, to: str, payload: dict) -> None:
        q = self._inboxes.setdefault(to, asyncio.Queue())
        await q.put(payload)

    def has_messages(self, agent_name: str) -> bool:
        q = self._inboxes.get(agent_name)
        return q is not None and not q.empty()
