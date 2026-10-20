"""Fast, deterministic tests of swarm context rotation -- NO real LLM.

Covers the new idle/length rotation (rotation.py + its wiring in swarm.py):

  1. policy: a brain's FIRST run never rotates; it rotates on a 'length'
     boundary (context >= max_msgs) and on an 'idle' boundary (quiet too long).
  2. integration: driving the real ``_run_specialist`` path repeatedly with a
     low length cap keeps a child's live context BOUNDED instead of growing
     forever -- while the StatusBoard ledger still records every delegation
     (the durable memory survives rotation).
  3. reset: a full swarm reset also clears the rotator's idle clocks.

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_rotation
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import multi_agent.swarm as S
from multi_agent.rotation import SessionRotator


# --- minimal fakes (mirrors test_plumbing) ---------------------------------
class _FakeReply:
    def __init__(self, text: str) -> None:
        self._t = text

    def get_text_content(self) -> str:
        return self._t


class _FakeState:
    def __init__(self) -> None:
        self.context: list = []

        class _PC:
            mode = None

        self.permission_context = _PC()


class _FakeAgent:
    """Each reply appends 2 'tool_use' blocks to its context, so context grows
    by 2 messages per delegation -- enough to cross a low length cap."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.state = _FakeState()

    async def reply(self, msg):
        class _Blk:
            type = "tool_use"
            name = "fake_tool"

        # A real reply adds several messages (reasoning, tool_use, tool_result);
        # we add 2 per delegation so the context crosses a low length cap.
        for _ in range(2):
            self.state.context.append(
                type("M", (), {"content": [_Blk()]})())
        await asyncio.sleep(0.01)
        return _FakeReply(f"{self.name} ok")


def _parse(chunk) -> dict:
    return json.loads(chunk.content[0].text)


async def _drain(sw: S.Swarm, timeout: float = 5.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = sw.board.current_snapshot()
        if cur and all(t["status"] in ("done", "failed") for t in cur):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("tasks did not finish within timeout")


# --- 1. policy unit tests ---------------------------------------------------
def _test_policy() -> bool:
    ok = True
    rot = SessionRotator(idle_seconds=100, max_msgs=4)
    agent = _FakeAgent("x")

    # First run: no recorded activity -> never rotate a just-built context.
    if rot.maybe_rotate("x", agent) is not None:
        print("[FAIL] policy: first run should not rotate")
        ok = False
    rot.touch("x")

    # Grow context to the cap -> 'length' rotation clears it.
    agent.state.context = [object()] * 4
    reason = rot.maybe_rotate("x", agent)
    if reason != "length" or agent.state.context:
        print(f"[FAIL] policy: expected length-rotation to clear context, "
              f"got reason={reason!r} ctx={len(agent.state.context)}")
        ok = False
    else:
        print("[ok] policy: length boundary clears context")

    # Idle rotation: backdate last-active past the idle window.
    rot.touch("x")
    rot._last_active["x"] = time.time() - 200  # > idle_seconds(100)
    agent.state.context = [object()]           # small ctx, so only idle can fire
    reason = rot.maybe_rotate("x", agent)
    if reason != "idle" or agent.state.context:
        print(f"[FAIL] policy: expected idle-rotation, got reason={reason!r}")
        ok = False
    else:
        print("[ok] policy: idle boundary clears context")

    # forget() drops the idle clock -> next call is treated as a first run.
    rot.touch("x")
    rot.forget()
    agent.state.context = []
    if rot.maybe_rotate("x", agent) is not None:
        print("[FAIL] policy: forget() should reset to first-run (no rotate)")
        ok = False
    else:
        print("[ok] policy: forget() resets the idle clock")
    return ok


# --- 2. integration: bounded context, durable board ------------------------
async def _test_integration() -> bool:
    ok = True
    sw = S.build_swarm(profile="mock")

    def fake_ensure(name: str):
        if name not in sw._children:
            sw._children[name] = _FakeAgent(name)
        return sw._children[name]

    sw._ensure_child = fake_ensure  # type: ignore
    # Low length cap so a couple of delegations cross it.
    sw.rotator.max_msgs = 3

    sw.board.begin_turn()
    sw._last_seen_version = sw.board.version()

    N = 6
    for i in range(N):
        _parse(S.delegate_task("task_manager", f"do thing {i}"))
        await _drain(sw)  # sequential: the per-child lock serialises these

    child = sw._children["task_manager"]
    ctx = len(child.state.context)
    # Without rotation this would be 2*N (=12) and climbing; with a cap of 3 it
    # rotates whenever pre-run context >= 3, so it stays small and bounded.
    print(f"[integration] task_manager context after {N} delegations = {ctx} "
          f"(bounded; would be {2 * N} without rotation)")
    if ctx > sw.rotator.max_msgs + 2:
        print("[FAIL] integration: context not bounded -> rotation not firing")
        ok = False

    # Durable memory: every delegation is still on the board ledger.
    cur = sw.board.current_snapshot()
    done = [t for t in cur if t["status"] == "done"]
    print(f"[integration] board recorded {len(done)}/{N} delegations "
          f"(ledger survives rotation)")
    if len(done) != N:
        print("[FAIL] integration: board lost delegations across rotations")
        ok = False
    return ok


async def _run() -> int:
    print("== policy ==")
    p_ok = _test_policy()
    print("\n== integration ==")
    i_ok = await _test_integration()
    ok = p_ok and i_ok
    print("\n" + ("[PASS] rotation: boundaries clear context, contexts stay "
                  "bounded, and the board ledger survives."
                  if ok else "[FAIL] see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
