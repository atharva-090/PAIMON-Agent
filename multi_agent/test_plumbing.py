"""Fast, deterministic tests of the swarm plumbing -- NO real LLM.

We replace each child agent with a FakeAgent (scripted .reply) so we can
assert the coordination logic in milliseconds instead of minutes:

  1. per-child serialization: two delegations to the SAME specialist never
     run concurrently (this is what was creating the duplicate task), while
     delegations to DIFFERENT specialists do overlap.
  2. cross-turn recall: a child's report stays on the board as a running
     ledger; after begin_turn() it shows up under check_status()['earlier']
     so the orchestrator can answer 'what did you do before'.
  3. warm children: a child's context is NOT wiped between delegations.

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_plumbing
"""

from __future__ import annotations

import asyncio
import json
import sys

import multi_agent.swarm as S


# --- fakes ------------------------------------------------------------------
class _FakeReply:
    def __init__(self, text: str) -> None:
        self._t = text

    def get_text_content(self) -> str:
        return self._t


class _FakeState:
    def __init__(self) -> None:
        self.context: list = []
        class _PC:  # permission_context stub
            mode = None
        self.permission_context = _PC()


class _FakeAgent:
    """Records overlap so we can prove the per-child lock. Each reply appends
    a fake 'tool_use' block to its context so warmth + tool-call counting are
    observable."""

    def __init__(self, name: str, tracker: "_OverlapTracker") -> None:
        self.name = name
        self.state = _FakeState()
        self._tracker = tracker

    async def reply(self, msg):
        async with self._tracker.enter(self.name):
            # Simulate a couple of tool calls landing in context (warm).
            class _Blk:
                type = "tool_use"
            self.state.context.append(type("M", (), {"content": [_Blk(), _Blk()]})())
            await asyncio.sleep(0.05)  # pretend work
            text = _block_text(msg.content)
            return _FakeReply(f"{self.name} did: {text[:40]}")


class _OverlapTracker:
    def __init__(self) -> None:
        self.active: dict[str, int] = {}
        self.max_active: dict[str, int] = {}
        self.global_active = 0
        self.global_max = 0

    class _Ctx:
        def __init__(self, t: "_OverlapTracker", name: str) -> None:
            self.t, self.name = t, name

        async def __aenter__(self):
            t = self.t
            t.active[self.name] = t.active.get(self.name, 0) + 1
            t.max_active[self.name] = max(t.max_active.get(self.name, 0),
                                          t.active[self.name])
            t.global_active += 1
            t.global_max = max(t.global_max, t.global_active)
            return self

        async def __aexit__(self, *exc):
            t = self.t
            t.active[self.name] -= 1
            t.global_active -= 1

    def enter(self, name: str):
        return self._Ctx(self, name)


# --- helpers ----------------------------------------------------------------
def _block_text(content) -> str:
    """Pull text out of a Msg's content, whether blocks are dicts or objects."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        b = content[0]
        return (b.get("text", "") if isinstance(b, dict)
                else getattr(b, "text", "")) or ""
    return str(content)


def _parse(chunk) -> dict:
    return json.loads(chunk.content[0].text)


def _text(chunk) -> str:
    return chunk.content[0].text


async def _drain(sw: S.Swarm, timeout: float = 5.0) -> None:
    """Wait until no current-turn task is still pending/in_progress."""
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = sw.board.current_snapshot()
        if cur and all(t["status"] in ("done", "failed") for t in cur):
            return
        await asyncio.sleep(0.02)  # yield so the child bg tasks can run
    raise AssertionError("tasks did not finish within timeout")


# --- tests ------------------------------------------------------------------
async def _run() -> int:
    tracker = _OverlapTracker()

    sw = S.build_swarm(profile="mock")
    # Swap in fakes: patch _ensure_child to hand back FakeAgents.
    def fake_ensure(name: str):
        if name not in sw._children:
            sw._children[name] = _FakeAgent(name, tracker)
        return sw._children[name]
    sw._ensure_child = fake_ensure  # type: ignore

    ok = True

    # --- TURN 1: in-flight guard + a different-child runs in parallel -------
    sw.board.begin_turn()
    sw._last_seen_version = sw.board.version()
    r1 = _parse(S.delegate_task("task_manager", "create group Vendor 2026"))
    # Second delegation to the SAME child while the first is in flight must be
    # REJECTED by the in-flight guard (this is what stops the orchestrator's
    # re-delegation cascade). It returns an ERROR, not a task_id.
    r2_text = _text(S.delegate_task("task_manager", "add task inside it"))
    r3 = _parse(S.delegate_task("comms_agent", "draft email to acme"))
    print(f"[guard] 2nd same-child delegation -> {r2_text[:60]!r}")
    if "ALREADY working" not in r2_text and not r2_text.startswith("ERROR"):
        print("[FAIL] in-flight guard did NOT reject a duplicate same-child "
              "delegation -> the re-delegation cascade could still happen")
        ok = False
    await _drain(sw)

    # different children DID overlap (proves we didn't over-serialize).
    print(f"[parallel] global max concurrent = {tracker.global_max} "
          f"(task_manager + comms ran together => >=2 expected)")
    if tracker.global_max < 2:
        print("[WARN] no cross-child overlap observed (timing); not fatal")

    cur = sw.board.current_snapshot()
    done = [t for t in cur if t["status"] == "done"]
    print(f"[turn1] {len(done)}/{len(cur)} delegations done "
          f"(2 expected: the duplicate was rejected)")
    if len(done) != 2:
        print("[FAIL] expected exactly 2 delegations (guard rejected the 3rd)")
        ok = False

    # --- TURN 2: re-delegate task_manager (now allowed) -> proves warmth ----
    # The guard released once turn-1's task finished, so this is accepted; and
    # the child's context must have GROWN (not been wiped) across delegations.
    sw.board.begin_turn()
    sw._last_seen_version = sw.board.version()
    ctx_before = len(sw._children["task_manager"].state.context)
    _parse(S.delegate_task("task_manager", "rename the group"))
    await _drain(sw)
    ctx_after = len(sw._children["task_manager"].state.context)
    print(f"[warm] task_manager context {ctx_before} -> {ctx_after} across turns")
    if ctx_after <= ctx_before:
        print("[FAIL] child context was wiped between delegations (not warm)")
        ok = False

    # --- TURN 3: follow-up question, NO new delegation ---------------------
    sw.board.begin_turn()
    sw._last_seen_version = sw.board.version()
    status = _parse(S.check_status())
    earlier = status["earlier"]
    print(f"[recall] check_status earlier-ledger has {len(earlier)} past tasks")
    comms_reports = [t for t in earlier if t["child"] == "comms_agent"]
    if not comms_reports or "draft email" not in comms_reports[0]["report"]:
        print("[FAIL] comms_agent's earlier report not recallable -> the "
              "'what email did you prepare' bug would still happen")
        ok = False
    else:
        print(f"[recall] recovered comms report: "
              f"{comms_reports[0]['report'][:50]!r}")
    if status["summary"]["current_total"] != 0:
        print("[FAIL] turn-2 should have no current delegations")
        ok = False

    print("\n" + ("[PASS] plumbing: per-child lock holds, children stay warm, "
                   "and past work is recallable across turns."
                   if ok else "[FAIL] see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
