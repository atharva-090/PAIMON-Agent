"""End-to-end test for the Hermes swarm against the REAL openai-mocker.

What this test does (no mocks, no fakes):
  1. Confirms the openai-mocker is reachable.
  2. Bumps the mocker's MAX_SESSIONS via env to 8 so the orchestrator + 4
     children can coexist on the single mock backend.
  3. Builds the swarm.
  4. Sends a real prompt that *needs* multiple specialists:
        "Create a group called 'Vendor Onboarding 2026', add a task
         'Email Acme accounts for vendor form' due 2026-06-15 inside it,
         and draft (don't send) an email to acme@example.com requesting
         the form back by Friday."
     - task_manager should create the group + task,
     - comms_agent should draft the email (no send).
  5. Polls the status board on a side coroutine while the orchestrator
     runs, printing each transition.
  6. Asserts: orchestrator delegated to >=2 distinct specialists, all
     delegations finished (done|failed), and the final reply text isn't
     empty.

Run:
    agent/.venv/Scripts/python.exe -m multi_agent.test_e2e

Why this is slow/flaky (per user question):
  * Slow: each brain call goes through chatgpt.com via Playwright
    (~7-13s/call). A child making 3-4 tool calls = ~30-50s. The
    orchestrator + 2 children doing this sequentially-ish is 2-3 min total.
  * Flaky: ChatGPT-web sometimes refuses, drifts character, or returns a
    stale snapshot. With more concurrent brains it compounds because they
    all share ONE ChatGPT login and rate-limit each other. The mock's
    log will show "segment 1 of N" if a prompt was too big, "no JSON in
    output" if Playwright disconnected, etc.
  * In real working (not just this test): the same constraints apply.
    For production, switch the swarm to a real LLM profile (openai /
    anthropic) -- the brain calls become 1-3s, and 4 children can truly
    run in parallel without sharing one chatgpt.com login.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

# Run in an ISOLATED session lane so the test's ChatGPT threads don't collide
# with a live server swarm (which uses the default 'hermes' prefix). MUST be
# set before roster.py is imported. Override with SWARM_SESSION_PREFIX.
os.environ.setdefault("SWARM_SESSION_PREFIX", "hermes-test")


# --- preflight --------------------------------------------------------------
def _bump_mock_sessions() -> None:
    """The mocker reads MAX_SESSIONS at boot, not per-request -- so this only
    matters for THE TEST'S knowledge. We print a reminder if the operator
    didn't bump it themselves."""
    cur = os.environ.get("MAX_SESSIONS")
    if cur is None:
        print("[setup] Reminder: start the openai-mock with MAX_SESSIONS>=8")
        print("        (orchestrator + 4 children + headroom). Current mock "
              "default is 3 -- the test may evict sessions mid-run if you "
              "didn't bump it.")
    else:
        print(f"[setup] MAX_SESSIONS={cur} (caller's responsibility to honour)")


def _check_mock_alive() -> bool:
    import config as cfg
    base = cfg.BASE_URL.rsplit("/v1", 1)[0]
    try:
        urlopen(base + "/v1/models", timeout=3)
        return True
    except URLError:
        return False
    except Exception:
        # /v1/models may 404 on the mock but the connection itself proves
        # reachability; only a network-level failure means it's down.
        return True


# --- one-prompt multi-specialist exercise -----------------------------------
PROMPT = (
    "Two things to do, in parallel where possible:\n"
    "(a) Create a group called 'Vendor Onboarding 2026' and add a task "
    "'Email Acme accounts for vendor form' inside it with due date "
    "2026-06-15.\n"
    "(b) Draft (DO NOT SEND) an email to acme@example.com asking them to "
    "return the completed vendor form by Friday 2026-06-12.\n"
    "Then report back to me with the task id and the draft email."
)


async def _status_printer(swarm, stop_event: asyncio.Event) -> None:
    """Side coroutine: every time the board changes, dump it. Gives us a real
    timeline of pending -> in_progress -> done."""
    seen = -1
    while not stop_event.is_set():
        v = await swarm.board.wait_for_change(seen, timeout=1.5)
        if v <= 0:
            continue
        seen = v
        snap = swarm.board.snapshot()
        ts = time.strftime("%H:%M:%S")
        print(f"\n[board @{ts} v={v}] -----------------------------------")
        for t in snap:
            line = (f"  {t['child']:14s} {t['status']:11s} "
                    f"({t['elapsed_s']:5.1f}s, {t['tool_calls']} tool calls) "
                    f"{t['id']}")
            print(line)
            if t["report"]:
                # First 200 chars of the report, single-lined.
                rep = t["report"].replace("\n", " ")[:200]
                print(f"    report: {rep}")
            if t["error"]:
                print(f"    error:  {t['error']}")


async def _run() -> int:
    _bump_mock_sessions()
    if not _check_mock_alive():
        print("[FATAL] openai-mocker isn't reachable on the configured BASE_URL.")
        print("        Start it with: cd openai-mock && npm start")
        return 2

    # Late import so it picks up our env tweaks.
    from multi_agent.swarm import build_swarm

    swarm = build_swarm(profile="mock")
    print(f"[setup] swarm built: orchestrator + {len(swarm._specs)} children "
          f"(lazy spawn). Concurrency cap: {swarm.max_concurrent_children}.")
    print(f"[setup] specialists: {list(swarm._specs.keys())}")

    print("\n[run] TURN 1 -- sending prompt to orchestrator:")
    print(f"      {PROMPT}\n")

    stop = asyncio.Event()
    printer = asyncio.create_task(_status_printer(swarm, stop))
    t0 = time.time()
    try:
        reply = await swarm.swarm_chat(PROMPT)
    finally:
        stop.set()
        await printer
    elapsed = time.time() - t0

    print("\n========================================================")
    print(f"[turn1 reply @{elapsed:.1f}s] {reply}")
    print("========================================================\n")

    # --- TURN 2: the exact follow-up that previously broke -----------------
    # Before the fix, the orchestrator wiped its memory each turn and said
    # "I don't have an email draft." Now the board is a running ledger and the
    # orchestrator stays warm, so it must recall comms_agent's draft.
    print("[run] TURN 2 -- recall follow-up: 'what email did you prepare? "
          "show it to me.'\n")
    stop2 = asyncio.Event()
    printer2 = asyncio.create_task(_status_printer(swarm, stop2))
    t1 = time.time()
    try:
        reply2 = await swarm.swarm_chat(
            "what email did you prepare for Acme just now? show me the draft.")
    finally:
        stop2.set()
        await printer2
    print("\n========================================================")
    print(f"[turn2 reply @{time.time()-t1:.1f}s] {reply2}")
    print("========================================================\n")

    snap = swarm.board.snapshot()
    children_used = sorted({t["child"] for t in snap})
    statuses = [t["status"] for t in snap]
    print(f"[summary] delegations: {len(snap)}")
    print(f"[summary] specialists used: {children_used}")
    print(f"[summary] terminal statuses: {statuses}")

    # --- assertions --------------------------------------------------------
    ok = True

    if len(snap) == 0:
        print("\n[FAIL] orchestrator made zero delegations -- it answered "
              "the user inline instead of using the swarm. Likely cause: "
              "the LLM didn't think the prompt warranted delegation. "
              "Re-run; if reproducible the orchestrator persona may need "
              "stronger 'always delegate' wording.")
        ok = False
    if len(children_used) < 2:
        print(f"\n[FAIL] expected delegation to >=2 specialists, "
              f"got {len(children_used)} ({children_used}). The orchestrator "
              "may have routed both subtasks to one child.")
        ok = False
    pending = [t for t in snap if t["status"] in ("pending", "in_progress")]
    if pending:
        print(f"\n[FAIL] {len(pending)} delegation(s) never reached a "
              "terminal status:")
        for t in pending:
            print(f"        {t['child']} ({t['id']}): {t['status']}")
        ok = False
    if not reply.strip():
        print("\n[FAIL] orchestrator returned an empty final reply.")
        ok = False

    # The headline assertion: the recall follow-up must actually surface the
    # email draft, not "I don't have one".
    low2 = reply2.lower()
    recalled = ("acme" in low2 or "vendor form" in low2 or "subject" in low2)
    denied = ("don't have" in low2 or "do not have" in low2
              or "nothing" in low2 or "no email" in low2)
    if recalled and not denied:
        print("\n[PASS-RECALL] turn-2 orchestrator recalled the prepared "
              "email across turns (the original bug is fixed).")
    else:
        print("\n[FAIL-RECALL] turn-2 orchestrator did NOT recall the email "
              "draft -- it should pull comms_agent's report from the ledger / "
              "its warm memory. Reply was:")
        print(f"        {reply2[:300]}")
        ok = False

    if ok:
        print("\n[PASS] orchestrator delegated to >=2 specialists, all "
              "delegations reached a terminal status, and a final reply was "
              "produced.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
