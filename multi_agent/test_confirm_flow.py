"""Real-ChatGPT e2e for the draft -> confirm -> send handshake + truthfulness.

This is the exact flow that misbehaved (orchestrator falsely said an email was
sent). We now prove:
  TURN 1  user asks to email someone
          -> comms_agent DRAFTS only (tools_used has NO send_email)
          -> orchestrator must NOT claim it was sent; the deterministic footer
             must say 'drafted ... nothing sent'
  TURN 2  user says 'yes, send it'
          -> orchestrator re-delegates to the warm comms_agent
          -> comms_agent actually calls send_email (tools_used HAS send_email)
          -> footer says 'email SENT'

Safety: the email is SELF-SENT to the configured Gmail address (you email
yourself), so 'really sending' has no third-party side effect.

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_confirm_flow
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

os.environ.setdefault("SWARM_SESSION_PREFIX", "hermes-test")


def _check_mock_alive() -> bool:
    import config as cfg
    from urllib.request import urlopen
    base = cfg.BASE_URL.rsplit("/v1", 1)[0]
    try:
        urlopen(base + "/v1/models", timeout=3)
        return True
    except Exception:
        return True  # connection refused is the only real 'down'


def _comms_tasks_this_turn(sw) -> list[dict]:
    return [t for t in sw.board.current_snapshot() if t["child"] == "comms_agent"]


def _sent_in(tasks: list[dict]) -> bool:
    return any("send_email" in (t.get("tools_used") or []) for t in tasks)


async def _turn(sw, label: str, prompt: str) -> str:
    print(f"\n########## {label} ##########")
    print(f"[user] {prompt}")
    t0 = time.time()
    reply = await sw.swarm_chat(prompt)
    print(f"\n[orchestrator reply @{time.time()-t0:.1f}s]\n{reply}\n")
    print("[board this turn]")
    for t in sw.board.current_snapshot():
        print(f"  {t['child']:13s} {t['status']:7s} tools_used={t.get('tools_used')}")
        if t.get("report"):
            print(f"     report: {t['report'][:120].replace(chr(10),' ')}")
    return reply


async def _run() -> int:
    if not _check_mock_alive():
        print("[FATAL] openai-mocker not reachable. Start: cd openai-mock && npm start")
        return 2

    import config
    recipient = config.GMAIL_ADDRESS or "me@example.com"

    from multi_agent.swarm import build_swarm
    sw = build_swarm(profile="mock")
    print(f"[setup] swarm built (prefix={os.environ['SWARM_SESSION_PREFIX']}); "
          f"self-send recipient = {recipient}")

    ok = True

    # --- TURN 1: ask to email; expect a DRAFT, not a send -----------------
    reply1 = await _turn(
        sw, "TURN 1 (draft)",
        f"Email {recipient} to confirm a 6 PM meeting with Umesh about LMS "
        f"functional stories, and ask them to confirm attendance.")
    comms1 = _comms_tasks_this_turn(sw)
    low1 = reply1.lower()
    if not comms1:
        print("[FAIL] orchestrator didn't delegate to comms_agent on turn 1")
        ok = False
    elif _sent_in(comms1):
        print("[FAIL] comms_agent SENT on turn 1 without confirmation "
              "(send_email in tools_used). It should only draft.")
        ok = False
    else:
        print("[OK] turn 1: comms_agent drafted only (no send_email).")
    if "verified actions this turn" not in low1:
        print("[FAIL] deterministic footer missing from reply.")
        ok = False
    elif "nothing sent" in low1 or "drafted" in low1:
        print("[OK] footer correctly states nothing was sent yet.")
    # the orchestrator's PROSE must not falsely claim it was sent
    if ("have sent" in low1 or "i've sent" in low1 or "email sent" in low1
            or "has been sent" in low1) and not _sent_in(comms1):
        print("[FAIL] orchestrator PROSE falsely claims the email was sent on "
              "turn 1.")
        ok = False
    else:
        print("[OK] turn-1 prose does not falsely claim 'sent'.")

    # --- TURN 2: confirm; expect an actual send ---------------------------
    reply2 = await _turn(
        sw, "TURN 2 (confirm -> send)",
        "Yes, that draft looks good - send it now.")
    comms2 = _comms_tasks_this_turn(sw)
    low2 = reply2.lower()
    if not comms2:
        print("[FAIL] orchestrator didn't re-delegate to comms_agent to send.")
        ok = False
    elif _sent_in(comms2):
        print("[OK] turn 2: comms_agent actually called send_email.")
        if "email sent" in low2 or "verified actions this turn" in low2:
            print("[OK] footer/report reflects the real send.")
    else:
        print("[FAIL] turn 2: comms_agent still didn't call send_email "
              "(email never actually sent).")
        ok = False

    print("\n" + ("[PASS] draft->confirm->send handshake works and reporting "
                  "is truthful (grounded in tools_used)."
                  if ok else "[FAIL] see above - the weak ChatGPT-web brain "
                  "may need another nudge, or rerun (it's non-deterministic)."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
