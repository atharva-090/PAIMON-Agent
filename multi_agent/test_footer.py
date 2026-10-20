"""Fast, deterministic tests for the ground-truth footer accounting -- NO LLM.

These guard the bug where ``_tool_names_used`` matched only ``type=="tool_use"``
while AgentScope's real tool-call block (ToolCallBlock) has ``type=="tool_call"``.
The result: every LIVE delegation reported an empty tools_used, so the footer
wrongly said "no actions taken (text only)" even when the child really called a
tool. The plumbing/rotation fakes use the "tool_use" spelling, so only a test
built from the REAL block type catches a regression -- that's what test #1 does.

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_footer
"""

from __future__ import annotations

import sys

from agentscope.message import Msg, ToolCallBlock, TextBlock

import multi_agent.swarm as S


def _ctx_with_real_tool_calls(*names: str) -> list:
    """A context list shaped like a real AgentScope run: an assistant Msg whose
    content holds ToolCallBlock(s) (type == 'tool_call')."""
    blocks = [ToolCallBlock(id=f"c{i}", name=n, input="{}")
              for i, n in enumerate(names)]
    return [Msg(name="task_manager", content=blocks, role="assistant")]


class _Board:
    """Minimal stand-in exposing just current_snapshot() for the footer."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def current_snapshot(self) -> list[dict]:
        return self._rows


def _test_real_blocks_counted() -> bool:
    # THE regression: real ToolCallBlock(type='tool_call') must be counted.
    ctx = _ctx_with_real_tool_calls("create_group", "create_task")
    used = S._tool_names_used(ctx, since=0)
    print(f"[real-blocks] tools_used from real ToolCallBlocks = {used}")
    if used != ["create_group", "create_task"]:
        print("[FAIL] real 'tool_call' blocks not counted -> footer would say "
              "'no actions taken' even though tools ran (the bug)")
        return False
    print("[ok] real 'tool_call' blocks are counted")
    return True


def _test_since_offset() -> bool:
    # `since` must skip earlier tool calls (so a warm follow-up only counts the
    # NEW calls, not the whole history).
    ctx = _ctx_with_real_tool_calls("create_group")
    ctx += _ctx_with_real_tool_calls("create_task")
    used = S._tool_names_used(ctx, since=1)
    if used != ["create_task"]:
        print(f"[FAIL] since-offset wrong: got {used}")
        return False
    print("[ok] since-offset counts only new tool calls")
    return True


def _test_footer_reports_actions() -> bool:
    ok = True
    # A task_manager that really created things, and a comms that only drafted.
    board = _Board([
        {"child": "task_manager", "status": "done",
         "tools_used": ["create_group", "create_task"]},
        {"child": "comms_agent", "status": "done", "tools_used": []},
    ])
    footer = S._ground_truth_footer(board)
    print(f"[footer] {footer}")
    if "group created" not in footer or "task created" not in footer:
        print("[FAIL] footer didn't credit task_manager's real actions")
        ok = False
    if "drafted" not in footer:
        print("[FAIL] footer didn't mark comms_agent as draft-only")
        ok = False
    if ok:
        print("[ok] footer credits real actions and flags draft-only")
    return ok


def _test_claims_sent_guard() -> bool:
    ok = True
    # With send_email actually in tools_used, a 'sent' claim is TRUTHFUL -> the
    # nudge must NOT fire. (Before the fix, send_email never made it into
    # tools_used, so this guard misfired on every real send.)
    if S._claims_sent_without_sending("STATUS: SENT", ["send_email"]):
        print("[FAIL] guard misfired on a real send (send_email present)")
        ok = False
    # Claims sent but no send_email -> guard SHOULD fire.
    if not S._claims_sent_without_sending("email sent successfully", []):
        print("[FAIL] guard missed a fake 'sent' claim")
        ok = False
    if ok:
        print("[ok] claims-sent guard: quiet on real send, fires on fake claim")
    return ok


def _test_file_guard() -> bool:
    ok = True
    # Scripter claims it can't find a file but never called a file tool -> the
    # runner SHOULD nudge it to actually look.
    if not S._claims_no_file_without_reading(
            "scripter", "I couldn't find any file named sales.xlsx.", []):
        print("[FAIL] file guard missed a 'cannot find' claim with no tool call")
        ok = False
    # It DID call list_uploaded_files -> a 'not found' is now honest, no nudge.
    if S._claims_no_file_without_reading(
            "scripter", "No such file in the uploads folder.",
            ["list_uploaded_files"]):
        print("[FAIL] file guard fired even though the scripter actually looked")
        ok = False
    # Only applies to the scripter, not other children.
    if S._claims_no_file_without_reading(
            "comms_agent", "couldn't find it", []):
        print("[FAIL] file guard should only apply to the scripter")
        ok = False
    if ok:
        print("[ok] file guard: nudges on blind 'not found', quiet once it looked")
    return ok


def _test_orchestrator_file_guard() -> bool:
    ok = True
    # File-related turn + orchestrator ducks (claims missing) -> should trigger.
    if not (S._looks_file_related("summarize the spreadsheet report.xlsx")
            and S._orchestrator_ducked_file("I couldn't find that file.")):
        print("[FAIL] orch guard should fire: file turn + 'couldn't find'")
        ok = False
    # Non-file turn must NOT be treated as file-related (avoid false positives).
    if S._looks_file_related("how many open tasks do I have?"):
        print("[FAIL] non-file turn wrongly flagged as file-related")
        ok = False
    # A normal answer (no missing-file claim) must not look ducked.
    if S._orchestrator_ducked_file("The report has 30 rows and 5 columns."):
        print("[FAIL] a normal file answer wrongly looks like a duck")
        ok = False
    if ok:
        print("[ok] orchestrator file guard: fires on file-turn duck, not otherwise")
    return ok


def main() -> int:
    results = [
        _test_real_blocks_counted(),
        _test_since_offset(),
        _test_footer_reports_actions(),
        _test_claims_sent_guard(),
        _test_file_guard(),
        _test_orchestrator_file_guard(),
    ]
    ok = all(results)
    print("\n" + ("[PASS] footer: real tool calls are counted, the footer "
                  "reports them truthfully, and the send guard is correct."
                  if ok else "[FAIL] see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
