"""Fast, deterministic tests for the findings ledger (record_answer) -- NO LLM.

record_answer is the scripter's external scratchpad: it writes a sub-answer onto
the active TaskRecord and returns the running ledger (which re-injects progress
into the child's context). The final report is assembled from these. We verify
the write, the dedup-by-question, the per-child scoping, and the no-active-task
guard.

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_findings
"""

from __future__ import annotations

import sys

from multi_agent.bus import TaskRecord
from multi_agent import findings as FD


def _text(chunk) -> str:
    return chunk.content[0].text


def main() -> int:
    ok = True
    rec = TaskRecord(id="t1", child="scripter", description="multi-part task")
    record = FD.make_record_answer("scripter")

    # No active task yet -> guarded error, nothing recorded.
    if "ERROR" not in _text(record("q", "1")):
        print("[FAIL] record_answer should error with no active task")
        ok = False

    FD.set_active("scripter", rec)

    # First record lands on the rec and the ack echoes the running ledger.
    out = _text(record("discount breaches", "80"))
    if not rec.findings or rec.findings[0]["a"] != "80" or "80" not in out:
        print(f"[FAIL] first record not stored/echoed: {rec.findings} / {out!r}")
        ok = False
    else:
        print("[ok] record stored + ledger echoed back (re-injects progress)")

    # Second distinct question appends.
    record("unapproved big orders", "36")
    if len(rec.findings) != 2:
        print(f"[FAIL] expected 2 findings, got {len(rec.findings)}")
        ok = False

    # Re-recording the SAME question overwrites (no duplicate).
    record("discount breaches", "80 (rechecked)")
    same = [f for f in rec.findings if f["q"] == "discount breaches"]
    if len(rec.findings) != 2 or len(same) != 1 or same[0]["a"] != "80 (rechecked)":
        print(f"[FAIL] re-record should overwrite, not duplicate: {rec.findings}")
        ok = False
    else:
        print("[ok] re-recording a question overwrites (no duplicates)")

    # findings_summary is the deterministic 'Q = A; Q = A' assembly.
    summary = FD.findings_summary(rec)
    if "discount breaches = 80 (rechecked)" not in summary \
            or "unapproved big orders = 36" not in summary:
        print(f"[FAIL] summary wrong: {summary!r}")
        ok = False
    else:
        print(f"[ok] deterministic summary: {summary!r}")

    # Scoping: a different child's record_answer doesn't touch scripter's rec.
    rec2 = TaskRecord(id="t2", child="task_manager", description="x")
    FD.set_active("task_manager", rec2)
    FD.make_record_answer("task_manager")("something", "9")
    if len(rec.findings) != 2 or len(rec2.findings) != 1:
        print("[FAIL] findings leaked across children")
        ok = False
    else:
        print("[ok] per-child scoping holds (no cross-session leak)")

    FD.clear_active("scripter")
    FD.clear_active("task_manager")
    if FD._active:
        print("[FAIL] clear_active left stale entries")
        ok = False

    print("\n" + ("[PASS] findings: record/echo/overwrite/summary/scoping all "
                  "correct." if ok else "[FAIL] see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
