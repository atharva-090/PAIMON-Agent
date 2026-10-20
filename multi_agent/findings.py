"""Per-task findings ledger -- the scripter's external scratchpad.

The weak ChatGPT-web brain is bad at holding many tool results in its head and
re-aggregating them in one final message: it eyeballs capped row dumps and mixes
up which number answers which sub-question (that's how a 5-part task came back
with 3 wrong numbers even though the tools returned the right ones).

So we give it EXTERNAL memory. The moment it has a definite value for a part of
the task, it calls record_answer(question, answer); the system stores it on the
task's board record, and record_answer RETURNS the running ledger -- which
re-injects the progress into the child's own context every time it records,
WITHOUT a middleware. The final report is then assembled deterministically from
the recorded findings (see swarm._run_specialist), not from the brain trying to
re-aggregate everything at the end.

Scoping: findings live on the active TaskRecord, looked up per child via a small
module registry the swarm runner sets around each delegation. Only one task runs
per child at a time (the per-child lock), so the registry never races within a
child; different children use different keys, so a recording can never bleed
into another agent's session.
"""

from __future__ import annotations

from agentscope.message import TextBlock
from agentscope.tool import ToolChunk

from .bus import TaskRecord


# child name -> the TaskRecord currently running for that child.
_active: dict[str, TaskRecord] = {}


def set_active(child: str, rec: TaskRecord) -> None:
    """Mark ``rec`` as the task record record_answer should write to for
    ``child``. Called by the runner before a delegation runs."""
    _active[child] = rec


def clear_active(child: str) -> None:
    """Drop the active record for ``child`` (called when its delegation ends)."""
    _active.pop(child, None)


def findings_summary(rec: TaskRecord) -> str:
    """One-line 'Q = A; Q = A' of everything recorded on ``rec`` (or '')."""
    if not rec or not rec.findings:
        return ""
    return "; ".join(f"{f['q']} = {f['a']}" for f in rec.findings)


def make_record_answer(child: str):
    """Build a record_answer tool bound to one ``child`` (so it writes to that
    child's active task). Returned as a plain function for FunctionTool."""

    def record_answer(question: str, answer: str, note: str = "") -> ToolChunk:
        """Record ONE finished sub-answer to your external scratchpad. Call this
        the moment you have a definite value for a part of the task -- a count, a
        total, a policy figure -- BEFORE moving on, so you never have to remember
        it. Your final report is built from what you record here, so record every
        sub-answer exactly once.

        Args:
            question (str): Which part you're answering, in a few words (e.g.
                'orders over discount cap', 'Chen total amount').
            answer (str): The exact value, COPIED from the tool output -- a COUNT,
                an excel_aggregate total, a quoted figure. Never estimate or count
                rows by eye.
            note (str): Optional one-line context (which tool / which condition).
        """
        rec = _active.get(child)
        if rec is None:
            return _tc("ERROR: no active task to record into.")
        q = (question or "").strip()
        a = (answer or "").strip()
        if not q or not a:
            return _tc("ERROR: record_answer needs both a question and an answer.")
        # Overwrite a re-recorded question instead of duplicating it.
        for f in rec.findings:
            if f["q"].lower() == q.lower():
                f["a"], f["note"] = a, (note or "").strip()
                break
        else:
            rec.findings.append({"q": q, "a": a, "note": (note or "").strip()})
        return _tc(
            f"Recorded: {q} = {a}.\n"
            f"Your answers so far ({len(rec.findings)}): {findings_summary(rec)}\n"
            "Keep going -- answer each remaining part of the task and record it "
            "the same way. When every part is recorded, write your final report "
            "stating exactly these recorded answers.")

    record_answer.__name__ = "record_answer"
    return record_answer


def _tc(text: str) -> ToolChunk:
    return ToolChunk(content=[TextBlock(type="text", text=text)])
