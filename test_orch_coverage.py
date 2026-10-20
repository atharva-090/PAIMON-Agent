"""The orchestrator must defer work it cannot plan yet, pick it up when the
report lands, and know what day it is.

    .venv/Scripts/python.exe test_orch_coverage.py

Run swarm-1792474238077 planned every delegation in its first 9 seconds and
never re-planned, so "book a demo with 3-4 of them" -- which needs the vendor
list from the research -- got no task at all, and the run answered without ever
mentioning the piece it had skipped. It also wrote "Thursday 26 March 2026"
into a meeting delegation sent on 20 October.
"""
import asyncio
import json
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAILS = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILS.append(label)


def truthy(label, got):
    print(f"  {'ok  ' if got else 'FAIL'} {label}")
    if not got:
        FAILS.append(label)


# ===========================================================================
#  the persona carries the duty
# ===========================================================================
print("the orchestrator is told to defer, re-plan, and count")
from multi_agent.roster import ORCHESTRATOR_PERSONA as P        # noqa: E402

truthy("a piece it cannot plan yet is deferred, not dropped",
       "DEFERRED, not finished and not dropped" in P)
truthy("the report landing is when deferred work gets picked up",
       "which piece was waiting on exactly this" in P)
truthy("it counts the pieces before answering",
       "count the pieces" in P and "every piece has a task" in P)
truthy("and it must name a piece it could not do",
       "say which piece and why, by name" in P)

steps = [int(m) for m in re.findall(r"^(\d+)\. ", P, re.M)]
check("the workflow is still numbered 1..9", steps, list(range(1, 10)))


# ===========================================================================
#  the nudge rides in the tool result, where the persona cannot decay
# ===========================================================================
print()
print("a finished task tells the orchestrator to re-plan")
import multi_agent.swarm as S                                   # noqa: E402

ROW = {"id": "tsk_1", "child": "browser_agent", "status": "done",
       "elapsed_s": 12.0, "tool_calls": 3, "tools_used": ["browse_open"],
       "report": "found 4 vendors"}


class _Board:
    def __init__(self, version):
        self._v = version

    async def wait_for_change(self, since_version, timeout):
        return self._v

    def current_snapshot(self):
        return [dict(ROW)]


class _Swarm:
    def __init__(self, version):
        self.board = _Board(version)
        self._last_seen_version = 0


def _call(version):
    real = S._current
    S._current = lambda: _Swarm(version)
    try:
        out = asyncio.run(S.wait_for_results(1))
    finally:
        S._current = real
    import browser_tools
    return json.loads(browser_tools.chunk_text(out))

done = _call(5)
truthy("a transition carries the re-plan instruction", "next" in done)
truthy("it points at the user's request",
       "Re-read the user's request" in done.get("next", ""))
truthy("and forbids answering with a piece unstarted",
       "has no task on the board" in done.get("next", ""))
truthy("the report still rides along for a finished task",
       done["current"][0].get("report") == "found 4 vendors")

# A timeout is not a transition: nothing finished, so there is nothing to
# re-plan against and the nudge would be noise on every poll.
timed = _call(-1)
truthy("a timeout says so", timed.get("timed_out") is True)
truthy("a timeout carries no re-plan nudge", "next" not in timed)


# ===========================================================================
#  it knows what day it is without asking
# ===========================================================================
print()
print("the orchestrator is handed the clock")
import scheduler                                                 # noqa: E402

line = ("NOW: " + scheduler.now_iso() + " (" + scheduler.now().strftime("%A")
        + "). Resolve every relative date against this before you write "
        "it into a delegation; never infer today's date.")
truthy("the line names the real date",
       scheduler.now().strftime("%Y-%m-%d") in line)
truthy("and the real weekday", scheduler.now().strftime("%A") in line)

src = open(S.__file__, encoding="utf-8").read()
truthy("swarm builds that line for every orchestrator turn",
       'extras.append(' in src and '"NOW: "' in src)
truthy("it is built from the scheduler, not from the model",
       "import scheduler" in src.split("extras: list[str] = []")[1][:900])

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
    sys.exit(1)
print("all orchestrator coverage checks passed")
