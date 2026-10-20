"""Run-history join + the shared browser lock.

    .venv/Scripts/python.exe test_run_history.py
"""
import json
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import run_history as R

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
#  the two ledgers are joined into one timeline
# ===========================================================================
print("run history — actions.jsonl + usage.jsonl")

_d = tempfile.mkdtemp()
ACT = os.path.join(_d, "actions.jsonl")
USE = os.path.join(_d, "usage.jsonl")

ACTIONS = [
    {"ts": "2026-09-22T10:00:00+00:00", "session": "s1", "agent": "orchestrator",
     "tool": "delegate_task", "ok": True,
     "args_text": "child=browser_agent, description=Log in and create fd_account",
     "result": '{"task_id": "tsk_aaa", "delegated_to": "browser_agent"}'},
    {"ts": "2026-09-22T10:00:20+00:00", "session": "s1", "agent": "browser_agent",
     "tool": "browse_open", "ok": True, "args_text": "url=https://x/login",
     "result": "OK"},
    {"ts": "2026-09-22T10:00:30+00:00", "session": "s1", "agent": "browser_agent",
     "tool": "browse_click_text", "ok": False, "args_text": "text=Create Category",
     "result": "Element not found."},
    # A different conversation, so nothing from it may leak into s1.
    {"ts": "2026-09-22T11:00:00+00:00", "session": "s2", "agent": "orchestrator",
     "tool": "check_status", "ok": True, "args_text": "", "result": "{}"},
]
# The first usage row carries its session; the second does not, which is how
# every row written before usage_log started stamping one looks.
USAGE = [
    {"ts": "2026-09-22T10:00:05+00:00", "session": "s1", "agent": "orchestrator",
     "model": "gpt-4o", "input_tokens": 9000, "cache_input_tokens": 7000,
     "output_tokens": 300, "latency_s": 4.5, "est_usd": 0.02, "tools_offered": 4},
    {"ts": "2026-09-22T10:00:25+00:00", "agent": "browser", "model": "gpt-4o",
     "input_tokens": 5000, "cache_input_tokens": 4000, "output_tokens": 120,
     "latency_s": 2.0, "est_usd": 0.01, "tools_offered": 16},
]
for path, rows in ((ACT, ACTIONS), (USE, USAGE)):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
R.ACTIONS_PATH, R.USAGE_PATH = ACT, USE

ss = R.sessions()
check("both conversations are found", [s["session"] for s in ss], ["s2", "s1"])
s1 = [s for s in ss if s["session"] == "s1"][0]
check("its llm calls are counted", s1["llm_calls"], 2)
check("its tool calls too", s1["tool_calls"], 3)
check("and the failure is visible", s1["failed_tools"], 1)
check("tokens are summed", s1["input_tokens"], 14000)
check("cost is summed", round(s1["est_usd"], 4), 0.03)

# usage_log labels the agent from its model session id ("browser") while
# action_log uses the roster name ("browser_agent"). Un-normalised, one agent
# becomes two nodes and a 2-agent run claims four participants.
check("one agent is one node", sorted(s1["agents"]),
      ["browser_agent", "orchestrator"])
check("the alias resolves", R.canonical_agent("task-mgr"), "task_manager")
check("an unknown name is left alone", R.canonical_agent("mystery"), "mystery")

print()
print("run detail — ordered events and the delegation edges")
d = R.detail("s1")
check("every event is present", len(d["events"]), 5)
check("they are in time order",
      [e["ts"] for e in d["events"]] == sorted(e["ts"] for e in d["events"]), True)
check("and numbered for the UI", [e["seq"] for e in d["events"]], [1, 2, 3, 4, 5])
# The unstamped usage row belongs to s1 by timestamp -- useful, but a guess,
# and the viewer says so rather than presenting it as recorded fact.
_inf = [e for e in d["events"] if e.get("inferred")]
check("the unstamped row is attributed", len(_inf), 1)
check("and flagged as inferred", _inf[0]["agent"], "browser_agent")

check("the delegation is an edge", len(d["delegations"]), 1)
check("pointing at the right child", d["delegations"][0]["to"], "browser_agent")
check("carrying its task id", d["delegations"][0]["task_id"], "tsk_aaa")

# A refused delegation is drawn differently, because two HOLDs in one run is
# the signal that something upstream is wrong.
R.ACTIONS_PATH = os.path.join(_d, "a2.jsonl")
with open(R.ACTIONS_PATH, "w", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "ts": "2026-09-22T10:00:00+00:00", "session": "s3",
        "agent": "orchestrator", "tool": "delegate_task", "ok": True,
        "args_text": "child=browser_agent, description=x",
        "result": "ERROR: HOLD. The delegation you just wrote..."}) + "\n")
check("a refusal is marked", R.detail("s3")["delegations"][0]["refused"], True)
check("a missing session yields nothing", R.detail("nope"), None)
check("an empty id yields nothing", R.detail(""), None)

# ===========================================================================
#  two children sharing one Chrome must not run at once
# ===========================================================================
# roster.py's header states the assumption: "v1 only has ONE browser-using
# specialist live at a time (browser_agent xor comms_agent for a given task),
# so they can SHARE the default Chrome". Nothing enforced it -- the lock was
# keyed by child NAME. In the 15:56 run the orchestrator delegated
# browser_agent and comms_agent in the same turn; they took different locks,
# drove the same Chrome, and comms_agent's set_context() wiped the browser
# child's goal, checklist, step counter and trail mid-run. Its next two clicks
# produced no page record at all.
print()
print("browser-capable children share one lock")
try:
    import multi_agent.swarm as SW
    sw = SW.Swarm()
    b = sw._child_lock("browser_agent")
    c = sw._child_lock("comms_agent")
    truthy("browser_agent and comms_agent share a lock", b is c)
    truthy("scripter keeps its own", sw._child_lock("scripter") is not b)
    truthy("task_manager keeps its own", sw._child_lock("task_manager") is not b)
    truthy("and the two non-browser children are not merged either",
           sw._child_lock("scripter") is not sw._child_lock("task_manager"))
except ImportError as exc:
    print(f"  skip  swarm not importable here ({exc})")

# ===========================================================================
#  the full request/response is kept, and the ledgers stay small
# ===========================================================================
# usage.jsonl has token counts and actions.jsonl clips arguments to 48 chars
# and results to 400. Neither keeps what was sent or returned, so the only way
# to read a child's actual prompt was to re-derive it from the code.
print()
print("call payloads — complete request and response")

import call_log                                                    # noqa: E402
import action_log                                                  # noqa: E402

_pd = tempfile.mkdtemp()
call_log.LOG_PATH = os.path.join(_pd, "payloads.jsonl")
action_log.LOG_PATH = os.path.join(_pd, "actions.jsonl")
action_log.start_run("sess_payload")

BIG_ARGS = {"filename": "x.xlsx", "data": {"columns": ["A" * 400]}}
BIG_RESULT = "SAVED XLSX spreadsheet: " + "z" * 3000
act = action_log.record("scripter", "create_spreadsheet", BIG_ARGS,
                        True, BIG_RESULT)

truthy("the ledger row carries a payload id", act.get("call_id"))
truthy("the ledger's own copy is still clipped",
       len(act["result"]) <= action_log.RESULT_CHARS + 2)
truthy("and its arguments too", len(act["args_text"]) < 200)

full = call_log.read(act["call_id"])
truthy("the payload is readable by id", full is not None)
check("it knows what kind of call it was", full["kind"], "tool")
check("and which agent made it", full["agent"], "scripter")
truthy("the untruncated result is there", len(full["response"]) >= 3000)
truthy("and the untruncated arguments", "A" * 400 in str(full["request"]))
check("an unknown id yields nothing", call_log.read("call_nope"), None)
check("an empty id yields nothing", call_log.read(""), None)

# Whatever the model layer hands over -- Msg objects, content blocks, pydantic
# models -- must serialise. A payload that cannot be stored is worth losing; a
# turn is not, so nothing in here may raise.
class _Block:
    def __init__(self): self.type, self.text = "text", "hello"


class _Msg:
    def __init__(self): self.role, self.content = "user", [_Block()]


call_log.write_llm("call_llm1", "browser_agent", "gpt-4o",
                   [_Msg()], _Msg(), tools=[{"function": {"name": "browse_open"}}])
p = call_log.read("call_llm1")
truthy("an object graph serialises", p is not None)
check("the request keeps its messages",
      p["request"]["messages"][0]["role"], "user")
check("the nested block survives",
      p["request"]["messages"][0]["content"][0]["text"], "hello")
check("the tool names are recorded", p["request"]["tools"], ["browse_open"])

# A single enormous field is clipped rather than written whole, and says so.
call_log.write_llm("call_big", "scripter", "gpt-4o", ["q" * 300000], "ok")
big = call_log.read("call_big")
truthy("an oversized field is clipped",
       len(big["request"]["messages"][0]) < 300000)
truthy("and marked as truncated", "truncated" in big["request"]["messages"][0])

# The switch exists because these files grow fast on a long run.
call_log._enabled = False
n_before = sum(1 for _ in open(call_log.LOG_PATH, encoding="utf-8"))
call_log.write_tool("call_off", "scripter", "t", {}, "r")
n_after = sum(1 for _ in open(call_log.LOG_PATH, encoding="utf-8"))
check("CALL_LOG=0 writes nothing", n_after, n_before)
call_log._enabled = True

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
    sys.exit(1)
print("all run-history + payload checks passed")
