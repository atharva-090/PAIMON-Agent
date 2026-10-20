"""Step 2: the deterministic action ledger.

The point of this module is that it CANNOT lie. Every line is derived from the
tool name, its arguments and its real result -- never from the model's account
of what it did, because that account has already been wrong in production (the
agent wrote a reminder instead of a Meet link and reported success).

Run:  .venv/Scripts/python.exe test_action_log.py
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

import action_log

FAILS: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILS.append(label)


def truthy(label: str, got) -> None:
    ok = bool(got)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        FAILS.append(label)


tmp = tempfile.mkdtemp(prefix="actionlog-test-")
real_log = action_log.LOG_PATH
action_log.LOG_PATH = os.path.join(tmp, "actions.jsonl")

try:
    # ------------------------------------------------------ argument rendering
    print("\nargument rendering")
    check("salient keys come first",
          action_log._fmt_args({"verbose": True, "to": "umesh@x.com"}),
          "to=umesh@x.com, verbose=true")
    check("lists become a count",
          action_log._fmt_args({"attachments": ["a.docx", "b.pdf"]}),
          "attachments=[2]")

    # An email body must never be inlined into a line that reaches a prompt.
    big = action_log._fmt_args({"body": "x" * 5000})
    truthy("huge payload reported as a size", "<5000 chars>" in big)
    truthy("huge payload content not inlined", "xxxxxxxxxx" not in big)

    many = action_log._fmt_args({f"k{i}": i for i in range(9)})
    truthy("key count is capped", many.endswith("+5"))
    check("no args renders empty", action_log._fmt_args(None), "")

    # AgentScope hands on_acting a ToolCallBlock whose `input` is a JSON STRING,
    # not a mapping. The first version only accepted a dict, so a real run
    # recorded "delegate_task() -> ok" with the child and task name missing --
    # the ledger looked fine and carried no information.
    check("JSON-string input is parsed (the real framework shape)",
          action_log._fmt_args('{"child": "browser_agent", "description": "Research X"}'),
          "child=browser_agent, description=Research X")
    check("non-JSON string is kept, not dropped",
          action_log._fmt_args("plain text"), "input=plain text")
    check("empty string renders empty", action_log._fmt_args(""), "")

    # ------------------------------------------------------ failure detection
    print("\nfailure detection (a tool can 'succeed' and still report an error)")
    truthy("structured ok=false", action_log.looks_failed('{"ok": false, "e": 1}'))
    truthy("leading 'Error:'", action_log.looks_failed("Error: element not found"))
    truthy("leading 'could not'", action_log.looks_failed("could not resolve path"))
    truthy("bulleted leading marker", action_log.looks_failed("- failed to send"))

    # The false positive that would matter: research prose mentioning failure.
    check("mid-text 'error' is NOT a failure",
          action_log.looks_failed(
              "Term loans carry risk; a payment error can trigger default."),
          False)
    check("empty is not a failure", action_log.looks_failed(""), False)

    # ------------------------------------------------------------- recording
    print("\nrecording")
    action_log.set_session("s1")
    action_log.record("comms", "send_email",
                      {"to": "umesh@x.com", "subject": "Term loans"},
                      ok=True, result='{"ok": true, "id": "abc"}', ms=120)

    # ok=True passed in, but the RESULT says otherwise -- ground truth wins.
    action_log.record("browser", "browse_click", {"text": "Attach files"},
                      ok=True, result="Error: element not found", ms=90)

    t = action_log.trail("s1")
    check("both recorded", len(t), 2)
    check("genuine success stays ok", t[0]["ok"], True)
    check("error-string result is marked FAILED", t[1]["ok"], False)

    line = action_log.line(t[1])
    truthy("failed line says FAILED", "FAILED" in line)
    truthy("failed line names the tool", "browse_click" in line)
    truthy("failed line keeps the arg", "Attach files" in line)

    # ------------------------------------------------------------- filtering
    print("\nfiltering and rendering")
    check("filter by agent", len(action_log.trail("s1", agent="comms")), 1)
    check("limit takes the newest",
          action_log.trail("s1", limit=1)[0]["tool"], "browse_click")
    truthy("render tags the agent", "comms:" in action_log.render("s1"))
    truthy("render can omit the agent",
           "comms:" not in action_log.render("s1", with_agent=False))

    s = action_log.stats("s1")
    check("stats counts actions", s["actions"], 2)
    check("stats counts failures", s["failed"], 1)

    # ------------------------------------------------------------ persistence
    truthy("appended to jsonl", os.path.exists(action_log.LOG_PATH))
    with open(action_log.LOG_PATH, encoding="utf-8") as fh:
        check("one json line per action", len(fh.read().strip().splitlines()), 2)

    # ----------------------------------------------------------- run rebinding
    print("\nrun key -> session id")
    key = action_log.start_run("chat")
    truthy("start_run makes it current", action_log.current_session() == key)
    action_log.record("hermes", "view_board", {}, ok=True, result="ok")
    check("recorded under the run key", len(action_log.trail(key)), 1)

    moved = action_log.rebind(key, "ses_20260921_abcd")
    check("rebind moves the trail", moved, 1)
    check("old key is empty", len(action_log.trail(key)), 0)
    check("new key has it", len(action_log.trail("ses_20260921_abcd")), 1)
    check("current follows the rebind",
          action_log.current_session(), "ses_20260921_abcd")
    check("rebinding nothing is a no-op", action_log.rebind("nope", "x"), 0)

    # ------------------------------------------------------------------- cap
    print("\nbounds")
    action_log.set_session("flood")
    for i in range(action_log.MAX_PER_SESSION + 25):
        action_log.record("browser", "browse_snapshot", {"i": i}, result="ok")
    check("trail is capped",
          len(action_log.trail("flood")), action_log.MAX_PER_SESSION)
    truthy("cap keeps the NEWEST",
           action_log.trail("flood", limit=1)[0]["args_text"].endswith("524"))

    # ------------------------------------- middleware records without a UI open
    print("\nmiddleware integration (records with NO subscribers)")
    import agent_events

    class FakeCall:
        id = "call_1"
        name = "create_meeting"
        input = {"title": "Loan review", "when": "tomorrow 5 PM"}

    class FakeChunk:
        content = "ok, meet.google.com/abc-defg-hij"

    async def fake_next(**kwargs):
        yield FakeChunk()

    async def drive():
        em = agent_events.StreamEmitter("comms")
        out = []
        async for item in em.on_acting(None, {"tool_call": FakeCall()}, fake_next):
            out.append(item)
        return out

    action_log.set_session("nosub")
    check("no UI subscribers", agent_events.has_listeners(), False)
    passed = asyncio.run(drive())
    check("tool output re-yielded unchanged", len(passed), 1)

    rec = action_log.trail("nosub")
    check("recorded anyway", len(rec), 1)
    check("tool name captured", rec[0]["tool"], "create_meeting")
    truthy("args captured", "Loan review" in rec[0]["args_text"])
    truthy("result captured", "meet.google.com" in rec[0]["result"])
    check("marked ok", rec[0]["ok"], True)
finally:
    action_log.LOG_PATH = real_log
    action_log.set_session("")
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    raise SystemExit(1)
print("all action-log checks passed")
