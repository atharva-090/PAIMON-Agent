"""Step 5: the four defects a real swarm run exposed.

Each of these was invisible to the unit tests and only showed up when a real
36-action run was read back line by line.

Run:  .venv/Scripts/python.exe test_run_fixes.py
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

import action_log
import agent_tools

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


# ------------------------------------------------- 5c: delegation description
print("\n5c — delegate_task keeps its description")
# The run logged `delegate_task(child=scripter, description=<582 chars>)`,
# recording that work was handed off but not what was asked for.
long_desc = ("Using the browser agent findings, write a DOCX covering business "
             "identity verification for individuals and non-individuals, citing "
             "Muthoot and HDFC, and save it under project/reports.")
out = action_log._fmt_args({"child": "scripter", "description": long_desc})
truthy("child is named", "child=scripter" in out)
truthy("description is readable, not a size", "<" not in out.split("description=")[1][:12])
truthy("the actual ask survives", "individuals and non-individuals" in out)

# But a payload must STILL be sized -- that guard is what keeps an email body
# or a generated report out of every prompt the trail is injected into.
big = action_log._fmt_args({"to": "u@x.com", "body": "y" * 3000})
truthy("email body still sized", "body=<3000 chars>" in big)
truthy("body content never inlined", "yyyyyyyyyy" not in big)
# A long description is capped too, just far later than a generic value.
huge = action_log._fmt_args({"description": "z" * 5000})
truthy("absurd description is still bounded", len(huge) < 400)

# --------------------------------------------------- 5d: recipient resolution
print("\n5d — send_email resolves a bare name")
# The run did: lookup_contact(Lesley) -> resolved ... then send_email(to=Lesley)
# -> SMTPRecipientsRefused 553. The model had the address and passed the name.
addr, note = agent_tools._resolve_recipient("lesley@example.com")
check("a real address passes through untouched",
      addr, "lesley@example.com")
check("no note when nothing was resolved", note, "")

addr, note = agent_tools._resolve_recipient("  a@b.com, c@d.com ")
truthy("multi-recipient string left alone", "a@b.com" in addr and "c@d.com" in addr)

addr, note = agent_tools._resolve_recipient("Nobody McNothing")
check("unknown name does NOT reach SMTP", addr, "")
truthy("and explains what to do instead",
       "lookup_contact" in note and "add_contact" in note)

addr, note = agent_tools._resolve_recipient("")
check("empty recipient rejected", addr, "")
truthy("empty gives a reason", "no recipient" in note.lower())

# ------------------------------------------------------------- 5b: via label
print("\n5b — 'via' distinguishes the three outcomes")
import memory_learn                                            # noqa: E402
import sessions                                                # noqa: E402
import wiki_store                                              # noqa: E402

tmp = tempfile.mkdtemp(prefix="runfix-")
real = (wiki_store.WIKI_DIR, wiki_store.INDEX_FILE, wiki_store.FTS_DB,
        memory_learn.DIGEST_FILE, action_log.LOG_PATH,
        sessions.SESSIONS_DIR, sessions.INDEX_FILE, sessions.FTS_DB)
wiki_store.WIKI_DIR = os.path.join(tmp, "wiki")
wiki_store.INDEX_FILE = os.path.join(tmp, "wiki", "_index.json")
wiki_store.FTS_DB = os.path.join(tmp, "wiki", "fts.db")
wiki_store._fts_ready = None
memory_learn.DIGEST_FILE = os.path.join(tmp, "digest.md")
action_log.LOG_PATH = os.path.join(tmp, "actions.jsonl")
sessions.SESSIONS_DIR = os.path.join(tmp, "sessions")
sessions.INDEX_FILE = os.path.join(tmp, "sessions", "index.json")
sessions.FTS_DB = os.path.join(tmp, "sessions", "fts.db")

try:
    os.makedirs(sessions.SESSIONS_DIR, exist_ok=True)
    sessions.reindex()

    def mk(text="Discussed the weather briefly."):
        return sessions.save_session(
            turns=[{"role": "user", "text": text},
                   {"role": "assistant", "text": "ok"}],
            title="t", summary="s")["id"]

    async def good(sysp, u):
        return '[{"entity":"Umesh","fact":"Umesh is the PM.",' \
               '"type":"person","confidence":0.9}]'

    async def empty(sysp, u):
        return "[]"                      # ran fine, honestly found nothing

    async def dead(sysp, u):
        raise RuntimeError("backend down")

    async def prose(sysp, u):
        return "I could not find any durable facts."   # replied, not JSON

    check("facts extracted -> llm",
          asyncio.run(memory_learn.learn_from_session(mk(), llm=good))["via"],
          "llm")

    # The bug: this used to report "deterministic", implying the brain was
    # down, when the brain had answered perfectly.
    check("LLM honestly returned [] -> llm-empty",
          asyncio.run(memory_learn.learn_from_session(mk(), llm=empty))["via"],
          "llm-empty")
    check("LLM raised -> llm-error",
          asyncio.run(memory_learn.learn_from_session(mk(), llm=dead))["via"],
          "llm-error")
    check("LLM replied with prose -> llm-unparsed",
          asyncio.run(memory_learn.learn_from_session(mk(), llm=prose))["via"],
          "llm-unparsed")

    # When the deterministic path actually rescues something, say so.
    via = asyncio.run(memory_learn.learn_from_session(
        mk("Remember that I prefer plain-text email."), llm=dead))["via"]
    truthy(f"deterministic rescue is reported ({via})",
           via.endswith("+deterministic"))
finally:
    (wiki_store.WIKI_DIR, wiki_store.INDEX_FILE, wiki_store.FTS_DB,
     memory_learn.DIGEST_FILE, action_log.LOG_PATH,
     sessions.SESSIONS_DIR, sessions.INDEX_FILE, sessions.FTS_DB) = real
    wiki_store._fts_ready = None
    action_log.set_session("")
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------ 5a: new-chat wiring
print("\n5a — 'New chat' ends the session on the SERVER")
ui = open("frontend/chat-ui/src/lib/useChat.ts", encoding="utf-8").read()
truthy("clear() calls the reset endpoint",
       '"/api/agent/new-session"' in ui and "method: \"POST\"" in ui)
truthy("a failed reset cannot strand the UI", "catch {" in ui)

built = "frontend/chat"
js = [f for f in os.listdir(os.path.join(built, "assets"))
      if f.endswith(".js")] if os.path.isdir(built) else []
truthy("built bundle exists", bool(js))
if js:
    blob = open(os.path.join(built, "assets", js[0]), encoding="utf-8",
                errors="ignore").read()
    truthy("endpoint is present in the SHIPPED bundle",
           "/api/agent/new-session" in blob)

srv = open("server.py", encoding="utf-8").read()
truthy("server exposes the endpoint", '"/api/agent/new-session"' in srv)
truthy("and it resets the swarm", "reset_swarm" in srv)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    raise SystemExit(1)
print("all run-fix checks passed")
