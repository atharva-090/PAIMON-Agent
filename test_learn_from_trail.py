"""Step 4: learn from the user's words + the action trail, not the transcript.

The old input was the whole transcript -- every assistant turn included. That is
both the bulk of the tokens and the least trustworthy part of the record: the
model narrating its own progress, which has been wrong in production (it once
reported sending a Meet link it had never created).

The deterministic extractor never trusted assistant turns either. This makes the
LLM path agree with it, and hands the model a mechanical record of what really
happened so it cannot learn a failure as a success.

Run:  .venv/Scripts/python.exe test_learn_from_trail.py
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

import action_log
import memory_learn
import sessions
import wiki_store

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


# --------------------------------------------------------- prompt assembly
print("\nprompt assembly")

p = memory_learn._extract_prompt(user_text="Email the report to Umesh.",
                                 actions="comms: send_email(to=u@x.com) → ok")
truthy("includes the user's words", "Email the report to Umesh." in p)
truthy("includes the actions", "send_email" in p)
truthy("labels actions as ground truth", "ground truth" in p)
check("no transcript section when both present", "Conversation:" in p, False)

p2 = memory_learn._extract_prompt(transcript="User: hi\nYou: hello")
truthy("falls back to transcript when nothing else", "Conversation:" in p2)

truthy("system prompt forbids learning failures as successes",
       "FAILED" in memory_learn._EXTRACT_SYS
       and "did not happen" in memory_learn._EXTRACT_SYS)

# ------------------------------------------------------- end-to-end, no LLM
print("\nend-to-end (fake llm, temp stores)")

tmp = tempfile.mkdtemp(prefix="learn-test-")
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

seen: dict = {}


async def fake_llm(system: str, user: str) -> str:
    seen["system"], seen["user"] = system, user
    return '[{"entity": "Umesh Patil", "fact": "Umesh is the vendor PM.", ' \
           '"type": "person", "confidence": 0.9}]'


try:
    os.makedirs(sessions.SESSIONS_DIR, exist_ok=True)
    sessions.reindex()

    # A conversation with a LOT of assistant prose -- the part we stop sending.
    turns = [
        {"role": "user", "text": "Umesh Patil is the vendor PM. Email him the report."},
        {"role": "assistant", "text": "Certainly! " + ("Let me think about this. " * 220)},
        {"role": "user", "text": "thanks"},
        {"role": "assistant", "text": "Done! " + ("I have completed the task. " * 220)},
    ]
    rec = sessions.save_session(turns=turns, title="Vendor", summary="s")
    sid = rec["id"]

    action_log.set_session(sid)
    action_log.record("comms", "send_email", {"to": "umesh@example.com"},
                      result='{"ok": true, "id": "m_1"}')
    action_log.record("comms", "browse_click", {"text": "Attach files"},
                      result="Error: element not found")

    out = asyncio.run(memory_learn.learn_from_session(sid, llm=fake_llm))
    check("facts written", out["facts"], 1)
    check("used the llm path", out["via"], "llm")
    check("reports that actions were available", out["actions"], True)

    sent = seen["user"]
    truthy("user words sent", "Umesh Patil is the vendor PM" in sent)
    truthy("actions sent", "send_email" in sent)
    truthy("the FAILED action is visible to the extractor", "FAILED" in sent)

    # The whole point: assistant narration is no longer paid for.
    truthy("assistant prose NOT sent", "Let me think about this" not in sent)
    truthy("assistant success claim NOT sent",
           "I have completed the task" not in sent)

    full = sessions.transcript_text(sid)
    saving = 100.0 * (1 - len(sent) / max(1, len(full)))
    print(f"       prompt {len(sent)} chars vs transcript {len(full)} "
          f"({saving:.0f}% smaller)")
    truthy("materially smaller than the transcript", len(sent) < len(full))

    # The fact landed in the wiki.
    page = wiki_store.read_page("umesh-patil")
    truthy("wiki page created", page is not None)
    truthy("fact recorded",
           page and any("vendor PM" in f["text"]
                        for f in wiki_store.parse_facts(page["body"])))

    # In-memory trail released; the durable jsonl copy remains.
    check("trail cleared after learning", len(action_log.trail(sid)), 0)
    truthy("durable copy kept", os.path.exists(action_log.LOG_PATH))

    # ---------------------------------------------- session with no actions
    print("\nsession with no recorded actions")
    rec2 = sessions.save_session(
        turns=[{"role": "user", "text": "Priya prefers morning meetings."},
               {"role": "assistant", "text": "Noted."}],
        title="Chat", summary="s")
    out2 = asyncio.run(memory_learn.learn_from_session(rec2["id"], llm=fake_llm))
    check("still learns without a trail", out2["facts"], 1)
    check("reports no actions", out2["actions"], False)
    truthy("user words still sent", "Priya prefers morning" in seen["user"])

    # ------------------------------------------------ deterministic fallback
    print("\nfallback when the brain is down")

    async def dead_llm(system: str, user: str) -> str:
        raise RuntimeError("backend down")

    rec3 = sessions.save_session(
        turns=[{"role": "user", "text": "Remember that I prefer plain-text email."}],
        title="Pref", summary="s")
    out3 = asyncio.run(memory_learn.learn_from_session(rec3["id"], llm=dead_llm))
    # 5b made `via` report WHY the llm produced nothing as well as what saved
    # it, so assert the deterministic rescue rather than one frozen string.
    truthy(f"falls back deterministically (via={out3['via']})",
           out3["via"].endswith("+deterministic"))
    truthy("and says the backend was the problem",
           out3["via"].startswith("llm-error"))
    truthy("still learned something", out3["facts"] >= 1)
finally:
    (wiki_store.WIKI_DIR, wiki_store.INDEX_FILE, wiki_store.FTS_DB,
     memory_learn.DIGEST_FILE, action_log.LOG_PATH,
     sessions.SESSIONS_DIR, sessions.INDEX_FILE, sessions.FTS_DB) = real
    wiki_store._fts_ready = None
    action_log.set_session("")
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    raise SystemExit(1)
print("all learn-from-trail checks passed")
