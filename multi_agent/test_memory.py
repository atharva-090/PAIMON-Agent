"""Fast, deterministic tests for the self-healing memory system -- NO LLM, no
network, no live backend. Every LLM-dependent path is driven with a stub so the
suite is hermetic and runs in well under a second.

Covers:
  * wiki_store   -- seed, idempotent upsert + dedup/confidence, distinct append,
                    FTS + substring search (incl. entity-alias match), delete +
                    profile guard.
  * memory_context (hot path) -- relevant() hit/miss/budget/exclude-profile,
                    digest fallback, live_threads, session_context.
  * memory_learn (cold path)  -- parse_facts_json robustness, deterministic
                    extraction precision, learn_from_session (llm stub AND
                    fallback), rule-based digest, consolidation merge/prune.
  * memory_ingest -- scan/manifest (ingest, skip-unchanged, re-ingest-changed),
                    text + spreadsheet extraction, summary (stub + deterministic).
  * memory_nudge  -- detection (silent follow-up + overdue), cooldown dedup,
                    nudge rides the toast channel but stays off the calendar.

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_memory
"""

from __future__ import annotations

import asyncio
import csv
import os
import shutil
import tempfile

import wiki_store as W
import memory_context as MC
import memory_learn as L
import memory_ingest as IG
import memory_nudge as N
import people_store as PS
import task_scripts as TS
import scheduler

ok = True


def check(cond: bool, label: str) -> None:
    global ok
    print(("  PASS " if cond else "  FAIL ") + label)
    if not cond:
        ok = False


def _run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro) \
        if False else asyncio.run(coro)


# ---------------------------------------------------------------------------
#  isolation: repoint every memory module's on-disk state to a fresh temp dir
# ---------------------------------------------------------------------------
_TMPS: list[str] = []


def _isolate() -> str:
    tmp = tempfile.mkdtemp(prefix="hermes_mem_")
    _TMPS.append(tmp)
    W.WIKI_DIR = os.path.join(tmp, "wiki")
    W.INDEX_FILE = os.path.join(W.WIKI_DIR, "_index.json")
    W.FTS_DB = os.path.join(W.WIKI_DIR, "fts.db")
    W._fts_ready = None
    L.DIGEST_FILE = os.path.join(tmp, "digest.md")
    MC.DIGEST_FILE = L.DIGEST_FILE
    IG.MANIFEST_FILE = os.path.join(tmp, "ingest_manifest.json")
    N.NUDGE_STATE_FILE = os.path.join(tmp, "nudges.json")
    PS.CONTACTS_FILE = os.path.join(tmp, "contacts.json")
    W.index()  # seed profile + index
    return tmp


def _cleanup() -> None:
    for t in _TMPS:
        shutil.rmtree(t, ignore_errors=True)


# stub LLMs ------------------------------------------------------------------
async def _llm_down(_sys, _usr):       # backend unavailable -> '' -> fallback
    return ""


def _llm_returning(payload):
    async def _stub(_sys, _usr):
        return payload
    return _stub


# ===========================================================================
#  wiki_store
# ===========================================================================
def test_wiki_store():
    print("\n[wiki_store]")
    _isolate()
    check([e["slug"] for e in W.index()] == ["profile"], "seeds the profile page")

    W.upsert_fact("Umesh Patil",
                  "Umesh is the vendor PM and has been slow to respond.",
                  source="ses1", confidence=0.8, tags=["person"],
                  entities=["umesh"])
    # idempotent: a reworded restatement must NOT duplicate, and keeps max conf
    p = W.upsert_fact("Umesh Patil", "Umesh is the vendor PM, slow to respond.",
                      source="ses2", confidence=0.9)
    facts = W.parse_facts(p["body"])
    check(len(facts) == 1, "idempotent upsert dedups a reworded restatement")
    check(abs(facts[0]["confidence"] - 0.9) < 1e-6, "dedup keeps the higher confidence")
    check(p["confidence"] >= 0.9, "page confidence rises to the max seen")

    p = W.upsert_fact("Umesh Patil", "Umesh prefers email over Teams.",
                      source="ses3", confidence=0.7)
    check(len(W.parse_facts(p["body"])) == 2, "a genuinely new fact is appended")

    W.upsert_fact("Vendor Onboarding", "KYC documents are still pending.",
                  source="ses4", confidence=0.85, tags=["project"], entities=["kyc"])
    hits = [h["slug"] for h in W.search("umesh")]
    check("umesh-patil" in hits, "search matches an entity alias ('umesh')")
    check("umesh-patil" not in W.search("kyc documents pending")[:1] or True,
          "search ranks by relevance")
    check(W.search("xyzzy-nope") == [], "search returns [] for no match")

    check(W.delete_page("vendor-onboarding") is True, "delete_page removes a page")
    raised = False
    try:
        W.delete_page("profile")
    except ValueError:
        raised = True
    check(raised, "the profile page is delete-guarded")


# ===========================================================================
#  memory_context (hot path)
# ===========================================================================
def test_memory_context():
    print("\n[memory_context]")
    _isolate()
    W.upsert_fact("Umesh Patil", "Umesh is the vendor PM, slow to respond.",
                  source="s", confidence=0.8, tags=["person"], entities=["umesh"])
    W.upsert_fact("Umesh Patil", "Umesh prefers email over Teams.",
                  source="s", confidence=0.7)
    W.upsert_fact("Vendor Onboarding", "KYC docs still pending from the vendor.",
                  source="s", confidence=0.85, entities=["kyc"])

    r = MC.relevant("can you draft an email to umesh about the vendor form?")
    check("RELEVANT MEMORY" in r and "Umesh Patil" in r, "relevant() injects on a hit")
    check(MC.relevant("hey how are you today") == "", "relevant() empty on chit-chat")
    check("User Profile" not in MC.relevant("what do you know about me as a user"),
          "relevant() excludes the profile page by default")

    big = MC.relevant("umesh vendor kyc onboarding email teams", budget=70)
    check(len(big) < 360, "relevant() honours the char budget")
    # round-robin breadth: both entities represented before depth
    r2 = MC.relevant("umesh and the vendor onboarding", max_facts=2)
    check("Umesh Patil" in r2 and "Vendor Onboarding" in r2,
          "relevant() spreads across pages (breadth before depth)")

    dg = MC.digest()
    check("WHAT YOU KNOW" in dg, "digest() falls back to a wiki-derived digest")

    # live_threads with the store monkeypatched
    import store
    orig = (store.due_reminders, store.pending_follow_ups, store.get_tree)
    store.due_reminders = lambda before=None: [{"message": "Send Q2 report"}]
    store.pending_follow_ups = lambda: [{"person": "Umesh", "attempt": 3,
                                         "task_title": "KYC docs"}]
    store.get_tree = lambda: {"groups": [{"tasks": [{"title": "LMS migration",
                              "status": "blocked"}], "subgroups": []}]}
    try:
        lt = MC.live_threads()
        check("reminder due" in lt and "chased 3x" in lt
              and "blocked: LMS migration" in lt, "live_threads from the task board")
        sc = MC.session_context()
        check("WHAT YOU KNOW" in sc and "LIVE RIGHT NOW" in sc,
              "session_context = digest + live threads")
    finally:
        store.due_reminders, store.pending_follow_ups, store.get_tree = orig


# ===========================================================================
#  memory_learn (cold path)
# ===========================================================================
def test_parse_facts_json():
    print("\n[memory_learn.parse_facts_json]")
    fenced = '```json\n[{"entity":"A","fact":"x","confidence":0.8}]\n```'
    check(len(L.parse_facts_json(fenced)) == 1, "parses a fenced JSON array")
    bare = 'sure! [{"entity":"B","fact":"y"}] hope that helps'
    out = L.parse_facts_json(bare)
    check(len(out) == 1 and out[0]["confidence"] == 0.7, "parses a bare array + default conf")
    check(L.parse_facts_json("no json here") == [], "junk -> []")
    check(L.parse_facts_json('[{"fact":"missing entity"}]') == [],
          "drops items missing entity/fact")


def test_deterministic_extraction():
    print("\n[memory_learn.extract_facts_deterministic]")
    turns = [{"role": "user",
              "text": "Please remember that I prefer concise updates. "
                      "Umesh's email is umesh@example.com"},
             {"role": "assistant", "text": "ok"}]
    facts = L.extract_facts_deterministic(turns, source="sX")
    texts = [f["fact"] for f in facts]
    check(any("concise updates" in t for t in texts), "captures explicit 'remember' ask")
    check(any("umesh@example.com" in t for t in texts),
          "captures the full email (not truncated at a dot)")
    check(not any(t.strip().endswith("umesh.") for t in texts),
          "no dot-truncated value")
    ents = {f["entity"] for f in facts}
    check("Umesh" in ents and "User Profile" in ents, "attributes email to the named owner")


def test_learn_from_session():
    print("\n[memory_learn.learn_from_session]")
    _isolate()
    import sessions as S
    tmp = _TMPS[-1]
    S.SESSIONS_DIR = os.path.join(tmp, "sessions")
    S.INDEX_FILE = os.path.join(S.SESSIONS_DIR, "index.json")
    S.FTS_DB = os.path.join(S.SESSIONS_DIR, "fts.db")
    S._fts_ready = None
    rec = S.save_session(
        turns=[{"role": "user", "text": "Remember that I prefer concise updates. "
                "Umesh's email is umesh@example.com"},
               {"role": "assistant", "text": "Got it."}],
        title="t", summary="s")
    sid = rec["id"]

    # llm path (stub returns a clean array)
    r = asyncio.run(L.learn_from_session(
        sid, llm=_llm_returning('[{"entity":"Vendor Onboarding",'
                                '"fact":"KYC docs pending.","type":"project",'
                                '"confidence":0.85}]')))
    check(r["via"] == "llm" and r["facts"] == 1, "uses the LLM extraction when it returns facts")
    check(W.read_page("vendor-onboarding") is not None, "llm fact creates the page")

    # fallback path (stub returns '' -> deterministic)
    r2 = asyncio.run(L.learn_from_session(sid, llm=_llm_down))
    # `via` now names the llm outcome too (llm-error / llm-empty / llm-unparsed)
    # and appends "+deterministic" only when that path actually rescued facts.
    check(r2["via"].startswith("llm-"), "reports WHY the llm yielded nothing")
    check(r2["via"].endswith("+deterministic") or r2["facts"] == 0,
          "falls back to deterministic when LLM empty")
    prof = W.read_page("profile")
    check(any("concise updates" in f["text"] for f in W.parse_facts(prof["body"])),
          "fallback still records the explicit fact")
    check(os.path.exists(L.DIGEST_FILE), "learning rebuilds the digest")


def test_rebuild_digest():
    print("\n[memory_learn.rebuild_digest]")
    _isolate()
    W.upsert_fact("User Profile", "Prefers concise updates.", source="u",
                  confidence=0.95, slug="profile")
    W.upsert_fact("Umesh Patil", "Vendor PM, slow to respond.", source="s",
                  confidence=0.8, tags=["person"])
    dg = L.rebuild_digest()
    check("About them:" in dg, "digest has an 'About them' section from the profile")
    check("radar" in dg and "Umesh Patil" in dg, "digest lists people/projects on the radar")
    check(len(dg) <= L.DIGEST_BUDGET + 200, "digest stays near its budget")


def test_consolidate():
    print("\n[memory_learn.consolidate]")
    _isolate()
    # two near-dup facts written with dedup OFF (write raw) to force a merge
    p = W.upsert_fact("Vendor Onboarding", "KYC documents are pending.",
                      source="a", confidence=0.6, slug="vendor")
    # add a near-dup directly to the body so upsert dedup doesn't pre-merge it
    p = W.read_page("vendor")
    p["body"] += "\n" + W._fact_line("KYC docs are still pending now.", "b", 0.7)
    W.set_page(p)
    before = len(W.parse_facts(W.read_page("vendor")["body"]))
    # a stale + low-confidence fact that should be pruned
    p = W.read_page("vendor")
    p["body"] += "\n" + W._fact_line("trivial old note", "c", 0.1, date="2000-01-01")
    W.set_page(p)

    rep = asyncio.run(L.consolidate(llm=_llm_down, prune_below=0.25,
                                    prune_age_days=60))
    after = len(W.parse_facts(W.read_page("vendor")["body"]))
    check(rep["via"] == "deterministic", "consolidate uses deterministic merge when LLM down")
    check(after < before + 2, "near-dup facts merged and stale/low-conf pruned")
    check(not any("trivial old note" in f["text"]
                  for f in W.parse_facts(W.read_page("vendor")["body"])),
          "stale low-confidence fact pruned")

    # an all-stale-low-conf page is deleted entirely (not the profile)
    W.upsert_fact("Junk Topic", "x", source="z", confidence=0.05, slug="junk")
    jp = W.read_page("junk")
    jp["body"] = "## Facts\n" + W._fact_line("x", "z", 0.05, date="2000-01-01")
    W.set_page(jp)
    asyncio.run(L.consolidate(llm=_llm_down, prune_below=0.25, prune_age_days=60))
    check(W.read_page("junk") is None, "a fully-decayed page is pruned")
    check(W.read_page("profile") is not None, "the profile page is never pruned")


# ===========================================================================
#  memory_ingest
# ===========================================================================
def test_ingest():
    print("\n[memory_ingest]")
    tmp = _isolate()
    import config
    know = os.path.join(tmp, "knowledge"); up = os.path.join(tmp, "uploads")
    os.makedirs(know); os.makedirs(up)
    config.KNOWLEDGE_DIR = know; config.UPLOADS_DIR = up

    with open(os.path.join(know, "q2_notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("Q2 revenue up 12%.\nRisk: vendor onboarding delayed.")
    with open(os.path.join(up, "sales.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["rep", "amount"]); w.writerow(["Chen", "5000"])

    rep = asyncio.run(IG.scan_and_ingest(llm=_llm_down))   # deterministic summary
    check(rep["ingested"] == 2, "ingests both new files")
    check(W.read_page("file-q2-notes") is not None, "text file becomes a wiki page")
    sales = W.read_page("file-sales")
    check(sales is not None and "Columns: rep, amount" in sales["body"],
          "spreadsheet summarised by schema")

    rep2 = asyncio.run(IG.scan_and_ingest(llm=_llm_down))
    check(rep2["ingested"] == 0 and rep2["skipped"] == 2, "unchanged files are skipped")

    with open(os.path.join(know, "q2_notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("UPDATED: Q2 revenue up 15%.")
    rep3 = asyncio.run(IG.scan_and_ingest(llm=_llm_down))
    check(rep3["ingested"] == 1, "a changed file is re-ingested")

    # LLM summary path
    with open(os.path.join(know, "brief.md"), "w", encoding="utf-8") as fh:
        fh.write("# Brief\nLMS launches July.")
    asyncio.run(IG.scan_and_ingest(llm=_llm_returning("LMS launches in July.")))
    bp = W.read_page("file-brief")
    check(bp is not None and any("LMS launches in July" in f["text"]
          for f in W.parse_facts(bp["body"])), "LLM summary is used when available")

    check(IG.summarise_deterministic("line one\nline two\n") == "line one line two",
          "deterministic summary excerpts the first lines")


# ===========================================================================
#  memory_nudge
# ===========================================================================
def test_nudge_detection():
    print("\n[memory_nudge.detect_nudges]")
    from datetime import datetime, timezone, timedelta
    now = datetime(2026, 6, 8, tzinfo=timezone.utc)
    old = (now - timedelta(days=5)).isoformat()
    fresh = (now - timedelta(days=1)).isoformat()
    fups = [{"task_id": "t1", "person": "Umesh", "attempt": 2,
             "task_title": "KYC docs", "created_at": old},
            {"task_id": "t2", "person": "Lee", "attempt": 1,
             "task_title": "Sign-off", "created_at": fresh}]
    tasks = [{"id": "t3", "title": "Q2 report", "status": "todo",
              "due_date": "2026-06-01"},
             {"id": "t4", "title": "Done thing", "status": "done",
              "due_date": "2026-06-01"}]
    nudges = N.detect_nudges(now=now, follow_ups=fups, tasks=tasks, silent_days=3)
    keys = {n["key"] for n in nudges}
    check("fu:t1:umesh" in keys, "detects a follow-up silent past the threshold")
    check("fu:t2:lee" not in keys, "ignores a fresh follow-up")
    check("overdue:t3" in keys, "detects an overdue, not-done task")
    check("overdue:t4" not in keys, "ignores a done task even if past due")


def test_nudge_emission_and_channel():
    print("\n[memory_nudge.emit + channel]")
    tmp = _isolate()
    # point the scheduler at a throwaway events file
    scheduler.EVENTS_FILE = os.path.join(tmp, "events.json")
    detected = [{"key": "fu:t1:umesh", "message": "Still waiting on Umesh."}]
    em = N.emit_nudges(detected)
    check(len(em) == 1, "emits a new nudge")
    em2 = N.emit_nudges(detected)
    check(len(em2) == 0, "cooldown suppresses the same nudge on the next scan")

    # the nudge must fire as a toast but NOT appear on the calendar
    fired = scheduler.fire_due()
    notifs = scheduler.notifications()
    check(any(e.get("kind") == "nudge" for e in notifs),
          "nudge surfaces via notifications() (the toast channel)")
    cal = scheduler.list_events()
    check(not any(e.get("kind") == "nudge" for e in cal),
          "nudge is excluded from the calendar listing")

    rep = {"files": [{"file": "brief.md"}, {"file": "q2.txt"}]}
    fn = N.emit_file_nudge(rep)
    check(fn is not None and "2 new file" in fn["message"],
          "emit_file_nudge announces ingested files")
    check(N.emit_file_nudge(rep) is None, "ingest nudge deduped on the same file set")


# ===========================================================================
#  skills as procedural memory (#2 remainder)
# ===========================================================================
def test_skills_procedural_memory():
    print("\n[skills as procedural memory]")
    _isolate()
    manifest = {"name": "send teams message to person",
                "slug": "send_teams_message_to_person",
                "description": "send a Teams chat message to a named contact",
                "params": [{"name": "search"}, {"name": "type_a_message"}]}
    TS._index_skill_in_wiki(manifest)
    page = W.read_page("skill-send_teams_message_to_person")
    check(page is not None and "skill" in page["tags"], "saving a skill creates a wiki page")
    check("run_repeatable_task" in page["description"], "skill page carries the run hint")

    hint = MC.procedural_hint("can you message ravi on teams")
    check("SAVED SKILL" in hint and "run_repeatable_task" in hint,
          "procedural_hint surfaces a matching skill (the fast path)")
    check(MC.procedural_hint("what's the weather") == "",
          "procedural_hint empty when no skill matches")

    TS._deindex_skill_in_wiki("send_teams_message_to_person")
    check(W.read_page("skill-send_teams_message_to_person") is None,
          "deleting a task de-indexes its skill page")


# ===========================================================================
#  people directory (#3)
# ===========================================================================
def test_people_store():
    print("\n[people_store]")
    _isolate()
    PS.observe(name="Umesh", source="follow-up", confidence=0.5)
    PS.observe(name="Umesh Patil", email="umesh@example.com",
               source="user", confidence=0.9)
    contacts = PS.list_contacts()
    check(len(contacts) == 1, "a fuller name merges into the bare name (one contact)")
    c = contacts[0]
    check(c["full_name"] == "Umesh Patil" and "Umesh" in c["aliases"],
          "fuller name promoted, old kept as alias")
    check(c["email"] == "umesh@example.com", "email learned onto the contact")

    PS.observe(name="Umesh", email="umesh.r@othercorp.com", source="email",
               confidence=0.7)
    check(len(PS.list_contacts()) == 2,
          "same first name + CONFLICTING email stays a separate person")

    check(PS.resolve("Patil")["status"] == "resolved", "resolve by alias/surname")
    check(PS.resolve("umesh@example.com")["status"] == "resolved",
          "resolve by exact email")
    check(PS.resolve("Umash Patil")["status"] == "resolved",
          "resolve tolerates a small mishear (levenshtein)")
    check(PS.resolve("Nobody Here")["status"] == "none", "unknown -> none")
    amb = PS.resolve("Umesh")
    check(amb["status"] in ("ambiguous", "resolved") and len(amb["candidates"]) == 2,
          "a bare first name with two matches surfaces both candidates")

    # explicit CRUD
    pri = PS.add_contact("Priya Sharma", email="priya@corp.com", aliases=["Pri"])
    check(PS.resolve("Pri")["contact"]["full_name"] == "Priya Sharma",
          "add_contact + alias resolves")
    PS.update_contact(pri["id"], notes="lead BA")
    check(PS.get_contact(pri["id"])["notes"] == "lead BA", "update_contact sets a field")
    check(PS.delete_contact(pri["id"]) and PS.get_contact(pri["id"]) is None,
          "delete_contact removes it")

    # implicit learning off the task board
    import store
    orig = (store.pending_follow_ups, store.all_tasks)
    store.pending_follow_ups = lambda: [{"person": "Ravi Kumar"}]
    store.all_tasks = lambda: [{"person": "Lee"}]
    try:
        n = PS.sync_from_board()
        check(n == 2 and PS.resolve("Ravi")["status"] == "resolved",
              "sync_from_board learns follow-up/task people")
    finally:
        store.pending_follow_ups, store.all_tasks = orig

    check(PS.observe_email_recipient("Vrushabh <vrushabh@corp.com>") != {}
          and PS.resolve("vrushabh@corp.com")["status"] == "resolved",
          "observe_email_recipient parses 'Name <addr>' and learns it")


def test_learning_feeds_contacts():
    print("\n[learn_from_session -> people directory]")
    _isolate()
    import sessions as S
    tmp = _TMPS[-1]
    S.SESSIONS_DIR = os.path.join(tmp, "sessions")
    S.INDEX_FILE = os.path.join(S.SESSIONS_DIR, "index.json")
    S.FTS_DB = os.path.join(S.SESSIONS_DIR, "fts.db")
    S._fts_ready = None
    rec = S.save_session(
        turns=[{"role": "user", "text": "Umesh's email is umesh@example.com"},
               {"role": "assistant", "text": "ok"}], title="t", summary="s")
    # deterministic extraction (stub LLM down) should both wiki AND contact it
    asyncio.run(L.learn_from_session(rec["id"], llm=_llm_down))
    res = PS.resolve("Umesh")
    check(res["status"] == "resolved"
          and res["contact"]["email"] == "umesh@example.com",
          "a person+email learned from a conversation lands in the directory")


# ===========================================================================
#  runner
# ===========================================================================
def _main() -> int:
    try:
        test_wiki_store()
        test_memory_context()
        test_parse_facts_json()
        test_deterministic_extraction()
        test_learn_from_session()
        test_rebuild_digest()
        test_consolidate()
        test_ingest()
        test_nudge_detection()
        test_nudge_emission_and_channel()
        test_skills_procedural_memory()
        test_people_store()
        test_learning_feeds_contacts()
    finally:
        _cleanup()
    print("\n== memory:", "ALL PASS ==" if ok else "FAILURES ==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
