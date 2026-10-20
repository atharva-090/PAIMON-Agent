"""Step 1: the digest corruption fixes.

Two bugs made the standing memory block — injected into EVERY conversation —
actively harmful:

  A. `profile.md` shipped its description as a plain paragraph above the
     `## Facts` heading. `parse_facts` only reads dated bullets, so the page
     parsed to ZERO facts and the digest had no "About them" section at all.
     The agent never knew the user's name.

  B. `memory_ingest._summarise` stored the backend's reply verbatim and then
     did `raw[:400]`. The mock returns a `{"content": "..."}` envelope, so the
     digest carried JSON — sliced mid-string, so it never even closed.

Run:  .venv/Scripts/python.exe test_digest_repair.py
"""
from __future__ import annotations

import os
import shutil
import tempfile

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


# ------------------------------------------------------------ sanitiser
print("\nclean_fact_text — unwraps envelopes, leaves prose alone")

check("well-formed envelope",
      wiki_store.clean_fact_text('{"content":"Net 30 terms."}'),
      "Net 30 terms.")

# The actual bug: sliced at 400 chars, so there is no closing quote or brace.
check("TRUNCATED envelope (the real bug)",
      wiki_store.clean_fact_text('{"content":"Master Agreement, 24 months'),
      "Master Agreement, 24 months")

check("fenced json",
      wiki_store.clean_fact_text('```json\n{"summary":"Revenue up 12%."}\n```'),
      "Revenue up 12%.")

check("escaped quotes survive",
      wiki_store.clean_fact_text('{"content":"He said \\"yes\\" firmly'),
      'He said "yes" firmly')

# Must NOT unwrap — these are the false positives that would corrupt good data.
check("plain prose untouched",
      wiki_store.clean_fact_text("Umesh Patil is the vendor PM."),
      "Umesh Patil is the vendor PM.")
check("prose CONTAINING json untouched",
      wiki_store.clean_fact_text('The config sets {"retries": 3} here.'),
      'The config sets {"retries": 3} here.')
check("json without a known key untouched",
      wiki_store.clean_fact_text('{"unrelated": "value"}'),
      '{"unrelated": "value"}')
check("empty stays empty", wiki_store.clean_fact_text("   "), "")

print("\ntruncate_words — cuts on a boundary, never mid-word")
check("short text passes through",
      wiki_store.truncate_words("short text", 40), "short text")
out = wiki_store.truncate_words("The quick brown fox jumps over the lazy dog", 20)
truthy("long text is shortened", len(out) <= 21)
truthy("no half-word at the cut", not out.rstrip("…").endswith(("jum", "ju")))

# ------------------------------------------------- write-path integration
print("\nwrite path — the sanitiser guards upsert_fact itself")

tmp = tempfile.mkdtemp(prefix="wiki-test-")
real = (wiki_store.WIKI_DIR, wiki_store.INDEX_FILE, wiki_store.FTS_DB)
wiki_store.WIKI_DIR = tmp
wiki_store.INDEX_FILE = os.path.join(tmp, "_index.json")
wiki_store.FTS_DB = os.path.join(tmp, "fts.db")
wiki_store._fts_ready = None

import memory_learn                                        # noqa: E402
real_digest = memory_learn.DIGEST_FILE
memory_learn.DIGEST_FILE = os.path.join(tmp, "digest.md")

try:
    # A producer that leaks an envelope must still land clean on disk.
    wiki_store.upsert_fact(title="Acme Corp",
                           fact='{"content":"Acme pays Net 30.',
                           source="file:test.pdf", confidence=0.7)
    facts = wiki_store.parse_facts(wiki_store.read_page("acme-corp")["body"])
    check("envelope cleaned at write time", facts[0]["text"], "Acme pays Net 30.")

    # Idempotence: re-upserting the CLEAN form must refresh, not duplicate.
    wiki_store.upsert_fact(title="Acme Corp", fact="Acme pays Net 30.",
                           source="file:test.pdf", confidence=0.8)
    facts = wiki_store.parse_facts(wiki_store.read_page("acme-corp")["body"])
    check("cleaned text dedupes against its envelope", len(facts), 1)

    # --- profile seeding ------------------------------------------------
    print("\nensure_profile_seeded")
    wiki_store.write_page(title="User Profile",
                          body="The user is a Business Analyst (BA).\n\n## Facts\n",
                          description="Who the user is.",
                          tags=["profile", "user"], slug=wiki_store.PROFILE_SLUG)
    before = wiki_store.parse_facts(
        wiki_store.read_page(wiki_store.PROFILE_SLUG)["body"])
    check("reproduces the bug: prose above ## Facts parses to 0", len(before), 0)

    r1 = memory_learn.ensure_profile_seeded()
    truthy("seeding added facts", r1["seeded"] > 0)
    after = wiki_store.parse_facts(
        wiki_store.read_page(wiki_store.PROFILE_SLUG)["body"])
    truthy("profile now yields facts", len(after) > 0)
    truthy("orphaned intro prose was rescued",
           any("Business Analyst" in f["text"] for f in after))

    r2 = memory_learn.ensure_profile_seeded()
    check("idempotent — second call is a no-op", r2["seeded"], 0)

    # --- repair pass ------------------------------------------------------
    print("\nrepair_fact_text")
    # Write a polluted fact the way the OLD code would have, bypassing the guard.
    page = wiki_store.read_page("acme-corp")
    page["body"] += '\n- (2026-06-08, file:old.pdf, c=0.70) {"content":"Legacy fact'
    wiki_store.set_page(page)

    dry = memory_learn.repair_fact_text(dry_run=True)
    check("dry run finds it", dry["facts_changed"], 1)
    unchanged = wiki_store.parse_facts(wiki_store.read_page("acme-corp")["body"])
    truthy("dry run changed nothing on disk",
           any(f["text"].startswith('{"content"') for f in unchanged))

    live = memory_learn.repair_fact_text()
    check("live run fixes it", live["facts_changed"], 1)
    fixed = wiki_store.parse_facts(wiki_store.read_page("acme-corp")["body"])
    truthy("envelope is gone from disk",
           not any(f["text"].startswith('{"content"') for f in fixed))
    truthy("text preserved", any("Legacy fact" in f["text"] for f in fixed))

    check("idempotent — nothing left to repair",
          memory_learn.repair_fact_text()["facts_changed"], 0)

    # --- the payoff -------------------------------------------------------
    print("\ndigest")
    digest = memory_learn.rebuild_digest()
    truthy("has an 'About them' section", "About them" in digest)
    truthy("carries no JSON envelope", '{"content"' not in digest)
finally:
    wiki_store.WIKI_DIR, wiki_store.INDEX_FILE, wiki_store.FTS_DB = real
    wiki_store._fts_ready = None
    memory_learn.DIGEST_FILE = real_digest
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    raise SystemExit(1)
print("all digest-repair checks passed")
