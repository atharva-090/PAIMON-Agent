"""The HOT PATH of the self-healing memory — instant, local, zero-LLM context.

This is what makes Paimon feel like it already knows you WITHOUT making you wait.
The expensive thinking (extracting facts, healing the wiki, re-rendering the
digest) happens on the cold path (memory_learn.py) in the background. Here, when
the user is actively waiting, we do nothing but cheap local lookups and string
assembly — no model call, no file scan beyond a couple of small reads.

Three things get spliced into the brain's prompt:

  1. ``session_context()`` — injected once at the start of a conversation
     (task_agent's primer / the orchestrator's first turn). It is the standing
     **digest** (pre-rendered ``memory/digest.md`` — who the user is, the people
     and projects that matter) plus **live open threads** computed fresh from the
     task board (overdue follow-ups, due reminders, blocked work). The digest is
     slow to build so it's cached; the open threads are cheap so they're always
     current.

  2. ``relevant(message)`` — injected alongside EVERY user turn. A local
     full-text lookup over the wiki for whatever the message is about, returning
     up to a few terse facts under a hard character budget. This is the "you're
     talking about Umesh → here's what I already know about Umesh" moment, and it
     costs one FTS query.

  3. (memory_learn handles the writing side.)

Everything degrades gracefully: if the digest file isn't built yet, we render a
minimal one from the wiki index on the fly; if the task board is empty, the
threads section is simply omitted.
"""

from __future__ import annotations

import os
import re

import wiki_store

DIGEST_FILE = os.path.join(os.path.dirname(__file__), "memory", "digest.md")

# Budgets — the ChatGPT-web brain is fragile and the mock chunks big prompts, so
# every injected block is hard-capped. These are the safety valve behind the
# "up to 3-4 facts" choice: richer context, but never a bloated prompt.
RELEVANT_MAX_FACTS = 4
RELEVANT_BUDGET = 800          # chars for the per-message relevant block
FACT_MAX_CHARS = 170           # truncate any single fact to keep lines tight
DIGEST_FALLBACK_BUDGET = 700   # chars for the on-the-fly digest fallback


# ===========================================================================
#  1. session-start context (digest + live open threads)
# ===========================================================================
def digest() -> str:
    """The standing 'what I know about you' block. Reads the pre-rendered
    ``memory/digest.md`` (built in the background by memory_learn.rebuild_digest);
    falls back to a minimal digest derived from the wiki index if it isn't built
    yet. Always cheap — at most one small file read."""
    try:
        if os.path.exists(DIGEST_FILE):
            with open(DIGEST_FILE, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            if text:
                return text
    except OSError:
        pass
    return _fallback_digest()


def _fallback_digest() -> str:
    """Build a tiny digest straight from the wiki index when no digest.md exists
    yet — top pages by confidence, title + one-liner. Keeps the first-ever
    conversation from being context-blind before the background job has run."""
    try:
        entries = wiki_store.index()
    except Exception:
        return ""
    if not entries:
        return ""
    ranked = sorted(entries, key=lambda e: -float(e.get("confidence", 0)))
    lines, used = [], 0
    for e in ranked:
        desc = (e.get("description") or "").strip()
        line = f"- {e['title']}" + (f" — {desc}" if desc else "")
        if used + len(line) > DIGEST_FALLBACK_BUDGET:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    return "WHAT YOU KNOW ABOUT THE USER (from memory):\n" + "\n".join(lines)


def live_threads(max_items: int = 5) -> str:
    """Open work that's live RIGHT NOW, computed fresh from the task board: due
    reminders, the most-chased open follow-ups, and blocked tasks. Deterministic
    and cheap (store.py only), so it's always current even though the digest is
    cached. Empty string when there's nothing pressing."""
    try:
        import store
    except Exception:
        return ""
    bits: list[str] = []

    try:
        due = store.due_reminders()  # remind_at <= now, not done
        for r in due[:max_items]:
            bits.append(f"reminder due: {r.get('message') or r.get('task_title')}")
    except Exception:
        pass

    try:
        fups = store.pending_follow_ups()
        # surface the most-chased first (highest attempt count = most overdue feel)
        fups = sorted(fups, key=lambda f: -int(f.get("attempt", 1)))
        for f in fups[:max_items]:
            n = int(f.get("attempt", 1))
            chase = f" (chased {n}x)" if n > 1 else ""
            bits.append(f"follow-up with {f.get('person','?')}{chase} "
                        f"re: {f.get('task_title','?')}")
    except Exception:
        pass

    try:
        tree = store.get_tree()
        blocked = []
        for g in tree.get("groups", []):
            for t in g.get("tasks", []):
                if t.get("status") == "blocked":
                    blocked.append(t["title"])
            for s in g.get("subgroups", []):
                for t in s.get("tasks", []):
                    if t.get("status") == "blocked":
                        blocked.append(t["title"])
        for title in blocked[:max_items]:
            bits.append(f"blocked: {title}")
    except Exception:
        pass

    if not bits:
        return ""
    bits = bits[:max_items]
    return "LIVE RIGHT NOW (open threads — raise these proactively if relevant):\n" \
           + "\n".join(f"- {b}" for b in bits)


def session_context() -> str:
    """The full memory block injected at the START of a conversation: standing
    digest + live open threads. Returns '' if there's genuinely nothing to say."""
    parts = [p for p in (digest(), live_threads()) if p]
    if not parts:
        return ""
    return "\n\n".join(parts)


# ===========================================================================
#  2. per-message relevant-memory injection ("the discussed thing")
# ===========================================================================
def relevant(message: str, max_facts: int = RELEVANT_MAX_FACTS,
             budget: int = RELEVANT_BUDGET, exclude: set | None = None) -> str:
    """The killer feature: given the user's CURRENT message, return up to a few
    terse facts the wiki holds about whatever they're discussing — so the brain
    answers as if it remembered. One local FTS query; no LLM.

    Returns a compact ``RELEVANT MEMORY:`` block, or '' when nothing pertinent is
    found (so chit-chat doesn't get polluted with irrelevant memory). The profile
    page is excluded by default — it's already carried by the session digest.
    """
    msg = (message or "").strip()
    if len(msg) < 3:
        return ""
    exclude = exclude if exclude is not None else {wiki_store.PROFILE_SLUG}
    try:
        hits = wiki_store.search(msg, limit=6)
    except Exception:
        return ""
    hits = [h for h in hits if h["slug"] not in exclude]
    hits = _on_topic(msg, hits)
    if not hits:
        return ""

    # Gather each page's facts, best-first (confidence then recency).
    per_page: list[tuple[dict, list[dict]]] = []
    for h in hits:
        page = wiki_store.read_page(h["slug"])
        if not page:
            continue
        facts = wiki_store.parse_facts(page["body"])
        facts.sort(key=lambda f: (-float(f.get("confidence", 0)), f.get("date", "")))
        if facts:
            per_page.append((page, facts))

    if not per_page:
        return ""

    # Round-robin across pages so we get BREADTH (one fact from each top page)
    # before depth, under the fact-count and char budgets.
    lines: list[str] = []
    used = 0
    round_i = 0
    while len(lines) < max_facts:
        progressed = False
        for page, facts in per_page:
            if round_i < len(facts):
                progressed = True
                f = facts[round_i]
                text = _trim(f["text"])
                line = f"- {page['title']}: {text}"
                if used + len(line) > budget:
                    return _wrap(lines)
                lines.append(line)
                used += len(line)
                if len(lines) >= max_facts:
                    break
        if not progressed:
            break
        round_i += 1

    return _wrap(lines)


# ===========================================================================
#  procedural memory — surface a saved SKILL relevant to the message
# ===========================================================================
def procedural_hint(message: str, max_skills: int = 2) -> str:
    """If the user is asking for something a SAVED browser skill already does,
    return a short 'run the fast way' hint. Skills live in the wiki as pages
    tagged 'skill' (task_scripts indexes them there). Instant + local; '' when
    no saved skill matches. This is the 'procedures' half of memory."""
    msg = (message or "").strip()
    if len(msg) < 3:
        return ""
    try:
        hits = wiki_store.search(msg, limit=8)
    except Exception:
        return ""
    skills = [h for h in hits if "skill" in (h.get("tags") or [])]
    if not skills:
        return ""
    lines = []
    for h in skills[:max_skills]:
        hint = (h.get("snippet") or h.get("description") or h.get("title")).strip()
        lines.append(f"- {hint}")
    return ("SAVED SKILL — you can do this the FAST way (run the saved automation) "
            "instead of driving the browser step by step:\n" + "\n".join(lines))


def _on_topic(message: str, hits: list[dict]) -> list[dict]:
    """Keep only hits whose IDENTITY matches the message, not just their body.

    FTS ORs every query token and searches the whole page blob, so a page can
    rank purely on an incidental body word. Asking "why is the specialist not
    available?" was pulling in Sales Q2, the Procurement Master Agreement and
    a Readme -- ~800 chars of spreadsheet trivia injected into a question about
    tooling, crowding the real context out of the turn.

    So require at least one distinctive query token to appear in the page's
    title, tags or entities -- the things that say what the page IS. That keeps
    "q2 sales report" -> the Q2 pages, and returns NOTHING for a meta question,
    which is the right answer: no block beats a misleading one.

    Trade-off: a body-only match is now dropped, costing some recall. Precision
    is worth more here, because an irrelevant block doesn't just waste budget --
    it actively misleads the model about what the user is discussing.
    """
    toks = {t for t in re.findall(r"\w+", (message or "").lower())
            if len(t) > 2 and t not in wiki_store._STOPWORDS}
    if not toks:
        return []
    kept = []
    for h in hits:
        identity = " ".join([
            h.get("title", ""), " ".join(h.get("tags") or []),
            " ".join(h.get("entities") or []),
        ]).lower()
        ident_toks = set(re.findall(r"\w+", identity))
        if toks & ident_toks:
            kept.append(h)
    return kept


def _trim(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= FACT_MAX_CHARS else text[:FACT_MAX_CHARS - 1] + "…"


def _wrap(lines: list[str]) -> str:
    if not lines:
        return ""
    return ("RELEVANT MEMORY (what you already know about this — use it "
            "naturally, don't recite it):\n" + "\n".join(lines))
