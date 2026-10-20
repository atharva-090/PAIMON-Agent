"""The COLD PATH of the self-healing memory — the slow thinking, off the hot path.

Nothing here ever runs while the user is waiting on a reply. It runs in the
background: after a conversation is archived, on the periodic ingest scan, and on
the nightly consolidation pass. Its job is to turn raw episodes (transcripts,
dropped files) into durable wiki facts, and to keep that wiki healthy — so that
the hot path (memory_context.py) has good, compact material to inject instantly.

Three jobs live here:

  * ``learn_from_session(session_id)`` — read an archived transcript and extract
    durable facts (people, projects, preferences, decisions), upserting them into
    the wiki with provenance (the session id) and a confidence. Then re-render
    the digest. Runs fire-and-forget after a session rotates/ends.

  * ``rebuild_digest()`` — deterministically render ``memory/digest.md`` from the
    wiki by an explicit set of RULES (the user's profile facts + the people and
    projects on their radar, highest-confidence first, under a tight budget).
    No LLM — pure assembly — so it's fast and predictable.

  * ``consolidate()`` — the self-healing pass (also see ingest in server.py):
    merge duplicate facts, let newer/higher-confidence facts win, and prune
    decayed low-confidence ones. Has an LLM merge step with a deterministic
    fallback, so the wiki still heals even when the ChatGPT-web brain is down.

The guiding principle: **the LLM is an enhancer, not a dependency.** Every path
has a deterministic fallback, so memory keeps working (just a little less richly)
when the backend is flaky — and the whole module is testable offline.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone

import wiki_store
import sessions

# A dedicated mock/browser session id so background memory work lives in its OWN
# ChatGPT window — never colliding with the live agent (hermes-pm) or the swarm.
# One window is reused across all memory jobs (extract / ingest / consolidate).
MEMORY_SESSION_ID = os.environ.get("MEMORY_SESSION_ID", "hermes-memory")

# Only one memory LLM job at a time (they share the one background brain).
_memory_lock = asyncio.Lock()

DIGEST_FILE = os.path.join(os.path.dirname(__file__), "memory", "digest.md")
DIGEST_BUDGET = 1100   # chars; this file is injected into the fragile prompt


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ===========================================================================
#  LLM helper — a throwaway, tool-less agent on the dedicated memory brain
# ===========================================================================
async def _llm_complete(system_prompt: str, user_prompt: str,
                        profile: str | None = None) -> str:
    """One plain completion on the background memory brain. Returns the text, or
    '' on any failure (so callers fall back to the deterministic path). Never
    raises."""
    try:
        import config
        from agentscope.agent import Agent, ReActConfig
        from agentscope.message import Msg, TextBlock
        from agentscope.state import AgentState
        from agentscope.permission import PermissionMode

        name = profile or config.LLM_PROFILE
        prof = config.LLM_PROFILES.get(name) or config.LLM_PROFILES["mock"]
        if prof["provider"] != "openai" and not prof.get("api_key"):
            prof, name = config.LLM_PROFILES["mock"], "mock"

        # Metered like every other brain. Memory learning runs in the
        # BACKGROUND after a session, so its spend is the easiest kind to miss
        # -- nobody is watching a progress bar when it happens.
        import usage_log

        provider = prof["provider"]
        if provider == "openai":
            from agentscope.credential import OpenAICredential
            credential = OpenAICredential(api_key=prof["api_key"],
                                          base_url=prof["base_url"])
            if name == "mock":
                from mock_model import SessionPinnedOpenAIModel
                model = usage_log.metered(SessionPinnedOpenAIModel)(
                    credential=credential, model=prof["model"], stream=False,
                    session_id=MEMORY_SESSION_ID)
            else:
                from agentscope.model import OpenAIChatModel
                model = usage_log.metered(OpenAIChatModel)(
                    credential=credential, model=prof["model"], stream=False)
        elif provider == "anthropic":
            from agentscope.credential import AnthropicCredential
            from agentscope.model import AnthropicChatModel
            model = usage_log.metered(AnthropicChatModel)(
                credential=AnthropicCredential(api_key=prof["api_key"]),
                model=prof["model"], stream=False)
        elif provider == "gemini":
            from agentscope.credential import GeminiCredential
            from agentscope.model import GeminiChatModel
            model = usage_log.metered(GeminiChatModel)(
                credential=GeminiCredential(api_key=prof["api_key"]),
                model=prof["model"], stream=False)
        else:
            return ""
        model = usage_log.attach(model, "memory_learn", name)

        state = AgentState()
        state.permission_context.mode = PermissionMode.BYPASS
        agent = Agent(name="hermes-memory", system_prompt=system_prompt,
                      model=model, state=state,
                      react_config=ReActConfig(max_iters=1))
        reply = await agent.reply(
            Msg(name="user", content=[TextBlock(type="text", text=user_prompt)],
                role="user"))
        return ((reply.get_text_content() if reply else "") or "").strip()
    except Exception as exc:  # pragma: no cover - network/backend dependent
        print(f"[memory] llm completion failed ({type(exc).__name__}: {exc})")
        return ""


# ===========================================================================
#  fact extraction from a session transcript
# ===========================================================================
_EXTRACT_SYS = (
    "You distil durable, long-term FACTS from a conversation so an assistant can "
    "remember them later. Output ONLY a JSON array, nothing else. Each item: "
    '{"entity": "<the person/project/topic the fact is about, Title Case>", '
    '"fact": "<one concise factual sentence>", '
    '"type": "person|project|preference|decision|other", '
    '"confidence": <0.0-1.0>}. '
    "Rules: capture only things worth remembering across sessions — who people "
    "are, what projects/decisions/preferences matter. Do NOT capture greetings, "
    "one-off chit-chat, or transient task mechanics. Use the SAME entity name for "
    "facts about the same thing. If there is nothing durable, output []. "
    "You may be shown a list of ACTIONS the system really performed. Those are "
    "mechanical records of real tool calls and their real results -- they are "
    "ground truth. NEVER record an outcome as achieved unless an action shows it "
    "succeeded; if an action is marked FAILED, the thing did not happen."
)


def _extract_prompt(user_text: str = "", actions: str = "",
                    transcript: str = "") -> str:
    """Assemble what the extractor reads.

    Previously this was the WHOLE transcript -- every assistant turn included.
    That is the bulk of the tokens and the least trustworthy part of the record:
    the model narrating its own progress, which has been wrong in production.
    The deterministic extractor never trusted it either ("their words are the
    durable signal; assistant turns are derived").

    So the LLM now reads the same two things a careful reader would: what the
    USER actually said, and what the system actually DID. The transcript remains
    the fallback for sessions with no recorded actions (older archives, or a
    pure chat where no tool ever ran).
    """
    parts = ["Extract the durable facts from this conversation as the JSON "
             "array described."]
    if user_text.strip():
        parts.append("WHAT THE USER SAID (their own words -- the durable "
                     "signal):\n" + user_text[:4000])
    if actions.strip():
        parts.append("WHAT THE SYSTEM ACTUALLY DID (ground truth -- real tool "
                     "calls and their real results):\n" + actions[:2500])
    if not user_text.strip() and not actions.strip():
        parts.append("Conversation:\n" + transcript[:6000])
    return "\n\n".join(parts)


def parse_facts_json(text: str) -> list[dict]:
    """Robustly pull a list of fact dicts out of an LLM reply that should be a
    JSON array but may be wrapped in prose / code fences (the weak brain does
    this constantly). Returns [] if nothing parseable."""
    if not text:
        return []
    # Prefer a fenced block, else the first [...] span.
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        start, end = text.find("["), text.rfind("]")
        blob = text[start:end + 1] if (start != -1 and end > start) else None
    if not blob:
        return []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    out = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity") or item.get("title") or "").strip()
            fact = str(item.get("fact") or item.get("text") or "").strip()
            if not entity or not fact:
                continue
            try:
                conf = float(item.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            out.append({
                "entity": entity, "fact": fact,
                "type": str(item.get("type") or "other").strip().lower(),
                "confidence": max(0.0, min(1.0, conf)),
            })
    return out


# --- deterministic fallback extraction (high-precision patterns only) -------
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# "X's email/number/phone/role/title is Y" — value runs to the end of the
# (already sentence-bounded) fragment, so it keeps a dotted email intact.
_ATTR_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\s+"
    r"(email|number|phone|role|title|manager|team)\s+(?:is|=)\s+(.+)",
    re.IGNORECASE)
# "remember (that) ..." / "note that ..." / "don't forget ..." — explicit asks
_REMEMBER_RE = re.compile(
    r"\b(?:remember(?:\s+that)?|note\s+that|don'?t\s+forget(?:\s+that)?)[:,]?\s+(.+)",
    re.IGNORECASE)
# "my <thing> is <value>" / "our <thing> is <value>"
_MINE_RE = re.compile(
    r"\b(?:my|our)\s+([a-z][a-z ]{1,28}?)\s+(?:is|are)\s+(.+)",
    re.IGNORECASE)
# A capitalised owner name immediately before an email ("Umesh: x@y", "Umesh <x@y>").
_OWNER_RE = re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[\(:<]?\s*$")


def _user_text(transcript_turns: list[dict]) -> str:
    return "\n".join(t.get("text", "") for t in transcript_turns
                     if t.get("role") == "user")


def _sentences(text: str) -> list[str]:
    """Split into rough sentences. We break on a newline or sentence-ending
    punctuation FOLLOWED BY whitespace — so 'umesh@example.com' (no space
    after its dots) is never split mid-email."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p.strip()]


def extract_facts_deterministic(turns: list[dict], source: str) -> list[dict]:
    """Conservative, no-LLM extraction: only HIGH-PRECISION patterns from the
    USER's own statements (their words are the durable signal; assistant turns
    are derived). Guarantees explicit 'remember X' asks and named contact emails
    are never lost even when the brain is offline. Precision over recall — we'd
    rather miss a fuzzy fact than write a garbled one. Returns fact dicts shaped
    like the LLM path."""
    facts: list[dict] = []
    attributed_emails: set[str] = set()

    for sent in _sentences(_user_text(turns)):
        m = _REMEMBER_RE.search(sent)
        if m:
            facts.append({"entity": "User Profile", "fact": m.group(1).strip(),
                          "type": "preference", "confidence": 0.95})
            continue
        m = _ATTR_RE.search(sent)
        if m:
            who, attr = m.group(1).strip(), m.group(2).lower()
            val = m.group(3).strip().rstrip(".!?,; ")
            facts.append({"entity": who.title(),
                          "fact": f"{who}'s {attr} is {val}.",
                          "type": "person", "confidence": 0.85})
            attributed_emails.update(_EMAIL_RE.findall(val))
            continue
        m = _MINE_RE.search(sent)
        if m:
            thing = m.group(1).strip()
            val = m.group(2).strip().rstrip(".!?,; ")
            facts.append({"entity": "User Profile",
                          "fact": f"Their {thing} is {val}.",
                          "type": "preference", "confidence": 0.8})

    # Standalone emails: only record one when a clear capitalised owner name sits
    # right before it (precision); skip bare emails (they'd make noisy pages) and
    # any already captured by the "X's email is …" pattern above.
    for sent in _sentences(_user_text(turns)):
        for email in _EMAIL_RE.findall(sent):
            if email in attributed_emails:
                continue
            before = sent[:sent.find(email)]
            nm = _OWNER_RE.search(before)
            if not nm:
                continue
            attributed_emails.add(email)
            facts.append({"entity": nm.group(1).strip().title(),
                          "fact": f"Email: {email}", "type": "person",
                          "confidence": 0.9})
    return facts


def _type_meta(ftype: str) -> tuple[list, list]:
    """(tags, ) helper — map a fact type to default page tags."""
    tag = {"person": "person", "project": "project",
           "preference": "preference", "decision": "decision"}.get(ftype, "topic")
    return [tag]


def _apply_facts(facts: list[dict], source: str) -> int:
    """Upsert a list of extracted fact dicts into the wiki. Returns count written.
    Person facts ALSO feed the people directory (a name, and an email if the fact
    states one) so contacts self-learn alongside the wiki."""
    n = 0
    for f in facts:
        entity = f.get("entity", "").strip()
        fact = f.get("fact", "").strip()
        if not entity or not fact:
            continue
        is_profile = entity.lower() in ("user profile", "user", "me", "the user")
        slug = wiki_store.PROFILE_SLUG if is_profile else wiki_store._slugify(entity)
        ftype = f.get("type", "other")
        tags = _type_meta(ftype)
        conf = float(f.get("confidence", 0.7))
        # entity aliases help the per-message lookup match casual mentions
        ents = list({entity.lower(), entity.split()[0].lower()}) if entity else []
        wiki_store.upsert_fact(
            title=entity, fact=fact, source=source,
            confidence=conf, tags=tags, entities=ents, slug=slug)
        n += 1
        # self-learning contacts: a person (not the user) with maybe an email
        if ftype == "person" and not is_profile:
            try:
                import people_store
                m = people_store._EMAIL_RE.search(fact)
                people_store.observe(name=entity, email=m.group(0) if m else "",
                                     source=source, confidence=min(conf, 0.85))
            except Exception:
                pass
    return n


async def learn_from_session(session_id: str, llm=None,
                             profile: str | None = None) -> dict:
    """Background entry point: extract durable facts from an archived session and
    fold them into the wiki, then re-render the digest. ``llm`` is an optional
    async ``(system, user) -> str`` override (used by tests); production uses the
    background memory brain with a deterministic fallback. Returns a small report.
    """
    rec = sessions.get(session_id)
    if not rec:
        return {"session": session_id, "facts": 0, "error": "no such session"}
    turns = rec.get("turns", [])

    # What the system really did, keyed to this session by action_log.rebind at
    # archive time. Empty for an older archive or a tool-less chat.
    actions = ""
    try:
        import action_log
        actions = action_log.render(session_id, with_agent=True)
    except Exception:
        actions = ""

    user_text = _user_text(turns)
    # Only pay for a full transcript when there is nothing better to send.
    transcript = ("" if (user_text.strip() or actions.strip())
                  else sessions.transcript_text(session_id))
    prompt = _extract_prompt(user_text, actions, transcript)

    async with _memory_lock:
        raw = ""
        try:
            if llm is not None:                       # test/override hook
                raw = await llm(_EXTRACT_SYS, prompt)
            else:
                raw = await _llm_complete(_EXTRACT_SYS, prompt, profile)
        except Exception as exc:
            print(f"[memory] extraction llm error: {exc}")
            raw = ""
        facts = parse_facts_json(raw)
        # Three outcomes, not two. The old code reported "deterministic"
        # whenever the LLM yielded no facts -- including when it ran perfectly
        # and correctly answered "nothing durable here". That made an empty
        # wiki impossible to diagnose: you could not tell a broken backend from
        # a quiet conversation.
        if facts:
            used = "llm"
        else:
            if not (raw or "").strip():
                used = "llm-error"          # no reply at all
            elif parse_facts_json(raw) == [] and "[" in raw:
                used = "llm-empty"          # replied, honestly found nothing
            else:
                used = "llm-unparsed"       # replied, but not as JSON
            facts = extract_facts_deterministic(turns, source=session_id)
            if facts:
                used += "+deterministic"
        written = _apply_facts(facts, source=session_id)
        try:
            rebuild_digest()
        except Exception as exc:
            print(f"[memory] digest rebuild failed: {exc}")

    # The trail has served its purpose; data/actions.jsonl keeps the durable
    # copy, so drop the in-memory one rather than holding a finished run.
    try:
        import action_log
        action_log.clear(session_id)
    except Exception:
        pass

    print(f"[memory] learned {written} fact(s) from {session_id} ({used}, "
          f"{len(prompt)} prompt chars)")
    return {"session": session_id, "facts": written, "via": used,
            "prompt_chars": len(prompt), "actions": bool(actions.strip())}


def learn_from_session_soon(session_id: str) -> None:
    """Fire-and-forget the learning job on the running loop, if there is one (the
    server). Safe no-op when called outside an event loop (CLI)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(learn_from_session(session_id))


# ===========================================================================
#  digest — the rule-based, read-only standing summary (no LLM)
# ===========================================================================
def _recency_bonus(updated_iso: str) -> float:
    """A small score nudge for freshness so recently-touched pages rank above
    equally-confident stale ones."""
    try:
        dt = datetime.fromisoformat(updated_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - dt).days
    except (ValueError, TypeError):
        return 0.0
    if days <= 1:
        return 0.30
    if days <= 7:
        return 0.20
    if days <= 30:
        return 0.10
    return 0.0


def _top_fact(page: dict) -> str:
    facts = wiki_store.parse_facts(page["body"])
    if not facts:
        return (page.get("description") or "").strip()
    facts.sort(key=lambda f: (-float(f.get("confidence", 0)), f.get("date", "")))
    return facts[0]["text"]


def repair_fact_text(dry_run: bool = False) -> dict:
    """Rewrite already-stored facts through the sanitiser.

    ``wiki_store.clean_fact_text`` guards new writes, but facts written BEFORE
    that guard existed are still sitting on disk as raw ``{"content": "..."}``
    envelopes -- including the two that `rebuild_digest` picks up, so the
    corruption is injected into every conversation until this runs.

    Idempotent: a fact already clean rewrites to itself and is not counted.
    """
    report = {"pages": 0, "facts_changed": 0, "pages_changed": [], "samples": []}
    for page in wiki_store.all_pages():
        facts = wiki_store.parse_facts(page["body"])
        if not facts:
            continue
        report["pages"] += 1
        changed = False
        for f in facts:
            cleaned = wiki_store.clean_fact_text(f["text"])
            if cleaned and cleaned != f["text"]:
                if len(report["samples"]) < 3:
                    report["samples"].append(
                        {"page": page["slug"],
                         "before": f["text"][:70], "after": cleaned[:70]})
                f["text"] = cleaned
                report["facts_changed"] += 1
                changed = True
        if changed:
            report["pages_changed"].append(page["slug"])
            if not dry_run:
                _rewrite_page_facts(page, facts)
    if not dry_run and report["facts_changed"]:
        rebuild_digest()
    return report


def ensure_profile_seeded() -> dict:
    """Give the profile page real fact BULLETS, so the digest can use it.

    The seeded page shipped its description as a plain paragraph above the
    ``## Facts`` heading. ``parse_facts`` only reads dated bullets, so the page
    parsed to ZERO facts and ``rebuild_digest`` emitted no "About them" section
    at all -- the single most important thing the agent could know about its
    user was silently absent from every conversation.

    Idempotent and non-destructive: it does nothing once the page has any fact,
    so it cannot overwrite what the agent has since learned. Safe to call on
    every startup.
    """
    out = {"seeded": 0, "facts": []}
    try:
        page = wiki_store.read_page(wiki_store.PROFILE_SLUG)
    except Exception:
        return out
    if page and wiki_store.parse_facts(page["body"]):
        return out                      # already has facts -- leave it alone

    seeds: list[tuple[str, float]] = []

    # 1. Identity, straight from config -- this is the fact the whole
    #    signing/addressing chain depends on.
    try:
        import config
        if config.USER_NAME:
            seeds.append((f"The user's name is {config.USER_NAME}.", 1.0))
        if getattr(config, "TIMEZONE", ""):
            seeds.append((f"The user is in the {config.TIMEZONE} timezone.", 0.9))
    except Exception:
        pass

    # 2. Rescue the orphaned intro prose sitting above "## Facts" -- it is real
    #    information that was never reachable as a fact.
    if page:
        intro = page["body"].split("## Facts", 1)[0].strip()
        for line in intro.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "_", "-")) and len(line) > 20:
                seeds.append((line, 0.9))

    for text, conf in seeds:
        try:
            wiki_store.upsert_fact(
                title="User Profile", fact=text, source="config",
                confidence=conf, slug=wiki_store.PROFILE_SLUG,
                tags=["profile", "user"], entities=["user", "me", "i"],
            )
            out["seeded"] += 1
            out["facts"].append(text)
        except Exception:
            continue
    return out


def rebuild_digest(budget: int = DIGEST_BUDGET, max_radar: int = 5) -> str:
    """Deterministically render ``memory/digest.md`` from the wiki by fixed RULES:

      1. 'About you'  — up to 5 highest-confidence facts from the profile page.
      2. 'On your radar' — the TOP ``max_radar`` people/projects pages, ranked by
         confidence + recency, each as its single best fact (also still capped by
         the char budget).

    The radar used to be bounded only by the char budget, so a long-lived wiki
    quietly grew the standing prompt until it crowded out the actual
    conversation. Five is what stays useful without dominating the turn.

    Returns the rendered text (also written to disk). Pure + cheap; this is what
    the hot path injects, so it's hard-capped to stay friendly to the brain.
    """
    pages = wiki_store.all_pages()
    profile = next((p for p in pages if p["slug"] == wiki_store.PROFILE_SLUG), None)
    others = [p for p in pages if p["slug"] != wiki_store.PROFILE_SLUG]
    others.sort(key=lambda p: -(float(p.get("confidence", 0))
                                + _recency_bonus(p.get("updated", ""))))

    lines = ["WHAT YOU KNOW ABOUT THE USER (durable memory — use it naturally, "
             "don't recite it):"]

    about = []
    if profile:
        pf = wiki_store.parse_facts(profile["body"])
        pf.sort(key=lambda f: (-float(f.get("confidence", 0)), f.get("date", "")))
        about = [f["text"] for f in pf[:5]]
    if about:
        lines.append("")
        lines.append("About them:")
        lines += [f"- {a}" for a in about]

    if others:
        lines.append("")
        lines.append("People & projects on their radar:")
        base = "\n".join(lines)
        used = len(base)
        shown = 0
        for p in others:
            if shown >= max_radar:
                break
            top = _top_fact(p)
            line = f"- {p['title']}" + (f": {top}" if top else "")
            if used + len(line) + 1 > budget:
                break
            lines.append(line)
            used += len(line) + 1
            shown += 1

    text = "\n".join(lines).strip()
    os.makedirs(os.path.dirname(DIGEST_FILE), exist_ok=True)
    tmp = DIGEST_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    os.replace(tmp, DIGEST_FILE)
    return text


# ===========================================================================
#  consolidation — the self-healing pass (LLM merge + deterministic fallback)
# ===========================================================================
_CONSOLIDATE_SYS = (
    "You are tidying one memory page about a single entity. You are given its "
    "current dated fact bullets. Merge duplicates and near-duplicates into one "
    "clear fact each, resolve contradictions in favour of the NEWER / higher-"
    "confidence statement, drop anything trivial, and keep the wording concise. "
    "Output ONLY a JSON array of the cleaned facts: "
    '[{"fact": "<sentence>", "confidence": <0-1>}]. Preserve every distinct piece '
    "of real information; do not invent anything new."
)


def _dedupe_facts_deterministic(facts: list[dict]) -> list[dict]:
    """No-LLM healing: collapse near-identical facts (keeping the longest text +
    max confidence + newest date) and drop very-low-confidence stragglers."""
    kept: list[dict] = []
    for f in sorted(facts, key=lambda x: (x.get("date", ""),)):
        merged = False
        for k in kept:
            if wiki_store._similar(k["text"], f["text"]) >= wiki_store._DUP_THRESHOLD:
                if len(f["text"]) > len(k["text"]):
                    k["text"] = f["text"]
                k["confidence"] = max(k["confidence"], f["confidence"])
                k["date"] = max(k["date"], f["date"])
                k["source"] = f.get("source") or k.get("source")
                merged = True
                break
        if not merged:
            kept.append(dict(f))
    return kept


async def consolidate(llm=None, profile: str | None = None,
                     prune_below: float = 0.25, prune_age_days: int = 60) -> dict:
    """The nightly self-healing pass over the whole wiki: per page, merge
    duplicate/contradictory facts (LLM when available, deterministic otherwise),
    prune facts that are both low-confidence AND stale, and delete any page left
    empty. Rebuilds the digest at the end. Returns a small report."""
    report = {"pages": 0, "facts_before": 0, "facts_after": 0,
              "pages_pruned": 0, "via": "deterministic"}
    async with _memory_lock:
        for page in wiki_store.all_pages():
            facts = wiki_store.parse_facts(page["body"])
            if not facts:
                continue
            report["pages"] += 1
            report["facts_before"] += len(facts)

            cleaned = None
            if llm is not None or _backend_profile_ok(profile):
                cleaned = await _consolidate_page_llm(page, facts, llm, profile)
                if cleaned:
                    report["via"] = "llm"
            if cleaned is None:
                cleaned = _dedupe_facts_deterministic(facts)

            # prune low-confidence + stale
            cleaned = [f for f in cleaned
                       if not (_is_stale(f, prune_age_days)
                               and f.get("confidence", 0) < prune_below)]
            report["facts_after"] += len(cleaned)

            if not cleaned and page["slug"] != wiki_store.PROFILE_SLUG:
                wiki_store.delete_page(page["slug"])
                report["pages_pruned"] += 1
                continue
            _rewrite_page_facts(page, cleaned)
        rebuild_digest()
    print(f"[memory] consolidated: {report}")
    return report


def _backend_profile_ok(profile: str | None) -> bool:
    """Whether a usable LLM profile exists (mock is always 'usable' — it'll just
    fail into the deterministic path if the backend is actually down)."""
    try:
        import config
        name = profile or config.LLM_PROFILE
        prof = config.LLM_PROFILES.get(name) or config.LLM_PROFILES["mock"]
        return prof["provider"] == "openai" or bool(prof.get("api_key"))
    except Exception:
        return False


async def _consolidate_page_llm(page, facts, llm, profile):
    bullets = "\n".join(f"- (c={f['confidence']:.2f}) {f['text']}" for f in facts)
    try:
        if llm is not None:
            raw = await llm(_CONSOLIDATE_SYS, bullets)
        else:
            raw = await _llm_complete(_CONSOLIDATE_SYS, bullets, profile)
    except Exception:
        return None
    parsed = parse_facts_json(raw) if raw else []
    if not parsed:
        # parse_facts_json wants 'entity'; consolidation JSON has only fact/conf.
        parsed = _parse_simple_facts(raw)
    if not parsed:
        return None
    out = []
    for p in parsed:
        out.append({"text": p.get("fact") or p.get("text") or "",
                    "confidence": float(p.get("confidence", 0.7)),
                    "date": _today(), "source": page["slug"]})
    return [o for o in out if o["text"].strip()]


def _parse_simple_facts(text: str) -> list[dict]:
    """Parse a `[{"fact","confidence"}]` array (consolidation output shape)."""
    if not text:
        return []
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e <= s:
        return []
    try:
        data = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for it in data if isinstance(data, list) else []:
        if isinstance(it, dict) and (it.get("fact") or it.get("text")):
            out.append(it)
    return out


def _is_stale(fact: dict, age_days: int) -> bool:
    try:
        dt = datetime.fromisoformat(fact.get("date", ""))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).days > age_days
    except (ValueError, TypeError):
        return False


def _rewrite_page_facts(page: dict, facts: list[dict]) -> None:
    """Replace a page's fact bullets with a cleaned set, preserving any non-fact
    prose above the '## Facts' section."""
    body = page["body"]
    head = body.split("## Facts", 1)[0].rstrip() if "## Facts" in body else \
        re.split(r"\n-\s*\(", body, 1)[0].rstrip()
    lines = [head, "", "## Facts"] if head else ["## Facts"]
    for f in sorted(facts, key=lambda x: (-float(x.get("confidence", 0)),
                                          x.get("date", ""))):
        lines.append(wiki_store._fact_line(
            f["text"], f.get("source", page["slug"]),
            float(f.get("confidence", 0.7)), f.get("date")))
    page["body"] = "\n".join(lines).strip()
    wiki_store.set_page(page)
