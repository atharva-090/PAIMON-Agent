"""The self-healing memory wiki — durable FACTS as human-readable Markdown.

This is the knowledge layer of Roadmap #2. It supersedes the old flat
``memory_store.py`` notes: instead of a couple of free-form notes, memory lives
as one Markdown page **per entity/topic** (a person, a project, a vendor, a
recurring subject) under ``memory/wiki/``, each carrying YAML-ish front matter
the index is built from, and a body of dated, provenance-stamped fact bullets.

Design (consistent with the rest of the codebase):
  * **PageIndex, not RAG.** A cheap ``_index.json`` table-of-contents (slug,
    title, description, tags, entities, confidence, updated) is what gets reasoned
    over first; a page body is the on-demand leaf fetch. No embeddings — the
    roadmap is explicit about reasoning over the tree, not vector search.
  * **FTS5 accelerator with a substring fallback**, exactly like ``sessions.py``:
    the Markdown files are the single source of truth; ``fts.db`` is a
    rebuildable index that degrades gracefully if this Python's sqlite lacks FTS5.
  * **Self-healing primitives baked into the data.** Every fact bullet records
    ``(date, source, confidence)`` so the nightly consolidation pass
    (memory_learn.consolidate) can merge duplicates, let newer/higher-confidence
    facts win over stale ones, and prune what decayed. Upserts are idempotent
    (a near-identical fact is refreshed, not duplicated).

Layout (under ``memory/wiki/``):
    profile.md          stable facts about the user (seeded once).
    <slug>.md           one page per entity/topic.
    _index.json         auto-generated TOC.
    fts.db              SQLite FTS5 search index (rebuildable from the .md files).

Everything here is dependency-free (no PyYAML): a tiny, forgiving front-matter
parser handles the handful of fields we use.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone

WIKI_DIR = os.path.join(os.path.dirname(__file__), "memory", "wiki")
INDEX_FILE = os.path.join(WIKI_DIR, "_index.json")
FTS_DB = os.path.join(WIKI_DIR, "fts.db")
PROFILE_SLUG = "profile"

# A fact whose normalised text already exists on the page is treated as the SAME
# fact (idempotent upsert): we refresh its date/confidence instead of appending a
# duplicate. This is the first line of self-healing, before consolidation runs.
_DUP_THRESHOLD = 0.86


# ===========================================================================
#  time / slug helpers
# ===========================================================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return slug or "note"


def _ensure_dir() -> None:
    os.makedirs(WIKI_DIR, exist_ok=True)


def _path(slug: str) -> str:
    return os.path.join(WIKI_DIR, f"{slug}.md")


# ===========================================================================
#  front-matter (dependency-free, forgiving)
# ===========================================================================
# Scalars are plain strings/numbers; lists use the inline `[a, b, c]` form.
_FM_BOUNDARY = "---"
_LIST_FIELDS = ("tags", "entities", "links")


def _parse_fm_value(raw: str):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
    return raw.strip('"').strip("'")


def _dump_fm_value(key: str, value) -> str:
    if key in _LIST_FIELDS or isinstance(value, (list, tuple, set)):
        items = ", ".join(str(v) for v in (value or []))
        return f"[{items}]"
    return str(value)


def _parse(slug: str) -> dict | None:
    """Read one page → dict with front-matter fields + ``body``, or None."""
    path = _path(slug)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    fm: dict = {}
    body = raw
    if raw.startswith(_FM_BOUNDARY):
        end = raw.find("\n" + _FM_BOUNDARY, len(_FM_BOUNDARY))
        if end != -1:
            block = raw[len(_FM_BOUNDARY):end].strip("\n")
            body = raw[end + len("\n" + _FM_BOUNDARY):].lstrip("\n")
            for line in block.splitlines():
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                fm[key.strip()] = _parse_fm_value(val)
    page = {
        "slug": slug,
        "title": fm.get("title") or slug,
        "description": fm.get("description") or "",
        "tags": fm.get("tags") if isinstance(fm.get("tags"), list) else [],
        "entities": fm.get("entities") if isinstance(fm.get("entities"), list) else [],
        "links": fm.get("links") if isinstance(fm.get("links"), list) else [],
        "confidence": _as_float(fm.get("confidence"), 0.7),
        "source": fm.get("source") or "",
        "created": fm.get("created") or "",
        "updated": fm.get("updated") or "",
        "body": body.strip(),
    }
    return page


def _as_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _write_raw(page: dict) -> None:
    """Serialise a page dict back to its .md file (front matter + body)."""
    _ensure_dir()
    order = ["title", "description", "tags", "entities", "confidence",
             "source", "links", "created", "updated"]
    lines = [_FM_BOUNDARY]
    for key in order:
        if key not in page:
            continue
        lines.append(f"{key}: {_dump_fm_value(key, page[key])}")
    lines.append(_FM_BOUNDARY)
    front = "\n".join(lines) + "\n"
    with open(_path(page["slug"]), "w", encoding="utf-8") as fh:
        fh.write(front + (page.get("body") or "").rstrip() + "\n")


# ===========================================================================
#  seed
# ===========================================================================
def _ensure() -> None:
    _ensure_dir()
    if not os.path.exists(_path(PROFILE_SLUG)):
        _write_raw({
            "slug": PROFILE_SLUG,
            "title": "User Profile",
            "description": "Who the user is and how they like to work.",
            "tags": ["profile", "user"],
            "entities": ["user", "me", "i"],
            "confidence": 1.0,
            "source": "seed",
            "links": [],
            "created": _now(),
            "updated": _now(),
            "body": (
                "The user is a Business Analyst (BA) who manages tasks, "
                "reminders and follow-ups across several work areas.\n\n"
                "## Facts\n"
                "_(durable facts about the user accrue here as dated bullets)_\n"
            ),
        })
        # First-ever boot: pull any legacy memory/notes/*.md into the wiki.
        try:
            migrate_from_notes()
        except Exception:
            pass
        rebuild_index()


# ===========================================================================
#  FTS5 search index (rebuildable accelerator; substring fallback)
# ===========================================================================
_fts_ready: bool | None = None


def _fts_blob(page: dict) -> str:
    """Everything searchable about a page flattened into one string — title,
    description, tags, entities AND body, so an entity alias (e.g. 'umesh')
    matches the page even when it only appears in front matter."""
    return " ".join([
        page.get("title", ""), page.get("description", ""),
        " ".join(page.get("tags", [])), " ".join(page.get("entities", [])),
        page.get("body", ""),
    ])


def _fts_conn() -> sqlite3.Connection | None:
    global _fts_ready
    if _fts_ready is False:
        return None
    try:
        _ensure_dir()
        conn = sqlite3.connect(FTS_DB)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5("
            "slug UNINDEXED, title, body, tokenize='porter unicode61')"
        )
        first_time = _fts_ready is None
        _fts_ready = True
        if first_time:
            _maybe_backfill(conn)
        return conn
    except sqlite3.OperationalError:
        _fts_ready = False
        return None


def _index_row(conn: sqlite3.Connection, page: dict) -> None:
    conn.execute("DELETE FROM wiki_fts WHERE slug = ?", (page["slug"],))
    conn.execute(
        "INSERT INTO wiki_fts (slug, title, body) VALUES (?,?,?)",
        (page["slug"], page.get("title", ""), _fts_blob(page)),
    )


def _maybe_backfill(conn: sqlite3.Connection) -> None:
    n = conn.execute("SELECT count(*) FROM wiki_fts").fetchone()
    if n and n[0] > 0:
        return
    did = False
    for page in all_pages():
        _index_row(conn, page)
        did = True
    if did:
        conn.commit()


def _index_one(page: dict) -> None:
    conn = _fts_conn()
    if conn is None:
        return
    try:
        _index_row(conn, page)
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()


def _drop_from_fts(slug: str) -> None:
    conn = _fts_conn()
    if conn is None:
        return
    try:
        conn.execute("DELETE FROM wiki_fts WHERE slug = ?", (slug,))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()


def _fts_match(query: str) -> str | None:
    tokens = re.findall(r"\w+", (query or "").lower())
    # Drop 1-char tokens and a few stopwords so a chatty message doesn't match
    # everything; prefix-match the rest and OR them (bm25 rank surfaces the best).
    tokens = [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]
    if not tokens:
        return None
    return " OR ".join(f"{t}*" for t in tokens[:24])


def reindex() -> int:
    """Rebuild the whole FTS index from the .md files. Returns pages indexed."""
    conn = _fts_conn()
    if conn is None:
        return 0
    try:
        conn.execute("DELETE FROM wiki_fts")
        count = 0
        for page in all_pages():
            _index_row(conn, page)
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


_STOPWORDS = {
    "the", "and", "for", "are", "was", "you", "your", "with", "this", "that",
    "have", "has", "can", "will", "what", "who", "how", "why", "where", "when",
    "did", "does", "from", "about", "they", "them", "his", "her", "its", "our",
    "but", "not", "all", "any", "some", "into", "out", "get", "got", "let",
    "please", "thanks", "hey", "hello", "hi", "okay", "yeah", "yes", "now",
}


# ===========================================================================
#  index (PageIndex TOC)
# ===========================================================================
def all_pages() -> list[dict]:
    _ensure_dir()
    out = []
    for fn in sorted(os.listdir(WIKI_DIR)):
        if fn.endswith(".md"):
            page = _parse(fn[:-3])
            if page:
                out.append(page)
    return out


def rebuild_index() -> list[dict]:
    """Regenerate the cheap table-of-contents over all pages."""
    _ensure_dir()
    entries = []
    for page in all_pages():
        entries.append({
            "slug": page["slug"],
            "title": page["title"],
            "description": page["description"],
            "tags": page["tags"],
            "entities": page["entities"],
            "confidence": page["confidence"],
            "updated": page["updated"],
            "facts": _count_facts(page["body"]),
            "chars": len(page["body"]),
        })
    with open(INDEX_FILE, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    return entries


def index() -> list[dict]:
    """The TOC the agent navigates first (cheap to load)."""
    _ensure()
    return rebuild_index()


# ===========================================================================
#  facts (dated, provenance-stamped bullets)
# ===========================================================================
# `- (2026-06-05, ses_xxx, c=0.80) Umesh is the vendor PM, slow to respond.`
_FACT_RE = re.compile(
    r"^-\s*\((?P<date>[\d-]+)(?:,\s*(?P<source>[^,)]+))?(?:,\s*c=(?P<conf>[\d.]+))?\)\s*"
    r"(?P<text>.*)$"
)


# Wrapper keys a backend might hand back instead of plain prose. The
# ChatGPT-web mock returns {"content": "..."} envelopes, and real providers
# sometimes wrap a reply in JSON despite being asked for plain text.
_ENVELOPE_KEYS = ("content", "text", "summary", "fact", "answer", "output")

# A truncated envelope: '{"content":"Master Procurement Agreement ...' with no
# closing quote or brace, because the producer sliced it mid-string.
_BROKEN_ENVELOPE = re.compile(
    r'^\{\s*"(?:' + "|".join(_ENVELOPE_KEYS) + r')"\s*:\s*"(?P<inner>.*)$',
    re.DOTALL,
)

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def clean_fact_text(text: str) -> str:
    """Unwrap a JSON/markdown envelope a backend leaked, returning plain prose.

    Every fact reaching the wiki goes through here, because the alternative --
    trusting each producer to hand over clean text -- already failed: ingested
    file summaries were stored as raw {"content": "..."} and then sliced at 400
    chars MID-STRING, so the digest injected into every single conversation was
    malformed JSON that never closed.

    Deliberately conservative. It only unwraps when the WHOLE string is an
    envelope carrying one known key, so a fact that legitimately quotes JSON
    ("the config sets {\"retries\": 3}") is left exactly as written.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # Markdown fences, possibly wrapped around JSON below.
    if t.startswith("```"):
        t = _FENCE.sub("", t).strip()

    # Well-formed envelope: parse it and take the first known key.
    if t.startswith("{") and t.endswith("}"):
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                for k in _ENVELOPE_KEYS:
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return _collapse(v)
        except (ValueError, TypeError):
            pass

    # Truncated envelope: no closing quote/brace to parse, so recover the inner
    # string by hand and undo the JSON escaping that survived the slice.
    m = _BROKEN_ENVELOPE.match(t)
    if m:
        inner = m.group("inner")
        # Drop a trailing '"}' / '"' if the slice happened to keep it.
        inner = re.sub(r'"\s*\}?\s*$', "", inner)
        inner = (inner.replace('\\"', '"').replace("\\n", " ")
                      .replace("\\t", " ").replace("\\\\", "\\"))
        return _collapse(inner)

    return _collapse(t)


def _collapse(text: str) -> str:
    """One-line prose: newlines folded, runs of space squeezed."""
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()


def truncate_words(text: str, limit: int) -> str:
    """Cut to `limit` chars WITHOUT splitting a word (or a JSON string).

    The 400-char slice in memory_ingest is what turned an envelope into
    unterminated garbage; cutting on a boundary keeps a truncation readable.
    """
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    cut = t[:limit]
    sp = cut.rfind(" ")
    if sp > limit * 0.6:          # only back off if it doesn't gut the text
        cut = cut[:sp]
    return cut.rstrip(" ,;:-") + "\u2026"


def _fact_line(text: str, source: str, confidence: float, date: str | None = None) -> str:
    date = date or _today()
    src = (source or "").strip() or "?"
    return f"- ({date}, {src}, c={confidence:.2f}) {text.strip()}"


def parse_facts(body: str) -> list[dict]:
    """Extract the dated fact bullets from a page body."""
    facts = []
    for line in (body or "").splitlines():
        m = _FACT_RE.match(line.strip())
        if m:
            facts.append({
                "date": m.group("date"),
                "source": (m.group("source") or "").strip(),
                "confidence": _as_float(m.group("conf"), 0.7),
                "text": m.group("text").strip(),
                "raw": line.strip(),
            })
    return facts


def _count_facts(body: str) -> int:
    return len(parse_facts(body))


def _normalise(text: str) -> str:
    """Loose normalisation for dedup: lowercase, drop punctuation/extra space."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())).strip()


def _content_tokens(text: str) -> set:
    """Meaningful words only — stopwords/articles dropped — so two restatements
    of the same fact that differ only by connectives compare as equal."""
    return {t for t in _normalise(text).split()
            if len(t) > 1 and t not in _STOPWORDS}


def _similar(a: str, b: str) -> float:
    """How alike two facts are (0..1). Blends Jaccard with a containment signal
    over CONTENT tokens, so 'Umesh is the vendor PM and has been slow to respond'
    and 'Umesh is the vendor PM, slow to respond' count as the same fact, while a
    genuinely different statement stays distinct."""
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    jac = inter / len(ta | tb)
    # Containment catches an elaboration/rewording (one fact's words ⊆ the
    # other's); only trust it when both sides carry real signal.
    if min(len(ta), len(tb)) >= 3:
        return max(jac, (inter / min(len(ta), len(tb))) * 0.95)
    return jac


# ===========================================================================
#  PUBLIC API — read
# ===========================================================================
def read_page(slug: str) -> dict | None:
    """Load one page's full body — the on-demand leaf fetch."""
    _ensure()
    return _parse(slug)


def exists(slug: str) -> bool:
    return os.path.exists(_path(slug))


def search(query: str, limit: int = 8) -> list[dict]:
    """Full-text search across pages (most relevant first). FTS5 when available,
    substring fallback otherwise. Returns index-style entries + a snippet."""
    q = (query or "").strip()
    if not q:
        return []
    hits = _search_fts(q, limit)
    if hits:
        return hits
    return _search_substring(q, limit)


def _entry_for(slug: str) -> dict | None:
    page = _parse(slug)
    if not page:
        return None
    body = page["body"]
    snippet = page["description"] or (body[:120] if body else "")
    return {
        "slug": page["slug"], "title": page["title"],
        "description": page["description"], "tags": page["tags"],
        "entities": page["entities"], "confidence": page["confidence"],
        "updated": page["updated"], "snippet": snippet.strip(),
    }


def _search_fts(query: str, limit: int) -> list[dict]:
    conn = _fts_conn()
    if conn is None:
        return []
    try:
        match = _fts_match(query)
        if not match:
            return []
        rows = conn.execute(
            "SELECT slug FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    out = []
    for r in rows:
        e = _entry_for(r[0])
        if e:
            out.append(e)
    return out


def _search_substring(query: str, limit: int) -> list[dict]:
    toks = [t for t in re.findall(r"\w+", query.lower())
            if len(t) > 1 and t not in _STOPWORDS]
    scored = []
    for page in all_pages():
        hay = _fts_blob(page).lower()
        score = sum(hay.count(t) for t in toks) if toks else 0
        if score:
            e = _entry_for(page["slug"])
            if e:
                scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:limit]]


# ===========================================================================
#  PUBLIC API — write (idempotent upsert + self-healing primitives)
# ===========================================================================
def write_page(
    title: str,
    body: str = "",
    description: str = "",
    tags: list | None = None,
    entities: list | None = None,
    confidence: float = 0.7,
    source: str = "",
    links: list | None = None,
    slug: str | None = None,
) -> dict:
    """Create or overwrite a whole page."""
    _ensure()
    slug = slug or _slugify(title)
    existing = _parse(slug)
    created = existing["created"] if existing else _now()
    page = {
        "slug": slug,
        "title": title or (existing["title"] if existing else slug),
        "description": description or (existing["description"] if existing else ""),
        "tags": list(tags if tags is not None else (existing["tags"] if existing else [])),
        "entities": list(entities if entities is not None
                          else (existing["entities"] if existing else [])),
        "confidence": float(confidence),
        "source": source or (existing["source"] if existing else ""),
        "links": list(links if links is not None else (existing["links"] if existing else [])),
        "created": created,
        "updated": _now(),
        "body": body if body else (existing["body"] if existing else ""),
    }
    _write_raw(page)
    _index_one(page)
    rebuild_index()
    return _parse(slug)


def upsert_fact(
    title: str,
    fact: str,
    source: str = "",
    confidence: float = 0.7,
    tags: list | None = None,
    entities: list | None = None,
    description: str = "",
    slug: str | None = None,
    links: list | None = None,
) -> dict:
    """Add a dated, provenance-stamped fact to an entity/topic page (creating the
    page if needed). IDEMPOTENT: if a near-identical fact already exists, refresh
    its date/source and keep the HIGHER confidence rather than duplicating it.
    New evidence with higher confidence wins — the first layer of self-healing.

    Returns the updated page dict.
    """
    _ensure()
    slug = slug or _slugify(title)
    page = _parse(slug)
    if page is None:
        page = {
            "slug": slug, "title": title or slug, "description": description,
            "tags": list(tags or []), "entities": list(entities or []),
            "confidence": float(confidence), "source": source, "links": list(links or []),
            "created": _now(), "updated": _now(),
            "body": f"## Facts\n",
        }
    else:
        # Merge metadata (don't clobber existing tags/entities; union them).
        if description and not page["description"]:
            page["description"] = description
        page["tags"] = _union(page["tags"], tags)
        page["entities"] = _union(page["entities"], entities)
        page["links"] = _union(page["links"], links)
        page["confidence"] = max(page["confidence"], float(confidence))

    facts = parse_facts(page["body"])
    # The chokepoint: every writer (ingest, session learning, agent tools,
    # consolidation) funnels through here, so one guard covers all of them.
    fact = clean_fact_text(fact)
    if not fact:
        return _parse(slug) or page
    dup_idx = None
    for i, f in enumerate(facts):
        if _similar(f["text"], fact) >= _DUP_THRESHOLD:
            dup_idx = i
            break

    today_line = _fact_line(fact, source, confidence)
    if dup_idx is not None:
        # Refresh the existing fact: newest date, source, max confidence.
        prev = facts[dup_idx]
        merged_conf = max(prev["confidence"], confidence)
        new_text = fact if len(fact) >= len(prev["text"]) else prev["text"]
        new_line = _fact_line(new_text, source or prev["source"], merged_conf)
        page["body"] = _replace_fact_line(page["body"], prev["raw"], new_line)
    else:
        page["body"] = _append_fact_line(page["body"], today_line)

    page["updated"] = _now()
    if source:
        page["source"] = source
    _write_raw(page)
    _index_one(page)
    rebuild_index()
    return _parse(slug)


def _union(existing: list, extra: list | None) -> list:
    out = list(existing or [])
    for x in (extra or []):
        if x not in out:
            out.append(x)
    return out


def _append_fact_line(body: str, line: str) -> str:
    body = (body or "").rstrip()
    if "## Facts" not in body:
        body = (body + "\n\n## Facts").strip()
    return body + "\n" + line


def _replace_fact_line(body: str, old_raw: str, new_line: str) -> str:
    out = []
    for ln in (body or "").splitlines():
        out.append(new_line if ln.strip() == old_raw.strip() else ln)
    return "\n".join(out)


def set_page(page: dict) -> dict:
    """Write a fully-formed page dict back (used by the consolidation pass after
    it has merged/rewritten a body). Stamps ``updated`` and reindexes."""
    _ensure()
    page = dict(page)
    page["updated"] = _now()
    if not page.get("created"):
        page["created"] = _now()
    _write_raw(page)
    _index_one(page)
    rebuild_index()
    return _parse(page["slug"])


def delete_page(slug: str) -> bool:
    """Prune a page entirely (used by consolidation to drop dead entries)."""
    if slug == PROFILE_SLUG:
        raise ValueError("the profile page cannot be deleted")
    path = _path(slug)
    if os.path.exists(path):
        os.remove(path)
        _drop_from_fts(slug)
        rebuild_index()
        return True
    return False


def remember(text: str, topic: str = PROFILE_SLUG, source: str = "user",
             confidence: float = 0.9) -> dict:
    """Convenience: durably record a fact, defaulting to the user profile page.
    This is what the `remember_fact` agent tool calls."""
    title = topic.replace("-", " ").title() if topic != PROFILE_SLUG else "User Profile"
    return upsert_fact(title, text, source=source, confidence=confidence,
                       slug=_slugify(topic))


# ===========================================================================
#  migration from the old flat memory_store notes
# ===========================================================================
def migrate_from_notes() -> int:
    """One-time import of any legacy ``memory/notes/*.md`` into the wiki. Safe to
    call repeatedly — it only imports notes whose slug isn't already a wiki page.
    Returns the number of notes imported."""
    notes_dir = os.path.join(os.path.dirname(__file__), "memory", "notes")
    if not os.path.isdir(notes_dir):
        return 0
    _ensure()
    imported = 0
    for fn in sorted(os.listdir(notes_dir)):
        if not fn.endswith(".md"):
            continue
        slug = fn[:-3]
        target = PROFILE_SLUG if slug == "profile" else slug
        # Don't clobber the seeded profile or an existing page; merge body in.
        try:
            with open(os.path.join(notes_dir, fn), "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError:
            continue
        # Strip the legacy front matter (title/description/updated) if present.
        body = raw
        m = re.match(r"^---\s*\n.*?\n---\s*\n(?P<b>.*)$", raw, re.DOTALL)
        if m:
            body = m.group("b")
        page = _parse(target)
        if page is None:
            write_page(title=slug.replace("-", " ").title(), body=body.strip(),
                       description="Imported from legacy notes.",
                       source="migrated", slug=target)
            imported += 1
        elif body.strip() and body.strip() not in page["body"]:
            page["body"] = (page["body"].rstrip() + "\n\n## Imported\n"
                            + body.strip())
            set_page(page)
            imported += 1
    return imported
