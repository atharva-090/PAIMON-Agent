"""Conversation session store — stops the whole transcript being re-injected.

Problem this solves: the agent's brain (ChatGPT via the openai-mocker) gets the
ENTIRE running conversation re-sent on every turn. That bloats tokens and, worse,
makes ChatGPT break character / refuse. So instead of one ever-growing context we
keep discrete **sessions**:

  * Each finished session is stored as a compact JSON record (id, title, one-line
    summary, and the plain user/assistant turns -- no tool-call noise).
  * A cheap `index.json` lists recent sessions (title + summary only).
  * A NEW session starts by seeing only that index (titles + one-liners of the
    last ~5). The agent decides if the current task relates to a past session and,
    if so, calls a tool to load that session's full context on demand
    (PageIndex-style: navigate the index, fetch the leaf only when needed).

Search is backed by an SQLite **FTS5** full-text index (`fts.db`) kept alongside
the JSON. The JSON files remain the single source of truth; the FTS index is a
disposable accelerator that is auto-(re)built from the JSON if missing, and that
degrades gracefully to a plain substring scan if this Python's sqlite has no
FTS5 compiled in.

Layout (under memory/sessions/):
  index.json            [{id, title, summary, ended_at, turns}]   (newest last)
  <id>.json             {id, title, summary, started_at, ended_at, turns:[{role,text}]}
  fts.db                SQLite FTS5 search index (rebuildable from the JSON)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "memory", "sessions")
INDEX_FILE = os.path.join(SESSIONS_DIR, "index.json")
FTS_DB = os.path.join(SESSIONS_DIR, "fts.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure() -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _read(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            txt = fh.read().strip()
        return json.loads(txt) if txt else default
    except (OSError, json.JSONDecodeError):
        return default


def _write(path: str, obj) -> None:
    _ensure()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _new_id() -> str:
    return ("ses_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_" + uuid.uuid4().hex[:4])


# --- SQLite FTS5 search index ----------------------------------------------
# The JSON files are the source of truth; this is a rebuildable accelerator.
# `_fts_ready`: None = not checked yet, True = FTS5 works, False = unavailable
# (so we stop trying and fall back to substring search).
_fts_ready: bool | None = None


def _record_body(rec: dict) -> str:
    """Flatten a session's turns into one searchable text blob."""
    return " ".join(t.get("text", "") for t in rec.get("turns", []))


def _fts_conn() -> sqlite3.Connection | None:
    """Return an sqlite connection with the FTS5 table ready, or None if FTS5
    isn't available in this Python's sqlite build. On first successful open,
    backfills the index from any existing JSON sessions."""
    global _fts_ready
    if _fts_ready is False:
        return None
    try:
        _ensure()
        conn = sqlite3.connect(FTS_DB)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5("
            "id UNINDEXED, title, summary, body, "
            "tokenize='porter unicode61')"
        )
        first_time = _fts_ready is None
        _fts_ready = True
        if first_time:
            _maybe_backfill(conn)
        return conn
    except sqlite3.OperationalError:
        # No FTS5 in this sqlite build — give up on the index for good.
        _fts_ready = False
        return None


def _index_record(conn: sqlite3.Connection, rec: dict) -> None:
    conn.execute("DELETE FROM sessions_fts WHERE id = ?", (rec["id"],))
    conn.execute(
        "INSERT INTO sessions_fts (id, title, summary, body) VALUES (?,?,?,?)",
        (rec["id"], rec.get("title", ""), rec.get("summary", ""),
         _record_body(rec)),
    )


def _maybe_backfill(conn: sqlite3.Connection) -> None:
    """If the index is empty but JSON sessions exist, index them all."""
    n = conn.execute("SELECT count(*) FROM sessions_fts").fetchone()
    if n and n[0] > 0:
        return
    indexed = False
    for entry in load_index():
        rec = get(entry["id"])
        if rec:
            _index_record(conn, rec)
            indexed = True
    if indexed:
        conn.commit()


def _index_session(rec: dict) -> None:
    """Add/refresh one session in the FTS index (no-op if FTS5 unavailable)."""
    conn = _fts_conn()
    if conn is None:
        return
    try:
        _index_record(conn, rec)
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()


def _fts_match(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression from a free-text query: keep only
    word characters, prefix-match each token, OR them together so any term can
    hit (bm25 `rank` then surfaces the most relevant sessions first)."""
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return None
    return " OR ".join(f"{t}*" for t in tokens)


def reindex() -> int:
    """Rebuild the whole FTS index from the JSON sessions. Returns the number
    of sessions indexed (0 if FTS5 is unavailable)."""
    conn = _fts_conn()
    if conn is None:
        return 0
    try:
        conn.execute("DELETE FROM sessions_fts")
        count = 0
        for entry in load_index():
            rec = get(entry["id"])
            if rec:
                _index_record(conn, rec)
                count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def load_index() -> list[dict]:
    return _read(INDEX_FILE, [])


def save_session(
    turns: list[dict], title: str, summary: str,
    started_at: str | None = None,
) -> dict:
    """Persist one finished session and append it to the index.

    Args:
        turns: [{"role": "user"|"assistant", "text": "..."}] -- plain text only.
        title: short label for the session.
        summary: one-line description.
    """
    _ensure()
    sid = _new_id()
    # Give every turn a stable message id (mid = "<sid>.<index>") so callers can
    # later fetch individual messages with get_messages([...]) without loading
    # a whole session.
    turns = [
        {**t, "mid": t.get("mid") or f"{sid}.{i}"}
        for i, t in enumerate(turns)
    ]
    record = {
        "id": sid,
        "title": title.strip()[:120] or "Untitled session",
        "summary": summary.strip()[:400],
        "started_at": started_at or _now(),
        "ended_at": _now(),
        "turns": turns,
    }
    _write(os.path.join(SESSIONS_DIR, f"{sid}.json"), record)
    index = load_index()
    index.append({
        "id": sid,
        "title": record["title"],
        "summary": record["summary"],
        "ended_at": record["ended_at"],
        "turns": len(turns),
    })
    _write(INDEX_FILE, index)
    _index_session(record)  # keep the FTS index in sync (best-effort)
    return record


def update_summary(session_id: str, summary: str) -> bool:
    """Replace a stored session's one-line summary (and its index entry + FTS
    row). Used by the background LLM summariser, which runs AFTER a session is
    archived with a cheap placeholder summary and then upgrades it."""
    rec = get(session_id)
    if rec is None:
        return False
    rec["summary"] = (summary or "").strip()[:400]
    _write(os.path.join(SESSIONS_DIR, f"{session_id}.json"), rec)
    index = load_index()
    for entry in index:
        if entry["id"] == session_id:
            entry["summary"] = rec["summary"]
            break
    _write(INDEX_FILE, index)
    _index_session(rec)
    return True


def recent(n: int = 5) -> list[dict]:
    """The cheap TOC a new session sees: the last `n` sessions, newest first."""
    return list(reversed(load_index()))[:n]


def get(session_id: str) -> dict | None:
    return _read(os.path.join(SESSIONS_DIR, f"{session_id}.json"), None)


def transcript_text(session_id: str, max_chars: int = 6000) -> str:
    """A compact plain-text rendering of a stored session for re-loading into
    the agent's context on demand."""
    rec = get(session_id)
    if rec is None:
        return ""
    lines = [f"# Past session: {rec['title']}", f"({rec['summary']})", ""]
    for t in rec.get("turns", []):
        who = "User" if t.get("role") == "user" else "You"
        mid = t.get("mid", "")
        tag = f" [{mid}]" if mid else ""
        lines.append(f"{who}{tag}: {t.get('text', '')}")
    text = "\n".join(lines)
    return text[:max_chars] + ("\n…[truncated]" if len(text) > max_chars else "")


def get_messages(message_ids: list[str]) -> list[dict]:
    """Fetch individual messages by their ids (mid = "<session_id>.<index>").
    Returns [{mid, session_id, title, role, text}] in the requested order,
    skipping any id that can't be resolved. Lets the agent pull a couple of
    specific messages instead of a whole session."""
    out: list[dict] = []
    cache: dict[str, dict] = {}
    for mid in message_ids or []:
        if not isinstance(mid, str) or "." not in mid:
            continue
        sid = mid.rsplit(".", 1)[0]
        rec = cache.get(sid)
        if rec is None:
            rec = get(sid) or {}
            cache[sid] = rec
        for t in rec.get("turns", []):
            if t.get("mid") == mid:
                out.append({
                    "mid": mid,
                    "session_id": sid,
                    "title": rec.get("title", ""),
                    "role": t.get("role", ""),
                    "text": t.get("text", ""),
                })
                break
    return out


def search(query: str, limit: int = 10) -> list[dict]:
    """Full-text search across stored sessions, returning matching index
    entries (most relevant first). Uses the SQLite FTS5 index when available
    and falls back to a plain substring scan otherwise."""
    q = (query or "").strip()
    if not q:
        return []
    hits = _search_fts(q, limit)
    if hits:
        return hits
    return _search_substring(q, limit)


def _search_fts(query: str, limit: int) -> list[dict]:
    """FTS5-backed search; [] if FTS5 is unavailable, errors, or has no hits."""
    conn = _fts_conn()
    if conn is None:
        return []
    try:
        match = _fts_match(query)
        if not match:
            return []
        rows = conn.execute(
            "SELECT id FROM sessions_fts WHERE sessions_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    by_id = {e["id"]: e for e in load_index()}
    return [by_id[r[0]] for r in rows if r[0] in by_id]


def _search_substring(query: str, limit: int = 10) -> list[dict]:
    """Plain substring search (the fallback when FTS5 isn't available)."""
    q = query.lower()
    hits = []
    for entry in reversed(load_index()):
        rec = get(entry["id"])
        if not rec:
            continue
        hay = (rec["title"] + " " + rec["summary"] + " "
               + _record_body(rec)).lower()
        if q in hay:
            hits.append(entry)
            if len(hits) >= limit:
                break
    return hits


def index_brief(n: int = 5) -> str:
    """Render the recent-sessions index as a short block for the agent's prompt."""
    rs = recent(n)
    if not rs:
        return "(no earlier sessions yet)"
    out = []
    for s in rs:
        out.append(f"- [{s['id']}] {s['title']} — {s['summary']}")
    return "\n".join(out)


def _within_minutes(iso: str | None, minutes: int) -> bool:
    """True if `iso` is a timestamp between now and `minutes` ago."""
    if not iso:
        return False
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (datetime.now(timezone.utc) - dt).total_seconds()
    return 0 <= delta <= minutes * 60


def _continuation_block(recent_minutes: int, max_chars: int = 700) -> str:
    """If the most recent session ended within `recent_minutes`, render its last
    user message + last assistant reply verbatim so a fresh session can pick the
    thread straight back up. Empty string if the last session is older (a cold
    return — the 5 summaries alone are enough)."""
    idx = load_index()
    if not idx:
        return ""
    last = idx[-1]  # newest (index is newest-last)
    if not _within_minutes(last.get("ended_at"), recent_minutes):
        return ""
    rec = get(last["id"])
    if not rec:
        return ""
    turns = rec.get("turns", [])
    user_t = next((t for t in reversed(turns) if t.get("role") == "user"), None)
    asst_t = next((t for t in reversed(turns)
                   if t.get("role") == "assistant"), None)
    if not (user_t or asst_t):
        return ""
    out = [
        f"CONTINUATION — your most recent exchange ([{last['id']}], just now); "
        "you are continuing this same conversation:",
    ]
    if user_t:
        out.append(f"  User said: {user_t.get('text', '').strip()[:max_chars]}")
    if asst_t:
        out.append(f"  You replied: {asst_t.get('text', '').strip()[:max_chars]}")
    return "\n".join(out)


def primer(n: int = 5, recent_minutes: int = 30) -> str:
    """The context block a fresh session sees: the last `n` session summaries
    (always), plus the previous exchange verbatim IF it happened within
    `recent_minutes` (so an active back-and-forth keeps continuity, while a cold
    return after a gap only sees the cheap summaries)."""
    lines = [
        "RECENT SESSIONS — your last conversations (titles + one-line summaries "
        "only, NOT full content):",
        index_brief(n),
    ]
    cont = _continuation_block(recent_minutes)
    if cont:
        lines += ["", cont]
    lines += [
        "",
        "If the current request continues one of these, call "
        "load_past_session('<id>') ONCE for a full transcript, or "
        "get_messages(['<mid>', ...]) for just specific messages. If it's a new "
        "or unrelated topic, proceed and load nothing (keep context small).",
    ]
    return "\n".join(lines)
