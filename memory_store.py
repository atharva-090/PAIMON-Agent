"""Persistent, cross-session memory for the agent, stored as Markdown files.

PageIndex approach (https://github.com/VectifyAI/PageIndex):
    Rather than stuffing every remembered fact into the model's context (which
    doesn't scale and gets expensive), memory lives in human-readable ``.md``
    files on disk and is exposed to the agent through a small *index* -- a
    table-of-contents of every memory note with a one-line description. The
    agent reads the cheap index first, reasons about which note is relevant,
    and only then loads that note's full body. This is the same "build a tree,
    navigate by reasoning, fetch the leaf on demand" idea PageIndex applies to
    long documents -- here applied to long-term memory.

Layout (under ``memory/notes/``):
    profile.md          stable facts about the user (a BA): role, people,
                        preferences, recurring workflows.
    <slug>.md           one note per topic/project, each with YAML-ish front
                        matter (title + description) used to build the index.
    _index.json         auto-generated TOC: [{slug, title, description, ...}]

The agent's *conversational* short-term memory (the running dialogue) is
handled separately by AgentScope's AgentState in task_agent.py. This module is
the durable, queryable knowledge base that survives across sessions.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone

NOTES_DIR = os.path.join(os.path.dirname(__file__), "memory", "notes")
INDEX_FILE = os.path.join(NOTES_DIR, "_index.json")
PROFILE_SLUG = "profile"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "note"


def _ensure() -> None:
    os.makedirs(NOTES_DIR, exist_ok=True)
    profile = os.path.join(NOTES_DIR, f"{PROFILE_SLUG}.md")
    if not os.path.exists(profile):
        _write_raw(
            PROFILE_SLUG,
            title="User Profile",
            description="Who the user is and how they like to work.",
            body=(
                "The user is a Business Analyst (BA) who manages tasks, "
                "reminders and follow-ups across several work areas.\n\n"
                "## People\n_(add the people they follow up with here)_\n\n"
                "## Preferences\n_(add working preferences here)_\n"
            ),
        )


def _path(slug: str) -> str:
    return os.path.join(NOTES_DIR, f"{slug}.md")


_FM_RE = re.compile(
    r"^---\s*\ntitle:\s*(?P<title>.*?)\n"
    r"description:\s*(?P<description>.*?)\n"
    r"(?:updated:\s*(?P<updated>.*?)\n)?---\s*\n(?P<body>.*)$",
    re.DOTALL,
)


def _write_raw(slug: str, title: str, description: str, body: str) -> None:
    _ensure_dir_only()
    front = (
        f"---\ntitle: {title}\n"
        f"description: {description}\n"
        f"updated: {_now()}\n---\n"
    )
    with open(_path(slug), "w", encoding="utf-8") as fh:
        fh.write(front + body.rstrip() + "\n")


def _ensure_dir_only() -> None:
    os.makedirs(NOTES_DIR, exist_ok=True)


def _parse(slug: str) -> dict | None:
    path = _path(slug)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    m = _FM_RE.match(raw)
    if m:
        d = m.groupdict()
        return {
            "slug": slug,
            "title": d["title"].strip(),
            "description": d["description"].strip(),
            "updated": (d.get("updated") or "").strip(),
            "body": d["body"].strip(),
        }
    # no front matter: treat whole file as body
    return {
        "slug": slug,
        "title": slug,
        "description": "",
        "updated": "",
        "body": raw.strip(),
    }


# ===========================================================================
#  PUBLIC API
# ===========================================================================
def rebuild_index() -> list[dict]:
    """Regenerate the cheap table-of-contents over all notes."""
    _ensure()
    import json
    entries = []
    for fn in sorted(os.listdir(NOTES_DIR)):
        if fn.endswith(".md"):
            slug = fn[:-3]
            note = _parse(slug)
            if note:
                entries.append(
                    {
                        "slug": note["slug"],
                        "title": note["title"],
                        "description": note["description"],
                        "updated": note["updated"],
                        "chars": len(note["body"]),
                    }
                )
    with open(INDEX_FILE, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    return entries


def index() -> list[dict]:
    """The TOC the agent navigates first (PageIndex-style). Cheap to load."""
    return rebuild_index()


def read_note(slug: str) -> dict | None:
    """Load one note's full body -- the on-demand 'leaf' fetch."""
    _ensure()
    return _parse(slug)


def write_note(title: str, body: str, description: str = "",
               slug: str | None = None) -> dict:
    """Create or overwrite a memory note."""
    _ensure()
    slug = slug or _slugify(title)
    _write_raw(slug, title=title, description=description or title, body=body)
    rebuild_index()
    return _parse(slug)


def append_note(slug: str, text: str, bullet: bool = True) -> dict:
    """Append a line/paragraph to an existing note (or create it)."""
    _ensure()
    note = _parse(slug)
    if note is None:
        return write_note(title=slug, body=text, slug=slug)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    line = f"- ({stamp}) {text}" if bullet else text
    _write_raw(
        slug,
        title=note["title"],
        description=note["description"],
        body=note["body"] + "\n" + line,
    )
    rebuild_index()
    return _parse(slug)


def remember(text: str, topic: str = PROFILE_SLUG) -> dict:
    """Convenience: append a durable fact to a topic note (defaults to the
    user profile)."""
    return append_note(topic, text)


def search_notes(query: str) -> list[dict]:
    """Substring search across note bodies; returns matching index entries
    with a short snippet."""
    q = query.lower()
    hits = []
    for entry in index():
        note = _parse(entry["slug"])
        if not note:
            continue
        hay = (note["title"] + " " + note["description"] + " "
               + note["body"]).lower()
        if q in hay:
            idx = note["body"].lower().find(q)
            snippet = note["body"][max(0, idx - 40): idx + 80] if idx >= 0 \
                else note["description"]
            hits.append({**entry, "snippet": snippet.strip()})
    return hits


def delete_note(slug: str) -> bool:
    if slug == PROFILE_SLUG:
        raise ValueError("the profile note cannot be deleted")
    path = _path(slug)
    if os.path.exists(path):
        os.remove(path)
        rebuild_index()
        return True
    return False
