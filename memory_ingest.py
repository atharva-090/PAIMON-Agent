"""Background ingest — fold dropped files into the memory wiki automatically.

Part of the cold path (Roadmap #2, the "ingest loop"). A periodic scan watches
two folders:

  * ``data/uploads/``   — files the user drops for the agent to analyse, and
  * ``data/knowledge/`` — a dedicated drop folder for reference material the
    agent should simply *know*.

For each NEW or CHANGED file (tracked by a small mtime+size+hash manifest so we
never re-ingest an unchanged file), we extract its text, summarise it (LLM when
available, deterministic otherwise), and upsert a wiki page for it. The user then
gets answers like "your Q2 deck says revenue was up 12%" without ever having
asked the agent to read that deck.

Self-contained + degrades gracefully: spreadsheets summarise via pandas, plain
text/markdown read directly, and richer documents go through markitdown if it's
installed (skipped with a note otherwise). The LLM is only an enhancer — the
deterministic summary (the file's first meaningful lines) is always available.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone

import config
import wiki_store

MANIFEST_FILE = os.path.join(os.path.dirname(__file__), "memory",
                             "ingest_manifest.json")

_SPREADSHEET_EXTS = {".xlsx", ".xls", ".xlsm", ".csv"}
_TEXT_EXTS = {".md", ".markdown", ".txt"}
_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".html", ".htm", ".rtf"}
_INGESTABLE = _SPREADSHEET_EXTS | _TEXT_EXTS | _DOC_EXTS

_MAX_HASH_BYTES = 1 << 16   # hash only the first 64 KB — cheap change detection
_MAX_TEXT = 6000            # cap extracted text fed to the summariser


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
#  manifest (what we've already ingested, and at what version)
# ===========================================================================
def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_FILE):
        return {}
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    tmp = MANIFEST_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, MANIFEST_FILE)


def _signature(path: str) -> dict:
    """A cheap change-signature for a file: size, mtime, and a hash of its first
    64 KB. Changing any of these re-ingests the file."""
    st = os.stat(path)
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            h.update(fh.read(_MAX_HASH_BYTES))
    except OSError:
        pass
    return {"size": st.st_size, "mtime": int(st.st_mtime), "hash": h.hexdigest()}


def _changed(sig: dict, prev: dict | None) -> bool:
    if not prev:
        return True
    return (sig["size"] != prev.get("size")
            or sig["mtime"] != prev.get("mtime")
            or sig["hash"] != prev.get("hash"))


# ===========================================================================
#  discovery + extraction
# ===========================================================================
def _watched_roots() -> list[str]:
    return [config.UPLOADS_DIR, config.KNOWLEDGE_DIR]


def watched_files() -> list[str]:
    """Every ingestable file across the watched folders (skips hidden files,
    the markitdown .cache, and temp files)."""
    out = []
    for root in _watched_roots():
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.startswith(".") or name.endswith(".tmp"):
                continue
            full = os.path.join(root, name)
            if not os.path.isfile(full):
                continue
            if os.path.splitext(name)[1].lower() in _INGESTABLE:
                out.append(full)
    return out


def _extract_text(path: str) -> str:
    """Pull readable text out of a file for summarisation. Spreadsheets become a
    schema+sample summary; text/markdown are read directly; richer documents go
    through markitdown if available. Returns '' if nothing extractable."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in _SPREADSHEET_EXTS:
            return _spreadsheet_text(path)
        if ext in _TEXT_EXTS:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()[:_MAX_TEXT]
        # richer docs: reuse file_tools' cached markitdown conversion
        import file_tools
        return (file_tools._to_markdown_cached(path) or "")[:_MAX_TEXT]
    except Exception as exc:
        print(f"[ingest] extract failed for {os.path.basename(path)}: "
              f"{type(exc).__name__}: {exc}")
        return ""


def _spreadsheet_text(path: str) -> str:
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    parts = [f"Spreadsheet {os.path.basename(path)}."]
    if ext == ".csv":
        sheets = {"(csv)": pd.read_csv(path, nrows=200)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, nrows=200)
    for label, df in sheets.items():
        cols = ", ".join(str(c) for c in df.columns)
        parts.append(f"Sheet '{label}': {len(df)} rows (sampled). Columns: {cols}.")
    return "\n".join(parts)[:_MAX_TEXT]


# ===========================================================================
#  summarisation (LLM enhancer + deterministic fallback)
# ===========================================================================
def summarise_deterministic(text: str, max_chars: int = 320) -> str:
    """No-LLM summary: the file's first meaningful, non-boilerplate lines. Crude
    but always available and never wrong (it's a literal excerpt)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out, used = [], 0
    for ln in lines:
        if used + len(ln) > max_chars:
            break
        out.append(ln)
        used += len(ln) + 1
    return " ".join(out)[:max_chars]


_SUMMARISE_SYS = (
    "Summarise this file for an assistant's long-term memory in 1-2 concise "
    "sentences: what it is and the key facts/figures worth remembering. Plain "
    "text only, no preamble."
)


async def _summarise(text: str, llm=None, profile: str | None = None) -> str:
    if not text.strip():
        return ""
    try:
        import memory_learn
        if llm is not None:                       # test/override hook
            raw = await llm(_SUMMARISE_SYS, text[:_MAX_TEXT])
        else:
            # Serialise on the shared memory lock so ingest summarisation and
            # session extraction never hit the one hermes-memory browser at once.
            async with memory_learn._memory_lock:
                raw = await memory_learn._llm_complete(
                    _SUMMARISE_SYS, text[:_MAX_TEXT], profile)
        # Unwrap any JSON/markdown envelope the backend leaked, THEN cut on a
        # word boundary. The old `raw[:400]` sliced envelopes mid-string, which
        # is how unterminated JSON ended up in the digest.
        raw = wiki_store.clean_fact_text(raw or "")
        if raw:
            return wiki_store.truncate_words(raw, 400)
    except Exception:
        pass
    return summarise_deterministic(text)


# ===========================================================================
#  ingest one file + the scan loop
# ===========================================================================
def _page_meta(path: str) -> tuple[str, str]:
    """(title, slug) for a file's wiki page."""
    stem = os.path.splitext(os.path.basename(path))[0]
    title = stem.replace("_", " ").replace("-", " ").strip().title() or "File"
    return title, "file-" + wiki_store._slugify(stem)


async def ingest_file(path: str, llm=None, profile: str | None = None) -> dict | None:
    """Extract → summarise → upsert one file as a wiki page. Returns a small
    record, or None if nothing extractable."""
    text = _extract_text(path)
    if not text.strip():
        return None
    summary = await _summarise(text, llm, profile)
    if not summary.strip():
        summary = summarise_deterministic(text)
    title, slug = _page_meta(path)
    base = os.path.basename(path)
    wiki_store.upsert_fact(
        title=title, fact=summary, source=f"file:{base}", confidence=0.7,
        tags=["file", "knowledge"], entities=[base.lower(), title.lower()],
        description=f"From file {base}", slug=slug)
    # A second, stable fact pinning the source path so it's always traceable.
    wiki_store.upsert_fact(title=title, fact=f"Source file: {base}",
                           source=f"file:{base}", confidence=0.6, slug=slug)
    return {"file": base, "slug": slug, "summary": summary[:80]}


async def scan_and_ingest(llm=None, profile: str | None = None,
                          max_files: int = 25) -> dict:
    """Scan the watched folders and ingest every new/changed file. Cheap when
    nothing changed (just stat+hash per file). Rebuilds the digest if anything
    was ingested. Returns a report."""
    manifest = _load_manifest()
    report = {"scanned": 0, "ingested": 0, "skipped": 0, "files": []}
    changed_any = False
    for path in watched_files():
        report["scanned"] += 1
        if report["ingested"] >= max_files:
            break
        try:
            sig = _signature(path)
        except OSError:
            continue
        if not _changed(sig, manifest.get(path)):
            report["skipped"] += 1
            continue
        rec = await ingest_file(path, llm, profile)
        if rec is None:
            # mark it seen anyway so we don't retry an unreadable file every scan
            manifest[path] = {**sig, "ingested_at": _now(), "slug": None}
            continue
        manifest[path] = {**sig, "ingested_at": _now(), "slug": rec["slug"]}
        report["ingested"] += 1
        report["files"].append(rec)
        changed_any = True
    _save_manifest(manifest)
    if changed_any:
        try:
            import memory_learn
            memory_learn.rebuild_digest()
        except Exception as exc:
            print(f"[ingest] digest rebuild failed: {exc}")
    if report["ingested"]:
        print(f"[ingest] {report['ingested']} file(s) folded into memory")
    return report
