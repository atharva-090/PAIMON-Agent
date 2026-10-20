"""Delivery / outbox tools for the Hermes agent (Roadmap #5 -- "Paimon anywhere").

This is how the agent SENDS A FILE to the user's phone. Where file_tools READS
the uploads folder, report_tools WRITES the reports folder, and fs_tools reaches
the allowlisted roots, deliver_tools takes any file the agent is allowed to read
and pushes a copy into the OUTBOX -- a single folder served to the mobile app.
The phone polls /api/deliveries and renders each delivery inline (image / audio /
video) or as a download chip (pdf / doc / sheet / anything else).

Design (matches the rest of the project):

  * READ THROUGH THE EXISTING ALLOWLIST. The source path is resolved with
    fs_tools (root-qualified: 'project/reports/q2.docx', 'downloads/photo.jpg',
    'documents/contract.pdf'), so the agent can only deliver files it is already
    allowed to read -- no new file-access surface. A BARE filename is also
    accepted as a convenience and looked up in the reports / uploads / outbox
    folders (the common "I just generated q2.docx -> send it" case).

  * THE OUTBOX IS SANDBOXED + BOUNDED. Files are copied (never moved) into
    OUTBOX_DIR under a unique, sanitised name; a single file is capped at
    DELIVER_MAX_MB. A JSON manifest records each delivery with a monotonic seq
    the mobile client uses to fetch only what's new.

Tools: send_file (the one the agent calls), list_deliveries.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import threading
from datetime import datetime, timezone

from agentscope.message import TextBlock
from agentscope.tool import ToolChunk

import config

_LOCK = threading.Lock()
_MANIFEST_NAME = "_deliveries.json"

# Extension -> coarse "kind" the mobile app switches on for rendering.
_KIND = {
    "image": {"png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "heic", "heif", "avif"},
    "audio": {"mp3", "wav", "m4a", "ogg", "oga", "flac", "aac", "opus", "weba"},
    "video": {"mp4", "mov", "webm", "mkv", "avi", "m4v", "3gp"},
    "pdf":   {"pdf"},
    "doc":   {"doc", "docx", "odt", "rtf"},
    "sheet": {"xls", "xlsx", "csv", "tsv", "ods"},
    "slides": {"ppt", "pptx", "odp"},
    "archive": {"zip", "rar", "7z", "tar", "gz", "tgz"},
    "text":  {"txt", "md", "markdown", "log", "json", "xml", "yaml", "yml", "html", "htm", "ini", "cfg"},
}


# ===========================================================================
#  helpers
# ===========================================================================
def _ok(text: str) -> ToolChunk:
    return ToolChunk(content=[TextBlock(type="text", text=text)])


def _err(msg: str) -> ToolChunk:
    return ToolChunk(content=[TextBlock(type="text", text=f"ERROR: {msg}")])


def _kind_of(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    for kind, exts in _KIND.items():
        if ext in exts:
            return kind
    return "other"


def _human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{int(f)} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GB"


def _outbox() -> str:
    os.makedirs(config.OUTBOX_DIR, exist_ok=True)
    return config.OUTBOX_DIR


def _manifest_path() -> str:
    return os.path.join(_outbox(), _MANIFEST_NAME)


def _load_manifest() -> list:
    try:
        with open(_manifest_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_manifest(rows: list) -> None:
    tmp = _manifest_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, _manifest_path())


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    base = os.path.basename(name or "file")
    base = _SAFE.sub("_", base).strip("._") or "file"
    return base[:120]


def _resolve_source(path: str) -> str:
    """Return the absolute path of the file to deliver, or raise ValueError.

    Tries the fs_tools allowlist first (root-qualified paths like
    'project/reports/x.docx' or 'downloads/y.pdf'); falls back to a BARE
    filename looked up in the reports / uploads / outbox folders and the project
    data root. Never escapes those locations."""
    p = (path or "").strip()
    if not p:
        raise ValueError("no path given")

    # 1) root-qualified path via the scoped filesystem allowlist.
    looks_qualified = ("/" in p.replace("\\", "/")) or (":" in p.split("/", 1)[0])
    if looks_qualified and config.FS_ENABLED:
        try:
            import fs_tools
            return fs_tools.resolve_readable(p.replace("\\", "/"))
        except Exception:
            pass  # fall through to the bare-name search

    # 2) bare filename -> search the agent's own output/input folders.
    base = os.path.basename(p.replace("\\", "/"))
    for folder in (config.REPORTS_DIR, config.UPLOADS_DIR, _outbox(), config.DATA_DIR):
        cand = os.path.realpath(os.path.join(folder, base))
        root = os.path.realpath(folder)
        if (cand == root or cand.startswith(root + os.sep)) and os.path.isfile(cand):
            return cand
    raise ValueError(
        f"could not find '{path}'. Use a root-qualified path "
        "('project/reports/q2.docx', 'downloads/photo.jpg') or a bare filename "
        "of a file you generated (in reports) or that was uploaded.")


# ===========================================================================
#  the core operation (shared by the tool + tests)
# ===========================================================================
def deliver(path: str, caption: str = "") -> dict:
    """Copy a readable file into the outbox and record a delivery. Returns the
    delivery record dict. Raises ValueError on a bad/oversized/missing source."""
    src = _resolve_source(path)
    size = os.path.getsize(src)
    cap = int(config.DELIVER_MAX_MB * 1024 * 1024)
    if size > cap:
        raise ValueError(
            f"file is {_human_size(size)} -- over the {config.DELIVER_MAX_MB} MB "
            "delivery cap. Share a smaller file or a download link instead.")

    with _LOCK:
        rows = _load_manifest()
        seq = (max((r.get("seq", 0) for r in rows), default=0)) + 1
        display = os.path.basename(src)
        stored = f"{seq:04d}_{_safe_name(display)}"
        dst = os.path.join(_outbox(), stored)
        shutil.copy2(src, dst)
        kind = _kind_of(display)
        mime = mimetypes.guess_type(display)[0] or "application/octet-stream"
        rec = {
            "id": f"dlv_{seq:04d}",
            "seq": seq,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "filename": display,
            "stored": stored,
            "kind": kind,
            "mime": mime,
            "size": size,
            "size_kb": round(size / 1024, 1),
            "size_human": _human_size(size),
            "caption": (caption or "").strip(),
            "url": f"/api/outbox/{stored}",
        }
        rows.append(rec)
        _save_manifest(rows)
    return rec


# ===========================================================================
#  agent tools
# ===========================================================================
def send_file(path: str, caption: str = "") -> ToolChunk:
    """Send a file (or media) to the user's phone. Use this whenever the user
    asks you to "send me", "share", or "give me" a document / report / image /
    audio / video / any file. The file shows up in their mobile chat as an inline
    image/audio/video or a downloadable chip.

    You can send: a file you just GENERATED (e.g. a report 'q2_summary.docx' --
    pass the bare filename or 'project/reports/q2_summary.docx'); a file in the
    user's Downloads or Documents (pass 'downloads/photo.jpg',
    'documents/contract.pdf'); or anything else you're allowed to read. You can
    only send files you can already read -- there is no new access here.

    Args:
        path (str): The file to send. A root-qualified path
            ('project/reports/q2.docx', 'downloads/photo.jpg',
            'documents/x.pdf') OR a bare filename of something you generated /
            that was uploaded ('q2_summary.docx').
        caption (str): Optional one-line note shown under the file on the phone.
    """
    try:
        rec = deliver(path, caption)
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:  # pragma: no cover
        return _err(f"could not send '{path}': {type(exc).__name__}: {exc}")
    return _ok(
        f"Sent '{rec['filename']}' to the user's phone "
        f"({rec['kind']}, {rec['size_human']}). It now appears in their mobile "
        f"chat as delivery {rec['id']}.")


def list_deliveries() -> ToolChunk:
    """List the files you've recently sent to the user's phone (newest first),
    so you can confirm what was delivered. Read-only."""
    rows = sorted(_load_manifest(), key=lambda r: r.get("seq", 0), reverse=True)
    slim = [{"id": r["id"], "filename": r["filename"], "kind": r["kind"],
             "size": r.get("size_human"), "when": r.get("ts")} for r in rows[:30]]
    return _ok(json.dumps({"count": len(rows), "recent": slim}, ensure_ascii=False))


# ===========================================================================
#  API helpers (for server.py -- the mobile delivery channel)
# ===========================================================================
def api_deliveries(since: int = 0, limit: int = 200) -> dict:
    """Deliveries with seq > ``since`` (oldest-first so the client can append),
    capped to the most recent ``limit``. The mobile app polls this."""
    rows = [r for r in _load_manifest() if r.get("seq", 0) > int(since or 0)]
    rows.sort(key=lambda r: r.get("seq", 0))
    if len(rows) > limit:
        rows = rows[-limit:]
    return {"deliveries": rows}


def resolve_outbox(stored: str) -> tuple:
    """(abs_path, download_filename) for a delivered file, or raise ValueError.
    Sandboxed: only a plain name inside OUTBOX_DIR is served (basename + realpath
    guard), and the download keeps the file's ORIGINAL display name."""
    base = os.path.realpath(_outbox())
    safe = os.path.basename(stored or "")
    full = os.path.realpath(os.path.join(base, safe))
    if (full != base and not full.startswith(base + os.sep)) or not os.path.isfile(full):
        raise ValueError("no such delivered file")
    display = safe
    for r in _load_manifest():
        if r.get("stored") == safe:
            display = r.get("filename", safe)
            break
    else:
        display = re.sub(r"^\d{4}_", "", safe)  # strip the seq prefix if unknown
    return full, display
