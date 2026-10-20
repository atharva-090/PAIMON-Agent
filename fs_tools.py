"""Scoped multi-root filesystem tools for the Hermes agent (Roadmap #4).

Where file_tools.py READS the uploads drop-folder and report_tools.py WRITES the
reports folder, fs_tools lets the agent work with files ANYWHERE ALLOWED -- the
project workspace AND the user's Downloads / Documents -- to list, read, edit,
move, delete and download files. It is the general "broader filesystem" layer
the attachment + download features build on.

Design (matches the rest of the project):

  * ALLOWLIST OF ROOTS, NOT THE WHOLE DISK. Every path is *root-qualified*: it
    begins with the NAME of a configured root, e.g. 'project/notes.txt' or
    'downloads/q2.csv'. There are no bare absolute paths -- a path that doesn't
    start with a known root name is refused. Each root (config.fs_roots()) has
    explicit read / write / delete permissions, and a realpath containment check
    rejects any '..' escape. This is the same sandbox shape as file_tools
    (locked to uploads/) and report_tools (locked to reports/), generalised to a
    short list of roots.

  * HARD CONFIRM-GATES OUTSIDE THE PROJECT. The 'project' root (DATA_DIR) is the
    agent's own sandbox -- writes/deletes there are free. Every OTHER root
    (Downloads, Documents, custom) is flagged confirm=True: a write / append /
    move / delete / download into it returns a PENDING_CONFIRMATION and does
    NOTHING until the tool is called again with confirm=True (the human's "yes").
    This mirrors browse_send_sensitive / browse_confirm_payment.

  * OUTPUT IS BOUNDED. Reads/listings are truncated to FS_MAX_OUTPUT and a
    download is capped at FS_DOWNLOAD_MAX_MB so neither the prompt nor the disk
    can be blown.

Tools: fs_rules (read first), fs_list_roots, fs_list, fs_find, fs_read,
fs_write, fs_append, fs_move, fs_delete, fs_download.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional, Tuple

from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, ToolChunk

import config

_MAX = config.FS_MAX_OUTPUT

# Text we read straight; documents we convert to markdown via markitdown (same
# split as file_tools). Anything else is treated as opaque binary (size only).
_TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml",
              ".yaml", ".yml", ".log", ".ini", ".cfg", ".html", ".htm", ".rtf",
              ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs",
              ".sh", ".bat", ".ps1", ".sql", ".toml", ".env"}
_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}
# How many entries fs_list / fs_find will ever return (keeps a giant folder from
# overflowing the prompt).
_LIST_CAP = 200
_FIND_CAP = 100
_FIND_MAX_SCAN = 20000  # don't walk forever in a huge tree


# ===========================================================================
#  helpers
# ===========================================================================
def _chunk(text: str) -> ToolChunk:
    if len(text) > _MAX:
        text = text[:_MAX] + f"\n…[truncated — {len(text)} chars total; read a " \
                             "narrower slice or a specific file]"
    return ToolChunk(content=[TextBlock(type="text", text=text)])


def _ok(obj) -> ToolChunk:
    return _chunk(obj if isinstance(obj, str)
                  else json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _err(msg: str) -> ToolChunk:
    return _chunk(f"ERROR: {msg}")


def _human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{int(f)} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GB"


def _roots() -> list:
    return config.fs_roots() if config.FS_ENABLED else []


def _find_root(name: str) -> Optional[dict]:
    for r in _roots():
        if r["name"].lower() == (name or "").strip().lower():
            return r
    return None


def _split(path: str) -> Tuple[str, str]:
    """Split a root-qualified path into (root_name, relpath). Accepts both
    'root/rel/sub' and 'root:rel/sub'; rejects an empty path."""
    s = (path or "").strip().replace("\\", "/").strip().lstrip("/")
    if not s:
        raise ValueError("path is empty -- use a root-qualified path like "
                         "'project/notes.txt' or 'downloads/file.csv'.")
    first_seg = s.split("/", 1)[0]
    if ":" in first_seg:                 # 'downloads:sub/file'
        head, _, rel = s.partition(":")
        return head.strip(), rel.strip().lstrip("/")
    head, _, rel = s.partition("/")      # 'downloads/sub/file'
    return head.strip(), rel.strip()


_PERM_KEY = {"read": "read", "write": "write", "delete": "delete"}


def _resolve(path: str, need: str) -> Tuple[dict, str, str]:
    """Resolve a root-qualified ``path`` to (root_spec, abs_path, display) while
    enforcing (a) the root exists, (b) it grants ``need`` (read/write/delete),
    and (c) the resolved absolute path stays inside the root. Raises ValueError
    with an actionable message otherwise."""
    name, rel = _split(path)
    root = _find_root(name)
    if root is None:
        allowed = [r["name"] for r in _roots()] or "(none -- fs disabled)"
        raise ValueError(
            f"unknown root '{name}'. Allowed roots: {allowed}. Paths must start "
            "with a root name, e.g. 'project/sub/file.txt'.")
    if not root.get(_PERM_KEY[need], False):
        raise ValueError(
            f"the '{name}' root does not allow {need} (permissions: "
            f"read={root['read']}, write={root['write']}, delete={root['delete']}).")
    base = os.path.realpath(root["path"])
    full = os.path.realpath(os.path.join(base, rel))
    if full != base and not full.startswith(base + os.sep):
        raise ValueError(f"'{path}' escapes the '{name}' root -- no '..' traversal.")
    return root, full, f"{name}/{rel}".rstrip("/")


def _confirm_gate(action: str, display: str, detail: str = "") -> ToolChunk:
    """Standard PENDING_CONFIRMATION for a destructive op outside the project.
    The op did NOT happen; the agent must get the user's explicit yes and then
    re-call the SAME tool with confirm=True."""
    tail = f" ({detail})" if detail else ""
    return _chunk(
        f"PENDING_CONFIRMATION ({action}). This would {action} OUTSIDE the "
        f"project workspace:\n  {display}{tail}\n"
        "Nothing was changed. Tell the user exactly what will happen and wait "
        "for an explicit 'yes'. Only after they approve, call this tool again "
        "with confirm=True.")


# ===========================================================================
#  guide tool (read first -- like report_rules / python_coding_rules)
# ===========================================================================
def fs_rules() -> ToolChunk:
    """Read this FIRST before using any fs_* tool. It explains root-qualified
    paths, which roots you can touch, and the confirm gate for anything written
    or deleted outside the project workspace.
    """
    lines = [
        "SCOPED FILESYSTEM -- how to work with files beyond the uploads folder",
        "====================================================================",
        "Every path is ROOT-QUALIFIED: it starts with a root NAME, then the path "
        "inside it, e.g. 'project/notes.txt', 'downloads/q2.csv', "
        "'documents/contracts/acme.pdf'. There are NO bare absolute paths "
        "(C:\\... is refused) -- always go through a root.",
        "",
        "ROOTS you can use right now (name -- permissions):",
    ]
    for r in _roots():
        perms = ",".join(p for p in ("read", "write", "delete") if r[p])
        gate = " [writes/deletes need confirm=True]" if r["confirm"] else " [free]"
        lines.append(f"  • {r['name']}  ({perms or 'no access'}){gate}  -> {r['path']}")
    lines += [
        "",
        "TOOLS:",
        "  fs_list_roots()                 -- the roots + permissions (machine form).",
        "  fs_list(path)                   -- list a folder, e.g. fs_list('downloads').",
        "  fs_find(root, query)            -- find files by name under a root.",
        "  fs_read(path)                   -- read a file (text, or a PDF/Word/"
        "Excel converted to text).",
        "  fs_write(path, content, confirm)-- create/overwrite a TEXT file.",
        "  fs_append(path, content, confirm)-- append to a text file.",
        "  fs_move(src, dst, confirm)      -- move/rename.",
        "  fs_delete(path, confirm)        -- delete a file.",
        "  fs_download(url, dest, confirm) -- fetch a URL into a folder.",
        "",
        "THE CONFIRM GATE (important): the 'project' root is your own sandbox -- "
        "writing/deleting there is free. EVERY other root (downloads, documents, "
        "...) is confirm-gated: a write/append/move/delete/download there returns "
        "PENDING_CONFIRMATION and does NOTHING. Show the user what will happen, "
        "get an explicit 'yes', THEN call the same tool again with confirm=True. "
        "Never write or delete outside the project without that yes.",
    ]
    return _ok("\n".join(lines))


# ===========================================================================
#  read-only tools
# ===========================================================================
def fs_list_roots() -> ToolChunk:
    """List the filesystem roots you're allowed to work in, with each root's
    read/write/delete permissions and whether writes there need confirmation.
    Call this (or fs_rules) before reaching outside the uploads/reports folders.
    """
    try:
        rows = [{
            "name": r["name"], "path": r["path"],
            "read": r["read"], "write": r["write"], "delete": r["delete"],
            "confirm_required": r["confirm"],
        } for r in _roots()]
        if not rows:
            return _ok("No filesystem roots are configured (FS_ENABLED=0).")
        return _ok({"roots": rows})
    except Exception as exc:  # pragma: no cover
        return _err(f"could not list roots: {type(exc).__name__}: {exc}")


def fs_list(path: str = "") -> ToolChunk:
    """List the contents of a folder inside an allowed root. Returns sub-folders
    and files (with sizes), each as a root-qualified path you can pass straight
    to fs_read / fs_move / etc.

    Args:
        path (str): A root, or a folder within one -- e.g. 'downloads',
            'project/reports', 'documents/contracts'. Empty lists the roots.
    """
    try:
        if not (path or "").strip():
            return fs_list_roots()
        root, full, disp = _resolve(path, need="read")
        if not os.path.exists(full):
            return _err(f"no such folder: {disp}")
        if not os.path.isdir(full):
            return _err(f"{disp} is a file, not a folder -- use fs_read to read it.")
        dirs, files, n = [], [], 0
        for name in sorted(os.listdir(full)):
            if name.startswith("."):
                continue
            child = os.path.join(full, name)
            qual = f"{disp}/{name}"
            n += 1
            if n > _LIST_CAP:
                continue
            if os.path.isdir(child):
                dirs.append({"path": qual + "/", "type": "folder"})
            else:
                files.append({
                    "path": qual,
                    "type": os.path.splitext(name)[1].lstrip(".").lower() or "?",
                    "size": _human_size(os.path.getsize(child)),
                })
        out = {"folder": disp, "folders": dirs, "files": files,
               "count": len(dirs) + len(files)}
        if n > _LIST_CAP:
            out["note"] = f"{n} entries; showing first {_LIST_CAP}. Use fs_find " \
                          "to narrow down."
        return _ok(out)
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except PermissionError as exc:
        return _err(f"permission denied: {exc}")
    except Exception as exc:
        return _err(f"could not list '{path}': {type(exc).__name__}: {exc}")


def fs_find(root: str, query: str) -> ToolChunk:
    """Find files whose NAME contains ``query`` (case-insensitive) anywhere under
    a root. Use it to locate a file before reading or attaching it -- e.g. find
    'invoice' in downloads.

    Args:
        root (str): The root (or sub-folder) to search, e.g. 'downloads' or
            'documents/work'.
        query (str): Substring to match in file names.
    """
    try:
        if not (query or "").strip():
            return _err("query is empty")
        _r, base, disp = _resolve(root or "", need="read")
        if not os.path.isdir(base):
            return _err(f"{disp} is not a folder to search.")
        q = query.strip().lower()
        hits, scanned = [], 0
        for cur, dnames, fnames in os.walk(base):
            dnames[:] = [d for d in dnames if not d.startswith(".")]
            for fn in fnames:
                scanned += 1
                if scanned > _FIND_MAX_SCAN:
                    break
                if q in fn.lower():
                    full = os.path.join(cur, fn)
                    rel = os.path.relpath(full, base).replace("\\", "/")
                    hits.append({"path": f"{disp}/{rel}".replace("//", "/"),
                                 "size": _human_size(os.path.getsize(full))})
                    if len(hits) >= _FIND_CAP:
                        break
            if len(hits) >= _FIND_CAP or scanned > _FIND_MAX_SCAN:
                break
        if not hits:
            return _ok(f"No files matching {query!r} under {disp}.")
        head = f"{len(hits)} match(es) for {query!r} under {disp}" + \
               (f" (capped at {_FIND_CAP})" if len(hits) >= _FIND_CAP else "")
        return _ok({"summary": head, "matches": hits})
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not search '{root}': {type(exc).__name__}: {exc}")


def _read_text(full: str) -> str:
    """Return a file's content as text: text files read directly; PDF/Word/Excel/
    PPT converted to markdown via markitdown; anything else summarised as binary."""
    ext = os.path.splitext(full)[1].lower()
    if ext in _TEXT_EXTS or ext == "":
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    if ext in _DOC_EXTS:
        from markitdown import MarkItDown
        md = MarkItDown().convert(full).text_content or ""
        if not md.strip() and ext == ".pdf":
            raise ValueError("this PDF has no extractable text (scanned/image-only).")
        return md
    raise ValueError(
        f"'{ext or 'no-extension'}' is a binary file ({_human_size(os.path.getsize(full))}); "
        "fs_read shows text only. For a spreadsheet you uploaded use the excel_* "
        "tools, or move it into uploads/ first.")


def fs_read(path: str) -> ToolChunk:
    """Read a file from an allowed root and return its text. Plain-text files are
    returned as-is; PDF / Word / PowerPoint / Excel are converted to text so you
    can read them. Large files are truncated -- read a specific file rather than a
    whole folder.

    Args:
        path (str): A root-qualified file path, e.g. 'downloads/notes.txt' or
            'documents/contracts/acme.pdf'.
    """
    try:
        root, full, disp = _resolve(path, need="read")
        if not os.path.exists(full):
            return _err(f"no such file: {disp}")
        if os.path.isdir(full):
            return _err(f"{disp} is a folder -- use fs_list to see its contents.")
        text = _read_text(full)
        head = f"{disp} · {len(text)} chars · {_human_size(os.path.getsize(full))}"
        return _chunk(f"{head}\n\n{text}")
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except PermissionError as exc:
        return _err(f"permission denied: {exc}")
    except Exception as exc:
        return _err(f"could not read '{path}': {type(exc).__name__}: {exc}")


# ===========================================================================
#  write tools (confirm-gated outside the project)
# ===========================================================================
def fs_write(path: str, content: str, confirm: bool = False) -> ToolChunk:
    """Create or OVERWRITE a text file in an allowed root. Inside the 'project'
    root this is free; anywhere else (downloads, documents, ...) it is confirm-
    gated -- the first call returns PENDING_CONFIRMATION and writes nothing; call
    again with confirm=True after the user says yes.

    Args:
        path (str): Destination, root-qualified, e.g. 'project/out/summary.md' or
            'documents/notes.txt'. Parent folders are created as needed.
        content (str): The text to write (replaces any existing content).
        confirm (bool): Set True ONLY after the user approved a write outside the
            project workspace.
    """
    try:
        root, full, disp = _resolve(path, need="write")
        if os.path.isdir(full):
            return _err(f"{disp} is a folder.")
        if root["confirm"] and not confirm:
            return _confirm_gate("write", disp, f"{len(content or '')} chars")
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        existed = os.path.isfile(full)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content or "")
        verb = "Overwrote" if existed else "Wrote"
        return _ok(f"{verb} {disp}  ({_human_size(os.path.getsize(full))}).")
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except PermissionError as exc:
        return _err(f"permission denied: {exc}")
    except Exception as exc:
        return _err(f"could not write '{path}': {type(exc).__name__}: {exc}")


def fs_append(path: str, content: str, confirm: bool = False) -> ToolChunk:
    """Append text to a file in an allowed root (creates it if missing). Confirm-
    gated outside the project, exactly like fs_write.

    Args:
        path (str): Root-qualified destination file.
        content (str): Text to append.
        confirm (bool): True only after approval for a write outside the project.
    """
    try:
        root, full, disp = _resolve(path, need="write")
        if os.path.isdir(full):
            return _err(f"{disp} is a folder.")
        if root["confirm"] and not confirm:
            return _confirm_gate("append to", disp, f"{len(content or '')} chars")
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "a", encoding="utf-8") as fh:
            fh.write(content or "")
        return _ok(f"Appended to {disp}  (now {_human_size(os.path.getsize(full))}).")
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except PermissionError as exc:
        return _err(f"permission denied: {exc}")
    except Exception as exc:
        return _err(f"could not append to '{path}': {type(exc).__name__}: {exc}")


def fs_move(src: str, dst: str, confirm: bool = False) -> ToolChunk:
    """Move or rename a file between/within allowed roots. Needs write on BOTH
    the source and destination roots; if EITHER is outside the project, it's
    confirm-gated (first call returns PENDING_CONFIRMATION, moves nothing).

    Args:
        src (str): Root-qualified source file, e.g. 'downloads/report.pdf'.
        dst (str): Root-qualified destination, e.g. 'project/uploads/report.pdf'
            (a folder, or a full new path).
        confirm (bool): True only after the user approved a move touching a
            non-project root.
    """
    try:
        sroot, sfull, sdisp = _resolve(src, need="write")
        if not os.path.isfile(sfull):
            return _err(f"no such file to move: {sdisp}")
        droot, dfull, ddisp = _resolve(dst, need="write")
        # If dst is an existing folder, keep the source filename inside it.
        if os.path.isdir(dfull):
            dfull = os.path.join(dfull, os.path.basename(sfull))
            ddisp = f"{ddisp}/{os.path.basename(sfull)}"
        if (sroot["confirm"] or droot["confirm"]) and not confirm:
            return _confirm_gate("move", f"{sdisp}  ->  {ddisp}")
        os.makedirs(os.path.dirname(dfull) or ".", exist_ok=True)
        shutil.move(sfull, dfull)
        return _ok(f"Moved {sdisp} -> {ddisp}.")
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except PermissionError as exc:
        return _err(f"permission denied: {exc}")
    except Exception as exc:
        return _err(f"could not move '{src}': {type(exc).__name__}: {exc}")


def fs_delete(path: str, confirm: bool = False) -> ToolChunk:
    """Delete a file from an allowed root. Requires the root to grant delete
    (Downloads/Documents do NOT by default). Confirm-gated outside the project:
    the first call returns PENDING_CONFIRMATION and deletes nothing.

    Args:
        path (str): Root-qualified file to delete.
        confirm (bool): True only after the user explicitly approved the delete.
    """
    try:
        root, full, disp = _resolve(path, need="delete")
        if os.path.isdir(full):
            return _err(f"{disp} is a folder; this tool only deletes files.")
        if not os.path.isfile(full):
            return _err(f"no such file: {disp}")
        if root["confirm"] and not confirm:
            return _confirm_gate("delete", disp,
                                 _human_size(os.path.getsize(full)))
        os.remove(full)
        return _ok(f"Deleted {disp}.")
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except PermissionError as exc:
        return _err(f"permission denied: {exc}")
    except Exception as exc:
        return _err(f"could not delete '{path}': {type(exc).__name__}: {exc}")


def fs_download(url: str, dest: str, confirm: bool = False) -> ToolChunk:
    """Download a file from a URL straight into an allowed folder. Confirm-gated
    outside the project (first call returns PENDING_CONFIRMATION, fetches
    nothing). Capped at FS_DOWNLOAD_MAX_MB; only http/https URLs.

    Args:
        url (str): The http(s) URL to fetch.
        dest (str): Root-qualified destination -- a folder ('downloads') to keep
            the URL's filename, or a full path with the filename you want
            ('project/out/data.csv').
        confirm (bool): True only after the user approved a download outside the
            project workspace.
    """
    try:
        url = (url or "").strip()
        if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
            return _err("only http(s) URLs are allowed.")
        root, full, disp = _resolve(dest, need="write")
        # A destination with no file extension (or an existing dir) is treated as
        # a FOLDER -> keep the URL's own filename inside it.
        if os.path.isdir(full) or not os.path.splitext(full)[1]:
            from urllib.parse import urlparse, unquote
            base = os.path.basename(unquote(urlparse(url).path)) or "download"
            full = os.path.join(full, base)
            disp = f"{disp}/{base}"
        if root["confirm"] and not confirm:
            return _confirm_gate("download into", disp, url)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)

        import urllib.request
        cap = int(config.FS_DOWNLOAD_MAX_MB * 1024 * 1024)
        req = urllib.request.Request(url, headers={"User-Agent": "Hermes/1.0"})
        written = 0
        with urllib.request.urlopen(req, timeout=60) as resp, open(full, "wb") as out:
            while True:
                buf = resp.read(65536)
                if not buf:
                    break
                written += len(buf)
                if written > cap:
                    out.close()
                    os.remove(full)
                    return _err(f"download exceeds the {config.FS_DOWNLOAD_MAX_MB} MB "
                                "cap -- aborted.")
                out.write(buf)
        return _ok(f"Downloaded {url}\n  -> {disp}  ({_human_size(written)}).")
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except PermissionError as exc:
        return _err(f"permission denied: {exc}")
    except Exception as exc:
        return _err(f"could not download '{url}': {type(exc).__name__}: {exc}")


# ===========================================================================
#  API helpers (raw dicts for the web file-explorer in server.py -- same
#  allowlist + containment as the tools, just without the ToolChunk wrapper)
# ===========================================================================
def api_roots() -> list:
    """Roots + permissions for the UI (no filesystem paths leaked to the page)."""
    return [{"name": r["name"], "read": r["read"], "write": r["write"],
             "delete": r["delete"], "confirm_required": r["confirm"]}
            for r in _roots()]


def api_list(path: str = "") -> dict:
    """Folder listing for the file-explorer. Empty path -> the roots. Raises
    ValueError on an unknown/forbidden/non-folder path (the server maps it to a
    4xx)."""
    if not (path or "").strip():
        return {"roots": api_roots()}
    _root, full, disp = _resolve(path, need="read")
    if not os.path.isdir(full):
        raise ValueError(f"{disp} is not a folder")
    dirs, files = [], []
    for name in sorted(os.listdir(full)):
        if name.startswith("."):
            continue
        child = os.path.join(full, name)
        qual = f"{disp}/{name}"
        if os.path.isdir(child):
            dirs.append({"name": name, "path": qual})
        else:
            files.append({
                "name": name, "path": qual,
                "type": os.path.splitext(name)[1].lstrip(".").lower() or "?",
                "size_kb": round(os.path.getsize(child) / 1024, 1)})
    return {"folder": disp, "folders": dirs, "files": files}


def resolve_readable(path: str) -> str:
    """Absolute path of a readable file in an allowed root, for the download
    endpoint (FileResponse). Raises ValueError if not allowed / not a file."""
    _root, full, disp = _resolve(path, need="read")
    if not os.path.isfile(full):
        raise ValueError(f"no such file: {disp}")
    return full


# ===========================================================================
#  factory
# ===========================================================================
# Reads/listings are read-only; the rest write/delete (sandboxed + confirm-gated).
_READONLY = {fs_rules, fs_list_roots, fs_list, fs_find, fs_read}
_FS_FUNCS = [
    fs_rules, fs_list_roots, fs_list, fs_find, fs_read,
    fs_write, fs_append, fs_move, fs_delete, fs_download,
]


def build_tools() -> list:
    """The full fs kit (read + write), for the file-analyst specialist."""
    return [FunctionTool(f, is_read_only=(f in _READONLY)) for f in _FS_FUNCS]


def build_read_tools() -> list:
    """Read-only subset -- for specialists (comms/browser) that need to LOCATE a
    file to attach but should not write/delete."""
    return [FunctionTool(f, is_read_only=True)
            for f in (fs_rules, fs_list_roots, fs_list, fs_find, fs_read)]
