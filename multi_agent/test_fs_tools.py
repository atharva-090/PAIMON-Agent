"""Fast, deterministic tests for fs_tools (Roadmap #4) -- NO LLM.

Exercises the scoped multi-root filesystem end-to-end against throwaway temp
roots and asserts the REAL behaviour, not just "a tool returned ok":

  * root resolution + containment (unknown root, '..' traversal, absolute path
    all rejected),
  * read/write/append round-trips inside the free 'project' root,
  * the CONFIRM GATE outside the project: write/append/move/delete/download into
    a confirm-required root does NOTHING without confirm=True, and actually does
    it WITH confirm=True,
  * per-root PERMISSIONS (a read+write root refuses delete),
  * fs_move across roots, fs_find, fs_download (real fetch from a local HTTP
    server + the size cap),
  * the API helpers the web explorer uses (api_roots / api_list / resolve_readable),
  * build_tools() wraps every tool cleanly (catches docstring/signature drift),
  * the browser attach/download tools' path validation + command construction
    (browser stack stubbed; skipped cleanly if it can't import).

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_fs_tools
"""

from __future__ import annotations

import http.server
import os
import threading
import tempfile

# --- point the FS allowlist at throwaway roots BEFORE config is imported -----
_PROJECT = tempfile.mkdtemp(prefix="hermes_fs_project_")
_EXT = tempfile.mkdtemp(prefix="hermes_fs_ext_")      # read+write, confirm, NO delete
_EXTD = tempfile.mkdtemp(prefix="hermes_fs_extd_")    # read+write+delete, confirm
os.environ["DATA_DIR"] = _PROJECT
os.environ["FS_ALLOW_DOWNLOADS"] = "0"                # don't touch the real ~/Downloads
os.environ["FS_ALLOW_DOCUMENTS"] = "0"
os.environ["FS_EXTRA_ROOTS"] = f"ext={_EXT}:rw;extd={_EXTD}:rwd"

import config  # noqa: E402
config.DATA_DIR = _PROJECT
import fs_tools as FS  # noqa: E402


def _text(chunk) -> str:
    return chunk.content[0].text


# A tiny local HTTP server so fs_download can do a REAL fetch deterministically.
class _Payload(http.server.BaseHTTPRequestHandler):
    BODY = b"hello,world\nrow,1\n"

    def do_GET(self):  # noqa: N802
        if self.path == "/big.bin":
            body = b"x" * (3 * 1024 * 1024)           # 3 MB, to trip the cap
        else:
            body = self.BODY
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                         # silence the server
        pass


def _serve() -> tuple:
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Payload)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _run() -> int:
    ok = True

    def check(cond: bool, msg: str):
        nonlocal ok
        if cond:
            print(f"[ok] {msg}")
        else:
            print(f"[FAIL] {msg}")
            ok = False

    # --- roots + guide -------------------------------------------------------
    roots_txt = _text(FS.fs_list_roots())
    check(all(n in roots_txt for n in ("project", "ext", "extd")),
          "fs_list_roots lists project + the two extra roots")
    rules = _text(FS.fs_rules())
    check("confirm=True" in rules and "root-qualified" in rules.lower(),
          "fs_rules explains the confirm gate + root-qualified paths")

    # --- resolution + containment -------------------------------------------
    check(_text(FS.fs_read("nope/x.txt")).startswith("ERROR")
          and "unknown root" in _text(FS.fs_read("nope/x.txt")),
          "unknown root is rejected")
    check("escapes" in _text(FS.fs_read("project/../escape.txt")),
          "'..' traversal is rejected")
    check(_text(FS.fs_read(r"C:\Windows\win.ini")).startswith("ERROR"),
          "a bare absolute path is rejected (not root-qualified)")

    # --- write / read / append inside the FREE project root -----------------
    w = _text(FS.fs_write("project/out/notes.md", "# Hi\nline1\n"))
    check(w.startswith("Wrote") and os.path.isfile(os.path.join(_PROJECT, "out", "notes.md")),
          "fs_write creates a file in project (no confirm needed)")
    rd = _text(FS.fs_read("project/out/notes.md"))
    check("# Hi" in rd and "line1" in rd, "fs_read returns the written content")
    FS.fs_append("project/out/notes.md", "line2\n")
    check("line2" in _text(FS.fs_read("project/out/notes.md")),
          "fs_append adds to the file")
    listing = _text(FS.fs_list("project/out"))
    check("notes.md" in listing, "fs_list shows the new file")

    # --- CONFIRM GATE outside the project -----------------------------------
    gated = _text(FS.fs_write("ext/report.txt", "secret"))
    check(gated.startswith("PENDING_CONFIRMATION")
          and not os.path.exists(os.path.join(_EXT, "report.txt")),
          "fs_write outside project is gated and writes NOTHING without confirm")
    done = _text(FS.fs_write("ext/report.txt", "secret", confirm=True))
    check(done.startswith("Wrote") and os.path.isfile(os.path.join(_EXT, "report.txt")),
          "fs_write with confirm=True actually writes outside project")
    check(_text(FS.fs_append("ext/report.txt", "more")).startswith("PENDING_CONFIRMATION"),
          "fs_append outside project is gated too")

    # --- PERMISSIONS: ext grants rw but NOT delete --------------------------
    perm = _text(FS.fs_delete("ext/report.txt", confirm=True))
    check(perm.startswith("ERROR") and "does not allow delete" in perm,
          "fs_delete refused on a no-delete root even with confirm")
    # extd DOES allow delete (but still confirm-gated).
    FS.fs_write("extd/tmp.txt", "x", confirm=True)
    check(_text(FS.fs_delete("extd/tmp.txt")).startswith("PENDING_CONFIRMATION"),
          "fs_delete gated without confirm on a deletable confirm root")
    check(_text(FS.fs_delete("extd/tmp.txt", confirm=True)).startswith("Deleted")
          and not os.path.exists(os.path.join(_EXTD, "tmp.txt")),
          "fs_delete with confirm actually removes the file")

    # --- MOVE: project -> extd is gated by the dest; with confirm it moves ---
    FS.fs_write("project/move_me.txt", "payload")
    mg = _text(FS.fs_move("project/move_me.txt", "extd/moved.txt"))
    check(mg.startswith("PENDING_CONFIRMATION")
          and os.path.isfile(os.path.join(_PROJECT, "move_me.txt"))
          and not os.path.exists(os.path.join(_EXTD, "moved.txt")),
          "fs_move into a confirm root is gated; source untouched")
    md = _text(FS.fs_move("project/move_me.txt", "extd/moved.txt", confirm=True))
    check(md.startswith("Moved")
          and not os.path.exists(os.path.join(_PROJECT, "move_me.txt"))
          and os.path.isfile(os.path.join(_EXTD, "moved.txt")),
          "fs_move with confirm relocates the file across roots")
    # A move within the free project root needs no confirm.
    FS.fs_write("project/a.txt", "a")
    check(_text(FS.fs_move("project/a.txt", "project/b.txt")).startswith("Moved"),
          "fs_move within project is free")

    # --- FIND ----------------------------------------------------------------
    found = _text(FS.fs_find("project", "notes"))
    check("notes.md" in found, "fs_find locates a file by name substring")

    # --- DOWNLOAD: confirm gate, real fetch, and the size cap ---------------
    srv, base_url = _serve()
    try:
        dg = _text(FS.fs_download(base_url + "/data.txt", "ext"))
        check(dg.startswith("PENDING_CONFIRMATION"),
              "fs_download into a confirm root is gated (no fetch)")
        dl = _text(FS.fs_download(base_url + "/data.txt", "project/dl"))
        got = os.path.join(_PROJECT, "dl", "data.txt")
        check(dl.startswith("Downloaded") and os.path.isfile(got)
              and open(got, "rb").read() == _Payload.BODY,
              "fs_download fetches into project and writes the real bytes")
        check(_text(FS.fs_download("ftp://x/y", "project")).startswith("ERROR"),
              "fs_download rejects non-http(s) schemes")
        old_cap = config.FS_DOWNLOAD_MAX_MB
        config.FS_DOWNLOAD_MAX_MB = 1                   # 1 MB cap vs a 3 MB body
        capped = _text(FS.fs_download(base_url + "/big.bin", "project/big.bin"))
        config.FS_DOWNLOAD_MAX_MB = old_cap
        check(capped.startswith("ERROR") and "cap" in capped
              and not os.path.exists(os.path.join(_PROJECT, "big.bin")),
              "fs_download aborts + cleans up when over the size cap")
    finally:
        srv.shutdown()

    # --- API helpers (web explorer) -----------------------------------------
    api_roots = FS.api_roots()
    check(any(r["name"] == "project" and r["confirm_required"] is False for r in api_roots)
          and any(r["name"] == "ext" and r["confirm_required"] is True for r in api_roots),
          "api_roots reports confirm_required correctly per root")
    al = FS.api_list("project/out")
    check(any(f["name"] == "notes.md" for f in al["files"]),
          "api_list returns files for the explorer")
    rr = FS.resolve_readable("project/out/notes.md")
    check(os.path.isfile(rr), "resolve_readable returns a real path for a readable file")
    try:
        FS.resolve_readable("ext/../../etc/passwd")
        check(False, "resolve_readable rejects traversal")
    except ValueError:
        check(True, "resolve_readable rejects traversal")

    # --- factory wraps every tool cleanly (docstring/signature parse) -------
    tools = FS.build_tools()
    check(len(tools) == 10, "build_tools wraps all 10 fs tools")
    check(len(FS.build_read_tools()) == 5, "build_read_tools is the 5 read-only tools")

    # --- browser attach/download: path validation + command (stubbed) -------
    try:
        import browser_tools as B

        calls = []

        def _fake(session, command, args=None, timeout=30):
            calls.append((command, list(args or [])))
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": "(stub)", "refs": {}}}
            return {"success": True}

        B.run_agent_browser_command = _fake          # stub the Node CLI shell-out
        # Attaching a real project file -> 'upload <sel> <abspath>'.
        real = os.path.join(_PROJECT, "out", "notes.md")
        att = _text(B.browse_attach_file("@e7", "project/out/notes.md"))
        up = [c for c in calls if c[0] == "upload"]
        check(att.startswith("OK") and up and up[0][1][0] == "@e7"
              and up[0][1][1] == real,
              "browse_attach_file resolves the path and calls upload <ref> <abspath>")
        # A file that doesn't exist / bad root -> clean error, no CLI call.
        calls.clear()
        bad = _text(B.browse_attach_file("@e1", "nope/ghost.pdf"))
        check(bad.startswith("ERROR") and not calls,
              "browse_attach_file rejects a non-allowlisted path before touching the browser")
        # Download into a confirm root is gated (no CLI call).
        calls.clear()
        dlg = _text(B.browse_download("@e2", "ext/export.csv"))
        check(dlg.startswith("PENDING_CONFIRMATION") and not calls,
              "browse_download into a confirm root is gated")
        # Download into project resolves + calls download.
        calls.clear()
        dlp = _text(B.browse_download("input#x", "project/export.csv"))
        dn = [c for c in calls if c[0] == "download"]
        check(dlp.startswith("OK") and dn and dn[0][1][0] == "input#x",
              "browse_download into project calls download <sel> <abspath>")
    except ImportError as exc:
        print(f"[skip] browser stack unavailable ({exc}); attach/download checks skipped")

    print("\n== fs_tools:", "ALL PASS ==" if ok else "FAILURES ==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_run())
