"""Fast, deterministic tests for deliver_tools + the mobile delivery API
(Roadmap #5) -- NO LLM.

Exercises the OUTBOX channel the agent uses to "send a file to the phone":

  * deliver() copies a readable file into the outbox, records a manifest entry
    with a monotonic seq + a /api/outbox/<stored> url, and detects the right
    kind (image/audio/video/doc/sheet/text/other),
  * source resolution works for a BARE filename (looked up in reports/uploads)
    AND a root-qualified fs path ('dl/...'); a missing file errors cleanly,
  * the per-file size cap aborts,
  * api_deliveries(since) returns only newer rows, oldest-first,
  * resolve_outbox enforces the basename/realpath sandbox and restores the
    original display name (stripping the seq prefix),
  * the send_file / list_deliveries tool wrappers,
  * (if FastAPI TestClient is importable) the REAL HTTP path end-to-end:
    /api/deliveries + /api/outbox/<f>, AND the MOBILE_TOKEN auth middleware
    (a non-loopback client is 401 without the token, 200 with it).

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_deliver_tools
"""

from __future__ import annotations

import os
import tempfile

# --- point DATA_DIR / outbox / fs roots at throwaway temp dirs BEFORE config --
_PROJECT = tempfile.mkdtemp(prefix="hermes_dlv_project_")
_DL = tempfile.mkdtemp(prefix="hermes_dlv_downloads_")     # a fake "downloads" root
os.environ["DATA_DIR"] = _PROJECT
os.environ["FS_ALLOW_DOWNLOADS"] = "0"                     # don't touch real ~/Downloads
os.environ["FS_ALLOW_DOCUMENTS"] = "0"
os.environ["FS_EXTRA_ROOTS"] = f"dl={_DL}:rw"
os.environ["OUTBOX_DIR"] = os.path.join(_PROJECT, "outbox")

import config  # noqa: E402
config.DATA_DIR = _PROJECT
config.OUTBOX_DIR = os.path.join(_PROJECT, "outbox")
config.REPORTS_DIR = os.path.join(_PROJECT, "reports")
config.UPLOADS_DIR = os.path.join(_PROJECT, "uploads")
config.DELIVER_MAX_MB = 1.0                                # 1 MB cap for the test
for d in (config.OUTBOX_DIR, config.REPORTS_DIR, config.UPLOADS_DIR):
    os.makedirs(d, exist_ok=True)

import deliver_tools as D  # noqa: E402


def _text(chunk) -> str:
    return chunk.content[0].text


def _write(path: str, data: bytes) -> str:
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _run() -> int:
    ok = True

    def check(cond: bool, msg: str):
        nonlocal ok
        if cond:
            print(f"[ok] {msg}")
        else:
            print(f"[FAIL] {msg}")
            ok = False

    # --- seed some source files ---------------------------------------------
    _write(os.path.join(config.REPORTS_DIR, "q2_summary.docx"), b"PK\x03\x04 docx")
    _write(os.path.join(_DL, "photo.png"), b"\x89PNG\r\n\x1a\n fake")
    _write(os.path.join(config.UPLOADS_DIR, "song.mp3"), b"ID3 fake audio")
    _write(os.path.join(config.REPORTS_DIR, "clip.mp4"), b"\x00\x00\x00 ftyp")

    # --- deliver a generated report by BARE name ----------------------------
    rec = D.deliver("q2_summary.docx", caption="Here's Q2")
    check(rec["seq"] == 1 and rec["kind"] == "doc" and rec["caption"] == "Here's Q2",
          "deliver() by bare name finds the report, kind=doc, keeps caption")
    stored = os.path.join(config.OUTBOX_DIR, rec["stored"])
    check(os.path.isfile(stored) and rec["url"] == f"/api/outbox/{rec['stored']}",
          "deliver() copies the file into the outbox and sets the url")
    check(rec["filename"] == "q2_summary.docx" and rec["stored"].startswith("0001_"),
          "stored name is seq-prefixed; display name is the original")

    # --- deliver from an fs root (root-qualified path) + kind detection ------
    img = D.deliver("dl/photo.png")
    check(img["kind"] == "image" and img["seq"] == 2,
          "deliver('dl/photo.png') resolves via the fs allowlist, kind=image, seq increments")
    aud = D.deliver("song.mp3")            # bare -> uploads
    vid = D.deliver("clip.mp4")            # bare -> reports
    check(aud["kind"] == "audio" and vid["kind"] == "video",
          "kind detection: mp3=audio, mp4=video")

    # --- missing file + disallowed path error cleanly -----------------------
    check(_text(D.send_file("does_not_exist.pdf")).startswith("ERROR"),
          "send_file on a missing file returns ERROR (no crash)")
    check(_text(D.send_file("nope_root/secret.txt")).startswith("ERROR"),
          "send_file on an unknown root returns ERROR")

    # --- size cap -----------------------------------------------------------
    _write(os.path.join(config.REPORTS_DIR, "big.bin"), b"x" * (2 * 1024 * 1024))
    big = _text(D.send_file("big.bin"))
    check(big.startswith("ERROR") and "cap" in big,
          "send_file aborts a file over the DELIVER_MAX_MB cap")

    # --- send_file tool wrapper success -------------------------------------
    sf = _text(D.send_file("q2_summary.docx"))
    check(sf.startswith("Sent") and "mobile chat" in sf,
          "send_file tool returns a friendly confirmation")

    # --- api_deliveries(since) ----------------------------------------------
    all_now = D.api_deliveries(0)["deliveries"]
    seqs = [r["seq"] for r in all_now]
    check(seqs == sorted(seqs), "api_deliveries returns oldest-first")
    since2 = D.api_deliveries(2)["deliveries"]
    check(all(r["seq"] > 2 for r in since2) and len(since2) < len(all_now),
          "api_deliveries(since=2) returns only newer rows")

    # --- list_deliveries tool -----------------------------------------------
    lst = _text(D.list_deliveries())
    check("q2_summary.docx" in lst and "photo.png" in lst,
          "list_deliveries lists what was sent")

    # --- resolve_outbox sandbox + display-name restore ----------------------
    full, display = D.resolve_outbox(rec["stored"])
    check(os.path.isfile(full) and display == "q2_summary.docx",
          "resolve_outbox returns the real path + restores the original display name")
    for bad in ("../config.py", "..\\secret", "nope.bin"):
        try:
            D.resolve_outbox(bad)
            check(False, f"resolve_outbox should reject {bad!r}")
        except ValueError:
            check(True, f"resolve_outbox rejects {bad!r}")

    # --- HTTP end-to-end + auth middleware (optional: needs TestClient) ------
    try:
        from fastapi.testclient import TestClient
        import server

        client = TestClient(server.app)   # no 'with' -> don't fire startup loops

        # deliveries + outbox over real HTTP (no token configured here).
        config.MOBILE_TOKEN = ""
        r = client.get("/api/deliveries?since=0")
        check(r.status_code == 200 and any(d["filename"] == "q2_summary.docx"
              for d in r.json()["deliveries"]),
              "GET /api/deliveries serves the manifest")
        rd = client.get(rec["url"])
        check(rd.status_code == 200
              and rd.headers.get("content-disposition", "").find("q2_summary.docx") >= 0,
              "GET /api/outbox/<f> downloads the bytes with the original filename")
        check(client.get("/api/outbox/..%2f..%2fconfig.py").status_code == 404,
              "GET /api/outbox traversal is 404")

        # auth: with a token set, a non-loopback client (TestClient host is
        # 'testclient') must present it.
        config.MOBILE_TOKEN = "s3cret-token"
        os.environ["MOBILE_AUTH_ALLOW_LOOPBACK"] = "1"
        check(client.get("/api/now").status_code == 401,
              "with MOBILE_TOKEN set, a remote client is 401 without the token")
        check(client.get("/api/now", headers={"Authorization": "Bearer s3cret-token"}
              ).status_code == 200,
              "Bearer token is accepted")
        check(client.get("/api/now?token=s3cret-token").status_code == 200,
              "?token= is accepted (for media/download URLs)")
        check(client.get("/api/now?token=wrong").status_code == 401,
              "a wrong token is rejected")
        config.MOBILE_TOKEN = ""          # reset
    except ImportError as exc:
        print(f"[skip] FastAPI TestClient unavailable ({exc}); HTTP/auth checks skipped")

    print("\n== deliver_tools:", "ALL PASS ==" if ok else "FAILURES ==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_run())
