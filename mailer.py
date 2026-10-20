"""Send email from the user's own Gmail via SMTP + an App Password.

Deterministic, no browser, no credentials through the LLM: the app password
lives in agent/.env (config.GMAIL_APP_PASSWORD) and is used straight with
smtplib. This sidesteps both Google's automated-browser login block and the
"can't handle credentials" refusal — the model only ever calls send_email(...).

One-time setup (see .env.example):
  1. Turn on 2-Step Verification on the Gmail account.
  2. Google Account -> Security -> App passwords -> generate one.
  3. Put GMAIL_ADDRESS and GMAIL_APP_PASSWORD in agent/.env.
"""

from __future__ import annotations

import mimetypes
import os
import smtplib
import ssl
from email.message import EmailMessage

import config

# Gmail rejects messages over 25MB. Cap below that so the failure is ours, with
# a useful message, rather than an opaque SMTP rejection after a slow upload.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_ATTACHMENTS = 10


def is_configured() -> bool:
    return bool(config.GMAIL_ADDRESS and config.GMAIL_APP_PASSWORD)


def _split_addrs(field: str | None) -> list[str]:
    """Split a comma-separated To/Cc field into individual addresses."""
    return [a.strip() for a in (field or "").split(",") if a.strip()]


def _attach(msg: EmailMessage, paths: list[str]) -> tuple[list[str], str]:
    """Attach each root-qualified path. Returns (attached names, error).

    Paths are resolved through fs_tools' allowlist, so the model can only ever
    attach something inside an allowed root -- it cannot talk this into mailing
    an arbitrary file off the disk. A single bad path fails the WHOLE send:
    silently mailing 1 of 2 promised documents is worse than not sending.
    """
    if len(paths) > MAX_ATTACHMENTS:
        return [], f"too many attachments ({len(paths)}); max {MAX_ATTACHMENTS}"

    names, total = [], 0
    for path in paths:
        try:
            import fs_tools
            full = fs_tools.resolve_readable(path)
        except Exception as exc:
            return [], (
                f"cannot attach '{path}': {exc}. Paths must be root-qualified "
                "(e.g. 'project/reports/Term_Loan_Overview.docx'). Use fs_find "
                "to locate the file first; generated reports live under "
                "'project/reports'."
            )
        try:
            size = os.path.getsize(full)
            total += size
            if total > MAX_ATTACHMENT_BYTES:
                return [], (
                    f"attachments exceed {MAX_ATTACHMENT_BYTES // (1024*1024)}MB "
                    f"in total at '{path}'")
            ctype, _enc = mimetypes.guess_type(full)
            maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
            with open(full, "rb") as fh:
                msg.add_attachment(fh.read(), maintype=maintype,
                                   subtype=subtype or "octet-stream",
                                   filename=os.path.basename(full))
            names.append(os.path.basename(full))
        except OSError as exc:
            return [], f"cannot read '{path}': {exc}"
    return names, ""


def send_email(to: str, subject: str, body: str, cc: str | None = None,
               attachments: list[str] | None = None) -> dict:
    """Send a plain-text email, optionally with files attached.

    Returns {ok, ...} — never raises.
    """
    if not is_configured():
        return {
            "ok": False,
            "error": "Gmail not configured. Add GMAIL_ADDRESS and "
                     "GMAIL_APP_PASSWORD to agent/.env (app password from "
                     "Google Account -> Security -> App passwords).",
        }
    if not to or not to.strip():
        return {"ok": False, "error": "missing recipient 'to'"}

    msg = EmailMessage()
    msg["From"] = config.GMAIL_ADDRESS
    msg["To"] = to
    if cc:
        msg["Cc"] = cc
    msg["Subject"] = subject or "(no subject)"
    msg.set_content(body or "")

    attached: list[str] = []
    if attachments:
        attached, err = _attach(msg, [p for p in attachments if str(p).strip()])
        if err:
            # Refused BEFORE the send, so nothing half-complete goes out.
            return {"ok": False, "error": err, "sent": False}

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as s:
            s.starttls(context=ctx)
            s.login(config.GMAIL_ADDRESS, config.GMAIL_APP_PASSWORD)
            s.send_message(msg)
        # Implicit self-learning: remember everyone we just emailed so the people
        # directory grows on its own (best-effort; never affects the send result).
        try:
            import people_store
            for addr in _split_addrs(to) + _split_addrs(cc):
                people_store.observe_email_recipient(addr, source="email-sent")
        except Exception:
            pass
        # Durable record that mail actually left the machine. Until this
        # existed, audit.jsonl had no email action of any kind -- the ONLY
        # trace of a send was the board's tools_used, which dies with the turn.
        try:
            import store
            store.audit("email_sent", to=to, cc=cc or "",
                        subject=msg["Subject"],
                        attachments=attached, body_chars=len(body or ""))
        except Exception:
            pass
        return {"ok": True, "from": config.GMAIL_ADDRESS, "to": to,
                "cc": cc, "subject": msg["Subject"],
                "attachments": attached,
                "attached_count": len(attached)}
    except smtplib.SMTPAuthenticationError:
        return {"ok": False, "error": "SMTP auth failed — check GMAIL_ADDRESS / "
                "GMAIL_APP_PASSWORD (must be an App Password, not the normal "
                "password; 2-Step Verification must be on)."}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
