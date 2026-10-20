"""A-F: the email path, from 'it cannot attach' to 'it cannot lie about it'.

Every check here corresponds to a defect that actually reached a real inbox or
a real board, not a hypothetical:

  A  send_email could not attach a file at all, so the agent pasted server
     paths into the body or tried to drive Gmail's UI (and failed).
  B  no USER_NAME existed, so bodies went out signed "[Your Name]".
  C  the comms persona's own worked example pointed fs_find at "downloads";
     generated reports live in project/reports, so the lookup could never work.
  D  nothing checked a body before SMTP.
  E  audit.jsonl had no email action of any kind, ever.
  F  a child could report "SEND FAILED" having made zero tool calls.

SMTP is never actually contacted: the send is stubbed, so this is free and
offline.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import agent_tools
import config
import mailer
import store
from multi_agent import swarm
from multi_agent.roster import by_name

fails: list[str] = []

# Redirect the audit log to a scratch file BEFORE anything sends: this test
# must not append fake "email_sent" rows to the real append-only history.
_audit_tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
_audit_tmp.close()
store.AUDIT_FILE = _audit_tmp.name


def check(cond: bool, msg: str) -> None:
    if not cond:
        fails.append(msg)
    print(("   ok   " if cond else "   FAIL ") + msg)


# ---------------------------------------------------------------- A: attach
print("--- A: send_email can attach a real file ---")
sent: dict = {}


class FakeSMTP:
    """Captures the message instead of talking to Gmail."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, **k):
        pass

    def login(self, *a):
        pass

    def send_message(self, msg):
        sent["msg"] = msg


mailer.smtplib.SMTP = FakeSMTP  # type: ignore[assignment]
config.GMAIL_ADDRESS = config.GMAIL_ADDRESS or "test@example.com"
config.GMAIL_APP_PASSWORD = config.GMAIL_APP_PASSWORD or "x" * 16

# A real file inside a real allowed root (project -> agent/data).
report_dir = config.REPORTS_DIR
os.makedirs(report_dir, exist_ok=True)
probe = os.path.join(report_dir, "_attach_probe.docx")
with open(probe, "wb") as fh:
    fh.write(b"PK\x03\x04probe-docx-bytes")

res = mailer.send_email(
    to="umesh@example.com", subject="Term loan overview",
    body="Hi Umesh,\n\nThe overview is attached.\n\nRegards,\nA",
    attachments=["project/reports/_attach_probe.docx"])

check(res.get("ok") is True, f"the send succeeded ({res.get('error','')})")
check(res.get("attachments") == ["_attach_probe.docx"],
      "the result names what was attached")
parts = [p.get_filename() for p in sent["msg"].iter_attachments()]
check(parts == ["_attach_probe.docx"],
      f"the MIME message really carries the file (got {parts})")

print("\n--- A: an attachment outside the allowed roots is refused ---")
res = mailer.send_email(to="x@example.com", subject="s", body="b",
                        attachments=["C:/Windows/System32/config/SAM"])
check(res.get("ok") is False and "root" in res.get("error", "").lower(),
      "an absolute off-root path is rejected, with an actionable reason")

print("\n--- A: one bad path fails the WHOLE send (no partial delivery) ---")
sent.clear()
res = mailer.send_email(
    to="x@example.com", subject="s", body="b",
    attachments=["project/reports/_attach_probe.docx", "project/reports/nope.xlsx"])
check(res.get("ok") is False, "a missing second file refuses the send")
check("msg" not in sent, "nothing was handed to SMTP")

# ---------------------------------------------------------------- B: name
print("\n--- B: there is a name to sign with ---")
check(bool(config.USER_NAME), f"config.USER_NAME resolves ({config.USER_NAME!r})")
_comms = by_name()["comms_agent"].persona
# "[Your Name]" SHOULD appear -- but only in the sentence forbidding it, never
# in a modelled body the agent might copy.
_placeholder_lines = [l for l in _comms.splitlines() if "[your name]" in l.lower()]
check(all("never" in l.lower() for l in _placeholder_lines),
      "the placeholder appears only in the rule forbidding it")
check(config.USER_NAME in _comms,
      "the persona tells the model which name to sign")

# ---------------------------------------------------------------- C: persona
print("\n--- C: the comms persona points at the right folder ---")
for label in ("persona", "persona_native"):
    p = getattr(by_name()["comms_agent"], label)
    check("project/reports" in p,
          f"{label}: names project/reports as where generated files live")
    # "downloads" may appear in a rule or a `wrong:` line -- it must never
    # appear in a line the model is told to COPY.
    right_lines = [l for l in p.splitlines() if l.lstrip().startswith("right:")]
    check(all("downloads" not in l.lower() for l in right_lines),
          f"{label}: no 'right:' example points at downloads")
    check(any("downloads" in l.lower() for l in p.splitlines()
              if l.lstrip().startswith(("wrong:", "'project/reports'", "NOT in"))
              or "NOT in 'downloads'" in l),
          f"{label}: downloads is explicitly called out as wrong")
    check("attachments" in p, f"{label}: teaches the attachments parameter")
    check("/api/reports" in p, f"{label}: warns against pasting /api/reports links")
    check("@@" not in p, f"{label}: no unsubstituted template markers")

# ---------------------------------------------------------------- D: guard
print("\n--- D: a defective body never reaches SMTP ---")
cases = [
    ("Regards,\n[Your Name]", "placeholder"),
    ("Get it at /api/reports/X.docx", "server path"),
    ("See /mnt/data/X.docx", "sandbox path"),
    ("Here is the **summary**", "markdown bold"),
    ("## Summary\nfine", "markdown heading"),
]
for body, label in cases:
    defects = agent_tools._email_body_defects(body)
    check(bool(defects), f"caught: {label}")

clean = ("Hi Umesh,\n\nBoth documents are attached: Term_Loan_Overview.docx and "
         "Term_Loan_Rules.xlsx.\n\n- Structures and eligibility\n- Approval "
         "rules\n\nRegards,\nAtharva")
check(agent_tools._email_body_defects(clean) == [],
      "a clean plain-text body passes untouched")

sent.clear()
out = agent_tools.send_email(to="x@example.com", subject="s",
                             body="Regards,\n[Your Name]")
check("msg" not in sent, "the defective body was NOT handed to SMTP")
rendered = getattr(out.content[0], "text", "")
check("NOT SENT" in rendered,
      "the tool tells the model plainly that it did not send")
check("placeholder" in rendered,
      "and names the defect so the model can fix it and retry")

# ---------------------------------------------------------------- E: audit
print("\n--- E: a send is recorded in the durable audit ---")
before = len(store.read_audit(limit=10_000))
mailer.send_email(to="umesh@example.com", subject="Audited",
                  body="plain body",
                  attachments=["project/reports/_attach_probe.docx"])
rows = store.read_audit(limit=10_000)
emails = [r for r in rows if r.get("action") == "email_sent"]
check(len(rows) > before, "a new audit row was written")
check(bool(emails), "there is now an 'email_sent' action in audit.jsonl")
if emails:
    last = emails[-1]
    check(last.get("to") == "umesh@example.com", "it records the recipient")
    check(last.get("attachments") == ["_attach_probe.docx"],
          "it records what was attached")

print("\n--- E: a REFUSED send is not audited as sent ---")
n_before = len([r for r in store.read_audit(limit=10_000)
                if r.get("action") == "email_sent"])
mailer.send_email(to="x@example.com", subject="s", body="b",
                  attachments=["project/reports/does_not_exist.docx"])
n_after = len([r for r in store.read_audit(limit=10_000)
               if r.get("action") == "email_sent"])
check(n_after == n_before, "a refused send writes no email_sent row")

# ---------------------------------------------------------------- F: honesty
print("\n--- F: claiming failure without trying is caught ---")
g = swarm._claims_failed_without_calling
check(g("STATUS: SEND FAILED", []), "SEND FAILED with zero tool calls -> nudge")
check(g("I was unable to send the email.", []), "prose failure, zero calls -> nudge")
check(not g("STATUS: SEND FAILED\nsmtplib error: auth failed", ["send_email"]),
      "a REAL failure after calling send_email is trusted, not nudged")
check(not g("STATUS: DRAFT - NOT SENT\nTo: x@y.com", []),
      "a legitimate draft ('NOT SENT' by design) is not flagged")
check(not g("Everything went fine.", []), "a success claim is left to the other guard")

# cleanup
for _path in (probe, _audit_tmp.name):
    try:
        os.remove(_path)
    except OSError:
        pass

print("\n" + "=" * 68)
if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
