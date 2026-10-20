"""Google Calendar / Meet: the request we build, without touching the network.

The thing that failed before was not the HTTP call -- it was that no HTTP call
was ever made: the agent clicked around meet.google.com, failed, and wrote a
local reminder instead. So what matters here is that the request we WOULD send
is correct (conferenceData present, conferenceDataVersion=1, attendees parsed,
times carry an offset) and that the tool is honest when it cannot do the job.

The authorised session is stubbed, so this runs offline with no credentials.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime

import calendar_tools as C
import config

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        fails.append(msg)
    print(("   ok   " if cond else "   FAIL ") + msg)


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records the request instead of sending it."""

    def __init__(self, payload=None, status=200) -> None:
        self.calls: list[dict] = []
        self._payload = payload if payload is not None else {
            "id": "evt_abc123",
            "summary": "LMS review",
            "hangoutLink": "https://meet.google.com/abc-defg-hij",
            "htmlLink": "https://calendar.google.com/event?eid=xyz",
            "start": {"dateTime": "2026-09-19T17:00:00+05:30"},
            "end": {"dateTime": "2026-09-19T17:30:00+05:30"},
            "attendees": [{"email": "umesh@example.com"}],
        }
        self._status = status

    def post(self, url, params=None, json=None, timeout=None):
        self.calls.append({"verb": "POST", "url": url, "params": params,
                           "body": json})
        return FakeResponse(self._payload, self._status)

    def get(self, url, params=None, timeout=None):
        self.calls.append({"verb": "GET", "url": url, "params": params})
        return FakeResponse({"items": [self._payload]}, self._status)

    def delete(self, url, params=None, timeout=None):
        self.calls.append({"verb": "DELETE", "url": url, "params": params})
        return FakeResponse({}, 204)


# ------------------------------------------------- not connected = honest
print("--- unconfigured: says how to fix it, does not pretend ---")
C._session = lambda: None  # type: ignore[assignment]
r = C.create_meeting("LMS review", "tomorrow 5 PM", "umesh@example.com")
check(r["ok"] is False, "create_meeting reports failure when not connected")
check("Calendar API" in r["error"] and "--login" in r["error"],
      "the error tells the human exactly what to do")
check("meet_link" not in r, "it does NOT return a fabricated link")

# ------------------------------------------------------- the real request
print("\n--- the request we build is a Meet request ---")
fake = FakeSession()
C._session = lambda: fake  # type: ignore[assignment]

r = C.create_meeting("LMS review", "tomorrow 5 PM",
                     "umesh@example.com, rakesh@example.com",
                     duration_minutes=45, description="Agenda: stories")
call = fake.calls[-1]
body, params = call["body"], call["params"]

check(r["ok"] is True, "create_meeting succeeds")
check(params.get("conferenceDataVersion") == 1,
      "conferenceDataVersion=1 (without this Google silently returns NO link)")
check("createRequest" in body.get("conferenceData", {}),
      "the body asks Google to create a conference")
check(body["conferenceData"]["createRequest"]["conferenceSolutionKey"]["type"]
      == "hangoutsMeet", "it asks specifically for a Meet")
check(r["meet_link"] == "https://meet.google.com/abc-defg-hij",
      "the Meet link is returned to the caller")
check(params.get("sendUpdates") == "all",
      "guests are actually emailed the invite")

print("\n--- attendees ---")
check([a["email"] for a in body["attendees"]]
      == ["umesh@example.com", "rakesh@example.com"],
      "a comma-separated string is parsed into individual attendees")
fake.calls.clear()
C.create_meeting("t", "tomorrow 5 PM", ["a@x.com", "a@x.com", "not an email"])
guests = [a["email"] for a in fake.calls[-1]["body"].get("attendees", [])]
check(guests == ["a@x.com"],
      f"duplicates and junk are dropped rather than failing the call (got {guests})")

print("\n--- times ---")
start = datetime.fromisoformat(body["start"]["dateTime"])
end = datetime.fromisoformat(body["end"]["dateTime"])
check(start.tzinfo is not None,
      "the start time carries a UTC offset (else 5 PM means 5 PM somewhere else)")
check(int((end - start).total_seconds() // 60) == 45,
      "duration_minutes is honoured")
check(("timeZone" in body["start"]) == bool(config.TIMEZONE),
      "timeZone is sent only when config.TIMEZONE is set")

print("\n--- a nonsense time is refused before any request ---")
fake.calls.clear()
r = C.create_meeting("t", "sometime-ish maybe")
check(r["ok"] is False and not fake.calls,
      "an unparseable 'when' fails locally without calling Google")
check("could not understand" in r["error"], "and says so usefully")

# --------------------------------------------- event created but no link
print("\n--- event created but Google returned no link: warn, don't fake it ---")
no_link = FakeSession(payload={
    "id": "evt_x", "summary": "t",
    "start": {"dateTime": "2026-09-19T17:00:00+05:30"},
    "end": {"dateTime": "2026-09-19T17:30:00+05:30"},
})
C._session = lambda: no_link  # type: ignore[assignment]
r = C.create_meeting("t", "tomorrow 5 PM", "a@x.com")
check(r["ok"] is True and r["meet_link"] == "",
      "the empty link is reported as empty")
check("warning" in r, "a warning explains that conferencing did not happen")

# ----------------------------------------------------- an API error
print("\n--- a Google error is surfaced verbatim ---")
bad = FakeSession(payload={"error": {"message": "Insufficient Permission"}},
                  status=403)
C._session = lambda: bad  # type: ignore[assignment]
r = C.create_meeting("t", "tomorrow 5 PM")
check(r["ok"] is False and "Insufficient Permission" in r["error"],
      "the real Google message reaches the caller")

# ----------------------------------------------------------- list/cancel
print("\n--- list and cancel ---")
C._session = lambda: fake  # type: ignore[assignment]
fake.calls.clear()
r = C.list_meetings(days=3)
check(r["ok"] and r["count"] == 1, "list_meetings returns events")
check(fake.calls[-1]["params"]["singleEvents"] == "true",
      "recurring events are expanded into real occurrences")
check(r["events"][0]["meet_link"].startswith("https://meet.google.com/"),
      "listed events carry their Meet link")

fake.calls.clear()
r = C.cancel_meeting("evt_abc123")
check(r["ok"] and fake.calls[-1]["verb"] == "DELETE", "cancel_meeting deletes")
check(fake.calls[-1]["params"]["sendUpdates"] == "all",
      "guests are told it was cancelled")
check(C.cancel_meeting("")["ok"] is False, "an empty event_id is refused")

# ------------------------------------------------------------- the persona
print("\n--- the comms persona teaches this ---")
from multi_agent.roster import by_name  # noqa: E402

for label in ("persona", "persona_native"):
    p = getattr(by_name()["comms_agent"], label)
    check("create_meeting" in p, f"{label}: names create_meeting")
    check("meet.google.com" in p,
          f"{label}: explicitly forbids clicking meet.google.com")
    check("schedule_reminder" in p,
          f"{label}: warns that a local reminder is not a meeting")

print("\n" + "=" * 68)
if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
