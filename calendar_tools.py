"""Google Calendar + Meet: real events, real Meet links.

Why this exists: the swarm was asked to "schedule a Google Meet at 5 PM", drove
meet.google.com in the browser, failed to find the "Take meeting" button, and
quietly fell back to writing a LOCAL reminder that said "Google Meet at 5 PM"
with no link in it. The run then reported success. A Meet link cannot be minted
by clicking around -- the only supported way is Calendar's events.insert with
``conferenceData`` and ``conferenceDataVersion=1``. So that is what this does.

Deliberately NOT using google-api-python-client: it drags in google-api-core,
which requires opentelemetry-api>=1.44 while AgentScope pins 1.42.1. The
Calendar REST API is a handful of JSON endpoints, so we call them directly with
an authorised session and keep the dependency tree flat.

Auth is OAuth (a Meet link is created as YOU, so an app password cannot work):
  1. Google Cloud Console -> new project -> enable the Google Calendar API.
  2. Credentials -> Create credentials -> OAuth client ID -> Desktop app.
  3. Download the JSON to agent/data/google_credentials.json.
  4. Run:  .venv/Scripts/python.exe calendar_tools.py --login
     A browser opens once; the refresh token is cached in
     agent/data/google_token.json and reused forever after.

The token file is a credential. It stays on disk, is never logged, and is never
passed to a model -- the tools return only event details and links.

This is a SEPARATE system from scheduler.py. scheduler.py is the agent's own
local reminder store (it nudges you in this app); this is the real Google
calendar other people can see and be invited to. Use both: a real event for
other humans, a local reminder for the agent's own follow-ups.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

import config

CREDENTIALS_PATH = os.path.join(config.DATA_DIR, "google_credentials.json")
TOKEN_PATH = os.path.join(config.DATA_DIR, "google_token.json")

# calendar.events is enough to create/read/delete events and request a Meet
# link. We deliberately do NOT ask for Gmail or Drive scopes.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

API_ROOT = "https://www.googleapis.com/calendar/v3"

SETUP_HINT = (
    "Google Calendar is not connected. One-time setup: (1) enable the Google "
    "Calendar API in Google Cloud Console, (2) create an OAuth client ID of "
    "type 'Desktop app', (3) save the downloaded JSON as "
    "agent/data/google_credentials.json, (4) run "
    "'.venv/Scripts/python.exe calendar_tools.py --login' once. This needs a "
    "human at a browser -- it cannot be done by the agent."
)


# ---------------------------------------------------------------------------
#  auth
# ---------------------------------------------------------------------------
def is_configured() -> bool:
    """True when a cached token exists, i.e. the one-time login has happened."""
    return os.path.exists(TOKEN_PATH)


def _load_credentials():
    """Return valid Credentials, refreshing if needed. None when not set up."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    if not os.path.exists(TOKEN_PATH):
        return None
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds)
    return creds if creds and creds.valid else None


def _save_token(creds) -> None:
    os.makedirs(os.path.dirname(TOKEN_PATH) or ".", exist_ok=True)
    with open(TOKEN_PATH, "w", encoding="utf-8") as fh:
        fh.write(creds.to_json())
    try:  # best-effort: keep the token readable only by this user
        os.chmod(TOKEN_PATH, 0o600)
    except OSError:
        pass


def login() -> str:
    """Run the one-time interactive consent flow. Needs a human at a browser."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not os.path.exists(CREDENTIALS_PATH):
        return (f"Missing {CREDENTIALS_PATH}.\n\n{SETUP_HINT}")
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    _save_token(creds)
    return f"Connected. Token cached at {TOKEN_PATH} — you will not be asked again."


def _session():
    """An authorised requests session, or None when not connected."""
    creds = _load_credentials()
    if creds is None:
        return None
    from google.auth.transport.requests import AuthorizedSession
    return AuthorizedSession(creds)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+")


def _attendee_list(attendees) -> list[dict]:
    """Accept a list, or a comma/semicolon-separated string, of addresses.

    Anything that is not an address is dropped rather than sent -- Google
    rejects the whole request for one malformed attendee, and losing a typo is
    better than losing the meeting.
    """
    if not attendees:
        return []
    if isinstance(attendees, str):
        found = _EMAIL_RE.findall(attendees)
    else:
        found = []
        for item in attendees:
            found.extend(_EMAIL_RE.findall(str(item)))
    seen, out = set(), []
    for addr in found:
        low = addr.lower()
        if low not in seen:
            seen.add(low)
            out.append({"email": addr})
    return out


def _resolve_when(when: str) -> tuple[str, str]:
    """Turn 'tomorrow 5 PM' / an ISO string into (iso, error).

    Reuses scheduler.parse_when so the phrasing the agent already understands
    for local reminders means the same thing here.
    """
    if not (when or "").strip():
        return "", "missing 'when' (e.g. 'tomorrow 5 PM' or '2026-09-19T17:00')"
    try:
        import scheduler
        iso = scheduler.parse_when(when)
        if iso:
            return iso, ""
    except Exception:
        pass
    try:
        return datetime.fromisoformat(when.strip()).isoformat(), ""
    except ValueError:
        return "", (f"could not understand the time {when!r}. Use something like "
                    "'tomorrow 5 PM', 'Friday 14:30' or '2026-09-19T17:00'.")


def _with_zone(dt: datetime) -> datetime:
    """Attach this machine's real UTC offset to a naive datetime.

    scheduler.parse_when returns naive local times ('2026-09-19T17:00:00').
    Sending that to Google without an offset means "17:00 in whatever zone the
    calendar defaults to", which is how a 5 PM meeting quietly becomes 5 PM
    somewhere else.
    """
    return dt if dt.tzinfo is not None else dt.astimezone()


def _time_field(dt: datetime) -> dict:
    """Google's {dateTime, timeZone?} pair.

    An explicit IANA name is better for recurring events (it survives DST), but
    guessing one on Windows is unreliable -- so we only send `timeZone` when the
    user set config.TIMEZONE, and otherwise rely on the offset already carried
    by the timestamp, which Google accepts and which is unambiguous.
    """
    field = {"dateTime": dt.isoformat()}
    if config.TIMEZONE:
        field["timeZone"] = config.TIMEZONE
    return field


def _event_view(ev: dict) -> dict:
    """The parts of a Google event worth handing back to a model/user."""
    start = ev.get("start", {})
    return {
        "event_id": ev.get("id", ""),
        "title": ev.get("summary", ""),
        "start": start.get("dateTime") or start.get("date", ""),
        "end": (ev.get("end", {}).get("dateTime")
                or ev.get("end", {}).get("date", "")),
        "meet_link": ev.get("hangoutLink", ""),
        "html_link": ev.get("htmlLink", ""),
        "attendees": [a.get("email", "") for a in ev.get("attendees", [])],
        "description": ev.get("description", ""),
    }


def _error(resp) -> str:
    """A useful message out of a Google error response."""
    try:
        body = resp.json()
        err = body.get("error", {})
        msg = err.get("message") or str(body)
        return f"Google Calendar error {resp.status_code}: {msg}"
    except Exception:
        return f"Google Calendar error {resp.status_code}: {resp.text[:200]}"


# ---------------------------------------------------------------------------
#  operations (plain dicts in/out; agent_tools wraps these as tools)
# ---------------------------------------------------------------------------
def create_meeting(title: str, when: str, attendees=None,
                   duration_minutes: int = 30,
                   description: str = "",
                   with_meet: bool = True) -> dict:
    """Create a real calendar event, with a Google Meet link by default.

    Returns {ok, event_id, meet_link, start, ...} or {ok: False, error}.
    """
    sess = _session()
    if sess is None:
        return {"ok": False, "error": SETUP_HINT}

    start_iso, err = _resolve_when(when)
    if err:
        return {"ok": False, "error": err}
    try:
        minutes = max(5, min(int(duration_minutes or 30), 24 * 60))
    except (TypeError, ValueError):
        minutes = 30

    start_dt = _with_zone(datetime.fromisoformat(start_iso))
    end_dt = start_dt + timedelta(minutes=minutes)

    body: dict = {
        "summary": title or "Meeting",
        "description": description or "",
        "start": _time_field(start_dt),
        "end": _time_field(end_dt),
    }
    guests = _attendee_list(attendees)
    if guests:
        body["attendees"] = guests
    if with_meet:
        # requestId must be unique per creation attempt; Google mints the Meet
        # link server-side and returns it as hangoutLink.
        body["conferenceData"] = {
            "createRequest": {
                "requestId": uuid.uuid4().hex,
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        }

    params = {
        "conferenceDataVersion": 1 if with_meet else 0,
        # Actually email the guests -- an invite nobody receives is not an invite.
        "sendUpdates": "all" if guests else "none",
    }
    try:
        resp = sess.post(f"{API_ROOT}/calendars/primary/events",
                         params=params, json=body, timeout=30)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if resp.status_code not in (200, 201):
        return {"ok": False, "error": _error(resp)}

    ev = resp.json()
    view = _event_view(ev)

    if with_meet and not view["meet_link"]:
        # The event exists but conferencing failed -- say so plainly rather
        # than returning a link-less "success" the caller will misread.
        view["warning"] = ("the event was created but Google did not return a "
                           "Meet link; check that Meet is enabled for this "
                           "Google account")
    try:
        import store
        store.audit("calendar_event_created", event_id=view["event_id"],
                    title=view["title"], start=view["start"],
                    meet_link=view["meet_link"],
                    attendees=view["attendees"])
    except Exception:
        pass
    return {"ok": True, **view}


def list_meetings(days: int = 7, max_results: int = 20) -> dict:
    """Upcoming events on the primary calendar, soonest first."""
    sess = _session()
    if sess is None:
        return {"ok": False, "error": SETUP_HINT}
    try:
        days = max(1, min(int(days or 7), 365))
    except (TypeError, ValueError):
        days = 7

    now = datetime.now().astimezone()
    params = {
        "timeMin": now.isoformat(),
        "timeMax": (now + timedelta(days=days)).isoformat(),
        "singleEvents": "true",     # expand recurrences into real occurrences
        "orderBy": "startTime",
        "maxResults": max(1, min(int(max_results or 20), 100)),
    }
    try:
        resp = sess.get(f"{API_ROOT}/calendars/primary/events",
                        params=params, timeout=30)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if resp.status_code != 200:
        return {"ok": False, "error": _error(resp)}
    items = resp.json().get("items", [])
    return {"ok": True, "count": len(items),
            "events": [_event_view(e) for e in items]}


def cancel_meeting(event_id: str, notify: bool = True) -> dict:
    """Delete an event, notifying the guests by default."""
    sess = _session()
    if sess is None:
        return {"ok": False, "error": SETUP_HINT}
    if not (event_id or "").strip():
        return {"ok": False, "error": "missing event_id"}
    try:
        resp = sess.delete(
            f"{API_ROOT}/calendars/primary/events/{event_id.strip()}",
            params={"sendUpdates": "all" if notify else "none"}, timeout=30)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    # 410 = already gone, which is the state the caller wanted.
    if resp.status_code not in (200, 204, 410):
        return {"ok": False, "error": _error(resp)}
    try:
        import store
        store.audit("calendar_event_cancelled", event_id=event_id)
    except Exception:
        pass
    return {"ok": True, "event_id": event_id, "cancelled": True}


if __name__ == "__main__":
    import sys
    if "--login" in sys.argv:
        print(login())
    elif "--list" in sys.argv:
        print(json.dumps(list_meetings(), indent=2))
    else:
        print(f"connected: {is_configured()}")
        print(f"credentials: {CREDENTIALS_PATH}")
        print(f"token:       {TOKEN_PATH}")
        print()
        print(SETUP_HINT)
