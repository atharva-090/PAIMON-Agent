"""A small scheduler — the "mini Google Calendar" behind the task manager.

It stores standalone timed *events* (reminders) in ``data/events.json`` and,
via a background loop in the web server, actually FIRES them when their time
arrives (unlike the per-task ``reminders`` in store.py, which are just data).

Key features:
  * Natural-language times: ``parse_when`` understands "in 20 mins", "in 2
    days", "tomorrow at 3pm", "next monday 09:30", bare "15:00", and ISO
    datetimes -- so both the UI quick-add box and the AI agent can schedule by
    just passing the user's phrase.
  * Recurrence: none / daily / weekly / monthly. When a recurring event fires,
    the next occurrence is scheduled automatically.
  * A notification queue: fired-but-unacknowledged events, which the frontend
    polls to pop toasts (and the browser/OS notification).

Everything is **local time** (the user's clock), which is what you want for
"remind me in 20 minutes". Times are stored as naive-local ISO strings.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime, timedelta
from typing import Optional

import store  # reuse atomic JSON IO + audit log + id generator

EVENTS_FILE = __import__("os").path.join(store.DATA_DIR, "events.json")
_LOCK = threading.RLock()

RECURRENCE = ("none", "daily", "weekly", "monthly")
_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}


# ---------------------------------------------------------------------------
#  local-time helpers
# ---------------------------------------------------------------------------
def now() -> datetime:
    return datetime.now()


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def now_iso() -> str:
    return _iso(now())


# ---------------------------------------------------------------------------
#  natural-language "when" parser
# ---------------------------------------------------------------------------
class WhenParseError(ValueError):
    pass


def _parse_time_of_day(text: str) -> Optional[tuple[int, int]]:
    """Find a clock time in `text`. Returns (hour, minute) or None."""
    t = text.strip()
    if "noon" in t:
        return (12, 0)
    if "midnight" in t:
        return (0, 0)
    if "tonight" in t:
        return (20, 0)
    # 3pm, 3:30 pm, 15:00, at 9
    m = re.search(r"\b(?:at\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1)) % 12
        if m.group(3) == "pm":
            h += 12
        return (h, int(m.group(2) or 0))
    m = re.search(r"\b(?:at\s*)?(\d{1,2}):(\d{2})\b", t)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"\bat\s+(\d{1,2})\b", t)
    if m:
        return (int(m.group(1)), 0)
    return None


def parse_when(text: str, base: Optional[datetime] = None) -> str:
    """Turn a phrase or ISO string into a concrete local-ISO datetime string.

    Raises WhenParseError if nothing recognisable is found.
    """
    if not text or not text.strip():
        raise WhenParseError("empty time")
    base = base or now()
    raw = text.strip()
    low = raw.lower()

    # 1) explicit ISO datetime, or date-only (default to 09:00 local)
    try:
        dt = datetime.fromisoformat(raw)
        if "t" not in low and ":" not in raw:  # date only
            dt = dt.replace(hour=9, minute=0)
        return _iso(dt)
    except ValueError:
        pass

    # 2) "in N <unit>" — precise offset from now
    m = re.search(
        r"\bin\s+(\d+)\s*(second|sec|minute|min|hour|hr|day|week|month)s?\b",
        low,
    )
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit.startswith("sec"):
            dt = base + timedelta(seconds=n)
        elif unit.startswith("min"):
            dt = base + timedelta(minutes=n)
        elif unit in ("hour", "hr"):
            dt = base + timedelta(hours=n)
        elif unit == "day":
            dt = base + timedelta(days=n)
        elif unit == "week":
            dt = base + timedelta(weeks=n)
        else:  # month
            dt = _add_months(base, n)
        return _iso(dt)

    # 3) a date anchor (today/tomorrow/weekday/next week/in N days) + time
    tod = _parse_time_of_day(low)
    day = None
    if "day after tomorrow" in low:
        day = (base + timedelta(days=2)).date()
    elif "tomorrow" in low:
        day = (base + timedelta(days=1)).date()
    elif "today" in low or "tonight" in low:
        day = base.date()
    elif "next week" in low:
        day = (base + timedelta(days=7)).date()
    else:
        wd = next((v for k, v in _WEEKDAYS.items()
                   if re.search(rf"\b{k}\b", low)), None)
        if wd is not None:
            ahead = (wd - base.weekday()) % 7
            if ahead == 0:
                ahead = 7  # "monday" means the next one
            day = (base + timedelta(days=ahead)).date()

    if day is not None:
        h, mi = tod if tod else (9, 0)
        return _iso(datetime(day.year, day.month, day.day, h, mi))

    # 4) time-only ("at 3pm", "15:30") -> today, or tomorrow if already passed
    if tod is not None:
        h, mi = tod
        cand = base.replace(hour=h, minute=mi, second=0, microsecond=0)
        if cand <= base:
            cand += timedelta(days=1)
        return _iso(cand)

    raise WhenParseError(f"couldn't understand time: {text!r}")


def _add_months(dt: datetime, n: int) -> datetime:
    month = dt.month - 1 + n
    year = dt.year + month // 12
    month = month % 12 + 1
    # clamp day to end of target month
    day = min(dt.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
                       else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return dt.replace(year=year, month=month, day=day)


def _next_occurrence(when_iso: str, recurrence: str) -> Optional[str]:
    if recurrence == "none" or recurrence not in RECURRENCE:
        return None
    dt = datetime.fromisoformat(when_iso)
    if recurrence == "daily":
        return _iso(dt + timedelta(days=1))
    if recurrence == "weekly":
        return _iso(dt + timedelta(weeks=1))
    if recurrence == "monthly":
        return _iso(_add_months(dt, 1))
    return None


# ---------------------------------------------------------------------------
#  storage
# ---------------------------------------------------------------------------
def _load() -> dict:
    return store._read_json(EVENTS_FILE, {"events": []})


def _save(data: dict) -> None:
    data["updated_at"] = now_iso()
    store._write_json(EVENTS_FILE, data)


def _public(e: dict) -> dict:
    return e


# ---------------------------------------------------------------------------
#  public API
# ---------------------------------------------------------------------------
def create_event(
    message: str,
    when: str,
    recurrence: str = "none",
    task_id: Optional[str] = None,
    kind: str = "reminder",
) -> dict:
    """Schedule a reminder. `when` may be natural language or an ISO datetime.

    `kind` marks what this event is: a normal 'reminder' (shows on the calendar
    AND pops a toast) or a 'nudge' (a proactive heads-up from the memory system —
    it still pops a toast via notifications(), but is kept OFF the calendar so it
    doesn't clutter it). See memory_nudge.py."""
    when_iso = parse_when(when)
    if recurrence not in RECURRENCE:
        recurrence = "none"
    with _LOCK:
        data = _load()
        event = {
            "id": store._new_id("evt"),
            "message": message.strip() or "Reminder",
            "when": when_iso,
            "recurrence": recurrence,
            "task_id": task_id,
            "kind": kind or "reminder",
            "fired": False,
            "fired_at": None,
            "acknowledged": False,
            "canceled": False,
            "created_at": now_iso(),
        }
        data["events"].append(event)
        _save(data)
        store._audit("schedule_created", event_id=event["id"],
                     message=event["message"], when=when_iso,
                     recurrence=recurrence, task_id=task_id, kind=event["kind"])
        return event


def get_event(event_id: str) -> Optional[dict]:
    with _LOCK:
        return next((e for e in _load()["events"] if e["id"] == event_id), None)


def cancel_event(event_id: str) -> bool:
    with _LOCK:
        data = _load()
        for e in data["events"]:
            if e["id"] == event_id and not e["canceled"]:
                e["canceled"] = True
                _save(data)
                store._audit("schedule_canceled", event_id=event_id,
                             message=e["message"])
                return True
        return False


def list_events(
    frm: Optional[str] = None,
    to: Optional[str] = None,
    include_canceled: bool = False,
    include_nudges: bool = False,
) -> list[dict]:
    """Events whose `when` falls in [frm, to] (ISO bounds). For the calendar.
    Proactive 'nudge' events are excluded by default so they don't clutter the
    calendar (they still surface as toasts via notifications())."""
    with _LOCK:
        out = []
        for e in _load()["events"]:
            if e["canceled"] and not include_canceled:
                continue
            if not include_nudges and e.get("kind") == "nudge":
                continue
            if frm and e["when"] < frm:
                continue
            if to and e["when"] > to:
                continue
            out.append(e)
        return sorted(out, key=lambda e: e["when"])


def upcoming(limit: int = 20) -> list[dict]:
    """Next not-yet-fired, not-canceled events."""
    with _LOCK:
        out = [
            e for e in _load()["events"]
            if not e["fired"] and not e["canceled"]
        ]
        return sorted(out, key=lambda e: e["when"])[:limit]


def fire_due(reference: Optional[str] = None) -> list[dict]:
    """Mark every due event as fired (and schedule recurrences). Returns the
    list of events that fired on this call. Called by the server loop."""
    ref = reference or now_iso()
    fired = []
    with _LOCK:
        data = _load()
        new_events = []
        for e in data["events"]:
            if (not e["fired"] and not e["canceled"] and e["when"] <= ref):
                e["fired"] = True
                e["fired_at"] = now_iso()
                e["acknowledged"] = False
                fired.append(e)
                store._audit("schedule_fired", event_id=e["id"],
                             message=e["message"], when=e["when"])
                nxt = _next_occurrence(e["when"], e["recurrence"])
                if nxt:
                    clone = dict(e)
                    clone.update(
                        id=store._new_id("evt"), when=nxt, fired=False,
                        fired_at=None, acknowledged=False,
                        created_at=now_iso(),
                    )
                    new_events.append(clone)
        if fired:
            data["events"].extend(new_events)
            _save(data)
    return fired


def notifications() -> list[dict]:
    """Fired but not-yet-acknowledged events (what the UI should pop)."""
    with _LOCK:
        return sorted(
            [e for e in _load()["events"]
             if e["fired"] and not e["acknowledged"] and not e["canceled"]],
            key=lambda e: e["fired_at"] or "",
        )


def acknowledge(ids: list[str]) -> int:
    with _LOCK:
        data = _load()
        n = 0
        idset = set(ids)
        for e in data["events"]:
            if e["id"] in idset and not e["acknowledged"]:
                e["acknowledged"] = True
                n += 1
        if n:
            _save(data)
        return n
