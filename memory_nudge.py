"""Proactive nudges — Paimon raising things before you ask.

The JARVIS-ish "alive" layer of the memory system. A periodic scan looks for
conditions worth a heads-up and pushes them, as toasts, through the EXISTING
notification channel: each nudge is created as a scheduler event of
``kind="nudge"`` due *now*, so the next scheduler tick fires it and the
frontend's existing 15-second poll of ``/api/schedule/notifications`` pops it —
no frontend changes, and nudges are kept off the calendar (see scheduler.py).

Conditions detected (all cheap, deterministic reads of the task board):
  * a follow-up has been silent for N+ days → "still waiting on <person>",
  * a task is past its due date and not done → "<task> is overdue",
  * (file-ingested nudges are emitted by the ingest loop, not here).

Dedup + cooldown: a small ``memory/nudges.json`` records when each nudge key was
last emitted, so the same condition doesn't toast every scan — only once per
cooldown window (default 12h) until it's resolved and re-arises.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

NUDGE_STATE_FILE = os.path.join(os.path.dirname(__file__), "memory", "nudges.json")

FOLLOWUP_SILENT_DAYS = int(os.environ.get("NUDGE_FOLLOWUP_SILENT_DAYS", "3"))
COOLDOWN_HOURS = int(os.environ.get("NUDGE_COOLDOWN_HOURS", "12"))
MAX_PER_SCAN = int(os.environ.get("NUDGE_MAX_PER_SCAN", "3"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(iso: str | None):
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (ValueError, TypeError):
        return None


def _safe(fn):
    try:
        return fn()
    except Exception:
        return []


# ===========================================================================
#  detection (pure — inject data in tests, else read the task board)
# ===========================================================================
def detect_nudges(now: datetime | None = None, follow_ups=None, tasks=None,
                  silent_days: int | None = None) -> list[dict]:
    """Return the nudges currently warranted: ``[{key, message}]``. ``key`` is a
    stable id used for cooldown dedup. Reads the live task board unless data is
    injected (tests)."""
    now = now or _now()
    silent_days = FOLLOWUP_SILENT_DAYS if silent_days is None else silent_days
    if follow_ups is None:
        import store
        follow_ups = _safe(store.pending_follow_ups)
    if tasks is None:
        import store
        tasks = _safe(store.all_tasks)

    out: list[dict] = []

    # 1) follow-ups gone quiet
    for f in follow_ups or []:
        created = _parse(f.get("created_at"))
        if not created:
            continue
        days = (now - created).days
        if days >= silent_days:
            person = f.get("person", "someone")
            task = f.get("task_title", "a task")
            attempt = int(f.get("attempt", 1))
            chased = f" (chased {attempt}x)" if attempt > 1 else ""
            out.append({
                "key": f"fu:{f.get('task_id','?')}:{person.lower()}",
                "message": (f"Still waiting on {person}{chased} about "
                            f"'{task}' — it's been {days} days. Chase again?"),
            })

    # 2) tasks past due and not done
    today = now.strftime("%Y-%m-%d")
    for t in tasks or []:
        if t.get("status") == "done":
            continue
        due = (t.get("due_date") or "")[:10]
        if due and due < today:
            out.append({
                "key": f"overdue:{t.get('id','?')}",
                "message": f"'{t.get('title','A task')}' was due {due} "
                           "and isn't done yet.",
            })

    return out


# ===========================================================================
#  cooldown state
# ===========================================================================
def _load_state() -> dict:
    if not os.path.exists(NUDGE_STATE_FILE):
        return {}
    try:
        with open(NUDGE_STATE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(NUDGE_STATE_FILE), exist_ok=True)
    tmp = NUDGE_STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, NUDGE_STATE_FILE)


def _in_cooldown(last_iso: str | None, now: datetime) -> bool:
    last = _parse(last_iso)
    if not last:
        return False
    return (now - last).total_seconds() < COOLDOWN_HOURS * 3600


# ===========================================================================
#  emission (rides the scheduler's existing toast channel)
# ===========================================================================
def emit_nudges(detected: list[dict], now: datetime | None = None,
                max_per_scan: int = MAX_PER_SCAN) -> list[dict]:
    """Emit any detected nudge not currently in cooldown, as a scheduler 'nudge'
    event due now (so it toasts). Records emission time for dedup. Returns the
    nudges actually emitted."""
    now = now or _now()
    state = _load_state()
    import scheduler
    emitted: list[dict] = []
    for nudge in detected:
        if len(emitted) >= max_per_scan:
            break
        key = nudge["key"]
        if _in_cooldown(state.get(key), now):
            continue
        try:
            scheduler.create_event(message=nudge["message"],
                                   when=scheduler.now_iso(), kind="nudge")
        except Exception as exc:
            print(f"[nudge] could not emit {key}: {exc}")
            continue
        state[key] = now.isoformat()
        emitted.append(nudge)
    if emitted:
        _save_state(state)
        print(f"[nudge] raised {len(emitted)} proactive nudge(s)")
    return emitted


def emit_file_nudge(report: dict, now: datetime | None = None) -> dict | None:
    """Emit a single 'I read these files into memory' nudge after an ingest scan
    that actually folded in new files. Deduped on the set of filenames so the
    same batch doesn't re-toast."""
    files = [r.get("file") for r in (report or {}).get("files", []) if r.get("file")]
    if not files:
        return None
    now = now or _now()
    key = "ingest:" + ",".join(sorted(files))
    state = _load_state()
    if _in_cooldown(state.get(key), now):
        return None
    shown = ", ".join(files[:3]) + (f" +{len(files)-3} more" if len(files) > 3 else "")
    msg = f"I read {len(files)} new file(s) into memory: {shown}."
    try:
        import scheduler
        scheduler.create_event(message=msg, when=scheduler.now_iso(), kind="nudge")
    except Exception as exc:
        print(f"[nudge] could not emit ingest nudge: {exc}")
        return None
    state[key] = now.isoformat()
    _save_state(state)
    return {"key": key, "message": msg}


def run_nudge_scan() -> list[dict]:
    """Detect + emit in one call. Used by the server's background loop."""
    return emit_nudges(detect_nudges())
