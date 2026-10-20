"""JSON-backed data store for the task manager.

Single source of truth used by BOTH the REST API (server.py) and the AI
agent's tools (agent_tools.py), so the agent and the UI never disagree.

Design (PageIndex-inspired, https://github.com/VectifyAI/PageIndex):
    Instead of holding everything in one giant blob, we keep a lightweight
    hierarchical *tree* (groups -> subgroups -> task stubs) in ``tree.json``
    that is cheap to load and reason over, and store each task's full body in
    its own ``tasks/<id>.json`` file that is only read on demand. The agent
    (or a client) navigates the tree first, then drills into the leaves it
    actually needs -- the same "navigate a table-of-contents, fetch the leaf"
    idea PageIndex uses for large documents.

Hierarchy:
    Group            top-level area/project (e.g. "Vendor Onboarding")
      Subgroup       optional nesting under a group (e.g. "Phase 1")
        Task         an individual to-do, with reminders & follow-ups

Nothing here touches a database; it's all flat JSON on disk.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

# --- paths -----------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TREE_FILE = os.path.join(DATA_DIR, "tree.json")
TASKS_DIR = os.path.join(DATA_DIR, "tasks")
AUDIT_FILE = os.path.join(DATA_DIR, "audit.jsonl")

# A single process-wide lock keeps concurrent writes (API + agent) consistent.
_LOCK = threading.RLock()

VALID_STATUS = ("todo", "in_progress", "blocked", "done")
VALID_PRIORITY = ("low", "medium", "high", "urgent")


# --- helpers ---------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(action: str, **details) -> None:
    """Append an immutable event to data/audit.jsonl (one JSON object per
    line). Nothing here ever deletes or rewrites past lines, so this is a full
    history of every mutation, including task completions and deletions."""
    _ensure_dirs()
    entry = {"ts": _now(), "action": action, **details}
    with open(AUDIT_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def audit(action: str, **details) -> None:
    """Public entry point for modules outside the board that need to record an
    event -- e.g. mailer.py logging that mail actually left the machine. Same
    append-only file as every task mutation, so one timeline covers everything
    the agent did, not just what it did to the board."""
    _audit(action, **details)


def read_audit(limit: int = 200, task_id: Optional[str] = None) -> list[dict]:
    """Return the most recent audit events (optionally filtered to one task)."""
    if not os.path.exists(AUDIT_FILE):
        return []
    out = []
    with open(AUDIT_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if task_id and e.get("task_id") != task_id:
                continue
            out.append(e)
    return out[-limit:][::-1]  # newest first


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _ensure_dirs() -> None:
    os.makedirs(TASKS_DIR, exist_ok=True)


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        txt = fh.read().strip()
    return json.loads(txt) if txt else default


def _write_json(path: str, obj: Any) -> None:
    _ensure_dirs()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic on the same filesystem


# --- tree (groups + subgroups + task stubs) --------------------------------
def _empty_tree() -> dict:
    return {"groups": [], "updated_at": _now()}


def _load_tree() -> dict:
    return _read_json(TREE_FILE, _empty_tree())


def _save_tree(tree: dict) -> None:
    tree["updated_at"] = _now()
    _write_json(TREE_FILE, tree)


def _find_group(tree: dict, group_id: str) -> Optional[dict]:
    for g in tree["groups"]:
        if g["id"] == group_id:
            return g
    return None


def _find_subgroup(group: dict, subgroup_id: str) -> Optional[dict]:
    for s in group.get("subgroups", []):
        if s["id"] == subgroup_id:
            return s
    return None


def _locate_task_stub(tree: dict, task_id: str):
    """Return (group, subgroup_or_None, stub) for a task id, or (None,None,None)."""
    for g in tree["groups"]:
        for stub in g.get("tasks", []):
            if stub["id"] == task_id:
                return g, None, stub
        for s in g.get("subgroups", []):
            for stub in s.get("tasks", []):
                if stub["id"] == task_id:
                    return g, s, stub
    return None, None, None


def _stub_from_task(task: dict) -> dict:
    """The lightweight view kept in the tree for cheap navigation."""
    return {
        "id": task["id"],
        "title": task["title"],
        "status": task["status"],
        "priority": task["priority"],
        "due_date": task.get("due_date"),
        "open_followups": sum(
            1 for f in task.get("follow_ups", []) if not f.get("done")
        ),
        "open_reminders": sum(
            1 for r in task.get("reminders", []) if not r.get("done")
        ),
    }


def _refresh_stub(tree: dict, task: dict) -> None:
    g, s, stub = _locate_task_stub(tree, task["id"])
    if stub is None:
        return
    stub.update(_stub_from_task(task))


# --- task bodies -----------------------------------------------------------
def _task_path(task_id: str) -> str:
    return os.path.join(TASKS_DIR, f"{task_id}.json")


def _load_task(task_id: str) -> Optional[dict]:
    return _read_json(_task_path(task_id), None)


def _save_task(task: dict) -> None:
    task["updated_at"] = _now()
    _write_json(_task_path(task["id"]), task)


# ===========================================================================
#  PUBLIC API
# ===========================================================================

# --- groups ----------------------------------------------------------------
def create_group(name: str, description: str = "") -> dict:
    with _LOCK:
        tree = _load_tree()
        group = {
            "id": _new_id("grp"),
            "name": name,
            "description": description,
            "created_at": _now(),
            "subgroups": [],
            "tasks": [],
        }
        tree["groups"].append(group)
        _save_tree(tree)
        return group


def create_subgroup(group_id: str, name: str, description: str = "") -> dict:
    with _LOCK:
        tree = _load_tree()
        group = _find_group(tree, group_id)
        if group is None:
            raise KeyError(f"group not found: {group_id}")
        sub = {
            "id": _new_id("sub"),
            "name": name,
            "description": description,
            "created_at": _now(),
            "tasks": [],
        }
        group.setdefault("subgroups", []).append(sub)
        _save_tree(tree)
        return sub


def rename_group(group_id: str, name: str) -> dict:
    with _LOCK:
        tree = _load_tree()
        group = _find_group(tree, group_id)
        if group is None:
            raise KeyError(f"group not found: {group_id}")
        group["name"] = name
        _save_tree(tree)
        return group


def delete_group(group_id: str) -> None:
    """Delete a group and every task (and subgroup) under it."""
    with _LOCK:
        tree = _load_tree()
        group = _find_group(tree, group_id)
        if group is None:
            return
        for stub in group.get("tasks", []):
            _delete_task_file(stub["id"])
        for sub in group.get("subgroups", []):
            for stub in sub.get("tasks", []):
                _delete_task_file(stub["id"])
        tree["groups"] = [g for g in tree["groups"] if g["id"] != group_id]
        _save_tree(tree)


def delete_subgroup(group_id: str, subgroup_id: str) -> None:
    with _LOCK:
        tree = _load_tree()
        group = _find_group(tree, group_id)
        if group is None:
            return
        sub = _find_subgroup(group, subgroup_id)
        if sub is None:
            return
        for stub in sub.get("tasks", []):
            _delete_task_file(stub["id"])
        group["subgroups"] = [
            s for s in group["subgroups"] if s["id"] != subgroup_id
        ]
        _save_tree(tree)


def _delete_task_file(task_id: str) -> None:
    path = _task_path(task_id)
    if os.path.exists(path):
        os.remove(path)


# --- tasks -----------------------------------------------------------------
def create_task(
    title: str,
    group_id: str,
    subgroup_id: Optional[str] = None,
    description: str = "",
    status: str = "todo",
    priority: str = "medium",
    due_date: Optional[str] = None,
    person: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    if status not in VALID_STATUS:
        raise ValueError(f"status must be one of {VALID_STATUS}")
    if priority not in VALID_PRIORITY:
        raise ValueError(f"priority must be one of {VALID_PRIORITY}")
    with _LOCK:
        tree = _load_tree()
        group = _find_group(tree, group_id)
        if group is None:
            raise KeyError(f"group not found: {group_id}")
        container = group
        if subgroup_id:
            sub = _find_subgroup(group, subgroup_id)
            if sub is None:
                raise KeyError(f"subgroup not found: {subgroup_id}")
            container = sub
        task = {
            "id": _new_id("tsk"),
            "title": title,
            "description": description,
            "status": status,
            "priority": priority,
            "due_date": due_date,
            "person": person,
            "tags": tags or [],
            "group_id": group_id,
            "subgroup_id": subgroup_id,
            "reminders": [],
            "follow_ups": [],
            "notes": [],
            "completed_at": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        _save_task(task)
        container.setdefault("tasks", []).append(_stub_from_task(task))
        _save_tree(tree)
        _audit("task_created", task_id=task["id"], title=title,
               group_id=group_id, subgroup_id=subgroup_id)
        return task


def get_task(task_id: str) -> Optional[dict]:
    with _LOCK:
        return _load_task(task_id)


def update_task(task_id: str, **fields) -> dict:
    """Patch top-level task fields (title, description, status, priority,
    due_date, person, tags)."""
    allowed = {
        "title", "description", "status", "priority",
        "due_date", "person", "tags",
    }
    with _LOCK:
        task = _load_task(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")
        old_status = task.get("status")
        changed = {}
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k == "status" and v not in VALID_STATUS:
                raise ValueError(f"status must be one of {VALID_STATUS}")
            if k == "priority" and v not in VALID_PRIORITY:
                raise ValueError(f"priority must be one of {VALID_PRIORITY}")
            if task.get(k) != v:
                changed[k] = v
            task[k] = v
        # Stamp / clear the completion time as status crosses the 'done' line.
        new_status = task.get("status")
        if new_status == "done" and old_status != "done":
            task["completed_at"] = _now()
        elif new_status != "done" and old_status == "done":
            task["completed_at"] = None
        _save_task(task)
        tree = _load_tree()
        _refresh_stub(tree, task)
        _save_tree(tree)
        if changed:
            _audit("task_updated", task_id=task_id, title=task["title"],
                   changes=changed)
        if new_status == "done" and old_status != "done":
            _audit("task_completed", task_id=task_id, title=task["title"])
        return task


def delete_task(task_id: str) -> None:
    with _LOCK:
        existing = _load_task(task_id)
        tree = _load_tree()
        g, s, stub = _locate_task_stub(tree, task_id)
        if stub is not None:
            container = s if s is not None else g
            container["tasks"] = [
                t for t in container["tasks"] if t["id"] != task_id
            ]
            _save_tree(tree)
        _delete_task_file(task_id)
        if existing is not None:
            _audit("task_deleted", task_id=task_id, title=existing.get("title"))


def move_task(
    task_id: str, group_id: str, subgroup_id: Optional[str] = None
) -> dict:
    """Move a task to a different group/subgroup."""
    with _LOCK:
        task = _load_task(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")
        tree = _load_tree()
        dest_group = _find_group(tree, group_id)
        if dest_group is None:
            raise KeyError(f"group not found: {group_id}")
        dest = dest_group
        if subgroup_id:
            sub = _find_subgroup(dest_group, subgroup_id)
            if sub is None:
                raise KeyError(f"subgroup not found: {subgroup_id}")
            dest = sub
        # remove old stub
        g, s, stub = _locate_task_stub(tree, task_id)
        if stub is not None:
            src = s if s is not None else g
            src["tasks"] = [t for t in src["tasks"] if t["id"] != task_id]
        # update body + add new stub
        task["group_id"] = group_id
        task["subgroup_id"] = subgroup_id
        _save_task(task)
        dest.setdefault("tasks", []).append(_stub_from_task(task))
        _save_tree(tree)
        return task


# --- notes -----------------------------------------------------------------
def add_note(task_id: str, text: str) -> dict:
    with _LOCK:
        task = _load_task(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")
        task.setdefault("notes", []).append(
            {"id": _new_id("note"), "text": text, "created_at": _now()}
        )
        _save_task(task)
        return task


# --- reminders -------------------------------------------------------------
def add_reminder(task_id: str, remind_at: str, message: str = "") -> dict:
    """remind_at: an ISO date/datetime string (e.g. '2026-06-05' or
    '2026-06-05T14:00')."""
    with _LOCK:
        task = _load_task(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")
        task.setdefault("reminders", []).append(
            {
                "id": _new_id("rem"),
                "remind_at": remind_at,
                "message": message or task["title"],
                "done": False,
                "created_at": _now(),
            }
        )
        _save_task(task)
        tree = _load_tree()
        _refresh_stub(tree, task)
        _save_tree(tree)
        return task


def complete_reminder(task_id: str, reminder_id: str) -> dict:
    with _LOCK:
        task = _load_task(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")
        for r in task.get("reminders", []):
            if r["id"] == reminder_id:
                r["done"] = True
                r["completed_at"] = _now()
        _save_task(task)
        tree = _load_tree()
        _refresh_stub(tree, task)
        _save_tree(tree)
        return task


# --- follow-ups ------------------------------------------------------------
def add_follow_up(
    task_id: str,
    person: str,
    note: str = "",
    due_date: Optional[str] = None,
) -> dict:
    """Log that a follow-up is needed with `person`. Each call bumps the
    follow-up count for that person on this task, so you can see how many
    times you've had to chase the same issue."""
    with _LOCK:
        task = _load_task(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")
        follow_ups = task.setdefault("follow_ups", [])
        prior = sum(
            1 for f in follow_ups
            if f.get("person", "").lower() == person.lower()
        )
        follow_ups.append(
            {
                "id": _new_id("fu"),
                "person": person,
                "note": note,
                "due_date": due_date,
                "attempt": prior + 1,   # 1st, 2nd, 3rd chase for this person
                "done": False,
                "created_at": _now(),
            }
        )
        _save_task(task)
        tree = _load_tree()
        _refresh_stub(tree, task)
        _save_tree(tree)
        _audit("follow_up_added", task_id=task_id, title=task["title"],
               person=person, attempt=prior + 1, note=note)
        return task


def complete_follow_up(task_id: str, follow_up_id: str) -> dict:
    with _LOCK:
        task = _load_task(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")
        for f in task.get("follow_ups", []):
            if f["id"] == follow_up_id:
                f["done"] = True
                f["completed_at"] = _now()
        _save_task(task)
        tree = _load_tree()
        _refresh_stub(tree, task)
        _save_tree(tree)
        return task


# --- read / query ----------------------------------------------------------
def get_tree() -> dict:
    """The cheap, navigable overview: groups -> subgroups -> task stubs."""
    with _LOCK:
        return _load_tree()


def all_tasks() -> list[dict]:
    with _LOCK:
        _ensure_dirs()
        tasks = []
        for fn in os.listdir(TASKS_DIR):
            if fn.endswith(".json") and not fn.endswith(".tmp"):
                t = _read_json(os.path.join(TASKS_DIR, fn), None)
                if t:
                    tasks.append(t)
        return tasks


def search_tasks(
    query: str = "",
    status: Optional[str] = None,
    person: Optional[str] = None,
    priority: Optional[str] = None,
) -> list[dict]:
    q = (query or "").lower()
    out = []
    for t in all_tasks():
        if q and q not in (t["title"] + " " + t.get("description", "")).lower():
            # also match notes / follow-up people
            blob = " ".join(
                [n.get("text", "") for n in t.get("notes", [])]
                + [f.get("person", "") for f in t.get("follow_ups", [])]
            ).lower()
            if q not in blob:
                continue
        if status and t["status"] != status:
            continue
        if priority and t["priority"] != priority:
            continue
        if person and not any(
            f.get("person", "").lower() == person.lower()
            for f in t.get("follow_ups", [])
        ) and (t.get("person") or "").lower() != person.lower():
            continue
        out.append(t)
    return out


def completed_tasks() -> list[dict]:
    """All tasks currently marked done, newest completion first. The data is
    never deleted on completion -- this just reads the 'done' ones back."""
    done = [t for t in all_tasks() if t.get("status") == "done"]
    return sorted(
        done,
        key=lambda t: t.get("completed_at") or t.get("updated_at") or "",
        reverse=True,
    )


def due_reminders(before: Optional[str] = None) -> list[dict]:
    """Open reminders with remind_at <= `before` (default: now). Returns a
    flat list of {task_id, task_title, ...reminder}."""
    cutoff = before or _now()
    out = []
    for t in all_tasks():
        for r in t.get("reminders", []):
            if not r.get("done") and r.get("remind_at", "") <= cutoff:
                out.append({"task_id": t["id"], "task_title": t["title"], **r})
    return sorted(out, key=lambda r: r.get("remind_at", ""))


def pending_follow_ups() -> list[dict]:
    """All open follow-ups across tasks, newest attempt first."""
    out = []
    for t in all_tasks():
        for f in t.get("follow_ups", []):
            if not f.get("done"):
                out.append({"task_id": t["id"], "task_title": t["title"], **f})
    return sorted(out, key=lambda f: (f.get("person", ""), -f.get("attempt", 0)))


def follow_up_counts() -> list[dict]:
    """How many times each person has been chased, per task -- answers
    'how many follow-ups have I taken for this issue'."""
    counts: dict[tuple, dict] = {}
    for t in all_tasks():
        for f in t.get("follow_ups", []):
            key = (t["id"], f.get("person", "?"))
            entry = counts.setdefault(
                key,
                {
                    "task_id": t["id"],
                    "task_title": t["title"],
                    "person": f.get("person", "?"),
                    "total": 0,
                    "open": 0,
                },
            )
            entry["total"] += 1
            if not f.get("done"):
                entry["open"] += 1
    return sorted(counts.values(), key=lambda e: -e["total"])


def stats() -> dict:
    tasks = all_tasks()
    by_status = {s: 0 for s in VALID_STATUS}
    for t in tasks:
        by_status[t["status"]] = by_status.get(t["status"], 0) + 1
    return {
        "total_tasks": len(tasks),
        "by_status": by_status,
        "open_reminders": len(due_reminders(before="9999")),
        "open_follow_ups": len(pending_follow_ups()),
    }
