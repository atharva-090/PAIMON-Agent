"""Full request/response payloads, kept out of the hot ledgers.

``usage.jsonl`` records what a model call COST and ``actions.jsonl`` records
that a tool was called, with its arguments clipped to 48 characters and its
result to 400. Neither keeps what was actually sent or returned, so the only
way to see the prompt a child received was to re-derive it by reading code.

This file keeps the whole thing: every message in the request, the text and
tool calls that came back, and for a tool the untruncated arguments and result.
It is written beside the ledgers and joined to them by ``call_id``.

It is deliberately a separate file. The ledgers are read on every board poll
and every trail injection; a megabyte of prompt per call has no business in
either. Nothing reads this one except the run viewer, on demand, for a single
call at a time.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone

import config

LOG_PATH = os.path.join(config.DATA_DIR, "call_payloads.jsonl")

# A single call's payload. A long browser run's prompt is ~40 KB; the cap is
# well clear of that and exists so one pathological call cannot write a
# gigabyte. Truncation is marked so the viewer never implies it has the whole
# thing when it does not.
MAX_PAYLOAD_CHARS = 400_000
MAX_FIELD_CHARS = 120_000

# Rotate rather than grow without bound. One generation is kept, so the
# current run and the one before it are always readable.
MAX_FILE_BYTES = int(os.environ.get("CALL_LOG_MAX_BYTES", str(256 * 1024 * 1024)))

_lock = threading.Lock()
_enabled = os.environ.get("CALL_LOG", "1").strip().lower() not in ("0", "false", "no")


def new_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def enabled() -> bool:
    return _enabled


def _clip(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    t = text or ""
    if len(t) <= limit:
        return t
    return t[:limit] + f"\n…[truncated, {len(t)} chars total]"


def _jsonable(obj, depth: int = 0):
    """Best-effort plain-data view of a message or response object.

    The model layer hands us whatever the provider SDK and AgentScope use --
    Msg objects, content blocks, pydantic models. Nothing here may raise: a
    payload that cannot be serialised is worth losing, a turn is not.
    """
    if depth > 6:
        return "…"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return _clip(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in obj]
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn(), depth + 1)
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: _jsonable(v, depth + 1)
                    for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:
            pass
    return _clip(str(obj))


def _rotate_if_needed() -> None:
    try:
        if os.path.getsize(LOG_PATH) < MAX_FILE_BYTES:
            return
        old = LOG_PATH + ".1"
        if os.path.exists(old):
            os.remove(old)
        os.replace(LOG_PATH, old)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _append(row: dict) -> None:
    if not _enabled:
        return
    try:
        line = json.dumps(row, ensure_ascii=False, default=str)
        if len(line) > MAX_PAYLOAD_CHARS:
            row = dict(row)
            row["oversize"] = len(line)
            for k in ("request", "response"):
                row[k] = _clip(json.dumps(row.get(k), ensure_ascii=False,
                                          default=str),
                               MAX_PAYLOAD_CHARS // 3)
            line = json.dumps(row, ensure_ascii=False, default=str)
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with _lock:
            _rotate_if_needed()
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass


def _session() -> str:
    try:
        import action_log
        return action_log.current_session()
    except Exception:
        return ""


def write_llm(call_id: str, agent: str, model: str, messages, response,
              tools=None, session: str = "") -> None:
    """One model call: everything sent, everything returned."""
    if not _enabled:
        return
    _append({
        "call_id": call_id,
        "kind": "llm",
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": session or _session(),
        "agent": agent or "?",
        "model": model or "?",
        "request": {
            "messages": _jsonable(messages),
            "tools": [t.get("function", {}).get("name")
                      if isinstance(t, dict) else str(t)
                      for t in (tools or [])],
        },
        "response": _jsonable(response),
    })


def write_tool(call_id: str, agent: str, tool: str, args, result: str,
               ok: bool = True, session: str = "") -> None:
    """One tool call, with the arguments and result the ledger had to clip."""
    if not _enabled:
        return
    _append({
        "call_id": call_id,
        "kind": "tool",
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": session or _session(),
        "agent": agent or "?",
        "tool": tool or "tool",
        "ok": bool(ok),
        "request": _jsonable(args),
        "response": _clip(result or ""),
    })


def read(call_id: str) -> dict | None:
    """One payload by id. Scans both generations, newest match wins."""
    if not call_id:
        return None
    found = None
    for path in (LOG_PATH, LOG_PATH + ".1"):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if call_id not in line:
                        continue        # cheap reject before parsing
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("call_id") == call_id:
                        found = row
        except FileNotFoundError:
            continue
        except Exception:
            continue
        if found:
            return found
    return found
