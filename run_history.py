"""Run history: every LLM call and every tool call, grouped by conversation.

Two ledgers already record what happened, from opposite ends:

  * ``data/actions.jsonl``  -- one row per TOOL call, with the session id, the
    agent that called it, its arguments and its result.
  * ``data/usage.jsonl``    -- one row per LLM call, with the agent, model,
    tokens and cost.

Neither is readable on its own. This module joins them into one timeline per
session so the viewer at /runs can draw it.

Usage rows written before ``usage_log`` started stamping a session are matched
by timestamp instead: a call that happened inside a session's window belongs to
that session. That is an inference, not a fact, so those rows are flagged.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import config

ACTIONS_PATH = os.path.join(config.DATA_DIR, "actions.jsonl")
USAGE_PATH = os.path.join(config.DATA_DIR, "usage.jsonl")

# A session is a conversation; anything this far past its last recorded action
# is a new one rather than a straggler. Only used to close the window we match
# unstamped usage rows against.
_TAIL_SECONDS = 120


def _read(path: str) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue          # a torn last line is not worth a 500
    except FileNotFoundError:
        pass
    return out


def _sid(row: dict) -> str:
    return (row.get("session") or row.get("session_id") or "").strip()


def _ts(row: dict) -> str:
    return (row.get("ts") or "").strip()


def _shift(iso: str, seconds: int) -> str:
    """ISO timestamp moved by N seconds, or the input when it cannot parse."""
    try:
        from datetime import datetime, timedelta
        return (datetime.fromisoformat(iso) + timedelta(seconds=seconds)).isoformat()
    except Exception:
        return iso


def _windows(actions: list[dict]) -> list[tuple]:
    """(session, first_ts, last_ts) per session, in first-seen order."""
    span: dict[str, list] = {}
    order: list[str] = []
    for a in actions:
        s, t = _sid(a), _ts(a)
        if not s or not t:
            continue
        if s not in span:
            span[s] = [t, t]
            order.append(s)
        else:
            if t < span[s][0]:
                span[s][0] = t
            if t > span[s][1]:
                span[s][1] = t
    return [(s, span[s][0], span[s][1]) for s in order]


def _attach_usage(usage: list[dict], windows: list[tuple]) -> None:
    """Give every usage row a session, in place.

    Rows stamped by usage_log keep theirs. Older rows are matched to the
    session whose window contains them and marked ``inferred``.
    """
    for u in usage:
        if _sid(u):
            continue
        t = _ts(u)
        if not t:
            continue
        for s, first, last in windows:
            if first <= t <= _shift(last, _TAIL_SECONDS):
                u["session"] = s
                u["inferred"] = True
                break


# The two ledgers name the same agent differently: usage_log takes its label
# from the model's session id ("browser", "task-mgr") while action_log uses the
# roster name ("browser_agent", "task_manager"). Un-normalised, one agent shows
# up as two nodes and every session looks like it had eight participants.
_AGENT_ALIASES = {
    "browser": "browser_agent",
    "comms": "comms_agent",
    "task-mgr": "task_manager",
    "task_mgr": "task_manager",
    "scripter": "scripter",
    "orchestrator": "orchestrator",
}


def canonical_agent(name: str) -> str:
    n = (name or "?").strip()
    return _AGENT_ALIASES.get(n.lower(), n)


def _tool_event(a: dict) -> dict:
    """One tool call, trimmed for the wire."""
    return {
        "kind": "tool",
        "ts": _ts(a),
        "agent": canonical_agent(a.get("agent")),
        "name": a.get("tool") or "?",
        "ok": bool(a.get("ok", True)),
        "args": a.get("args_text") or "",
        "result": a.get("result") or "",
        "task_id": a.get("task_id") or "",
        # Points at the untruncated arguments and result in call_payloads.jsonl.
        "call_id": a.get("call_id") or "",
    }


def _llm_event(u: dict) -> dict:
    """One model call, trimmed for the wire."""
    inp = int(u.get("input_tokens") or 0)
    cached = int(u.get("cache_input_tokens") or 0)
    return {
        "kind": "llm",
        "ts": _ts(u),
        "agent": canonical_agent(u.get("agent")),
        "name": u.get("model") or "?",
        "ok": True,
        "profile": u.get("profile") or "",
        "input_tokens": inp,
        "cache_input_tokens": cached,
        "output_tokens": int(u.get("output_tokens") or 0),
        "latency_s": float(u.get("latency_s") or 0.0),
        "est_usd": float(u.get("est_usd") or 0.0),
        "tools_offered": int(u.get("tools_offered") or 0),
        "inferred": bool(u.get("inferred")),
        # Points at the full request/response in call_payloads.jsonl.
        "call_id": u.get("call_id") or "",
    }


# Tool calls that hand work to another agent. The viewer draws these as the
# edges of the graph; everything else is work done inside one node.
DELEGATION_TOOLS = ("delegate_task",)


def _delegations(actions: list[dict]) -> list[dict]:
    """Orchestrator -> child hand-offs, parsed out of delegate_task rows."""
    out = []
    for a in actions:
        if a.get("tool") not in DELEGATION_TOOLS:
            continue
        args = a.get("args_text") or ""
        child = ""
        for part in args.split(","):
            part = part.strip()
            if part.startswith("child="):
                child = part[6:].strip()
                break
        task_id = ""
        res = a.get("result") or ""
        if '"task_id"' in res:
            try:
                task_id = json.loads(res).get("task_id", "")
            except Exception:
                task_id = ""
        out.append({
            "ts": _ts(a),
            "from": canonical_agent(a.get("agent") or "orchestrator"),
            "to": child or "?",
            "task_id": task_id,
            "description": args,
            "ok": bool(a.get("ok", True)),
            "refused": res.startswith("ERROR:"),
        })
    return out


def _summarise(events: list[dict]) -> dict:
    llm = [e for e in events if e["kind"] == "llm"]
    tools = [e for e in events if e["kind"] == "tool"]
    return {
        "llm_calls": len(llm),
        "tool_calls": len(tools),
        "failed_tools": sum(1 for e in tools if not e["ok"]),
        "input_tokens": sum(e["input_tokens"] for e in llm),
        "cache_input_tokens": sum(e["cache_input_tokens"] for e in llm),
        "output_tokens": sum(e["output_tokens"] for e in llm),
        "est_usd": round(sum(e["est_usd"] for e in llm), 6),
        "latency_s": round(sum(e["latency_s"] for e in llm), 2),
    }


def sessions(limit: int = 50) -> list[dict]:
    """Newest-first session summaries, without their event lists."""
    actions = _read(ACTIONS_PATH)
    usage = _read(USAGE_PATH)
    wins = _windows(actions)
    _attach_usage(usage, wins)

    by_session: dict[str, list] = {}
    for a in actions:
        s = _sid(a)
        if s:
            by_session.setdefault(s, []).append(_tool_event(a))
    for u in usage:
        s = _sid(u)
        if s:
            by_session.setdefault(s, []).append(_llm_event(u))

    out = []
    for s, first, last in wins:
        evs = sorted(by_session.get(s, []), key=lambda e: e["ts"])
        agents = []
        for e in evs:
            if e["agent"] not in agents:
                agents.append(e["agent"])
        row = {
            "session": s,
            "started": first,
            "ended": max([last] + [e["ts"] for e in evs]) if evs else last,
            "agents": agents,
        }
        row.update(_summarise(evs))
        out.append(row)
    out.reverse()
    return out[:max(1, int(limit or 50))]


def detail(session: str) -> Optional[dict]:
    """One session: its events in order, plus the delegation edges."""
    if not session:
        return None
    actions = [a for a in _read(ACTIONS_PATH)]
    wins = _windows(actions)
    usage = _read(USAGE_PATH)
    _attach_usage(usage, wins)

    mine = [a for a in actions if _sid(a) == session]
    if not mine:
        return None
    events = [_tool_event(a) for a in mine]
    events += [_llm_event(u) for u in usage if _sid(u) == session]
    events.sort(key=lambda e: e["ts"])
    for i, e in enumerate(events):
        e["seq"] = i + 1

    agents = []
    for e in events:
        if e["agent"] not in agents:
            agents.append(e["agent"])

    out = {
        "session": session,
        "started": events[0]["ts"] if events else "",
        "ended": events[-1]["ts"] if events else "",
        "agents": agents,
        "events": events,
        "delegations": _delegations(mine),
    }
    out.update(_summarise(events))
    return out


def payload(call_id: str) -> Optional[dict]:
    """The full request/response for one call, or None when it was not kept.

    Payloads predate nothing and outlive nothing: a call logged before
    call_log existed, or while CALL_LOG=0, simply has no row here. The viewer
    says so rather than rendering an empty panel.
    """
    try:
        import call_log
        return call_log.read(call_id)
    except Exception:
        return None
