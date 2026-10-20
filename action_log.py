"""Deterministic per-action ledger — what the system DID, not what it said.

Why this is not model-authored
------------------------------
The obvious design is to ask the model for a one-line summary after each step.
This project has already paid for that mistake: on a real run the agent could
not click through the Google Meet UI, quietly wrote a *reminder* instead, and
reported success. The fix was making ``tools_used`` the ground truth and adding
``_claims_failed_without_calling`` to catch the model contradicting the ledger.

A model-authored action summary would re-open exactly that hole, and worse than
before: these lines feed ``learn_from_session``, so a wrong one becomes a dated,
sourced, confidence-scored FACT in the wiki. That launders a hallucination into
long-term memory, where nothing downstream can tell it from an observation.

So every line here is derived mechanically from the tool name, its arguments and
its actual result. The model's job is to reason; the bookkeeping is ours. This
costs nothing -- no extra LLM call, no added latency -- and it cannot lie.

What it is for
--------------
  * **Working memory.** ``BrowserSnapshotPruner`` collapses old page snapshots on
    the real-API path, which saves a lot of tokens and destroys the record of
    what happened on those pages. A compact action trail is the right
    replacement: dense, cheap, and it survives pruning.
  * **Learning input.** ``learn_from_session`` currently feeds the LLM a whole
    raw transcript -- mostly the model thinking out loud. Twenty action lines
    carry more signal for a fraction of the tokens.
  * **Audit.** Appended to ``data/actions.jsonl`` so a finished run can be
    reconstructed after the process is gone.

Relationship to the other records: ``agent_events`` is an ephemeral UI bus (it
goes quiet with no subscribers); ``usage_log`` counts tokens; ``audit.jsonl``
records a few high-consequence operations. This records every tool call.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from agentscope.middleware import MiddlewareBase

import config

LOG_PATH = os.path.join(config.DATA_DIR, "actions.jsonl")

# Per-session cap. A long browser run is hundreds of steps; keeping the newest
# bounds memory while leaving far more than any consumer asks for.
MAX_PER_SESSION = 500

# Argument rendering budgets. An email body or a generated report is kilobytes
# and must never be inlined -- it is summarised by size instead.
ARG_VALUE_CHARS = 48
ARG_MAX_KEYS = 4

# Two budgets, because the ledger serves two readers with opposite needs.
#
# STORED: data/actions.jsonl is the audit record. Diagnosing the empty-form
# failure needed the post-action page state, and 90 chars cut it off exactly
# where the evidence was -- "ran 2/2 actions. Current page: [ORIENT…". Disk is
# cheap; losing the one line that proves what happened is not.
#
# RENDERED: the trail is injected into prompts every step, so what the model
# reads stays short. A long result there is paid for on every subsequent call.
RESULT_CHARS = 400
LINE_RESULT_CHARS = 110

# Args worth showing first when a tool takes more than we can render. These are
# the ones that identify WHAT was acted on rather than how.
_SALIENT = ("to", "title", "subject", "when", "query", "url", "path", "name",
            "slug", "text", "event_id", "task_id", "file", "filename",
            "child", "description")

# Keys whose value IS the content, so squashing them to "<582 chars>" destroys
# the only informative part of the line. A real run logged
#   delegate_task(child=scripter, description=<582 chars>)
# which records that work was handed off but not what was asked for -- useless
# both to the trail injector and to the extractor that learns from it.
_LONG_KEYS = {"description", "task", "instruction", "goal", "prompt"}
LONG_VALUE_CHARS = 220

# Structured failure signals, checked before any text heuristic.
_FALSE_MARKERS = ('"ok": false', '"ok":false', "'ok': false", "ok=false")

# Leading-text failure markers. Only consulted at the START of a result, so a
# research summary that merely mentions "error" somewhere is not misread.
_FAIL_PREFIXES = ("error", "failed", "failure", "could not", "couldn't",
                  "unable to", "not found", "no such", "missing", "denied",
                  "permission", "timeout", "timed out", "refused", "invalid")

_lock = threading.Lock()
_trails: dict[str, list[dict]] = {}

# The session every action is attributed to. One server, one event loop, one
# turn in flight, so a module global is sufficient and avoids threading a
# session id through the middleware chain.
_current_session = ""


def set_session(session_id: str) -> None:
    """Attribute subsequent actions to this session. Safe to call repeatedly."""
    global _current_session
    _current_session = (session_id or "").strip()


def current_session() -> str:
    return _current_session


# ===========================================================================
#  rendering — deterministic, from the call and its real result
# ===========================================================================
def _fmt_value(v: Any, limit: int | None = None) -> str:
    """One argument value, short and single-line. Big payloads become a size."""
    limit = ARG_VALUE_CHARS if limit is None else limit
    try:
        if v is None:
            return "None"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, (list, tuple)):
            return f"[{len(v)}]"
        if isinstance(v, dict):
            return f"{{{len(v)}}}"
        s = " ".join(str(v).split())
        if len(s) > limit * 6:
            # An email body / report text: report its size, never its content.
            return f"<{len(s)} chars>"
        return s if len(s) <= limit else s[:limit] + "…"
    except Exception:
        return "?"


def _coerce_args(args: Any) -> dict:
    """Get a dict of arguments out of whatever the framework handed us.

    AgentScope's ``ToolCallBlock.input`` is annotated ``str`` and arrives as a
    JSON-encoded string, not a mapping. The first version of this module only
    accepted a dict, so every recorded line came out with EMPTY arguments --
    ``delegate_task() -> ok`` instead of naming the child and the task. The
    ledger still worked, which is exactly why it went unnoticed until a real
    run was inspected by eye.

    Accepts both shapes, because providers and versions differ on this.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        t = args.strip()
        if t.startswith("{"):
            try:
                parsed = json.loads(t)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, TypeError):
                pass
        # Not JSON: keep it rather than dropping it on the floor.
        return {"input": t} if t else {}
    return {}


def _fmt_args(args: Any) -> str:
    """`to=umesh@x.com, attachments=[1]` — salient keys first, budget-capped."""
    args = _coerce_args(args)
    if not args:
        return ""
    keys = [k for k in _SALIENT if k in args]
    keys += [k for k in args if k not in keys]
    parts = []
    for k in keys[:ARG_MAX_KEYS]:
        cap = LONG_VALUE_CHARS if k in _LONG_KEYS else ARG_VALUE_CHARS
        parts.append(f"{k}={_fmt_value(args[k], cap)}")
    if len(args) > ARG_MAX_KEYS:
        parts.append(f"+{len(args) - ARG_MAX_KEYS}")
    return ", ".join(parts)


def looks_failed(text: str) -> bool:
    """Whether a tool RESULT reports a failure despite not raising.

    A tool returning an error string while "succeeding" is this project's
    recurring failure mode, so the result text is inspected, not just the
    absence of an exception. Conservative on purpose: a structured ok=false
    counts anywhere, but prose markers count only at the very start, so a
    research result that happens to contain the word "error" is not misread.
    """
    t = " ".join((text or "").split()).strip()
    if not t:
        return False
    low = t.lower()
    if any(m in low[:400] for m in _FALSE_MARKERS):
        return True
    head = low.lstrip("{[\"' -•")
    return head.startswith(_FAIL_PREFIXES)


_BARE_OK = {"ok", "ok.", "true", "done", "success", "succeeded",
            '{"ok": true}', '{"ok":true}'}


def _clip(text: str, limit: int) -> str:
    """Shorten a stored result for a rendered line."""
    t = text or ""
    return t if len(t) <= limit else t[:limit] + "\u2026"


def _is_bare_ok(result: str) -> bool:
    """Whether a result says only 'it worked' and carries no other detail."""
    return " ".join((result or "").split()).strip().lower() in _BARE_OK


def line(action: dict, with_agent: bool = True) -> str:
    """Render one action as the single line consumers inject or summarise."""
    who = f"{action.get('agent', '?')}: " if with_agent else ""
    args = action.get("args_text", "")
    call = f"{action.get('tool', 'tool')}({args})"
    if action.get("ok"):
        res = _clip(action.get("result", ""), LINE_RESULT_CHARS)
        # Don't render "ok — ok": a result that only restates success adds
        # nothing, and these lines are injected into prompts where every
        # redundant token is paid for on every subsequent step.
        tail = "ok" if (not res or _is_bare_ok(res)) else f"ok — {res}"
    else:
        tail = (f"FAILED — {_clip(action.get('result', ''), LINE_RESULT_CHARS)}"
                .rstrip(" —"))
    return f"{who}{call} → {tail}"


# ===========================================================================
#  record + read
# ===========================================================================
def record(agent: str, tool: str, args: Any = None, ok: bool = True,
           result: str = "", ms: int = 0, session_id: str | None = None) -> dict:
    """Append one action. Never raises — bookkeeping must not break a turn."""
    try:
        sid = session_id if session_id is not None else _current_session
        res = " ".join((result or "").split())
        failed = (not ok) or looks_failed(res)
        action = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": sid,
            "agent": agent or "?",
            "tool": tool or "tool",
            "args_text": _fmt_args(args),
            "ok": not failed,
            "result": res[:RESULT_CHARS] + ("…" if len(res) > RESULT_CHARS else ""),
            "ms": int(ms or 0),
        }
        # The ledger's own copy is clipped on purpose -- it is re-read on every
        # board poll and injected into prompts. The untruncated arguments and
        # result go to call_log instead, linked by id, and are read only when
        # somebody opens that call in the viewer.
        try:
            import call_log
            if call_log.enabled():
                cid = call_log.new_id()
                action["call_id"] = cid
                call_log.write_tool(cid, agent, tool, args, result or "",
                                    ok=not failed, session=sid)
        except Exception:
            pass
        with _lock:
            trail = _trails.setdefault(sid, [])
            trail.append(action)
            if len(trail) > MAX_PER_SESSION:
                del trail[:len(trail) - MAX_PER_SESSION]
        _append_jsonl(action)
        return action
    except Exception:
        return {}


def _append_jsonl(action: dict) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(action, ensure_ascii=False) + "\n")
    except Exception:
        pass


def trail(session_id: str | None = None, agent: str | None = None,
          limit: int | None = None) -> list[dict]:
    """Actions for a session, optionally only one agent's, newest last."""
    sid = session_id if session_id is not None else _current_session
    with _lock:
        items = list(_trails.get(sid, []))
    if agent:
        items = [a for a in items if a.get("agent") == agent]
    if limit is not None and limit >= 0:
        items = items[-limit:]
    return items


def render(session_id: str | None = None, agent: str | None = None,
           limit: int | None = None, with_agent: bool = True) -> str:
    """The trail as newline-joined lines — what gets injected or summarised."""
    return "\n".join(line(a, with_agent=with_agent)
                     for a in trail(session_id, agent, limit))


def stats(session_id: str | None = None) -> dict:
    """Counts for a session: total, failed, and per-agent totals."""
    items = trail(session_id)
    per: dict[str, int] = {}
    failed = 0
    for a in items:
        per[a.get("agent", "?")] = per.get(a.get("agent", "?"), 0) + 1
        if not a.get("ok"):
            failed += 1
    return {"actions": len(items), "failed": failed, "per_agent": per}


def rebind(old_key: str, new_key: str) -> int:
    """Move a trail from a provisional run key to the archived session id.

    A turn's actions happen BEFORE the session exists -- the id is only minted
    by ``sessions.save_session`` at archive time -- so actions accumulate under
    a temporary run key and are re-keyed here. Without this, the trail would be
    unreachable by the one consumer that needs it most: ``learn_from_session``,
    which is handed a session id and nothing else.
    """
    if not old_key or not new_key or old_key == new_key:
        return 0
    with _lock:
        items = _trails.pop(old_key, [])
        if not items:
            return 0
        for a in items:
            a["session"] = new_key
        _trails.setdefault(new_key, []).extend(items)
        moved = len(items)
    global _current_session
    if _current_session == old_key:
        _current_session = new_key
    return moved


def start_run(prefix: str = "run") -> str:
    """Begin a fresh trail and make it current. Returns the provisional key."""
    key = f"{prefix}-{int(time.time() * 1000)}"
    set_session(key)
    return key


# ===========================================================================
#  middleware — put the trail back in front of the agent
# ===========================================================================
# How many of an agent's own recent actions to show it. Enough to cover a
# failed attempt AND the recovery that followed (the click-then-fallback shape
# this project keeps producing); few enough to stay a footnote, not a chapter.
TRAIL_LINES = int(os.environ.get("ACTION_TRAIL_LINES", "6"))

_TRAIL_OPEN = "[what you have already done this run — ground truth, newest last]"
_TRAIL_CLOSE = ("[end of record. This is derived from the actual tool calls and "
                "their real results, not from anything you reported.]")


def trail_message_text(agent: str, session_id: str | None = None,
                       limit: int | None = None) -> str:
    """The injected block for one agent, or '' when it has done nothing yet."""
    body = render(session_id, agent=agent,
                  limit=TRAIL_LINES if limit is None else limit,
                  with_agent=False)
    if not body.strip():
        return ""
    return f"{_TRAIL_OPEN}\n{body}\n{_TRAIL_CLOSE}"


def _is_trail_msg(msg) -> bool:
    try:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for b in content:
                t = getattr(b, "text", None)
                if t is None and isinstance(b, dict):
                    t = b.get("text")
                if isinstance(t, str) and t.startswith(_TRAIL_OPEN):
                    return True
        elif isinstance(content, str):
            return content.startswith(_TRAIL_OPEN)
    except Exception:
        pass
    return False


def result_text(chunk) -> str:
    """Pull displayable text out of a tool-result chunk, whatever its shape.

    Lives here rather than in agent_events so the ledger and the UI bus can
    never disagree about what a tool returned -- they would silently drift
    apart, and the ledger's whole value is that it matches reality.
    """
    try:
        out = getattr(chunk, "output", None)
        if isinstance(out, list):
            joined = " ".join(p for p in
                              (getattr(b, "text", "") or "" for b in out) if p)
            if joined:
                return joined
        content = getattr(chunk, "content", None)
        if isinstance(content, list):
            joined = " ".join(p for p in
                              (getattr(b, "text", "") or "" for b in content) if p)
            if joined:
                return joined
        if isinstance(content, str):
            return content
    except Exception:
        pass
    return ""


class ActionRecorder(MiddlewareBase):
    """Record every tool call this agent makes. Nothing else.

    The swarm gets its recording from ``agent_events.StreamEmitter`` (which also
    feeds the UI bus). The single-agent path has no StreamEmitter, so without
    this it would produce no ledger at all -- and the trail injector below would
    have nothing to inject. The two are never installed on the same agent, so a
    call is never counted twice.
    """

    def __init__(self, agent_label: str) -> None:
        self.label = agent_label

    async def on_acting(self, agent, input_kwargs: dict, next_handler):
        call = input_kwargs.get("tool_call")
        tool_name = getattr(call, "name", "") or "tool"
        raw_args = getattr(call, "input", None)
        started = time.time()
        last, ok = "", True
        try:
            async for item in next_handler(**input_kwargs):
                text = result_text(item)
                if text:
                    last = text
                yield item
        except Exception as exc:
            ok = False
            last = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                record(self.label, tool_name, raw_args, ok=ok, result=last,
                       ms=int((time.time() - started) * 1000))
            except Exception:
                pass


def _supervisor_owns_the_trail() -> bool:
    """Whether a supervised browser run is rendering its own step trail.

    Two trails were being maintained over the same actions: this one, and the
    ORIENTATION header's. They disagreed in useful ways -- this one carried
    ok/failure and a result excerpt, the header's carried the goal -- and the
    agent got both, six lines each, describing the same steps.

    The header's is now the goal-scoped one, so it is the better of the two
    while a run is active, and this block stands down rather than competing
    with it. Outside a supervised run (an unsupervised browse, comms working a
    chat UI with no goal plan) nothing else is keeping a trail, so this one
    still does its job.
    """
    try:
        import browser_supervisor as _sup
        return bool(_sup.context().active)
    except Exception:
        return False


class ActionTrailInjector(MiddlewareBase):
    """Keep one up-to-date action trail at the tail of an agent's context.

    Why this exists: ``BrowserSnapshotPruner`` collapses old page snapshots on
    the real-API path. That is a large, correct saving -- two consecutive
    snapshots of the same page are ~98% identical -- but it also destroys the
    only record of what the agent DID on those pages. The agent can end up
    re-trying something it already tried, or forgetting that a click failed and
    a different route worked.

    So the trail is the compact replacement for what the pruner removed: a few
    lines instead of several kilobytes of accessibility tree.

    Scoped to the agent's OWN actions. A specialist does not need the
    orchestrator's bookkeeping, and cross-agent noise is what the message bus
    and the status board are for.

    Cache note: the block is removed and re-appended each step, so everything
    before its previous position stays byte-identical and still cache-hits.
    Only the tail is re-sent. This is the same trade the pruner already makes on
    this path, and it is why the injector is not enabled where the pruner isn't.
    """

    def __init__(self, agent_label: str, lines: int | None = None) -> None:
        self.label = agent_label
        self.lines = TRAIL_LINES if lines is None else lines

    async def on_reasoning(self, agent, input_kwargs: dict, next_handler):
        try:
            self._refresh(agent)
        except Exception as exc:          # never let bookkeeping break a turn
            print(f"[trail] skipped ({type(exc).__name__}: {exc})")
        async for event in next_handler(**input_kwargs):
            yield event

    def _refresh(self, agent) -> None:
        context = getattr(getattr(agent, "state", None), "context", None)
        if not isinstance(context, list):
            return
        text = "" if _supervisor_owns_the_trail() else \
            trail_message_text(self.label, limit=self.lines)

        # Drop the previous block wherever it sits. Done even when there is no
        # new text, so a cleared trail cannot leave a stale record behind.
        for i in range(len(context) - 1, -1, -1):
            if _is_trail_msg(context[i]):
                del context[i]
        if not text:
            return

        from agentscope.message import Msg, TextBlock
        # role="user" matches every other synthetic injection in this codebase
        # (the swarm's nudges, the resume prompt) -- a role the providers here
        # already accept mid-ReAct.
        context.append(Msg(name="user",
                           content=[TextBlock(type="text", text=text)],
                           role="user"))


def clear(session_id: str | None = None) -> None:
    """Drop a session's in-memory trail (the jsonl file is untouched)."""
    sid = session_id if session_id is not None else _current_session
    with _lock:
        _trails.pop(sid, None)
