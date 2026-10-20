"""Browser-control tools for the Hermes-PM agent.

These wrap the `agent-browser-middleware` (which drives a real Chrome via the
Node `agent-browser` CLI) and expose them as AgentScope FunctionTools, so the
agent can browse the web autonomously inside its normal ReAct loop:

    browse_open -> browse_snapshot -> (reason about @eN refs) ->
    browse_click / browse_type / browse_press -> ... -> report

Design notes:
  * We drive the user's installed Chrome (system chrome.exe), headed, with a
    persistent profile -- all configured via environment variables that
    agent-browser reads (pushed below at import time). No middleware edits.
  * Element refs (@e1, @e2, ...) only exist AFTER a snapshot, and they go stale
    once the page changes. So state-changing actions here AUTO-RETURN a fresh
    compact snapshot -- this both keeps the agent oriented and halves the number
    of slow LLM round-trips.
  * Output is truncated (AGENT_BROWSER_MAX_OUTPUT + a Python backstop) so big
    accessibility trees don't overflow the LLM context.
  * Every action is logged to data/browser_runs/run_<id>.jsonl -- the raw
    material for the future "compile a repeated task into a script" phase.

Two high-risk actions are intentionally NOT done autonomously and are gated
behind confirmation tools: paying money and sending a sensitive message
(see browse_confirm_payment / browse_send_sensitive).
"""

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, ToolChunk

import config
from middleware.utils import (run_agent_browser_command, normalize_ref,
                              find_agent_browser)

# --- push browser settings into the env for the agent-browser subprocess -----
if config.BROWSER_CHROME_PATH and os.path.exists(config.BROWSER_CHROME_PATH):
    os.environ.setdefault("AGENT_BROWSER_EXECUTABLE_PATH", config.BROWSER_CHROME_PATH)
if config.BROWSER_HEADED:
    os.environ.setdefault("AGENT_BROWSER_HEADED", "1")
os.environ.setdefault("AGENT_BROWSER_PROFILE", config.BROWSER_PROFILE_DIR)
os.environ.setdefault("AGENT_BROWSER_MAX_OUTPUT", str(config.BROWSER_MAX_OUTPUT))
if config.BROWSER_ALLOWED_DOMAINS:
    os.environ.setdefault("AGENT_BROWSER_ALLOWED_DOMAINS", config.BROWSER_ALLOWED_DOMAINS)

SESSION = config.BROWSER_SESSION
_TEXT_CAP = config.BROWSER_MAX_OUTPUT + 1500  # Python backstop over agent-browser's

# A2 / A3 / A4 — see the block comments at each use site. The element-table
# flag lives in config because the pruner's install site reads it too.
COMPACT_SNAPSHOT = config.BROWSER_ELEMENT_TABLE
# Consecutive page-changing actions that changed nothing before the supervisor
# stops asking and recovers the page itself.
NO_CHANGE_LIMIT = int(os.environ.get("BROWSER_NO_CHANGE_LIMIT", "3"))


# --- helpers ---------------------------------------------------------------
def _chunk(text: str) -> ToolChunk:
    if len(text) > _TEXT_CAP:
        text = text[:_TEXT_CAP] + "\n…[truncated]"
    return ToolChunk(content=[TextBlock(type="text", text=text)])


def _err(msg: str) -> ToolChunk:
    return _chunk(f"ERROR: {msg}")


def chunk_text(out) -> str:
    """The text a browse_* tool returned, whatever shape it came back in.

    ToolChunk is a pydantic model holding TextBlocks; the autopilot has to read
    its own tool results to tell a stalled click from a working one, and it
    must not care which of dict / model / str it gets.
    """
    if isinstance(out, str):
        return out
    try:
        blocks = out["content"] if isinstance(out, dict) else out.content
    except Exception:
        return str(out)
    parts = []
    for b in blocks or []:
        if isinstance(b, dict):
            parts.append(str(b.get("text", "")))
        else:
            parts.append(str(getattr(b, "text", "") or ""))
    return "\n".join(p for p in parts if p)


_run_id: Optional[str] = None
# Refs map ({"e3": {"role": "...", "name": "..."}}) from the MOST RECENT
# snapshot. Element refs (@eN) are ephemeral — they only mean something relative
# to the snapshot the agent last saw. We cache that map here so that when the
# agent acts on @eN we can resolve it to a DURABLE locator (role + accessible
# name) and log that — the raw material the script generator needs to emit a
# replayable Playwright call (a bare "@e3" would be useless on a later run).
_last_refs: dict = {}


def _ensure_run() -> str:
    global _run_id
    if _run_id is None:
        _run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _run_id


def _capture_refs(data: dict) -> None:
    """Remember the refs map from a snapshot so later @eN actions resolve."""
    global _last_refs
    refs = (data or {}).get("refs")
    if isinstance(refs, dict):
        _last_refs = refs


def _resolve_locator(ref: str) -> Optional[dict]:
    """Look up a durable {role, name} locator for an @eN ref from the last
    snapshot the agent saw. Returns None if unknown (e.g. stale ref)."""
    key = ref.lstrip("@")
    loc = _last_refs.get(key)
    if isinstance(loc, dict) and (loc.get("role") or loc.get("name")):
        return {"role": loc.get("role"), "name": loc.get("name")}
    return None


def _log(op: str, label: str, args: dict, result: dict,
         locator: Optional[dict] = None) -> None:
    """Append one action to the current run log (best-effort).

    `op` is a canonical verb (open/click/type/press/scroll/back/eval/close/...)
    the script generator keys off; `label` is the human-readable action;
    `locator` is the resolved {role, name} for ref-based actions.
    """
    try:
        os.makedirs(config.BROWSER_LOG_DIR, exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": _ensure_run(),
            "op": op,
            "action": label,
            "args": args,
            "locator": locator,
            "ok": bool(result.get("success")),
            "error": result.get("error"),
        }
        path = os.path.join(config.BROWSER_LOG_DIR, f"run_{_ensure_run()}.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _run(op: str, label: str, command: str, args: list, log_args: dict,
         locator: Optional[dict] = None) -> dict:
    res = run_agent_browser_command(SESSION, command, args)
    _log(op, label, log_args, res, locator)
    return res


def _format_snapshot(data: dict) -> str:
    snap = (data or {}).get("snapshot") or ""
    origin = (data or {}).get("origin") or ""
    # Collapse a giant dropdown/option list (e.g. a 100-category picker) before it
    # bloats the prompt and drifts the weak brain into refusing — it's pointed at
    # the Search box instead. Best-effort; a normal page is unchanged.
    try:
        import browser_supervisor as sup
        snap = sup.collapse_option_overflow(snap)
    except Exception:
        pass
    head = f"URL: {origin}\n" if origin else ""
    out = (head + snap).strip() or "(empty page)"
    # A2: render the tree as an element table. Same refs, ~a third of the
    # characters. The knob exists because this reshapes every page the agent
    # ever sees; BROWSER_ELEMENT_TABLE=0 puts the raw tree back with no restart
    # of anything else.
    if COMPACT_SNAPSHOT:
        try:
            import browser_supervisor as sup
            out = sup.compact_snapshot(out)
        except Exception:
            pass
    return out


def _snapshot_text() -> str:
    res = run_agent_browser_command(SESSION, "snapshot", ["-c"])
    if not res.get("success"):
        return f"(could not read page: {res.get('error')})"
    data = res.get("data") or {}
    _capture_refs(data)
    return _format_snapshot(data)


def _header_for(op: str, target: str, snap: str) -> str:
    """Run one snapshot through the supervisor (P1/P4/P5/P6) and return the
    orientation header to prepend. Best-effort: "" if the supervisor is off."""
    try:
        import browser_supervisor as sup
        return sup.after_action(op=op, target=target, snapshot=snap,
                                run_id=_ensure_run(),
                                log_dir=config.BROWSER_LOG_DIR)
    except Exception:
        return ""


def _snap(op: str = "snapshot", target: str = "") -> str:
    """Take a fresh snapshot AND run it through the supervisor, returning the
    GOAL/CHECKLIST/TRAIL/STEP orientation header APPENDED to the page (never a
    rewrite of an older message — mock-safe). Just the snapshot when no
    supervised run is active."""
    snap = _snapshot_text()
    header = _header_for(op, target, snap)
    return f"{header}{snap}" if header else snap


def _blocked(op: str, target: str) -> Optional[str]:
    """Tier-2 auditor (rule 58): a block message if this action would just repeat
    a flagged loop, else None. Best-effort."""
    try:
        import browser_supervisor as sup
        return sup.should_block(op, target)
    except Exception:
        return None


def _current_href() -> Optional[str]:
    """Best-effort full URL (including path) of the page the browser is on right
    now, or None if no page is open / it can't be read. Used by browse_open to
    avoid pointlessly reloading a page we're already on."""
    res = run_agent_browser_command(SESSION, "eval", ["location.href"])
    if not res.get("success"):
        return None
    data = res.get("data") or {}
    href = data.get("result")
    return str(href).strip() if href else None


def _same_url(a: str, b: str) -> bool:
    """Compare two URLs ignoring only scheme and a trailing slash (path is kept,
    so /login and /attributes on the same host are correctly treated as
    DIFFERENT — we must never skip a genuine navigation)."""
    def norm(u: str) -> str:
        u = (u or "").strip()
        for p in ("https://", "http://"):
            if u.lower().startswith(p):
                u = u[len(p):]
                break
        return u.rstrip("/").lower()
    return bool(a) and bool(b) and norm(a) == norm(b)


# ===========================================================================
#  A3 / A4 — the wedged-page gate and what it reports
# ===========================================================================
# On 13 Oct the agent clicked "Add Source attributes" eleven times in 3.5
# minutes. Every click returned `OK: clicked` and a byte-identical page,
# because something was on top of the button and the click never reached it.
# The supervisor printed "⚠ AUDITOR: don't keep submitting" and the model read
# it and clicked again.
#
# So: at the limit the action is NOT run. The page is recovered instead
# (Escape, then a reload if Escape achieved nothing) and the agent is told what
# was in the way.

# What is painted over the page right now. Only ever run after a stall, so it
# costs nothing on the happy path. Borrowed in spirit from jev-ultrafast's
# pre-click hit test -- see the note in _overlay_report for what we do instead.
_OVERLAY_JS = r"""(() => {
  const hit = [];
  for (const e of document.querySelectorAll('body *')) {
    const s = getComputedStyle(e);
    if (s.display === 'none' || s.visibility === 'hidden' || +s.opacity === 0)
      continue;
    if (!['fixed', 'absolute', 'sticky'].includes(s.position)) continue;
    const r = e.getBoundingClientRect();
    if (r.width < 40 || r.height < 20) continue;
    const z = parseInt(s.zIndex) || 0;
    const tag = (String(e.className || '') + ' ' + (e.getAttribute('role') || ''));
    if (z < 1 && !/backdrop|overlay|modal|toast|snack|popup|dialog|loading|spinner|alert/i.test(tag))
      continue;
    hit.push((e.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 100)
             || (tag.trim().slice(0, 60) || e.tagName.toLowerCase()));
  }
  return hit.slice(0, 4).join(' // ');
})()"""


def _overlay_report() -> str:
    """One line naming whatever is painted on top, or "".

    jev-ultrafast hit-tests the chosen element itself
    (`el.contains(document.elementFromPoint(x, y))`). It can, because it owns
    the DOM node. Our refs belong to agent-browser's Playwright aria map and
    are not reachable from page JavaScript, so we answer the agent's actual
    question -- "what is in my way?" -- instead of testing one node.
    ponytail: scan, not hit test; swap in a real hit test if we ever carry
    element geometry through the snapshot.
    """
    try:
        res = run_agent_browser_command(SESSION, "eval", [_OVERLAY_JS])
        if not res.get("success"):
            return ""
        out = str(((res.get("data") or {}).get("result") or "")).strip()
        return out[:300]
    except Exception:
        return ""


# True when a form on this page is holding typed text. A reload would throw it
# away -- which is exactly what happened on the 09:02 run: the category form
# held "Retail_LN" and its description, the submit button could not be located,
# the gate counted four dead actions and reloaded, and the next read showed two
# empty fields. Escape is reversible; a reload is not.
_HAS_TYPED_JS = ("[...document.querySelectorAll('input,textarea')]"
                 ".some(e => (e.value || '').trim().length > 0)")


def _page_has_typed_input() -> bool:
    try:
        res = run_agent_browser_command(SESSION, "eval", [_HAS_TYPED_JS])
        return bool((res.get("data") or {}).get("result")) if res.get(
            "success") else True          # unreadable -> assume it has, and
    except Exception:                      # keep the destructive branch shut
        return True


def _recover_wedged_page() -> str:
    """Force the page out of a stuck state. Returns a one-line account.

    Escape first: it is reversible and it closes the usual culprit, a dialog or
    a leftover backdrop. Reload only when Escape achieved nothing AND no form
    is holding typed text -- a wedged page with a filled form is the one case
    where the cure is worse than the disease, because the agent loses the work
    and has to type it all again.
    """
    import browser_supervisor as sup
    before = sup.snapshot_hash(_snapshot_text())
    steps = ["pressed Escape"]
    run_agent_browser_command(SESSION, "press", ["Escape"])
    if sup.snapshot_hash(_snapshot_text()) == before:
        if _page_has_typed_input():
            steps.append("did NOT reload, because a form here still holds what "
                         "you typed — find the real submit control instead "
                         "(it may not be a <button type=submit>)")
        else:
            run_agent_browser_command(SESSION, "eval", ["location.reload()"])
            time.sleep(1.2)
            steps.append("reloaded the page")
    sup.clear_wedge()
    return " and ".join(steps)


def _wedge_guard(op: str) -> Optional[str]:
    """A4: the gate. A refusal message when the page is wedged, else None."""
    if (op or "") not in ("click", "type", "press", "batch", "click_text",
                          "select", "submit", "repeat", "attach_file"):
        return None
    try:
        import browser_supervisor as sup
        if not sup.wedged(NO_CHANGE_LIMIT):
            return None
        report = sup.wedge_report(NO_CHANGE_LIMIT)
        overlay = _overlay_report()
        did = _recover_wedged_page()
        covered = (f"\nOn top of the page: {overlay}" if overlay else "")
        return (report.replace("I have pressed Escape and re-read the page",
                               "I have " + did)
                + covered + "\n\nCurrent page:\n" + _snap("recover", "wedge"))
    except Exception:
        return None


def _action_then_snapshot(op: str, label: str, command: str, args: list,
                          log_args: dict,
                          locator: Optional[dict] = None) -> ToolChunk:
    """Run a state-changing action, then return a fresh compact snapshot so the
    agent immediately sees the new page state (saves an LLM round-trip).

    The locator is resolved from the PREVIOUS snapshot (what the agent saw when
    it chose the ref) BEFORE the action runs, because the action then refreshes
    the refs map to the new page.
    """
    wedge = _wedge_guard(op)          # A4: refuse + recover, before anything runs
    if wedge:
        return _chunk(wedge)
    blk = _blocked(op, label)
    if blk:
        return _err(blk)
    res = _run(op, label, command, args, log_args, locator)
    if not res.get("success"):
        return _err(res.get("error") or f"{label} failed")
    return _chunk(f"OK: {label}.\n\nCurrent page:\n{_snap(op, label)}")


# ===========================================================================
#  BATCH EXECUTION (run many actions in ONE agent-browser process)
# ===========================================================================
# agent-browser ships a native `batch` command: it takes a JSON array of arg-
# arrays on stdin and runs them in sequence in ONE process (the browser persists
# via its session daemon). We use it to collapse a known run of actions (fill a
# form, click submit, …) into a SINGLE subprocess + a SINGLE LLM round-trip,
# instead of one slow tool call per action. --bail stops at the first failure so
# a broken step doesn't cascade.

def _action_to_command(a: dict) -> list[str]:
    """Translate one high-level action dict into agent-browser CLI args. Raises
    ValueError with an actionable message on a malformed action."""
    if not isinstance(a, dict):
        raise ValueError(f"each action must be an object, got {type(a).__name__}")
    op = (a.get("action") or a.get("op") or "").strip().lower()
    if op == "click":
        return ["click", normalize_ref(a["ref"])]
    if op in ("type", "fill"):
        r = normalize_ref(a["ref"])
        bad = _typeable_error(r, str(a.get("text", "")))
        if bad:
            # ValueError is how this translator reports a malformed action;
            # browse_batch surfaces it and runs nothing, which is what we want:
            # a batch that fills the wrong node and then submits is worse than
            # a batch that refuses, because the submit looks like it worked.
            raise ValueError(bad)
        return ["fill", r, str(a.get("text", ""))]
    if op == "press":
        return ["press", str(a.get("key", "Enter"))]
    if op == "click_text":
        return ["find", "text", str(a["text"]), "click"]
    if op == "click_role":
        args = ["find", "role", str(a.get("role", "button")), "click"]
        if a.get("name"):
            args += ["--name", str(a["name"])]
        return args
    # REF-FREE FILLS — target a field by its label/placeholder instead of an @eN
    # ref, so the SAME action works every iteration even when the form rebuilds
    # (which is what breaks ref-based repetition). This is what makes
    # browse_repeat usable for "open form -> fill -> submit" cycles.
    if op in ("type_label", "fill_label"):
        return ["find", "label", str(a["label"]), "fill", str(a.get("text", ""))]
    if op in ("type_placeholder", "fill_placeholder"):
        return ["find", "placeholder", str(a["placeholder"]), "fill",
                str(a.get("text", ""))]
    if op == "select":  # native <select>: by ref or CSS selector
        target = normalize_ref(a["ref"]) if a.get("ref") else str(a.get("selector", ""))
        if not target:
            raise ValueError("select needs 'ref' or 'selector'")
        return ["select", target, str(a.get("value", ""))]
    if op == "scroll":
        return ["scroll", str(a.get("direction", "down"))]
    if op == "wait":
        return ["wait", str(int(a.get("ms", 800)))]
    raise ValueError(
        f"unknown action '{op}'. Use one of: click, type, press, click_text, "
        "click_role, type_label, type_placeholder, select, scroll, wait.")


def _describe_action(a: dict) -> str:
    op = (a.get("action") or a.get("op") or "?").lower()
    if op in ("type", "fill"):
        return f"type {a.get('text','')!r} into {a.get('ref','?')}"
    if op in ("type_label", "type_placeholder", "fill_label", "fill_placeholder"):
        return f"type {a.get('text','')!r} into {a.get('label') or a.get('placeholder','?')}"
    if op == "click":
        return f"click {a.get('ref','?')}"
    if op in ("click_text", "click_role"):
        return f"click '{a.get('text') or a.get('name','?')}'"
    return op


def _batch_label(actions: list) -> str:
    """A short human summary of a batch — used in the result head AND the
    orientation TRAIL, so the agent sees 'type 'X', click Submit' instead of the
    useless 'batch batch'."""
    parts = [_describe_action(a) for a in actions[:4]]
    s = ", ".join(parts)
    if len(actions) > 4:
        s += f", +{len(actions) - 4} more"
    return s[:80] or "batch"


def _action_op(a: dict) -> str:
    """Canonical op for the run log — so batched sub-actions log as their REAL op
    (click/type/...) and stay replayable by the script generator, instead of an
    opaque 'batch' that the generator drops."""
    op = (a.get("action") or a.get("op") or "").lower()
    if op in ("click", "click_text", "click_role"):
        return "click"
    if op in ("type", "fill", "type_label", "type_placeholder",
              "fill_label", "fill_placeholder"):
        return "type"
    if op in ("press", "scroll", "select", "wait"):
        return op
    return op or "action"


def _action_locator(a: dict) -> Optional[dict]:
    """A durable {role, name} locator for a batched action so its log entry is
    replayable. Name-based actions carry their own target; ref-based ones resolve
    against the last snapshot."""
    op = (a.get("action") or a.get("op") or "").lower()
    if op == "click_text":
        return {"role": None, "name": str(a.get("text", ""))}
    if op == "click_role":
        return {"role": str(a.get("role", "")) or None,
                "name": str(a.get("name", "")) or None}
    if op in ("type_label", "fill_label"):
        return {"role": "textbox", "name": str(a.get("label", ""))}
    if op in ("type_placeholder", "fill_placeholder"):
        return {"role": "textbox", "name": str(a.get("placeholder", ""))}
    if a.get("ref"):
        return _resolve_locator(normalize_ref(a["ref"]))
    return None


def _result_error(elem) -> Optional[str]:
    """An error string if one batch result element represents a failure, else
    None. Tolerant of the element being a dict ({success/ok/error}) or a string."""
    if isinstance(elem, dict):
        if elem.get("success") is False or elem.get("ok") is False:
            return str(elem.get("error") or "action failed")
        if elem.get("error"):
            return str(elem["error"])
    return None


def _run_batch(commands: list, bail: bool = True, timeout: int = 90) -> dict:
    """Run several agent-browser commands in ONE process via `batch` (JSON on
    stdin). Returns {success, results:[...]} or {success: False, error}. results
    is agent-browser's per-command JSON array, in order."""
    try:
        cmd_parts = find_agent_browser()
    except FileNotFoundError as exc:
        return {"success": False, "error": str(exc)}
    cmd = cmd_parts + ["batch"] + (["--bail"] if bail else []) \
        + ["--json", "--session", SESSION]
    payload = json.dumps(commands)
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", cwd=os.getcwd(),
        )
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    chunks: list[str] = []

    def _read():
        # Feed the commands in, then accumulate stdout until the JSON array we're
        # building parses cleanly (don't kill mid-output) — robust to results
        # whose own text contains brackets.
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except Exception:
            pass
        try:
            for line in proc.stdout:
                chunks.append(line)
                acc = "".join(chunks)
                start = acc.find("[")
                if start >= 0:
                    try:
                        json.loads(acc[start:])
                        break
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    reader.join(timeout=timeout)
    try:
        proc.kill()
        proc.wait(timeout=1)
    except Exception:
        pass

    out = "".join(chunks).strip()
    start = out.find("[")
    if start < 0:
        return {"success": False,
                "error": f"no JSON from batch (timeout or error). {out[:300]}"}
    try:
        results = json.loads(out[start:])
    except json.JSONDecodeError as exc:
        return {"success": False, "error": f"bad JSON from batch: {exc}",
                "raw": out[:300]}
    return {"success": True, "results": results if isinstance(results, list) else []}


def _batch_then_snapshot(actions: list, label: str, noun: str) -> ToolChunk:
    """Shared core for browse_batch / browse_repeat: translate actions, run them
    as one --bail batch, log each, and return how far it got + a fresh snapshot."""
    if not isinstance(actions, list) or not actions:
        return _err("no actions given")
    wedge = _wedge_guard("batch")     # A4: a batch is the most common disguise
    if wedge:                         # a repeated dead click wears
        return _chunk(wedge)
    try:
        commands = [_action_to_command(a) for a in actions]
    except (ValueError, KeyError) as exc:
        return _err(str(exc))
    res = _run_batch(commands, bail=True)
    if not res.get("success"):
        return _err(res.get("error") or "batch failed")
    results = res.get("results") or []
    # Log each action against its result (best-effort, for the script generator).
    first_err, ran_ok = None, 0
    for i, a in enumerate(actions):
        elem = results[i] if i < len(results) else None
        err = _result_error(elem) if elem is not None else (
            "did not run (a prior action failed)" if first_err else None)
        # Log under the REAL op (click/type/...) with the resolved locator, so the
        # script generator can still replay a batched run (was logged as 'batch'
        # and silently dropped — the regression noted in AUTOMATION_RULES rule 60).
        # Record WHY, not just that it failed. Without the reason a run log of
        # 55 failed clicks is unreadable: you cannot tell a real failure from
        # the actions that merely "did not run (a prior action failed)" after
        # the batch bailed -- which is most of them.
        _log(_action_op(a), _describe_action(a), {"action": a},
             {"success": err is None and elem is not None, "error": err},
             locator=_action_locator(a))
        if err and first_err is None:
            first_err = (i, err)
        elif err is None and elem is not None:
            ran_ok += 1
    # --bail truncates the results array at the failing step, so fewer results
    # than actions ALSO means a failure stopped the run (even if the element's
    # shape didn't expose an explicit error above).
    if first_err is None and len(results) < len(actions):
        first_err = (len(results), "a step failed and stopped the batch "
                     "(see the page below)")
    total = len(actions)
    head = (f"OK: {label} — ran {ran_ok}/{total} {noun}."
            if first_err is None
            else f"PARTIAL: {label} — {ran_ok}/{total} {noun} done, then "
                 f"step {first_err[0] + 1} failed: {first_err[1]}")
    return _chunk(f"{head}\n\nCurrent page:\n{_snap('batch', label)}")


# ===========================================================================
#  BROWSER TOOLS
# ===========================================================================
# ---------------------------------------------------------------------------
#  Typing into something that cannot hold text
# ---------------------------------------------------------------------------
# A real run typed a category name into a PAGE HEADING and reported success:
#
#   type 'CC and Hypo Loans' into @e6  ->  role=heading, name='Source Category'
#   click @e10                         ->  role=button,  name='Submit'
#
# The form submitted empty, the page never changed, and the agent announced the
# category had been created. The refs were not stale -- the agent had the right
# snapshot and simply picked the wrong element from it. Nothing downstream can
# recover from that, because a `fill` against a non-editable node is a silent
# no-op in the browser: it "succeeds" and writes nothing.
#
# So the check has to be here, at the point of the call: the resolved role is
# already known before the action runs, and a heading is never a typo for an
# input -- it is a different kind of thing.
#
# A DENYLIST, not an allowlist. Roles we do not recognise (custom widgets,
# 'generic' wrappers round a real input) are allowed through, because refusing
# them would break flows that work today. This only has to catch the class of
# mistake that actually happens: aiming at the visible label instead of the box.
_NON_TYPEABLE_ROLES = frozenset({
    "heading", "button", "link", "image", "img", "separator", "list",
    "listitem", "table", "row", "cell", "columnheader", "rowheader",
    "navigation", "banner", "contentinfo", "main", "article", "paragraph",
    "statictext", "menuitem", "tab", "tablist", "checkbox", "radio", "option",
    "progressbar", "alert", "dialog", "form", "figure", "note", "toolbar",
})


def _typeable_error(ref: str, text: str) -> str:
    """'' when `ref` can hold text, else an actionable message for the agent."""
    loc = _resolve_locator(ref)
    if not loc:
        return ""                      # unknown ref: let the browser decide
    role = (loc.get("role") or "").strip().lower()
    if role not in _NON_TYPEABLE_ROLES:
        return ""
    name = loc.get("name") or "?"
    return (
        f"REFUSED: {ref} is a {role} ({name!r}), which cannot hold text — "
        f"typing into it silently does nothing and any Submit that follows "
        f"sends an EMPTY field.\n"
        f"You are probably aiming at the label instead of the box. Take a "
        f"browse_snapshot and pick the ref whose role is textbox/searchbox/"
        f"combobox (often the line just BELOW this one), then type again. "
        f"If the field has a visible label, prefer the ref-free form: "
        f'browse_batch([{{"action": "type_label", "label": "{name}", '
        f'"text": "{text[:40]}"}}]).'
    )


# ---------------------------------------------------------------------------
#  Never fetch one of OUR generated reports over HTTP
# ---------------------------------------------------------------------------
# `/api/reports/<name>` is this project's own download endpoint, served by the
# Hermes server. A real run handed the browser child that path while it was
# sitting on a customer portal, and it navigated to
#
#   https://anandniyamweb.drutam.in:9634/api/reports/Letter_of_Credit_Loans_Info.xlsx
#
# -- our path resolved against THEIR origin. The portal is a single-page app,
# so it answered with its own shell and a title, and the navigation looked like
# a success. The child then carried on believing it had opened the spreadsheet.
#
# Fetching the file is the wrong idea even against the right origin: the child
# would get a binary download into a browser tab, not the rows it needs. The
# contents belong in the delegation (use_findings_from), which is why this
# child has no filesystem tools -- files are somebody else's job.
_REPORT_PATH_RE = re.compile(r"/api/reports/", re.I)


def _report_url_error(url: str) -> str:
    """'' unless this URL is one of our report downloads, else what to do."""
    if not _REPORT_PATH_RE.search(url or ""):
        return ""
    return (
        "REFUSED: /api/reports/... is THIS system's own download endpoint, not "
        "a page on the site you are browsing. Opening it against the current "
        "origin fetches nothing — a single-page app will answer with its own "
        "shell and look like it worked.\n"
        "You cannot read a generated file from the browser. If you need what is "
        "IN that file, you were meant to be given it: report back that the "
        "delegation referenced "
        f"{(url or '').rsplit('/', 1)[-1] or 'a file'} without including its "
        "contents, and ask for the values in the task description "
        "(the orchestrator attaches them with use_findings_from). "
        "Do NOT invent the contents and do NOT try another URL."
    )


def browse_open(url: str) -> ToolChunk:
    """Open a URL in the browser and return the page's interactive elements.

    Start any web task with this. The returned snapshot lists elements with
    [ref=eN] tags -- use those refs (e.g. '@e3') with browse_click / browse_type.

    Args:
        url (str): The URL to navigate to (include https://).
    """
    bad = _report_url_error(url)
    if bad:
        return _err(bad)
    # Don't reload a page we're already on. Re-opening the current URL throws
    # away in-progress form state and, across re-delegations, is the "page keeps
    # reloading" loop. If we're already here, just return the live page so the
    # agent resumes where it left off instead of starting over.
    here = _current_href()
    if here and _same_url(here, url):
        return _chunk(
            f"Already on {here} — not reloading (resuming). "
            f"Current page:\n{_snap('open', url)}"
        )
    res = _run("open", "open", "open", [url], {"url": url})
    if not res.get("success"):
        return _err(res.get("error") or "navigation failed")
    data = res.get("data") or {}
    info = f"Opened {data.get('url', url)} — title: {data.get('title', '?')}"
    return _chunk(f"{info}\n\nPage elements:\n{_snap('open', url)}")


def browse_snapshot() -> ToolChunk:
    """Re-read the current page's accessibility tree (interactive elements with
    [ref=eN]).

    You usually do NOT need this: click / type / press / scroll / back already
    return the fresh snapshot. Call it only when RESUMING a run (a browser is
    already open and you don't know where you are), or right after browse_open.
    """
    res = run_agent_browser_command(SESSION, "snapshot", ["-c"])
    if not res.get("success"):
        return _err(res.get("error") or "snapshot failed")

    # The weak brain habitually tails every batch with a snapshot even though
    # the action it just ran already returned one -- a redundant ~2-3k char page
    # dump per batch. If the page has not changed since the snapshot we last
    # handed it, say so in one line instead of repeating the whole tree. The
    # refs are still valid, because the page is byte-identical.
    # NB: compare the EXACT text, not sup.snapshot_hash(). That hash is
    # deliberately structural -- it strips refs, dates and bare numbers so a
    # repeated action can be detected -- which means 'ak_attribute_1' and
    # 'ak_attribute_2' hash the SAME. Fine for loop detection, wrong here: it
    # would hide a real change from the agent. Byte-identical or resend.
    try:
        import browser_supervisor as sup
        data = res.get("data") or {}
        fresh = _format_snapshot(data)
        ctx = sup.context()
        if ctx.active and ctx.last_snapshot and fresh == ctx.last_snapshot:
            _capture_refs(data)              # keep refs live for the next click
            return _chunk(
                "Page UNCHANGED since the snapshot you already have — the refs "
                "in it are still valid. (Not repeating the tree: click / type / "
                "press / scroll / back already return a fresh snapshot, so you "
                "rarely need browse_snapshot.)"
            )
    except Exception:
        pass

    # _snap re-reads + runs the supervisor (orientation header). A plain
    # browse_snapshot is a READ, so the loop monitor ignores it.
    return _chunk(_snap("snapshot"))


def browse_click(ref: str) -> ToolChunk:
    """Click an element by its ref. You MUST have a current snapshot first so the
    ref exists.

    Args:
        ref (str): Element ref from a snapshot, e.g. '@e5' (or 'e5' / '5').
    """
    r = normalize_ref(ref)
    loc = _resolve_locator(r)
    return _action_then_snapshot(
        "click", f"click {r}", "click", [r], {"ref": r}, locator=loc
    )


def browse_type(ref: str, text: str) -> ToolChunk:
    """Type text into an input/textarea element (clears it first).

    Args:
        ref (str): Element ref from a snapshot, e.g. '@e3'.
        text (str): The text to enter.
    """
    r = normalize_ref(ref)
    bad = _typeable_error(r, text)
    if bad:
        return _err(bad)
    loc = _resolve_locator(r)
    return _action_then_snapshot(
        "type", f"type into {r}", "fill", [r, text],
        {"ref": r, "text": text}, locator=loc
    )


def browse_press(key: str) -> ToolChunk:
    """Press a keyboard key (e.g. 'Enter', 'Tab', 'Escape', 'Control+a').

    Args:
        key (str): The key (or chord) to press.
    """
    return _action_then_snapshot(
        "press", f"press {key}", "press", [key], {"key": key}
    )


def browse_scroll(direction: str = "down") -> ToolChunk:
    """Scroll the page to reveal more content.

    Args:
        direction (str): 'up', 'down', 'left', or 'right'.
    """
    return _action_then_snapshot(
        "scroll", f"scroll {direction}", "scroll", [direction],
        {"direction": direction}
    )


def browse_back() -> ToolChunk:
    """Go back to the previous page in history."""
    return _action_then_snapshot("back", "back", "back", [], {})


def browse_get_text(selector: str = "body") -> ToolChunk:
    """Get the visible text content of the current page (use when you need to
    read the page body rather than just its interactive elements).

    Args:
        selector (str): CSS selector to read text from. Defaults to 'body'
            (the whole page); pass e.g. 'main' or '#content' to narrow it.
    """
    sel = (selector or "body").strip() or "body"
    res = run_agent_browser_command(SESSION, "get", ["text", sel])
    if not res.get("success"):
        return _err(res.get("error") or "get text failed")
    data = res.get("data") or {}
    return _chunk(str(data.get("result") or data.get("text") or data))


def browse_eval(code: str) -> ToolChunk:
    """Run JavaScript in the page and return the result. Useful for reading
    values or doing things the other tools don't cover.

    Args:
        code (str): JavaScript expression, e.g. 'document.title'.
    """
    res = _run("eval", "eval", "eval", [code], {"code": code})
    if not res.get("success"):
        return _err(res.get("error") or "eval failed")
    data = res.get("data") or {}
    return _chunk(str(data.get("result", data)))


def browse_close() -> ToolChunk:
    """Close the browser session when the task is finished."""
    global _run_id
    res = _run("close", "close", "close", [], {})
    _run_id = None  # next browse_open starts a fresh run log
    if not res.get("success"):
        return _err(res.get("error") or "close failed")
    return _chunk("Browser closed.")


def browse_click_text(text: str, role: str = "") -> ToolChunk:
    """Click an element by its VISIBLE TEXT / accessible name instead of a ref.
    Robust when refs go stale — it re-finds the element on the LIVE page, so it
    survives the page rebuilding after a submit. Prefer this over browse_click for
    buttons and links you can name: browse_click_text('Submit'),
    browse_click_text('Source category'). Optionally narrow by ARIA role
    ('button' / 'link' / 'tab' ...).

    Args:
        text (str): The visible label / accessible name to click.
        role (str): Optional ARIA role to disambiguate (e.g. 'button', 'link').
    """
    if not (text or "").strip():
        return _err("text is required")
    blk = _blocked("click", text)
    if blk:
        return _err(blk)
    if role.strip():
        args, loc = (["role", role.strip(), "click", "--name", text],
                     {"role": role.strip(), "name": text})
    else:
        args, loc = ["text", text, "click"], {"role": None, "name": text}
    res = _run("click", f"click '{text}'", "find", args,
               {"text": text, "role": role}, locator=loc)
    if not res.get("success"):
        return _err(res.get("error") or
                    f"couldn't find/click '{text}' — re-snapshot and check the "
                    "exact label, or use a ref.")
    return _chunk(f"OK: clicked '{text}'.\n\nCurrent page:\n{_snap('click', text)}")


def browse_batch(actions: list) -> ToolChunk:
    """Run several browser actions in ONE shot — one round-trip instead of one
    slow tool call per action. Use this whenever you already know the next few
    steps (e.g. type a value then click Submit, or fill a multi-field form).
    Actions run IN ORDER and STOP at the first failure (so a bad ref doesn't
    cascade); a single fresh page snapshot is returned at the end.

    Each action is a small dict (use refs from the LATEST snapshot):
      {"action":"click","ref":"@e5"}
      {"action":"type","ref":"@e7","text":"Applicant"}   (clears then types)
      {"action":"press","key":"Enter"}
      {"action":"click_text","text":"Submit"}            (find by visible text)
      {"action":"click_role","role":"button","name":"Submit"}
      {"action":"type_label","label":"Attribute Name","text":"x"}   (ref-free)
      {"action":"type_placeholder","placeholder":"Search","text":"x"}
      {"action":"select","ref":"@e8","value":"Applicant"}  (native <select>)
      {"action":"scroll","direction":"down"}
      {"action":"wait","ms":800}

    Tip: if the form REBUILDS after a submit (refs change), use click_text /
    click_role in the batch rather than @eN refs — they re-find the element.

    Args:
        actions (list): Ordered list of action dicts (see above).
    """
    return _batch_then_snapshot(actions, _batch_label(actions), "actions")


def _item_label(it) -> str:
    if isinstance(it, dict):
        return str(it.get("value") or it.get("text") or it)
    return str(it)


def browse_repeat(template: list, items: list,
                  defaults: Optional[dict] = None, verify: bool = False) -> ToolChunk:
    """Repeat the SAME short action sequence for many values in ONE shot — the
    fast way to fill a form N times (e.g. create 5 attributes). Give a `template`
    of actions with the placeholder "{}" where the per-item value goes, and a
    list of `items`.

    The template can include the WHOLE create cycle (open the form, fill it,
    submit) — do it ref-free so it survives the form rebuilding each iteration:
      template=[{"action":"click_text","text":"Add Source attributes"},
                {"action":"type_label","label":"Attribute Name","text":"{}"},
                {"action":"click_text","text":"{category}"},
                {"action":"click_text","text":"Save"}]
      items=["Existing_EMI","Credit_Score",
             {"value":"Employment_Type","category":"Employment Details"}]
      defaults={"category":"Applicant"}

    OVERRIDES (rule 60): an item may be a plain value OR a dict — its 'value'
    fills "{}" and its other keys fill "{name}" placeholders, overriding
    `defaults` for that one iteration (the odd-one-out, inline). `defaults` apply
    to every iteration.

    VERIFY (rule 61): pass verify=True to run iteration-by-iteration and STOP at
    the first one that DEVIATES from the first good iteration (an error appeared,
    or the form didn't submit) — slower, but it catches a per-item failure
    instead of blindly creating the rest. Default (verify=False) runs all items
    as one fast batch.

    IMPORTANT: refs (@eN) change when the form rebuilds, so prefer click_text /
    type_label / type_placeholder in the template. Do ONE iteration manually
    first to confirm the steps, THEN browse_repeat the rest — finish them here,
    don't hand the remainder back.

    Args:
        template (list): Action dicts with "{}" / "{name}" placeholders.
        items (list): Per-iteration values (plain) or override dicts.
        defaults (dict): Named values applied to every iteration (optional).
        verify (bool): Check each iteration against the first; stop on deviation.
    """
    if not isinstance(template, list) or not template:
        return _err("template must be a non-empty list of actions")
    if not isinstance(items, list) or not items:
        return _err("items must be a non-empty list of values")
    try:
        for a in template:                       # fail fast on a bad template
            if not isinstance(a, dict):
                return _err("each template action must be an object")
            _action_to_command(a)
    except (ValueError, KeyError) as exc:
        return _err(str(exc))
    import browser_supervisor as sup
    if not verify:
        expanded = sup.expand_template(template, items, defaults)
        return _batch_then_snapshot(
            expanded, f"repeat over {len(items)} item(s)", "actions")
    return _repeat_verified(template, items, defaults)


def _repeat_verified(template: list, items: list,
                     defaults: Optional[dict]) -> ToolChunk:
    """Iteration-by-iteration repeat with a per-item baseline deviation check
    (rule 61). The first iteration sets the baseline; later ones that deviate
    (error appeared / wrong page) stop the run so the agent can handle it."""
    import browser_supervisor as sup
    baseline: Optional[str] = None
    done = 0
    for idx, it in enumerate(items):
        actions = sup.expand_template(template, [it], defaults)
        res = _run_batch([_action_to_command(a) for a in actions], bail=True)
        if not res.get("success"):
            return _chunk(f"PARTIAL (verified): created {done}/{len(items)} — "
                          f"item {idx + 1} ({_item_label(it)}) could not run: "
                          f"{res.get('error')}\n\nCurrent page:\n{_snap('repeat', _item_label(it))}")
        results = res.get("results") or []
        for i, a in enumerate(actions):           # log sub-actions under real ops
            elem = results[i] if i < len(results) else None
            _log(_action_op(a), _describe_action(a), {"action": a},
                 {"success": elem is not None and _result_error(elem) is None},
                 locator=_action_locator(a))
        snap = _snapshot_text()
        header = _header_for("repeat", f"{_item_label(it)}", snap)  # updates supervisor
        if baseline is None:
            baseline = snap
        else:
            dev = sup.iteration_deviation(baseline, snap)
            if dev:
                return _chunk(
                    f"PARTIAL (verified): created {done}/{len(items)}. Item "
                    f"{idx + 1} ({_item_label(it)}) DEVIATED: {dev}. Stopped so "
                    f"you can handle it before continuing.\n\nCurrent page:\n"
                    f"{header}{snap}")
        done += 1
    final = _snapshot_text()
    head = _header_for("repeat", f"{done} items", final)
    return _chunk(f"OK (verified): created {done}/{len(items)} item(s), each "
                  f"checked against the first.\n\nCurrent page:\n{head}{final}")


def browse_repeat_last(items: list, value_was: str = "",
                       defaults: Optional[dict] = None,
                       verify: bool = False) -> ToolChunk:
    """Repeat the create-cycle you JUST did, for a list of new values — without
    re-writing the steps (rule 60). It reads the LAST completed iteration from
    your run log, turns the value you typed into a "{}" placeholder, and repeats
    it over `items`. Use it right after you've created ONE record successfully:
    'do that again for these names'.

    Args:
        items (list): The new per-iteration values (plain or override dicts).
        value_was (str): The value you used in the recorded iteration (e.g.
            'Monthly_Income'). Optional — auto-detected when the last iteration
            typed exactly one value.
        defaults (dict): Named values applied to every iteration (optional).
        verify (bool): Check each iteration against the first (see browse_repeat).
    """
    if not isinstance(items, list) or not items:
        return _err("items must be a non-empty list of values")
    import browser_supervisor as sup
    records = sup.read_run_log(_ensure_run(), log_dir=config.BROWSER_LOG_DIR)
    template, why = sup.derive_template_from_log(records, value_was=value_was)
    if not template:
        return _err(f"can't derive a repeatable template: {why}")
    return browse_repeat(template, items, defaults=defaults, verify=verify)


# ===========================================================================
#  FILE ATTACHMENT / DOWNLOAD (Roadmap #4 -- the clean Playwright path)
# ===========================================================================
def _as_selector(ref: str) -> str:
    """Accept either a snapshot ref ('@e7' / 'e7' / '7') or a raw CSS selector
    ('input[type=file]'). Refs are normalised to @eN; anything else is passed
    through as a selector (agent-browser accepts both)."""
    r = (ref or "").strip()
    core = r[1:] if r.startswith("@") else r
    if r.startswith("@") or r.isdigit() or (core.startswith("e") and core[1:].isdigit()):
        return normalize_ref(r)
    return r


def browse_attach_file(ref: str, path: str) -> ToolChunk:
    """Attach a local file to a web form's file input -- the clean way to add an
    attachment in Gmail/Teams/Slack web, set a profile picture, upload a
    document, etc. This sets the file on a file-<input> element programmatically
    (Playwright setInputFiles), so no native OS dialog is involved.

    The file must live in an ALLOWED filesystem root (see fs_list_roots / fs_rules)
    -- pass its root-qualified path (e.g. 'project/reports/q2.pdf' or
    'downloads/photo.png'). Attaching does NOT send anything; send/submit is still
    a separate, guarded step (browse_send_sensitive before an irreversible Send).

    Args:
        ref (str): The file-input element's ref from a snapshot (e.g. '@e7'), or a
            CSS selector like 'input[type=file]'.
        path (str): Root-qualified path to the local file to attach.
    """
    try:
        import fs_tools
        _root, full, disp = fs_tools._resolve(path, need="read")
    except Exception as exc:
        return _err(f"cannot attach '{path}': {exc}")
    if not os.path.isfile(full):
        return _err(f"no such file to attach: {disp}")
    sel = _as_selector(ref)
    res = _run("upload", f"attach {disp}", "upload", [sel, full],
               {"ref": sel, "path": disp})
    if not res.get("success"):
        return _err(res.get("error") or "attach failed")
    return _chunk(f"OK: attached {disp} to {sel}.\n\nCurrent page:\n{_snapshot_text()}")


def browse_download(ref: str, path: str) -> ToolChunk:
    """Download a file that the page produces by clicking a link/button, saving it
    into an ALLOWED folder. Use this for browser-driven downloads (an export
    button, a "Download" link) -- it captures Playwright's download event and
    writes the file where you ask.

    The destination must be in an allowed root; OUTSIDE the project workspace this
    is the same as any write -- only do it after the user is okay with it. Pass a
    root-qualified destination ('downloads' to keep the site's filename, or
    'project/out/export.csv' for a specific name).

    Args:
        ref (str): Ref (e.g. '@e4') or CSS selector of the element to click that
            triggers the download.
        path (str): Root-qualified destination folder or file path.
    """
    try:
        import fs_tools
        root, full, disp = fs_tools._resolve(path, need="write")
    except Exception as exc:
        return _err(f"cannot download to '{path}': {exc}")
    if root.get("confirm"):
        # A browser download into Downloads/Documents writes outside the sandbox.
        # Keep the same human-in-the-loop gate fs_download uses.
        if os.path.isdir(full) or not os.path.splitext(full)[1]:
            disp_target = f"{disp}".rstrip("/")
        else:
            disp_target = disp
        return _chunk(
            "PENDING_CONFIRMATION (browser download). This saves a downloaded file "
            f"OUTSIDE the project workspace:\n  {disp_target}\nNothing was saved. "
            "Confirm with the user, then save it into 'project/...' instead, or "
            "have them approve this location.")
    sel = _as_selector(ref)
    res = _run("download", f"download to {disp}", "download", [sel, full],
               {"ref": sel, "path": disp})
    if not res.get("success"):
        return _err(res.get("error") or "download failed")
    return _chunk(f"OK: downloaded to {disp}.")


# ===========================================================================
#  GUARDED ACTIONS (the only two things that need the user's OK)
# ===========================================================================
def browse_confirm_payment(summary: str) -> ToolChunk:
    """Use this INSTEAD of clicking when an action would spend money / complete a
    payment / place an order / transfer funds. It does NOT perform the action --
    it asks the user to approve first. After the user says yes, proceed with the
    actual browse_click on the pay/confirm button.

    Args:
        summary (str): A precise summary of the payment (what, how much, to whom).
    """
    _log("confirm_payment", "confirm_payment", {"summary": summary}, {"success": True})
    return _chunk(
        "PENDING_CONFIRMATION (payment). Do NOT proceed yet. Tell the user "
        "exactly this and wait for an explicit 'yes' before clicking the "
        f"pay/confirm button:\n\n{summary}"
    )


def browse_send_sensitive(summary: str) -> ToolChunk:
    """Use this INSTEAD of clicking 'Send' when about to send a clearly sensitive
    email or message (legal, financial, HR, external/important recipients, or
    anything irreversible and consequential). It does NOT send -- it asks the
    user to approve first. After an explicit yes, proceed with the real send.

    Args:
        summary (str): Recipient(s) + the gist of the message being sent.
    """
    _log("send_sensitive", "send_sensitive", {"summary": summary}, {"success": True})
    return _chunk(
        "PENDING_CONFIRMATION (sensitive message). Do NOT send yet. Tell the "
        "user exactly this and wait for an explicit 'yes' before clicking "
        f"Send:\n\n{summary}"
    )


# ===========================================================================
#  FACTORY
# ===========================================================================

# ===========================================================================
#  Stage B — browse_autopilot: Jev decides the step, we execute it
# ===========================================================================
# A browser step on gpt-6-luna costs ~13,000 input tokens and 3-9 seconds, and
# most steps are mechanical: which of these forty controls is the Submit
# button. TypeSafe's Jev answers exactly that in a few hundred milliseconds off
# an element table, for ~1,000 tokens.
#
# This tool runs that loop. Each iteration executes through the SAME helpers a
# hand-driven step uses, so the supervisor header, the checklist, the A4 wedge
# gate and the action ledger all still apply -- the only thing replaced is the
# model call that chose the element.
#
# It is a tool, not a replacement agent: the browser child still owns the goal,
# still writes the report, and can drop back to clicking by hand whenever the
# autopilot stops. That is deliberate -- Jev's own README says DONE is never
# independent evidence of success, and our checklist is what settles that.

AUTOPILOT_MAX_STEPS = int(os.environ.get("AUTOPILOT_MAX_STEPS", "14"))
# Seconds to let a filtered list repaint before the next decision.
AUTOPILOT_SETTLE_S = float(os.environ.get("AUTOPILOT_SETTLE_S", "0.5"))


def autopilot_available() -> bool:
    """True when a TypeSafe key is configured, so the roster can offer
    the tool only where it works."""
    try:
        import jev_policy
        return jev_policy.enabled()
    except Exception:
        return False


def _autopilot_step(goal: str, snap: str, history: list) -> dict:
    """One decision + one execution. Returns a record for the trail."""
    import jev_policy

    # A form whose fields have no accessible name needs one DOM probe before
    # anyone can reason about it. Skipped on every page that does not.
    hints = []
    els, _t, _r = jev_policy.action_space(snap)
    if jev_policy.needs_labels(els):
        try:
            res = run_agent_browser_command(SESSION, "eval",
                                            [jev_policy.LABEL_PROBE_JS])
            if res.get("success"):
                hints = str((res.get("data") or {}).get("result")
                            or "").split("|")
        except Exception:
            hints = []
    d = jev_policy.choose(snap, goal, history, label_hints=hints)
    op, ref = d["operation"], d.get("ref", "")
    rec = {"action": d.get("label") or op, "kind": op.lower(),
           "operation": op, "ref": ref, "text": None,
           "latency_ms": d.get("latency_ms"), "usage": d.get("usage"),
           "page_changed": None, "result": ""}
    if op in ("DONE", "BLOCKED"):
        return rec

    if op == "TYPE_TEXT":
        text, meta = jev_policy.field_text(
            goal, {"label": d.get("label"), "role": d.get("role")},
            jev_policy.page_context(snap), history)
        rec["text_latency_ms"] = meta.get("latency_ms")
        if not text:
            # The helper refused to invent a value. Typing nothing is correct;
            # pretending we typed something is how a form gets submitted blank.
            rec["operation"] = "BLOCKED"
            rec["result"] = "the text helper returned no value for this field"
            return rec
        rec["text"] = text
        out = browse_type(ref, text)
    else:
        out = browse_click(ref)

    body = chunk_text(out)
    rec["result"] = body[:400]
    rec["page_changed"] = "NOTHING CHANGED" not in body
    rec["refused"] = body.startswith("REFUSED")
    return rec


def browse_autopilot(goal: str, max_steps: int = 0) -> ToolChunk:
    """Drive the CURRENT page toward one small goal, several steps at a time.

    Use this for the mechanical part of a form: opening the Add dialog, filling
    the fields and submitting. It picks the element itself, so it costs a
    fraction of deciding each click yourself. Give it ONE outcome, with the
    literal values in the sentence.

    It stops on its own when the goal looks done, when it is blocked, or after
    max_steps. It always returns the final page, so you can check the result
    and carry on by hand.

    Args:
        goal (str): One outcome, with its values, e.g. "create an attribute
            named Monthly_income of type Number with description 'verified net
            monthly income'".
        max_steps (int): Optional cap on actions (default 14).
    """
    import jev_policy
    if not jev_policy.enabled():
        return _err("Autopilot is off (no TYPESAFE_API_KEY). Drive the page "
                    "with browse_click / browse_type as usual.")
    goal = (goal or "").strip()
    if len(goal) < 8:
        return _err("Give browse_autopilot one concrete outcome, with values.")

    limit = max(1, min(int(max_steps or AUTOPILOT_MAX_STEPS), 40))
    history: list = []
    lines: list = []
    snap = _snap("snapshot", "autopilot")
    stop = "ran out of steps"
    for i in range(limit):
        try:
            rec = _autopilot_step(goal, snap, history)
        except Exception as exc:
            stop = f"autopilot error: {type(exc).__name__}: {exc}"
            break
        op = rec["operation"]
        if op == "DONE":
            stop = "the policy reports the goal is satisfied"
            lines.append(f"  {i + 1}. DONE")
            break
        if op == "BLOCKED":
            stop = "blocked: " + (rec["result"] or "no supported operation")
            lines.append(f"  {i + 1}. BLOCKED")
            break
        mark = "✓" if rec["page_changed"] else "✗"
        what = rec["action"] if not rec["text"] else \
            f"{rec['action']} = {rec['text'][:40]!r}"
        lines.append(f"  {i + 1}. {mark} {op} {what} ({rec['latency_ms']}ms)")
        history.append({k: rec[k] for k in
                        ("action", "kind", "text", "page_changed")})
        if rec.get("refused"):
            stop = "the wedge gate refused the action — the page is stuck"
            break
        # A filter box repaints its list asynchronously. Without this the next
        # snapshot still shows the unfiltered dropdown, so the policy cannot
        # see the option it just searched for and clicks the toggle again --
        # the open/close oscillation seen on the first live run. jev waits for
        # visible suggestions after typing into a combobox for the same reason.
        if op == "TYPE_TEXT":
            time.sleep(AUTOPILOT_SETTLE_S)
        snap = _snapshot_text()

    head = (f"AUTOPILOT ran {len(lines)} step(s) toward: {goal[:120]}\n"
            + "\n".join(lines) + f"\nStopped: {stop}\n")
    return _chunk(head + "\nCurrent page:\n" + _snap("autopilot", goal[:40]))


_BROWSER_FUNCS = [
    browse_open, browse_snapshot, browse_click, browse_type, browse_press,
    browse_scroll, browse_back, browse_get_text, browse_eval, browse_close,
    # Fast paths: one round-trip for many actions, loop-unroll a repeated form
    # (+ derive-from-log + per-item verify), and act-by-name (stale-ref-proof).
    browse_batch, browse_repeat, browse_repeat_last, browse_click_text,
    browse_attach_file, browse_download,
    # Stage B: offered only when TYPESAFE_API_KEY is set -- an un-taught tool
    # sitting in the schema is worse than a missing one (see the note on the
    # repeatable-task tools in roster.py).
    *( [browse_autopilot] if __import__("jev_policy").enabled() else [] ),
    browse_confirm_payment, browse_send_sensitive,
]


def build_tools() -> list:
    """Return the browser tools as AgentScope FunctionTools."""
    return [FunctionTool(f, is_read_only=False) for f in _BROWSER_FUNCS]
