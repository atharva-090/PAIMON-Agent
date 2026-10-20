"""Deterministic Playwright script generator.

Takes the action log of a SUCCESSFUL browser run (data/browser_runs/run_<id>.jsonl,
produced by browser_tools.py) and compiles it into a standalone, PARAMETERISED
Playwright script that replays the same UI actions. No AI is involved — plain,
deterministic codegen from the log.

Two things make this work:
  * Durable locators — the agent acts on ephemeral refs (@e5) that are useless on
    a later run, so browser_tools.py resolves each to a {role, name} locator at
    action time and logs it; we emit `page.get_by_role("button", name="Send")`.
  * Parameterisation — every text the agent TYPED becomes a named input (default =
    the recorded value), so one script handles "remind anyone anything", not just
    the exact words from the recording. `run(page, params)` + DEFAULT_PARAMS.

Registering a generated script as a reusable agent tool lives in task_scripts.py.

Usage:
    python script_generator.py                # compile the most recent run
    python script_generator.py <run_id>       # compile a specific run
    python script_generator.py --list         # list available runs
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from datetime import datetime

import config

# Ops that represent real UI actions we replay. Read-only/structural ops
# (snapshot, get_text) are never logged; `close` is handled by the wrapper.
_REPLAY_OPS = {"open", "click", "type", "press", "scroll", "back", "eval"}
_GUARDED_OPS = {"confirm_payment", "send_sensitive"}

_SCROLL_DELTA = {
    "down": "(0, 800)", "up": "(0, -800)",
    "right": "(800, 0)", "left": "(-800, 0)",
}


# --- log loading -----------------------------------------------------------
def _log_path(run_id: str) -> str:
    return os.path.join(config.BROWSER_LOG_DIR, f"run_{run_id}.jsonl")


def _op_of(rec: dict) -> str | None:
    """The canonical op for a record. New logs carry `op` directly; for older
    logs (pre locator-capture) we derive it from the human action label."""
    if rec.get("op"):
        return rec["op"]
    action = (rec.get("action") or "").strip().lower()
    if action.startswith("type into"):
        return "type"
    for verb in ("open", "click", "press", "scroll", "back", "eval", "close",
                 "confirm_payment", "send_sensitive"):
        if action.startswith(verb):
            return verb
    return None


def list_runs() -> list[dict]:
    """Available runs (newest first) with a light summary for picking one."""
    out = []
    pattern = os.path.join(config.BROWSER_LOG_DIR, "run_*.jsonl")
    # Newest first by MODIFICATION TIME (not filename) so "latest run" is
    # genuinely the most recent, regardless of run-id naming.
    paths = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for path in paths:
        run_id = os.path.basename(path)[len("run_"):-len(".jsonl")]
        recs = load_run(run_id)
        first_url = next(
            (r["args"].get("url") for r in recs if _op_of(r) == "open"), None
        )
        actions = sum(1 for r in recs if _op_of(r) in _REPLAY_OPS)
        failures = sum(1 for r in recs if r.get("ok") is False)
        out.append({
            "run_id": run_id, "first_url": first_url,
            "actions": actions, "failures": failures, "total": len(recs),
        })
    return out


def load_run(run_id: str) -> list[dict]:
    """Read a run log into a list of action records."""
    path = _log_path(run_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no run log: {path}")
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


# --- code emission ---------------------------------------------------------
def _q(value) -> str:
    """A safe Python string literal (handles quotes, newlines, unicode)."""
    return json.dumps(value if value is not None else "", ensure_ascii=False)


def _slug(s: str | None) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s or "value"


def _locator_expr(loc: dict | None) -> str | None:
    """Build a Playwright locator expression from a {role, name} locator."""
    role = (loc or {}).get("role")
    name = (loc or {}).get("name")
    if role and name:
        return f"page.get_by_role({_q(role)}, name={_q(name)})"
    if role:
        return f"page.get_by_role({_q(role)}).first"
    if name:
        return f"page.get_by_text({_q(name)}).first"
    return None


def _assign_params(recs: list[dict]) -> tuple[dict, list[dict]]:
    """Assign a unique param name to every successful `type` step. Returns
    (rec_id -> param_name, [param spec])."""
    rec_param: dict = {}
    specs: list[dict] = []
    used: dict = {}
    for rec in recs:
        if _op_of(rec) != "type" or rec.get("ok") is False:
            continue
        loc = rec.get("locator") or {}
        base = _slug(loc.get("name"))
        n = used.get(base, 0)
        name = base if n == 0 else f"{base}_{n + 1}"
        used[base] = n + 1
        rec_param[id(rec)] = name
        specs.append({
            "name": name,
            "default": (rec.get("args") or {}).get("text", ""),
            "label": loc.get("name") or "(unnamed field)",
        })
    return rec_param, specs


def _emit(rec: dict, param_name: str | None) -> list[str]:
    """Translate one log record into Playwright source lines (already indented
    inside run()). Returns [] for records that produce no replay step."""
    op = _op_of(rec)
    args = rec.get("args") or {}
    loc = rec.get("locator")

    if op == "open":
        return [f"page.goto({_q(args.get('url'))})"]

    if op in ("click", "type"):
        expr = _locator_expr(loc)
        if not expr:
            ref = args.get("ref")
            return [
                f"# TODO: no durable locator was captured for {op} on {ref!r} "
                f"(old/stale ref). Re-record the run or hand-edit this step.",
            ]
        if op == "click":
            return [f"{expr}.click()"]
        target = f"params[{_q(param_name)}]" if param_name else _q(args.get("text"))
        return [f"{expr}.fill({target})"]

    if op == "press":
        return [f"page.keyboard.press({_q(args.get('key'))})"]

    if op == "scroll":
        delta = _SCROLL_DELTA.get(args.get("direction", "down"), "(0, 800)")
        return [f"page.mouse.wheel{delta}"]

    if op == "back":
        return ["page.go_back()"]

    if op == "eval":
        return [
            "# review: this eval may have been exploration — remove if not needed",
            f"page.evaluate({_q(args.get('code'))})",
        ]

    if op in _GUARDED_OPS:
        summary = args.get("summary", "")
        kind = "payment" if op == "confirm_payment" else "sensitive message"
        return [
            f"# GUARDED STEP ({kind}) — the original run paused for human approval:",
            f"#   {summary}",
            f"input({_q('Approve the guarded step above, then press Enter to continue...')})",
        ]

    return []


def build_source(run_id: str, recs: list[dict]) -> tuple[str, dict, list[dict]]:
    """Build the full script source from a run's records.
    Returns (source, stats, param_specs)."""
    rec_param, param_specs = _assign_params(recs)
    body_lines: list[str] = []
    stats = {"steps": 0, "skipped_failed": 0, "unresolved": 0, "guarded": 0}

    for rec in recs:
        op = _op_of(rec)
        if op == "close" or op not in (_REPLAY_OPS | _GUARDED_OPS):
            continue
        if rec.get("ok") is False:
            stats["skipped_failed"] += 1
            body_lines.append(
                f"# skipped failed step: {rec.get('action')} "
                f"(error: {rec.get('error')})"
            )
            continue
        lines = _emit(rec, rec_param.get(id(rec)))
        if not lines:
            continue
        if any(l.startswith("# TODO: no durable locator") for l in lines):
            stats["unresolved"] += 1
        if op in _GUARDED_OPS:
            stats["guarded"] += 1
        else:
            stats["steps"] += 1
        body_lines.extend(lines)

    indent = "    "
    body = "\n".join(indent + l for l in body_lines) if body_lines else indent + "pass"
    defaults = {s["name"]: s["default"] for s in param_specs}
    defaults_src = json.dumps(defaults, ensure_ascii=False, indent=4)

    first_url = next(
        (r["args"].get("url") for r in recs if _op_of(r) == "open"), "(unknown)"
    )
    profile_dir = config.BROWSER_PROFILE_DIR
    param_doc = ("\n".join(f"      - {s['name']}: fills {s['label']!r} "
                           f"(default {s['default']!r})" for s in param_specs)
                 or "      (none — this task takes no inputs)")

    source = f'''"""Auto-generated Playwright script — compiled from browser run {run_id}.

Generated {datetime.now().isoformat(timespec="seconds")} by script_generator.py
(deterministic codegen — no AI). Source log: {os.path.basename(_log_path(run_id))}
Starting URL: {first_url}

Replays the run's successful UI actions using durable role/name locators. The
text typed during the run is exposed as PARAMETERS (override them to reuse this
task with different inputs); each defaults to the recorded value:
{param_doc}

Run:  python {run_id}.py --params "{{\\"some_field\\": \\"new value\\"}}"
      python {run_id}.py            (replays with the recorded defaults)
"""
from playwright.sync_api import sync_playwright

# Reuse the same persistent Chrome profile the agent used, so logins/cookies carry over.
USER_DATA_DIR = r"{profile_dir}"
CHANNEL = "chrome"

DEFAULT_PARAMS = {defaults_src}


def run(page, params=None):
    params = {{**DEFAULT_PARAMS, **(params or {{}})}}
{body}


def main(headless: bool = False, params=None):
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            USER_DATA_DIR, channel=CHANNEL, headless=headless
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            run(page, params)
        finally:
            ctx.close()


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="{{}}", help="JSON object of param overrides")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    try:
        main(headless=a.headless, params=json.loads(a.params))
        print("TASK_OK")
    except Exception as exc:  # surfaced to the task runner
        print(f"TASK_FAIL: {{type(exc).__name__}}: {{exc}}")
        raise
'''
    return source, stats, param_specs


# --- public API ------------------------------------------------------------
def generate_script(run_id: str, out_dir: str | None = None) -> dict:
    """Compile run <run_id> into a parameterised Playwright script. Returns
    {path, stats, params, run_id}."""
    recs = load_run(run_id)
    source, stats, params = build_source(run_id, recs)
    out_dir = out_dir or config.BROWSER_SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_id}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    return {"path": path, "stats": stats, "params": params, "run_id": run_id}


def generate_latest(out_dir: str | None = None) -> dict:
    runs = list_runs()
    if not runs:
        raise FileNotFoundError("no browser runs to compile")
    return generate_script(runs[0]["run_id"], out_dir)


# --- CLI -------------------------------------------------------------------
def _cli(argv: list[str]) -> None:
    if argv and argv[0] in ("--list", "-l"):
        runs = list_runs()
        if not runs:
            print("No browser runs found in", config.BROWSER_LOG_DIR)
            return
        print(f"{'run_id':<22} {'actions':>7} {'fails':>5}  first_url")
        for r in runs:
            print(f"{r['run_id']:<22} {r['actions']:>7} {r['failures']:>5}  "
                  f"{r['first_url'] or ''}")
        return

    run_id = argv[0] if argv else None
    result = generate_script(run_id) if run_id else generate_latest()
    s = result["stats"]
    print(f"Wrote {result['path']}")
    print(f"  {s['steps']} replay steps, {s['guarded']} guarded, "
          f"{s['unresolved']} unresolved locators, {s['skipped_failed']} failed-skipped")
    print(f"  params: {[p['name'] for p in result['params']] or '(none)'}")
    if s["unresolved"]:
        print("  NOTE: some steps had no durable locator (old log). Re-record "
              "after the browser_tools.py locator-capture update for a clean script.")


if __name__ == "__main__":
    _cli(sys.argv[1:])
