"""Saved repeatable browser tasks — registry + runner.

Phase 2 of the "repeatable task" feature. Once the agent has done a web task
once (slowly, via the ReAct browser loop) and it succeeded cleanly, the user can
choose to SAVE it. We then:

  1. compile that run's log into a parameterised Playwright script
     (script_generator.py), and
  2. register it here under a friendly name with its parameter spec.

After that the agent doesn't need to re-drive the browser step-by-step: it calls
the saved task by name with the inputs (run_task), and we execute the compiled
script directly — fast and deterministic. No AI is used to build or run the script.

Registry: data/browser_scripts/registry.json
    { "<task_name>": {name, description, script, params, source_run, created} }
Scripts:  data/browser_scripts/task_<slug>.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import config
import script_generator

AGENT_DIR = os.path.dirname(__file__)
REGISTRY = os.path.join(config.BROWSER_SCRIPT_DIR, "registry.json")
_RUN_TIMEOUT = int(os.environ.get("TASK_RUN_TIMEOUT", "300"))


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s or "task"


def _load_registry() -> dict:
    if not os.path.exists(REGISTRY):
        return {}
    try:
        with open(REGISTRY, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_registry(reg: dict) -> None:
    os.makedirs(config.BROWSER_SCRIPT_DIR, exist_ok=True)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, REGISTRY)


# --- public API ------------------------------------------------------------
def save_task(name: str, description: str = "", run_id: str | None = None) -> dict:
    """Compile a browser run into a reusable, parameterised task and register it.

    Args:
        name: friendly task name (also the key used to run it later).
        description: what the task does.
        run_id: which run to compile; defaults to the most recent run.
    Returns the saved manifest. Raises if there's no run or no replayable steps.
    """
    gen = (script_generator.generate_script(run_id) if run_id
           else script_generator.generate_latest())
    if gen["stats"]["steps"] == 0:
        raise ValueError(
            "that run has no replayable UI steps (nothing to save). Make sure the "
            "task actually clicked/typed and that locators were captured."
        )

    slug = _slug(name)
    script_path = os.path.join(config.BROWSER_SCRIPT_DIR, f"task_{slug}.py")
    shutil.copyfile(gen["path"], script_path)

    manifest = {
        "name": name,
        "slug": slug,
        "description": description,
        "script": script_path,
        "params": gen["params"],            # [{name, default, label}]
        "source_run": gen["run_id"],
        "stats": gen["stats"],
        "created": datetime.now(timezone.utc).isoformat(),
    }
    reg = _load_registry()
    reg[name] = manifest
    _save_registry(reg)
    _index_skill_in_wiki(manifest)
    return manifest


def _index_skill_in_wiki(manifest: dict) -> None:
    """Procedural memory: register this saved skill as a wiki page so the
    hot-path memory lookup can surface it ("you have a saved skill for this —
    run it the fast way"). Memory = facts (the wiki) + procedures (these skills).
    Best-effort: never let memory wiring break saving a task."""
    try:
        import wiki_store
        name = manifest["name"]
        params = [p["name"] for p in manifest.get("params", [])]
        desc = manifest.get("description", "").strip()
        param_hint = (" with params " + ", ".join(params)) if params else ""
        run_hint = (f"Run with run_repeatable_task('{name}'{param_hint}) instead "
                    f"of driving the browser step by step.")
        use_when = f"Use when: {desc}" if desc else ""
        page_desc = f"Saved browser skill. {run_hint} {use_when}".strip()
        wiki_store.write_page(
            title=f"Skill: {name}",
            body=f"## Skill\n- {run_hint}\n" + (f"- {use_when}\n" if use_when else ""),
            description=page_desc,
            tags=["skill", "procedure"],
            entities=[name.lower()] + [w for w in re.split(r"\W+", name.lower()) if len(w) > 2],
            confidence=0.95, source="skill",
            slug="skill-" + manifest["slug"],
        )
    except Exception as exc:  # pragma: no cover
        print(f"[skills] could not index skill in wiki: {exc}")


def _deindex_skill_in_wiki(slug: str) -> None:
    try:
        import wiki_store
        wiki_store.delete_page("skill-" + slug)
    except Exception:
        pass


def list_tasks() -> list[dict]:
    """All saved tasks (name, description, parameter names)."""
    out = []
    for m in _load_registry().values():
        out.append({
            "name": m["name"],
            "description": m.get("description", ""),
            "params": [p["name"] for p in m.get("params", [])],
        })
    return out


def get_task(name: str) -> dict | None:
    return _load_registry().get(name)


def delete_task(name: str) -> bool:
    reg = _load_registry()
    if name not in reg:
        return False
    script = reg[name].get("script")
    slug = reg[name].get("slug")
    reg.pop(name)
    _save_registry(reg)
    try:
        if script and os.path.exists(script):
            os.remove(script)
    except OSError:
        pass
    if slug:
        _deindex_skill_in_wiki(slug)
    return True


def run_task(name: str, params: dict | None = None, headless: bool = False) -> dict:
    """Execute a saved task's compiled Playwright script with the given param
    overrides. Returns {ok, name, output, error}."""
    m = get_task(name)
    if not m:
        return {"ok": False, "name": name, "error": f"no saved task '{name}'"}
    script = m["script"]
    if not os.path.exists(script):
        return {"ok": False, "name": name, "error": f"script missing: {script}"}

    cmd = [sys.executable, script, "--params", json.dumps(params or {})]
    if headless:
        cmd.append("--headless")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_RUN_TIMEOUT, cwd=AGENT_DIR
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "name": name, "error": f"timed out after {_RUN_TIMEOUT}s"}

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    ok = proc.returncode == 0 and "TASK_OK" in out
    return {
        "ok": ok,
        "name": name,
        "output": out[-1500:],
        "error": None if ok else (err[-1500:] or out[-1500:] or "non-zero exit"),
    }


# --- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Saved repeatable browser tasks")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list")
    p_save = sub.add_parser("save")
    p_save.add_argument("name")
    p_save.add_argument("--desc", default="")
    p_save.add_argument("--run", default=None)
    p_run = sub.add_parser("run")
    p_run.add_argument("name")
    p_run.add_argument("--params", default="{}")
    p_run.add_argument("--headless", action="store_true")
    p_del = sub.add_parser("delete")
    p_del.add_argument("name")

    a = ap.parse_args()
    if a.cmd == "list":
        for t in list_tasks():
            print(f"- {t['name']}: {t['description']}  params={t['params']}")
    elif a.cmd == "save":
        m = save_task(a.name, a.desc, a.run)
        print(f"Saved '{m['name']}' -> {m['script']}")
        print(f"  params: {[p['name'] for p in m['params']] or '(none)'}")
    elif a.cmd == "run":
        r = run_task(a.name, json.loads(a.params), a.headless)
        print(json.dumps(r, indent=2, ensure_ascii=False))
    elif a.cmd == "delete":
        print("deleted" if delete_task(a.name) else "not found")
    else:
        ap.print_help()
