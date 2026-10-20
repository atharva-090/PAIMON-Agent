"""Code mode for the scripter: a single sandboxed ``run_python`` tool.

WHY (CodeAct / "code as the universal action"). The scripter's brain is
ChatGPT-web driven through the mock -- slow (~10 s/turn) and a weak
instruction-follower. Answering a *procedural* question (a loop + a cross-file
join + per-row conditionals) as a chain of JSON tool calls takes many fragile
turns where the weak brain loses the thread. It is far more reliable and cheaper
to let the brain write ONE short Python script: the control flow then runs in
deterministic Python -- fast and exact -- and only "write a correct script"
stays AI. This is an ESCALATION tool; plain counts/totals still go through the
simpler excel_*/query_data tools.

SECURITY -- generated code is treated as UNTRUSTED (it may be buggy or
prompt-injected via file contents). Layered, in-process containment:

  1. Static AST gate (_validate), BEFORE anything runs: no imports; no dunder
     attribute access (``x.__class__``/``__globals__`` ...) and no bare dunder
     names (``__builtins__``); an attribute denylist (every ``read_*``, the
     pandas file-writers ``to_csv``/``to_excel``/``to_pickle``/..., plus
     ``eval``/``query``/``format``/``format_map``/``system``/``popen``/``tofile``/
     ``dump``); and a name denylist (``open``/``eval``/``exec``/``getattr`` ...).
  2. Curated builtins: a tiny safe set; NO open/eval/exec/getattr/__import__,
     so the ONLY path to a dunder is literal ``.__x__`` syntax -- which (1) kills.
  3. No filesystem inside the sandbox: the script reaches data ONLY through
     vetted, path-sandboxed helpers (load_table/load_rows/sql/read_doc), built on
     file_tools internals. It cannot import or open anything; a real ``pd`` is
     exposed but its file readers are absent and its writers blocked by (1).
  4. Resource bounds: a ``sys.settrace`` wall-clock deadline that interrupts
     pure-Python loops, a tool-call cap on the data helpers, and bounded output.

RESULTS flow through the EXISTING findings ledger: the script calls
record_answer(question, answer) -- so the swarm's report is assembled
deterministically from the ledger, exactly like the other scripter tools -- and/
or sets ``result``. run_python returns captured prints + result + the running
ledger, or a clean error + the offending line for bounded self-repair.

Limitation (honest): the settrace deadline can't interrupt a long *C-level*
pandas call; acceptable because data only ever comes from row-bounded helpers.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import collections as _collections
import datetime as _datetime
import io
import json as _json
import math as _math
import os
import re as _re
import statistics as _statistics
import sys
import threading
import time

from agentscope.message import TextBlock
from agentscope.tool import ToolChunk

import config
import file_tools as F
from multi_agent import findings as _findings

_MAX = config.FILE_MAX_OUTPUT
_TIMEOUT_S = float(os.environ.get("CODE_MODE_TIMEOUT", "8"))
_MAX_TOOL_CALLS = int(os.environ.get("CODE_MODE_MAX_TOOL_CALLS", "300"))
_MAX_ROWS_OUT = 50          # rows shown when a DataFrame/Series is the result
_TRACE_EVERY = 200          # check the wall-clock deadline every N trace events


# ===========================================================================
#  ToolChunk helpers (mirror file_tools)
# ===========================================================================
def _chunk(text: str) -> ToolChunk:
    if len(text) > _MAX:
        text = text[:_MAX] + f"\n…[truncated — {len(text)} chars; print/emit less]"
    return ToolChunk(content=[TextBlock(type="text", text=text)])


def _ok(text: str) -> ToolChunk:
    return _chunk(text)


def _err(msg: str) -> ToolChunk:
    return _chunk(f"ERROR: {msg}")


# ===========================================================================
#  Layer 1 — static AST gate
# ===========================================================================
# Bare names the script may never use (escape hatches / dynamic eval / file I/O).
_DENY_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "getattr",
    "setattr", "delattr", "globals", "locals", "vars", "dir", "type",
    "memoryview", "breakpoint", "help", "exit", "quit", "object", "super",
    "classmethod", "staticmethod", "property",
}
# Attribute names blocked on ANY object (in addition to: anything dunder, and any
# attribute starting with "read_"). The to_* set is exactly the pandas/numpy
# serializers that can write a file; safe converters (to_dict/to_list/to_numpy/
# to_datetime/to_frame...) are intentionally NOT here.
_DENY_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_hdf", "to_feather",
    "to_json", "to_sql", "to_stata", "to_gbq", "to_clipboard", "to_orc",
    "to_xml", "to_html", "tofile", "dump",
    "eval", "query", "format", "format_map", "system", "popen", "Popen",
}


def _validate(code: str):
    """Return None if ``code`` is safe to run, else a precise human-readable
    rejection reason. Pure static analysis -- nothing executes."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return ("imports are not allowed — the data API is already available "
                    "(load_table / load_rows / sql / read_doc / list_files / "
                    "list_sheets / pd).")
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("__"):
                return f"access to dunder attribute '.{a}' is not allowed."
            if a.startswith("read_") or a in _DENY_ATTRS:
                return (f"attribute '.{a}' is not allowed (no file I/O / eval / "
                        "string-format escape inside code mode).")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in _DENY_NAMES:
                return f"use of name '{node.id}' is not allowed."
    return None


# ===========================================================================
#  Layer 2 — curated builtins
# ===========================================================================
# A deliberately small, safe set. Note the ABSENCE of open/eval/exec/getattr/
# __import__/type/vars/globals: with those gone, the only route to a dunder is
# literal `.__x__`, which the AST gate rejects. Exception classes are included so
# try/except works; their .__subclasses__() is dunder (blocked).
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "int", "isinstance", "len", "list", "map", "max",
    "min", "pow", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip", "bytes",
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
    "ZeroDivisionError", "RuntimeError", "AttributeError", "StopIteration",
)
_SAFE_BUILTINS = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES}

# Pure-computation stdlib modules pre-injected into the sandbox BY NAME (the brain
# uses `re.findall(...)` etc. WITHOUT writing `import`, which the AST gate forbids).
# These have no filesystem/network entry point reachable without a dunder attribute
# -- and dunder access is blocked by the AST gate -- so they widen capability
# (text/date/math/stats work) without widening the escape surface.
_SAFE_MODULES = {
    "re": _re, "math": _math, "statistics": _statistics,
    "collections": _collections, "datetime": _datetime, "json": _json,
}


# ===========================================================================
#  Layer 3 — curated data surface (no filesystem; built on file_tools internals)
# ===========================================================================
def _is_spreadsheet(full: str) -> bool:
    return os.path.splitext(full)[1].lower() in F._SPREADSHEET_EXTS


def _make_pd_shim():
    """A pandas namespace exposing only safe constructors/combinators -- NO
    read_* (file readers) and NO eval. DataFrame *instances* are real, but their
    file-writing methods are blocked by the AST gate, not here."""
    import pandas as pd
    from types import SimpleNamespace
    return SimpleNamespace(
        DataFrame=pd.DataFrame, Series=pd.Series, concat=pd.concat,
        merge=pd.merge, to_datetime=pd.to_datetime, to_numeric=pd.to_numeric,
        isna=pd.isna, notna=pd.notna, isnull=pd.isnull, notnull=pd.notnull,
        cut=pd.cut, qcut=pd.qcut, date_range=pd.date_range, NA=pd.NA, NaT=pd.NaT,
        unique=pd.unique,
    )


def _make_data_api():
    """Return the dict of data-access callables injected into the sandbox. Each
    reaches data ONLY through file_tools' path-sandboxed internals."""
    def load_table(filename, sheet=None):
        full = F._resolve(filename)
        if not _is_spreadsheet(full):
            raise ValueError(f"{filename!r} isn't a spreadsheet; use read_doc().")
        df, _label = F._load_table(full, sheet)
        return df

    def load_rows(filename, sheet=None):
        return load_table(filename, sheet).to_dict("records")

    def sql(filename, query):
        full = F._resolve(filename)
        if not _is_spreadsheet(full):
            raise ValueError(f"{filename!r} isn't a spreadsheet; use read_doc().")
        return F._run_sql(full, query)

    def list_files():
        out = []
        for name in sorted(os.listdir(F.UPLOADS_DIR)):
            full = os.path.join(F.UPLOADS_DIR, name)
            if not name.startswith(".") and os.path.isfile(full):
                out.append(name)
        return out

    def list_sheets(filename):
        full = F._resolve(filename)
        if full.lower().endswith(".csv"):
            return ["(csv)"]
        import pandas as pd
        return list(pd.ExcelFile(full).sheet_names)

    def read_doc(filename):
        full = F._resolve(filename)
        if _is_spreadsheet(full):
            raise ValueError(f"{filename!r} is a spreadsheet; use load_table().")
        return F._to_markdown_cached(full)

    return {"load_table": load_table, "load_rows": load_rows, "sql": sql,
            "list_files": list_files, "list_sheets": list_sheets,
            "read_doc": read_doc}


# ===========================================================================
#  Layer 4 — bounded execution (settrace deadline; runs on a worker thread)
# ===========================================================================
def _run_with_limits(code_obj, ns: dict, timeout_s: float):
    """Execute ``code_obj`` under a wall-clock deadline enforced by a per-line
    trace function (which interrupts pure-Python loops). Returns the captured
    exception (or None). Runs on a daemon thread so a thread wedged in C code
    can't block the server."""
    deadline = time.monotonic() + timeout_s
    counter = {"n": 0}
    box = {"exc": None}

    def _tracer(frame, event, arg):
        counter["n"] += 1
        if counter["n"] % _TRACE_EVERY == 0 and time.monotonic() > deadline:
            raise TimeoutError(f"script exceeded the {timeout_s:.0f}s time limit "
                               "— loop fewer times or aggregate with one sql() call.")
        return _tracer

    def _target():
        sys.settrace(_tracer)
        try:
            exec(code_obj, ns)
        except BaseException as exc:   # noqa: BLE001 — surface ANY failure to the brain
            box["exc"] = exc
        finally:
            sys.settrace(None)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_s + 1.0)           # small grace beyond the tracer deadline
    if t.is_alive():                  # stuck in a C call the tracer can't break
        box["exc"] = TimeoutError(
            f"script exceeded {timeout_s:.0f}s and could not be interrupted "
            "(avoid heavy pandas ops on unbounded data).")
    return box["exc"]


# ===========================================================================
#  rendering / errors
# ===========================================================================
def _render_value(v) -> str:
    try:
        import pandas as pd
        if isinstance(v, pd.DataFrame):
            return f"DataFrame {v.shape}:\n{F._df_to_md(v.head(_MAX_ROWS_OUT))}"
        if isinstance(v, pd.Series):
            return "Series:\n" + v.head(_MAX_ROWS_OUT).to_string()
    except Exception:
        pass
    return repr(v)


def _offending_line(exc: BaseException, src: str) -> str:
    tb, lineno = exc.__traceback__, None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<run_python>":
            lineno = tb.tb_lineno
        tb = tb.tb_next
    if lineno:
        lines = src.splitlines()
        if 1 <= lineno <= len(lines):
            return f"line {lineno}: {lines[lineno - 1].strip()}"
    return ""


_CODING_RULES = """HOW TO WRITE CODE FOR THE SCRIPTER — read this BEFORE writing any Python.

You are NOT in ChatGPT's code interpreter. There is NO /mnt/data, NO internet, NO
pip, and NO `import`. Code runs in a small restricted in-process sandbox and can
read the uploaded files ONLY through the helper functions listed below.

── HOW TO CALL (exact JSON) ───────────────────────────────────────────────
Emit ONE tool call whose name is EXACTLY  run_python  and whose arguments is an
OBJECT with a single "code" key holding your whole script as ONE string:
  {"name": "run_python", "arguments": {"code": "rows = load_rows('sales.xlsx')\\nprint(len(rows))"}}
Do NOT call `python`, `python_user_visible`, or `python_user_visible.exec` — those
tools DO NOT EXIST here and will just error. The ONLY code tool is run_python, and
its argument is {"code": "..."} (an object), never a bare string.

── IMPORTS: NONE — use the pre-provided names ─────────────────────────────
Never write `import` (it is rejected). These are ALREADY available by name —
just use them directly:
  pd  -- safe pandas subset (pd.DataFrame/Series/concat/merge/to_datetime/to_numeric/isna…)
  re · math · statistics · collections · datetime · json
e.g.  re.findall(r'\\d+', text)  ·  collections.Counter(items)  ·  math.sqrt(x).
You do NOT need (and cannot import) os, sys, csv, openpyxl, requests, etc.

── READ DATA (the only way to touch files; reference files by BARE name) ───
  load_table(filename, sheet=None) -> pandas DataFrame
  load_rows(filename, sheet=None)  -> list[dict]   (one dict per row)
  sql(filename, query)             -> DataFrame     (read-only SELECT; each sheet is a table)
  read_doc(filename)               -> str           (PDF/Word/PPT/txt as markdown)
  list_files()                     -> [filenames]
  list_sheets(filename)            -> [sheet names]
There is NO pd.read_excel / pd.read_csv / open(). Use load_table('file.xlsx','Sheet').

── FORBIDDEN (your script is REJECTED before running if it uses any) ───────
  import · open · eval · exec · getattr · any __dunder__ attribute ·
  df.to_csv/to_excel/to_pickle… (no file writing) · df.query / df.eval.
  Time limit ~8 s — keep loops bounded; prefer one sql() or a vectorised pandas
  op over a giant Python loop.

── RETURN YOUR RESULTS ────────────────────────────────────────────────────
  record_answer(question, answer)  -- record EACH sub-answer (it goes into your report)
  result = <value>                 -- shown back to you
  print(...)                       -- captured (bounded), for checking
After run_python returns, write your normal short reply stating the answers.

── EXAMPLE (our style — no imports, load_table, record_answer) ────────────
  orders = load_table('q2_sales_report.xlsx', 'Orders')
  reps   = load_table('q2_sales_report.xlsx', 'Reps')
  closed = orders[orders['status'] == 'closed']
  by_rep = closed.groupby('rep')['amount'].sum()
  for rep, total in by_rep.items():
      quota = reps.loc[reps['rep'] == rep, 'quota'].iloc[0]
      record_answer(rep + ' attainment %', round(100 * total / quota, 1))
  result = 'done'
"""


def python_coding_rules() -> ToolChunk:
    """The Python coding contract for code mode — CALL THIS FIRST, before writing
    any script, and follow it exactly. It tells you the ONE correct way to run
    Python here: the exact run_python tool-call JSON format, that there are NO
    imports and NO /mnt/data (this is a restricted sandbox, NOT ChatGPT's code
    interpreter), which data-reading helpers to use (load_table/load_rows/sql/
    read_doc), what is forbidden, and how to return your answers. Never call
    `python`/`python_user_visible` — they don't exist; use run_python per these
    rules.
    """
    return _ok(_CODING_RULES)


# ===========================================================================
#  the tool factory
# ===========================================================================
def make_run_python(child: str):
    """Build a ``run_python`` tool bound to ``child`` so its in-sandbox
    record_answer writes to that child's active task (same pattern as
    findings.make_record_answer). Returned as a plain function for FunctionTool."""

    recorder = _findings.make_record_answer(child)

    def run_python(code: str = "") -> ToolChunk:
        """Run ONE short Python script to answer a question that needs real
        procedural logic over the uploaded files — a loop, a cross-file join,
        per-row conditionals — that the excel_*/query_data tools can't express in
        one call. ESCALATION ONLY: for a plain count/total/filter use
        excel_query or excel_aggregate instead.

        Call python_coding_rules() FIRST (or run_python() with no code) to get the
        exact rules + the required tool-call JSON format. Your script reads data
        via load_table/load_rows/sql/read_doc (read-only, sandboxed), computes in
        plain Python/pandas, and returns findings via record_answer(question,
        answer) and/or a `result` variable. No imports, no file writing, ~8s limit.

        Args:
            code (str): The Python script to run. Leave EMPTY to get the coding
                rules and the required call format.
        """
        if not (code or "").strip() or code.strip().lower() in ("help", "?"):
            return _ok(_CODING_RULES)

        bad = _validate(code)
        if bad:
            return _err("rejected before running — " + bad +
                        "\nCall run_python() with no code to see the API + rules.")
        try:
            code_obj = compile(code, "<run_python>", "exec")
        except SyntaxError as exc:
            return _err(f"SyntaxError: {exc.msg} (line {exc.lineno})")

        # --- build the per-call sandbox namespace ---------------------------
        out = io.StringIO()

        def _print(*args, **kw):
            kw.pop("file", None)
            sep, end = kw.get("sep", " "), kw.get("end", "\n")
            out.write(sep.join(str(a) for a in args) + end)
            if out.tell() > _MAX * 4:           # stop a runaway print loop early
                raise RuntimeError("too much output — print less / aggregate.")

        calls = {"n": 0}

        def _capped(fn):
            def wrapper(*a, **k):
                calls["n"] += 1
                if calls["n"] > _MAX_TOOL_CALLS:
                    raise RuntimeError(
                        f"exceeded {_MAX_TOOL_CALLS} data-helper calls — load "
                        "once and reuse, or aggregate with a single sql() call.")
                return fn(*a, **k)
            wrapper.__name__ = getattr(fn, "__name__", "helper")
            return wrapper

        recorded = {"n": 0}

        def record_answer(question, answer, note=""):
            recorder(str(question), str(answer), str(note))
            recorded["n"] += 1

        emitted: dict = {}

        def emit(label, value):
            emitted[str(label)] = value

        builtins = dict(_SAFE_BUILTINS)
        builtins["print"] = _print
        ns: dict = {"__builtins__": builtins, "pd": _make_pd_shim(),
                    "record_answer": record_answer, "emit": emit}
        ns.update(_SAFE_MODULES)          # re / math / statistics / collections / datetime / json
        for name, fn in _make_data_api().items():
            ns[name] = _capped(fn)

        # --- run under the limits -------------------------------------------
        exc = _run_with_limits(code_obj, ns, _TIMEOUT_S)
        if exc is not None:
            where = _offending_line(exc, code)
            tail = f"\n  at {where}" if where else ""
            return _err(f"{type(exc).__name__}: {exc}{tail}\n"
                        "(fix the script and call run_python again.)")

        # --- assemble the (bounded) result ----------------------------------
        sections = []
        printed = out.getvalue().rstrip()
        if printed:
            sections.append("output:\n" + printed)
        result = ns.get("result")
        if result is not None:
            sections.append("result = " + _render_value(result))
        if emitted:
            rendered = "; ".join(f"{k} = {_render_value(v)}" for k, v in emitted.items())
            sections.append("emitted: " + rendered)
        if recorded["n"]:
            summ = _findings.findings_summary(_findings._active.get(child))
            sections.append(f"recorded {recorded['n']} answer(s)."
                            + (f" ledger: {summ}" if summ else ""))
        if not sections:
            sections.append("Script ran but returned nothing. Set `result`, "
                            "print(...), or call record_answer(question, answer).")
        # Truthful action log: the footer only sees one 'run_python' tool call,
        # so we surface what the script actually did here.
        sections.append(f"[code actions] data-helper calls: {calls['n']}")
        return _chunk("\n\n".join(sections))

    run_python.__name__ = "run_python"
    return run_python
