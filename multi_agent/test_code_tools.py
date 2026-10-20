"""Fast, deterministic tests for code_tools.run_python (CODE MODE) -- NO LLM.

Builds a two-sheet workbook in a temp uploads folder, then exercises the
sandbox the way the security model promises:

  * a valid multi-step script computes a known cross-sheet answer and its
    record_answer calls land on the active task record (the findings ledger);
  * sql()/load_table/pd are wired and return real data;
  * the AST gate rejects imports, dunder access, open/eval/exec/getattr, the
    pandas file-writers (to_csv) and readers (pd.read_csv), df.query, format;
  * the settrace deadline kills `while True: pass`;
  * the data-helper call cap trips;
  * output is truncated to the cap;
  * NO file can be written (the attempts are blocked AND nothing appears on disk).

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_code_tools
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import file_tools as F
import code_tools as C
from multi_agent import findings
from multi_agent.bus import TaskRecord


def _text(chunk) -> str:
    return chunk.content[0].text


def _make_fixture(d: str) -> None:
    import pandas as pd
    reps = pd.DataFrame({
        "rep_id": [1, 2, 3, 4],
        "region": ["West", "West", "East", "East"],
        "name": ["Ann", "Bob", "Cara", "Dan"],
    })
    sales = pd.DataFrame({
        "rep_id": [1, 1, 2, 3, 3, 4, 4],
        "amount": [100, 200, 50, 300, 400, 500, 600],
    })
    # West = 100+200+50 = 350 ; East = 300+400+500+600 = 1800 ; grand = 2150.
    with pd.ExcelWriter(os.path.join(d, "company.xlsx")) as xw:
        reps.to_excel(xw, sheet_name="Reps", index=False)
        sales.to_excel(xw, sheet_name="Sales", index=False)


def _run() -> int:
    ok = True
    tmp = tempfile.mkdtemp(prefix="hermes_code_")
    F.UPLOADS_DIR = tmp
    F._CACHE_DIR = os.path.join(tmp, ".cache")
    _make_fixture(tmp)
    run = C.make_run_python("scripter")

    # --- cheat-sheet on empty code ------------------------------------------
    cs = _text(run(""))
    if "load_table" not in cs or "record_answer" not in cs:
        print(f"[FAIL] empty run_python should return the cheat-sheet:\n{cs[:200]}")
        ok = False
    else:
        print("[ok] run_python(): returns the API cheat-sheet")

    # --- python_coding_rules: the contract that anchors the brain -----------
    rules = _text(C.python_coding_rules())
    needs = ['"name": "run_python"', '"code"', "load_table", "python_user_visible",
             "import", "re", "collections"]
    missing = [s for s in needs if s not in rules]
    if missing:
        print(f"[FAIL] python_coding_rules missing {missing}:\n{rules[:200]}")
        ok = False
    else:
        print("[ok] python_coding_rules: JSON format + no-import + anti-reflex rules")

    # --- valid multi-step cross-sheet script + record_answer lands ----------
    rec = TaskRecord(id="t1", child="scripter", description="x")
    findings.set_active("scripter", rec)
    script = (
        "sales = load_table('company.xlsx', 'Sales')\n"
        "reps = load_table('company.xlsx', 'Reps')\n"
        "m = sales.merge(reps, on='rep_id')\n"
        "by_region = m.groupby('region')['amount'].sum()\n"
        "for region, tot in by_region.items():\n"
        "    record_answer(region + ' total', int(tot))\n"
        "result = int(by_region.sum())\n"
    )
    res = _text(run(script))
    findings.clear_active("scripter")
    got = {f["q"]: f["a"] for f in rec.findings}
    if (got.get("West total") != "350" or got.get("East total") != "1800"
            or "2150" not in res):
        print(f"[FAIL] cross-sheet script wrong. findings={got}\n{res[:300]}")
        ok = False
    else:
        print("[ok] valid script: cross-sheet merge + record_answer -> ledger")

    # --- sql() helper inside code mode returns real data --------------------
    sres = _text(run(
        "df = sql('company.xlsx', \"SELECT region, SUM(amount) t \"\n"
        "         \"FROM sales s JOIN reps r ON s.rep_id=r.rep_id \"\n"
        "         \"GROUP BY region HAVING SUM(amount) > 400\")\n"
        "result = df\n"))
    if "East" not in sres or "1800" not in sres or "West" in sres:
        print(f"[FAIL] sql() in code mode wrong (want East/1800 only):\n{sres[:300]}")
        ok = False
    else:
        print("[ok] sql() helper: join + HAVING inside code mode")

    # --- pre-injected modules (re / collections / math / statistics) work ---
    mres = _text(run(
        "txt = 'a1 b2 a3 c4 a5'\n"
        "nums = re.findall(r'[0-9]+', txt)\n"
        "first = collections.Counter(w[0] for w in txt.split())\n"
        "result = (len(nums), first['a'], round(math.sqrt(16), 1), "
        "statistics.mean([2, 4, 6]))\n"))
    if "5" not in mres or "3" not in mres or "4.0" not in mres:
        print(f"[FAIL] pre-injected modules not usable:\n{mres[:200]}")
        ok = False
    else:
        print("[ok] modules: re / collections / math / statistics usable (no import)")

    # --- AST gate rejects the dangerous forms -------------------------------
    rejects = {
        "import os": "import os",
        "from-import": "from sys import path",
        "open()": "open('x.txt', 'w')",
        "eval()": "eval('1+1')",
        "exec()": "exec('y=1')",
        "__import__": "__import__('os')",
        "dunder attr": "z = (1).__class__",
        "dunder mro": "z = ().__class__.__bases__",
        "bare __builtins__": "z = __builtins__",
        "getattr": "z = getattr(sql, 'x')",
        "df.to_csv (file write)": ("df = load_table('company.xlsx','Sales')\n"
                                   "df.to_csv('out.csv')\n"),
        "pd.read_csv (file read)": "df = pd.read_csv('company.xlsx')",
        "df.query (eval engine)": ("df = load_table('company.xlsx','Sales')\n"
                                   "df.query('amount > 100')\n"),
        "str.format escape": "z = '{0.__class__}'.format(sql)",
    }
    gate_ok = True
    for label, code in rejects.items():
        r = _text(run(code))
        if not r.startswith("ERROR"):
            print(f"[FAIL] AST gate let through {label!r}:\n{r[:160]}")
            gate_ok = ok = False
    if gate_ok:
        print(f"[ok] AST gate rejects all {len(rejects)} dangerous forms")

    # --- settrace deadline kills an infinite loop ---------------------------
    saved_to = C._TIMEOUT_S
    try:
        C._TIMEOUT_S = 2.0
        t0 = time.time()
        r = _text(run("while True:\n    pass\n"))
        dt = time.time() - t0
        if not r.startswith("ERROR") or "time limit" not in r or dt > 6:
            print(f"[FAIL] infinite loop not bounded (dt={dt:.1f}s):\n{r[:160]}")
            ok = False
        else:
            print(f"[ok] settrace deadline kills `while True` in {dt:.1f}s")
    finally:
        C._TIMEOUT_S = saved_to

    # --- data-helper call cap trips -----------------------------------------
    saved_cap = C._MAX_TOOL_CALLS
    try:
        C._MAX_TOOL_CALLS = 5
        r = _text(run("for i in range(50):\n    load_table('company.xlsx','Sales')\n"))
        if not r.startswith("ERROR") or "data-helper calls" not in r:
            print(f"[FAIL] tool-call cap not enforced:\n{r[:160]}")
            ok = False
        else:
            print("[ok] data-helper call cap trips")
    finally:
        C._MAX_TOOL_CALLS = saved_cap

    # --- output truncation respects the cap ---------------------------------
    saved_max = C._MAX
    try:
        C._MAX = 200
        r = _text(run("print('x' * 300)\n"))   # 300 < cap*4 (no runaway), > cap (trunc)
        if "truncated" not in r or len(r) > 400:
            print(f"[FAIL] output not truncated to the cap (len={len(r)}):\n{r[:120]}")
            ok = False
        else:
            print("[ok] output truncated to FILE_MAX_OUTPUT")
    finally:
        C._MAX = saved_max

    # --- NO file written: attempts blocked AND nothing on disk --------------
    before = set(os.listdir(tmp))
    for code in ("open('evil.txt', 'w')",
                 "df = load_table('company.xlsx','Sales')\ndf.to_csv('evil2.csv')\n"):
        run(code)
    after = set(os.listdir(tmp))
    if after - before:
        print(f"[FAIL] sandbox wrote file(s): {after - before}")
        ok = False
    elif os.path.exists("evil.txt") or os.path.exists("evil2.csv"):
        print("[FAIL] sandbox wrote a file into the cwd")
        ok = False
    else:
        print("[ok] no file written (open/to_csv blocked, disk untouched)")

    print("\n" + ("[PASS] code_tools: valid scripts compute + record, sql/pd wired, "
                  "AST gate holds, timeout + call-cap + output bounded, no file I/O."
                  if ok else "[FAIL] see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_run())
