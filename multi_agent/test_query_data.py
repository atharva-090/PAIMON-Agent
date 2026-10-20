"""Fast, deterministic tests for query_data (DuckDB SQL) -- NO LLM.

Builds a real two-sheet workbook in a temp uploads folder, then asserts the
behaviour that matters:

  * query_data(filename) with no sql lists every sheet as a table + its columns,
  * a CROSS-SHEET JOIN + GROUP BY ... HAVING returns the right aggregated rows
    (the whole point of the tool -- something excel_aggregate can't express),
  * the read-only guard rejects mutations / multi-statement SQL,
  * the sandbox blocks file-system access (enable_external_access=false),
  * a bad table/column returns the schema so the brain can self-correct.

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_query_data
"""

from __future__ import annotations

import os
import sys
import tempfile

import file_tools as F


def _text(chunk) -> str:
    return chunk.content[0].text


def _make_fixture(d: str) -> None:
    import pandas as pd
    # reps: 2 West + 2 East. sales: per-rep amounts.
    reps = pd.DataFrame({
        "rep_id": [1, 2, 3, 4],
        "region": ["West", "West", "East", "East"],
        "name": ["Ann", "Bob", "Cara", "Dan"],
    })
    sales = pd.DataFrame({
        "rep_id": [1, 1, 2, 3, 3, 4, 4],
        "amount": [100, 200, 50, 300, 400, 500, 600],
    })
    # Region totals: West = 100+200+50 = 350; East = 300+400+500+600 = 1800.
    with pd.ExcelWriter(os.path.join(d, "company.xlsx")) as xw:
        reps.to_excel(xw, sheet_name="Reps", index=False)
        sales.to_excel(xw, sheet_name="Sales", index=False)


def _run() -> int:
    ok = True
    tmp = tempfile.mkdtemp(prefix="hermes_sql_")
    F.UPLOADS_DIR = tmp
    F._CACHE_DIR = os.path.join(tmp, ".cache")
    _make_fixture(tmp)

    # --- schema view (no sql) lists both tables + columns --------------------
    sch = _text(F.query_data("company.xlsx"))
    if ("reps" not in sch.lower() or "sales" not in sch.lower()
            or "region" not in sch or "amount" not in sch):
        print(f"[FAIL] no-sql schema view missing tables/columns:\n{sch[:300]}")
        ok = False
    else:
        print("[ok] query_data(no sql): lists every sheet as a table + columns")

    # --- cross-sheet JOIN + GROUP BY ... HAVING ------------------------------
    # Only East (1800) clears HAVING > 400; West (350) must be excluded.
    join_sql = ("SELECT r.region, SUM(s.amount) AS total "
                "FROM sales s JOIN reps r ON s.rep_id = r.rep_id "
                "GROUP BY r.region HAVING SUM(s.amount) > 400 ORDER BY total DESC")
    res = _text(F.query_data("company.xlsx", join_sql))
    if "East" not in res or "1800" not in res or "West" in res:
        print(f"[FAIL] join+HAVING wrong (want East/1800, no West):\n{res[:300]}")
        ok = False
    else:
        print("[ok] cross-sheet JOIN + GROUP BY ... HAVING")

    # --- DISTINCT aggregate (a plain power-tool can't do COUNT DISTINCT) -----
    nd = _text(F.query_data("company.xlsx",
                            "SELECT COUNT(DISTINCT region) AS n FROM reps"))
    if "2" not in nd:
        print(f"[FAIL] COUNT(DISTINCT region) should be 2:\n{nd[:200]}")
        ok = False
    else:
        print("[ok] COUNT(DISTINCT ...) returns 2")

    # --- read-only guard: mutations + multi-statement are rejected -----------
    guard_ok = True
    for bad in ("DROP TABLE sales",
                "INSERT INTO sales VALUES (9, 9)",
                "UPDATE sales SET amount = 0",
                "SELECT 1; DROP TABLE sales",
                "CREATE TABLE x AS SELECT * FROM sales"):
        r = _text(F.query_data("company.xlsx", bad))
        if not r.startswith("ERROR"):
            print(f"[FAIL] read-only guard let through: {bad!r}\n{r[:160]}")
            guard_ok = ok = False
    if guard_ok:
        print("[ok] read-only guard rejects mutations + multi-statement SQL")

    # --- sandbox: no file-system access (external access disabled) -----------
    evil = _text(F.query_data("company.xlsx",
                              "SELECT * FROM read_csv_auto('config.py')"))
    if not evil.startswith("ERROR"):
        print(f"[FAIL] file-system access NOT blocked:\n{evil[:200]}")
        ok = False
    else:
        print("[ok] sandbox blocks read_csv_auto (no file-system access)")

    # --- bad column -> SQL error WITH the schema for self-correction ---------
    badcol = _text(F.query_data("company.xlsx", "SELECT nope FROM sales"))
    if not badcol.startswith("ERROR") or "amount" not in badcol:
        print(f"[FAIL] bad column should error AND echo real columns:\n{badcol[:300]}")
        ok = False
    else:
        print("[ok] bad column errors and returns the real schema")

    # --- non-spreadsheet is refused -----------------------------------------
    import pandas as pd  # noqa: F401  (fixture builder already imported it)
    with open(os.path.join(tmp, "notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("hello")
    notxt = _text(F.query_data("notes.txt", "SELECT 1"))
    if not notxt.startswith("ERROR") or "spreadsheet" not in notxt:
        print(f"[FAIL] query_data should refuse a .txt:\n{notxt[:200]}")
        ok = False
    else:
        print("[ok] query_data refuses non-spreadsheet files")

    print("\n" + ("[PASS] query_data: SQL joins/HAVING/DISTINCT work, read-only "
                  "guard + sandbox hold, errors echo the schema."
                  if ok else "[FAIL] see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_run())
