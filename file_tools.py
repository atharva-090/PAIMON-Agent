"""File-reading tools for the Hermes agent (used by the swarm's scripter).

Lets the agent read and query files a user drops into ``data/uploads/`` --
Excel, CSV, PDF, Word, PowerPoint, Markdown, plain text. The design is shaped by
two facts about our setup:

  * The brain is ChatGPT's web UI driven through the mock, so it has a tight
    prompt budget and is a weak instruction-follower. We therefore NEVER paste a
    whole file at it. Instead the agent QUERIES the file through small tools and
    only the relevant slice comes back (truncated to FILE_MAX_OUTPUT). This is
    the same "tool-based querying beats full-text dump" approach production
    agent frameworks use, minus the embeddings/RAG (overkill + hallucination
    risk for a weak brain).

  * Spreadsheets and documents want different handling:
      - Excel/CSV  -> pandas: show the schema (columns + row count), then peek
        (first 10 + last 10 rows when there are more than 20), search, or filter.
        Big sheets never hit the prompt in full. For questions the point-tools
        can't express in one shot -- cross-sheet JOINs, GROUP BY ... HAVING,
        window functions -- query_data runs a READ-ONLY DuckDB SELECT with each
        sheet exposed as a table (in-memory, no file-system access).
      - PDF/Word/PPT -> converted ONCE to markdown via markitdown (cached under
        uploads/.cache/), then keyword-searched. No embeddings.

Every tool returns a ``ToolChunk`` and is READ-ONLY. All file access is confined
to ``UPLOADS_DIR`` -- absolute paths and ``..`` traversal are rejected.

Tools: list_uploaded_files, excel_info, excel_peek, excel_search, excel_filter,
excel_query (compound multi-column conditions), excel_aggregate (sum/mean/count/
min/max with optional group-by + where), query_data (read-only DuckDB SQL --
joins / HAVING / windows the point-tools can't express), doc_read, doc_search.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, ToolChunk

import config

UPLOADS_DIR = config.UPLOADS_DIR
_CACHE_DIR = os.path.join(UPLOADS_DIR, ".cache")
_MAX = config.FILE_MAX_OUTPUT

# How many rows we ever return at once, and the head/tail peek sizes. Keeping
# these small is what stops a 50k-row sheet from blowing the prompt.
_ROW_CAP = 20
_PEEK_HEAD = 10
_PEEK_TAIL = 10
# For the COUNT-bearing tools (query/filter/search) we show only a TINY sample.
# The weak brain was counting/summing the visible rows by eye instead of trusting
# the exact COUNT the tool already computed -- a 5-row sample removes the
# temptation while still showing what the rows look like.
_SAMPLE_CAP = 5
# query_data (SQL) returns a result SET, not a count -- cap the rows we render so
# a wide SELECT can't blow the prompt (chars are still bounded by _MAX on top).
_SQL_ROW_CAP = 50
# Whole-word tokens that must never appear in a query_data SQL statement. The
# real containment is (a) a single statement that must start with SELECT/WITH and
# (b) enable_external_access=false on a throwaway in-memory connection; this list
# is belt-and-suspenders that also gives the brain a clear "read-only" message
# instead of a raw engine error. (No scalar SQL functions are in here -- e.g.
# replace()/list_*() stay allowed.)
_SQL_FORBIDDEN = (
    "attach", "detach", "copy", "install", "load", "pragma", "call", "export",
    "import", "vacuum", "checkpoint", "insert", "update", "delete", "drop",
    "alter", "create", "truncate",
)

# Extension -> family. Spreadsheets go through pandas; documents through
# markitdown. (markitdown also handles md/txt/html, but we read those directly.)
_SPREADSHEET_EXTS = {".xlsx", ".xls", ".xlsm", ".csv"}
_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".md", ".markdown",
             ".txt", ".html", ".htm", ".json", ".xml", ".rtf"}


# ===========================================================================
#  helpers
# ===========================================================================
def _chunk(text: str) -> ToolChunk:
    if len(text) > _MAX:
        text = text[:_MAX] + f"\n…[truncated — {len(text)} chars total; narrow " \
                             "your query (search/filter/peek) to see the rest]"
    return ToolChunk(content=[TextBlock(type="text", text=text)])


def _ok(obj) -> ToolChunk:
    return _chunk(obj if isinstance(obj, str)
                  else json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _err(msg: str) -> ToolChunk:
    return _chunk(f"ERROR: {msg}")


def _resolve(filename: str) -> str:
    """Resolve ``filename`` to an absolute path INSIDE UPLOADS_DIR, or raise.

    The agent passes a bare name (or a name with subfolders) like 'report.xlsx'
    or 'uploads/report.xlsx'; we strip a leading 'uploads/' for convenience and
    refuse anything that escapes the drop folder."""
    name = (filename or "").strip().replace("\\", "/").lstrip("/")
    for prefix in ("uploads/", "data/uploads/"):
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
    base = os.path.realpath(UPLOADS_DIR)
    full = os.path.realpath(os.path.join(base, name))
    if full != base and not full.startswith(base + os.sep):
        raise ValueError(f"'{filename}' is outside the uploads folder")
    if not os.path.isfile(full):
        raise FileNotFoundError(
            f"no file '{name}' in the uploads folder. Call list_uploaded_files() "
            "to see what's available.")
    return full


def _df_to_md(df) -> str:
    """Render a DataFrame as a markdown table; fall back to plain text if the
    tabulate backend isn't available."""
    show = df.fillna("")
    try:
        return show.to_markdown(index=False)
    except Exception:
        return show.to_string(index=False)


def _load_table(full: str, sheet: Optional[str]):
    """Load a spreadsheet/CSV into a DataFrame. Returns (df, sheet_label).
    Raises ValueError with the available sheet names on a bad sheet."""
    import pandas as pd
    ext = os.path.splitext(full)[1].lower()
    if ext == ".csv":
        return pd.read_csv(full), "(csv)"
    xls = pd.ExcelFile(full)
    names = xls.sheet_names
    if sheet is None:
        target = names[0]
    elif sheet in names:
        target = sheet
    else:
        raise ValueError(f"no sheet '{sheet}'. Available sheets: {names}")
    return pd.read_excel(xls, sheet_name=target), target


def _fmt_num(v) -> str:
    """Render an aggregate value compactly: whole floats as ints, else 2 dp."""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.2f}"
    except (TypeError, ValueError):
        return str(v)


def _count_result(count: int, total: int, desc: str, hits, label: str) -> ToolChunk:
    """Format a COUNT-first result: the exact count is the answer, with only a
    tiny labelled sample below it. This stops the brain eyeballing/summing the
    visible rows instead of using the count the tool already computed."""
    head = (f"COUNT = {count}   (rows where {desc}; out of {total} total rows "
            f"in sheet {label})")
    if count == 0:
        return _ok(head)
    note = (f"^^ Use this COUNT ({count}) as your answer. The {min(count, _SAMPLE_CAP)} "
            "row(s) below are ONLY a sample so you can see the shape -- do NOT "
            "count or sum them by eye; the COUNT above is exact.")
    return _chunk(f"{head}\n{note}\n\nsample:\n{_df_to_md(hits.head(_SAMPLE_CAP))}")


# ===========================================================================
#  discovery
# ===========================================================================
def list_uploaded_files() -> ToolChunk:
    """List the files available to read in the uploads folder.

    Call this FIRST so you know the exact filenames (and their types) before
    reading. Spreadsheets (xlsx/csv) are read with the excel_* tools; documents
    (pdf/docx/pptx/txt/md) with the doc_* tools.
    """
    try:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        rows = []
        for name in sorted(os.listdir(UPLOADS_DIR)):
            full = os.path.join(UPLOADS_DIR, name)
            if name.startswith(".") or not os.path.isfile(full):
                continue
            ext = os.path.splitext(name)[1].lower()
            kind = ("spreadsheet" if ext in _SPREADSHEET_EXTS
                    else "document" if ext in _DOC_EXTS else "other")
            rows.append({
                "filename": name,
                "type": ext.lstrip(".") or "?",
                "category": kind,
                "size_kb": round(os.path.getsize(full) / 1024, 1),
                "read_with": ("excel_* tools" if kind == "spreadsheet"
                              else "doc_* tools" if kind == "document"
                              else "unsupported"),
            })
        if not rows:
            return _ok(f"The uploads folder is empty ({UPLOADS_DIR}). Drop files "
                       "there, then call this again.")
        return _ok({"uploads_dir": UPLOADS_DIR, "files": rows})
    except Exception as exc:
        return _err(f"could not list uploads: {type(exc).__name__}: {exc}")


# ===========================================================================
#  spreadsheets (pandas)
# ===========================================================================
def excel_info(filename: str, sheet: str = "") -> ToolChunk:
    """Show a spreadsheet's schema: its sheet names, and for one sheet the
    columns, their data types, and the total row count. Call this BEFORE peeking
    or filtering so you use real column names (never invent them).

    Args:
        filename (str): A spreadsheet in the uploads folder (.xlsx/.xls/.csv).
        sheet (str): Optional sheet name; defaults to the first sheet.
    """
    try:
        full = _resolve(filename)
        if os.path.splitext(full)[1].lower() not in _SPREADSHEET_EXTS:
            return _err(f"'{filename}' isn't a spreadsheet; use doc_read/doc_search.")
        import pandas as pd
        all_sheets = (["(csv)"] if full.lower().endswith(".csv")
                      else pd.ExcelFile(full).sheet_names)
        df, label = _load_table(full, sheet or None)
        cols = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]
        return _ok({
            "filename": os.path.basename(full),
            "all_sheets": all_sheets,
            "sheet": label,
            "rows": int(len(df)),
            "columns": cols,
            "hint": "use excel_peek to see sample rows, excel_filter to select "
                    "rows by a column, or excel_search for a keyword.",
        })
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not read '{filename}': {type(exc).__name__}: {exc}")


def excel_peek(filename: str, sheet: str = "") -> ToolChunk:
    """Preview a sheet's rows. If the sheet has more than 20 rows, returns the
    FIRST 10 and LAST 10 rows (so you see its shape without loading it all);
    otherwise returns every row. Use excel_filter/excel_search for specific rows.

    Args:
        filename (str): A spreadsheet in the uploads folder (.xlsx/.xls/.csv).
        sheet (str): Optional sheet name; defaults to the first sheet.
    """
    try:
        full = _resolve(filename)
        if os.path.splitext(full)[1].lower() not in _SPREADSHEET_EXTS:
            return _err(f"'{filename}' isn't a spreadsheet; use doc_read/doc_search.")
        df, label = _load_table(full, sheet or None)
        n = len(df)
        header = f"{os.path.basename(full)} · sheet {label} · {n} rows × " \
                 f"{len(df.columns)} cols"
        if n > _PEEK_HEAD + _PEEK_TAIL:
            head_md = _df_to_md(df.head(_PEEK_HEAD))
            tail_md = _df_to_md(df.tail(_PEEK_TAIL))
            body = (f"{header}\n\nFIRST {_PEEK_HEAD} ROWS:\n{head_md}\n\n"
                    f"… {n - _PEEK_HEAD - _PEEK_TAIL} rows hidden …\n\n"
                    f"LAST {_PEEK_TAIL} ROWS:\n{tail_md}")
        else:
            body = f"{header}\n\n{_df_to_md(df)}"
        return _chunk(body)
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not read '{filename}': {type(exc).__name__}: {exc}")


def excel_search(filename: str, query: str, sheet: str = "") -> ToolChunk:
    """Find rows where ANY cell contains ``query`` (case-insensitive). Returns up
    to 20 matching rows plus the total match count.

    Args:
        filename (str): A spreadsheet in the uploads folder (.xlsx/.xls/.csv).
        query (str): Text to look for in any cell.
        sheet (str): Optional sheet name; defaults to the first sheet.
    """
    try:
        if not (query or "").strip():
            return _err("query is empty")
        full = _resolve(filename)
        df, label = _load_table(full, sheet or None)
        mask = df.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False, regex=False)
        ).any(axis=1)
        hits = df[mask]
        return _count_result(len(hits), len(df), f"any cell contains {query!r}",
                             hits, label)
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not search '{filename}': {type(exc).__name__}: {exc}")


def excel_filter(filename: str, column: str, op: str, value: str,
                 sheet: str = "") -> ToolChunk:
    """Select rows where a column meets a condition. Validates the column name
    and tells you the real columns if you got it wrong (never guess columns --
    check excel_info first). Returns up to 20 matching rows + the total count.

    Args:
        filename (str): A spreadsheet in the uploads folder (.xlsx/.xls/.csv).
        column (str): The exact column name to test.
        op (str): One of ==, !=, >, <, >=, <=, contains.
        value (str): The value to compare against (numbers compared numerically
            when the column is numeric).
        sheet (str): Optional sheet name; defaults to the first sheet.
    """
    try:
        full = _resolve(filename)
        df, label = _load_table(full, sheet or None)
        if column not in df.columns:
            return _err(f"no column '{column}' in sheet {label}. Available "
                        f"columns: {[str(c) for c in df.columns]}")
        allowed = {"==", "!=", ">", "<", ">=", "<=", "contains"}
        if op not in allowed:
            return _err(f"unsupported op '{op}'. Use one of: {sorted(allowed)}")
        col = df[column]
        if op == "contains":
            mask = col.astype(str).str.contains(str(value), case=False,
                                                na=False, regex=False)
        else:
            import pandas as pd
            cmp_val, cmp_col = value, col
            num_col = pd.to_numeric(col, errors="coerce")
            num_val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if num_val == num_val and num_col.notna().any():  # value & column numeric
                cmp_col, cmp_val = num_col, num_val
            else:
                cmp_col = col.astype(str)
                cmp_val = str(value)
            ops = {
                "==": cmp_col == cmp_val, "!=": cmp_col != cmp_val,
                ">": cmp_col > cmp_val, "<": cmp_col < cmp_val,
                ">=": cmp_col >= cmp_val, "<=": cmp_col <= cmp_val,
            }
            mask = ops[op].fillna(False)
        hits = df[mask]
        return _count_result(len(hits), len(df), f"{column} {op} {value!r}",
                             hits, label)
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not filter '{filename}': {type(exc).__name__}: {exc}")


def excel_query(filename: str, where: str, sheet: str = "") -> ToolChunk:
    """Select rows matching a COMPOUND condition across one or more columns --
    the multi-condition counterpart to excel_filter (which tests a single
    column). Returns the total match COUNT plus up to 20 sample rows. This is
    how you answer 'how many rows where X AND Y'.

    Args:
        filename (str): A spreadsheet in the uploads folder (.xlsx/.xls/.csv).
        where (str): A pandas query expression over the REAL column names (run
            excel_info first). Quote text values with single quotes; combine
            conditions with `and` / `or`; wrap a column name that has spaces in
            backticks. Examples:
              "discount_pct > 15"
              "amount > 5000 and approval == 'none'"
              "rep == 'Chen' and status == 'closed'"
        sheet (str): Optional sheet name; defaults to the first sheet.
    """
    try:
        if not (where or "").strip():
            return _err("where expression is empty")
        full = _resolve(filename)
        if os.path.splitext(full)[1].lower() not in _SPREADSHEET_EXTS:
            return _err(f"'{filename}' isn't a spreadsheet; use doc_search.")
        df, label = _load_table(full, sheet or None)
        try:
            hits = df.query(where)
        except Exception as exc:
            return _err(
                f"could not evaluate where={where!r}: {type(exc).__name__}: {exc}. "
                f"Real columns: {[str(c) for c in df.columns]}. Quote text values "
                "with single quotes (e.g. approval == 'none') and combine with "
                "and/or.")
        return _count_result(len(hits), len(df), where, hits, label)
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not query '{filename}': {type(exc).__name__}: {exc}")


def excel_aggregate(filename: str, column: str, func: str, sheet: str = "",
                    group_by: str = "", where: str = "") -> ToolChunk:
    """Compute an aggregate over a column -- sum, mean, count, min, max, median,
    or nunique -- optionally GROUPED BY another column and/or pre-filtered with a
    WHERE expression. This is how you get totals the row tools can't show: a
    grand total, an average, or a per-group breakdown over ALL rows (not just
    the 20 a peek/filter returns).

    Args:
        filename (str): A spreadsheet in the uploads folder (.xlsx/.xls/.csv).
        column (str): The column to aggregate (for func='count' it's ignored --
            count counts matching rows).
        func (str): One of sum, mean, count, min, max, median, nunique.
        sheet (str): Optional sheet name; defaults to the first sheet.
        group_by (str): Optional column to group by, e.g. 'region' -> one total
            per region (sorted high to low).
        where (str): Optional pandas filter applied BEFORE aggregating, e.g.
            "rep == 'Chen'" or "amount > 5000 and approval == 'none'".
    """
    try:
        full = _resolve(filename)
        if os.path.splitext(full)[1].lower() not in _SPREADSHEET_EXTS:
            return _err(f"'{filename}' isn't a spreadsheet; use doc_search.")
        df, label = _load_table(full, sheet or None)
        func = (func or "").strip().lower()
        func = {"avg": "mean", "average": "mean", "total": "sum"}.get(func, func)
        if func not in {"sum", "mean", "count", "min", "max", "median", "nunique"}:
            return _err(f"unsupported func '{func}'. Use one of: sum, mean, "
                        "count, min, max, median, nunique.")
        if (where or "").strip():
            try:
                df = df.query(where)
            except Exception as exc:
                return _err(f"could not evaluate where={where!r}: "
                            f"{type(exc).__name__}: {exc}. Real columns: "
                            f"{[str(c) for c in df.columns]}.")
        nrows = len(df)
        if func != "count" and column not in df.columns:
            return _err(f"no column '{column}' in sheet {label}. Available "
                        f"columns: {[str(c) for c in df.columns]}")
        if group_by and group_by not in df.columns:
            return _err(f"no group_by column '{group_by}'. Available columns: "
                        f"{[str(c) for c in df.columns]}")
        import pandas as pd

        def _agg(series):
            if func == "count":
                return len(series)
            if func == "nunique":
                return int(series.nunique())
            num = pd.to_numeric(series, errors="coerce")
            return {"sum": num.sum(), "mean": num.mean(), "median": num.median(),
                    "min": num.min(), "max": num.max()}[func]

        where_note = f", where {where}" if (where or "").strip() else ""
        if group_by:
            if func == "count":
                res = df.groupby(group_by).size()
            else:
                res = df.groupby(group_by)[column].apply(_agg)
            res = res.sort_values(ascending=False)
            body = "\n".join(f"  {k}: {_fmt_num(v)}" for k, v in res.items())
            return _ok(f"{func}({column or 'rows'}) by {group_by} "
                       f"(sheet {label}, {nrows} rows{where_note}):\n{body}")
        value = nrows if func == "count" else _agg(df[column])
        return _ok(f"{func}({column or 'rows'}) = {_fmt_num(value)}  "
                   f"(sheet {label}, over {nrows} rows{where_note})")
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not aggregate '{filename}': {type(exc).__name__}: {exc}")


# ===========================================================================
#  SQL over a workbook (DuckDB, read-only)
# ===========================================================================
def _strip_sql_comments(sql: str) -> str:
    """Remove /* block */ and -- line comments so the guard can't be slipped a
    forbidden keyword hidden behind a comment."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def _check_select(sql: str) -> Optional[str]:
    """Return None if ``sql`` is a single read-only SELECT/WITH statement, else a
    human-readable reason it's rejected. Defence in depth on top of the sandboxed
    in-memory connection (enable_external_access=false)."""
    body = _strip_sql_comments(sql).strip().rstrip(";").strip()
    if not body:
        return "empty SQL -- pass a SELECT, or call query_data(filename) with no " \
               "sql to see the tables first."
    if ";" in body:
        return "only a SINGLE read-only SELECT statement is allowed (found ';')."
    low = body.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return ("query_data is READ-ONLY: your SQL must start with SELECT or WITH "
                "(a CTE). No INSERT/UPDATE/CREATE/COPY/ATTACH etc.")
    for kw in _SQL_FORBIDDEN:
        if re.search(rf"\b{kw}\b", low):
            return (f"'{kw}' is not allowed -- query_data is read-only (SELECT "
                    "only) and has no file-system access. Rephrase as a SELECT.")
    return None


def _safe_table_name(label: str, taken: set) -> str:
    """Turn a sheet name into a safe, case-insensitive SQL identifier (snake_case),
    de-duplicating against names already used in this workbook."""
    base = re.sub(r"[^0-9a-zA-Z]+", "_", str(label).strip().lower()).strip("_")
    if not base:
        base = "sheet"
    if base[0].isdigit():
        base = "t_" + base
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    taken.add(name)
    return name


def _register_workbook(con, full: str) -> list:
    """Register every sheet of a workbook (or the single CSV) as a DuckDB view.
    Returns [(table_name, original_sheet_label, dataframe), ...] in sheet order."""
    import pandas as pd
    if full.lower().endswith(".csv"):
        stem = os.path.splitext(os.path.basename(full))[0]
        sheets = {stem: pd.read_csv(full)}
    else:
        sheets = pd.read_excel(full, sheet_name=None)  # ordered dict of DataFrames
    taken, out = set(), []
    for label, df in sheets.items():
        df = df.copy()
        df.columns = [str(c) for c in df.columns]  # SQL needs string identifiers
        tname = _safe_table_name(label, taken)
        con.register(tname, df)
        out.append((tname, str(label), df))
    return out


def _workbook_schema_text(full: str, tables: list) -> str:
    """Compact, brain-readable listing of the tables a query_data call exposes --
    one per sheet, with columns and types. Returned both on a no-sql call and
    appended to SQL errors so the brain can self-correct to real names."""
    lines = [f"{os.path.basename(full)} -- SQL tables (one per sheet; names are "
             "case-insensitive). Available tables:"]
    for tname, label, df in tables:
        aka = f'  (sheet "{label}")' if label.lower() != tname else ""
        cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        lines.append(f"  • {tname}{aka} — {len(df)} rows")
        lines.append(f"      columns: {cols}")
    return "\n".join(lines)


def _connect_sandboxed():
    """Open a throwaway in-memory DuckDB connection with external access OFF --
    the SINGLE place we configure the SQL sandbox (no file I/O, no extensions,
    no network). Shared by query_data and code_tools.sql so the containment
    settings can never drift between the two entry points."""
    import duckdb
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_external_access=false")
    return con


def _run_sql(full: str, sql: str):
    """Execute ONE read-only SELECT over a workbook's sheets and return a pandas
    DataFrame. Raises ValueError on a non-SELECT / guard rejection or a DuckDB
    error. This is the data-returning core that code mode's sql() helper calls;
    query_data wraps the same machinery but renders a ToolChunk + schema."""
    bad = _check_select(sql)
    if bad:
        raise ValueError(bad)
    con = _connect_sandboxed()
    try:
        _register_workbook(con, full)
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def query_data(filename: str, sql: str = "") -> ToolChunk:
    """Run a READ-ONLY SQL SELECT over a spreadsheet, with each sheet exposed as a
    table (powered by DuckDB). This is the power tool for questions the excel_*
    tools can't express in one shot: cross-sheet JOINs, GROUP BY ... HAVING,
    window functions, DISTINCT, sub-queries, multi-step CTEs.

    Two-step use:
      1. Call query_data(filename) with NO sql to see the table names (one per
         sheet) and each table's columns -- so you use real names, never guess.
      2. Call query_data(filename, sql) with a single SELECT over those tables.

    Only SELECT/WITH is allowed (no INSERT/UPDATE/CREATE/COPY/ATTACH) and there is
    no file-system access. Prefer the simple excel_* tools for a plain count or
    total; reach for query_data only when one SQL statement is genuinely the
    clearest way to get the answer.

    Args:
        filename (str): A spreadsheet in the uploads folder (.xlsx/.xls/.csv).
            ALL of its sheets become tables in the same query, so you can JOIN
            across sheets of one workbook.
        sql (str): A single read-only SELECT. Leave empty to list the tables and
            columns first. Table/column names are case-insensitive; wrap a name
            containing spaces in double quotes. Example:
              SELECT r.region, SUM(s.amount) AS total
              FROM sales s JOIN reps r ON s.rep_id = r.rep_id
              GROUP BY r.region HAVING SUM(s.amount) > 100000
    """
    try:
        full = _resolve(filename)
        if os.path.splitext(full)[1].lower() not in _SPREADSHEET_EXTS:
            return _err(f"'{filename}' isn't a spreadsheet; query_data works on "
                        ".xlsx/.xls/.csv. Use doc_read/doc_search for documents.")
        try:
            import duckdb
        except ImportError:
            return _err("duckdb isn't installed; use excel_query/excel_aggregate "
                        "instead.")
        # Sandboxed throwaway connection: no file I/O, no extensions, no network.
        # The registered views live only in this connection (see _connect_sandboxed).
        con = _connect_sandboxed()
        try:
            tables = _register_workbook(con, full)
            sql = (sql or "").strip()
            if not sql:
                return _ok(_workbook_schema_text(full, tables) +
                           "\n\nNow call query_data(filename, sql) with a single "
                           "read-only SELECT over these tables.")
            bad = _check_select(sql)
            if bad:
                return _err(bad)
            try:
                df = con.execute(sql).fetchdf()
            except Exception as exc:
                return _err(f"SQL error: {type(exc).__name__}: {exc}\n\n"
                            + _workbook_schema_text(full, tables))
            n = len(df)
            if n == 0:
                return _ok("Query OK — 0 rows matched.")
            head = (f"{n} row(s) returned"
                    + (f" (showing first {_SQL_ROW_CAP})" if n > _SQL_ROW_CAP else ""))
            return _chunk(f"{head}:\n{_df_to_md(df.head(_SQL_ROW_CAP))}")
        finally:
            con.close()
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not run SQL on '{filename}': {type(exc).__name__}: {exc}")


# ===========================================================================
#  documents (markitdown -> cached markdown)
# ===========================================================================
def _to_markdown_cached(full: str) -> str:
    """Convert a document to markdown, caching the result next to the source.
    Re-converts only when the source is newer than the cache. Raises a clear
    error if a PDF yields no extractable text (i.e. it's image-only / scanned)."""
    ext = os.path.splitext(full)[1].lower()
    # md/txt are already text -- read them straight, no conversion.
    if ext in (".md", ".markdown", ".txt"):
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache = os.path.join(_CACHE_DIR, os.path.basename(full) + ".md")
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(full):
        with open(cache, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    from markitdown import MarkItDown
    md = MarkItDown().convert(full).text_content or ""
    if ext == ".pdf" and not md.strip():
        raise ValueError(
            "this PDF has no extractable text — it looks scanned / image-only. "
            "OCR isn't supported yet; ask the user for a text-based version.")
    with open(cache, "w", encoding="utf-8") as fh:
        fh.write(md)
    return md


def doc_read(filename: str) -> ToolChunk:
    """Read a document (PDF/Word/PPT/Markdown/text) as markdown. Returns the
    start of the document plus its total length; for anything large, use
    doc_search to jump to the parts you need instead of reading it all.

    Args:
        filename (str): A document in the uploads folder.
    """
    try:
        full = _resolve(filename)
        if os.path.splitext(full)[1].lower() in _SPREADSHEET_EXTS:
            return _err(f"'{filename}' is a spreadsheet; use excel_info/excel_peek.")
        md = _to_markdown_cached(full)
        head = f"{os.path.basename(full)} · {len(md)} chars of markdown"
        if len(md) > _MAX:
            return _chunk(f"{head}\n(showing the first part; use "
                          f"doc_search('{os.path.basename(full)}', <keyword>) to "
                          f"find specific sections)\n\n{md}")
        return _chunk(f"{head}\n\n{md}")
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not read '{filename}': {type(exc).__name__}: {exc}")


def doc_search(filename: str, query: str, context_lines: int = 2) -> ToolChunk:
    """Search a document for a keyword and return the matching lines with a few
    lines of surrounding context (no embeddings -- a literal, case-insensitive
    match). Use this instead of doc_read for long documents.

    Args:
        filename (str): A document in the uploads folder.
        query (str): Text to look for.
        context_lines (int): Lines of context to show around each match (0-5).
    """
    try:
        if not (query or "").strip():
            return _err("query is empty")
        full = _resolve(filename)
        if os.path.splitext(full)[1].lower() in _SPREADSHEET_EXTS:
            return _err(f"'{filename}' is a spreadsheet; use excel_search.")
        md = _to_markdown_cached(full)
        lines = md.splitlines()
        ctx = max(0, min(int(context_lines), 5))
        q = query.lower()
        blocks, count, last_end = [], 0, -1
        for i, line in enumerate(lines):
            if q in line.lower():
                count += 1
                if len(blocks) >= 12:  # cap snippets so we stay in budget
                    continue
                start = max(0, i - ctx)
                end = min(len(lines), i + ctx + 1)
                if start <= last_end:  # merge with the previous overlapping block
                    blocks[-1][1] = end
                else:
                    blocks.append([start, end])
                last_end = end
        if not count:
            return _ok(f"No lines in {os.path.basename(full)} contain {query!r}.")
        out = [f"{count} match(es) for {query!r} in {os.path.basename(full)}:"]
        for start, end in blocks:
            snippet = "\n".join(f"{j+1:>4}: {lines[j]}" for j in range(start, end))
            out.append(snippet)
        if count > 12:
            out.append(f"(showing first 12 of {count} matches — refine the query)")
        return _chunk("\n\n".join(out))
    except (ValueError, FileNotFoundError) as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"could not search '{filename}': {type(exc).__name__}: {exc}")


# ===========================================================================
#  factory
# ===========================================================================
_FILE_FUNCS = [
    list_uploaded_files,
    excel_info, excel_peek, excel_search, excel_filter,
    excel_query, excel_aggregate, query_data,
    doc_read, doc_search,
]


def build_tools() -> list:
    """Return the file tools as AgentScope FunctionTools (all read-only)."""
    return [FunctionTool(f, is_read_only=True) for f in _FILE_FUNCS]
