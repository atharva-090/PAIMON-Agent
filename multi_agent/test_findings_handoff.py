"""Step 6: a teammate's REAL report reaches the next child.

The failure this fixes, from a real run:

    browser_agent  researched Muthoot (8 tool calls, goals verified on page)
    orchestrator   delegate_task(child=scripter, description=<582 chars>)
    scripter       fs_download(muthootfinance.com)  FAILED
    scripter       fs_download(hdfcbank.com)        ok -> read 1.3 MB of HTML
    scripter       write_document(...)

The scripter never saw the research, re-gathered the sources itself, got nothing
usable out of raw HTML and fell back to generic knowledge. The document it
produced mentioned Muthoot zero times and never made the individual /
non-individual distinction the user asked for.

The old design assumed the orchestrator would paste the report into the
description. It does not -- it paraphrases, and detail dies there.

Run:  .venv/Scripts/python.exe -m multi_agent.test_findings_handoff
"""
from __future__ import annotations

import multi_agent.swarm as S

FAILS: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILS.append(label)


def truthy(label: str, got) -> None:
    ok = bool(got)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        FAILS.append(label)


REPORT = ("Muthoot Finance verifies INDIVIDUALS with PAN, Aadhaar and a "
          "passport-size photo. NON-INDIVIDUALS (partnerships, companies) "
          "additionally need the partnership deed or certificate of "
          "incorporation, board resolution and GST registration. "
          "Source: muthootfinance.com/services/corporate-business-loan")

ROWS = [
    {"id": "tsk_research", "child": "browser_agent", "status": "done",
     "report": REPORT},
    {"id": "tsk_running", "child": "browser_agent", "status": "in_progress",
     "report": ""},
    {"id": "tsk_empty", "child": "task_manager", "status": "done", "report": ""},
]

# ------------------------------------------------------------ block assembly
print("\nfindings block")
block = S._findings_block(ROWS, ["tsk_research"])
truthy("header tells the child not to re-gather", "Do NOT re-gather" in block)
truthy("names the source child and task", "browser_agent" in block
       and "tsk_research" in block)
truthy("carries the REAL report text, verbatim",
       "certificate of incorporation" in block)
truthy("carries the distinction the run lost",
       "INDIVIDUALS" in block and "NON-INDIVIDUALS" in block)

check("unknown ids yield nothing", S._findings_block(ROWS, ["tsk_nope"]), "")
check("no ids yield nothing", S._findings_block(ROWS, []), "")
check("a done task with an EMPTY report is skipped",
      S._findings_block(ROWS, ["tsk_empty"]), "")
check("an unfinished task contributes nothing",
      S._findings_block(ROWS, ["tsk_running"]), "")

# Budgets: a runaway report must not eat the child's whole context.
huge = [{"id": "h", "child": "browser_agent", "status": "done",
         "report": "x" * 50000}]
b = S._findings_block(huge, ["h"])
truthy("one huge report is capped", len(b) < S._FINDINGS_PER_REPORT + 400)
truthy("and says it was truncated", "[truncated]" in b)

many = [{"id": f"t{i}", "child": "browser_agent", "status": "done",
         "report": "y" * 1500} for i in range(10)]
b = S._findings_block(many, [f"t{i}" for i in range(10)])
truthy("many reports respect the total budget",
       len(b) < S._FINDINGS_TOTAL + 400)

# ------------------------------------------------------------- auto-detection
print("\nauto-detection (reuses the sequencing guard's back-reference logic)")
dep = "Using the browser specialist's research, write the policy DOCX."
check("a back-reference finds the finished source",
      S._auto_finding_ids(dep, ROWS), ["tsk_research"])

selfcontained = "Write a DOCX titled Q3 Plan with three sections."
check("a self-contained delegation attaches nothing",
      S._auto_finding_ids(selfcontained, ROWS), [])

check("nothing to attach when the source has not finished",
      S._auto_finding_ids(dep, [ROWS[1]]), [])

# --------------------------------------------------- the guard still refuses
print("\nsequencing guard still refuses a premature delegation")
src = S._delegation_needs_earlier_output(dep, [ROWS[1]])
check("refuses while the source is still running", src, "browser_agent")
check("allows once the source is done",
      S._delegation_needs_earlier_output(dep, ROWS), "")

nudge = S._sequencing_nudge("scripter", "browser_agent")
truthy("nudge now points at use_findings_from",
       "use_findings_from" in nudge)
truthy("nudge no longer tells it to paste the report",
       "pasted into the description" not in nudge)

# --------------------------------------------------------------- tool surface
print("\ntool surface")
import inspect                                                  # noqa: E402
sig = inspect.signature(S.delegate_task)
truthy("delegate_task accepts use_findings_from",
       "use_findings_from" in sig.parameters)
check("and it is optional",
      sig.parameters["use_findings_from"].default, None)

# The schema must actually BUILD. swarm.py has `from __future__ import
# annotations`, so every annotation is a string that pydantic evaluates when
# AgentScope generates the tool schema. Annotating this `Optional[list]` raised
#   PydanticUserError: `_StructuredOutputDynamicClass` is not fully defined;
#   you should define `Optional`
# at AGENT CONSTRUCTION time -- the whole swarm failed to build, and calling the
# function directly (as every test above does) never touches that path.
import asyncio                                                  # noqa: E402
from agentscope.tool import Toolkit, FunctionTool                # noqa: E402

tk = Toolkit(tools=[FunctionTool(S.delegate_task, is_read_only=False)])
schemas = asyncio.run(tk.get_tool_schemas())
fn = schemas[0].get("function", schemas[0])
check("the tool schema builds at all", fn.get("name"), "delegate_task")
props = (fn.get("parameters") or {}).get("properties", {})
truthy("use_findings_from is in the generated schema",
       "use_findings_from" in props)
truthy("so are the required args",
       "child" in props and "description" in props)

# Normalise: a docstring wraps, so a phrase is rarely a contiguous substring.
doc = " ".join((S.delegate_task.__doc__ or "").split())
truthy("documented for the model", "use_findings_from" in doc)
truthy("tells it not to copy reports by hand",
       "do NOT have to copy it into the description" in doc)

import multi_agent.roster as R                                  # noqa: E402
for name, persona in (("mock", R.ORCHESTRATOR_PERSONA),
                      ("native", R.ORCHESTRATOR_PERSONA_NATIVE)):
    truthy(f"{name} persona teaches use_findings_from",
           "use_findings_from" in persona)
    truthy(f"{name} persona drops the paste-it-in example",
           "report text pasted here in full" not in persona)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    raise SystemExit(1)
print("all findings-handoff checks passed")
