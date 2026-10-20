"""Fixes 1-3: stop the browser reporting mechanical success as real success.

Replays the exact failure from data/browser_runs/run_20260922_104700.jsonl:

    click @e6                         -> button  "Add Source category"   (navigates)
    type 'CC and Hypo Loans' into @e6 -> heading "Source Category"       (silent no-op)
    click @e10                        -> button  "Submit"                (empty form)

    step 8  /newCategory  hash=b7b4225b5dfb0547
    step 9  /newCategory  hash=b7b4225b5dfb0547     <- unchanged

The agent then announced the category had been created. The refs were NOT
stale: it had the right snapshot and picked the wrong element out of it. Every
signal needed to catch this already existed and none was checked.

Run:  .venv/Scripts/python.exe test_browser_guards.py
"""
from __future__ import annotations

import browser_supervisor as S
import browser_tools as B

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


def text_of(chunk) -> str:
    try:
        return "".join(b["text"] for b in chunk["content"])
    except Exception:
        return str(chunk)


# The refs map exactly as it stood on /newCategory in the real run.
REFS = {
    "e6": {"role": "heading", "name": "Source Category"},
    "e8": {"role": "textbox", "name": "Category Name"},
    "e10": {"role": "button", "name": "Submit"},
    "e12": {"role": "combobox", "name": "Type"},
    "e14": {"role": "searchbox", "name": "Find"},
    "e16": {"role": "link", "name": "Back"},
    "e18": {"role": "generic", "name": "wrapper"},
}
B._last_refs = dict(REFS)

# ------------------------------------------------------ fix 1: browse_type
print("\nfix 1 — browse_type refuses a target that cannot hold text")
out = text_of(B.browse_type("@e6", "CC and Hypo Loans"))
truthy("the exact failing call is now refused", "REFUSED" in out)
truthy("it names the role that blocked it", "heading" in out)
truthy("it names the element", "Source Category" in out)
truthy("it explains the real danger (empty submit)", "EMPTY field" in out)
truthy("it says what to do instead", "browse_snapshot" in out)
truthy("it offers the ref-free escape hatch", "type_label" in out)

print("\n  …and does not block legitimate targets")
for ref, role in (("@e8", "textbox"), ("@e12", "combobox"),
                  ("@e14", "searchbox")):
    check(f"{role} allowed", B._typeable_error(ref, "x"), "")
# Unrecognised roles pass: refusing them would break flows that work today.
check("unknown role ('generic') allowed", B._typeable_error("@e18", "x"), "")
check("unresolvable ref allowed (browser decides)",
      B._typeable_error("@e99", "x"), "")

# ------------------------------------------------------- fix 2: the batch
print("\nfix 2 — the same guard inside browse_batch")
# The real run used a BATCH, so a guard only on browse_type would have missed
# it entirely: batches translate straight to CLI args.
try:
    B._action_to_command({"action": "type", "ref": "@e6", "text": "CC and Hypo"})
    FAILS.append("batch refuses the heading")
    print("  FAIL batch refuses the heading")
except ValueError as exc:
    truthy("batch refuses the heading", "REFUSED" in str(exc))

check("a real field still translates",
      B._action_to_command({"action": "type", "ref": "@e8", "text": "CC"}),
      ["fill", "@e8", "CC"])
check("clicks are untouched",
      B._action_to_command({"action": "click", "ref": "@e10"}),
      ["click", "@e10"])
check("ref-free fills are untouched",
      B._action_to_command({"action": "type_label", "label": "Category Name",
                            "text": "CC"}),
      ["find", "label", "Category Name", "fill", "CC"])

# ------------------------------------------------- fix 3: nothing changed
print("\nfix 3 — an action that changes nothing says so")
H = "b7b4225b5dfb0547"          # the real hash from both step 8 and step 9

note = S._no_change_note("batch", H, H)
truthy("the real batch would now be flagged", "NOTHING CHANGED" in note)
truthy("it forbids reporting the step as done", "Do NOT report" in note)
truthy("it suggests looking for validation", "validation" in note)

truthy("a click that changes nothing is flagged",
       S._no_change_note("click", H, H))
check("a click that DOES change is silent",
      S._no_change_note("click", H, "different"), "")
check("no previous hash is silent (first action)",
      S._no_change_note("click", "", H), "")

# These are excluded on purpose: our snapshots carry neither input values nor
# scroll offset, so they legitimately leave the hash alone. A warning that
# fires on every keystroke is one the model learns to ignore.
for op in ("type", "fill", "scroll", "snapshot", "get_text"):
    check(f"'{op}' does not cry wolf", S._no_change_note(op, H, H), "")

# ------------------------------------------------------------- regression
print("\nthe supervisor still works")
ctx = S.context()
truthy("after_action is a no-op with no active run",
       S.after_action("click", "@e10", "URL: http://x\\nstuff") == "")



# ===========================================================================
#  fix 4 — a goal is reported as VERIFIED only when the page shows it
# ===========================================================================
print("\nfix 4 — claimed vs verified")

# The three goals from the real run, and the page as it actually looked after
# the empty submit: still on the form, no saved record anywhere.
GOALS = ["Login to the website with the given credentials",
         "'CC and Hypo Loans' category created",
         "Attributes from 'Loan_Attributes.xlsx' file created"]
PAGE_AFTER_EMPTY_SUBMIT = (
    "URL: https://anandniyamweb.drutam.in:9634/newCategory\n"
    "heading \"Source Category\"\n"
    "textbox \"Category Name\" []\n"
    "button \"Submit\"\n"
)

cl = S.Checklist(GOALS)
cl.mark_present_in(PAGE_AFTER_EMPTY_SUBMIT)
check("the page gives NO evidence the category exists",
      cl.items["'CC and Hypo Loans' category created"], False)

# This is the bug in one line: mark_done makes the tick true by construction,
# so reading it AFTERWARDS can never be a check on anything.
cl.mark_done("'CC and Hypo Loans' category created")
check("mark_done ticks it regardless of the page",
      cl.items["'CC and Hypo Loans' category created"], True)

# And the evidence path still works when the record really is on the page.
cl2 = S.Checklist(GOALS)
cl2.mark_present_in(
    "URL: .../unposted/SOURCE_CATEGORY\n"
    "cell \"'CC and Hypo Loans' category created\"\n")
truthy("a real saved record DOES tick by evidence",
       cl2.items["'CC and Hypo Loans' category created"])

# The checklist tick must never BE the evidence. 4b moved verification off the
# tick entirely (see goal_is_evidenced below) precisely because the tick is
# forced by mark_done and so can only ever agree with itself.
src = open("multi_agent/swarm.py", encoding="utf-8").read()
truthy("the forced tick still runs (it orients the agent between goals)",
       "ctx.checklist.mark_done(goal)" in src)
check("but the tick is no longer read back as evidence",
      "evidenced = bool(ctx.checklist.items.get(goal))" in src, False)
truthy("evidence comes from the page instead",
       "_sup.goal_is_evidenced(" in src)
# Normalise: the literal wraps across source lines, so match on the joined form.
flat = " ".join(src.split())
truthy("an unverified goal is reported as such",
       "CLAIMED BUT NOT \" f\"VERIFIED" in flat or "CLAIMED BUT NOT" in flat)
truthy("the phrase reaches the report intact",
       "CLAIMED BUT NOT " in src and "VERIFIED" in src)
truthy("and the report says not to trust it",
       "Treat it as unconfirmed" in src)

# --- fix 4b: verify by the NAME in the goal, not the sentence --------------
# The checklist is set to the planned GOALS, but mark_present_in matches an
# item against the exact value of a table cell. A sentence never equals a cell,
# so NO goal has ever ticked by evidence -- every check mark came from the
# forced mark_done. Checking the sentence would therefore report every goal as
# unverified, including the ones that worked: a warning that always fires.
print("\nfix 4b — evidence comes from the name inside the goal")

check("a sentence goal yields no evidence name",
      S.goal_evidence_name("Log in to the website using the credentials"), "")
check("a quoted record name is extracted",
      S.goal_evidence_name("Create a category named 'LC_loans'"), "LC_loans")
check("an underscore identifier is extracted",
      S.goal_evidence_name("Create the attributes Monthly_Income and Existing_EMI"),
      "Monthly_Income")
# A spreadsheet feeding the task is not a record the page will show.
check("a quoted FILENAME is not mistaken for a record",
      S.goal_evidence_name("Attributes from 'Loan_Attributes.xlsx' file created"),
      "")
check("a research goal is not checkable",
      S.goal_evidence_name("Summary of financial aspects of cash credit loans"), "")

LIST_PAGE = ('URL: .../unposted/SOURCE_CATEGORY\n'
             'cell "LC_loans"\ncell "CC and Hypo Loans"\n')
EMPTY_FORM = ('URL: .../newCategory\nheading "Source Category"\n'
              'textbox "Category Name" []\nbutton "Submit"\n')

ck, ev = S.goal_is_evidenced("Create a category named 'LC_loans'", LIST_PAGE)
check("checkable when the goal names a record", ck, True)
check("evidenced when the record is on the page", ev, True)

ck, ev = S.goal_is_evidenced("Create a category named 'LC_loans'", EMPTY_FORM)
check("still checkable on the empty form", ck, True)
check("NOT evidenced when the form never saved", ev, False)

# The un-checkable case must NOT produce a warning -- that is the whole point.
ck, ev = S.goal_is_evidenced("Log in to the website", LIST_PAGE)
check("a login goal is not checkable", ck, False)

# A half-typed value sitting in an input is not a saved record.
ck, ev = S.goal_is_evidenced(
    "Create a category named 'LC_loans'",
    'URL: /newCategory\ntextbox "Category Name" "LC_loans"\n')
check("a value still in the input box does not count as evidence", ev, False)

src2 = open("multi_agent/swarm.py", encoding="utf-8").read()
truthy("swarm warns only when checkable AND unevidenced",
       "checkable and not evidenced" in src2)
truthy("and says so when nothing could confirm it",
       "nothing on a page could confirm this" in src2)

# ===========================================================================
#  fix 5 — back-reference detection
# ===========================================================================
print("\nfix 5 — findings attach for the descriptions that really occurred")
import multi_agent.swarm as SW                                   # noqa: E402

ROWS = [{"id": "tsk_a4dda73ad3", "child": "browser_agent",
         "status": "done", "report": "real research"}]

check("'from task tsk_...' in the text (row 50)",
      SW._auto_finding_ids(
          "Using the summaries from task tsk_a4dda73ad3, create an Excel file.",
          ROWS), ["tsk_a4dda73ad3"])
check("'the browser_agent's report' — underscore (row 55)",
      SW._auto_finding_ids(
          "Using the summaries from the browser_agent's report, create a file.",
          ROWS), ["tsk_a4dda73ad3"])
check("'the browser agent's report' — space, still works",
      SW._auto_finding_ids(
          "Using the summaries from the browser agent's report, create a file.",
          ROWS), ["tsk_a4dda73ad3"])
check("a self-contained delegation attaches nothing",
      SW._auto_finding_ids("Write a DOCX titled Q3 Plan.", ROWS), [])
check("an id that matches no task attaches nothing",
      SW._auto_finding_ids("Use task tsk_deadbeef99.", ROWS), [])

# The sequencing guard shares the marker list and must normalise the same way.
check("the guard also sees the underscore form",
      SW._delegation_needs_earlier_output(
          "Using the browser_agent's research, write the DOCX.",
          [{"id": "t", "child": "browser_agent", "status": "in_progress"}]),
      "browser_agent")

# ===========================================================================
#  fix 6 — store the evidence, render it short
# ===========================================================================
print("\nfix 6 — stored vs rendered result budgets")
import action_log as A                                           # noqa: E402

truthy("stored budget is big enough for a page state", A.RESULT_CHARS >= 300)
truthy("rendered budget stays prompt-sized",
       A.LINE_RESULT_CHARS <= 160 and A.LINE_RESULT_CHARS < A.RESULT_CHARS)

import os, tempfile                                              # noqa: E402
_real = A.LOG_PATH
A.LOG_PATH = os.path.join(tempfile.mkdtemp(prefix="guards-"), "a.jsonl")
try:
    A.set_session("g")
    real_result = (
        "OK: type 'CC and Hypo Loans' into @e6, click @e10 - ran 2/2 actions. "
        "Current page: [ORIENTATION] GOAL: create the category  CHECKLIST: "
        "Login done · category created NOT DONE · attributes NOT DONE  "
        "URL: https://anandniyamweb.drutam.in:9634/newCategory")
    rec = A.record("browser_agent", "browse_batch", {"actions": [1, 2]},
                   result=real_result)
    # The whole point of raising it: the checklist state survives into the log.
    truthy("the stored row keeps the checklist evidence",
           "category created NOT DONE" in rec["result"])
    truthy("the stored row keeps the final URL", "newCategory" in rec["result"])
    truthy("the rendered line stays short",
           len(A.line(rec)) < 260)
finally:
    A.LOG_PATH = _real
    A.set_session("")

# ===========================================================================
#  fix 7 — never fetch one of OUR generated reports over HTTP
# ===========================================================================
# The run handed the browser child a /api/reports/ path while it was on a
# customer portal, and it navigated to
#   https://anandniyamweb.drutam.in:9634/api/reports/Letter_of_Credit_Loans_Info.xlsx
# -- our path against THEIR origin. The portal is a single-page app, so it
# answered with its own shell and a title, and the navigation looked fine.
print()
print("fix 7 — cross-origin report fetching")

REAL_URL = ("https://anandniyamweb.drutam.in:9634/api/reports/"
            "Letter_of_Credit_Loans_Info.xlsx")
err = B._report_url_error(REAL_URL)
truthy("the exact URL from the run is refused", "REFUSED" in err)
truthy("it explains the SPA-shell trap", "single-page app" in err)
truthy("it names the file that was wanted", "Letter_of_Credit_Loans_Info" in err)
truthy("it points at the delegation as the route",
       "use_findings_from" in err)
truthy("it forbids inventing the contents", "Do NOT invent" in err)

# Our OWN origin is refused too: a binary download into a browser tab is not
# the rows the child needs either.
truthy("our own origin is refused as well",
       "REFUSED" in B._report_url_error("http://127.0.0.1:8800/api/reports/x.xlsx"))
check("an ordinary page is untouched",
      B._report_url_error("https://anandniyamweb.drutam.in:9634/login"), "")
check("an ordinary deep link is untouched",
      B._report_url_error("https://example.com/reports/quarterly"), "")

# --- the other half: the child should have been GIVEN the contents ----------
# "create attributes based on the Excel ... Loan_Attributes.xlsx" matched no
# phrase in _BACKREF_MARKERS, so nothing was attached and it went looking.
print()
print("fix 7b — a generated filename attaches its producer's report")
FROWS = [{"id": "tsk_scripter1", "child": "scripter", "status": "done",
          "report": "SAVED XLSX spreadsheet: Letter_of_Credit_Loans_Info.xlsx "
                    "with columns Attribute, Details"}]
check("the real delegation now attaches",
      SW._auto_finding_ids(
          "Log in and create a category named 'LC_loans'. Then create "
          "attributes based on the Excel file Letter_of_Credit_Loans_Info.xlsx.",
          FROWS),
      ["tsk_scripter1"])
check("no filename mentioned -> nothing attached",
      SW._auto_finding_ids("Log in and create a category.", FROWS), [])
check("a file nobody produced -> nothing attached",
      SW._auto_finding_ids("Use Unknown_File.xlsx for the values.", FROWS), [])
check("a bare word like 'the spreadsheet' is too vague",
      SW._auto_finding_ids("Use the spreadsheet for the values.", FROWS), [])

# --- and the persona says so ------------------------------------------------
import multi_agent.roster as RO                                  # noqa: E402
for name in ("_BROWSER_REMINDER", "_BROWSER_REMINDER_NATIVE"):
    txt = getattr(RO, name)
    truthy(f"{name} states it has no filesystem tools",
           "NO filesystem tools" in txt)
    # It used to say "ask for the values in the delegation". The child did
    # exactly that -- "Please provide the attribute specifications directly" --
    # and the run ended, because nothing answers a child mid-run. The blocker
    # has to travel in the report, which IS the channel back.
    truthy(f"{name} forbids asking and waiting",
           "do NOT ask" in txt and "do NOT wait" in txt)
    truthy(f"{name} gives it the blocker line",
           "BLOCKED: need" in txt)

# ===========================================================================
#  fix 8 - quote the page's own validation error
# ===========================================================================
# Run 3 typed the category name into the RIGHT field and clicked Submit, and
# the form still refused: a SECOND required field, Description, was empty. The
# page said so in red -- "Category Description is required." -- and the agent
# reported "no change or confirmation visible on the page" and asked the user
# what to do, because nothing pointed at the line that explained it.
print()
print("fix 8 - the page's own error is quoted back")

FORM_PAGE = "\n".join([
    "URL: https://anandniyamweb.drutam.in:9634/newCategory",
    'heading "Source Category"',
    'textbox "Category Name" "sav_account"',
    'textbox "Description" []',
    "Category Description is required.",
    'button "Submit"',
])

check("the real error is found",
      S.validation_errors(FORM_PAGE), ["Category Description is required."])

note = S._no_change_note("click", "same", "same", FORM_PAGE)
truthy("the no-change warning still fires", "NOTHING CHANGED" in note)
truthy("and now carries the reason", "THE PAGE SAYS WHY" in note)
truthy("quoting the page verbatim", "Category Description is required." in note)

# A field LABEL is not a complaint. Reporting 'textbox "Category Name" required'
# as an error would point the agent at every field on the form, not the failing
# one.
LABELS = "\n".join(['textbox "Category Name" required',
                    'combobox "Type" required'])
check("field labels are not mistaken for errors", S.validation_errors(LABELS), [])
check("a clean page yields nothing",
      S.validation_errors('heading "Source Category"'), [])

for phrase in ("Please enter a valid email",
               "Name already exists",
               "This field cannot be empty",
               "Amount is invalid"):
    truthy("recognised: " + phrase, S.validation_errors(phrase))

# Only when the page actually stalled; a successful submit stays silent.
check("no warning when the page did change",
      S._no_change_note("click", "before", "after", FORM_PAGE), "")

# ===========================================================================
#  fix 4c - only a CREATION claim is checkable
# ===========================================================================
# Run 3 reported "Navigation to the 'Source Category' section: CLAIMED BUT NOT
# VERIFIED". Navigating creates nothing, so there was never anything to find,
# and a warning that fires on goals it cannot possibly confirm is one everyone
# learns to ignore.
print()
print("fix 4c - navigation and research goals are not reported 'unverified'")

for goal in ("Navigation to the 'Source Category' section on the dashboard",
             "Successful login to https://anandniyamweb.drutam.in:9634/login",
             "Detailed summary of savings account information",
             "List of sources used for gathering the information"):
    check("not a creation claim: " + goal[:40], S.goal_claims_creation(goal), False)
    ck, _ = S.goal_is_evidenced(goal, 'cell "anything"')
    check("  -> not checkable, so never warned", ck, False)

for goal in ("Creation of a category named 'sav_account'",
             "Create the attributes Monthly_Income and Existing_EMI",
             "'CC and Hypo Loans' category created"):
    check("is a creation claim: " + goal[:40], S.goal_claims_creation(goal), True)

# Bounded stems: a verb list that quietly matches nouns makes every goal look
# like a creation claim.
check("'savings' is not the verb 'save'",
      S.goal_claims_creation("Summary of savings account information"), False)
check("'address' is not the verb 'add'",
      S.goal_claims_creation("Update the address on file"), False)

# ===========================================================================
#  what is actually IN the prompt — tools, persona, goal message
# ===========================================================================
print()
print("prompt payload — the browser child's tool list")

import multi_agent.roster as _RO                                  # noqa: E402
_spec = _RO.by_name()["browser_agent"]
_tools = {f.__name__ for f in _spec.tool_factory()}

# ~471 tokens of schema on every call for three tools this persona never
# mentions; their instructions live in task_agent.py, a different agent. An
# un-taught tool is worse than a missing one -- a child stuck on a form sees
# list_repeatable_tasks looking like a way out, and a wasted step is one of 40.
for dead in ("list_repeatable_tasks", "run_repeatable_task",
             "save_repeatable_task"):
    check(f"dropped: {dead}", dead in _tools, False)

# click_text is load-bearing, not spare: [ref=eN] dies when a form rebuilds
# after a submit, and this is the tool that re-finds the element on the live
# page. Two of the persona's examples depend on it.
for keep in ("browse_click_text", "browse_batch", "browse_repeat_last",
             "browse_send_sensitive", "browse_get_text"):
    truthy(f"kept: {keep}", keep in _tools)

print()
print("prompt payload — the persona says what the last three runs needed")
for variant, native in (("native", False), ("mock", True)):
    txt = _spec.persona_for(native)
    # D-next-action: the turn used to be framed as "produce a report", which is
    # what produced "I've compiled a detailed summary..." with no summary.
    truthy(f"{variant}: the turn is about the next action",
           "NEXT ACTION" in txt)
    # Run 3: the right field was filled, Description was empty, the form
    # refused, and the empty submit was reported as a created category.
    # Matched on the collapsed text: the persona is hard-wrapped, so asserting
    # a frozen line would break on a reflow rather than on a lost rule.
    _flat = " ".join(txt.split())
    truthy(f"{variant}: every required field, same submit",
           "field marked required must have a value in the SAME submit" in _flat)
    # Run 1: typed into a heading.
    truthy(f"{variant}: only a field accepts text", "Typing into a heading" in txt)
    # The run ended by asking the user a question nobody was reading.
    truthy(f"{variant}: no questions mid-run",
           "Never end your turn with a question" in txt)
    # The honest exit: an unconditional 'never stop' produces fabrication, so
    # stopping truthfully has to be the cheap path, not a removed one.
    truthy(f"{variant}: stuck has a legitimate ending",
           'browse_get_text("body")' in txt)
    # D1: the report IS the work.
    truthy(f"{variant}: the report must carry content",
           "Your report IS the work" in txt)
    truthy(f"{variant}: with the source it came from", "source URL" in txt)
    # D2: the canonical example no longer teaches fill-one-field-then-Submit.
    truthy(f"{variant}: the batch example fills every field",
           '"label":"Description"' in txt or '"label": "Description"' in txt)

print()
print("prompt payload — the goal message")

import multi_agent.swarm as _SW                                   # noqa: E402

_goals = ["Log in to the portal",
          "Creation of a category named 'sav_account'",
          "Creation of the attributes under it"]
_results = ["1. Log in to the portal -> logged in; dashboard shows 145 attributes",
            "2. Creation of a category named 'sav_account' -> created, id CAT_2291"]
_msg = _SW._goal_message("full original task text", _goals,
                         [True, True, False], 2, "carry line", _results)

# Only the PREVIOUS goal's first line used to travel, so by goal 3 the result
# of goal 1 -- where the ids and the naming rule are discovered -- was gone.
truthy("goal 1's result still travels at goal 3", "145 attributes" in _msg)
truthy("goal 2's result too", "CAT_2291" in _msg)
truthy("under a heading that says what they are",
       "RESULTS OF THE GOALS ALREADY DONE" in _msg)
truthy("this goal is still singled out", "GOAL 3 OF 3" in _msg)

# No results yet -> the old single-line carry is still used rather than an
# empty heading.
_first = _SW._goal_message("task", _goals, [False, False, False], 0, "", [])
check("goal 1 has no results block",
      "RESULTS OF THE GOALS ALREADY DONE" in _first, False)

# The budget keeps a long run's results from crowding out the goal itself.
_many = [f"{i}. goal {i} -> " + "x" * 300 for i in range(1, 12)]
truthy("the results block is capped",
       len(_SW._results_block(_many)) <= _SW._RESULTS_BUDGET + 400)
truthy("and keeps the most recent", "11. goal 11" in _SW._results_block(_many))

# ===========================================================================
#  A — "schedule a meeting" must reach the only child with a calendar
# ===========================================================================
# It went to task_manager, which has no calendar tools: it made a task and a
# 6pm reminder whose own text read "Google Meet link included" with no link
# anywhere, and no calendar_event_created in the whole session.
print()
print("A - meetings route to comms_agent")

_specs = _RO.by_name()
_orch = _RO.ORCHESTRATOR_PERSONA
_flat_orch = " ".join(_orch.split())
truthy("the routing table sends meetings to comms",
       "ALSO real calendar meetings" in _flat_orch)
truthy("and says a reminder is not a meeting",
       "a reminder at 6pm is not a meeting" in _flat_orch)
truthy("comms advertises create_meeting",
       "create_meeting" in _specs["comms_agent"].capabilities)
truthy("comms says it is the only source of a Meet link",
       "ONLY child that can produce a Meet link"
       in " ".join(_specs["comms_agent"].capabilities.split()))
truthy("task_manager disclaims meetings",
       "NOT for meetings" in _specs["task_manager"].capabilities)

# The tool actually has to be there for the routing to mean anything.
truthy("comms really holds create_meeting",
       "create_meeting" in {f.__name__ for f in _specs["comms_agent"].tool_factory()})
check("task_manager really does not",
      "create_meeting" in {f.__name__ for f in _specs["task_manager"].tool_factory()},
      False)

# ===========================================================================
#  B — a report of failure is not a finding
# ===========================================================================
# The research child reported "I was unable to access consistent information
# from the web pages I visited". The orchestrator passed that as
# use_findings_from anyway and the scripter built a spreadsheet whose second
# row echoed its own column headers.
print()
print("B - a failed report cannot be used as input")

_REAL_FAILURE = ("goal 1/3 (Identify the key financial details typically "
                 "captured in a savings account): I was unable to access "
                 "consistent information from the web pages I visited due to "
                 "issues with the sites.")
check("the real failing report is unusable",
      _SW.finding_is_usable(_REAL_FAILURE), False)
truthy("a report with actual facts is usable",
       _SW.finding_is_usable("APY 0.42% national average, 4.50% top rate "
                             "(source: nerdwallet.com); minimum deposit $0-100."))
# Conservative: one setback mentioned late in a long, real report is an aside,
# not a headline, and must not disqualify the work around it.
truthy("a late aside does not disqualify a real report",
       _SW.finding_is_usable("x" * 400 + " I was unable to open one of them."))

_rows_fail = [{"id": "tsk_a", "child": "browser_agent", "status": "done",
               "report": _REAL_FAILURE}]
check("it is named as unusable",
      _SW.unusable_finding_ids(_rows_fail, ["tsk_a"]), ["tsk_a"])
# Passed through but labelled -- dropping it silently would leave the child
# with no explanation for why its input is missing.
_blk = _SW._findings_block(_rows_fail, ["tsk_a"])
truthy("the briefing warns instead of hiding it",
       "DID NOT PRODUCE USABLE OUTPUT" in _blk)
truthy("and forbids filling the gap by invention", "do NOT invent" in _blk)

# ===========================================================================
#  C — a blocked child is re-delegated, not abandoned
# ===========================================================================
# The exact sentence the browser child ended on. It was doing what the reminder
# told it to; the run ended there because a report is the only channel back.
print()
print("C - BLOCKED becomes an instruction on the board")

_REAL_BLOCK = ("I am unable to access the content of the Excel file "
               "'savings_account_details.xlsx' directly through the browser, "
               "as my current tools do not support file download and reading "
               "from the internal endpoint (/api/reports/...). Please provide "
               "the attribute specifications directly.")
truthy("the real sentence is recognised as blocked",
       _SW.blocked_request(_REAL_BLOCK)[1])
check("and it is a DATA blocker the orchestrator can fix",
      _SW.blocked_request(_REAL_BLOCK)[0], "need")
truthy("the explicit line is recognised too",
       _SW.blocked_request("Created the category.\nBLOCKED: need the "
                           "attribute names and their data types")[1])
check("a finished report is not blocked",
      _SW.blocked_request("Created sav_account and 5 attributes, all visible "
                          "on the list page."), ("", ""))
check("an empty report is not blocked", _SW.blocked_request(""), ("", ""))

_row = _SW._row({"id": "tsk_x", "child": "browser_agent", "status": "done",
                 "report": _REAL_BLOCK})
truthy("the board carries what is missing", _row.get("blocked_on"))
truthy("and tells the orchestrator to re-delegate",
       "Re-delegate the same task" in (_row.get("hint") or ""))
truthy("and not to mark it done",
       "Do NOT mark this done" in (_row.get("hint") or ""))

# The request usually sits at the END of the report, which is exactly what a
# cap removes -- so it must be read before truncation.
_row_cap = _SW._row({"id": "tsk_x", "child": "browser_agent", "status": "done",
                     "report": _REAL_BLOCK}, report_cap=60)
truthy("it survives a truncated report", _row_cap.get("blocked_on"))

# And the prompt side says the same thing.
truthy("the orchestrator persona handles BLOCKED",
       "BLOCKED: need" in _orch)

# ===========================================================================
#  D — "as researched" is a back-reference
# ===========================================================================
print()
print("D - the delegation's back-reference resolves")

_FROWS = [{"id": "tsk_r", "child": "browser_agent", "status": "done",
           "report": "APY 0.42%; minimum deposit $0-100 (source: nerdwallet)"},
          {"id": "tsk_s", "child": "scripter", "status": "done",
           "report": "SAVED XLSX spreadsheet: savings_account_details.xlsx"}]
for phrase in ("create attributes related to savings accounts as researched",
               "create the attributes you researched earlier",
               "build the list based on the research"):
    check(f"resolves: {phrase[:42]!r}",
          _SW._auto_finding_ids(phrase, _FROWS), ["tsk_r"])
# A dangling reference names no child but still means "whatever just finished".
check("a dangling reference takes the last usable report",
      _SW._auto_finding_ids("use the findings above to create them", _FROWS),
      ["tsk_s"])
check("no reference at all attaches nothing",
      _SW._auto_finding_ids("create a category called sav_account", _FROWS), [])

# ===========================================================================
#  E — comms finishes what it starts
# ===========================================================================
# It called lookup_contact and stopped. No send_email in the session at all.
print()
print("E - comms acts, it does not stop at lookup")

for _native in (False, True):
    _c = " ".join(_specs["comms_agent"].persona_for(_native).split())
    truthy(f"comms turn is about the next action (native={not _native})",
           "decide THE NEXT ACTION and take it" in _c)
    truthy(f"lookup is not the task (native={not _native})",
           "Resolving a contact is not the task" in _c)
    truthy(f"no questions mid-run (native={not _native})",
           "never end a turn with a question" in _c)
    truthy(f"comms knows it owns meetings (native={not _native})",
           "real calendar meetings" in _c)

# ===========================================================================
#  a WALL is not a gap
# ===========================================================================
# Three research delegations died on the same security check in one run: the
# report said "BLOCKED: need access past the security verification", that was
# read as a request for data, and the answer to a request for data is to ask
# again. Nothing the orchestrator writes gets past a bot check.
print()
print("blocked: a wall and a gap need opposite responses")

_REAL_WALL = ("goal 1/2 (Gather data on the financial-related information "
              "captured in current accounts): BLOCKED: need access to the "
              "website past the security verification to gather data.")
_kind, _what = _SW.blocked_request(_REAL_WALL)
check("the real loop-causing report is a SITE blocker", _kind, "site")
truthy("and it still says what happened", "security verification" in _what)

# The marker sits mid-line in a segmented report; the anchored pattern this
# replaces matched none of the real ones.
truthy("a mid-line marker is found",
       _SW.blocked_request("goal 2/3 (X): BLOCKED: site cloudflare bot check")[1])

for _wall in ("BLOCKED: site the page returned 403 Forbidden",
              "BLOCKED: need to get past the captcha",
              "BLOCKED: need a subscription required to read the rates",
              "BLOCKED: site sign in to continue"):
    check(f"wall: {_wall[9:44]!r}", _SW.blocked_request(_wall)[0], "site")

for _gap in ("BLOCKED: need the attribute names and data types",
             "BLOCKED: need the column headers from the spreadsheet"):
    check(f"gap:  {_gap[9:44]!r}", _SW.blocked_request(_gap)[0], "need")

_site_row = _SW._row({"id": "t", "child": "browser_agent", "status": "done",
                      "report": _REAL_WALL})
truthy("a wall tells the orchestrator NOT to retry",
       "do NOT send it back for the same thing" in (_site_row.get("hint") or ""))
check("and does not tell it to re-delegate",
      "Re-delegate the same task" in (_site_row.get("hint") or ""), False)

_flat_orch2 = " ".join(_RO.ORCHESTRATOR_PERSONA.split())
truthy("the persona distinguishes the two kinds",
       "BLOCKED: site" in _flat_orch2 and "BLOCKED: need" in _flat_orch2)
truthy("and caps repeat delegations", "more than twice" in _flat_orch2)
for _n in ("_BROWSER_REMINDER", "_BROWSER_REMINDER_NATIVE"):
    truthy(f"{_n} can report a wall", "BLOCKED: site" in getattr(_RO, _n))

# ===========================================================================
#  the veto keeps its narrow vocabulary
# ===========================================================================
# "...include attributes for transaction logs and commercial flow, as
# identified in the report." matched the widened list, resolved "report" to
# scripter, and two ordinary BRE delegations were refused outright. A phrase
# list that ATTACHES a report may over-fire; the same list driving a VETO may
# not, because over-firing there costs the task.
print()
print("the attach vocabulary and the veto vocabulary are separate")

_D9 = ("Access the BRE website and navigate to the 'Source Category' section. "
       "Use this section to create a category named 'curr_account' and include "
       "attributes for transaction logs and commercial flow, as identified in "
       "the report.")
check("the delegation that was HELD now passes",
      _SW._delegation_needs_earlier_output(_D9, []), "")
# The original narrow phrases must still veto -- that guard earns its keep.
check("an explicit unmet dependency is still refused",
      _SW._delegation_needs_earlier_output(
          "Attach the files you created and send them.", []), "scripter")
_AR = [{"id": "tsk_r", "child": "browser_agent", "status": "done",
        "report": "APY 0.42%; minimum deposit $0-100"}]
check("the wide phrase still ATTACHES",
      _SW._auto_finding_ids("create the attributes as researched", _AR),
      ["tsk_r"])

# ===========================================================================
#  the browser is handed values, not a teammate briefing
# ===========================================================================
# set_context(goal=rec.description) read the whole string, so once a briefing
# was prepended the GOAL line -- re-sent on every step -- read "FINDINGS FROM
# YOUR TEAMMATES (real reports from the shared board..." and the goal itself
# never appeared.
print()
print("the browser gets a task, not a briefing")

_WITH_BRIEF = ("FINDINGS FROM YOUR TEAMMATES (...):" + _SW._TASK_MARKER
               + "Log in and create curr_account.")
check("task_text strips the briefing",
      _SW.task_text(_WITH_BRIEF), "Log in and create curr_account.")
check("a plain description is untouched",
      _SW.task_text("Log in and create curr_account."),
      "Log in and create curr_account.")
_flat_rem = " ".join(_RO.ORCHESTRATOR_REMINDER_NATIVE.split())
truthy("the orchestrator is told to write the values out",
       "write the VALUES out" in _flat_rem)

# ===========================================================================
#  only the page awards a tick
# ===========================================================================
# mark_done ran unconditionally at every goal boundary, so the header read
# "'curr_account' category is created in the 'Source Category' section ✓" on
# every step while the category did not exist -- and the header is what the
# agent orients by. It read its own unverified claim back as fact.
print()
print("a goal nothing can confirm is '?', never a tick")

_G = "'curr_account' category is created in the 'Source Category' section"
_cl = S.Checklist([_G, "Transaction logs attributes are added"])
_cl.mark_claimed(_G)
truthy("it renders as a question", " ?" in _cl.render())
check("and never as a tick", "✓" in _cl.render(), False)
check("a claim does not count as progress", _cl.progress()[0], 0)
truthy("it stays on the remaining list", _G in _cl.remaining)
_cl.mark_done(_G)
truthy("evidence still upgrades it to a tick", "✓" in _cl.render())

_ctx = S.set_context(goal="Create curr_account", max_steps=40, items=[])
_ctx.set_items([_G, "Transaction logs attributes are added"], as_goals=True)
S.set_goal(2, 2, "Transaction logs attributes are added")
_ctx.step = 4
_ctx.trail = [
    {"step": 2, "url": "https://x/unposted/SOURCE_CATEGORY", "op": "click",
     "target": "Source category", "goal": 1, "ok": True, "err": ""},
    {"step": 3, "url": "https://x/newCategory", "op": "click",
     "target": "Add Source category", "goal": 1, "ok": True, "err": ""},
    {"step": 4, "url": "https://x/unposted/SOURCE_ATTRIBUTES", "op": "click",
     "target": "Source attributes", "goal": 2, "ok": True, "err": ""},
]
_ctx.checklist.mark_claimed(_G)
_hdr = S.render_header(_ctx)
truthy("the header explains the mark", "reported done but NOT found" in _hdr)
# It cannot judge a claim from the claim. The steps that supposedly produced
# it belong to the PREVIOUS goal, which the window would otherwise drop.
truthy("the claimed goal's own steps come along",
       "Add Source category" in _hdr)
truthy("and the label says so", "the one before it" in _hdr)
_ctx.active = False

# ===========================================================================
#  the orchestrator stops inventing UI
# ===========================================================================
# It wrote "Use the 'Create Category' section in the sidebar" for a portal with
# no such section. The child obeyed: click 'Create Category' -> Element not
# found.
print()
print("outcomes, not buttons")
truthy("the persona forbids naming buttons it has not seen",
       "never WHICH BUTTON to press" in _flat_orch2)
truthy("quoting the sentence that caused it", "Create Category" in _flat_orch2)
truthy("and the reminder repeats it, because personas decay",
       "never which button to press" in _flat_rem)
truthy("the browser must be given values, not references",
       "no files and no board" in _flat_rem)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    raise SystemExit(1)
print("all browser-guard checks passed (fixes 1-8)")
