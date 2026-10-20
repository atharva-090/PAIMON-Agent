"""A1-A4: the element table, the wedged-page gate, and the memory trim.

    .venv/Scripts/python.exe test_a1_a4.py

Every fixture here is real output from session swarm-1791913203232 (13 Oct
2026) -- the run where the agent clicked one button eleven times.
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import browser_supervisor as S

FAILS = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILS.append(label)


def truthy(label, got):
    print(f"  {'ok  ' if got else 'FAIL'} {label}")
    if not got:
        FAILS.append(label)


# The real tree agent-browser returned at 17:54:44, trimmed to the shapes that
# matter: nav links, a toolbar, a table, a native select, and the wrappers.
TREE = """URL: https://anandniyamweb.drutam.in:9634/unposted/SOURCE_ATTRIBUTES
- generic [ref=e1] clickable [onclick]
  - generic
    - complementary
      - StaticText "N"
      - StaticText "NIYAM"
      - button "Collapse" [ref=e4]
        - image
      - navigation
        - list
          - listitem [level=1]
            - link "Dashboard Dashboard" [ref=e92]
          - listitem [level=1]
            - link "Source attributes Source attributes" [ref=e95]
      - StaticText "Powered by Netwin"
    - generic
      - SvgRoot
        - StaticText "Total Attributes"
        - StaticText "162"
      - generic
        - searchbox "Search" [ref=e12]
      - generic
        - button "All Category" [expanded=false, ref=e28]
          - StaticText "All Category"
      - button "Add Source attributes" [ref=e6]
      - table
        - rowgroup
          - row
            - columnheader "Sr. No" [ref=e102]
            - columnheader "Attribute ID" [ref=e103]
        - row [ref=e14] clickable [onclick]
          - cell "01" [ref=e29]
          - cell "ATT_174" [ref=e30]
          - cell "Business_Status" [ref=e31]
          - cell "Commercial_BL" [ref=e32]
          - cell
        - row [ref=e15] clickable [onclick]
          - cell "02" [ref=e35]
          - cell "ATT_173" [ref=e36]
          - cell "Application_complete" [ref=e37]
          - cell "Commercial_BL" [ref=e38]
          - cell
      - combobox "Show Entries" [ref=e7]
        - option "10" [selected=true, ref=e89]
        - option "20" [ref=e90]
      - button "Previous" [disabled, ref=e109]
      - StaticText "Parameter with same name already exist."
"""


# ===========================================================================
#  A2 — the element table
# ===========================================================================
print("A2 - accessibility tree -> element table")
flat = S.compact_snapshot(TREE)

truthy("it is much smaller", len(flat) < len(TREE) * 0.65)
print(f"       {len(TREE)} -> {len(flat)} chars "
      f"({round(100 * len(flat) / len(TREE))}%)")

# Every ref the agent could act on must survive, or a click stops resolving.
for ref in ("e4", "e92", "e95", "e12", "e28", "e6", "e7", "e89", "e109"):
    truthy(f"ref {ref} survives", f"[{ref}]" in flat)

truthy("a control keeps its name", "Add Source attributes" in flat)
truthy("and its state", "expanded=false" in flat)
truthy("a disabled control says so", "disabled=true" in flat)
truthy("a selected option says so", "selected=true" in flat)

# A table row is one line, not seven.
truthy("a row collapses to one line",
       "01 | ATT_174 | Business_Status | Commercial_BL" in flat)
check("each row is a single line",
      sum(1 for ln in flat.splitlines() if ln.startswith("[e14]")), 1)

# The wrappers are what made the tree 7.5 KB.
for noise in ("generic", "rowgroup", "listitem", "SvgRoot", "complementary"):
    truthy(f"{noise!r} is gone", noise not in flat)

# The toast is the whole reason the 13 Oct run could not diagnose itself.
truthy("page text survives", "Parameter with same name already exist." in flat)
truthy("so do the page's numbers", "162" in flat)
truthy("the URL is still the first line", flat.startswith("URL: https://"))

# A StaticText that only repeats the button above it is dead weight.
check("a label is not repeated as page text", flat.count("All Category"), 1)

# Nothing recognisable? Pass it through rather than lose it.
odd = 'URL: x\n- button "Go" [ref=e1]\n!! not a tree line at all\n'
truthy("an unparsable line is kept",
       "!! not a tree line at all" in S.compact_snapshot(odd))
check("empty input is safe", S.compact_snapshot(""), "")

# The hash must not see refs. It did not before, because it stripped
# "[ref=eN]"; the table spells them "[eN]", and missing that would make every
# re-render of one page look like a change and silence the stall detector.
a = S.compact_snapshot(TREE)
b = S.compact_snapshot(TREE.replace("ref=e92", "ref=e777"))
check("a renumbered ref does not change the hash",
      S.snapshot_hash(a), S.snapshot_hash(b))
truthy("a real edit does change it",
       S.snapshot_hash(a) != S.snapshot_hash(
           S.compact_snapshot(TREE.replace("Add Source attributes", "Delete"))))

# The supervisor reads evidence off the rendered page; it must still find it.
truthy("the checklist can still see a record",
       S.goal_is_evidenced("Commercial_BL category created", flat)[1])


# ===========================================================================
#  A4 — the wedged-page gate
# ===========================================================================
print()
print("A4 - three dead actions and the gate closes")

PAGE_A = 'URL: https://x/list\n- button "Add" [ref=e6]\n'
PAGE_B = 'URL: https://x/form\n- textbox "Name" [ref=e7]\n'

S.set_context(goal="create the attributes", max_steps=30)
check("a fresh run is not wedged", S.wedged(), False)

# The 13 Oct sequence: three spellings of one button, each leaving the page
# byte-identical. The old monitor needed the SAME (url, op, target) three times
# and so never flagged it.
S.after_action("snapshot", "", PAGE_A)
S.after_action("click", "Add Source attributes", PAGE_A)
check("one dead click is not a wedge", S.wedged(), False)
S.after_action("batch", "click @e6", PAGE_A)
check("two is still not", S.wedged(), False)
S.after_action("click", "click @e6", PAGE_A)
check("three different spellings DO wedge it", S.wedged(), True)
check("the counter is readable", S.context().no_change, 3)

msg = S.wedge_report()
truthy("the refusal says it did not run", msg.startswith("REFUSED"))
truthy("it names the three actions", "click @e6" in msg)
truthy("it offers the real cause", "on top of it" in msg)

# A page that actually changed clears it.
S.after_action("click", "Add", PAGE_B)
check("a working action clears the wedge", S.wedged(), False)
check("and zeroes the counter", S.context().no_change, 0)

# A read is neither a success nor a failure: it must not mask a wedge.
for _ in range(3):
    S.after_action("click", "x", PAGE_B)
check("wedged again after three dead clicks", S.wedged(), True)
S.after_action("snapshot", "", PAGE_B)
check("a snapshot does not clear it", S.wedged(), True)
S.after_action("get_text", "body", PAGE_B)
check("nor does reading the text", S.wedged(), True)
S.clear_wedge()
check("only the recovery clears it", S.wedged(), False)

# The old gate stays, and still refuses nothing when no run is open.
S.clear_context()
check("no run means no gate", S.wedged(), False)
check("and should_block stays quiet", S.should_block("click", "x"), None)


# ===========================================================================
#  A3 — what is on top of the page
# ===========================================================================
print()
print("A3 - the overlay probe")
import browser_tools as B                                        # noqa: E402

truthy("the probe is one expression", B._OVERLAY_JS.startswith("(() =>"))
truthy("it looks for a backdrop", "backdrop" in B._OVERLAY_JS)
truthy("and a toast", "toast" in B._OVERLAY_JS)
truthy("it reads computed style, not the a11y tree",
       "getComputedStyle" in B._OVERLAY_JS)

check("the gate ignores a read", B._wedge_guard("get_text"), None)
check("and a snapshot", B._wedge_guard("snapshot"), None)
# With no supervised run open there is nothing to refuse.
check("and does nothing when no run is active", B._wedge_guard("click"), None)
check("the limit is configurable", B.NO_CHANGE_LIMIT, 3)
truthy("the element table is on by default", B.COMPACT_SNAPSHOT)

# The whole gate, against a stubbed browser: wedge the run, then call the
# guard and check that the action is refused, Escape is tried, the reload
# follows because Escape changed nothing, and the run is released afterwards.
CALLS = []
STUCK = 'URL: https://x/list\n- button "Add" [ref=e6]\n'
FREED = ('URL: https://x/list\n- dialog "New attribute"\n'
         '- textbox "Name" [ref=e9]\n')
_page = [STUCK]
TYPED = [False]          # does a form on the page hold typed text?


def _fake(session, command, args=None, **kw):
    CALLS.append((command, list(args or [])))
    if command == "snapshot":
        return {"success": True, "data": {"snapshot": _page[0], "origin": "https://x/list"}}
    if command == "eval" and args and "location.reload" in args[0]:
        _page[0] = FREED           # the reload is what finally unsticks it
        return {"success": True, "data": {"result": None}}
    if command == "eval" and args and "querySelectorAll" in args[0]             and "value" in args[0]:
        return {"success": True, "data": {"result": TYPED[0]}}
    if command == "eval":
        return {"success": True, "data": {"result": "modal-backdrop // Parameter with same name already exist."}}
    return {"success": True, "data": {}}


_real = B.run_agent_browser_command
B.run_agent_browser_command = _fake
try:
    S.set_context(goal="create the attributes", max_steps=30)
    S.after_action("snapshot", "", STUCK)
    for spelling in ("Add Source attributes", "click @e6", "batch click @e6"):
        S.after_action("click", spelling, STUCK)
    truthy("the run is wedged before the guard runs", S.wedged())

    out = B._wedge_guard("click")
    truthy("the guard returns a refusal", bool(out) and out.startswith("REFUSED"))
    truthy("it pressed Escape", ("press", ["Escape"]) in CALLS)
    truthy("and then reloaded, because Escape changed nothing",
           any(c == "eval" and a and "location.reload" in a[0] for c, a in CALLS))
    truthy("it names what was on top", "modal-backdrop" in out)
    truthy("it quotes the page's own complaint", "already exist" in out)
    truthy("it hands back the recovered page", "New attribute" in out)
    check("and the run may act again", S.wedged(), False)

    # Cleared, so the very next action goes through untouched.
    check("the guard steps aside once recovered", B._wedge_guard("click"), None)

    # A filled form must survive the recovery. On the 09:02 live run the
    # category form held "Retail_LN" and its description, the submit control
    # could not be located, and the reload threw both away -- the agent then
    # had to type everything again and ran out of steps.
    CALLS.clear()
    TYPED[0] = True
    _page[0] = STUCK
    S.set_context(goal="create the attributes", max_steps=30)
    S.after_action("snapshot", "", STUCK)
    for spelling in ("type name", "click Submit", "press Enter"):
        S.after_action("click", spelling, STUCK)
    out2 = B._wedge_guard("click")
    truthy("it still refuses the action", bool(out2)
           and out2.startswith("REFUSED"))
    truthy("it still presses Escape", ("press", ["Escape"]) in CALLS)
    truthy("but it does NOT reload over typed text",
           not any(c == "eval" and a and "location.reload" in a[0]
                   for c, a in CALLS))
    truthy("and it says why", "still holds what you typed" in out2)
    check("the run may act again either way", S.wedged(), False)

    # An unreadable page must take the safe branch, not the destructive one.
    CALLS.clear()
    S.set_context(goal="g", max_steps=30)
    S.after_action("snapshot", "", STUCK)
    for _ in range(3):
        S.after_action("click", "x", STUCK)
    B.run_agent_browser_command = lambda *a, **k: {"success": False}
    B._wedge_guard("click")
    truthy("a dead browser does not trigger a reload",
           not any(c == "eval" and a and "location.reload" in a[0]
                   for c, a in CALLS))
finally:
    B.run_agent_browser_command = _real
    S.clear_context()


# ===========================================================================
#  A1 — no memory block for a browser child
# ===========================================================================
print()
print("A1 - the memory lookup skips the browser child")
import inspect                                                   # noqa: E402
import multi_agent.swarm as SW                                   # noqa: E402

src = inspect.getsource(SW.Swarm._run_specialist)
truthy("relevant() is gated on has_browser",
       'if getattr(spec, "has_browser", False)' in src
       and "memory_context.relevant" in src)
truthy("procedural_hint is NOT gated",
       "skill = memory_context.procedural_hint(rec.description)" in src)

# The gate has to agree with the roster, or it trims the wrong child.
from multi_agent.roster import by_name                           # noqa: E402
specs = by_name()
check("browser_agent is a browser child",
      specs["browser_agent"].has_browser, True)
check("comms_agent shares the browser", specs["comms_agent"].has_browser, True)
check("scripter does not", specs["scripter"].has_browser, False)
check("task_manager does not", specs["task_manager"].has_browser, False)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
    sys.exit(1)
print("all A1-A4 checks passed")
