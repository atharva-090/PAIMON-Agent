"""Unit tests for browser_supervisor (Section-K foundations P1-P6).

Dependency-free — runs without agentscope. Exercises every primitive with the
real Niyam-shaped data, plus the loop-detection edge cases.

    python test_browser_supervisor.py
"""

import os
import sys
import tempfile

try:  # the header uses ✓/☐/⚠; let the test console print them on Windows (cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import browser_supervisor as S


def _check(name, cond):
    print(("  ok  " if cond else "FAIL  ") + name)
    if not cond:
        _check.failed += 1
_check.failed = 0


# A trimmed Niyam Source-Attributes snapshot (the list page after Monthly_Income).
SNAP_LIST_1 = """URL: https://anandniyamweb.drutam.in:9634/unposted/SOURCE_ATTRIBUTES
- heading "Source Category"
- button "Add Source attributes" [ref=e6]
- cell "Monthly_Income" [ref=e27]
- cell "Applicant_Information" [ref=e28]
- StaticText "Total Attributes"
- StaticText "145"
"""

# Same page a moment later: a count changed (145 -> 146) but structure identical.
SNAP_LIST_1B = SNAP_LIST_1.replace("145", "146")

# After creating Existing_EMI (a new row appears).
SNAP_LIST_2 = SNAP_LIST_1 + '- cell "Existing_EMI" [ref=e31]\n'

# The new-category form.
SNAP_FORM = """URL: https://anandniyamweb.drutam.in:9634/newCategory
- heading "Source Category"
- textbox [ref=e7]
- button "Submit" [ref=e10]
"""

GOAL = ("Open the portal, create category 'Applicant', then create attributes "
        "Monthly_Income, Existing_EMI, Credit_Score, Employment_Type, "
        "Requested_Tenure and verify them.")


def test_p3_hash_and_diff():
    print("\n[P3] snapshot hash + diff")
    # Structural-equal pages (only a count differs) must hash the SAME.
    _check("count change -> same hash (normalized)",
           S.snapshot_hash(SNAP_LIST_1) == S.snapshot_hash(SNAP_LIST_1B))
    # A genuinely different page must hash differently.
    _check("different page -> different hash",
           S.snapshot_hash(SNAP_LIST_1) != S.snapshot_hash(SNAP_FORM))
    # diff surfaces the new row, ignores the count.
    d = S.diff_snapshots(SNAP_LIST_1, SNAP_LIST_2)
    _check("diff reports the new Existing_EMI row", "Existing_EMI" in d)
    _check("diff ignores the count-only change",
           S.diff_snapshots(SNAP_LIST_1, SNAP_LIST_1B) == "")


def test_p4_checklist():
    print("\n[P4] checklist parse + deterministic ticking")
    items = S.parse_checklist_items(GOAL)
    want = ["Monthly_Income", "Existing_EMI", "Credit_Score",
            "Employment_Type", "Requested_Tenure"]
    _check("parsed the 5 underscore attributes", items == want)
    cl = S.Checklist(items)
    _check("nothing done initially", cl.progress() == (0, 5))
    cl.mark_present_in(SNAP_LIST_1)         # page shows Monthly_Income
    _check("Monthly_Income ticked from the page", cl.progress() == (1, 5))
    cl.mark_present_in(SNAP_LIST_2)         # now shows Existing_EMI too
    _check("Existing_EMI ticked from the page", cl.progress() == (2, 5))
    _check("remaining is the other three",
           cl.remaining == ["Credit_Score", "Employment_Type", "Requested_Tenure"])
    _check("render marks done with check, pending with box",
           "Monthly_Income ✓" in cl.render() and "Credit_Score ☐" in cl.render())
    # comma-list fallback (no underscore names)
    alt = S.parse_checklist_items("Please create the following: Alpha, Beta and Gamma.")
    _check("comma/and fallback list", alt == ["Alpha", "Beta", "Gamma"])


def test_p6_loop_monitor():
    print("\n[P6] loop monitor")
    # Repeat: same click on the same page 3x -> flagged.
    m = S.LoopMonitor(repeat_threshold=3)
    for _ in range(3):
        m.record("u/form", "click", "Submit", "HASH_A")
    _check("3x identical click flagged as repeat",
           (m.flag() or "").startswith("repeat"))
    # Read ops repeating are fine (snapshots), not a loop.
    m2 = S.LoopMonitor(repeat_threshold=3)
    for _ in range(4):
        m2.record("u/form", "snapshot", "", "HASH_A")
    _check("repeated snapshots are NOT flagged", m2.flag() is None)
    # Stall: several real actions, page hash never changes -> flagged.
    m3 = S.LoopMonitor(stall_threshold=4)
    m3.record("u/form", "type", "name", "HASH_A")
    m3.record("u/form", "click", "Submit", "HASH_A")
    m3.record("u/form", "type", "name", "HASH_A")
    m3.record("u/form", "click", "Submit", "HASH_A")
    _check("page-unchanged-despite-actions flagged as stall",
           (m3.flag() or "").startswith("stall"))
    # Healthy progress: page hash changes each step -> no flag.
    m4 = S.LoopMonitor()
    for i in range(6):
        m4.record(f"u/p{i}", "click", f"t{i}", f"HASH_{i}")
    _check("changing pages -> no flag", m4.flag() is None)


def test_p1_p5_context_and_header():
    print("\n[P1+P5] run context + header via after_action")
    tmp = tempfile.mkdtemp()
    S.set_context(goal=GOAL, max_steps=30)
    ctx = S.context()
    _check("context active with parsed checklist",
           ctx.active and ctx.checklist.progress() == (0, 5))
    # Simulate the agent landing on the list page after creating Monthly_Income.
    hdr = S.after_action("click", "Submit", SNAP_LIST_1, run_id="testrun",
                         log_dir=tmp)
    _check("header carries GOAL", "GOAL:" in hdr)
    _check("header carries CHECKLIST with 1/5 done",
           "CHECKLIST (1/5)" in hdr and "Monthly_Income ✓" in hdr)
    _check("header carries TRAIL", "TRAIL:" in hdr)
    _check("header carries STEP budget", "STEP 1/30" in hdr)
    _check("step advanced", ctx.step == 1)
    _check("current url tracked from snapshot",
           "SOURCE_ATTRIBUTES" in ctx.current_url)
    # Next action shows Existing_EMI -> checklist auto-advances to 2/5.
    hdr2 = S.after_action("click", "Save", SNAP_LIST_2, run_id="testrun",
                          log_dir=tmp)
    _check("checklist auto-advanced to 2/5", "CHECKLIST (2/5)" in hdr2)
    _check("inactive context -> empty header (opt-in safety)",
           (S.clear_context() or S.render_header()) == "")


def test_p6_correction_in_header():
    print("\n[P6->P5] auditor correction surfaces in header")
    tmp = tempfile.mkdtemp()
    S.set_context(goal="fill the form", max_steps=20)
    # Click Submit on the SAME unchanged form 3x -> monitor flags -> header warns.
    hdr = ""
    for _ in range(3):
        hdr = S.after_action("click", "Submit", SNAP_FORM, run_id="r2", log_dir=tmp)
    _check("auditor correction injected after a repeat loop", "⚠ AUDITOR:" in hdr)
    _check("stuck_summary reports the stuck run",
           (S.stuck_summary() or "").startswith("STUCK"))
    S.clear_context()


def test_p2_log_reader():
    print("\n[P2] durable log: page records, trail compressor, segmentation")
    tmp = tempfile.mkdtemp()
    rid = "logtest"
    path = os.path.join(tmp, f"run_{rid}.jsonl")
    # Hand-write an action+page interleaving like browser_tools would produce:
    # one create-attribute iteration (type name, click Save) then the list page.
    import json
    rows = [
        {"op": "click", "action": "click 'Add Source attributes'",
         "args": {"text": "Add Source attributes"}, "ok": True},
        {"op": "type", "action": "type into name", "args": {"text": "Existing_EMI"}, "ok": True},
        {"op": "click", "action": "click 'Save'", "args": {"text": "Save"}, "ok": True},
        {"type": "page", "step": 3, "url": ".../SOURCE_ATTRIBUTES", "page_hash": "h1"},
        {"op": "click", "action": "click 'Add Source attributes'",
         "args": {"text": "Add Source attributes"}, "ok": True},
        {"op": "type", "action": "type into name", "args": {"text": "Credit_Score"}, "ok": True},
        {"op": "click", "action": "click 'Save'", "args": {"text": "Save"}, "ok": True},
        {"type": "page", "step": 6, "url": ".../SOURCE_ATTRIBUTES", "page_hash": "h2"},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    recs = S.read_run_log(rid, log_dir=tmp)
    _check("read all 8 records", len(recs) == 8)
    trail = S.compress_trail(recs)
    _check("trail mentions both attribute values",
           "Existing_EMI" in trail and "Credit_Score" in trail)
    segs = S.segment_iterations(recs)
    _check("segmented into 2 create-iterations (split on Save)", len(segs) == 2)
    _check("each iteration ends at the Save click",
           all(s[-1]["op"] == "click" for s in segs))
    # round-trip the real append_page_record path too.
    S.append_page_record(rid, 9, "http://x/y", "hh", "A ✓", log_dir=tmp)
    _check("append_page_record persisted a 9th line",
           len(S.read_run_log(rid, log_dir=tmp)) == 9)


SNAP_ATTR_FORM = """URL: https://x/newAttribute
- heading "Source Attribute"
- button "Applicant" [ref=e11]
- LabelText
  - StaticText "Attribute Name"
- textbox [ref=e7]: Existing_EMI
  - StaticText "Existing_EMI"
- button "Submit" [ref=e10]
"""


def test_fix_checklist_no_false_tick():
    print("\n[Fix] checklist must NOT tick a half-typed form value (live bug)")
    cl = S.Checklist(["Existing_EMI", "Credit_Score"])
    ticked = cl.mark_present_in(SNAP_ATTR_FORM)   # name only in the input box
    _check("typing a name into the form input does NOT tick it (not saved yet)",
           ticked == 0 and cl.progress() == (0, 2))
    cl.mark_present_in(SNAP_LIST_2)               # now it's a row in the list
    _check("ticks once it appears as a created row (a table cell)",
           "Existing_EMI" in cl.done)


SNAP_CAT_DROPDOWN = """URL: https://x/newAttribute
- button "Select an option" [ref=e11]
- searchbox "Search..." [ref=e22]
- list
  - listitem [level=1]
    - link "Credit_Score" [ref=e75]
  - listitem [level=1]
    - link "Monthly_Income" [ref=e78]
"""


def test_fix_no_tick_from_dropdown():
    print("\n[Fix] a category dropdown must NOT tick attribute items (live bug)")
    cl = S.Checklist(["Credit_Score", "Monthly_Income"])
    n = cl.mark_present_in(SNAP_CAT_DROPDOWN)   # both names exist as CATEGORIES here
    _check("dropdown option names don't tick attribute checklist items",
           n == 0 and cl.progress() == (0, 2))


def test_c1_collapse_dropdown():
    print("\n[Fix/C1] collapse a giant option list (root cause of the bail)")
    lines = ['URL: https://x/form', '- searchbox "Search" [ref=e22]', '- list']
    for i in range(20):
        lines.append("  - listitem [level=1]")
        lines.append(f'    - link "Category_{i}" [ref=e{i}]')
    lines.append('- button "Submit" [ref=e99]')
    out = S.collapse_option_overflow("\n".join(lines), keep=8)
    _check("search box + submit are preserved",
           "Search" in out and "Submit" in out)
    _check("only the first 8 options are kept",
           out.count('link "Category_') == 8)
    _check("summary states the remainder + points to Search",
           "+12 more items" in out and "Search box" in out)
    small = '- list\n  - listitem\n    - link "A" [ref=e1]\n  - listitem\n    - link "B" [ref=e2]'
    _check("a small list is returned unchanged",
           S.collapse_option_overflow(small, keep=8) == small)


def test_fix_exact_match_no_substring():
    print("\n[Fix] checklist exact-cell match (Credit_Score ≠ Credit_Score_BRE)")
    # The dashboard has a policy-API table cell 'Credit_Score_BRE' — it must NOT
    # tick the 'Credit_Score' attribute item (the live false-tick).
    dash = ('- row\n  - cell "Credit_Bureau_BRE" [ref=e35]\n'
            '  - cell "Credit_Score_BRE" [ref=e40]\n')
    cl = S.Checklist(["Credit_Score", "Existing_EMI"])
    cl.mark_present_in(dash)
    _check("'Credit_Score_BRE' cell does NOT tick 'Credit_Score'",
           cl.progress() == (0, 2))
    # the exact cell value DOES tick
    cl.mark_present_in('- cell "Credit_Score" [ref=e9]')
    _check("exact cell value 'Credit_Score' ticks",
           "Credit_Score" in cl.done and "Credit_Score" in cl.last_ticked)
    # space/underscore tolerance: a goal item with SPACES ticks from an
    # underscore-joined created record (the form enforces underscores).
    cl2 = S.Checklist(["Business Rating", "Number of existing loans"])
    cl2.mark_present_in('- cell "Business_Rating" [ref=e1]\n'
                        '- cell "Number_of_existing_loans" [ref=e2]')
    _check("'Business Rating' ticks from cell 'Business_Rating' (space↔_)",
           cl2.progress() == (2, 2))


def test_fastpath_hint():
    print("\n[Auditor] FAST-PATH hint steers to the code path (browse_repeat_last)")
    S.set_context(goal="create A_one, B_two, C_three, D_four", max_steps=30)
    ctx = S.context()
    # no hint before any cycle is done
    _check("no fast-path hint at 0 done", S.repeat_hint(ctx) == "")
    # one created (appears as a cell), 3 remain -> hint appears with the call
    S.after_action("click", "Save", '- cell "A_one" [ref=e9]', run_id="fp",
                   log_dir=tempfile.mkdtemp())
    hint = S.repeat_hint(ctx)
    _check("hint appears after 1 done with >=2 remaining",
           "browse_repeat_last" in hint and "verify=True" in hint)
    _check("hint lists the remaining items + value_was",
           "B_two" in hint and "C_three" in hint and "value_was='A_one'" in hint)
    _check("the header carries the fast-path line",
           "FAST PATH" in S.render_header(ctx))
    S.clear_context()


def test_rule60_overrides():
    print("\n[Rule 60a] expand_template with defaults + per-item override")
    tmpl = [{"action": "type_label", "label": "Attribute Name", "text": "{}"},
            {"action": "click_text", "text": "{category}"},
            {"action": "click_text", "text": "Save"}]
    items = ["Existing_EMI",
             {"value": "Employment_Type", "category": "Employment Details"}]
    out = S.expand_template(tmpl, items, defaults={"category": "Applicant"})
    _check("expanded to 2 items x 3 actions", len(out) == 6)
    _check("scalar item: {} = value, {category} = default 'Applicant'",
           out[0]["text"] == "Existing_EMI" and out[1]["text"] == "Applicant")
    _check("override item: {} = value, {category} overridden",
           out[3]["text"] == "Employment_Type" and out[4]["text"] == "Employment Details")
    _check("constant 'Save' is untouched",
           out[2]["text"] == "Save" and out[5]["text"] == "Save")


def test_rule60_derive_from_log():
    print("\n[Rule 60b] derive a repeatable template from the run log")
    recs = [
        {"op": "click", "action": "click 'Add Source attributes'", "args": {},
         "locator": {"name": "Add Source attributes"}},
        {"op": "type", "action": "type into Attribute Name",
         "args": {"text": "Monthly_Income"},
         "locator": {"role": "textbox", "name": "Attribute Name"}},
        {"op": "click", "action": "click 'Save'", "args": {},
         "locator": {"name": "Save"}},
        {"type": "page", "step": 3, "url": ".../SOURCE_ATTRIBUTES", "page_hash": "h"},
    ]
    tmpl, why = S.derive_template_from_log(recs)
    _check("derived a template (auto-detected the typed value)", tmpl is not None)
    _check("the typed value became a {} placeholder",
           any(a.get("text") == "{}" for a in (tmpl or [])))
    _check("reconstructed REF-FREE (type_label + click_text, no @eN)",
           any(a.get("action") == "type_label" for a in (tmpl or [])) and
           all("ref" not in a for a in (tmpl or [])))
    # ambiguity is reported, not guessed
    recs2 = recs[:1] + [
        {"op": "type", "action": "t", "args": {"text": "AAA"},
         "locator": {"name": "f1"}},
        {"op": "type", "action": "t", "args": {"text": "BBB"},
         "locator": {"name": "f2"}},
        {"op": "click", "action": "click 'Save'", "args": {}, "locator": {"name": "Save"}},
    ]
    t2, why2 = S.derive_template_from_log(recs2)
    _check("two distinct typed values -> asks for value_was (no guess)",
           t2 is None and "ambiguous" in why2)


def test_rule58_should_block():
    print("\n[Rule 58] auditor tier-2 hard-block of a runaway repeat")
    tmp = tempfile.mkdtemp()
    S.set_context(goal="fill the form", max_steps=20)
    _check("no block before any repeat", S.should_block("click", "Save") is None)
    for _ in range(3):  # same click on the SAME unchanged form 3x
        S.after_action("click", "Save", SNAP_FORM, run_id="blk", log_dir=tmp)
    _check("4th identical click is BLOCKED",
           (S.should_block("click", "Save") or "").startswith("BLOCKED"))
    _check("a DIFFERENT click is NOT blocked",
           S.should_block("click", "Cancel") is None)
    _check("scroll is never blocked (legit repeat)",
           S.should_block("scroll", "down") is None)
    S.clear_context()
    _check("no block when no run active", S.should_block("click", "Save") is None)


def test_rule61_iteration_deviation():
    print("\n[Rule 61] baseline deviation of a later iteration")
    base = SNAP_LIST_1
    _check("benign new data row -> NO deviation",
           S.iteration_deviation(base, SNAP_LIST_2) == "")
    err = base + '- StaticText "Attribute name already exists"\n'
    _check("a new error/alert -> deviation",
           "already exists" in S.iteration_deviation(base, err).lower())
    _check("stuck on the form (url changed) -> deviation",
           S.iteration_deviation(base, SNAP_FORM).startswith("ended on"))



# ===========================================================================
#  The trail is scoped to the CURRENT goal, and records FAILURE
# ===========================================================================
TRAIL_2GOALS = [
    {"step": 2, "url": "https://x/login", "op": "click_text",
     "target": "Sign in", "goal": 1, "ok": True, "err": ""},
    {"step": 3, "url": "https://x/login", "op": "click_text",
     "target": "Sign in", "goal": 1, "ok": False, "err": "Password is required."},
    {"step": 4, "url": "https://x/dashboard", "op": "click_text",
     "target": "Source Category", "goal": 2, "ok": True, "err": ""},
    {"step": 5, "url": "https://x/newCategory", "op": "type_label",
     "target": "Category Name=sav_account", "goal": 2, "ok": True, "err": ""},
    {"step": 6, "url": "https://x/newCategory", "op": "click_text",
     "target": "Submit", "goal": 2, "ok": False,
     "err": "Category Description is required."},
]


def test_trail_is_goal_scoped():
    print("\ntrail: scoped to the goal in flight")
    out = S.compress_trail_inmem(TRAIL_2GOALS, goal=2)
    # Goal 2's own steps are all there.
    for step in ("4:", "5:", "6:"):
        _check(f"goal 2 step {step} present", step in out)
    # Goal 1's SUCCESSFUL step is not -- its outcome travels as a one-line
    # result in the goal message instead of replaying its steps every turn.
    _check("goal 1's successful step is dropped", "2:" not in out)
    # ...but its FAILURE survives: it usually explains the shape of this goal.
    _check("goal 1's failure is carried over", "Password is required." in out)

    # No goal running (unsupervised browse) -> the old last-N behaviour.
    plain = S.compress_trail_inmem(TRAIL_2GOALS)
    _check("ungated trail keeps every step", "2:" in plain and "6:" in plain)


def test_trail_shows_failure():
    print("\ntrail: a step that errored does not read like one that worked")
    out = S.compress_trail_inmem(TRAIL_2GOALS, goal=2)
    _check("the failure is marked", "\u2717" in out)
    _check("and says why, in the page's words",
           "Category Description is required." in out)
    # The bug this replaces: the entry carried no ok/err field at all, so
    # "clicked Submit" and "clicked Submit and the form refused" were the same
    # string. Anything without the marker must be a step that worked.
    ok_only = S.compress_trail_inmem(
        [t for t in TRAIL_2GOALS if t["ok"]], goal=2)
    _check("a clean trail carries no marker", "\u2717" not in ok_only)


def test_after_action_records_the_outcome():
    print("\nobserve: the outcome is recorded, not just the action")
    import tempfile as _tf
    d = _tf.mkdtemp()
    ctx = S.set_context(goal="t", max_steps=20, items=[])
    S.set_goal(1, 2, "Creation of a category")
    page = ('URL: https://x/newCategory\n'
            'textbox "Category Name" "sav_account"\n'
            'textbox "Description" []\n'
            'Category Description is required.\n'
            'button "Submit"')
    S.after_action("click", "Submit", page, log_dir=d)
    last = ctx.trail[-1]
    _check("the step is tagged with its goal", last["goal"] == 1)
    _check("it is recorded as failed", last["ok"] is False)
    _check("carrying the page's own reason",
           last["err"] == "Category Description is required.")

    clean = 'URL: https://x/list\ncell "sav_account"\nbutton "Add"'
    S.after_action("click", "Add", clean, log_dir=d)
    _check("a step that worked is recorded ok", ctx.trail[-1]["ok"] is True)
    ctx.active = False


def test_fastpath_not_offered_on_goals():
    print("\nFAST PATH: goals are not form values")
    ctx = S.set_context(goal="t", max_steps=40, items=[])
    goals = ["Successful login to the portal",
             "Creation of a category named 'sav_account'",
             "Creation of the attributes under it"]
    ctx.set_items(goals, as_goals=True)
    ctx.checklist.mark_done(goals[0])
    # It used to emit browse_repeat_last(items=["Creation of a category named
    # 'sav_account'", ...]) -- an instruction to type the goal SENTENCE into
    # the form. Same class as the checklist/evidence mismatch.
    _check("no fast path when the checklist holds goals",
           S.repeat_hint(ctx) == "")
    _check("and the header does not mention one",
           "FAST PATH" not in S.render_header(ctx))

    # Real record values still get the hint -- that is what it is for.
    ctx.set_items(["Monthly_Income", "Existing_EMI", "Credit_Score"])
    ctx.checklist.mark_done("Monthly_Income")
    hint = S.repeat_hint(ctx)
    _check("values still get the fast path", "browse_repeat_last" in hint)
    _check("naming the remaining items", "Existing_EMI" in hint)
    ctx.active = False


def main():
    test_p3_hash_and_diff()
    test_p4_checklist()
    test_p6_loop_monitor()
    test_p1_p5_context_and_header()
    test_p6_correction_in_header()
    test_p2_log_reader()
    test_fix_checklist_no_false_tick()
    test_fix_no_tick_from_dropdown()
    test_c1_collapse_dropdown()
    test_fix_exact_match_no_substring()
    test_fastpath_hint()
    test_rule60_overrides()
    test_rule60_derive_from_log()
    test_rule58_should_block()
    test_rule61_iteration_deviation()
    test_trail_is_goal_scoped()
    test_trail_shows_failure()
    test_after_action_records_the_outcome()
    test_fastpath_not_offered_on_goals()
    print(f"\n{'ALL PASSED' if _check.failed == 0 else str(_check.failed)+' FAILED'}")
    sys.exit(1 if _check.failed else 0)


if __name__ == "__main__":
    main()
