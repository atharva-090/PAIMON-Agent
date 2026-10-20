"""Stage B: the Jev policy, the ref mapping, and the autopilot loop.

    .venv/Scripts/python.exe test_jev_policy.py

No network. TypeSafe and the text helper are both stubbed, so this checks the
things that would be dangerous or expensive to discover live: that a model
answer can never become a selector we did not offer, and that the loop stops.
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import jev_policy as J

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


SNAP = (
    "URL: https://portal.test/newAttribute\n\n"
    'CONTROLS — act on one by its ref, e.g. browse_click(ref="e6")\n'
    "[e4]    button     Collapse\n"
    "[e7]    textbox    Attribute Name\n"
    "[e8]    textbox    Description\n"
    "[e9]    combobox   Data Type  · expanded=false\n"
    "[e11]   button     Submit\n"
    "[e14]   row       01 | ATT_174 | Business_Status\n"
    "\nPAGE TEXT\nAdd Source attributes\nCategory\nRetail_UL\n")


# ===========================================================================
#  the action space and the ref mapping
# ===========================================================================
print("the element table becomes Jev's action space")
els, tgts, refs = J.action_space(SNAP)

check("rows are context, not controls", len(els), 5)
check("the indexes map onto our own refs", refs,
      {"1": "e4", "2": "e7", "3": "e8", "4": "e9", "5": "e11"})
check("only editable roles get TYPE_TEXT",
      sorted(tgts["TYPE_TEXT"]), ["2", "3", "4"])
check("everything is clickable", len(tgts["CLICK"]), 5)
truthy("state travels with the element",
       any(e.get("state") == "expanded=false" for e in els))
check("the page url is extracted",
      J.page_context(SNAP)["url"], "https://portal.test/newAttribute")
truthy("the page text is extracted", "Retail_UL" in J.page_context(SNAP)["text"])
check("an empty page offers nothing", J.action_space("")[0], [])


# ===========================================================================
#  the guard — a model answer must never become a selector
# ===========================================================================
print()
print("a malformed or out-of-range answer executes nothing")

GOOD = {"choice": "1", "probabilities": {"1": 0.9, "2": 0.1},
        "confidence": 0.9}
truthy("a well-formed answer passes",
       J.validate_choice(dict(GOOD), {"1": 1, "2": 1})["choice"] == "1")

BAD = [
    ("an index we never offered",
     {"choice": "99", "probabilities": {"1": 0.9, "2": 0.1}, "confidence": 1}),
    ("a distribution over the wrong set",
     {"choice": "1", "probabilities": {"1": 1.0}, "confidence": 1}),
    ("probabilities that do not sum to 1",
     {"choice": "1", "probabilities": {"1": 0.2, "2": 0.2}, "confidence": 1}),
    ("a choice that is not the argmax",
     {"choice": "2", "probabilities": {"1": 0.9, "2": 0.1}, "confidence": 1}),
    ("a confidence out of range",
     {"choice": "1", "probabilities": {"1": 0.9, "2": 0.1}, "confidence": 7}),
    ("a non-numeric probability",
     {"choice": "1", "probabilities": {"1": "x", "2": 0.1}, "confidence": 1}),
    ("nothing at all", {}),
]
for label, ans in BAD:
    try:
        J.validate_choice(ans, {"1": 1, "2": 1})
        FAILS.append(label)
        print(f"  FAIL {label}: was accepted")
    except ValueError:
        print(f"  ok   {label}: refused")


# ===========================================================================
#  one decision, end to end, against a stubbed provider
# ===========================================================================
print()
print("a decision maps back to one of our refs")

POSTS = []


def fake_post(url, key, body, timeout=30, who=""):
    POSTS.append((url, body))
    if "typesafe" in url:
        ops = list(body["questions"]["operation"]["criteria"])
        tt = list(body["questions"]["type_text_target"]["criteria"])
        return {"model": "jev-test", "usage": {"input_tokens": 600},
                "answers": {
                    "operation": {"choice": "TYPE_TEXT", "confidence": 0.95,
                                  "probabilities": {o: (0.95 if o == "TYPE_TEXT"
                                                        else 0.05 / (len(ops) - 1))
                                                    for o in ops}},
                    "type_text_target": {
                        "choice": "2", "confidence": 0.9,
                        "probabilities": {t: (0.9 if t == "2"
                                              else 0.1 / (len(tt) - 1))
                                          for t in tt}}}}
    return {"choices": [{"message": {"content": '{"text": "Monthly_income"}'}}],
            "usage": {"total_tokens": 120}}


_real = J._post
J._post = fake_post
try:
    d = J.choose(SNAP, "create an attribute named Monthly_income", [])
    check("the operation comes back", d["operation"], "TYPE_TEXT")
    check("the index is resolved to OUR ref", d["ref"], "e7")
    check("and it is the field it named", d["label"], "Attribute Name")
    truthy("the usage is recorded", d["usage"].get("input_tokens") == 600)

    url, body = POSTS[0]
    truthy("one request carries every head",
           {"operation", "click_target", "type_text_target"} <= set(
               body["questions"]))
    check("the page travels with it",
          body["state"]["page"]["url"], "https://portal.test/newAttribute")
    truthy("so does the goal",
           "Monthly_income" in body["questions"]["operation"]
           ["instructions"]["goal"])

    txt, meta = J.field_text("create Monthly_income",
                             {"label": "Attribute Name", "role": "textbox"},
                             J.page_context(SNAP), [])
    check("the text helper returns the value", txt, "Monthly_income")
    truthy("and is pointed at our own model",
           "typesafe" not in POSTS[-1][0])
    truthy("with reasoning off, because this is a lookup",
           POSTS[-1][1].get("reasoning_effort") == "none")

    # A helper that will not commit must type nothing, not a guess.
    J._post = lambda u, k, b, timeout=30, who="": (
        {"choices": [{"message": {"content": '{"text": null}'}}]})
    check("a refusing helper types nothing",
          J.field_text("g", {}, {}, [])[0], "")
    J._post = lambda u, k, b, timeout=30, who="": (
        {"choices": [{"message": {"content": 'I think it is Monthly_income'}}]})
    check("so does a chatty one", J.field_text("g", {}, {}, [])[0], "")
finally:
    J._post = _real


# ===========================================================================
#  the autopilot loop
# ===========================================================================
print()
print("the loop executes through our own tools, and stops")
import browser_tools as B                                        # noqa: E402

truthy("the tool exists", callable(getattr(B, "browse_autopilot", None)))
truthy("it is offered only with a key", B.autopilot_available() == J.enabled())
check("the step cap is configurable", B.AUTOPILOT_MAX_STEPS, 14)

CLICKS = []
PAGE = [SNAP]


def fake_click(ref):
    CLICKS.append(("click", ref))
    PAGE[0] = SNAP.replace("newAttribute", "saved")
    return {"content": [{"type": "text", "text": "OK: click @" + ref}]}


def fake_type(ref, text):
    CLICKS.append(("type", ref, text))
    return {"content": [{"type": "text", "text": "OK: typed"}]}


_rc, _rt, _rs, _rst = (B.browse_click, B.browse_type, B._snap, B._snapshot_text)
B.browse_click, B.browse_type = fake_click, fake_type
B._snap = lambda *a, **k: PAGE[0]
B._snapshot_text = lambda: PAGE[0]

SEQ = ["TYPE_TEXT", "CLICK", "DONE"]
STEP = [0]


def scripted(snapshot, goal, history, label_hints=None):
    op = SEQ[min(STEP[0], len(SEQ) - 1)]
    STEP[0] += 1
    if op == "DONE":
        return {"operation": "DONE", "ref": "", "index": "", "label": "",
                "role": "", "latency_ms": 7, "usage": {}}
    ref = "e7" if op == "TYPE_TEXT" else "e11"
    return {"operation": op, "ref": ref, "index": "2", "label": "Submit",
            "role": "button", "latency_ms": 7, "usage": {}}


_rch, _rft, _rnl = J.choose, J.field_text, J.needs_labels
J.needs_labels = lambda els: False   # no DOM probe in a stubbed run
J.choose = scripted
J.field_text = lambda *a, **k: ("Monthly_income", {"latency_ms": 5})
try:
    out = B.browse_autopilot("create an attribute named Monthly_income")
    body = B.chunk_text(out)
    truthy("it reports each step", "1. " in body and "2. " in body)
    truthy("it says why it stopped", "Stopped:" in body)
    truthy("DONE ends the loop", "satisfied" in body)
    check("it typed through our own tool", CLICKS[0], ("type", "e7",
                                                       "Monthly_income"))
    check("and clicked through ours", CLICKS[1], ("click", "e11"))
    check("it did not run past DONE", len(CLICKS), 2)

    # A goal with no detail is the caller's bug, and a cheap one to catch.
    short = B.chunk_text(B.browse_autopilot("go"))
    truthy("a vague goal is refused", "concrete outcome" in short)

    # The step cap must hold even when the policy never says DONE.
    CLICKS.clear()
    STEP[0] = 0
    SEQ[:] = ["CLICK"]
    body = B.chunk_text(B.browse_autopilot("create something", max_steps=3))
    check("the cap is honoured", len(CLICKS), 3)
    truthy("and it says so", "ran out of steps" in body)

    # The A4 gate refusing an action must stop the loop, not be clicked past.
    CLICKS.clear()
    B.browse_click = lambda ref: (CLICKS.append(("click", ref)) or
                                  {"content": [{"type": "text",
                                                "text": "REFUSED — not run."}]})
    body = B.chunk_text(B.browse_autopilot("create something", max_steps=6))
    check("the wedge gate stops the autopilot", len(CLICKS), 1)
    truthy("and it says the page is stuck", "stuck" in body)

    # A provider outage must end the turn with a report, never an exception.
    J.choose = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("TypeSafe unreachable: timed out"))
    body = B.chunk_text(B.browse_autopilot("create something"))
    truthy("an outage is reported, not raised", "autopilot error" in body)
finally:
    J.choose, J.field_text, J.needs_labels = _rch, _rft, _rnl
    B.browse_click, B.browse_type = _rc, _rt
    B._snap, B._snapshot_text = _rs, _rst

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
    sys.exit(1)
print("all Stage B checks passed")
