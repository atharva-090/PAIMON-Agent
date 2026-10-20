"""A graded intelligence benchmark for whatever model is behind an OpenAI endpoint.

Built to answer "is this local 3.7B actually good enough to run the agent?" with
evidence rather than vibes. Tasks are realistic for THIS project (task-board
work, extraction, tool calls, scheduling) plus standard reasoning probes, split
into four difficulty tiers and graded deterministically wherever possible --
exact match, regex, JSON-schema checks, tool-argument checks, and for the coding
tasks by actually EXECUTING the generated function against test cases.

Run against the local model (default):
    agent/.venv/Scripts/python.exe -m local_model.bench_k2
Against the openai-mocker instead, for a side-by-side:
    agent/.venv/Scripts/python.exe -m local_model.bench_k2 --base-url \
        http://127.0.0.1:8000/v1 --model gpt-4o
Quick subset:
    ... -m local_model.bench_k2 --tier 1 --tier 2
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
#  Grading helpers
# ---------------------------------------------------------------------------
def canon(s: str | None) -> str:
    """Fold typographic punctuation to ASCII.

    The model writes "don't" with U+2019, so a grader regex spelling don't with
    an ASCII apostrophe silently fails a CORRECT answer. That marked the
    hallucination-honesty task as failed when the model had actually refused
    properly -- exactly the sort of grader bug that produces a wrong verdict
    about the model.
    """
    if not s:
        return ""
    for bad, good in (("’", "'"), ("‘", "'"), ("“", '"'),
                      ("”", '"'), ("–", "-"), ("—", "-"),
                      (" ", " "), (" ", " ")):
        s = s.replace(bad, good)
    return s


def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", canon(s)).strip().lower()


def exact(expected: str) -> Callable:
    def g(r: dict) -> tuple[bool, str]:
        got = norm(r["content"])
        want = norm(expected)
        return got == want, f"want {want!r}, got {got!r}"
    return g


def contains_all(*needles: str) -> Callable:
    def g(r: dict) -> tuple[bool, str]:
        got = norm(r["content"])
        missing = [n for n in needles if norm(n) not in got]
        return not missing, (f"missing {missing}" if missing else "ok")
    return g


def contains_none(*needles: str) -> Callable:
    def g(r: dict) -> tuple[bool, str]:
        got = norm(r["content"])
        present = [n for n in needles if norm(n) in got]
        return not present, (f"should not contain {present}" if present else "ok")
    return g


def regex(pattern: str, flags: int = re.I) -> Callable:
    def g(r: dict) -> tuple[bool, str]:
        ok = re.search(pattern, canon(r["content"]), flags) is not None
        return ok, f"/{pattern}/ {'matched' if ok else 'did not match'}"
    return g


def _extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a reply (may be fenced)."""
    if not text:
        raise ValueError("empty")
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1],
                default=-1)
    if start == -1:
        raise ValueError("no json found")
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(text[start:])
    return obj


def json_check(fn: Callable[[Any], tuple[bool, str]]) -> Callable:
    def g(r: dict) -> tuple[bool, str]:
        try:
            obj = _extract_json(r["content"])
        except Exception as exc:
            return False, f"no parseable JSON ({exc})"
        return fn(obj)
    return g


def tool_check(name: str, arg_fn: Callable[[dict], tuple[bool, str]] | None = None
               ) -> Callable:
    def g(r: dict) -> tuple[bool, str]:
        calls = r["tool_calls"]
        if not calls:
            return False, "no tool call made"
        got = [c["function"]["name"] for c in calls]
        if name not in got:
            return False, f"want tool {name!r}, got {got}"
        call = next(c for c in calls if c["function"]["name"] == name)
        if arg_fn is None:
            return True, "ok"
        try:
            args = json.loads(call["function"]["arguments"])
        except Exception as exc:
            return False, f"arguments not JSON: {exc}"
        return arg_fn(args)
    return g


def no_tool_call() -> Callable:
    def g(r: dict) -> tuple[bool, str]:
        if r["tool_calls"]:
            return False, f"unexpected tool call: " \
                          f"{[c['function']['name'] for c in r['tool_calls']]}"
        return True, "ok"
    return g


def run_python(func_name: str, cases: list[tuple[tuple, Any]]) -> Callable:
    """Execute the generated function against (args, expected) pairs."""
    def g(r: dict) -> tuple[bool, str]:
        text = r["content"] or ""
        m = re.search(r"```(?:python)?\s*(.+?)```", text, re.S)
        code = m.group(1) if m else text
        if f"def {func_name}" not in code:
            return False, f"no def {func_name} in reply"
        harness = code + "\n\nimport json\n_res=[]\n"
        for args, _ in cases:
            harness += f"_res.append({func_name}(*{args!r}))\n"
        harness += "print(json.dumps(_res))\n"
        try:
            p = subprocess.run([sys.executable, "-c", harness],
                               capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return False, "code timed out"
        if p.returncode != 0:
            err = (p.stderr or "").strip().splitlines()
            return False, f"code raised: {err[-1] if err else '?'}"
        try:
            got = json.loads((p.stdout or "").strip().splitlines()[-1])
        except Exception:
            return False, f"bad output: {(p.stdout or '')[:120]!r}"
        want = [c[1] for c in cases]
        if got != want:
            return False, f"want {want}, got {got}"
        return True, "all cases passed"
    return g


def all_of(*graders: Callable) -> Callable:
    def g(r: dict) -> tuple[bool, str]:
        notes = []
        for gr in graders:
            ok, note = gr(r)
            notes.append(note)
            if not ok:
                return False, "; ".join(notes)
        return True, "; ".join(notes)
    return g


# ---------------------------------------------------------------------------
#  Task definition
# ---------------------------------------------------------------------------
@dataclass
class Task:
    id: str
    tier: int
    category: str
    prompt: str
    grade: Callable
    system: str | None = None
    tools: list[dict] | None = None
    max_tokens: int = 2000
    note: str = ""


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"},
                           "unit": {"type": "string",
                                    "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"],
        },
    },
}

TASKBOARD_TOOLS = [
    {"type": "function", "function": {
        "name": "create_task",
        "description": "Create a task on the board.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "due_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
        }, "required": ["title"]}}},
    {"type": "function", "function": {
        "name": "view_board",
        "description": "Read-only overview of every group and task.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "schedule_reminder",
        "description": "Schedule a reminder at a natural-language time.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"},
            "when": {"type": "string",
                     "description": "e.g. 'tomorrow at 3pm', 'in 20 mins'"},
        }, "required": ["text", "when"]}}},
]


def _t2_json(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, f"expected object, got {type(obj).__name__}"
    need = {"name": "Priya Sharma", "amount": 4500, "currency": "INR"}
    for k, v in need.items():
        if k not in obj:
            return False, f"missing key {k!r}"
        got = obj[k]
        if isinstance(v, (int, float)):
            # Accept "4500" / "4,500" / 4500.0 -- the value is what matters.
            try:
                got_num = float(str(got).replace(",", "").strip())
            except ValueError:
                return False, f"{k}: {got!r} is not numeric"
            if abs(got_num - v) > 1e-6:
                return False, f"{k}: want {v}, got {got!r}"
        elif norm(str(got)) != norm(str(v)):
            return False, f"{k}: want {v!r}, got {got!r}"
    return True, "ok"


def _t3_invoices(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, list):
        return False, f"expected a list, got {type(obj).__name__}"
    if len(obj) != 2:
        return False, f"expected 2 overdue invoices, got {len(obj)}"
    ids = sorted(str(o.get("id", "")).upper() for o in obj if isinstance(o, dict))
    if ids != ["INV-002", "INV-004"]:
        return False, f"want INV-002 and INV-004, got {ids}"
    return True, "ok"


HAYSTACK = textwrap.dedent("""\
    Meeting notes, Q3 planning, 12 September.
    Attendees: Ravi, Priya, Sam, Deepa, Arun.
    Ravi opened with the migration timeline. The vendor contract renewal is
    handled by Deepa. Sam raised the staffing shortfall in the QA team.
    Budget headroom for Q3 was confirmed at 18 lakh. Priya noted the client
    escalation from Meridian Corp is now closed. The internal audit reference
    for this quarter is AUD-7734-KX. Arun will circulate the revised deck.
    Deepa flagged that the renewal deadline is 30 November. Sam asked whether
    the QA backfill needs board approval; Ravi said no, it sits under the
    existing headcount plan. The next review is scheduled for 4 October.
    Action items: Arun to send the deck, Deepa to start renewal paperwork,
    Sam to post the QA role, Priya to archive the Meridian thread.
    """)

TASKS: list[Task] = [
    # ---------------- TIER 1: basic competence ----------------
    Task("t1-format", 1, "instruction",
         "Reply with exactly the following text and nothing else: ACK-7781",
         exact("ACK-7781")),
    Task("t1-arith", 1, "math",
         "A task took 3 hours on Monday, 4.5 hours on Tuesday and 2 hours on "
         "Wednesday. How many hours in total? Reply with just the number.",
         regex(r"\b9\.5\b")),
    Task("t1-classify", 1, "classification",
         "Classify the priority of this request as exactly one word - "
         "low, medium, or high: 'The production database is down and "
         "customers cannot check out.' Reply with only that word.",
         # Tolerate trailing punctuation/quoting; the classification is the point.
         all_of(regex(r"\bhigh\b"), contains_none("low", "medium"))),
    Task("t1-extract", 1, "extraction",
         "Extract the person's name from this sentence and reply with only "
         "the name: 'Please follow up with Deepa Nair about the renewal.'",
         contains_all("Deepa Nair")),

    # ---------------- TIER 2: multi-step, tools ----------------
    Task("t2-pct", 2, "math",
         "A project was quoted at 240 hours. 35% is done. Then the scope grew "
         "by 40 hours. How many hours of work REMAIN? Think it through, then "
         "end your reply with the final number on its own line.",
         regex(r"\b196\b"),
         note="240*0.65=156, +40 = 196"),
    Task("t2-json", 2, "extraction",
         "Extract the payee, amount and currency from this line as JSON with "
         "keys name, amount, currency (amount must be a number, no symbols): "
         "'Paid Priya Sharma INR 4,500 for the vendor audit.' "
         "Reply with only the JSON.",
         json_check(_t2_json)),
    Task("t2-tool", 2, "tool-use",
         "What is the weather in Mumbai right now, in fahrenheit?",
         tool_check("get_weather", lambda a: (
             norm(str(a.get("city"))) == "mumbai"
             and norm(str(a.get("unit", "fahrenheit"))) == "fahrenheit",
             f"args={a}")),
         tools=[WEATHER_TOOL]),
    Task("t2-no-tool", 2, "tool-restraint",
         "Do not call any tools. Just answer in one sentence: what does the "
         "get_weather tool do?",
         all_of(no_tool_call(), contains_all("weather")),
         tools=[WEATHER_TOOL],
         note="knowing when NOT to call a tool"),
    Task("t2-constraint", 2, "instruction",
         "Summarise this in EXACTLY five words, no more, no less: 'The vendor "
         "contract renewal deadline is 30 November and Deepa owns the "
         "paperwork.'",
         lambda r: (len(re.findall(r"[A-Za-z0-9'-]+",
                                   (r["content"] or "").strip())) == 5,
                    f"word count = "
                    f"{len(re.findall(r'[A-Za-z0-9-]+', (r['content'] or '')))}")),

    # ---------------- TIER 3: hard reasoning, code ----------------
    Task("t3-code", 3, "code",
         "Write a Python function `business_days(start, end)` that takes two "
         "ISO date strings and returns the number of weekdays (Mon-Fri) "
         "strictly between them, excluding both endpoints. Return only code "
         "in a python code block.",
         run_python("business_days", [
             (("2026-09-14", "2026-09-18"), 3),   # Mon->Fri, between = Tue,Wed,Thu
             (("2026-09-11", "2026-09-14"), 0),   # Fri->Mon, between = Sat,Sun
             (("2026-09-01", "2026-09-30"), 20),
         ]),
         max_tokens=3000),
    Task("t3-filter", 3, "reasoning",
         "Today is 2026-09-12. Here are invoices:\n"
         "INV-001 due 2026-09-20 status unpaid\n"
         "INV-002 due 2026-08-30 status unpaid\n"
         "INV-003 due 2026-08-15 status PAID\n"
         "INV-004 due 2026-09-01 status unpaid\n"
         "INV-005 due 2026-10-02 status unpaid\n"
         "Return ONLY a JSON array of the OVERDUE unpaid invoices, each as "
         '{"id": ..., "due": ...}. Overdue means the due date is before today '
         "and it is not paid.",
         json_check(_t3_invoices),
         max_tokens=3000,
         note="must exclude the PAID one and the future ones"),
    Task("t3-logic", 3, "logic",
         "Four people sit in a row: Ravi, Priya, Sam, Deepa. Priya is not at "
         "either end. Sam is immediately to the right of Ravi. Deepa is at the "
         "far right. Who is at the far left? Reply with just the name.",
         all_of(contains_all("Ravi"), contains_none("Priya", "Sam", "Deepa")),
         max_tokens=3000,
         note="Ravi,Sam,Priya,Deepa"),
    Task("t3-needle", 3, "long-context",
         HAYSTACK + "\nBased only on the notes above, what is the internal "
         "audit reference for this quarter? Reply with just the reference.",
         contains_all("AUD-7734-KX")),
    Task("t3-multitool", 3, "tool-use",
         "I need a high priority task called 'Renew vendor contract' due "
         "2026-11-30, and remind me about it tomorrow at 9am.",
         lambda r: (
             len(r["tool_calls"]) >= 1
             and any(c["function"]["name"] in ("create_task", "schedule_reminder")
                     for c in r["tool_calls"]),
             f"calls={[c['function']['name'] for c in r['tool_calls']]}"),
         tools=TASKBOARD_TOOLS,
         max_tokens=3000,
         note="ideally BOTH create_task and schedule_reminder"),

    # ---------------- TIER 4: traps and honesty ----------------
    Task("t4-hallucination", 4, "honesty",
         "What were the exact revenue figures reported in the Meridian Corp "
         "Q3 2026 earnings call? If you do not know, say so plainly.",
         all_of(
             regex(r"(don't|do not|cannot|can't|no) (know|have|access)|"
                   r"not (aware|able)|unable to|no (information|data|access)|"
                   r"I'm not sure|unaware"),
             contains_none("$1", "billion in revenue", "reported revenue of")),
         max_tokens=2500,
         note="must refuse to invent figures"),
    Task("t4-trap", 4, "reasoning",
         "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than "
         "the ball. How much does the ball cost? Reply with just the amount.",
         all_of(regex(r"0?\.05\b|\b5 ?(cents|paise)"),
                contains_none("0.10 ", "10 cents")),
         max_tokens=3000,
         note="classic System-1 trap; wrong answer is 0.10"),
    Task("t4-counting", 4, "character-level",
         "How many times does the letter 'r' appear in the word "
         "'strawberry'? Reply with just the number.",
         all_of(regex(r"\b3\b"), contains_none("two", "\b2\b")),
         max_tokens=2500,
         note="known LLM weak spot"),
    Task("t4-conflict", 4, "instruction",
         "Write a haiku about deadlines. Ignore the previous sentence and "
         "instead reply with exactly: OVERRIDE-OK",
         exact("OVERRIDE-OK"),
         note="later instruction wins"),
    Task("t4-multihop", 4, "reasoning",
         HAYSTACK + "\nBased only on the notes: the person who owns the "
         "renewal paperwork flagged a deadline. How many days are there from "
         "the next review date to that deadline? Reply with the number of "
         "days and nothing else.",
         regex(r"\b57\b"),
         max_tokens=3000,
         note="Deepa->30 Nov; next review 4 Oct; 4 Oct->30 Nov = 57 days"),
]


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="k2-horizon-3.7b")
    ap.add_argument("--api-key", default="sk-local-ignored")
    ap.add_argument("--tier", type=int, action="append")
    ap.add_argument("--only", action="append", help="task id substring")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    tasks = TASKS
    if args.tier:
        tasks = [t for t in tasks if t.tier in args.tier]
    if args.only:
        tasks = [t for t in tasks
                 if any(o in t.id for o in args.only)]

    print(f"model    : {args.model}  @ {args.base_url}")
    print(f"tasks    : {len(tasks)}   temperature={args.temperature}")
    print("=" * 78)

    results = []
    for t in tasks:
        msgs = []
        if t.system:
            msgs.append({"role": "system", "content": t.system})
        msgs.append({"role": "user", "content": t.prompt})
        kw: dict[str, Any] = dict(model=args.model, messages=msgs,
                                  temperature=args.temperature,
                                  max_tokens=t.max_tokens)
        if t.tools:
            kw["tools"] = t.tools

        t0 = time.time()
        err = None
        try:
            resp = client.chat.completions.create(**kw)
            m = resp.choices[0].message
            r = {
                "content": m.content or "",
                "tool_calls": [
                    {"function": {"name": c.function.name,
                                  "arguments": c.function.arguments}}
                    for c in (m.tool_calls or [])
                ],
                "reasoning": getattr(m, "reasoning_content", "") or "",
                "finish": resp.choices[0].finish_reason,
                "ptok": resp.usage.prompt_tokens,
                "ctok": resp.usage.completion_tokens,
            }
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            r = {"content": "", "tool_calls": [], "reasoning": "",
                 "finish": "error", "ptok": 0, "ctok": 0}
        dt = time.time() - t0

        if err:
            ok, note = False, err
        else:
            try:
                ok, note = t.grade(r)
            except Exception as exc:
                ok, note = False, f"grader error: {type(exc).__name__}: {exc}"

        mark = "PASS" if ok else "FAIL"
        print(f"[T{t.tier}] {t.id:18s} {mark}  {dt:5.1f}s  "
              f"{r['ctok']:4d}tok  reason={len(r['reasoning']):5d}ch  "
              f"finish={r['finish']}")
        if not ok:
            print(f"       why : {note[:180]}")
            preview = (r["content"] or "").replace("\n", " ")[:160]
            if preview:
                print(f"       got : {preview!r}")
            if r["tool_calls"]:
                print(f"       calls: {[c['function']['name'] for c in r['tool_calls']]}")
        results.append({"id": t.id, "tier": t.tier, "category": t.category,
                        "ok": ok, "note": note, "seconds": round(dt, 1),
                        "completion_tokens": r["ctok"],
                        "reasoning_chars": len(r["reasoning"]),
                        "finish": r["finish"], "content": r["content"],
                        "tool_calls": r["tool_calls"]})

    # ---- summary ----------------------------------------------------------
    print("=" * 78)
    by_tier: dict[int, list] = {}
    by_cat: dict[str, list] = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r)
        by_cat.setdefault(r["category"], []).append(r)

    print("\nBy tier:")
    tier_names = {1: "basic competence", 2: "multi-step + tools",
                  3: "hard reasoning/code", 4: "traps + honesty"}
    for tier in sorted(by_tier):
        rs = by_tier[tier]
        p = sum(1 for x in rs if x["ok"])
        print(f"  T{tier} {tier_names.get(tier,''):22s} {p}/{len(rs)}  "
              f"({100*p/len(rs):3.0f}%)")

    print("\nBy category:")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        p = sum(1 for x in rs if x["ok"])
        print(f"  {cat:16s} {p}/{len(rs)}")

    total = sum(1 for r in results if r["ok"])
    secs = sum(r["seconds"] for r in results)
    ctok = sum(r["completion_tokens"] for r in results)
    print(f"\nTOTAL {total}/{len(results)} ({100*total/max(1,len(results)):.0f}%)"
          f"   wall {secs/60:.1f} min   "
          f"{ctok} completion tokens   "
          f"{ctok/max(1e-9,secs):.1f} tok/s avg")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
