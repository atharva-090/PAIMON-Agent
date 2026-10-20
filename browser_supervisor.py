"""Supervisory primitives for browser runs — the Section-K foundations (P1-P6).

This module is deliberately DEPENDENCY-FREE (stdlib + config only, no agentscope)
so every primitive is unit-testable without the agent runtime. browser_tools.py
and the swarm runner wire these in as thin glue; the logic lives here.

What's inside (maps 1:1 to the prerequisite list in AUTOMATION_RULES.md):

  P1  RunContext           — per-run state (goal, step, checklist, url, trail,
                             monitor) the tools read; set by the runner.
  P2  log reader/compressor— read_run_log / compress_trail / segment_iterations,
                             plus a durable per-step page-state record.
  P3  snapshot hash + diff — normalize_snapshot / snapshot_hash / diff_snapshots:
                             the "did the page actually change?" primitive.
  P4  Checklist            — parse goal items, tick them DETERMINISTICALLY by
                             reading the page (not the brain's say-so).
  P5  header rendering     — render_header(): the GOAL/CHECKLIST/TRAIL/STEP block
                             appended to the snapshot (mock-safe, never a history
                             rewrite).
  P6  LoopMonitor          — rolling (url,op,target,page_hash) window; flag()
                             detects repeat/stall loops; the auditor's eyes.

The single integration entry point is ``after_action(...)``: browser_tools calls
it once per action with the fresh snapshot, and it updates the context, ticks the
checklist, feeds the monitor, writes the durable page record, and returns the
header to prepend. When no context is active (runner didn't opt in) it returns
"" so unwired code paths are completely unaffected.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import config


# ===========================================================================
#  P3 — snapshot normalization, hashing, diff
# ===========================================================================
# Both spellings of a ref: the raw tree's "[ref=e6]" and the compact element
# table's leading "[e6]". Missing the second one would leave refs inside the
# hash, every re-render of an unchanged page would look different, and the
# whole no-change/stall detection would go quiet.
_REF_RE = re.compile(r"\[(?:ref=)?e\d+\]")
_LEVEL_RE = re.compile(r"\[level=\d+\]")
_DATE_RE = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
_NUM_RE = re.compile(r"\b\d+\b")
_WS_RE = re.compile(r"\s+")


def normalize_snapshot(snap: str) -> str:
    """Strip the volatile parts of an accessibility-tree dump so two renders of
    the SAME structural page hash equal. We drop element refs (reassigned every
    snapshot), level markers, dates, and bare numbers (counts/totals) — none of
    which mean the page structurally changed."""
    s = snap or ""
    s = _REF_RE.sub("", s)
    s = _LEVEL_RE.sub("", s)
    s = _DATE_RE.sub("<date>", s)
    s = _NUM_RE.sub("<n>", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def snapshot_hash(snap: str) -> str:
    """Short stable hash of the structural page (post-normalization)."""
    return hashlib.sha1(normalize_snapshot(snap).encode("utf-8")).hexdigest()[:16]


def _meaningful_lines(snap: str) -> list[str]:
    """Interactive/text lines worth diffing (skip pure structural scaffolding)."""
    out = []
    for raw in (snap or "").splitlines():
        line = _REF_RE.sub("", raw).strip(" -")
        if not line or line.startswith("URL:"):
            continue
        if any(k in line for k in ("button", "link", "textbox", "searchbox",
                                   "StaticText", "heading", "combobox", "cell",
                                   "option", "alert", "error", "Notification")):
            # Normalize volatile numbers/dates per line so a pure count change
            # (145 -> 146) is NOT reported as a structural change; a genuinely
            # new/removed element (text label) still shows.
            line = _NUM_RE.sub("<n>", _DATE_RE.sub("<date>", line))
            out.append(_WS_RE.sub(" ", line))
    return out



# ===========================================================================
#  A2 — element table: the accessibility tree, rendered as one line per control
# ===========================================================================
# An agent-browser snapshot of the attribute list page is 7,513 characters of
# indented tree, and it was still cut off mid-table. Most of it is structure the
# agent cannot act on: generic/list/listitem/rowgroup wrappers, and a StaticText
# child that repeats the name of the button above it.
#
# This renders the SAME refs as a flat table. The refs are agent-browser's own,
# so browse_click(ref="e6") keeps working exactly as before -- only the
# presentation changes. Measured on the real page: 7,513 -> ~1,900 characters.
#
# Lifted from browser-use/jev-ultrafast (references/jev-ultrafast), whose
# element table is why its decisions cost ~5k tokens where ours cost 21k.

# Roles the agent can actually act on. Everything else is scaffolding.
_ACTIONABLE = frozenset({
    "button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio",
    "switch", "tab", "menuitem", "menuitemradio", "menuitemcheckbox", "option",
    "spinbutton", "slider", "listbox", "textarea", "treeitem",
})
# Containers that carry meaning only as part of a result row.
_ROW_ROLES = frozenset({"gridcell", "cell", "columnheader", "rowheader"})
# Never worth a line of their own.
_WRAPPERS = frozenset({
    "generic", "list", "listitem", "rowgroup", "table", "grid", "navigation",
    "complementary", "main", "banner", "contentinfo", "region", "group",
    "paragraph", "separator", "image", "img", "svgroot", "labeltext",
    "document", "section", "article", "form", "figure", "none", "presentation",
})
# Kept as a visible marker: a dialog changes what every other ref means, and an
# alert is where this portal puts "Parameter with same name already exist."
_LANDMARKS = frozenset({"dialog", "alertdialog", "alert", "status", "log",
                        "heading", "tooltip"})

_NODE_RE = re.compile(r'^(\s*)-\s+([A-Za-z][\w-]*)\s*(.*)$')
# A name containing a quote is rare enough not to be worth an escape-aware
# pattern; such a line simply fails to parse and is passed through unchanged.
_NAME_RE = re.compile(r'^"([^"]*)"\s*(.*)$')
_ATTR_RE = re.compile(r'^\[([^\]]*)\]\s*(.*)$')
# Attributes worth showing. "ref" is handled separately; level/cursor and the
# rest are not decision-relevant and cost a line's width.
_KEEP_ATTRS = ("checked", "selected", "expanded", "disabled", "pressed",
               "current", "value", "active")


def parse_snapshot_node(line: str):
    """One accessibility-tree line -> (indent, role, name, attrs, rest), or None.

    Handles every shape agent-browser emits:
        - button "Add Source attributes" [ref=e6]
        - button "All Category" [expanded=false, ref=e28]
        - row [ref=e14] clickable [onclick]
        - StaticText "Approved"
        - cell
    """
    m = _NODE_RE.match(line.rstrip())
    if not m:
        return None
    indent, role, rest = len(m.group(1)), m.group(2), m.group(3)
    name = ""
    nm = _NAME_RE.match(rest)
    if nm:
        name, rest = nm.group(1), nm.group(2).strip()
    attrs: dict = {}
    am = _ATTR_RE.match(rest)
    if am:
        for part in am.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                attrs[k.strip().lower()] = v.strip()
            else:
                attrs[part.lower()] = "true"
        rest = am.group(2).strip()
    return indent, role, name, attrs, rest


def _cell(ref: str) -> str:
    """The ref column, padded so the role column lines up at any ref width."""
    return (("[" + ref + "]") if ref else "").ljust(7) + " "


def _state_suffix(attrs: dict, rest: str) -> str:
    bits = [k + "=" + attrs[k] for k in _KEEP_ATTRS if k in attrs]
    if "clickable" in rest:
        bits.append("clickable")
    return ("  · " + " ".join(bits)) if bits else ""


def compact_snapshot(text: str, max_text: int = 2500) -> str:
    """Render an accessibility tree as an element table plus a page-text block.

    Refs are preserved verbatim, so every existing browse_* call still resolves.
    A line the parser does not recognise is passed through unchanged, so an
    unexpected shape degrades to the old behaviour instead of losing content.
    """
    head, body = "", text or ""
    if body.startswith("URL:"):
        cut = body.find("\n")
        head, body = (body[:cut], body[cut + 1:]) if cut >= 0 else (body, "")
    rows: list[str] = []
    page_text: list[str] = []
    seen_text: set = set()
    named: set = set()
    pending_row = None          # [ref, [cell names]] while inside a <row>
    row_indent = -1

    def flush_row():
        nonlocal pending_row
        if pending_row:
            ref, cells = pending_row
            cells = [c for c in cells if c]
            if cells:
                rows.append(_cell(ref) + "row".ljust(10) + " | ".join(cells))
            pending_row = None

    for line in body.splitlines():
        node = parse_snapshot_node(line)
        if node is None:
            if line.strip():                      # keep anything unparsed
                flush_row()
                rows.append(line.rstrip())
            continue
        indent, role, name, attrs, rest = node
        low = role.lower()
        ref = attrs.get("ref", "")

        if pending_row is not None and indent <= row_indent:
            flush_row()

        if low in ("statictext", "text"):
            if pending_row is None and name and name not in seen_text \
                    and name not in named:
                seen_text.add(name)
                page_text.append(name)
            continue

        if low == "row" and ref:
            flush_row()
            pending_row, row_indent = [ref, []], indent
            continue
        if low in _ROW_ROLES and pending_row is not None:
            pending_row[1].append(name)
            continue

        if low in _LANDMARKS:
            flush_row()
            rows.append((_cell(ref) + low.ljust(10) + " " + name).rstrip())
            if name:
                named.add(name)
            continue

        if (low in _ACTIONABLE or low not in _WRAPPERS) and ref:
            flush_row()
            named.add(name)
            rows.append(_cell(ref) + low.ljust(10) + " " + name
                        + _state_suffix(attrs, rest))
            continue

    flush_row()
    blob = "\n".join(t for t in page_text if t.strip())
    if len(blob) > max_text:
        blob = blob[:max_text] + "\n…[+" + str(len(blob) - max_text) \
               + " chars of page text]"
    parts = [head] if head else []
    if rows:
        parts.append('CONTROLS — act on one by its ref, e.g. '
                     'browse_click(ref="e6")\n' + "\n".join(rows))
    if blob:
        parts.append("PAGE TEXT\n" + blob)
    return "\n\n".join(parts).strip() or (text or "")


_OPTION_LINE = re.compile(r"^\s*-\s+(link|option|menuitem)\b")
_LISTITEM_LINE = re.compile(r"^\s*-\s+listitem\b")


def collapse_option_overflow(text: str, keep: int = 12) -> str:
    """Collapse a long run of dropdown/list OPTION lines to the first `keep`
    plus a one-line summary. A 100-item category dropdown otherwise bloats the
    snapshot, and the bloat tips the weak ChatGPT-web brain into a refusal
    ('I can't use these tools') — the exact failure seen live. The summary points
    the agent at the Search box (filter) instead of scrolling 100 refs. A normal
    page (few options) is returned unchanged."""
    lines = text.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if _OPTION_LINE.match(lines[i]) or _LISTITEM_LINE.match(lines[i]):
            j, opt = i, 0
            while j < n and (_OPTION_LINE.match(lines[j])
                             or _LISTITEM_LINE.match(lines[j])):
                if _OPTION_LINE.match(lines[j]):
                    opt += 1
                j += 1
            if opt > keep:
                kept, k = 0, i
                while k < j and kept < keep:
                    out.append(lines[k])
                    if _OPTION_LINE.match(lines[k]):
                        kept += 1
                    k += 1
                indent = re.match(r"^(\s*)", lines[i]).group(1)
                out.append(f"{indent}… [+{opt - keep} more items hidden to save "
                           "space — if the one you need isn't shown, type it into "
                           "the page's Search box to filter]")
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def diff_snapshots(prev: str, cur: str, max_items: int = 6) -> str:
    """A compact, human-readable 'what changed' between two snapshots. Empty
    string when nothing meaningful changed. This is what rule 48 (diffing) and
    rule 61 (baseline replay) escalate on."""
    a, b = set(_meaningful_lines(prev)), set(_meaningful_lines(cur))
    added = [x for x in _meaningful_lines(cur) if x in (b - a)]
    removed = [x for x in _meaningful_lines(prev) if x in (a - b)]
    if not added and not removed:
        return ""
    parts = []
    if added:
        parts.append("appeared: " + " | ".join(added[:max_items]))
    if removed:
        parts.append("gone: " + " | ".join(removed[:max_items]))
    return "; ".join(parts)


# ===========================================================================
#  P4 — Checklist (parse goal items; tick deterministically from the page)
# ===========================================================================
# `_+` not `_`: a user's name can carry a DOUBLE underscore ("anand__test") and
# the single-underscore form silently dropped it -- the item never made it onto
# the checklist, so the run "verified" without ever tracking that record.
_UNDERSCORE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_+[A-Za-z0-9]+)+\b")
# Our own vocabulary looks exactly like a record name (browse_snapshot,
# type_label, browse_repeat_last ...), and delegation text is FULL of it -- a
# continuation that says "take ONE browse_snapshot, then create ar_test_1"
# extracted `browse_snapshot` as a record to create, put it on the checklist,
# and fed it to browse_repeat as a value to type into the form. Anything in a
# tool namespace is machinery, never a record.
_TOOL_PREFIXES = (
    "browse_", "fs_", "memory_", "excel_", "doc_", "csv_", "pdf_",
    "run_", "save_", "list_", "check_", "wait_", "delegate_", "send_",
    "add_", "get_", "view_", "create_", "record_", "current_", "lookup_",
    "report_", "write_", "read_", "open_", "close_", "click_", "type_",
    "press_", "scroll_", "select_", "wait_for_", "task_", "set_", "mark_",
)
_TOOL_EXACT = {"tool_calls", "tool_call", "task_id", "tool_use", "in_progress"}


def _is_machinery(token: str) -> bool:
    """True for our own tool/protocol vocabulary, which must never be mistaken
    for a record the user asked us to create."""
    t = (token or "").lower()
    return t in _TOOL_EXACT or t.startswith(_TOOL_PREFIXES)
# A created record shows up in a TABLE CELL — NOT in a form input being typed,
# and NOT in a dropdown's listitem/link (a category dropdown full of names that
# collide with attribute names would otherwise false-tick). Restricting to real
# table cells is what makes the checklist trustworthy.
_CELL_RE = re.compile(r"\b(cell|gridcell|rowheader)\b", re.I)
_INPUT_RE = re.compile(r"\b(textbox|searchbox|combobox|spinbutton)\b", re.I)
_QUOTED_RE = re.compile(r'"([^"]+)"')


def _norm_name(s: str) -> str:
    """Lower-case and treat runs of space/underscore as one separator, so a goal
    item 'Business Rating' matches a created record 'Business_Rating' (the form
    enforces underscores). Still an EXACT (normalized) match — 'Credit_Score'
    won't match 'Credit_Score_BRE'."""
    return re.sub(r"[\s_]+", " ", (s or "").strip().lower())


def _whole_word(needle: str, haystack: str) -> bool:
    """needle present in haystack at word boundaries (so 'Credit_Score' does NOT
    match inside 'Credit_Score_BRE' — '_' is a word char, so the boundary fails)."""
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


# ---------------------------------------------------------------------------
#  Verifying a GOAL against the page
# ---------------------------------------------------------------------------
# `Checklist.mark_present_in` matches an item against the EXACT value of a table
# cell. That works for the items `parse_checklist_items` produces -- record
# names like Monthly_Income -- but the swarm sets the checklist to the planned
# GOALS instead, and a goal is a sentence:
#
#   item:  "Create a category named 'LC_loans'"
#   cell:  "LC_loans"
#
# Those never match, so no goal has ever ticked by evidence; every ✓ came from
# the forced `mark_done`. Checking `evidenced` on the sentence would therefore
# report EVERY goal as unverified, including the ones that really worked --
# which is worse than not checking, because a warning that always fires is one
# nobody reads.
#
# So pull the thing the goal is ABOUT out of the sentence and look for that.
_GOAL_QUOTED_RE = re.compile(r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]{2,60})['\"\u2018\u2019\u201c\u201d]")
_FILENAME_RE = re.compile(r"\.[a-z0-9]{2,5}$", re.I)


# Only a goal that claims to CREATE something can be verified by looking for
# it on a page. "Navigation to the 'Source Category' section" names a quoted
# thing but creates nothing, and reporting it as CLAIMED BUT NOT VERIFIED --
# which is what happened -- teaches everyone to ignore the warning.
_CREATION_VERB_RE = re.compile(
    # Bounded stems. `sav\w*` matched "savings" and `add\w*` matched
    # "address" -- a verb list that quietly matches nouns makes every goal
    # look like a creation claim, which is how the warning loses its meaning.
    r"\b(creat(?:e|es|ed|ing|ion)|add(?:s|ed|ing|ition)?|mak(?:e|es|ing)|"
    r"new|register(?:s|ed|ing)?|insert(?:s|ed|ing)?|submit(?:s|ted|ting)?|"
    r"sav(?:e|es|ed|ing)|upload(?:s|ed|ing)?)\b", re.I)


def goal_claims_creation(goal: str) -> bool:
    """Whether this goal asserts something now EXISTS that did not before."""
    return bool(_CREATION_VERB_RE.search(goal or ""))


def goal_evidence_name(goal: str) -> str:
    """The record name a goal claims to create, or '' when it is not checkable.

    '' is a first-class answer. "Log in to the website" and "Summarise the
    findings" leave nothing on a page to look for, and pretending otherwise
    would manufacture false alarms.
    """
    g = (goal or "").strip()
    if not g:
        return ""
    # Nothing was claimed to be created, so there is nothing to look for.
    if not goal_claims_creation(g):
        return ""
    for m in _GOAL_QUOTED_RE.finditer(g):
        name = m.group(1).strip()
        # A filename is an input to the task, not a record the page will show.
        if name and not _FILENAME_RE.search(name):
            return name
    # Drop any quoted filename before scanning for identifiers, or the goal
    # "Attributes from 'Loan_Attributes.xlsx' created" yields Loan_Attributes --
    # the spreadsheet feeding the task, not a record the page will ever show.
    # Looking for it would report a perfectly good goal as unverified.
    stripped = _GOAL_QUOTED_RE.sub(
        lambda m: "" if _FILENAME_RE.search(m.group(1).strip()) else m.group(0), g)
    toks = [t for t in _UNDERSCORE_RE.findall(stripped) if not _is_machinery(t)]
    for t in toks:
        if not _FILENAME_RE.search(t):
            return t
    return ""


# A saved record, in either rendering of the page. The raw tree spells one as a
# run of quoted `cell` lines; the A2 element table spells the same row as
#     [e14]   row       01 | ATT_174 | Business_Status | Commercial_BL
# Both must count, or the compact page silently stops ticking the checklist and
# every created record reads as "claimed, not found on the page".
_ROW_LINE_RE = re.compile(r"^\[e\d+\]\s+row\s+(.*)$")


def _record_values(page_text: str) -> set:
    """Normalised values that appear on the page AS A SAVED RECORD.

    Never a form input: a half-typed name still sitting in a text box is not
    created, and treating it as created is how a run reported a category that
    did not exist.
    """
    out = set()
    for line in (page_text or "").splitlines():
        row = _ROW_LINE_RE.match(line.strip())
        if row:
            for v in row.group(1).split("|"):
                v = v.strip()
                if v:
                    out.add(_norm_name(v))
            continue
        if _CELL_RE.search(line) and not _INPUT_RE.search(line):
            for v in _QUOTED_RE.findall(line):
                out.add(_norm_name(v))
    return out


def cell_values(page_text: str) -> set:
    """Normalised values of the page's saved records (never form inputs)."""
    return _record_values(page_text)


def goal_is_evidenced(goal: str, page_text: str) -> tuple:
    """(checkable, evidenced) for one goal against the page it left behind."""
    name = goal_evidence_name(goal)
    if not name:
        return (False, False)
    return (True, _norm_name(name) in cell_values(page_text))


def parse_checklist_items(goal: str) -> list[str]:
    """Best-effort extraction of the discrete items a task asks you to create.
    Prefers underscore identifiers (Monthly_Income, Existing_EMI — the common
    shape), else a comma/and list after a 'create/add/following ...:' lead-in.
    Conservative: returns [] rather than guessing noise."""
    if not goal:
        return []
    toks, seen = _UNDERSCORE_RE.findall(goal), []
    for t in toks:
        if t not in seen and not _is_machinery(t):
            seen.append(t)
    if len(seen) >= 2:
        return seen
    m = re.search(r"(?:create|add|make|following)[^:]*:\s*(.+)", goal, re.I)
    if m:
        seg = re.split(r"[.\n]", m.group(1))[0]
        parts = [p.strip().strip("'\"") for p in re.split(r",| and ", seg)]
        parts = [p for p in parts if 1 <= len(p) <= 40]
        if len(parts) >= 2:
            return parts
    return seen


class Checklist:
    """Items to complete + whether each is done. Ticked DETERMINISTICALLY by
    reading the page (mark_present_in) — never by the model's claim."""

    def __init__(self, items: list[str]):
        # preserve order, dedupe
        self.items: dict[str, bool] = {}
        for it in items or []:
            it = (it or "").strip()
            if it and it not in self.items:
                self.items[it] = False
        self.last_ticked: list[str] = []   # names newly ticked by the last call

    def mark_claimed(self, item: str) -> None:
        """The child says this is finished and the page cannot confirm it.

        Renders as '?', never '✓'. The run that prompted this showed
        "'curr_account' category is created in the 'Source Category' section ✓"
        on every step while the category did not exist, because the goal loop
        ticked the item unconditionally at each boundary. The agent read its
        own unverified claim back as fact and moved on to the attributes.

        It stays in `remaining`, so the agent can still go and check it.
        """
        for k in self.items:
            if k.lower() == (item or "").strip().lower() and not self.items[k]:
                self.claimed.add(k)
                return

    def mark_done(self, item: str) -> None:
        for k in self.items:
            if k.lower() == (item or "").strip().lower():
                self.items[k] = True

    @property
    def claimed(self) -> set:
        # Lazily created so every construction path gets one, including the
        # ones that build a Checklist without going through __init__.
        if not hasattr(self, "_claimed"):
            self._claimed = set()
        return self._claimed

    def mark_present_in(self, page_text: str, cells_only: bool = True) -> int:
        """Tick any item that appears on the page AS A CREATED RECORD — by
        default only as the EXACT value of a table cell, never in a form input
        and never as a substring. Two guards learned from live runs: a half-typed
        name in the input box must NOT tick (it isn't saved), and 'Credit_Score'
        must NOT tick from a cell reading 'Credit_Score_BRE' (a different record).
        Returns how many were newly ticked (also recorded in ``last_ticked``)."""
        lines = (page_text or "").splitlines()
        self.last_ticked = []
        if cells_only:
            values = _record_values(page_text)
        else:
            low = "\n".join(lines).lower()
        for k in list(self.items):
            if self.items[k]:
                continue
            hit = (_norm_name(k) in values) if cells_only \
                else _whole_word(k.lower(), low)
            if hit:
                self.items[k] = True
                self.last_ticked.append(k)
        return len(self.last_ticked)

    @property
    def done(self) -> list[str]:
        return [k for k, v in self.items.items() if v]

    @property
    def remaining(self) -> list[str]:
        return [k for k, v in self.items.items() if not v]

    def render(self) -> str:
        if not self.items:
            return ""
        return " · ".join(
            f"{k} {'✓' if v else ('?' if k in self.claimed else '☐')}"
            for k, v in self.items.items())

    def progress(self) -> tuple[int, int]:
        return (len(self.done), len(self.items))


# ===========================================================================
#  P6 — LoopMonitor (detect no-progress loops)
# ===========================================================================
_READ_OPS = {"snapshot", "open", "get_text", "eval"}


class LoopMonitor:
    """Rolling window of (url, op, target, page_hash). flag() returns a reason
    string when the run looks stuck, else None. Cheap; runs every action."""

    def __init__(self, repeat_threshold: int = 3, stall_threshold: int = 4,
                 window: int = 12):
        self.repeat_threshold = repeat_threshold
        self.stall_threshold = stall_threshold
        self.window_size = window
        self.window: list[tuple] = []

    def record(self, url: str, op: str, target: str, page_hash: str) -> None:
        self.window.append((url or "", op or "", target or "", page_hash or ""))
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

    def reset(self) -> None:
        self.window = []

    def flag(self) -> Optional[str]:
        w = self.window
        # (1) same real action repeated back-to-back with no benefit.
        if len(w) >= self.repeat_threshold:
            recent = w[-self.repeat_threshold:]
            first = recent[0]
            if (first[1] not in _READ_OPS
                    and all(x[0] == first[0] and x[1] == first[1]
                            and x[2] == first[2] for x in recent)):
                return (f"repeat: '{first[1]} {first[2]}' {self.repeat_threshold}× "
                        f"on the same page with no change")
        # (2) several real actions but the page never structurally changed.
        if len(w) >= self.stall_threshold:
            recent = w[-self.stall_threshold:]
            real = [x for x in recent if x[1] not in _READ_OPS]
            hashes = {x[3] for x in recent if x[3]}
            if len(real) >= self.stall_threshold - 1 and len(hashes) == 1:
                return (f"stall: page unchanged across {self.stall_threshold} "
                        "actions — a validation error or a no-op is likely")
        return None


def _correction_for(flag: str) -> str:
    """Turn a monitor flag into a concrete instruction for the agent (tier-1)."""
    if flag.startswith("repeat"):
        return ("you're repeating the same action with no effect. STOP — call "
                "browse_snapshot and READ the page (look for a validation error, "
                "a disabled button, or a changed layout) before doing anything "
                "else; do NOT repeat that click.")
    if flag.startswith("stall"):
        return ("your last few actions changed nothing on the page. There is "
                "probably an error message or a required field you missed — read "
                "the page text (browse_get_text) and fix the cause; don't keep "
                "submitting.")
    return "you appear stuck — re-read the current page before continuing."


# ===========================================================================
#  P1 — RunContext (the spine) + module-global current context
# ===========================================================================
class RunContext:
    def __init__(self, goal: str = "", max_steps: int = 0,
                 items: Optional[list[str]] = None):
        self.active = False
        self.goal = (goal or "").strip()
        self.max_steps = int(max_steps or 0)
        self.step = 0
        self.run_id: Optional[str] = None
        self.current_url = ""
        self.last_page_hash = ""
        self.last_snapshot = ""
        self.checklist = Checklist(items if items is not None
                                   else parse_checklist_items(goal))
        self.monitor = LoopMonitor()
        self.trail: list[dict] = []     # [{step,url,op,target,goal,ok,err}]
        self.last_flag: Optional[str] = None
        self.last_completed = ""        # most-recent checklist item that ticked
        # Goal segmentation (swarm._run_goals): which outcome is being worked on
        # right now. 0 = not running goal-by-goal.
        # A4: consecutive page-CHANGING actions that changed nothing. The
        # loop monitor already noticed this and printed a warning; the run of
        # 13 Oct shows the model reading six consecutive '✗ page did not
        # change' lines and clicking a seventh time. A counter the tool layer
        # can refuse on is the difference between advice and a gate.
        self.no_change = 0
        self.goal_index = 0             # 1-based
        self.goal_total = 0
        self.goal_name = ""
        # Whether `checklist` holds planned GOALS (sentences) rather than the
        # record values a form would be filled with. The FAST PATH reads the
        # checklist as values, so on a planned run it produced
        # browse_repeat_last(items=["Creation of a category named 'x'"]) --
        # an instruction to type the goal sentence into the form.
        self.items_are_goals = False

    def tick(self) -> int:
        self.step += 1
        return self.step

    def set_items(self, items: list[str], as_goals: bool = False) -> None:
        """Install the checklist on an ALREADY-OPEN run, without resetting the
        step counter, trail or loop monitor.

        The runner opens the run before it knows its goals (planning needs the
        child), so the checklist arrives a moment later. Calling set_context()
        again to deliver it threw away the run's progress and, worse, silently
        re-derived the items from whatever description was at hand. This keeps
        the run and replaces only the list.

        `as_goals` marks the items as planned outcomes rather than form values,
        which is what the FAST PATH needs to know before offering to bulk-create
        them."""
        self.checklist = Checklist(items or [])
        self.items_are_goals = bool(as_goals)


_CTX = RunContext()


def set_context(goal: str = "", max_steps: int = 0,
                items: Optional[list[str]] = None) -> RunContext:
    """Begin a supervised browser run. Called by the runner BEFORE the child
    acts. Items default to whatever parse_checklist_items finds in the goal."""
    global _CTX
    _CTX = RunContext(goal=goal, max_steps=max_steps, items=items)
    _CTX.active = True
    return _CTX


def context() -> RunContext:
    return _CTX


def set_goal(index: int, total: int, name: str = "") -> None:
    """Record which segmented goal is in flight (swarm._run_goals calls this)."""
    ctx = context()
    ctx.goal_index = int(index or 0)
    ctx.goal_total = int(total or 0)
    ctx.goal_name = (name or "").strip()


def progress_line() -> str:
    """One-line 'still alive and moving' summary of the active browser run.

    The orchestrator only ever saw ``status: in_progress`` for a child that was
    busy driving a 40-step web flow, which looks indistinguishable from a hang --
    so it gave up and told the user the system was stuck. Real evidence of
    movement is far more persuasive than any instruction telling it to be
    patient, and on a segmented run the GOAL counter is the most legible
    evidence there is: it shows work completed and work remaining, not just that
    a step counter ticked.

    Returns "" when no supervised run is active.
    """
    ctx = context()
    if not ctx.active:
        return ""
    bits = []
    if ctx.goal_total:
        bits.append(f"goal {ctx.goal_index}/{ctx.goal_total}")
        if ctx.goal_name:
            bits.append(f"“{_short(ctx.goal_name, 40)}”")
        pending = max(0, ctx.goal_total - ctx.goal_index)
        bits.append(f"{ctx.goal_index - 1} done, {pending} pending")
    if ctx.checklist and ctx.checklist.items:
        done = sum(1 for v in ctx.checklist.items.values() if v)
        # On a segmented run the checklist IS the goal list, so only add it when
        # it tells us something the goal counter did not (deterministic ticks
        # can confirm an outcome the model has not reported yet).
        label = "verified on page" if ctx.goal_total else "checklist"
        bits.append(f"{label} {done}/{len(ctx.checklist.items)}")
    bits.append(f"step {ctx.step}")
    if ctx.current_url:
        bits.append(_url_tail(ctx.current_url))
    if ctx.trail:
        last = ctx.trail[-1]
        tgt = _short(str(last.get("target") or ""), 24)
        bits.append(f"last: {last.get('op')} {tgt}".strip())
    return " · ".join(bits)


def clear_context() -> None:
    global _CTX
    _CTX = RunContext()


# ===========================================================================
#  P5 — header rendering (appended to the snapshot; mock-safe)
# ===========================================================================
def _short(s: str, n: int) -> str:
    s = _WS_RE.sub(" ", (s or "").strip())
    return s if len(s) <= n else s[:n - 1] + "…"


def _url_tail(url: str) -> str:
    if not url:
        return "?"
    path = re.sub(r"^https?://[^/]+", "", url) or "/"
    return path.split("?")[0][-40:] or "/"


def compress_trail_inmem(trail: list[dict], max_lines: int = 6,
                         goal: int = 0, carry_previous: bool = False) -> str:
    """One-line compression of the recent action trail for the header.

    Scoped to the CURRENT goal when one is running. A fixed last-N window
    spanning the whole delegation meant that at goal 2 step 3 most of the trail
    was goal 1's tail -- steps on pages the agent has left, competing for
    attention with the two that matter. Each finished goal is represented by
    its one-line result in the goal message instead.

    The exception is a step that FAILED under an earlier goal: that usually
    explains the shape of this one, so a couple of those survive the window.
    """
    if not trail:
        return ""
    if goal:
        current = [t for t in trail if t.get("goal", 0) == goal]
        prior_fails = [t for t in trail
                       if t.get("goal", 0) and t.get("goal") != goal
                       and not t.get("ok", True)][-2:]
        prior = []
        if carry_previous:
            # Something is marked '?' -- reported done, not found on a page.
            # The agent is being asked to judge that claim, and it cannot do
            # that from the claim alone: it needs the steps that supposedly
            # produced it. A claim is virtually always about the goal that
            # just finished.
            prior = [t for t in trail
                     if t.get("goal", 0) == goal - 1][-max_lines:]
        seen = {id(t) for t in prior}
        recent = prior + [t for t in prior_fails if id(t) not in seen] \
            + current[-max_lines:]
    else:
        recent = trail[-max_lines:]
    parts = []
    for t in recent:
        # An action that errored is the single most useful thing in the trail
        # and used to be invisible: the entry carried no ok/err field at all,
        # so "clicked Submit" and "clicked Submit and the form refused" read
        # identically. The one-shot warning on that step's result was the only
        # signal, and the snapshot pruner ate it two steps later.
        mark = "" if t.get("ok", True) else "✗ "
        err = (t.get("err") or "").strip()
        tail = f" -> {_short(err, 64)}" if err else ""
        parts.append(f"{mark}{t['step']}:{_url_tail(t.get('url',''))} "
                     f"{t.get('op','')} "
                     f"{_short(t.get('target',''), 24)}{tail}".strip())
    return " · ".join(parts)


def repeat_hint(ctx: RunContext) -> str:
    """The 'switch to the CODE path' directive — the intelligent split between
    LLM and code. Once at least ONE create-cycle is done and ≥2 items remain,
    doing the rest by hand is where the weak brain wastes steps and eventually
    mis-formats a tool call. So we compute the EXACT one-shot call (with the
    remaining items + the value to parameterize) and put it in front of the
    model, so finishing the bulk is a single copy-paste, not N manual cycles."""
    cl = ctx.checklist
    if not cl:
        return ""
    # The items must be VALUES to type, not planned outcomes. On a planned run
    # this fired with goal sentences and told the agent to bulk-create
    # ["Creation of a category named 'sav_account'", ...] -- i.e. to type the
    # sentence into the form. Same class as the checklist/evidence mismatch:
    # a list of goals is not a list of records.
    if getattr(ctx, "items_are_goals", False):
        return ""
    done, remaining = cl.done, cl.remaining
    if len(done) < 1 or len(remaining) < 2:
        return ""
    vw = ctx.last_completed or (done[-1] if done else "")
    vw_part = f", value_was='{vw}'" if vw else ""
    return ("⚡ FAST PATH (use it now): you've already created "
            f"{len(done)} ({', '.join(done[:3])}). Do NOT fill the form by hand "
            f"again — create the remaining {len(remaining)} in ONE call: "
            f"browse_repeat_last(items={remaining}{vw_part}, verify=True). "
            "If that can't derive the steps, use browse_repeat(template, items) "
            "with click_text/type_label so it survives the form rebuilding.")


def render_header(ctx: Optional[RunContext] = None,
                  flag: Optional[str] = None) -> str:
    """The orientation block prepended to a tool result. Empty unless a run is
    active. APPEND-ONLY by construction — callers attach it to the NEW tool
    result, never to an older message (that would break mock continuity)."""
    ctx = ctx or context()
    if not ctx.active:
        return ""
    lines: list[str] = []
    if ctx.goal:
        lines.append("GOAL: " + _short(ctx.goal, 240))
    if ctx.checklist and ctx.checklist.items:
        d, total = ctx.checklist.progress()
        lines.append(f"CHECKLIST ({d}/{total}): " + ctx.checklist.render())
        if ctx.checklist.claimed:
            # ✓ is deterministic -- the record was seen on a page. '?' is the
            # previous turn's claim and nothing more, so say so plainly rather
            # than letting the agent read it as done.
            lines.append(
                "  ? = reported done but NOT found on any page since. Judge it "
                "from the steps below: if no step actually filled the form and "
                "saved it, it does not exist and you must do it now.")
    hint = repeat_hint(ctx)
    if hint:
        lines.append(hint)
    _claimed = bool(ctx.checklist and ctx.checklist.claimed)
    trail = compress_trail_inmem(ctx.trail, goal=ctx.goal_index,
                                 carry_previous=_claimed)
    if trail:
        lines.append((("STEPS (this goal, and the one before it): "
                       if _claimed else "STEPS THIS GOAL: ")
                      if ctx.goal_index else "TRAIL: ") + trail)
    if ctx.max_steps:
        lines.append(f"STEP {ctx.step}/{ctx.max_steps} "
                     f"(~{max(0, ctx.max_steps - ctx.step)} left)")
    if flag:
        lines.append("⚠ AUDITOR: " + _correction_for(flag))
    if not lines:
        return ""
    return "[ORIENTATION — your goal & progress, re-read it]\n" + \
           "\n".join(lines) + "\n\n"


# ===========================================================================
#  P2 — durable per-step page record + log reader / compressor / segmenter
# ===========================================================================
def _log_path(run_id: str, log_dir: Optional[str]) -> str:
    return os.path.join(log_dir or config.BROWSER_LOG_DIR, f"run_{run_id}.jsonl")


def append_page_record(run_id: str, step: int, url: str, page_hash: str,
                       checklist_state: str = "",
                       log_dir: Optional[str] = None) -> None:
    """Persist a per-step page-state line into the SAME run log the actions go
    to (so the trail/compressor can pair an action with the page it produced)."""
    if not run_id:
        return
    try:
        os.makedirs(log_dir or config.BROWSER_LOG_DIR, exist_ok=True)
        rec = {"type": "page", "run_id": run_id, "step": step, "url": url,
               "page_hash": page_hash, "checklist": checklist_state,
               "ts": datetime.now(timezone.utc).isoformat()}
        with open(_log_path(run_id, log_dir), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read_run_log(run_id: str, log_dir: Optional[str] = None) -> list[dict]:
    """Read a run's jsonl back into a list of records (actions + page records)."""
    out: list[dict] = []
    try:
        with open(_log_path(run_id, log_dir), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def _is_action(rec: dict) -> bool:
    return rec.get("type") != "page" and bool(rec.get("op"))


def compress_trail(records: list[dict], max_lines: int = 8) -> str:
    """Durable-log version of the trail (pairs each action with the URL of the
    page record that follows it). Used where the in-memory trail isn't handy."""
    lines: list[str] = []
    pending: list[dict] = []
    for rec in records:
        if _is_action(rec):
            pending.append(rec)
        elif rec.get("type") == "page":
            url = _url_tail(rec.get("url", ""))
            for a in pending:
                tgt = _short(str((a.get("args") or {}).get("text")
                                 or a.get("action") or ""), 24)
                lines.append(f"{a.get('op','')} {tgt} →{url}".strip())
            pending = []
    for a in pending:  # trailing actions with no page record yet
        lines.append(f"{a.get('op','')} {_short(a.get('action',''), 24)}".strip())
    return " · ".join(lines[-max_lines:])


# COMMIT verbs only — these end one create-cycle. Deliberately excludes "add" /
# "create" / "new", which usually OPEN a form rather than commit it (so we don't
# split an iteration at its first click). Heuristic; good for the common form.
_SUBMIT_RE = re.compile(r"\b(save|submit|confirm|apply|update|finish)\b", re.I)


def segment_iterations(records: list[dict]) -> list[list[dict]]:
    """Split the action stream into iterations, ending each at a submit/save-type
    click — the boundary of one 'create a record' cycle. Used by the future
    browse_repeat_last (rule 60) to derive a template from the last iteration."""
    actions = [r for r in records if _is_action(r)]
    segments: list[list[dict]] = []
    cur: list[dict] = []
    for a in actions:
        cur.append(a)
        label = f"{a.get('action','')} {json.dumps(a.get('args') or {})}"
        if a.get("op") in ("click",) and _SUBMIT_RE.search(label):
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)
    return segments


# ===========================================================================
#  THE INTEGRATION ENTRY POINT — one call per action from browser_tools
# ===========================================================================
# Ops after which the page is EXPECTED to change. A click that submits a form,
# follows a link or opens a modal always leaves a different page; when it does
# not, something was rejected -- which is precisely what went unnoticed when a
# category form was submitted empty:
#
#   step 8  /newCategory  hash=b7b4225b5dfb0547
#   step 9  /newCategory  hash=b7b4225b5dfb0547   <- after type + Submit
#
# Both hashes were already recorded. Nothing compared them, so the agent
# reported the category as created.
#
# Deliberately EXCLUDES 'type' and 'scroll'. Our snapshots do not carry input
# values or scroll offset, so those legitimately leave the hash untouched and
# would warn on every keystroke -- and a warning that fires constantly is one
# the model learns to ignore.
_EXPECT_PAGE_CHANGE = frozenset({
    "click", "click_text", "click_role", "press", "batch", "select",
    "submit", "repeat", "attach_file",
})

# Text a form shows when it rejects a submit. The page that blocked category
# creation said, in red, "Category Description is required." -- the exact reason,
# already on screen, in a snapshot the agent had. It reported "no change or
# confirmation visible" and asked the user what to do, because nothing pointed
# at the line that explained it.
_VALIDATION_RE = re.compile(
    r"^[^\n]{0,160}?\b("
    r"is required|are required|cannot be empty|can't be empty|must not be empty|"
    r"please enter|please select|please provide|please fill|"
    r"is invalid|invalid |must be |should be |too short|too long|"
    r"already exists|does not match|is mandatory|required field"
    r")\b[^\n]{0,160}$",
    re.I | re.M)

# Lines that merely LABEL a required field ("Category Name *") rather than
# complain about one. Reporting those as errors would point the agent at every
# field on the form instead of the one that failed.
_NOT_AN_ERROR_RE = re.compile(r"^\s*(textbox|combobox|searchbox|heading|link|"
                              r"button|cell|row)\b", re.I)


def validation_errors(page_text: str, limit: int = 3) -> list:
    """Validation complaints visible on the page, most specific first."""
    out = []
    for line in (page_text or "").splitlines():
        t = " ".join(line.split())
        if not t or _NOT_AN_ERROR_RE.match(t):
            continue
        if _VALIDATION_RE.search(t):
            t = t.strip("\"' ")
            if t not in out:
                out.append(t)
        if len(out) >= limit:
            break
    return out


_NO_CHANGE_NOTE = (
    "⚠ NOTHING CHANGED: the page is byte-identical to before this {op}. "
    "Whatever you expected to happen did NOT happen — a form was rejected, a "
    "required field was empty, or the click missed. Do NOT report this step as "
    "done. Take a browse_snapshot, look for a validation message, and check "
    "that the value you typed is actually IN the field before submitting again."
)


def _no_change_note(op: str, prev_hash: str, new_hash: str,
                    page_text: str = "") -> str:
    """The warning for an action that left the page untouched, or ''.

    When the page itself says why, quote it: "the page did not change" sends
    the agent hunting, whereas "Category Description is required." ends the
    problem in one line.
    """
    o = (op or "").strip().lower()
    if o not in _EXPECT_PAGE_CHANGE:
        return ""
    if not prev_hash or prev_hash != new_hash:
        return ""
    note = _NO_CHANGE_NOTE.format(op=o)
    errs = validation_errors(page_text)
    if errs:
        note += ("\nTHE PAGE SAYS WHY — fix exactly this and submit again: "
                 + " | ".join(errs))
    return note


def after_action(op: str, target: str, snapshot: str,
                 run_id: Optional[str] = None,
                 log_dir: Optional[str] = None) -> str:
    """Update the supervisor with the result of one action and return the header
    to prepend to the tool result. No-op (returns "") when no run is active, so
    unwired callers are unaffected.

      * P1: advances the step counter and current url,
      * P4: ticks the checklist from the page text (deterministic),
      * P6: feeds the loop monitor and computes a flag,
      * P2: writes the durable per-step page record,
      * P5: renders the orientation header (incl. a correction line when flagged).
    """
    ctx = context()
    if not ctx.active:
        return ""
    url = _url_from_snapshot(snapshot) or ctx.current_url
    phash = snapshot_hash(snapshot)
    # Capture the comparison before the new hash overwrites the old one.
    stalled = _no_change_note(op, ctx.last_page_hash, phash, snapshot)
    if run_id:
        ctx.run_id = run_id
    ctx.tick()
    ctx.current_url = url
    ctx.last_page_hash = phash
    ctx.last_snapshot = snapshot
    if ctx.checklist:
        ctx.checklist.mark_present_in(snapshot)
        if ctx.checklist.last_ticked:        # remember the most-recent cycle done
            ctx.last_completed = ctx.checklist.last_ticked[-1]
    ctx.monitor.record(url, op, target, phash)
    # Record WHETHER the step worked, not just that it happened. Two signals,
    # both already computed: the page refused to change, or it is showing a
    # complaint. Either one means this step did not do what it looks like.
    # A4: only a mutating action can be judged. A read that returns the same
    # page is not a failure, so it neither increments nor clears the count.
    if (op or "").strip().lower() in _EXPECT_PAGE_CHANGE:
        ctx.no_change = (ctx.no_change + 1) if stalled else 0
    _errs = validation_errors(snapshot, limit=1)
    ctx.trail.append({"step": ctx.step, "url": url, "op": op, "target": target,
                      "goal": ctx.goal_index,
                      "ok": not (stalled or _errs),
                      "err": (_errs[0] if _errs
                              else ("page did not change" if stalled else ""))})
    append_page_record(ctx.run_id or "", ctx.step, url, phash,
                       ctx.checklist.render() if ctx.checklist else "", log_dir)
    ctx.last_flag = ctx.monitor.flag()
    header = render_header(ctx, ctx.last_flag)
    # First line, not last: this contradicts what the agent is about to
    # conclude, so it has to be read before the orientation text it trusts.
    return f"{stalled}\n{header}" if stalled else header


def _url_from_snapshot(snapshot: str) -> str:
    """Pull the 'URL: ...' line our snapshots start with."""
    for line in (snapshot or "").splitlines():
        line = line.strip()
        if line.startswith("URL:"):
            return line[4:].strip()
    return ""


# ===========================================================================
#  Rule 60 — template expansion with overrides + derive-a-macro-from-the-log
# ===========================================================================
def _subst_action(a: dict, subs: dict, primary) -> dict:
    """Substitute placeholders in one template action: '{}' -> primary value,
    '{name}' -> subs[name]. Only string fields are touched. Unmatched
    placeholders are left as-is (caller's mistake to surface, not crash)."""
    out = dict(a)
    for k, v in list(out.items()):
        if isinstance(v, str):
            s = v.replace("{}", str(primary))
            for name, val in subs.items():
                s = s.replace("{" + name + "}", str(val))
            out[k] = s
    return out


def expand_template(template: list, items: list,
                    defaults: Optional[dict] = None) -> list:
    """Expand a repeat template over items into a flat action list (rule 60a).

    Each item is a plain value (substituted into '{}') OR a dict of named
    overrides (its 'value'/'text' is the primary for '{}', other keys fill
    '{name}'). ``defaults`` apply to every iteration and are overridden per item
    — so the odd-one-out (e.g. a different category) is expressed inline."""
    defaults = defaults or {}
    out: list = []
    for it in items:
        if isinstance(it, dict):
            subs = {**defaults, **it}
            primary = it.get("value", it.get("text", ""))
        else:
            subs = {**defaults, "value": it}
            primary = it
        for a in template:
            out.append(_subst_action(a, subs, primary))
    return out


def _record_to_action(rec: dict) -> Optional[dict]:
    """Reconstruct a REF-FREE action dict from one logged action record (uses the
    durable {role,name} locator, never the stale @eN). Returns None if it can't
    be made replayable."""
    args = rec.get("args") or {}
    a = args.get("action")
    if isinstance(a, dict):                 # batched action: stored verbatim
        return dict(a)
    op, loc = rec.get("op"), (rec.get("locator") or {})
    name, role = loc.get("name"), loc.get("role")
    if op == "click" and name:
        return ({"action": "click_role", "role": role, "name": name}
                if role else {"action": "click_text", "text": name})
    if op == "type" and name:
        return {"action": "type_label", "label": name, "text": args.get("text", "")}
    if op == "press":
        return {"action": "press", "key": args.get("key", "Enter")}
    if op == "select":
        return {"action": "select", "ref": args.get("ref", ""),
                "value": args.get("value", "")}
    return None


def _subst_literal(a: dict, old: str, new: str) -> dict:
    out = dict(a)
    for k, v in out.items():
        if isinstance(v, str) and old and old in v:
            out[k] = v.replace(old, new)
    return out


def derive_template_from_log(records: list,
                             value_was: str = "") -> tuple[Optional[list], str]:
    """Rule 60b: turn the LAST completed iteration in the run log into a
    parameterized template (the typed value -> '{}'). Returns (template, "") or
    (None, reason). ``value_was`` is the value you used last time; auto-detected
    when there's exactly one typed value."""
    segs = segment_iterations(records)
    if not segs:
        return None, "no completed iteration in the run log yet — do one first."
    template = [act for act in (_record_to_action(r) for r in segs[-1]) if act]
    if not template:
        return None, ("couldn't rebuild ref-free actions from the last iteration "
                      "— do it once using click_text / type_label, not raw @eN refs.")
    if not value_was:
        typed = [a.get("text", "") for a in template
                 if a.get("action") in ("type", "type_label", "type_placeholder",
                                        "fill") and a.get("text")]
        distinct = list(dict.fromkeys(typed))
        if len(distinct) == 1:
            value_was = distinct[0]
        elif not distinct:
            return None, "the last iteration typed no value to parameterize."
        else:
            return None, (f"ambiguous which typed value varies ({distinct}); pass "
                          "value_was=<the value you used last time>.")
    return [_subst_literal(a, value_was, "{}") for a in template], ""


# ===========================================================================
#  A4 — the page is wedged: stop asking, and say so
# ===========================================================================
# should_block() below only fires on the monitor's "repeat" flag, which needs
# the SAME (url, op, target) three times running. In the 13 Oct run the agent
# alternated three spellings of one button -- click_text "Add Source
# attributes", click @e6, batch click @e6 -- so the repeat flag never fired and
# nothing was ever blocked. Eleven dead clicks in 3.5 minutes.
#
# This counts the only thing that matters: page-changing actions that changed
# nothing, whatever they were called. At the limit the tool layer refuses the
# action and recovers the page instead (browser_tools._recover_wedged_page).
NO_CHANGE_LIMIT = 3


def wedged(limit: int = NO_CHANGE_LIMIT) -> bool:
    """True when the last `limit` page-changing actions all changed nothing."""
    ctx = context()
    return bool(ctx.active and ctx.no_change >= max(1, int(limit)))


def clear_wedge() -> None:
    """Called after the page has been forcibly changed, so the agent may act."""
    context().no_change = 0


def wedge_report(limit: int = NO_CHANGE_LIMIT) -> str:
    """What to tell the agent instead of running its action."""
    ctx = context()
    acts = [t for t in ctx.trail if t.get("op") in _EXPECT_PAGE_CHANGE][-limit:]
    what = " · ".join(f"{t.get('op')} {_short(str(t.get('target') or ''), 28)}"
                      for t in acts) or "your last few actions"
    errs = validation_errors(ctx.last_snapshot or "", limit=2)
    why = ("\nThe page says: " + " | ".join(errs)) if errs else ""
    return (f"REFUSED — not run. {ctx.no_change} page-changing actions in a row "
            f"left the page byte-identical ({what}). The click is landing on "
            f"nothing, or a dialog, toast or backdrop is on top of it."
            f"{why}\n"
            "I have pressed Escape and re-read the page for you. Work from the "
            "page below: if your target is still there, something is covering "
            "it — close that first. Do NOT repeat the same action.")


# ===========================================================================
#  Rule 58 — auditor tier-2: deterministic hard-block of a runaway repeat
# ===========================================================================
_BLOCKABLE = {"click", "type"}


def should_block(op: str, target: str) -> Optional[str]:
    """Tier-2 hard stop: refuse an action that would just repeat a flagged loop.
    Returns a block message, or None to proceed. Conservative — only blocks an
    identical click/type once the monitor has already flagged a repeat AND this
    is the same action, so legitimate retries/scrolls are never blocked."""
    ctx = context()
    if not ctx.active or op not in _BLOCKABLE:
        return None
    flag = ctx.monitor.flag()
    if not flag or not flag.startswith("repeat") or not ctx.monitor.window:
        return None
    last = ctx.monitor.window[-1]
    if last[1] == op and last[2] == (target or ""):
        return (f"BLOCKED: you've already done '{op} {target}' several times on "
                "this page with no change — do NOT repeat it. Read the page "
                "(browse_get_text) for an error/validation message or a different "
                "control, or move on to the next item.")
    return None


# ===========================================================================
#  Rule 61 — baseline-DOM compare: deviation of a later iteration from the first
# ===========================================================================
_ERR_RE = re.compile(
    r"\b(error|invalid|required|already exists|failed|must|cannot|denied|"
    r"duplicate|not allowed)\b", re.I)


def error_lines(snapshot: str) -> list[str]:
    """Lines that look like an error/validation/alert message on the page."""
    out = []
    for line in (snapshot or "").splitlines():
        low = line.lower()
        if _ERR_RE.search(line) and ("statictext" in low or "alert" in low
                                     or "error" in low or "notification" in low):
            out.append(_WS_RE.sub(" ", _REF_RE.sub("", line)).strip(" -"))
    return out


def _same_path(a: str, b: str) -> bool:
    def p(u: str) -> str:
        u = re.sub(r"^https?://[^/]+", "", u or "")
        return u.split("?")[0].rstrip("/").lower()
    return p(a) == p(b)


def iteration_deviation(baseline_snapshot: str, live_snapshot: str) -> str:
    """Rule 61: how a later iteration deviates from the first (good) one. Benign
    new data rows are IGNORED; what we flag is (a) a URL change — the form didn't
    submit / a modal blocked it — or (b) a NEW error/alert that wasn't there
    before. Empty string when the iteration looks healthy."""
    b_url, l_url = _url_from_snapshot(baseline_snapshot), _url_from_snapshot(live_snapshot)
    if b_url and l_url and not _same_path(b_url, l_url):
        return f"ended on {l_url} (expected {b_url}) — the form may not have submitted"
    new_err = [e for e in error_lines(live_snapshot)
               if e not in error_lines(baseline_snapshot)]
    if new_err:
        return "error appeared: " + " | ".join(new_err[:3])
    return ""


def stuck_summary() -> Optional[str]:
    """For the runner to call AFTER a child reply: a one-line 'this run looks
    stuck' note (loop flag + checklist progress) so the orchestrator can report
    the truth instead of a fabricated success. None when the run made progress."""
    ctx = context()
    if not ctx.active:
        return None
    flag = ctx.monitor.flag()
    d, total = ctx.checklist.progress() if ctx.checklist else (0, 0)
    if not flag:
        return None
    where = _url_tail(ctx.current_url)
    prog = f"{d}/{total} items done" if total else "no checklist"
    return f"STUCK at {where} ({flag}); {prog}; remaining: " + \
           ", ".join(ctx.checklist.remaining[:8]) if ctx.checklist else \
           f"STUCK at {where} ({flag})"
