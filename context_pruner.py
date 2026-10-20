"""Context pruning for browser agents.

A long browser task is a ReAct loop of dozens of steps, and every browser action
returns a fresh page snapshot (the accessibility tree -- ~7.5KB each). Those pile
up in the agent's live context, and because the WHOLE context is re-sent to the
model on every reasoning step, the bloat compounds: measured on real pages, two
consecutive snapshots of the SAME page are 98% line-identical, so a 40-step run
re-sends roughly 1.6M tokens of which the overwhelming majority is duplicate.

The original fix was blunt: keep the newest snapshot, collapse every older one to
a fixed placeholder. That is nearly optimal on cost, but it is wrong in three
ways this module fixes -- the ideas are lifted from headroomlabs-ai/headroom
(read_lifecycle.py, cross_turn_dedup.py, error_detection.py):

T1  CLASSIFY, DON'T COUNT. Collapsing by POSITION is blind to which page a
    snapshot shows. Navigate A -> B -> back to A and the old blunt rule
    collapsed the A snapshot even though A is what is on screen. Snapshots are
    now keyed on their URL and labelled by WHY they are dead:
      * SUPERSEDED -- the same page was read again later; merely redundant.
      * STALE      -- the browser left that page; the content is now factually
                      wrong, not just old.
    The two get different markers, because "the page has since changed, take a
    fresh snapshot" is true and actionable where a generic placeholder is not.

T2  SAY WHAT CHANGED. A collapsed snapshot used to carry no information at all.
    It now carries a one-line summary of what changed between it and the copy
    that superseded it (via browser_supervisor.diff_snapshots), so the model
    keeps a usable trace of the run for ~40 chars instead of ~7,500.

T3  NEVER SHRED THE ERROR. A page showing "Category name already exists" is the
    single most important page in a failing run, and the blunt rule collapsed it
    one step later. Error-bearing snapshots are retained; once they do age out,
    the error lines themselves are preserved verbatim inside the marker.

CACHE SAFETY. Every marker is frozen at the moment it is written: it depends
only on this snapshot and the ones BEFORE it, never on what arrives later, so a
collapsed snapshot is never rewritten a second time. Each snapshot therefore
changes exactly once (expanded -> collapsed) and everything earlier in the
context stays byte-stable, which is what keeps a provider's cached prefix (and
the mock's prefix match) intact for the whole run.

Only snapshot-bearing browser tools are touched. browse_get_text / browse_eval
results are left intact -- they carry data the agent deliberately read, not
stale refs.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator, Callable

from agentscope.middleware import MiddlewareBase

# Browser tools whose result is a page snapshot (accessibility tree with @eN
# refs). Their older outputs are safe to collapse -- the refs go stale anyway.
SNAPSHOT_TOOLS = {
    "browse_open", "browse_snapshot", "browse_click", "browse_type",
    "browse_press", "browse_scroll", "browse_back", "browse_click_text",
    "browse_batch", "browse_repeat", "browse_repeat_last",
}

# Any marker this module writes starts with this, so a second pass recognises an
# already-collapsed snapshot and leaves it exactly as it is.
MARKER_PREFIX = "[snapshot "

# Retained for callers/tests that import it; new collapses use the richer
# markers built by _marker().
PLACEHOLDER = (
    "[older browser snapshot pruned to save context — its @eN refs are now "
    "stale; call browse_snapshot to read the current page]"
)

_URL_RE = re.compile(r"^URL:\s*(\S+)", re.M)

# Phrases that mark a page as carrying something the agent must not lose. Kept
# deliberately narrow -- an accessibility tree says "alert" and "error" for
# structural reasons, so we match the wording of a real, user-visible failure.
_ERROR_RE = re.compile(
    r"(already exists|is required|cannot be (?:empty|blank|used)|must be|"
    r"invalid|not allowed|not permitted|failed to|could not be|"
    r"does not match|duplicate|try again|error:|denied)",
    re.I,
)
# Cap how much of a page's error text rides along in a marker.
_ERR_LINE_CAP = 2
_ERR_CHARS = 160


def _text_of(block) -> str:
    out = getattr(block, "output", None)
    if isinstance(out, list) and out and hasattr(out[0], "text"):
        return out[0].text or ""
    return ""


def _is_snapshot_result(block) -> bool:
    """True for a tool_result block produced by a snapshot-bearing browser tool
    whose output is the usual [TextBlock] shape we can collapse."""
    if getattr(block, "type", None) != "tool_result":
        return False
    if getattr(block, "name", None) not in SNAPSHOT_TOOLS:
        return False
    out = getattr(block, "output", None)
    return isinstance(out, list) and len(out) > 0 and hasattr(out[0], "text")


def _is_collapsed(text: str) -> bool:
    return text.startswith(MARKER_PREFIX) or text == PLACEHOLDER


def snapshot_url(text: str) -> str:
    """The page a snapshot describes. '' when the snapshot carries no URL line
    (an error result, or a page with no origin)."""
    m = _URL_RE.search(text or "")
    return m.group(1) if m else ""


def error_lines(text: str) -> list[str]:
    """Visible lines that read like a real page error, newest-page-first order."""
    hits = []
    for raw in (text or "").splitlines():
        line = raw.strip(" -\t")
        if not line or line.startswith("URL:"):
            continue
        if _ERROR_RE.search(line):
            # Drop the accessibility scaffolding, keep the human sentence.
            line = re.sub(r"\[ref=e\d+\]", "", line).strip()
            line = re.sub(r"\s+", " ", line)
            if line and line not in hits:
                hits.append(line[:_ERR_CHARS])
        if len(hits) >= _ERR_LINE_CAP:
            break
    return hits


def _short(url: str, n: int = 52) -> str:
    if not url:
        return "the page"
    return url if len(url) <= n else "…" + url[-(n - 1):]


def _marker(state: str, url: str, changed: str, errors: list[str]) -> str:
    """The frozen replacement text for one collapsed snapshot.

    Built only from this snapshot and earlier ones, so it never changes on a
    later pass -- that is what keeps the cached prefix stable.
    """
    where = _short(url)
    if state == "stale":
        head = (f"{MARKER_PREFIX}stale — the browser has since left {where}. "
                f"Its @eN refs are dead")
    else:
        head = (f"{MARKER_PREFIX}superseded — {where} was read again later. "
                f"Its @eN refs are stale")
    parts = [head]
    # Errors first: when both are present the error is the thing the agent must
    # not lose, and a long change summary would push it out of a skim.
    if errors:
        parts.append("page showed: " + " / ".join(f"“{e}”" for e in errors))
    if changed:
        parts.append(f"changed after this: {changed}")
    parts.append("browse_snapshot for the current page]")
    return ". ".join(parts)


def prune_browser_snapshots(context: list, keep: int = 1,
                            error_keep: int = 1) -> int:
    """Collapse dead browser snapshots in ``context`` (a list of Msg).

    ``keep``       -- always retain this many most-recent snapshots expanded;
                      the newest is the live page and the agent needs its refs.
    ``error_keep`` -- additionally retain this many most-recent error-bearing
                      snapshots, whatever their age (T3).

    Mutates blocks in place; returns how many were collapsed. Never raises.
    """
    if keep < 0:
        keep = 0

    # Every snapshot block still expanded, in conversation order.
    live: list[tuple[int, int, str]] = []          # (msg_i, block_i, text)
    for mi, msg in enumerate(context):
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if not _is_snapshot_result(block):
                continue
            text = _text_of(block)
            if _is_collapsed(text):
                continue                            # frozen; never revisited
            live.append((mi, bi, text))

    if len(live) <= keep:
        return 0

    urls = [snapshot_url(t) for _, _, t in live]
    errs = [error_lines(t) for _, _, t in live]

    # T3: the most recent error-bearing snapshots are retained.
    protected = set()
    if error_keep > 0:
        for i in range(len(live) - 1, -1, -1):
            if errs[i]:
                protected.add(i)
                if len(protected) >= error_keep:
                    break

    keep_from = len(live) - keep                    # indices >= this stay whole
    pruned = 0

    for i, (mi, bi, text) in enumerate(live):
        if i >= keep_from or i in protected:
            continue

        # T1: superseded (this page is read again later) vs stale (we left it).
        later_same = next((j for j in range(i + 1, len(live))
                           if urls[j] == urls[i] and urls[i]), None)
        state = "superseded" if later_same is not None else "stale"

        # T2: what changed between this copy and the one that replaced it.
        changed = ""
        if later_same is not None:
            try:
                from browser_supervisor import diff_snapshots
                changed = diff_snapshots(text, live[later_same][2],
                                         max_items=3)[:180]
            except Exception:
                changed = ""

        block = context[mi].content[bi]
        text_block_cls = type(block.output[0])
        block.output = [text_block_cls(
            type="text", text=_marker(state, urls[i], changed, errs[i]))]
        pruned += 1

    return pruned


class BrowserSnapshotPruner(MiddlewareBase):
    """Middleware that collapses dead browser snapshots before each reasoning
    (model) call, keeping the live context -- and the persisted state -- small
    during long browser tasks."""

    def __init__(self, keep: int = 1, error_keep: int = 1) -> None:
        self.keep = keep
        self.error_keep = error_keep

    async def on_reasoning(
        self,
        agent,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        try:
            n = prune_browser_snapshots(agent.state.context, keep=self.keep,
                                        error_keep=self.error_keep)
            if n:
                print(f"[prune] collapsed {n} dead browser snapshot(s)")
        except Exception as exc:  # never let pruning break a turn
            print(f"[prune] skipped ({type(exc).__name__}: {exc})")
        async for event in next_handler():
            yield event
