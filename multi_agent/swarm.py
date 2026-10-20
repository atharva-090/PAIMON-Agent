"""The Hermes swarm: an orchestrator agent + 4 specialist children.

Public entry point: ``Swarm.swarm_chat(user_text)`` -- the FastAPI server
calls this. Architecture mirrors ruflo's queen + workers but trimmed to one
process and one asyncio loop:

  user message -> orchestrator agent (LLM)
        |
        | delegate_task(child, "...")  -> schedules child as a bg task
        | delegate_task(child, "...")  -> schedules another in parallel
        | wait_for_results(60)         -> blocks until a child reports
        | check_status()               -> reads the shared StatusBoard
        | (loop until everyone is done/failed)
        |
        v
   final summary to user

Each specialist is built LAZILY (on first delegation) so we don't open 5
ChatGPT browsers idle at boot. After construction we reuse the agent
instance across delegations to keep the brain warm; we just RESET its
state.context before each new task so the child's context stays small and
the mock opens a fresh ChatGPT chat in the same browser (its
``completeWithSession`` already does this: if the incoming context isn't a
prefix-extension of what it saw, it's treated as a new task → new chat
in the same window).

The orchestrator's tool surface (delegate_task / check_status /
wait_for_results / list_specialists) is the ONLY way it acts. It never
gets the task-board, browser, or email tools -- those go to the
specialists, by design.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Optional

from agentscope.agent import Agent, ReActConfig
from agentscope.message import Msg, TextBlock
from agentscope.permission import PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, Toolkit, ToolChunk

import agent_events
import config
import memory_context
from .bus import MessageBus, StatusBoard, TaskRecord
from .roster import (ORCHESTRATOR_PERSONA, ORCHESTRATOR_PERSONA_NATIVE,
                     ORCHESTRATOR_REMINDER, ORCHESTRATOR_REMINDER_NATIVE,
                     ORCHESTRATOR_SESSION_ID, ROSTER, Specialist, by_name)
from .rotation import SessionRotator
from .findings import set_active, clear_active, findings_summary

# Rotator key for the orchestrator brain (children are keyed by their name).
_ORCH_KEY = "orchestrator"

# How many orchestrator messages survive a turn: 5 user + 5 assistant.
ORCH_CONTEXT_MSGS = int(os.environ.get("SWARM_ORCH_CONTEXT_MSGS", "10"))
# Nudge on the warm thread, then one fresh-chat re-roll, then give up and let
# the honest answer stand.
_ORCH_REFUSAL_RETRIES = int(
    os.environ.get("SWARM_ORCH_REFUSAL_RETRIES", "2"))

# Only compact once the context is ACTUALLY big. Compacting breaks the mock's
# prefix match, so the next turn starts a brand-new ChatGPT chat -- and a cold
# chat re-evaluates the request from scratch. Compacting on every turn made that
# the norm, and ChatGPT began soft-refusing a browser task whose delegation text
# carries site credentials ("I can't execute the requested browser task"),
# something it had done happily a few turns into a warm thread. So: stay warm
# while the context is small, and pay the cold start only when there is real
# bloat to shed.
# 12000 was too eager: ONE browser delegation with its poll results clears
# that, so the next user turn was cold nearly every time. The rotator's
# 80-message cap is the real backstop against unbounded growth; this only
# has to catch a genuinely bloated thread.
ORCH_COMPACT_AFTER_CHARS = int(
    os.environ.get("SWARM_ORCH_COMPACT_AFTER", "40000"))

# The same trap, one level down. Wiping a browser child's context at EVERY
# goal boundary made every goal a brand-new ChatGPT chat: the full system
# prompt and tool list were re-sent, and the model re-evaluated a
# first-contact message that restates the task (credentials included) with no
# evidence it had just been driving that browser -- so it soft-refused ("the
# live browse_* controls are not available in this run") on goals 3 and 4 of a
# 5-goal plan. Goals now share ONE warm thread, exactly like the single-shot
# run that worked; the wipe is a safety valve for real bloat, not the norm.
CHILD_COMPACT_AFTER_CHARS = int(
    os.environ.get("SWARM_CHILD_COMPACT_AFTER", "60000"))


def _context_chars(agent) -> int:
    return sum(len(str(getattr(m, "content", "")))
               for m in (getattr(agent.state, "context", []) or []))

# Headers of the blocks we splice into a user turn. They are useful ONCE, for
# the turn they were computed for; kept in history they go stale and re-bill
# every turn, so they are stripped when the turn is compacted.
_INJECTED_MARKERS = (
    "RELEVANT MEMORY (",
    "SAVED SKILL —",
    "SAVED SKILL -",
    "WHAT YOU KNOW ABOUT THE USER",
    "LIVE RIGHT NOW",
    "(Tool discipline:",
)


def _visible_text(msg) -> str:
    """The plain text of a message, ignoring tool_use / tool_result blocks."""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    parts = []
    for b in (content or []):
        btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
        if btype == "text":
            t = b.get("text") if isinstance(b, dict) else getattr(b, "text", "")
            if t:
                parts.append(t)
    return "\n".join(parts)


def _strip_injections(text: str) -> str:
    """Cut a user turn back to what the human actually typed."""
    cut = len(text)
    for marker in _INJECTED_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def compact_orchestrator_context(agent, keep: int = ORCH_CONTEXT_MSGS) -> int:
    """Collapse a finished turn down to what the user actually sees.

    A single orchestrator turn generates delegate_task calls, several
    wait_for_results polls and their payloads, then the reply. All of it stays
    in context and is re-sent to the brain forever. Measured on a real run: two
    user messages produced ~10k chars of internal echo, and the same email draft
    was delivered three times in one turn.

    None of that is needed once the turn is over -- the board is the durable
    ledger, and check_status() can re-read it on demand. So after each turn we
    keep only the visible conversation: the user's own words (injected memory
    and reminder blocks stripped, since they are recomputed fresh each turn) and
    the assistant's final reply, capped at the last ``keep`` messages.

    NOTE ON THE MOCK: this deliberately breaks the delta/prefix match, so the
    mocker treats the next turn as a new task and opens a fresh ChatGPT chat
    with this compact context as the full prompt. That is the intended trade:
    a bounded, deterministic window every turn instead of a thread that grows
    without limit -- and it re-sends the persona each turn, which also stops the
    system prompt decaying (the cause of the specialist-name tool calls).
    Returns the number of messages dropped.
    """
    ctx = list(getattr(agent.state, "context", []) or [])
    before = len(ctx)
    kept = []
    for m in ctx:
        role = getattr(m, "role", None)
        if role not in ("user", "assistant"):
            continue                      # tool results
        text = _visible_text(m)
        if role == "user":
            text = _strip_injections(text)
        if not text.strip():
            continue                      # tool_use-only assistant turns
        kept.append(Msg(name=getattr(m, "name", role),
                        content=[TextBlock(type="text", text=text)],
                        role=role))
    kept = kept[-keep:] if keep > 0 else kept
    agent.state.context = kept
    return before - len(kept)


# ===========================================================================
#  Brain factory (mirrors task_agent.build_model but per-session-id)
# ===========================================================================
def _build_model(session_id: str, profile: str | None = None):
    """Return an OpenAIChatModel pinned to ``session_id`` against the mock
    (so each agent lives in its own ChatGPT browser), or a real-provider
    model when the profile is openai/anthropic/gemini (no session pinning,
    real APIs ignore unknown body fields)."""
    name = profile or config.LLM_PROFILE
    prof = config.LLM_PROFILES.get(name) or config.LLM_PROFILES["mock"]
    # Must match config.resolve_profile exactly: is_mock_backend() decides which
    # PERSONA the agent gets, and this decides which MODEL it talks to. If they
    # disagree, an agent gets native-tool-calling instructions while actually
    # talking to the mock (or the reverse), which breaks tool calling outright.
    if not prof.get("api_key"):
        prof, name = config.LLM_PROFILES["mock"], "mock"

    # Every model is wrapped so each call's token usage lands in
    # data/usage.jsonl, tagged with the agent that spent it. On a real API this
    # is the only way to see where a run's cost actually went; the mock reports
    # no usage, so the wrapper simply records nothing there.
    import usage_log
    provider = prof["provider"]
    if provider == "openai":
        from agentscope.credential import OpenAICredential
        credential = OpenAICredential(api_key=prof["api_key"],
                                      base_url=prof["base_url"])
        if name == "mock":
            from mock_model import SessionPinnedOpenAIModel
            model = usage_log.metered(SessionPinnedOpenAIModel)(
                credential=credential, model=prof["model"], stream=False,
                session_id=session_id,
            )
        else:
            from agentscope.model import (OpenAIChatModel,
                                          OpenAIResponseModel)
            if prof.get("responses"):
                # The stock Responses formatter puts the function_call ITEM id
                # on function_call_output.call_id, which OpenAI rejects; see
                # response_formatter_fix.
                from response_formatter_fix import (
                    PatchedOpenAIResponseFormatter)
                model = usage_log.metered(OpenAIResponseModel)(
                    credential=credential, model=prof["model"], stream=False,
                    formatter=PatchedOpenAIResponseFormatter())
            else:
                model = usage_log.metered(OpenAIChatModel)(
                    credential=credential, model=prof["model"], stream=False)
    elif provider == "anthropic":
        from agentscope.credential import AnthropicCredential
        from agentscope.model import AnthropicChatModel
        model = usage_log.metered(AnthropicChatModel)(
            credential=AnthropicCredential(api_key=prof["api_key"]),
            model=prof["model"], stream=False,
        )
    elif provider == "gemini":
        from agentscope.credential import GeminiCredential
        from agentscope.model import GeminiChatModel
        model = usage_log.metered(GeminiChatModel)(
            credential=GeminiCredential(api_key=prof["api_key"]),
            model=prof["model"], stream=False,
        )
    else:
        raise ValueError(f"unknown provider: {provider}")
    # session_id doubles as the agent label: "hermes-browser" -> "browser".
    return usage_log.attach(model, session_id.split("-", 1)[-1], name)


# ===========================================================================
#  Specialist builder
# ===========================================================================
def _build_specialist(spec: Specialist, profile: str | None = None) -> Agent:
    """Construct one child agent. Reused across delegations; state.context is
    reset between tasks so each delegation looks like a fresh ChatGPT chat
    to the mock."""
    tools = [FunctionTool(f, is_read_only=False) for f in spec.tool_factory()]
    toolkit = Toolkit(tools=tools)
    state = AgentState()
    state.permission_context.mode = PermissionMode.BYPASS

    # A pure observer that copies this child's reasoning and tool calls onto
    # the event bus for the live UI. First in the chain so it sees everything
    # the later middlewares do; it never alters what flows through.
    middlewares = [agent_events.StreamEmitter(spec.name)]
    # The snapshot pruner REWRITES already-delivered messages (collapses old
    # snapshots). Against the mock that breaks its delta/prefix-match continuity,
    # so it opens a NEW ChatGPT chat every step (the "new tabs back to back"
    # bug). The mock already keeps the prompt small by sending only the delta, so
    # the pruner is unnecessary there and actively harmful -- enable it ONLY for
    # real, stateless providers that re-send the full context each call.
    # ...and only while a page is still big enough to be worth the damage.
    # A2 turned a 7.5 KB accessibility tree into a ~2.3 KB element table, which
    # a cached prefix carries for almost nothing, so the trade reversed: the
    # pruner now costs more in lost cache than it saves in tokens.
    if (spec.has_browser and not config.is_mock_backend(profile)
            and not config.BROWSER_ELEMENT_TABLE):
        from context_pruner import BrowserSnapshotPruner
        middlewares.append(BrowserSnapshotPruner(keep=1))
        # Paired with the pruner on purpose. Collapsing snapshots removes the
        # record of what happened on those pages, so put a compact, factual
        # trail of this agent's own actions back at the tail of its context.
        # Enabled ONLY where the pruner is: against the mock nothing is pruned,
        # and rewriting its context breaks the delta/prefix-match continuity.
        import action_log
        middlewares.append(action_log.ActionTrailInjector(spec.name))

    return Agent(
        name=spec.name,
        # The mock parses tool calls out of the model's TEXT, so its persona
        # teaches that JSON protocol. A real provider takes tools natively and
        # returns structured calls -- there the same examples would coach the
        # model to write JSON as message content instead of calling anything.
        system_prompt=spec.persona_for(config.is_mock_backend(profile)),
        model=_build_model(spec.session_id, profile),
        toolkit=toolkit,
        state=state,
        middlewares=middlewares,
        react_config=ReActConfig(max_iters=spec.max_iters),
    )


# ===========================================================================
#  Swarm
# ===========================================================================
class Swarm:
    """One swarm = one orchestrator + N lazily-constructed specialist children.

    The board is the single source of truth for "what's running". The
    orchestrator's tools all close over this instance.
    """

    def __init__(self, profile: str | None = None) -> None:
        self.profile = profile
        self.board = StatusBoard()
        self.bus = MessageBus()
        self._specs = by_name()
        self._children: dict[str, Agent] = {}    # built lazily
        # One lock per child so two delegations to the SAME specialist
        # serialise instead of racing on one shared agent instance (that race
        # was what created the duplicate task). Different children still run
        # concurrently.
        self._child_locks: dict[str, asyncio.Lock] = {}
        self.orchestrator: Agent = _build_orchestrator(self, profile)
        # Tracks the board version the orchestrator has already observed,
        # so wait_for_results only wakes on genuinely NEW transitions.
        self._last_seen_version: int = self.board.version()
        # Concurrency cap: total non-orchestrator brains we'll have running
        # at once. The mock's MAX_SESSIONS gates this hard; bump it via env
        # when you spin up the swarm (we suggest 8 -- orchestrator + 4 kids
        # + headroom).
        self.max_concurrent_children = int(
            os.environ.get("SWARM_MAX_CONCURRENT", "4"))
        self._concurrent_sem = asyncio.Semaphore(self.max_concurrent_children)
        # Idle/length context rotation for every warm brain (orchestrator +
        # children). The board is the durable memory, so a rotation just clears
        # a brain's live context before it next runs -- keeping these five
        # contexts from growing unbounded over a long conversation. See
        # rotation.py.
        self.rotator = SessionRotator()
        # A compact transcript of THIS conversation (plain user/assistant text),
        # archived to sessions.py on reset() so the cold-path learner can extract
        # durable facts from swarm conversations too (mirrors the single agent).
        self._turns: list[dict] = []
        self._started_at: str | None = None

    # -- specialist lifecycle ----------------------------------------------
    def _ensure_child(self, name: str) -> Agent:
        if name not in self._specs:
            raise KeyError(f"no such specialist '{name}'")
        if name not in self._children:
            spec = self._specs[name]
            self._children[name] = _build_specialist(spec, self.profile)
        return self._children[name]

    def _child_lock(self, name: str) -> asyncio.Lock:
        """One lock per child -- except that every browser-capable child shares
        a single one.

        roster.py's header states the standing assumption plainly: "v1 only has
        ONE browser-using specialist live at a time (browser_agent xor
        comms_agent for a given task), so they can SHARE the default Chrome --
        no concurrent-collision risk". Nothing enforced it. The lock was keyed
        by child NAME, so browser_agent and comms_agent took different locks
        and ran together under a semaphore of 4.

        They then shared one Chrome AND one module-global supervisor context:
        comms_agent's set_context() wiped browser_agent's goal, checklist, step
        counter and trail in the middle of its run. In the log the BRE run
        opens with a five-item checklist and its next two clicks produce no
        page record at all, because the context they would have written to had
        been replaced.

        Serialising them is the small, correct fix. Running them in parallel
        needs distinct browser_session names and a per-session toolset, which
        is the refactor that header describes and defers.
        """
        spec = self._specs.get(name)
        key = "__browser__" if getattr(spec, "has_browser", False) else name
        return self._child_locks.setdefault(key, asyncio.Lock())

    async def _run_specialist(self, rec: TaskRecord) -> None:
        """Background coroutine that actually runs a delegated child.

        The child is WARM: we do NOT wipe its context between delegations, so
        a follow-up delegation ('send the draft you prepared', 'revise it to
        say Monday') continues the same ChatGPT chat and the child remembers
        its own prior work -- that's how the orchestrator coordinates with a
        pre-existing agent. The per-child lock guarantees one delegation at a
        time per specialist, keeping that warm context coherent (and killing
        the duplicate-task race). The global semaphore still caps how many
        DIFFERENT children run at once."""
        lock = self._child_lock(rec.child)
        async with self._concurrent_sem, lock:
            try:
                agent = self._ensure_child(rec.child)
                # Point record_answer at THIS task so the child's recorded
                # sub-answers land on this record (scoped per child).
                set_active(rec.child, rec)
                # Supervised browser run (Section K P1/P5/P6): browser-capable
                # children get a goal/checklist/trail header + loop monitor woven
                # into every snapshot they receive. No-op for non-browser
                # specialists (they never call browse_*, so after_action never
                # fires). One browser child runs at a time, so the module-global
                # context is safe (matches browser_tools' existing globals).
                spec = self._specs.get(rec.child)
                _supervised = bool(spec and getattr(spec, "has_browser", False))
                if _supervised:
                    try:
                        import browser_supervisor as _sup
                        # NOTE: items are deliberately NOT set here. This call
                        # only opens the supervised run; the checklist is
                        # installed once, below, from the planned goals (or from
                        # auto-extraction when planning produced none). Setting
                        # it twice is what corrupted the last run: this call
                        # auto-extracted from a CONTINUATION description and
                        # replaced a good 5-goal checklist with
                        # [browse_snapshot, an_attribute_1, ...].
                        _sup.set_context(goal=task_text(rec.description),
                                         max_steps=getattr(spec, "max_iters", 0),
                                         items=[])
                    except Exception:
                        _supervised = False
                # Rotate this child's context BEFORE we snapshot `pre`, so a
                # rotated run measures its tool calls from a clean slate and the
                # mock opens a fresh ChatGPT chat. Only fires on an idle/length
                # boundary; a normal warm follow-up is untouched.
                self.rotator.maybe_rotate(rec.child, agent)
                pre = len(agent.state.context)
                # The child sees ONLY this description (not the wiki, not the
                # user's words). Fold in any memory relevant to the task so it
                # works with full context — e.g. a delegation mentioning "Umesh"
                # arrives with what we know about Umesh attached. Instant + local.
                child_text = rec.description
                try:
                    # A1: not for a browser child. memory_context.relevant() is
                    # an FTS match over whole page bodies, and a web task is
                    # long prose full of common words, so it matched on them:
                    # every browser delegation in the 13 Oct run arrived with
                    # the same four unrelated documents (a vendor agreement, a
                    # sales workbook, a readme, a procurement contract) --
                    # 1,273 characters, in the goal message, every goal. The
                    # orchestrator's own persona already says to copy any fact
                    # the child needs into the delegation, and it does.
                    # procedural_hint stays: it only fires on a page tagged
                    # 'skill', so it cannot produce this kind of noise.
                    rel = ("" if getattr(spec, "has_browser", False)
                           else memory_context.relevant(rec.description))
                    skill = memory_context.procedural_hint(rec.description)
                except Exception:
                    rel = skill = ""
                # The child's persona is a system prompt: the mock delivers it
                # once at thread start and never again, so its rules decay over
                # a long thread. spec.reminder carries the few rules the child
                # must not forget (e.g. comms_agent's send discipline) and rides
                # in the highest-recency slot -- last, right before the mock's
                # own protocol footer.
                spec_reminder = self._specs[rec.child].reminder_for(
                    config.is_mock_backend(self.profile))
                addon = "\n\n".join(
                    x for x in (rel, skill, spec_reminder) if x
                )
                if addon:
                    child_text = f"{rec.description}\n\n{addon}"
                # Long multi-outcome web runs die on context, not capability.
                # Plan the outcomes once, then execute them one at a time with a
                # compaction between each. Browser children only -- everyone
                # else's work is short enough to fit in one pass.
                goals = []
                goal_tools: list[str] = []
                # Goal segmentation is a BROWSER concern: only a web flow runs
                # long enough to die on context, and only it can resume from a
                # snapshot between goals. comms_agent also carries has_browser,
                # so gating on that flag alone made it pay a planning
                # round-trip on every long delegation for nothing.
                if _supervised and rec.child in _GOAL_PLANNING_CHILDREN:
                    goals = await self._plan_goals(agent, rec.description)
                    # Install the checklist ONCE, here, now that we know whether
                    # planning produced goals. Planned outcomes are always better
                    # than name-scraping; without them we fall back to the
                    # scraper, which is also what the un-planned path relies on.
                    try:
                        import browser_supervisor as _sup
                        _sup.context().set_items(
                            goals or _sup.parse_checklist_items(rec.description),
                            as_goals=bool(goals))
                    except Exception:
                        pass
                if goals:
                    print(f"[swarm] {rec.child} planned {len(goals)} goals: "
                          f"{goals}", flush=True)
                    text, goal_tools = await self._run_goals(
                        rec, agent, goals, _supervised, addon)
                else:
                    msg = Msg(
                        name="user",
                        content=[TextBlock(type="text", text=child_text)],
                        role="user",
                    )
                    reply = await agent.reply(msg)
                    text = (reply.get_text_content() if reply else "") or ""
                used = _tool_names_used(agent.state.context, since=pre)
                if goal_tools:
                    # The per-goal tally is the accurate one -- it was taken
                    # inside each goal, before any boundary reset could move
                    # the ground under `pre`.
                    used = goal_tools + [t for t in used
                                         if t not in goal_tools]
                # Deterministic anti-hallucination guards: the weak ChatGPT-web
                # brain sometimes NARRATES a result instead of emitting the tool
                # call -- it claims an email "sent successfully" without calling
                # send_email, or (the scripter) claims a file "couldn't be found"
                # without calling any file tool. We detect that mismatch and force
                # the warm child to actually call the tool. If it still won't
                # after a couple of nudges, the truth (no action) stands and the
                # footer/board report it honestly.
                for _ in range(3):
                    if _claims_sent_without_sending(text, used):
                        nudge_text = _SEND_NUDGE
                    elif _claims_blocked_without_trying(text, used):
                        nudge_text = _FALSE_BLOCKER_NUDGE
                    elif _claims_failed_without_calling(text, used):
                        nudge_text = _FAILED_WITHOUT_TRYING_NUDGE
                    elif _claims_no_file_without_reading(rec.child, text, used):
                        nudge_text = _FILE_NUDGE
                    elif _claims_made_report_without_writing(rec.child, text,
                                                             used):
                        nudge_text = _REPORT_NUDGE
                    elif _browser_bailed_with_work_left(rec.child, text, used):
                        nudge_text = _BROWSER_CONTINUE_NUDGE
                    else:
                        break
                    mark = len(agent.state.context)
                    nudge = Msg(
                        name="user",
                        content=[TextBlock(type="text", text=nudge_text)],
                        role="user",
                    )
                    reply = await agent.reply(nudge)
                    text = (reply.get_text_content() if reply else "") or text
                    used = _tool_names_used(agent.state.context, since=pre)
                    # REFUSAL-LOOP ESCAPE: a browser child that, AFTER this nudge,
                    # STILL made no browse_* call and is still bailing is stuck in
                    # a poisoned ChatGPT thread — arguing with it again is futile.
                    # Wipe its context (fresh chat) and re-issue the task WITH the
                    # progress so far, so it resumes instead of restarting. One shot.
                    if (nudge_text is _BROWSER_CONTINUE_NUDGE
                            and not _used_browser_tool(
                                _tool_names_used(agent.state.context, since=mark))
                            and _browser_bailed_with_work_left(rec.child, text, used)):
                        text, used = await self._rotate_and_resume(rec, agent)
                        break
                # Deterministic assembly: if the child recorded structured
                # sub-answers (record_answer), surface them as the authoritative
                # block of the report -- so a multi-part answer comes from the
                # ledger, not from the weak brain re-aggregating in its last line.
                ledger = findings_summary(rec)
                report = text.strip()
                if ledger:
                    report = f"RECORDED ANSWERS: {ledger}\n\n{report}"
                # Deterministic loop signal (P6 tier-2): if the run is spinning /
                # stalled with the checklist incomplete, surface the TRUTH on the
                # board so the orchestrator reports "stuck", not a fake success.
                if _supervised:
                    try:
                        import browser_supervisor as _sup
                        stuck = _sup.stuck_summary()
                    except Exception:
                        stuck = None
                    if stuck:
                        report = f"[AUDITOR] {stuck}\n\n{report}"
                self.board.mark(rec.id, "done", report=report,
                                tool_calls=len(used), tools_used=used)
                agent_events.publish("report", rec.child, task_id=rec.id,
                                     status="done", report=report,
                                     tools_used=list(used))
            except Exception as exc:
                self.board.mark(rec.id, "failed", error=f"{type(exc).__name__}: {exc}")
                agent_events.publish("report", rec.child, task_id=rec.id,
                                     status="failed",
                                     report=f"{type(exc).__name__}: {exc}",
                                     tools_used=[])
            finally:
                # Reset this child's idle clock whether it succeeded or failed --
                # either way it just ran, so 'idle' is measured from now.
                self.rotator.touch(rec.child)
                clear_active(rec.child)
                try:  # end the supervised browser run (Section K)
                    import browser_supervisor as _sup
                    _sup.clear_context()
                except Exception:
                    pass

    async def _rotate_and_resume(self, rec: TaskRecord, agent) -> tuple:
        """Recover a browser child stuck in a refusal loop ('my tools aren't
        available') that survived a nudge. Re-prompting the SAME poisoned ChatGPT
        thread is hopeless, so we wipe the child's context (which makes the mock
        open a FRESH chat) and re-issue the task WITH the progress it already made
        — so it resumes, not restarts. Returns (text, tools_used) of the fresh run.

        Progress comes from the supervisor's deterministic checklist, so the
        resumed run is told exactly what's done and what's left."""
        done, remaining = [], []
        try:
            import browser_supervisor as _sup
            cl = _sup.context().checklist
            if cl:
                done, remaining = cl.done, cl.remaining
        except Exception:
            pass
        agent.state.context = []  # fresh chat — escape the poisoned thread
        prog = ""
        if done or remaining:
            prog = (f"\n\nPROGRESS (already created — do NOT recreate these): "
                    f"{', '.join(done) or 'none'}. STILL TO CREATE: "
                    f"{', '.join(remaining) or 'snapshot the list to check'}.")
        resume = (rec.description + prog +
                  "\n\nThe browser is ALREADY OPEN on this task and your browse_* "
                  "tools ARE available right now. Call browse_snapshot FIRST to see "
                  "the current page, then finish every remaining item. Do NOT say "
                  "any tool is unavailable — just call it.")
        pre = len(agent.state.context)
        reply = await agent.reply(
            Msg(name="user", content=[TextBlock(type="text", text=resume)],
                role="user"))
        text = (reply.get_text_content() if reply else "") or ""
        used = _tool_names_used(agent.state.context, since=pre)
        print(f"[swarm] rotated {rec.child} out of a refusal loop and resumed "
              f"({len(used)} tool calls on the fresh chat)")
        return text, used

    # -- public entry point used by the server ------------------------------
    async def swarm_chat(self, user_text: str) -> str:
        """One full orchestrator turn. The board is NOT cleared (it's the
        team's running ledger across the whole conversation) and the
        orchestrator stays WARM (it remembers what it delegated and what each
        child reported), so follow-ups like 'what email did you prepare?'
        are answered from memory / the ledger instead of starting blank.
        Returns the orchestrator's final reply."""
        self.board.begin_turn()
        # Rotate the orchestrator's own context on an idle/length boundary
        # before it runs. Safe to clear: the board ledger is the durable memory,
        # and the orchestrator recovers earlier work via check_status(). This is
        # what keeps its per-turn context growth from running away over a long
        # conversation.
        fresh = len(self.orchestrator.state.context) == 0
        self.rotator.maybe_rotate(_ORCH_KEY, self.orchestrator)
        fresh = fresh or len(self.orchestrator.state.context) == 0
        # Only react to transitions that happen FROM HERE in wait_for_results;
        # don't immediately "wake" on last turn's already-finished tasks.
        self._last_seen_version = self.board.version()
        if self._started_at is None:
            from datetime import datetime, timezone
            self._started_at = datetime.now(timezone.utc).isoformat()
            # One trail per swarm conversation, shared by every child: the
            # specialists' actions belong to the same run as the orchestrator's.
            try:
                import action_log
                action_log.start_run("swarm")
            except Exception:
                pass
        # Splice memory into the orchestrator's prompt (instant, local): the
        # standing digest + live open threads on a FRESH conversation, and the
        # per-message relevant block every turn. The orchestrator passes specifics
        # down to children (its persona says so), and delegate_task also re-injects
        # relevant memory per child.
        msg_text = user_text
        extras: list[str] = []
        # The orchestrator has no current_datetime tool, so a relative date in a
        # delegation was pure guesswork -- one run wrote "Thursday 26 March
        # 2026" into a meeting request sent on 20 October. One local read, no
        # model call, and "tomorrow" resolves correctly.
        try:
            import scheduler
            _n = scheduler.now()
            extras.append(
                "NOW: " + scheduler.now_iso() + " (" + _n.strftime("%A") +
                "). Resolve every relative date against this before you write "
                "it into a delegation; never infer today's date.")
        except Exception:
            pass
        try:
            if fresh:
                sc = memory_context.session_context()
                if sc:
                    extras.append(sc)
            rel = memory_context.relevant(user_text)
            if rel:
                extras.append(rel)
            skill = memory_context.procedural_hint(user_text)
            if skill:
                extras.append(skill)
        except Exception:
            extras = []
        # Tool discipline goes on EVERY turn, not just the first. The persona's
        # <specialists_are_not_tools> block decays over a long ChatGPT thread and
        # the orchestrator starts calling a specialist's NAME as a tool; this
        # rides in the highest-recency slot to keep the shape in front of it.
        # Appended last so it is the final thing before the protocol footer.
        extras.append(ORCHESTRATOR_REMINDER
                      if config.is_mock_backend(self.profile)
                      else ORCHESTRATOR_REMINDER_NATIVE)
        if extras:
            msg_text = user_text + "\n\n" + "\n\n".join(extras)
        msg = Msg(
            name="user",
            content=[TextBlock(type="text", text=msg_text)],
            role="user",
        )
        reply = await self.orchestrator.reply(msg)
        self.rotator.touch(_ORCH_KEY)
        text = (reply.get_text_content() if reply else "") or "(no text)"
        # Orchestrator-level anti-hallucination guard: it has no file-reading
        # ability, so on a file-related turn it must NOT answer by claiming the
        # file is missing/unreadable without delegating to the scripter. If it
        # did, force a real delegation and re-answer from the scripter's report.
        if _looks_file_related(user_text):
            for _ in range(2):
                if any(t["child"] == "scripter"
                       for t in self.board.current_snapshot()):
                    break
                if not _orchestrator_ducked_file(text):
                    break
                nudge = Msg(
                    name="user",
                    content=[TextBlock(type="text", text=_ORCH_FILE_NUDGE)],
                    role="user",
                )
                reply = await self.orchestrator.reply(nudge)
                self.rotator.touch(_ORCH_KEY)
                text = (reply.get_text_content() if reply else "") or text
        # Cold-start refusal guard. A cold ChatGPT chat is unavoidable (first
        # turn after a restart, after the idle rotation, after a length
        # rotation) and on a cold chat the brain sometimes denies having any
        # tools. That denial used to be terminal -- it went straight to the
        # user. Push back instead: the nudge is a second message, so it lands
        # on a NOW-WARM thread, which is the condition the brain behaves in.
        for attempt in range(_ORCH_REFUSAL_RETRIES):
            if not _orchestrator_refused_to_act(
                    text, self.board.current_snapshot()):
                break
            print(f"[swarm] orchestrator refused without delegating "
                  f"(attempt {attempt + 1}/{_ORCH_REFUSAL_RETRIES}); "
                  f"nudging", flush=True)
            nudge_text = _ORCH_REFUSAL_NUDGE
            if attempt:
                # Still refusing on a warm thread means THIS chat is poisoned,
                # not that the request is impossible. Drop it, re-issue the
                # user's own turn into a fresh one, and take the new roll.
                self.orchestrator.state.context = []
                nudge_text = msg_text
                print("[swarm] orchestrator thread poisoned; reissuing on a "
                      "fresh chat", flush=True)
            reply = await self.orchestrator.reply(
                Msg(name="user",
                    content=[TextBlock(type="text", text=nudge_text)],
                    role="user"))
            self.rotator.touch(_ORCH_KEY)
            text = (reply.get_text_content() if reply else "") or text
        # Patience/finish guard: the orchestrator must not end its turn while its
        # OWN delegations are still running and hand the user a "still working /
        # please restart" non-answer (its wait_for_results timed out and it gave
        # up early). If any current-turn task is still in flight, DRAIN it here,
        # then make the orchestrator write the real answer from the finished
        # reports instead of punting.
        text = await self._finish_pending_delegations(text)
        # Capture the plain turn (the user's own words + our reply) for this
        # conversation's transcript, archived + learned-from on reset().
        self._turns.append({"role": "user", "text": user_text})
        self._turns.append({"role": "assistant", "text": text})
        # The turn is over: throw away its machinery (delegations, polls, tool
        # payloads) and keep only the visible conversation. Done LAST so every
        # guard above still sees the full working context -- and only once the
        # context is genuinely large, because compacting costs a cold ChatGPT
        # chat on the next turn (see ORCH_COMPACT_AFTER_CHARS).
        size = _context_chars(self.orchestrator)
        if size > ORCH_COMPACT_AFTER_CHARS:
            dropped = compact_orchestrator_context(self.orchestrator)
            print(f"[swarm] compacted orchestrator context: {size} chars, "
                  f"-{dropped} msgs, "
                  f"{len(self.orchestrator.state.context)} kept", flush=True)
        footer = _ground_truth_footer(self.board)
        return f"{text}\n\n{footer}" if footer else text

    async def _plan_goals(self, agent, task: str) -> list[str]:
        """Ask the child ONCE for the list of outcomes. [] = run it single-shot."""
        if not GOAL_PLANNING or len(task) < GOAL_PLAN_MIN_CHARS:
            return []
        saved = list(getattr(agent.state, "context", []) or [])
        try:
            msg = Msg(name="user",
                      content=[TextBlock(type="text",
                                         text=_PLAN_PROMPT.format(task=task))],
                      role="user")
            reply = await agent.reply(msg)
            goals = parse_plan(reply.get_text_content() if reply else "")
        except Exception as exc:
            print(f"[swarm] goal planning failed ({type(exc).__name__}: {exc}); "
                  f"running single-shot", flush=True)
            goals = []
            # Rewind ONLY on failure -- a half-written turn is worth dropping.
            # On success the exchange STAYS in context: the mock sends a delta
            # only while the history it already saw is an unbroken prefix, so
            # truncating here would force goal 1 into a fresh chat -- the very
            # cold start this path exists to avoid. Two short messages, and the
            # agent keeps its own plan in view.
            agent.state.context = saved
        return goals if len(goals) > 1 else []

    async def _run_goals(self, rec: TaskRecord, agent, goals: list[str],
                         supervised: bool,
                         addon: str = "") -> tuple[str, list[str]]:
        """Execute one goal per turn, compacting between them.

        `addon` is the memory context + the specialist reminder. It was being
        built in the caller and then dropped here: only the single-shot branch
        used it, so on a PLANNED run -- the long supervised web flows that need
        it most -- the browser child never saw its reminder at all, and the
        rules it carries (no filesystem tools, never fetch /api/reports) were
        absent from every call.

        The browser is NOT touched between goals -- same Chrome, same login,
        same page. Only the conversation is reset, and the next goal's message
        tells the agent to re-orient with a single snapshot.
        """
        done = [False] * len(goals)
        reports: list[str] = []
        # What each finished goal actually produced, one line each. Travels
        # with every later goal so the ids and quirks found early are still in
        # front of the agent at the end.
        results: list[str] = []
        carry = ""
        # Accumulated across goals, because a boundary reset invalidates any
        # index-based measurement the caller took before this loop started.
        all_used: list[str] = []
        for i, goal in enumerate(goals):
            # Publish which goal is in flight so the orchestrator's status poll
            # can show "goal 2/5 · 1 done, 3 pending" instead of a bare
            # "in_progress" it reads as a hang.
            if supervised:
                try:
                    import browser_supervisor as _sup
                    _sup.set_goal(i + 1, len(goals), goal)
                except Exception:
                    pass
            text = _goal_message(rec.description, goals, done, i, carry,
                                 results)
            if addon:
                # Last, so it sits in the highest-recency slot -- the same
                # reasoning the single-shot branch uses.
                text = f"{text}\n\n{addon}"
            msg = Msg(name="user", content=[TextBlock(type="text", text=text)],
                      role="user")
            pre_goal = len(getattr(agent.state, "context", []) or [])
            try:
                reply = await agent.reply(msg)
            except Exception as exc:
                reports.append(f"goal {i+1} ({goal}) FAILED: "
                               f"{type(exc).__name__}: {exc}")
                break
            out = (reply.get_text_content() if reply else "") or ""
            used = _tool_names_used(agent.state.context, since=pre_goal)

            # A goal that touched NO browse_* tool did not happen, whatever the
            # reply says. The whole-delegation guards below cannot catch this:
            # they see the aggregate report and a tools_used that goal 1 already
            # filled in, so a refusal on goal 3 slid through and was ticked done.
            # Nudge once here, per goal, against THIS goal's own tool record.
            if not _used_browser_tool(used) and _browser_bailed_with_work_left(
                    rec.child, out, used):
                print(f"[swarm] {rec.child} goal {i+1} bailed without touching "
                      f"the browser; nudging", flush=True)
                nudge = Msg(name="user",
                            content=[TextBlock(type="text",
                                               text=_BROWSER_CONTINUE_NUDGE)],
                            role="user")
                try:
                    reply = await agent.reply(nudge)
                    out = (reply.get_text_content() if reply else "") or out
                    used = _tool_names_used(agent.state.context, since=pre_goal)
                except Exception:
                    pass

            carry = out.strip().splitlines()[0][:200] if out.strip() else ""
            acted = _used_browser_tool(used)
            for t in used:
                if t not in all_used:
                    all_used.append(t)

            if not acted:
                # Do not tick it and do not march on: later goals depend on this
                # one (a container before its contents), so continuing would pile
                # fiction on fiction. Stop and report the truth.
                reports.append(f"goal {i+1}/{len(goals)} ({goal}) NOT DONE - the "
                               f"browser child made no browse_* call. It said: "
                               f"{carry or 'nothing'}")
                skipped = [f"{j+1}. {g}" for j, g in enumerate(goals) if j > i]
                if skipped:
                    reports.append("NOT ATTEMPTED: " + "; ".join(skipped))
                print(f"[swarm] {rec.child} goal {i+1}/{len(goals)} NOT DONE "
                      f"(no browser tool used) - stopping", flush=True)
                break

            # Was this goal EVIDENCED on the page, or only claimed?
            #
            # Checklist items are ticked deterministically by
            # `mark_present_in` -- the item's text appearing as a saved record
            # on the page -- and its class docstring is explicit that they are
            # "never [ticked] by the model's claim". But the line below used to
            # call `mark_done(goal)` FIRST and then read the tick back, so
            # `ticked` was true by construction and was used for nothing but a
            # log suffix. `done[i] = True` ran unconditionally.
            #
            # That is how a run announced "'CC and Hypo Loans' category
            # created" while the supervisor's own checklist still read ☐: the
            # form had submitted empty and nothing was ever on the page.
            #
            # So read the evidence BEFORE force-ticking. mark_done still runs,
            # because the rendered checklist is what orients the agent between
            # goals and freezing it at ☐ would tell it nothing was done -- but
            # it no longer decides what we REPORT.
            # Three outcomes, not two: verified, contradicted, or simply
            # not checkable. Only the middle one deserves a warning.
            checkable = evidenced = False
            if supervised:
                try:
                    import browser_supervisor as _sup
                    ctx = _sup.context()
                    checkable, evidenced = _sup.goal_is_evidenced(
                        goal, ctx.last_snapshot or "")
                    if ctx.checklist:
                        # Only the page may award a ✓. mark_done ran here
                        # unconditionally, so the header told the agent a
                        # category existed that had never been created -- and
                        # the header is what it orients by. A goal nothing can
                        # confirm gets '?', and the step trail lets the agent
                        # judge it for itself.
                        if evidenced or not checkable:
                            ctx.checklist.mark_done(goal)
                        else:
                            ctx.checklist.mark_claimed(goal)
                except Exception:
                    pass
            done[i] = True
            results.append(
                f"{i + 1}. {goal[:80]} -> "
                + (carry or "no report")
                + ("" if not (checkable and not evidenced)
                   else "  [CLAIMED, not found on the page]"))
            if supervised and checkable and not evidenced:
                # Flow continues -- many goals ("log in", "summarise X") leave
                # no record a page check could find, so refusing to proceed
                # would stall on false negatives. What changes is that the
                # report stops asserting something we cannot show.
                reports.append(
                    f"goal {i+1}/{len(goals)} ({goal}): CLAIMED BUT NOT "
                    f"VERIFIED — nothing matching this goal was found on the "
                    f"page afterwards. Treat it as unconfirmed. It said: "
                    f"{carry or 'nothing'}")
            else:
                reports.append(f"goal {i+1}/{len(goals)} ({goal}): "
                               f"{carry or 'no report'}")
            _how = ("done (verified on page)" if evidenced
                    else "CLAIMED (NOT on the page)" if checkable
                    else "done (nothing on a page could confirm this)")
            print(f"[swarm] {rec.child} goal {i+1}/{len(goals)} {_how}",
                  flush=True)

            # Goal boundary. The conversation stays WARM by default -- dropping
            # it here is what turned each goal into a fresh chat. Wipe only when
            # the accumulated snapshots are genuinely too big to carry; then the
            # next goal's message re-states the task, the plan, the status and
            # the carry-over note, and tells it to re-orient with one snapshot.
            # The browser is never touched either way.
            if i + 1 < len(goals):
                size = _context_chars(agent)
                if size > CHILD_COMPACT_AFTER_CHARS:
                    agent.state.context = []
                    print(f"[swarm] {rec.child} context {size} chars > "
                          f"{CHILD_COMPACT_AFTER_CHARS}; reset at goal boundary "
                          f"(fresh chat, browser stays open)", flush=True)
        return "\n".join(reports), all_used

    async def _finish_pending_delegations(self, text: str) -> str:
        """If the orchestrator returned with current-turn tasks still
        pending/in_progress, wait for them to settle (bounded) and force a final
        answer built from their reports -- so the user never gets a premature
        'it's still running' reply. No-op when nothing is in flight."""
        def _pending() -> list:
            return [t for t in self.board.current_snapshot()
                    if t["status"] in ("pending", "in_progress")]

        if not _pending():
            return text
        deadline = time.time() + float(
            os.environ.get("SWARM_FINISH_TIMEOUT", "210"))
        while _pending() and time.time() < deadline:
            remaining = max(1.0, deadline - time.time())
            await self.board.wait_for_change(self.board.version(),
                                             timeout=min(15.0, remaining))
        # Whether everything settled or a straggler remains, make the
        # orchestrator answer NOW from the board (check_status shows done/failed
        # reports and any still-running task), rather than bailing to the user.
        nudge = Msg(
            name="user",
            content=[TextBlock(type="text", text=_FINISH_NUDGE)],
            role="user",
        )
        reply = await self.orchestrator.reply(nudge)
        self.rotator.touch(_ORCH_KEY)
        return (reply.get_text_content() if reply else "") or text

    def reset(self) -> None:
        """Wipe the whole conversation: board ledger, orchestrator memory, and
        every child's memory. Used by the UI's 'New session' so a fresh
        conversation starts clean. Browsers (real + mock) stay warm.

        Before wiping, archive this conversation's transcript and fire the
        background learner — so the swarm grows the memory wiki exactly like the
        single agent does."""
        self._archive_and_learn()
        self.board.clear()
        self._last_seen_version = self.board.version()
        self.orchestrator.state.context = []
        for agent in self._children.values():
            agent.state.context = []
        self.rotator.forget()

    def _archive_and_learn(self) -> None:
        """Persist the swarm conversation as a session and schedule fact
        extraction. Best-effort: never let it break a reset."""
        turns = list(self._turns)
        self._turns = []
        started = self._started_at
        self._started_at = None
        if not turns:
            return
        try:
            import sessions
            import memory_learn
            users = [t["text"] for t in turns if t.get("role") == "user"]
            title = (users[0][:60] + ("…" if users and len(users[0]) > 60 else "")) \
                if users else "Swarm session"
            summary = f"{len(turns)} turns (swarm). " + " | ".join(u[:60] for u in users[:3])
            rec = sessions.save_session(turns=turns, title=title,
                                        summary=summary[:300], started_at=started)
            try:
                import action_log
                action_log.rebind(action_log.current_session(), rec["id"])
            except Exception:
                pass
            memory_learn.learn_from_session_soon(rec["id"])
        except Exception as exc:
            print(f"[swarm] archive/learn skipped: {exc}")


# Singleton ref the orchestrator's tools close over (FunctionTool can't carry
# bound state out-of-band, so we route through this module-global). One swarm
# per process is the realistic case anyway.
_CURRENT: Swarm | None = None


def _current() -> Swarm:
    if _CURRENT is None:
        raise RuntimeError("swarm not initialised")
    return _CURRENT


# ===========================================================================
#  Orchestrator agent + its tools
# ===========================================================================
def _ok(obj) -> ToolChunk:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return ToolChunk(content=[TextBlock(type="text", text=text)])


def _err(msg: str) -> ToolChunk:
    return ToolChunk(content=[TextBlock(type="text", text=f"ERROR: {msg}")])


# --- the orchestrator's tools ----------------------------------------------
def list_specialists() -> ToolChunk:
    """List the specialist children you can delegate to: name, role, and one-
    line capability summary. Call this FIRST when you're not sure who should
    handle a request."""
    info = [
        {"name": s.name, "title": s.title, "capabilities": s.capabilities}
        for s in ROSTER
    ]
    return _ok(info)


# ---------------------------------------------------------------------------
#  Findings hand-off -- a teammate's REAL report, not the orchestrator's recap
# ---------------------------------------------------------------------------
# `_delegation_needs_earlier_output` used to assume that once a dependency had
# finished, "its report could then have been pasted in, which is exactly what we
# want the orchestrator to do". A real run showed it does not:
#
#   browser_agent  researched Muthoot (8 tool calls, goals verified on page)
#   orchestrator   delegate_task(child=scripter, description=<582 chars>)
#   scripter       fs_download(muthootfinance.com)  FAILED
#   scripter       fs_download(hdfcbank.com)        ok -> read 1.3 MB of HTML
#   scripter       write_document(...)
#
# The scripter never saw the research, so it went and re-gathered the sources
# itself, got nothing usable out of raw HTML, and fell back to generic
# knowledge. The resulting document mentioned Muthoot zero times and never made
# the individual / non-individual distinction the user actually asked for.
#
# The orchestrator is the wrong place to carry this: asking a model to copy a
# long report verbatim into an argument is asking it to paraphrase and drop
# detail, which is what it did. So the board -- which already holds the exact
# report text -- supplies it directly.
_FINDINGS_PER_REPORT = 1600     # one teammate's report
_FINDINGS_TOTAL = 4200          # all attached reports together

_FINDINGS_HEADER = (
    "FINDINGS FROM YOUR TEAMMATES (real reports from the shared board -- these "
    "are what was actually found. Use them. Do NOT re-gather this information "
    "yourself, and do not contradict them):"
)


# A report that says the work did NOT happen. The research child reported "I
# was unable to access consistent information from the web pages I visited",
# the orchestrator passed that as use_findings_from anyway, and the scripter
# built a spreadsheet out of it whose second row echoed its own column headers.
# Nothing downstream can tell a finding apart from an apology unless we look.
_FINDING_FAILED_RE = re.compile(
    r"\b(?:"
    r"i (?:was |am )?(?:un)?able to (?:not )?|i could ?n[o']t|i can ?n[o']t|"
    r"unable to (?:access|find|retrieve|gather|read|open|complete)|"
    r"could not (?:access|find|retrieve|gather|read|open|complete)|"
    r"no (?:results|data|information|content) (?:were |was )?(?:found|available)|"
    r"nothing was (?:found|created|returned)|"
    r"did not (?:find|return|produce)|"
    r"NOT DONE|CLAIMED BUT NOT VERIFIED|failed to"
    r")\b", re.I)


def finding_is_usable(report: str) -> bool:
    """Whether a teammate's report carries WORK, or only an account of failing.

    Conservative: a long report that mentions one setback among real content is
    still usable, so the marker has to be a large share of a short report or
    appear in its opening -- where a child puts the headline when the answer is
    "I couldn't".
    """
    t = " ".join((report or "").split())
    if not t:
        return False
    m = _FINDING_FAILED_RE.search(t)
    if not m:
        return True
    # In the first 300 characters it IS the headline; later on it is an aside.
    return m.start() > 300


def unusable_finding_ids(rows: list, task_ids: list) -> list:
    """Of the requested findings, the ones that report failure."""
    wanted = {r.get("id"): r for r in (rows or [])}
    return [i for i in (task_ids or [])
            if i in wanted and (wanted[i].get("report") or "").strip()
            and not finding_is_usable(wanted[i].get("report"))]


_TASK_MARKER = "\n\n---\n\nYOUR TASK:\n"


def task_text(description: str) -> str:
    """The delegation minus any findings briefing prepended to it.

    set_context(goal=rec.description) was reading the whole string, so once a
    briefing was attached the agent's GOAL line -- re-sent on every single step
    -- was the briefing's first 240 characters and the goal never appeared.
    """
    d = description or ""
    i = d.find(_TASK_MARKER)
    return d[i + len(_TASK_MARKER):] if i != -1 else d


def _findings_block(rows: list, task_ids: list) -> str:
    """Render the named tasks' real reports as a briefing, or '' if none."""
    wanted = [r for r in (rows or []) if r.get("id") in set(task_ids or [])]
    parts, used = [], 0
    for r in wanted:
        text = (r.get("report") or "").strip()
        if not text:
            continue
        if not finding_is_usable(text):
            # Passed through, but labelled. Silently dropping it would leave
            # the child with no explanation for why its input is missing.
            text = ("⚠ THIS TEAMMATE DID NOT PRODUCE USABLE OUTPUT — do NOT "
                    "build on it, and do NOT invent what it should have "
                    "contained. Its report was:\n" + text)
        if len(text) > _FINDINGS_PER_REPORT:
            text = text[:_FINDINGS_PER_REPORT].rstrip() + " …[truncated]"
        chunk = f"[{r.get('child', '?')} · {r.get('id', '?')}]\n{text}"
        if used + len(chunk) > _FINDINGS_TOTAL:
            break
        parts.append(chunk)
        used += len(chunk)
    if not parts:
        return ""
    return _FINDINGS_HEADER + "\n\n" + "\n\n".join(parts)


# Task ids the orchestrator names inline. It writes "from task tsk_a4dda73ad3"
# in the description instead of passing use_findings_from, and the receiving
# child then tried get_task(tsk_a4dda73ad3) -- a TASK-BOARD tool on a different
# id space -- and failed. If the id is right there in the text, use it.
_TASK_ID_RE = re.compile(r"\btsk_[0-9a-f]{6,}\b", re.I)

# A file one of the children generated. Extensions only -- a bare word like
# "the spreadsheet" is too vague to resolve to a specific task.
_GENERATED_FILE_RE = re.compile(
    r"\b[\w.-]+\.(?:xlsx|xls|docx|doc|pdf|pptx|csv|md|txt)\b", re.I)


def _normalise_backref(text: str) -> str:
    """Lowercase, and treat 'browser_agent' the same as 'browser agent'.

    The marker list was written the human way ("the browser agent's") while the
    orchestrator writes the machine way ("the browser_agent's"), so a real
    delegation missed by exactly one character and no findings were attached.
    """
    return re.sub(r"[_\-]+", " ", (text or "").lower())


def _auto_finding_ids(description: str, rows: list) -> list:
    """Task ids this delegation leans on and that have already produced a report.

    Reuses the same back-reference detection the sequencing guard uses: when it
    names a source child that is NOT done we refuse the delegation, and when it
    IS done we now attach the real report instead of hoping it was pasted in.
    """
    # A generated filename in the description points at whoever produced it.
    # The real delegation said "create attributes based on the Excel ..." and
    # named Loan_Attributes.xlsx; no phrase in _BACKREF_MARKERS matched, so
    # nothing was attached and the child went looking for the file over HTTP.
    # Matching the FILENAME against the reports that mention it is both more
    # precise than phrase-matching and harder to phrase around.
    files = {f.lower() for f in _GENERATED_FILE_RE.findall(description or "")}
    if files:
        by_file = [r.get("id") for r in (rows or [])
                   if r.get("status") == "done"
                   and any(f in (r.get("report") or "").lower() for f in files)]
        if by_file:
            return by_file

    # An id written straight into the description wins: it is unambiguous.
    named = {m.lower() for m in _TASK_ID_RE.findall(description or "")}
    if named:
        hits = [r.get("id") for r in (rows or [])
                if (r.get("id") or "").lower() in named
                and (r.get("report") or "").strip()]
        if hits:
            return hits

    d = _normalise_backref(description)
    if not any(m in d for m in _ATTACH_MARKERS):
        return []
    source = ""
    for needle, child in _BACKREF_SOURCES:
        if needle in d:
            source = child
            break
    if not source:
        # The phrase is a back-reference ("use the findings above") but names
        # no child. It still means "whatever just finished", so fall back to
        # the most recent done task that actually produced something -- a
        # dangling reference is exactly how a delegation went out empty.
        usable = [r.get("id") for r in (rows or [])
                  if r.get("status") == "done"
                  and (r.get("report") or "").strip()
                  and finding_is_usable(r.get("report"))]
        return usable[-1:]
    return [r.get("id") for r in (rows or [])
            if r.get("child") == source and r.get("status") == "done"
            and (r.get("report") or "").strip()]


def delegate_task(child: str, description: str,
                  use_findings_from: list | None = None) -> ToolChunk:
    # NOTE: `list | None`, not `Optional[list]`. This module has
    # `from __future__ import annotations`, so every annotation is a STRING,
    # and AgentScope builds each tool's schema by having pydantic evaluate it.
    # "Optional[list]" needs `typing.Optional` in that evaluation namespace and
    # does not get it, which raises PydanticUserError and kills agent
    # construction entirely. "list | None" resolves from builtins alone.
    """Hand a piece of work to a specialist child. Returns IMMEDIATELY with a
    task_id; the child runs in the background. Call this several times in a
    row to fan work out in parallel.

    Args:
        child (str): The specialist's name (see list_specialists).
        description (str): A self-contained instruction for the child. It
            sees ONLY this -- it does not see the user's original message or
            your reasoning -- so include all the context it needs (which
            task / which person / what URL / etc).
        use_findings_from (list): Task ids whose REPORTS this child needs, e.g.
            ["tsk_96f44776bf"] to give the writer what the researcher found.
            The board attaches the real report text itself, so you do NOT have
            to copy it into the description -- just name the task. When your
            description clearly refers to an earlier child's work this is
            attached automatically, but naming it is better.
    """
    sw = _current()
    if child not in sw._specs:
        return _err(f"no such specialist '{child}'. Available: "
                    f"{list(sw._specs.keys())}")
    # In-flight guard: refuse a SECOND delegation to a child that already has an
    # unfinished task THIS turn. Re-delegating an in-progress child spawns
    # overlapping runs that just queue on its per-child lock -- which is what
    # made the swarm look "stuck" and pushed the orchestrator into giving up.
    # The child is already working; the orchestrator should wait for it.
    busy = [t for t in sw.board.current_snapshot()
            if t["child"] == child and t["status"] in ("pending", "in_progress")]
    if busy:
        return _err(
            f"{child} is ALREADY working on task {busy[0]['id']} this turn "
            f"(status: {busy[0]['status']}). Do NOT delegate to it again -- that "
            "just queues a duplicate. Call wait_for_results() until it finishes, "
            "then check_status() to read its report.")
    # Sequencing guard: refuse a delegation that leans on another child's
    # output before that child has produced it. Refused at the point of the
    # call, not after the fact, because once the child runs it will invent the
    # missing content and the fabrication is already in the report.
    source = _delegation_needs_earlier_output(description,
                                              sw.board.current_snapshot())
    if source and source != child:
        return _err(_sequencing_nudge(child, source))
    # Attach the real reports this child needs. Explicit ids win; otherwise
    # fall back to whatever the back-reference detector spotted. Folded into the
    # description rather than carried beside it, so every downstream path --
    # goal planning, the per-goal message, resume, and check_status -- sees
    # exactly what the child was told.
    rows = sw.board.current_snapshot() + sw.board.history_snapshot()
    ids = [i for i in (use_findings_from or []) if i]
    if not ids:
        ids = _auto_finding_ids(description, rows)
    # The browser child does not get a teammate briefing. A report is prose
    # about work; what a form needs is values. Prepending several kilobytes of
    # another child's narrative pushed the actual instruction down the page,
    # and -- because set_context() reads the same string -- turned the
    # ORIENTATION header's GOAL line into "FINDINGS FROM YOUR TEAMMATES (real
    # reports from the shared board...". The orchestrator writes the values it
    # wants created straight into the description instead.
    if child == "browser_agent":
        ids = []
    bad = unusable_finding_ids(rows, ids)
    if bad and len(bad) == len(ids):
        # Every source this delegation leans on reported failure. Handing it
        # over anyway is how a research failure became a spreadsheet of column
        # headers and then an empty set of attributes.
        return json.dumps({
            "ok": False,
            "error": "the findings you asked for report FAILURE, not results",
            "unusable": bad,
            "hint": ("Do not build on them. Either re-delegate the source task "
                     "with clearer instructions, or write the values into this "
                     "description yourself. Never ask a child to invent them."),
        })
    block = _findings_block(rows, ids)
    if block:
        description = f"{block}{_TASK_MARKER}{description}"
        print(f"[swarm] attached {len(ids)} teammate report(s) to {child}'s "
              f"delegation ({len(block)} chars)", flush=True)

    rec = sw.board.new_task(child, description)
    sw.board.mark(rec.id, "in_progress")
    agent_events.publish("delegation", "orchestrator", task_id=rec.id,
                         child=child, description=description)
    # A loop is always running here: this sync tool is invoked inline inside
    # the orchestrator's async reply. The child runs when the orchestrator
    # next awaits (e.g. wait_for_results).
    loop = asyncio.get_running_loop()
    loop.create_task(sw._run_specialist(rec))
    return _ok({"task_id": rec.id, "delegated_to": child,
                "status": "in_progress",
                "hint": "call wait_for_results() to block until any child "
                        "finishes, then check_status() to read reports"})


# How much of a past child's report the running ledger carries. The full text
# stays on the board; this only bounds what is re-sent to the brain.
_LEDGER_REPORT_CAP = 400

# ---------------------------------------------------------------------------
#  Goal segmentation for long browser runs
# ---------------------------------------------------------------------------
# A multi-outcome web task ("create 3 categories and 6 attributes, 2 in each")
# is ~9 outcomes x ~8 steps x ~2.5k chars of snapshot ≈ 180k chars. It does not
# fail because the agent can't do it -- it just proved it can -- it fails
# because the context dies around outcome 2.
#
# So: ask the child ONCE for a plan of OUTCOMES, then hand it back one goal at a
# time. Goal boundaries are the browser equivalent of the orchestrator's turn
# boundary: the one place it is safe to drop accumulated snapshots, because the
# agent re-orients with a single snapshot (the RESUME case its persona already
# describes). The BROWSER IS NEVER CLOSED between goals -- same Chrome, same
# login, same page; only the conversation is compacted.
GOAL_PLANNING = os.environ.get("SWARM_GOAL_PLANNING", "1") not in ("0", "", "false")
# Only the browser child is segmented into goals (see _delegate).
_GOAL_PLANNING_CHILDREN = {"browser_agent"}
# Don't pay a planning round-trip on a one-liner like "check site X".
GOAL_PLAN_MIN_CHARS = int(os.environ.get("SWARM_GOAL_PLAN_MIN_CHARS", "120"))
GOAL_MAX = int(os.environ.get("SWARM_GOAL_MAX", "12"))

_PLAN_PROMPT = (
    "PLANNING STEP — do NOT touch the browser yet and do NOT call any tool.\n\n"
    "Break the task below into the list of OUTCOMES that must exist when it is "
    "done. Outcomes, not UI steps: name the thing being created or found, never "
    "'click the dropdown' (you have not seen the page yet, so any step-level "
    "plan would be wrong).\n"
    "Put each dependency in order — a container before the things inside it.\n"
    "If the task has only ONE outcome, return just that one.\n\n"
    "Reply with ONLY a numbered list, one outcome per line, nothing else:\n"
    "1. <outcome>\n2. <outcome>\n\n"
    "TASK:\n{task}"
)

_PLAN_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.{2,160}?)\s*$")


def parse_plan(text: str, limit: int = GOAL_MAX) -> list[str]:
    """Pull the numbered outcomes out of the planning reply."""
    goals: list[str] = []
    for line in (text or "").splitlines():
        m = _PLAN_LINE_RE.match(line)
        if not m:
            continue
        g = m.group(1).strip().rstrip(".").strip()
        # Drop a restated header like "Outcomes:" or a lone bullet of prose.
        if g and g.lower() not in ("outcomes", "goals", "plan") and g not in goals:
            goals.append(g)
        if len(goals) >= limit:
            break
    return goals


# A "password ..." clause in the restated task is dead weight from goal 2 on --
# the login already happened and the message says so -- and a credential in
# what may be a first-contact message is exactly what tips the model into a
# soft refusal. Goal 1 still gets it verbatim (it may have to log in).
_CRED_RE = re.compile(
    r"(?i)\b(pass(?:word|wd)?|pwd|pw)\b\s*(?:is|=|:)?\s*"
    r"[\"']?(\S+?)[\"']?(?=[\s,.;]|$)")


def _redact_creds(text: str) -> str:
    return _CRED_RE.sub(
        lambda m: f"{m.group(1)} [already entered - you are logged in]",
        text or "")


_RESULTS_BUDGET = 1200


def _results_block(results: list[str]) -> str:
    """The one-line outcome of each finished goal, newest kept first under a
    character budget.

    Only the PREVIOUS goal's first line used to travel, so by goal 3 the result
    of goal 1 was gone -- and goal 1 is usually where the ids, the naming rule
    and the form's quirks were discovered. These are one line each, sent once
    per goal, which is far cheaper than replaying their steps every turn.
    """
    if not results:
        return ""
    kept: list[str] = []
    used = 0
    for r in reversed(results):
        if used + len(r) > _RESULTS_BUDGET and kept:
            break
        kept.append(r)
        used += len(r)
    kept.reverse()
    return "\n".join("  " + r for r in kept)


def _goal_message(task: str, goals: list[str], done: list[bool], idx: int,
                  carry: str = "", results: list[str] | None = None) -> str:
    """The per-goal delegation: full task + plan + status + THIS goal only."""
    lines = []
    for i, g in enumerate(goals):
        mark = "✓" if done[i] else ("▶" if i == idx else "☐")
        lines.append(f"  {mark} {i + 1}. {g}")
    plan = "\n".join(lines)
    resume = (
        "A browser is ALREADY OPEN from the previous goal — same window, same "
        "login. Do NOT browse_open the app again and do NOT browse_close. Take "
        "ONE browse_snapshot to see where you are, then continue.\n"
        if idx > 0 else ""
    )
    block = _results_block(results or [])
    notes = (f"\nRESULTS OF THE GOALS ALREADY DONE:\n{block}\n" if block
             else (f"\nNOTES FROM THE PREVIOUS GOAL:\n{carry}\n" if carry
                   else ""))
    task = task if idx == 0 else _redact_creds(task)
    return (
        f"ORIGINAL TASK (for context — do NOT do all of it now):\n{task}\n\n"
        f"PLAN:\n{plan}\n{notes}\n"
        f"YOUR ONLY JOB THIS TURN IS GOAL {idx + 1} OF {len(goals)}:\n"
        f"  {goals[idx]}\n\n"
        f"{resume}"
        "Do ONLY that goal. Do not start any other goal — the system will hand "
        "you the next one. When it is done, reply with a one-line report of what "
        "you created and how (any quirk the next goal should know about, e.g. a "
        "dropdown that needs typing into its search box). Do NOT close the "
        "browser: more goals follow."
    )


# A child that could not finish for want of data it was never given. The
# browser child asked, in prose, for the attribute values -- which is what the
# reminder told it to do -- and the run ended, because a child's report is the
# only channel back and nobody re-read it as a request. The prompt now tells it
# to write "BLOCKED: need ...", and this turns that line into an instruction on
# the board, so acting on it does not depend on the orchestrator noticing.
# NOT anchored to the start of a line. A segmented run's report reads
# "goal 1/2 (Gather data ...): BLOCKED: need access past the security check",
# so the marker sits mid-line and the ^-anchored version matched no real one.
_BLOCKED_RE = re.compile(
    r"\bBLOCKED\s*[:\-]\s*((?:need|site)?[^\n]{4,300})", re.I)

# The shapes a child produces when it is asking rather than reporting. Kept
# narrow: an ordinary report that happens to contain "provide" must not trip it.
_ASKING_RE = re.compile(
    r"\b(?:please provide|please supply|please share|could you provide|"
    r"can you provide|provide the .{0,40}(?:specification|values|details|"
    r"contents|list)|awaiting (?:your |the )?(?:input|response|details))\b",
    re.I)


# A wall, not a gap. Re-delegating into one of these is pure waste: the run
# that prompted this hit a security check on nerdwallet, then bankrate, then
# Google, because every blocker looked like a request for data.
_SITE_WALL_RE = re.compile(
    r"\b(?:security (?:check|verification)|captcha|are you a human|"
    r"bot (?:check|detection)|cloudflare|access denied|forbidden|403|429|"
    r"rate.?limit|paywall|subscri\w* required|sign in to (?:continue|view)|"
    r"log ?in (?:wall|required)|blocked by|refused to connect|"
    r"(?:page|site) (?:did ?n[o']t|would ?n[o']t|failed to) load|timed out"
    r")\b", re.I)


def blocked_request(report: str) -> tuple:
    """(kind, what) for a blocked child, or ("", "") when it is not blocked.

    kind is "site" for something no amount of re-delegation can change, and
    "need" for data a teammate or the user can hand over.
    """
    t = (report or "").strip()
    if not t:
        return ("", "")
    what = ""
    m = _BLOCKED_RE.search(t)
    if m:
        what = " ".join(m.group(1).split())[:300]
    else:
        # It asked in prose instead of using the line. Same situation, and the
        # run dies just as quietly, so treat it the same and quote the sentence.
        m = _ASKING_RE.search(t)
        if not m:
            return ("", "")
        start = t.rfind(".", 0, m.start()) + 1
        end = t.find(".", m.end())
        what = " ".join(t[start:end if end != -1 else len(t)].split())[:300]
    # The explicit "BLOCKED: site ..." form, or the wall described in prose.
    kind = "site" if (what.lower().startswith("site")
                      or _SITE_WALL_RE.search(what)) else "need"
    if what.lower().startswith(("site ", "site:")):
        what = what[5:].strip() or what
    elif what.lower().startswith("need "):
        what = what[5:].strip() or what
    return (kind, what)


def _row(rec: dict, *, with_report: bool = True,
         report_cap: int | None = None) -> dict:
    """One status-board row, trimmed for the brain.

    The board's to_dict() is a blanket asdict(): it echoes the FULL description
    the orchestrator itself just wrote, the full report, and raw epoch floats --
    and wait_for_results returns that on EVERY poll. A single turn that polled
    three times re-sent the same draft three times (~5.4k chars of tool results
    for one email). The orchestrator never needs its own description read back,
    and an unfinished task has no report worth sending.
    """
    out = {
        "id": rec.get("id"),
        "child": rec.get("child"),
        "status": rec.get("status"),
        "elapsed_s": rec.get("elapsed_s"),
        "tool_calls": rec.get("tool_calls"),
        # Ground truth for <truthful_reporting> -- always send it.
        "tools_used": rec.get("tools_used") or [],
    }
    if rec.get("error"):
        out["error"] = rec["error"]
    if rec.get("findings"):
        out["findings"] = rec["findings"]
    # A browser child driving a long web flow looked identical to a hang from
    # the board (just "in_progress"), so the orchestrator bailed and told the
    # user the system was stuck. Show it the child is genuinely moving, and say
    # plainly what to do -- in the tool result, which cannot decay like the
    # persona can. Only for a live browser run; everything else is unaffected.
    if rec.get("status") == "in_progress":
        prog, goals_left, goals_total = "", 0, 0
        try:
            import browser_supervisor as sup
            prog = sup.progress_line()
            ctx = sup.context()
            if ctx.active and ctx.goal_total:
                goals_total = ctx.goal_total
                goals_left = max(0, ctx.goal_total - ctx.goal_index)
        except Exception:
            pass
        if prog:
            out["progress"] = prog
            more = (f" It is working through a {goals_total}-goal plan with "
                    f"{goals_left} still to go after this one, so this WILL "
                    f"take several more minutes." if goals_total else "")
            out["hint"] = ("STILL WORKING and making progress — call "
                           "wait_for_results() again. Do NOT give up, do NOT "
                           "re-delegate, and do NOT tell the user it is stuck."
                           + more)
    # A report only exists once the child is finished; sending "" on every poll
    # is pure noise.
    if with_report and rec.get("report"):
        report = rec["report"]
        kind, need = blocked_request(report)
        if need:
            # Before truncation: the request is usually at the END of the
            # report, which is exactly what a cap removes.
            out["blocked_on"] = need
            if kind == "site":
                # Telling it to re-delegate here is what sent three identical
                # research tasks into the same bot check.
                out["hint"] = (
                    f"{rec.get('child', 'this child')} was stopped by the SITE: "
                    f"{need}. Nothing you write changes that, so do NOT send it "
                    f"back for the same thing. Supply the values yourself in "
                    f"the next delegation, or tell the user that source is "
                    f"unreachable and what you did instead.")
            else:
                out["hint"] = (
                    f"{rec.get('child', 'this child')} finished what it could "
                    f"but is MISSING: {need}. Do NOT mark this done and do NOT "
                    f"move on. Re-delegate the same task to the same child with "
                    f"those values written into the description.")
        if report_cap and len(report) > report_cap:
            report = (report[:report_cap].rstrip()
                      + f"… [truncated; {len(report)} chars total — "
                        f"call check_status() for the full report]")
        out["report"] = report
    return out


def _rows(recs: list[dict], **kw) -> list[dict]:
    return [_row(r, **kw) for r in recs]


def check_status() -> ToolChunk:
    """Your team's status board. Returns:
      * 'current' -- tasks you delegated THIS turn, with status (pending /
        in_progress / done / failed), the child's report (if done), elapsed
        seconds, tool-call count.
      * 'earlier' -- the RUNNING LEDGER of everything your team did earlier in
        THIS conversation, including each child's full report. THIS is how you
        recall past work: if the user asks 'what email did you prepare', 'what
        did the researcher find', 'what task id was that', look here (or in
        your own memory of their reports) and answer -- do NOT say you have
        nothing without checking.
    Free to call -- it costs no child LLM time.
    """
    sw = _current()
    current = sw.board.current_snapshot()
    earlier = sw.board.history_snapshot()
    sw._last_seen_version = sw.board.version()
    return _ok({
        # check_status is the DEEP read: full reports for this turn's work.
        "current": _rows(current),
        # The ledger grows for the whole conversation, so cap each past report.
        "earlier": _rows(earlier, report_cap=_LEDGER_REPORT_CAP),
        "summary": {
            "current_total": len(current),
            "pending": sum(1 for t in current if t["status"] == "pending"),
            "in_progress": sum(1 for t in current if t["status"] == "in_progress"),
            "done": sum(1 for t in current if t["status"] == "done"),
            "failed": sum(1 for t in current if t["status"] == "failed"),
            "earlier_total": len(earlier),
        },
    })


async def wait_for_results(timeout_s: int = 60) -> ToolChunk:
    """BLOCK until at least one delegated task transitions (typically:
    finishes). Use this whenever you have work in flight rather than
    busy-looping on check_status. Returns the latest status snapshot.

    Args:
        timeout_s (int): Max seconds to wait. Reasonable default 60 (single
            tool call on the mock takes ~10s). Pass a bigger number for
            slow browser tasks; pass a smaller one if you want to interleave
            more orchestration steps.
    """
    sw = _current()
    new_version = await sw.board.wait_for_change(
        since_version=sw._last_seen_version, timeout=float(timeout_s)
    )
    if new_version < 0:
        snap = sw.board.current_snapshot()
        return _ok({
            "timed_out": True,
            # Still waiting: status only. Nothing has finished, so there is no
            # new report to send -- re-sending the last one wastes the window.
            "current": _rows(snap, with_report=False),
            "hint": "no child finished within the timeout; you can wait "
                    "again, or check_status() and decide whether to give up "
                    "on the slowest one",
        })
    sw._last_seen_version = new_version
    snap = sw.board.current_snapshot()
    return _ok({
        # Something transitioned: carry reports for the tasks that FINISHED,
        # and status-only rows for the ones still running.
        "current": [
            _row(r, with_report=r.get("status") in ("done", "failed"))
            for r in snap
        ],
        "next": "a task just changed. Re-read the user's request: which piece "
                "was waiting on this report? Delegate it now with the real "
                "values. Do not answer the user while any piece they asked for "
                "has no task on the board.",
        "summary": {
            "current_total": len(snap),
            "pending": sum(1 for t in snap if t["status"] == "pending"),
            "in_progress": sum(1 for t in snap if t["status"] == "in_progress"),
            "done": sum(1 for t in snap if t["status"] == "done"),
            "failed": sum(1 for t in snap if t["status"] == "failed"),
        },
    })


_ORCHESTRATOR_TOOL_FUNCS = [
    list_specialists, delegate_task, check_status, wait_for_results,
]


_ORCH_MEMORY_NOTE = (
    "\n\nMEMORY: Some messages carry trusted background you already know — a "
    "'WHAT YOU KNOW ABOUT THE USER' digest, 'LIVE RIGHT NOW' open threads, and/or "
    "a 'RELEVANT MEMORY' block about what's being discussed. Use them to route "
    "and to word your reply naturally; never recite them back verbatim. CRUCIAL: "
    "a child sees ONLY your delegation description, not this memory — so when a "
    "fact matters to the work (a person's email, a project's status, a stated "
    "preference), copy that specific fact INTO the delegation description."
)


def _build_orchestrator(swarm: Swarm, profile: str | None = None) -> Agent:
    """The orchestrator gets ONLY the four delegation tools, never the
    domain ones. By construction it can't accidentally do specialist work
    itself."""
    tools = [FunctionTool(f, is_read_only=False)
             for f in _ORCHESTRATOR_TOOL_FUNCS]
    toolkit = Toolkit(tools=tools)
    state = AgentState()
    state.permission_context.mode = PermissionMode.BYPASS
    return Agent(
        name="hermes-orchestrator",
        system_prompt=(ORCHESTRATOR_PERSONA
                       if config.is_mock_backend(profile)
                       else ORCHESTRATOR_PERSONA_NATIVE) + _ORCH_MEMORY_NOTE,
        model=_build_model(ORCHESTRATOR_SESSION_ID, profile),
        toolkit=toolkit,
        state=state,
        middlewares=[agent_events.StreamEmitter("orchestrator")],
        # 30 iterations is enough for: list -> delegate xN -> wait -> check
        # -> maybe re-delegate -> wait -> final.
        react_config=ReActConfig(max_iters=30),
    )


# ===========================================================================
#  Utility: count tool_use blocks added to a child's context during one reply
# ===========================================================================
# Human labels for the tools that matter most in the footer; anything not
# here is shown by its raw name (still unambiguous).
_TOOL_LABELS = {
    "send_email": "email SENT",
    "create_task": "task created",
    "create_group": "group created",
    "create_subgroup": "subgroup created",
    "update_task": "task updated",
    "move_task": "task moved",
    "delete_task": "task deleted",
    "add_note": "note added",
    "add_reminder": "reminder set",
    "schedule_reminder": "reminder scheduled",
    "add_follow_up": "follow-up logged",
    "complete_follow_up": "follow-up closed",
}


def _ground_truth_footer(board) -> str:
    """A deterministic, NON-LLM one-liner appended to the orchestrator's reply,
    built straight from each child's tools_used. This is the user's safety net:
    even if the orchestrator's prose lies ('I sent the email'), this footer
    states what actually executed ('comms_agent: drafted, NOT sent'). Empty
    when nothing was delegated this turn (e.g. chit-chat / recall)."""
    cur = board.current_snapshot()
    if not cur:
        return ""
    parts: list[str] = []
    for t in cur:
        child = t["child"]
        if t["status"] == "failed":
            parts.append(f"{child}: FAILED ({(t.get('error') or '')[:50]})")
            continue
        if t["status"] != "done":
            parts.append(f"{child}: {t['status']}")
            continue
        used = t.get("tools_used") or []
        if not used:
            # Nothing executed -> it only produced text. For comms that means a
            # draft was staged, not sent; for others, advice/analysis only.
            parts.append(f"{child}: drafted/advised only - nothing sent or saved"
                         if child == "comms_agent"
                         else f"{child}: no actions taken (text only)")
            continue
        seen: list[str] = []
        for u in used:
            lbl = _TOOL_LABELS.get(u, u)
            if lbl not in seen:
                seen.append(lbl)
        parts.append(f"{child}: " + ", ".join(seen))
    return "[verified actions this turn] " + "  |  ".join(parts)


# Sent to a child that claimed it sent an email without actually calling the
# send tool — forces the real tool call (or an honest correction).
_SEND_NUDGE = (
    "STOP. The system records which tools you actually call, and you did NOT "
    "call send_email — so NOTHING was sent, even though your last reply claimed "
    "it was. If you were asked to send the email, call the send_email tool NOW "
    "with the exact recipient, subject, and body of the draft (do not reply "
    "with text — emit the tool call). If you were only asked to draft, correct "
    "yourself: reply 'STATUS: DRAFT - NOT SENT' and do not claim it was sent."
)

# Phrases that indicate a child is CLAIMING an email went out.
_SENT_CLAIM_MARKERS = (
    "status: sent", "has been sent", "sent successfully", "email sent",
    "i have sent", "i've sent", "have sent the email", "message sent",
)


def _claims_sent_without_sending(report: str, tools_used: list[str]) -> bool:
    """True when a child's report claims an email was sent but it never called
    send_email. This is the exact hallucination the mock brain produces; the
    runner uses it to force a real send (or an honest correction)."""
    if "send_email" in tools_used:
        return False
    r = (report or "").lower()
    return any(m in r for m in _SENT_CLAIM_MARKERS)


# Phrases that indicate a child is blaming a MISSING CAPABILITY for not acting.
# ChatGPT-web drifts back into being ChatGPT and reaches for its own ecosystem:
# observed live reporting "the Gmail connector is unavailable in this session"
# and offering to install "Superhuman Mail" -- neither of which exists here. It
# has send_email the whole time and simply never called it.
_FALSE_BLOCKER_MARKERS = (
    "connector is unavailable", "connector is not available",
    "no email connector", "not connected", "once connected",
    "unavailable in this session", "not available in this session",
    "unavailable in this run", "not available in this run",
    "unavailable in this turn", "not available in this turn",
    # "the runtime did not expose /api/reports" -- a capability denial worded
    # as a fact about the runtime, which matched none of the markers above.
    "did not expose", "does not expose", "didn't expose", "doesn't expose",
    "not exposed", "runtime did not", "not wired up", "not provided by the",
    "tool is currently unavailable", "isn't available in this session",
    "install", "plugin", "extension", "integration is",
)


def _claims_blocked_without_trying(report: str, tools_used: list[str]) -> bool:
    """True when a child says it was BLOCKED by a missing capability while its
    tools_used is empty -- i.e. it never actually tried.

    This is the mirror image of _claims_sent_without_sending: that one catches
    claiming success without acting, this one catches claiming impossibility
    without acting. The honest-but-wrong case (a truthful 'NOT SENT' with a
    fabricated reason) slipped through every existing guard.

    A child that DID call tools is trusted -- a real tool error is a real
    blocker and must be reported, not nudged away.
    """
    if tools_used:
        return False
    r = (report or "").lower()
    return any(m in r for m in _FALSE_BLOCKER_MARKERS)


_FALSE_BLOCKER_NUDGE = (
    "STOP. You made ZERO tool calls, so you did not actually try. There are no "
    "connectors, plugins, apps or integrations in this system and nothing needs "
    "installing or connecting -- you have real tools and they are available to "
    "you right now. Do not suggest installing anything. If you were asked to "
    "act, EMIT THE TOOL CALL now (e.g. send_email with the exact recipient, "
    "subject and body). If the tool then returns an error, report that error "
    "verbatim. If you genuinely cannot proceed, say exactly what you are "
    "missing from the USER -- not from the system."
)


# --- reporting a FAILURE it never attempted ---------------------------------
# The mirror of _claims_blocked_without_trying, which only fires on capability
# DENIAL wording ("connector unavailable", "not exposed"). A child that simply
# writes "STATUS: SEND FAILED" -- no excuse, no explanation -- with an empty
# tools_used slipped through every guard, and the board recorded a failure that
# never happened. Observed on comms: it announced SEND FAILED having made zero
# calls, so nothing was tried and nothing was learned.
_FAILURE_CLAIM_MARKERS = (
    "send failed", "status: failed", "failed to send", "could not send",
    "unable to send", "email failed", "sending failed", "not sent",
    "delivery failed", "could not complete", "unable to complete",
    "was unsuccessful", "did not succeed",
)


def _claims_failed_without_calling(report: str, tools_used: list[str]) -> bool:
    """True when a child reports a FAILURE while its tools_used is empty.

    A failure is only real if something was attempted. With zero tool calls
    there is no error to report -- there is only an untried task, and saying
    otherwise puts a false negative on the board that the orchestrator then
    relays to the user as fact.

    A child that DID call tools is trusted absolutely: a genuine tool error is
    exactly what we want reported, verbatim and un-nudged.
    """
    if tools_used:
        return False
    r = (report or "").lower()
    # "NOT SENT" is legitimate when the child was asked to DRAFT rather than
    # send -- the comms persona uses "STATUS: DRAFT - NOT SENT" by design.
    if "draft" in r:
        return False
    return any(m in r for m in _FAILURE_CLAIM_MARKERS)


_FAILED_WITHOUT_TRYING_NUDGE = (
    "STOP. You reported a FAILURE but you made ZERO tool calls -- so nothing "
    "was attempted and there is no error to report. A failure is only real "
    "once a tool has actually returned one. EMIT THE TOOL CALL now. If it "
    "returns an error, quote that error verbatim and report SEND FAILED with "
    "it. If you are missing something you need from the USER (an address, a "
    "file, an approval), say exactly what -- but do not report a failure you "
    "never tried."
)


# Sent to the scripter when it claims a file is missing without ever calling a
# file tool — the weak brain assuming files are attached to the chat instead of
# reading the uploads folder. Forces it to actually look.
_FILE_NUDGE = (
    "STOP. The file IS available to you — it lives in the uploads FOLDER that "
    "your tools read; it is NOT attached to this chat, so 'please upload it' is "
    "wrong. You did NOT call any file tool. Call list_uploaded_files NOW to see "
    "the real filenames, then excel_info / excel_peek / excel_filter (for "
    "spreadsheets) or doc_read / doc_search (for documents) to actually answer. "
    "Emit the tool call — do not reply with text claiming you can't find it."
)

# Phrases that indicate a child is CLAIMING a file is absent / needs uploading.
_NO_FILE_MARKERS = (
    "couldn't find", "could not find", "couldn't locate", "could not locate",
    "can't find", "cannot find", "unable to find", "unable to locate",
    "no file", "not find any", "please upload", "no uploaded file",
    "isn't uploaded", "is not uploaded", "wasn't uploaded", "no such file",
    "doesn't appear", "does not appear",
)

# The file-reading tools — calling any of these proves the scripter actually
# looked (so a "can't find it" after one of these is an honest result, not a
# hallucination).
_FILE_TOOL_NAMES = {
    "list_uploaded_files", "excel_info", "excel_peek", "excel_search",
    "excel_filter", "excel_query", "excel_aggregate", "doc_read", "doc_search",
}


# Sent to the ORCHESTRATOR when it tries to end its turn while its own delegated
# tasks are still finishing. Forces it to read the now-available reports and give
# the user the real answer instead of an "it's still running / restart" punt.
_FINISH_NUDGE = (
    "Your delegated specialist task(s) for this turn have now finished (or are "
    "recorded on the board). Call check_status() to read their FINAL reports, "
    "then give the user ONE complete answer built from the actual results and "
    "the verified tool outputs -- include the specific numbers/findings. Do NOT "
    "say the work is 'still running', do NOT ask the user to restart or re-run, "
    "and do NOT re-delegate -- the results are ready; read them and answer now. "
    "If a task genuinely failed, say what failed and give the parts that "
    "succeeded."
)


def _claims_no_file_without_reading(child: str, report: str,
                                    tools_used: list[str]) -> bool:
    """True when the scripter claims it can't find / needs an upload but never
    actually called a file tool — the weak brain assuming the file isn't there
    instead of looking. The runner forces it to read first."""
    if child != "scripter":
        return False
    if any(t in _FILE_TOOL_NAMES for t in tools_used):
        return False
    r = (report or "").lower()
    return any(m in r for m in _NO_FILE_MARKERS)


# --- orchestrator fanning out work that is actually sequential --------------
# A child sees ONLY its own description. So a delegation that says "using the
# research output from the browser specialist" or "attach the Word doc you
# created", sent in the SAME turn as the delegation that produces that output,
# is referring to something the child will never receive. Observed twice: the
# scripter wrote two finance documents from its own knowledge because the
# research never reached it, and comms was told to attach files that did not
# exist yet. The prompt now teaches sequencing; this catches the relapse.
# NARROW, and it stays narrow: this tuple drives _delegation_needs_earlier_output,
# which REFUSES a delegation. Widening it (with "as identified", "as researched"
# and friends) put two ordinary BRE delegations on HOLD -- "include attributes
# for transaction logs and commercial flow, as identified in the report" matched
# "as identified", resolved "report" to scripter, and the work was blocked.
#
# A phrase list that ATTACHES a report is safe when it over-fires; the same list
# driving a VETO is not. They are separate lists now, and _ATTACH_MARKERS below
# is the one allowed to be generous.
_BACKREF_MARKERS = (
    "using the research", "from the research", "the research output",
    "the research from", "research provided by", "research gathered by",
    "the browser specialist", "the browser agent's", "the researcher's",
    "the scripter's", "the output from the", "the report from the",
    "the file you created", "the files you created", "the document you created",
    "the attached word", "the attached excel", "the attached document",
    "as attachments", "the generated files", "the generated document",
    "the previously created", "produced by the",
)

# WIDE, and only ever used to decide what to attach. Over-firing here costs a
# teammate's report riding along in a description; over-firing in the veto costs
# the task.
_ATTACH_MARKERS = _BACKREF_MARKERS + (
    "as researched", "you researched", "that you found", "you found earlier",
    "as found", "from your research", "based on the research",
    "the research you", "researched above", "identified earlier",
    "from the findings", "the findings above", "as identified",
)
# Which child the reference points at, so we can check the right dependency.
_BACKREF_SOURCES = (
    ("browser", "browser_agent"), ("research", "browser_agent"),
    ("scripter", "scripter"), ("word", "scripter"), ("excel", "scripter"),
    ("document", "scripter"), ("file", "scripter"), ("report", "scripter"),
    ("attach", "scripter"), ("generated", "scripter"),
)


def _delegation_needs_earlier_output(description: str,
                                     board_rows: list) -> str:
    """Name the child whose output this delegation depends on but has not got.

    Returns "" when the delegation is self-contained, or when the dependency has
    already finished this turn (its report could then have been pasted in, which
    is exactly what we want the orchestrator to do).
    """
    d = _normalise_backref(description)
    if not any(m in d for m in _BACKREF_MARKERS):
        return ""
    source = ""
    for needle, child in _BACKREF_SOURCES:
        if needle in d:
            source = child
            break
    if not source:
        return ""
    for row in board_rows or []:
        if row.get("child") == source and row.get("status") == "done":
            return ""          # the report exists; assume it was pasted in
    return source


def _sequencing_nudge(child: str, source: str) -> str:
    return (
        f"HOLD. The delegation you just wrote for {child} refers to work from "
        f"{source} that {source} has not finished, and a child can only see the "
        f"description you give it -- it cannot read another child's report. As "
        f"written, {child} would have to invent the content.\n"
        f"Do it in order instead: delegate {source} first, wait_for_results, "
        f"then delegate {child} with use_findings_from=[\"<that task id>\"] -- "
        f"the board attaches {source}'s real report itself, so you do not have "
        f"to copy it. Emit the {source} delegation now."
    )


# --- scripter claiming a report it never wrote (the /mnt/data failure) ------
# The nastiest shape yet, because it does not LOOK like a failure. Told the
# report tools weren't there ("the runtime did not expose /api/reports"), the
# scripter fell back to ChatGPT's OWN code interpreter, generated real .docx /
# .xlsx files inside ChatGPT's sandbox, and reported their sandbox paths as the
# deliverables -- with download chips in the web UI to match. Nothing exists on
# this machine, so the files cannot be attached, emailed, or opened, and every
# existing guard passed it: it claimed success (so the blocked-guard missed it)
# and never said a file was missing (so the file-guard missed it).
_REPORT_TOOL_NAMES = {
    "write_document", "create_spreadsheet", "create_chart", "list_reports",
}
# A path only ChatGPT's sandbox produces. Seeing it in a report is proof the
# work happened somewhere this system cannot reach.
_FOREIGN_PATH_MARKERS = ("/mnt/data", "mnt/data", "sandbox:/", "/home/sandbox")
_MADE_REPORT_MARKERS = (
    "created both", "created the", "generated the", "generated both",
    "i created", "i generated", "i've created", "i have created",
    "deliverable", "the exact generated path", "downloads:",
    ".docx", ".xlsx", ".pdf", ".pptx",
)


def _claims_made_report_without_writing(child: str, report: str,
                                        tools_used: list[str]) -> bool:
    """True when the scripter says a document exists but called no report tool.

    Mirror of _claims_sent_without_sending: that one catches claiming an action
    happened without acting; this catches claiming an ARTIFACT exists without
    making it. A foreign sandbox path is decisive on its own -- no real run of
    our tools can produce one.
    """
    if child != "scripter":
        return False
    if any(t in _REPORT_TOOL_NAMES for t in (tools_used or [])):
        return False
    r = (report or "").lower()
    if any(m in r for m in _FOREIGN_PATH_MARKERS):
        return True
    return any(m in r for m in _MADE_REPORT_MARKERS)


_REPORT_NUDGE = (
    "STOP. You did NOT call any report tool, so no file exists. Whatever you "
    "produced by thinking or by running code in your own environment is NOT in "
    "this system: paths like /mnt/data/... belong to a different sandbox, this "
    "software cannot read them, and they cannot be attached to an email. Do "
    "not report them as deliverables.\n"
    "The report tools ARE available to you right now. Emitting the tool call "
    "is always possible -- the software runs it and hands you the result. Call "
    "write_document(filename, markdown, fmt='docx') for the document and "
    "create_spreadsheet(filename, data) for the workbook NOW, then report the "
    "/api/reports/<filename> link each one returns. If a tool returns an "
    "error, report that error verbatim."
)


# --- browser child bailing with work left (the Niyam failure) ---------------
# Scoped to browser_agent ONLY -- NOT comms_agent, whose draft-and-wait is a
# DESIGNED pause (nudging it to "finish" could bypass the send-confirm gate).
_BROWSER_FINISH_CHILDREN = {"browser_agent"}

# Two failure shapes from the weak browser brain: (1) it hallucinates a tool
# (often its own specialist name, "browser_agent") and claims the tool isn't
# available, and (2) it does PART of a repetitive task then hands the rest back
# ("remaining attributes still need to be created", "continue creating ...").
_BROWSER_STUCK_MARKERS = (
    "not available in the current toolset", "tool is not available",
    "the tool named", "browser automation tool", "required browser",
    "cannot continue", "could not continue", "unable to continue",
    "cannot proceed", "could not proceed", "tool is unavailable",
    # the "I can't run my own tools" hallucination, more phrasings:
    "unable to execute browser", "cannot execute browser", "can't execute browser",
    "browser tools in this chat", "tools in this chat", "in this chat",
    "browser tools are not", "browser tools aren't", "no browser tools",
    "can't use the browser", "cannot use the browser",
    # Seen on the cold-start goals of a segmented run. "in this run" and
    # "browse_* controls" matched no marker above, so the refusal was
    # accepted as a finished goal.
    "in this run", "in this turn", "browse_* controls", "browse_ controls",
    "live browser controls", "browser controls are not",
    "can't execute goal", "cannot execute goal", "unable to execute goal",
)
_BROWSER_HANDOFF_MARKERS = (
    "still need to be created", "remaining attribute", "continue creating",
    "not yet created", "still need to create", "remaining items",
    "remaining records", "create the remaining", "yet to be created",
)

_BROWSER_CONTINUE_NUDGE = (
    "STOP — do NOT hand this back and do NOT report it as blocked. You ARE the "
    "browser agent and your browse_* tools ARE available right now. There is NO "
    "'browser_agent' or 'browser' tool — never call one; use browse_open / "
    "browse_snapshot / browse_batch / browse_repeat / browse_click_text "
    "directly. The browser is still open on this task. FIRST call browse_snapshot "
    "to see the current page, THEN finish EVERY remaining item yourself in this "
    "same run. For several similar records, do them in ONE shot with "
    "browse_repeat(template, items): put the whole create cycle in the template "
    "using browse_click_text for buttons and type_label / type_placeholder for "
    "fields (so it keeps working as the form rebuilds), and list ALL the "
    "remaining values in items. Only report back once every item is created AND "
    "you have verified them on the list page."
)


def _used_browser_tool(tools_used: list[str]) -> bool:
    """True if any browse_* tool appears — proof the child actually drove the
    browser (vs. only refusing/talking)."""
    return any(str(t).startswith("browse_") for t in (tools_used or []))


def _browser_bailed_with_work_left(child: str, report: str,
                                   tools_used: list[str]) -> bool:
    """True when browser_agent stops with work remaining — either claiming a tool
    is unavailable (a hallucination; its browse_* tools exist) or handing the
    rest of a repetitive job back to the orchestrator. The runner forces it to
    reorient (browse_snapshot) and finish the remaining items in the same run."""
    if child not in _BROWSER_FINISH_CHILDREN:
        return False
    r = (report or "").lower()
    return (any(m in r for m in _BROWSER_STUCK_MARKERS)
            or any(m in r for m in _BROWSER_HANDOFF_MARKERS))


# The orchestrator claiming it has no way to act. Always a hallucination: its
# four tools are in the prompt it was just given. Kept narrow -- these are
# capability-denial phrasings, not ordinary "I couldn't find it" answers --
# and the zero-delegations check below is what makes it safe.
_ORCH_REFUSAL_MARKERS = (
    "can't", "cannot", "can not", "unable to", "couldn't", "could not",
    "not available", "aren't available", "unavailable", "not accessible",
    "no access to", "don't have access", "do not have access",
    "not supported", "without the required", "i have no ",
)
# Length guard: a capability denial is a one-liner. A long answer that merely
# CONTAINS "not available" is almost always a real report being relayed, and
# must not be second-guessed.
_ORCH_REFUSAL_MAX_CHARS = int(
    os.environ.get("SWARM_ORCH_REFUSAL_MAX_CHARS", "600"))


def _orchestrator_refused_to_act(text: str, current_tasks: list) -> bool:
    """True when the orchestrator answered with a capability denial while
    delegating NOTHING this turn.

    The zero-delegations test is what keeps this honest: once it HAS delegated,
    a report of unavailability is a child's real finding being relayed (a site
    down, a record missing) and must be passed through untouched. Only a turn
    with no delegation at all can be this hallucination.

    Deliberately NOT keyed on "in this run" / "in this session" alone -- those
    appear in ordinary recaps ("here's what we did in this session"). The
    trigger is an inability phrase in a short, tool-less reply.
    """
    if current_tasks:
        return False
    t = (text or "").strip()
    if not t or len(t) > _ORCH_REFUSAL_MAX_CHARS:
        return False
    t = t.lower()
    return any(m in t for m in _ORCH_REFUSAL_MARKERS)


_ORCH_REFUSAL_NUDGE = (
    "STOP. That answer is wrong: you DO have the tools and they are available "
    "to you right now, in this very turn. Your four tools are "
    "list_specialists, delegate_task, check_status and wait_for_results, and "
    "they are listed in the tool block you were given. There is no "
    "'delegation interface' to enable, nothing to install, and nothing to "
    "connect.\n"
    "You made ZERO tool calls, so you did not actually try. Emit the "
    "delegation NOW as your entire reply:\n"
    'RIGHT: {"tool_calls":[{"name":"delegate_task","arguments":'
    '{"child":"browser_agent","description":"<the full task, verbatim>"}}]}\n'
    "Pick the child that fits: browser_agent for anything on a website, "
    "comms_agent for email or chat, scripter for files and reports, "
    "task_manager for the board. Then wait_for_results and answer from what "
    "comes back. Do not explain, do not apologise -- emit the JSON."
)


# Sent to the ORCHESTRATOR when it answered a file question itself by claiming
# the file is missing/unreadable instead of delegating to the scripter. The
# orchestrator has no file-reading ability, so such a claim is always a
# hallucination; this forces it to delegate.
_ORCH_FILE_NUDGE = (
    "You do NOT read files yourself and cannot know whether a file exists — only "
    "the scripter specialist can look in the uploads folder. You just claimed a "
    "file is missing or can't be accessed WITHOUT delegating to the scripter. "
    "That is wrong. Delegate NOW with delegate_task('scripter', '<name the exact "
    "file and what to extract>'), then wait_for_results and answer from the "
    "scripter's report. Never tell the user a file is missing or to re-upload it "
    "on your own authority."
)

# Keywords that mark the user's message as being about a file, so the
# orchestrator file-guard only fires on genuinely file-related turns (avoids
# false positives like "I couldn't find any open tasks").
_FILE_TOPIC_KEYWORDS = (
    "file", "spreadsheet", "excel", "csv", "workbook", "sheet", "document",
    "pdf", "word doc", "powerpoint", "upload", "attachment", ".xlsx", ".xls",
    ".csv", ".pdf", ".docx", ".pptx", ".txt",
)


def _looks_file_related(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(k in t for k in _FILE_TOPIC_KEYWORDS)


def _orchestrator_ducked_file(report: str) -> bool:
    """True when the orchestrator's reply claims a file is missing/unreadable —
    something it can't actually know, since only the scripter reads files."""
    r = (report or "").lower()
    return any(m in r for m in _NO_FILE_MARKERS)


# A tool call lands in the agent's context as a block whose ``type`` marks it.
# AgentScope's own block (ToolCallBlock) uses "tool_call"; Anthropic's wire
# format -- and the fake agents in our tests -- call it "tool_use". We accept
# BOTH. Matching only "tool_use" (the original code) never matched a real
# AgentScope run, so tools_used came back empty for every live delegation: the
# ground-truth footer then wrongly reported "no actions taken (text only)" even
# when the child really called create_group / send_email, and the comms
# claims-sent guard misfired. This set is the fix.
_TOOL_CALL_TYPES = {"tool_call", "tool_use"}


def _block_attr(block, key):
    """Read a block field whether the block is a pydantic object (``.attr``) or
    a plain dict (``["key"]``) -- a context reloaded from JSON can be either."""
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _tool_names_used(context: list, *, since: int) -> list[str]:
    """Return the NAMES of the tools the agent actually called since position
    ``since`` in its context. This is the orchestrator's ground truth: a
    child's prose report is just talk, but tools_used is what it really did.
    It's how the orchestrator distinguishes 'comms_agent drafted an email'
    (tools_used == []) from 'comms_agent SENT it' (tools_used contains
    'send_email')."""
    names: list[str] = []
    for msg in context[since:]:
        blocks = _block_attr(msg, "content")
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            if _block_attr(b, "type") in _TOOL_CALL_TYPES:
                nm = _block_attr(b, "name")
                if nm:
                    names.append(nm)
    return names


# ===========================================================================
#  Public factory + chat entry
# ===========================================================================
def build_swarm(profile: str | None = None) -> Swarm:
    """Build (or rebuild) the swarm singleton."""
    global _CURRENT
    _CURRENT = Swarm(profile=profile)
    return _CURRENT


def get_swarm(profile: str | None = None) -> Swarm:
    """Return the current swarm, building it on first call. Rebuilds if the
    LLM profile changes (so a switch from mock -> openai takes effect)."""
    global _CURRENT
    if _CURRENT is None or (profile and profile != _CURRENT.profile):
        return build_swarm(profile)
    return _CURRENT


# Per-request serialisation so two concurrent /api/agent/multi/chat calls
# don't race for the same orchestrator context.
_swarm_lock = asyncio.Lock()


async def server_swarm_chat(user_text: str, profile: str | None = None) -> str:
    async with _swarm_lock:
        sw = get_swarm(profile)
        return await sw.swarm_chat(user_text)


def status_snapshot() -> dict:
    """Live view of the board for the UI (any delegations still in flight,
    plus their reports). Returns ``{"running": bool, "tasks": [...]}``."""
    if _CURRENT is None:
        return {"running": False, "tasks": []}
    snap = _CURRENT.board.snapshot()
    running = any(t["status"] in ("pending", "in_progress") for t in snap)
    return {"running": running, "tasks": snap}


def reset_swarm() -> bool:
    """Clear the swarm's conversation memory (board ledger + orchestrator +
    children). No-op (returns False) if no swarm has been built yet."""
    if _CURRENT is None:
        return False
    _CURRENT.reset()
    return True
