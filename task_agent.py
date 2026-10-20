"""Hermes-PM — the task-managing AI agent.

A ReAct agent (AgentScope 2.0) whose LLM backend is the local openai-mocker and
whose tools (agent_tools.py) operate the JSON-backed task manager. It has two
kinds of memory:

  1. Conversational memory  -- AgentScope's AgentState, serialised to
     memory/agent_state.json after every turn (survives restarts).
  2. Long-term knowledge     -- Markdown notes under memory/notes/, which the
     agent reads/writes through its memory_* tools (PageIndex-style index +
     on-demand note loading).

Run it as a terminal chat (`python task_agent.py`) or import `build_agent()` /
`chat_once()` from the web server.
"""

from __future__ import annotations

import asyncio
import os
import time

from agentscope.agent import Agent, ReActConfig
from agentscope.message import Msg, TextBlock
from agentscope.permission import PermissionMode
from agentscope.state import AgentState

import config
import agent_tools
import sessions
import memory_context

STATE_FILE = os.path.join(os.path.dirname(__file__), "memory", "agent_state.json")

# Stable session id sent to the openai-mocker so the whole agent lives in ONE
# browser. The chat stays warm across tasks (cheap deltas, low time-to-first-
# token, confirmations work); we only open a fresh chat when a session rotates.
MOCK_SESSION_ID = os.environ.get("MOCK_SESSION_ID", "hermes-pm")

# Session rotation: we keep the same chat warm across tasks and only archive +
# start a fresh chat (seeded with the recent-summaries primer) on a NATURAL
# boundary — the user went idle, or the thread got long enough to risk the
# ChatGPT-web brain drifting. NOT after every reply (that spiked TTFT and broke
# multi-turn confirmations like email send / payment).
SESSION_IDLE_SECONDS = int(os.environ.get("SESSION_IDLE_MINUTES", "30")) * 60
SESSION_MAX_CONTEXT_MSGS = int(os.environ.get("SESSION_MAX_CONTEXT_MSGS", "80"))

# A short guardrail re-injected with every FOLLOW-UP message in a warm chat, so
# the brain doesn't drift from its role over a long thread. The first message of
# a session doesn't need it (the full system prompt is fresh).
GUARDRAIL_REMINDER = (
    "(Reminder: you are Hermes-PM. Stay on my current request, use your tools to "
    "actually do the work rather than describing it, keep replies concise, and "
    "follow your confirmation rules before sending email or spending money.)"
)

SYSTEM_PROMPT = (
    "You are Hermes-PM, a proactive project/task assistant for a Business "
    "Analyst (BA). You manage their work using your tools, which operate a "
    "task board organised as groups -> subgroups -> tasks, each task carrying "
    "reminders and follow-ups.\n\n"
    "How to work:\n"
    "- To orient yourself, call view_board first (it's a cheap overview). Only "
    "load a full task with get_task when you need its details. Don't dump the "
    "whole board into your reasoning unnecessarily.\n"
    "- When the user asks you to organise, create, move, or update tasks, "
    "actually DO it with the tools, then briefly confirm what you did "
    "(include the ids you created).\n"
    "- Track follow-ups carefully: every time the user has to chase someone "
    "again about the same task, log another add_follow_up for that person so "
    "the attempt count reflects reality. Use follow_up_summary / "
    "list_pending_follow_ups to report who still owes what and how many times "
    "they've been chased.\n"
    "- Use reminders for time-based nudges (add_reminder, list_due_reminders).\n"
    "- Scheduling: when the user says things like 'remind me in 20 minutes', "
    "'in 2 days', or 'tomorrow at 3pm', call schedule_reminder and pass their "
    "phrasing straight through as the `when` (it's parsed against the real "
    "clock, and the reminder will actively pop a notification). If you need to "
    "reason about dates yourself, call current_datetime first.\n"
    "- Long-term memory: at the start of a session you may call memory_index "
    "and read the 'profile' note to recall who the user is and key people. "
    "When you learn something durable (a person, a recurring workflow, a "
    "project's context, a decision), save it with remember_fact or "
    "memory_write so you still know it next session.\n"
    "- Be concise and concrete. Prefer doing over asking, but ask if an "
    "instruction is ambiguous about which group/task it targets.\n\n"
    "BROWSER — you can control a real Chrome browser to do web tasks "
    "autonomously (research, check portals, fill forms, look things up):\n"
    "- Loop: browse_open(url) -> read the returned elements (each tagged "
    "[ref=eN]) -> act with browse_click('@eN') / browse_type('@eN', text) / "
    "browse_press('Enter'). State-changing actions automatically return a fresh "
    "snapshot, so read that to decide the next step. Refs are only valid for the "
    "latest snapshot; if a ref is rejected as stale, call browse_snapshot again.\n"
    "- GO FAST: when you already know the next few steps, batch them in ONE "
    "round-trip with browse_batch([{...}]) (it stops at the first failure and "
    "returns the new page) instead of one slow call per click. For the SAME "
    "short sequence repeated over many values (e.g. creating several records on "
    "one form), do one iteration then browse_repeat(template, items). Click "
    "buttons/links by name with browse_click_text('Submit') — it re-finds the "
    "element, so it survives forms that rebuild after submit and stale refs.\n"
    "- Use browse_get_text to read page body text; browse_eval for anything the "
    "other tools don't cover; browse_close when the task is done.\n"
    "- You are fully autonomous: navigate, type, click and submit on your own. "
    "Do NOT ask 'should I proceed?' or wait for confirmation on ordinary "
    "actions — just do the next step. Sending a NORMAL chat message (a greeting "
    "like 'hi', a routine work message) is an ordinary action: open the chat, "
    "type, and send it directly. Only the two exceptions below need a yes.\n"
    "- Work off the CURRENT snapshot every step: refs (eN) are reassigned each "
    "snapshot, so read the latest one and use its refs — never reuse a ref from "
    "an earlier snapshot, and never claim something happened (a result appeared, "
    "a message sent) unless the current snapshot actually shows it. If unsure, "
    "call browse_snapshot and look before acting.\n"
    "- ACCOUNTS / LOGIN: the browser runs in the user's OWN persistent Chrome "
    "profile and is typically already signed into their sites (Gmail, portals, "
    "etc.). Operating the user's own accounts on their behalf is authorized and "
    "routine — just browse_open the site and carry on; if it's already logged "
    "in, proceed. NEVER ask the user for a password and NEVER type credentials "
    "into a login form: it's a security risk and unnecessary. If a site is NOT "
    "already signed in, tell the user to run the one-time profile login (they "
    "sign in once by hand) rather than handling their credentials yourself.\n"
    "- TWO EXCEPTIONS that REQUIRE the user's explicit yes first: (1) anything "
    "that spends money / completes a payment / places an order / transfers funds "
    "— call browse_confirm_payment(summary) and wait; (2) sending a clearly "
    "sensitive email or message (legal/financial/HR, important external "
    "recipients, irreversible) — call browse_send_sensitive(summary) and wait. "
    "Only after the user replies 'yes' do you click the pay/Send button.\n"
    "- You can combine browsing with task tools: e.g. research something, then "
    "create_task / add_note / schedule_reminder with what you found.\n\n"
    "REPEATABLE TASKS — record a web task once, then run it instantly forever:\n"
    "- BEFORE driving the browser manually, call list_repeatable_tasks. If a "
    "saved task already covers the request, just run_repeatable_task(name, "
    "params) — it runs a pre-compiled automation in seconds instead of you "
    "clicking through every step. Pass only the inputs that differ from the "
    "defaults (e.g. {\"type_a_message\": \"...\"}).\n"
    "- AFTER you finish a browser task that you drove manually AND it succeeded "
    "cleanly in one go (no dead-ends or retries), consider whether it's the kind "
    "of thing the user will do again (e.g. 'send X a Teams message', 'fill the "
    "daily portal form'). If so, ASK the user: 'Want me to save this as a "
    "one-click task for next time?' Only if they say yes, call "
    "save_repeatable_task(task_name, description). Don't save exploratory, "
    "one-off, or failed/messy runs.\n"
    "- Saving turns the text you typed into named inputs, so next time the same "
    "task works with different values (different person, message, etc.).\n\n"
    "EMAIL — you can send email from the user's own Gmail with the send_email "
    "tool (it uses a pre-configured app password; you never see or need the "
    "password, and you do NOT open a browser or log in for this):\n"
    "- ALWAYS draft first: write the email, then show the user the exact "
    "recipient, subject, and full body and ask them to confirm. Only call "
    "send_email AFTER they explicitly say yes/send. Never send an unapproved "
    "email.\n"
    "- Tone: firm, clear, professional. You may state facts plainly — a task is "
    "overdue, a follow-up is pending, a dependency is blocked on a named person "
    "— but do NOT write personal attacks, insults, or defamatory claims about "
    "anyone. Frame complaints as professional escalations about the work.\n"
    "- If send_email reports it isn't configured, tell the user to add "
    "GMAIL_ADDRESS and GMAIL_APP_PASSWORD to agent/.env (one-time). Never ask "
    "them to paste a password into the chat."
)


def build_model(profile: str | None = None):
    """Build the LLM 'brain'. Profile selects the backend (see config.LLM_PROFILES):
    'mock' (default, the openai-mocker / ChatGPT), or a real provider
    ('openai' / 'anthropic' / 'gemini') for faster, more reliable long browser
    runs. Falls back to 'mock' if the chosen profile has no API key."""
    name = profile or config.LLM_PROFILE
    prof = config.LLM_PROFILES.get(name) or config.LLM_PROFILES["mock"]
    # Key check only -- 'mock' and 'local' are provider 'openai' but ship dummy
    # keys, so this also catches LLM_PROFILE=openai with no OPENAI_API_KEY,
    # which previously reached api.openai.com with an empty credential.
    if not prof.get("api_key"):
        print(f"[agent] profile '{name}' has no API key; using 'mock'.")
        prof, name = config.LLM_PROFILES["mock"], "mock"

    # Every model built here is wrapped with usage_log.metered so its tokens
    # land in data/usage.jsonl. Without it the single-agent path spent real
    # money invisibly: only the swarm was metered, so a run's recorded cost was
    # missing every call this builder made.
    import usage_log

    provider = prof["provider"]
    if provider == "openai":
        from agentscope.credential import OpenAICredential
        from agentscope.model import OpenAIChatModel
        credential = OpenAICredential(
            api_key=prof["api_key"], base_url=prof["base_url"]
        )
        if name == "mock":
            # Pin a stable session id so the mock keeps ONE browser for this
            # agent and opens a new chat per task (see mock_model.py).
            from mock_model import SessionPinnedOpenAIModel
            model = usage_log.metered(SessionPinnedOpenAIModel)(
                credential=credential, model=prof["model"], stream=False,
                session_id=MOCK_SESSION_ID,
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
            model=prof["model"],
            stream=False,
        )
    elif provider == "gemini":
        from agentscope.credential import GeminiCredential
        from agentscope.model import GeminiChatModel
        model = usage_log.metered(GeminiChatModel)(
            credential=GeminiCredential(api_key=prof["api_key"]),
            model=prof["model"],
            stream=False,
        )
    else:
        raise ValueError(f"unknown provider: {provider}")
    return usage_log.attach(model, "task_agent", name)


def _load_state() -> AgentState | None:
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            data = fh.read()
        return AgentState.model_validate_json(data) if data.strip() else None
    except Exception as exc:
        print(f"[warn] couldn't load agent memory ({exc}); starting fresh.")
        return None


def save_state(agent: Agent) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        fh.write(agent.state.model_dump_json())


def discard_live_state() -> int | None:
    """Drop any leftover live AgentState so a fresh server run starts lean.

    Under the per-turn-session design the live state is ephemeral (each turn
    archives + clears it), so a state file present at startup is an abandoned
    turn — typically a browser-heavy one bloated with snapshots. Resuming it
    would re-send the whole blob on the first message (the mock then chunks it
    into many 'segment N of M' pieces). We back it up once, then remove it.

    Returns the size in bytes of the discarded state, or None if there was none.
    """
    global _shared_agent, _current_turns, _session_started_at
    _shared_agent = None
    _current_turns = []
    _session_started_at = None
    if not os.path.exists(STATE_FILE):
        return None
    size = os.path.getsize(STATE_FILE)
    try:
        if size > 2:  # keep a one-shot backup of anything non-trivial
            from datetime import datetime
            bak = STATE_FILE + ".bak-" + datetime.now().strftime("%Y%m%d_%H%M%S")
            os.replace(STATE_FILE, bak)
            print(f"[agent] discarded leftover live state ({size} B) -> {bak}")
        else:
            os.remove(STATE_FILE)
    except OSError as exc:
        print(f"[agent] could not discard live state: {exc}")
        return None
    return size


def _session_primer() -> str:
    """The context a fresh session sees, all assembled locally (no LLM, instant):
      * the standing memory digest + live open threads (memory_context), so the
        agent opens already knowing who the user is and what's pressing, and
      * the last 5 session summaries + the immediately-preceding exchange verbatim
        if it happened in the last 30 min (sessions.primer()).
    """
    block = "\n\n" + sessions.primer(n=5, recent_minutes=30)
    try:
        mem = memory_context.session_context()
    except Exception:  # memory must never break agent construction
        mem = ""
    if mem:
        block = "\n\n" + mem + block
    return block


def build_agent(profile: str | None = None) -> Agent:
    state = _load_state() or AgentState()
    # Autonomous assistant: auto-approve its own tool calls instead of pausing
    # for per-tool confirmation (the user is in control via the chat itself).
    state.permission_context.mode = PermissionMode.BYPASS
    # The snapshot pruner rewrites already-delivered messages, which breaks the
    # mock's delta/prefix-match session continuity and makes it open a fresh
    # ChatGPT chat every step. The mock already sends only the delta, so skip the
    # pruner there; keep it for real, stateless providers that re-send full
    # context each call and genuinely benefit from collapsing old snapshots.
    middlewares = []
    # The ledger is recorded on every profile -- it costs nothing and the cold
    # path learns from it -- but the trail is only INJECTED where the pruner
    # runs, because that is the only place the context loses the history the
    # trail replaces.
    import action_log
    middlewares.append(action_log.ActionRecorder("hermes"))
    # The pruner comes off wherever the element table is on -- see
    # config.BROWSER_ELEMENT_TABLE for the cache arithmetic. The trail injector
    # is paired with it on purpose: it exists to put back the history the
    # pruner deleted, so with nothing deleted it is pure duplication.
    if not config.is_mock_backend(profile) and not config.BROWSER_ELEMENT_TABLE:
        from context_pruner import BrowserSnapshotPruner
        middlewares.append(BrowserSnapshotPruner(keep=1))
        middlewares.append(action_log.ActionTrailInjector("hermes"))
    return Agent(
        name="Hermes-PM",
        system_prompt=SYSTEM_PROMPT + _session_primer(),
        model=build_model(profile),
        toolkit=agent_tools.build_toolkit(),
        state=state,
        middlewares=middlewares,
        # Browser tasks need many tool steps (navigate->snapshot->click->...).
        react_config=ReActConfig(max_iters=40),
    )


# --- current (live) session turn capture -----------------------------------
# We keep a compact transcript of the live session (plain user/assistant text,
# no tool-call noise). On session end it's summarised + archived via sessions.py.
_current_turns: list[dict] = []
_session_started_at: str | None = None
_last_activity_at: float | None = None  # wall-clock of the last turn (idle check)


def _touch_activity() -> None:
    global _last_activity_at
    _last_activity_at = time.time()


def _summarize_turns(turns: list[dict]) -> tuple[str, str]:
    """Deterministic, token-free title + one-line summary from the user's
    messages. Used as the instant placeholder when a session is archived; the
    one-liner is then upgraded by the background LLM summariser."""
    user_msgs = [t["text"].strip().replace("\n", " ")
                 for t in turns if t.get("role") == "user" and t.get("text", "").strip()]
    if not user_msgs:
        return ("Empty session", "No user messages.")
    title = user_msgs[0][:60] + ("…" if len(user_msgs[0]) > 60 else "")
    joined = " | ".join(m[:70] for m in user_msgs[:4])
    summary = f"{len(turns)} turns. {joined}"[:300]
    return (title, summary)


_SUMMARY_PROMPT = (
    "Summarise THIS whole conversation in ONE concise sentence (max 18 words): "
    "what I asked for and what you did. Reply with plain text only — just the "
    "sentence, no preamble, no quotes, and do not call any tool."
)


async def _summarize_via_agent(agent: Agent) -> str:
    """Generate the one-line session summary by sending one more message to the
    agent — so it lands as the LAST message of the SAME ChatGPT chat (a
    continuation), not a disconnected call that would open a new browser.

    Best-effort: the caller falls back to the deterministic summary on empty.
    The summary reply is internal (stored on the session record, not shown to
    the user)."""
    msg = Msg(
        name="user",
        content=[TextBlock(type="text", text=_SUMMARY_PROMPT)],
        role="user",
    )
    reply = await agent.reply(msg)
    line = ((reply.get_text_content() if reply else "") or "").strip()
    return line.splitlines()[0].strip()[:300] if line else ""


async def chat_once(agent: Agent, user_text: str) -> str:
    """Send one user turn, return the agent's text reply, and persist memory.
    Used by both the CLI loop and the web server."""
    global _session_started_at
    if _session_started_at is None:
        from datetime import datetime, timezone
        _session_started_at = datetime.now(timezone.utc).isoformat()
        # Open a fresh action trail for this conversation. The real session id
        # does not exist until archive time, so actions accrue under a run key
        # and get re-keyed there.
        try:
            import action_log
            action_log.start_run("chat")
        except Exception:
            pass
    # Build what we actually SEND the brain: the user's words, plus (a) any
    # relevant memory about what they're discussing — the instant, local
    # "you're talking about Umesh, here's what you know" injection — and (b) on a
    # warm follow-up, the short role-guardrail so the brain doesn't drift. We
    # archive only the user's own words (extras never pollute the transcript).
    extras: list[str] = []
    try:
        mem = memory_context.relevant(user_text)
    except Exception:  # memory is best-effort; never break the turn
        mem = ""
    if mem:
        extras.append(mem)
    try:
        skill = memory_context.procedural_hint(user_text)
    except Exception:
        skill = ""
    if skill:
        extras.append(skill)
    if _current_turns:
        extras.append(GUARDRAIL_REMINDER)
    sent_text = user_text + ("\n\n" + "\n\n".join(extras) if extras else "")
    msg = Msg(
        name="user",
        content=[TextBlock(type="text", text=sent_text)],
        role="user",
    )
    # Supervised browser run (Section K): a goal/checklist/trail header + loop
    # monitor for any browse_* the agent does this turn. Inert on non-browser
    # turns — the header only appears when a browser tool takes a snapshot.
    try:
        import browser_supervisor as _sup
        _sup.set_context(goal=user_text, max_steps=40)
    except Exception:
        pass
    reply = None
    for attempt in range(2):  # tolerate the mock's cold-start on first call
        try:
            reply = await agent.reply(msg)
            break
        except Exception as exc:
            if attempt == 0:
                continue
            raise exc
    save_state(agent)
    try:
        import browser_supervisor as _sup
        _sup.clear_context()
    except Exception:
        pass
    text = (reply.get_text_content() if reply else "") or "(no text)"
    # Capture the plain turn for this session's compact transcript.
    _current_turns.append({"role": "user", "text": user_text})
    _current_turns.append({"role": "assistant", "text": text})
    return text


def _archive_current(summary_override: str | None = None) -> dict | None:
    """Archive the live session (if any), clear the live transcript, and wipe the
    on-disk AgentState. Uses `summary_override` (the LLM one-liner) when given,
    else a deterministic placeholder. Returns the saved record, or None."""
    global _current_turns, _session_started_at
    if not _current_turns:
        return None
    turns = list(_current_turns)
    title, summary = _summarize_turns(turns)
    if summary_override:
        summary = summary_override.strip()[:300]
    rec = sessions.save_session(
        turns=turns, title=title, summary=summary,
        started_at=_session_started_at,
    )
    # Re-key the action trail onto the id learn_from_session will be given.
    try:
        import action_log
        action_log.rebind(action_log.current_session(), rec["id"])
    except Exception:
        pass
    _current_turns = []
    _session_started_at = None
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    # Cold path: extract durable facts from this just-archived conversation into
    # the memory wiki, in the BACKGROUND (the user never waits on it). No-op when
    # there's no running event loop (e.g. the CLI on exit).
    try:
        import memory_learn
        memory_learn.learn_from_session_soon(rec["id"])
    except Exception as exc:  # learning must never break archiving
        print(f"[memory] could not schedule learning: {exc}")
    return rec


def end_current_session() -> dict | None:
    """Summarise + archive the live session (if any) and clear it. Returns the
    saved session record, or None if nothing to save."""
    return _archive_current()


def _rotation_reason() -> str | None:
    """Decide whether the current session should rotate (archive + fresh chat)
    BEFORE handling the next message. Returns the reason, or None to keep going.

    We rotate on natural boundaries only — never mid-burst:
      * 'idle'   — the user has been away longer than SESSION_IDLE_SECONDS, or
      * 'length' — the live context grew past SESSION_MAX_CONTEXT_MSGS (guards
        against the ChatGPT-web brain drifting on a very long thread).
    """
    if not _current_turns:
        return None
    if (_last_activity_at is not None
            and time.time() - _last_activity_at > SESSION_IDLE_SECONDS):
        return "idle"
    if _shared_agent is not None:
        try:
            if len(_shared_agent.state.context) >= SESSION_MAX_CONTEXT_MSGS:
                return "length"
        except Exception:
            pass
    return None


async def _archive_and_rotate(reason: str, profile: str | None = None) -> dict | None:
    """End the current session: summarise it (in the SAME chat, as that chat's
    last message, when the chat is still warm), archive it, and drop the shared
    agent so the next message opens a fresh chat seeded with the primer.

    The summary is skipped for an 'idle' rotation — the chat is cold/gone, so we
    fall back to the deterministic one-liner rather than cold-starting a chat
    just to summarise. Caller must hold the agent lock.
    """
    global _shared_agent
    warm = reason != "idle" and _shared_agent is not None and bool(_current_turns)
    summary = ""
    if warm:
        try:
            summary = await _summarize_via_agent(_shared_agent)
        except Exception as exc:  # keep the deterministic summary on failure
            print(f"[session] summary failed: {exc}")
    rec = _archive_current(summary_override=summary or None)
    _shared_agent = None  # next message rebuilds fresh → mock opens a new chat
    if rec:
        print(f"[session] rotated ({reason}) — archived {rec['id']}: {rec['summary']}")
    return rec


async def _maybe_rotate_session(profile: str | None = None) -> None:
    """Rotate the session if a boundary was reached. Caller holds the lock."""
    reason = _rotation_reason()
    if reason:
        await _archive_and_rotate(reason, profile)


# A single shared agent instance for the web server (one BA, one brain).
_shared_agent: Agent | None = None
_agent_lock = asyncio.Lock()


_shared_profile: str | None = None


def get_shared_agent(profile: str | None = None) -> Agent:
    """Return the shared agent, rebuilding it if the LLM profile changed."""
    global _shared_agent, _shared_profile
    if profile and profile != _shared_profile and _shared_agent is not None:
        # Switch brain but keep conversation memory.
        save_state(_shared_agent)
        _shared_agent = None
    if _shared_agent is None:
        _shared_agent = build_agent(profile)
        _shared_profile = profile or config.LLM_PROFILE
    return _shared_agent


async def start_new_session(profile: str | None = None) -> dict | None:
    """Explicitly end + archive the current session (summarising it in its own
    chat while it's still warm) and reset the agent so the next message begins a
    fresh chat seeded with the updated recent-5 primer. Returns the record."""
    async with _agent_lock:
        return await _archive_and_rotate("manual", profile)


async def server_chat(user_text: str, profile: str | None = None) -> str:
    """Serialised entry point for the web /api/agent/chat endpoint.

    The chat stays WARM across tasks: a follow-up continues the same ChatGPT
    thread (cheap delta, low time-to-first-token) and multi-turn confirmations
    (email send, payment) keep their context. We only archive + open a fresh
    chat at a natural boundary — the user went idle, or the thread got long —
    which is checked here, before the turn runs."""
    async with _agent_lock:
        await _maybe_rotate_session(profile)        # rotate only if idle/long
        reply = await chat_once(get_shared_agent(profile), user_text)
        _touch_activity()
    return reply


async def _cli() -> None:
    agent = build_agent()
    print("\n  Hermes-PM ready. Backend:", config.BASE_URL,
          f"(model {config.MODEL})")
    print("  /exit quit · /reset wipe memory · /new start a fresh session "
          "(archives this one).\n")
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        low = text.lower()
        if low in ("/exit", "/quit"):
            break
        if low == "/new":
            rec = end_current_session()
            agent = build_agent()  # fresh state + updated recent-sessions index
            if rec:
                print(f"  Archived session '{rec['title']}'. Fresh start.\n")
            else:
                print("  Fresh session.\n")
            continue
        if low == "/reset":
            agent.state = AgentState()
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
            print("  Conversation memory wiped.\n")
            continue
        print("Hermes-PM > ", end="", flush=True)
        try:
            print(await chat_once(agent, text), "\n")
        except Exception as exc:
            print(f"\n  [backend error: {exc}]\n")
    # Archive the session on exit so it shows up in the index next time.
    end_current_session()
    print("  Session archived. Bye!")


if __name__ == "__main__":
    asyncio.run(_cli())
