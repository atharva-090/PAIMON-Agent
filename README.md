# BA Task Manager + Hermes-PM agent

> 📋 **New here / picking up the project?** Read [`HANDOFF.md`](HANDOFF.md) first — full context, file map, gotchas, and roadmap.

A mini Notion-style task manager for a Business Analyst, with an AI agent
(built on [AgentScope 2.0](https://github.com/agentscope-ai/agentscope)) that
can organise your work for you. **No database** — everything is plain JSON and
Markdown files on disk. The LLM backend is your local **openai-mocker**
(ChatGPT driven through a real browser).

```
Browser UI ─┐
            ├─► FastAPI (server.py) ─► store.py  ─► data/*.json   (tasks)
Hermes-PM ──┘        ▲                  memory_store.py ─► memory/notes/*.md
  agent  ────────────┘  (same store, so agent + UI never disagree)
   │
   └─► openai-mocker (http://127.0.0.1:8000) ─► chatgpt.com
```

## What it does

- **Hierarchy**: Groups → Subgroups → Tasks (a Notion-style tree).
- **Per task**: status, priority, due date, owner/person, notes, **reminders**,
  and **follow-ups**.
- **Follow-up tracking**: every time you log a follow-up with a person on a
  task, the *attempt count* increments — so you can see "I've chased Ravi 3
  times on this issue". Side panels show pending follow-ups and due reminders.
- **Scheduler / mini calendar** (`scheduler.py`): set time-based reminders in
  plain English — "in 20 mins", "in 2 days", "tomorrow at 3pm", "next monday
  09:30" — with optional daily/weekly/monthly recurrence. A background loop in
  the server fires them when due, and the web UI pops a **toast + sound +
  browser notification**. The **📅 Calendar** view shows a month grid of your
  reminders and task due-dates. The agent can schedule for you too
  (`schedule_reminder`).
- **Completed view + audit log**: completing a task never deletes it — it moves
  to the **✅ Completed** view with a completion time, and every change is
  recorded in an append-only audit log (`data/audit.jsonl`), surfaced as a
  History section on each task.
- **Sessions / context management** (`sessions.py`): instead of re-sending the
  entire ever-growing conversation to the LLM every turn (which bloats tokens and
  makes the ChatGPT mock break character), each conversation is a discrete
  **session**. Hitting **＋ New session** archives the current one as a compact
  JSON record with an auto **title + one-line summary**. A fresh session starts
  seeing only the **last-5 sessions index** (titles + summaries); the agent
  decides if your new request relates to one and, if so, calls
  `load_past_session(id)` to pull that transcript back on demand
  (PageIndex-style). Stored under `memory/sessions/`.
- **Autonomous browser control** (`browser_tools.py` + `agent-browser-middleware`):
  the agent can drive a real Chrome to do web tasks — open pages, read the
  accessibility tree, click, type, submit, scrape, then act on what it finds
  (e.g. "check this portal and log a follow-up"). It runs **fully autonomously**
  except two guarded actions — **payments** and **sending a sensitive message** —
  which it pauses to confirm with you first. Every browser action is logged to
  `data/browser_runs/` (raw material for a future "compile a repeated task into a
  script" feature). **Hybrid brain:** pick the LLM from the chat panel — `mock`
  (ChatGPT via the mocker, default/free/slow) or a real `openai`/`anthropic`/
  `gemini` key (much faster for the long multi-step browser loops).
- **AI agent (Hermes-PM)**: a chat panel on the right. Ask it to create groups,
  organise/move tasks, set reminders, log follow-ups, or summarise what's
  pending — it uses tools that operate the *same* JSON store, so the board
  updates live.
- **Two kinds of memory**:
  - *Conversational* — the agent's dialogue state is saved to
    `memory/agent_state.json` after every turn (survives restarts).
  - *Long-term knowledge* — Markdown notes under `memory/notes/`. The agent
    reads a cheap **index** first and only loads the note it needs
    (PageIndex-style, https://github.com/VectifyAI/PageIndex), so memory scales
    without dumping everything into the prompt.

## Run it

**1. Start the openai-mocker** (the LLM backend) — in a terminal:
```powershell
cd "..\openai-mock"
npm start          # serves http://127.0.0.1:8000  (run `npm run login` once first)
```

**2. Start the task manager** — in a second terminal:
```powershell
cd "agent"
.\start.bat
# or:  .\.venv\Scripts\python.exe -m uvicorn server:app --port 8800
```
> Keep this window open — it's the web server. If you close it, the page shows
> "Failed to fetch".

**3. Open the app**: http://127.0.0.1:8800

Prefer a pure terminal chat with the agent (no web UI)? Run:
```powershell
.\.venv\Scripts\python.exe task_agent.py
```

## Try these in the chat panel

- "Create a group *Vendor Onboarding* with a subgroup *Phase 1 – KYC*."
- "Add a high-priority task *Collect KYC docs from Acme* for Ravi, due next Friday."
- "Ravi still hasn't replied about the KYC docs — log another follow-up."
- "Who do I still need to chase, and how many times each?"
- "Remind me to email Priya in 20 minutes." / "Remind me to review the BRD in 2 days."
- "Set a daily standup reminder at 9:30am."
- "Remember that Priya in Legal signs off all vendor contracts." (saved to long-term memory)

## Browser automation setup (one-time)

The agent's browser control needs the Node `agent-browser` CLI and Chrome:

```powershell
npm install -g agent-browser          # the browser backend
pip install -e ..\agent-browser-middleware\agent-browser-middleware  # into agent\.venv
```

It drives your **installed Google Chrome** (no Chrome-for-Testing download needed)
in a **headed, persistent profile** so logins stick and you can watch. Tunables
(all optional, env-driven) live in `config.py`: `BROWSER_CHROME_PATH`,
`BROWSER_HEADED` (1/0), `BROWSER_PROFILE_DIR`, `BROWSER_MAX_OUTPUT`,
`BROWSER_ALLOWED_DOMAINS`. For a faster/reliable brain on long browser tasks, set
`OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`) and pick the brain in
the chat panel.

## Notes & limits

- **Latency**: each agent reply takes ~10–120s. The agent may make several
  ChatGPT calls per message (decide a tool → run it → look at the result →
  answer), and each call is ChatGPT generating in a browser (~10–25s). That's
  inherent to the mocker; the UI shows a "thinking" placeholder meanwhile.
- **Data location**: `data/tree.json` (the navigable hierarchy + task stubs)
  and `data/tasks/<id>.json` (full task bodies). Long-term memory:
  `memory/notes/*.md`. All human-readable and editable.
- **Config**: `config.py` or env vars `MOCK_BASE_URL`, `MOCK_MODEL`.

## Files

| file | role |
| ---- | ---- |
| `store.py` | JSON data layer (groups/subgroups/tasks/reminders/follow-ups, audit log) |
| `scheduler.py` | the scheduler: NL time parsing, events store, due-firing, recurrence |
| `sessions.py` | conversation sessions: compact JSON store, recent-5 index, summarize-on-end |
| `browser_tools.py` | browser-control tools (wrap `agent-browser-middleware`, drive real Chrome) |
| `memory_store.py` | PageIndex-style Markdown long-term memory |
| `agent_tools.py` | 24 AgentScope tools wrapping the two stores |
| `task_agent.py` | the Hermes-PM agent (toolkit + persistent memory) + CLI |
| `server.py` | FastAPI: REST API + serves the frontend + agent chat endpoint |
| `frontend/` | the web UI (vanilla HTML/CSS/JS, no build step) |
| `start.bat` | convenience launcher |
| `chat_agent.py` | the earlier plain chat agent (no tools) — still works |
