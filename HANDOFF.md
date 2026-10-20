# Project Handoff — Hermes-PM (BA Task Manager + Autonomous AI Agent)

Context transfer for a fresh AI session. Read this top-to-bottom to continue the
work with full context. Everything described is **built and working** unless
marked otherwise.

---

## 1. What this project is

A personal **task-management app for a Business Analyst (BA)**, plus an AI agent
("Hermes-PM") that manages the tasks AND can act on the web autonomously. The
LLM "brain" is a local **openai-mocker** that impersonates the OpenAI API but is
actually driving **chatgpt.com** in a real browser (so it's free but slow/flaky).

**No database** — all state is plain JSON + Markdown files on disk.

Three sub-projects live under `E:\KIRO projects\Open AI mocker (hermes agent try)\`:
- `openai-mock/` — Node/Express server (port **8000**) that fakes the OpenAI
  Chat Completions API by driving ChatGPT in Chrome via Playwright. *Pre-existing;
  we mostly consume it.*
- `agent/` — the app we built: FastAPI backend + web UI + the AgentScope agent
  (port **8800**). **This is the main workspace.**
- `agent-browser-middleware/agent-browser-middleware/` — a pure-Python CLI that
  wraps the Node `agent-browser` CLI to drive a real Chrome (the agent's "hands").

There is also a persistent project memory at
`%USERPROFILE%\.claude\projects\E--KIRO-projects-...\memory\` (`MEMORY.md` +
`openai-mock-project.md` + `agentscope-chat-agent.md`) — auto-loaded each chat.

---

## 2. The LLM backend — openai-mocker (port 8000)

- Speaks OpenAI `POST /v1/chat/completions` (+ `/v1/models`, `/health`) but
  fulfils each call by typing into chatgpt.com and reading the SSE reply.
- Point any OpenAI client at `http://127.0.0.1:8000/v1`; API key ignored; model
  label `gpt-4o`.
- Run: `cd openai-mock` → `npm run login` (one-time ChatGPT auth) → `npm start`.
- **Hard limits / quirks (important):**
  - ~10–25 s per call (it's ChatGPT generating in a browser). Token counts are
    estimated.
  - **Flaky** — it has crashed mid-session several times (agent then gets
    "Connection error", and `:8000` stops listening). Fix = restart `npm start`.
  - It re-injects **the whole conversation** into one prompt every turn, wrapped
    in a "you are an API endpoint / BEGIN SESSION…END SESSION" framing to stop
    ChatGPT breaking character. Big transcripts → ChatGPT refuses / breaks
    character. (This motivated our session feature, §6.)
  - Very large prompts (~70k+) get refused as prompt-injection; keep prompts/
    snapshots lean.
  - **Warm pool**: `WARM_POOL=true, WARM_POOL_SIZE=4` pre-launches 4 logged-in
    ChatGPT browsers to skip the ~5 s cold-start. It IS wired in
    (`sessionManager.js` `prime()`/`acquire()`). Currently **over-provisioned**
    for single-user (4 idle Chromes ≈ 2 GB; also `WARM_POOL_SIZE=4` >
    `MAX_SESSIONS=3`). Recommendation: drop to 1–2 now; for multi-agent set it =
    concurrent agents and keep `WARM_POOL_SIZE ≤ MAX_SESSIONS`. (Decision still
    pending with the user.)

---

## 3. The agent app (`agent/`) — file map

Runs in its own venv: `agent\.venv` (Python 3.14, AgentScope 2.0.0 installed).

| file | role |
|---|---|
| `store.py` | JSON data layer: groups→subgroups→tasks, reminders, follow-ups (auto-incrementing per-person `attempt` count), `completed_at`, append-only audit log `data/audit.jsonl`. PageIndex-style `data/tree.json` (stubs) + `data/tasks/<id>.json` (bodies). |
| `memory_store.py` | Long-term memory as Markdown notes under `memory/notes/` + cheap `_index.json` TOC (read index → load note on demand). |
| `sessions.py` | Conversation sessions: compact JSON under `memory/sessions/` + `index.json`; recent-5 index; summarize-on-end. |
| `scheduler.py` | Reminders/calendar: natural-language `when` parser, `data/events.json`, recurrence, firing. |
| `browser_tools.py` | 12 browser tools wrapping the middleware (drive real Chrome). |
| `agent_tools.py` | All the AgentScope FunctionTools (tasks/memory/scheduler/sessions) + merges browser tools. `build_toolkit()` → **45 tools**. |
| `task_agent.py` | The Hermes-PM agent: model profiles (hybrid brain), system prompt, session lifecycle, CLI + `server_chat`. |
| `server.py` | FastAPI: REST API + serves `frontend/` (desktop) + the `/m` mobile PWA + `/api/agent/chat` + scheduler/memory loops + session/reports/files/**deliveries** endpoints + the **`MOBILE_TOKEN` auth middleware**. |
| `config.py` | All config: LLM profiles, browser settings, paths, fs roots, **outbox + mobile auth (#5)**. |
| `frontend/` | Vanilla HTML/CSS/JS desktop UI (no build step): sidebar tree, Calendar view, Completed view, chat panel w/ brain selector + New-session button. |
| `mobile/` | **(#5)** The installable **phone PWA** at `/m`: minimalist voice+text chat (`index.html`/`app.js`/`style.css`), `manifest.webmanifest` + `sw.js`, `icons/` + `make_icons.py` (Pillow). `mobile/apk/` = Capacitor scaffold for a real `.apk`. |
| `deliver_tools.py` | **(#5)** Agent→phone file delivery: `send_file(path,caption)` copies a readable file into `data/outbox/` + records it; the phone polls `/api/deliveries` and renders inline/as a download. |
| `mobile_serve.py` / `start_mobile.bat` | **(#5)** Phone-mode launcher: binds `0.0.0.0`, prints the phone URL + QR, auto-detects the Tailscale IP. |
| `MOBILE_APP.md` | **(#5)** Install routes (PWA / PWABuilder / Capacitor), Tailscale for any-network access, token auth, the delivery feature. |
| `start.bat` / `start_mobile.bat` | Launch the server — localhost desktop, or phone-reachable (`0.0.0.0` + QR). |

Run the app: `cd agent` → `.\start.bat` (or
`.\.venv\Scripts\python.exe -m uvicorn server:app --port 8800`) → open
**http://127.0.0.1:8800**. Keep the window open (else UI shows "Failed to fetch").

---

## 4. Features built (all verified)

1. **Task board** — groups → subgroups → tasks; status/priority/due/person/notes;
   web UI with sidebar tree, create/edit/move, search.
2. **Reminders & follow-ups** — per-task; follow-ups track how many times each
   person was chased (`attempt` count); side panels for due/pending.
3. **Completed view + audit log** — done tasks hidden from group lists, shown in
   ✅ Completed with `completed_at`; every mutation appended to `data/audit.jsonl`,
   surfaced as a History section per task.
4. **Scheduler / mini-calendar** — "remind me in 20 mins / in 2 days / tomorrow
   3pm", recurrence (daily/weekly/monthly); a background loop in the server fires
   due reminders → browser **toast + sound + Notification**; 📅 Calendar month view.
5. **Autonomous browser control** (§5).
6. **Sessions / context management** (§6).
7. **Hybrid brain** — chat panel dropdown picks the LLM: `mock` (default) or real
   `openai`/`anthropic`/`gemini` (env keys: `OPENAI_API_KEY`, etc.). `config.LLM_PROFILES`.
8. **Persistent memory** — conversational (AgentState) + long-term MD notes.

---

## 5. Autonomous browser automation

- Backend: `agent-browser` (Node CLI, v0.27, installed globally) drives the
  **user's installed Chrome** (`C:\Program Files\Google\Chrome\Application\chrome.exe`)
  — set via `AGENT_BROWSER_EXECUTABLE_PATH` because the Chrome-for-Testing
  download timed out. **Headed**, persistent profile (`agent/.browser-profile`),
  output truncated. All via env vars that `browser_tools.py` sets at import — no
  middleware edits needed.
- `browser_tools.py` exposes 12 tools: `browse_open / snapshot / click / type /
  press / scroll / back / get_text / eval / close` + 2 guarded:
  `browse_confirm_payment`, `browse_send_sensitive`.
- **Loop:** `browse_open(url)` → returns accessibility tree with `[ref=eN]` →
  `browse_click('@eN')` / `browse_type('@eN', text)`. State-changing actions
  **auto-return a fresh snapshot** (halves slow LLM round-trips). Must snapshot
  before clicking (refs only exist after a snapshot; go stale on navigation).
- **Safety (user's choice): fully autonomous** (navigate/type/click/submit on its
  own) EXCEPT **payments** and **sending sensitive messages** → the two guarded
  tools return `PENDING_CONFIRMATION`, agent asks user, proceeds only on "yes".
- Every browser action logged to `data/browser_runs/run_<id>.jsonl`.
- Verified: agent autonomously opened example.com, read the snapshot, reported the
  heading, closed — confirmed in the action log.
- v0.27 gotchas: response shape is `{success, data, error}`; the middleware's
  `eval`→`evaluate` mapping is WRONG (correct subcommand is `eval`, handled in
  `browser_tools`).

---

## 6. Sessions / context management (newest feature)

**Problem:** the mocker re-sends the entire transcript every turn → token bloat +
ChatGPT breaks character / refuses (e.g. it refused browser/login tasks mid-run).

**Solution (PageIndex-style):**
- Each finished conversation → compact `memory/sessions/<id>.json` (id, title,
  one-line summary, plain user/assistant turns — no tool-call noise) + `index.json`.
- A **new session** only sees the **last-5 index** (titles+summaries) appended to
  the system prompt by `task_agent._session_primer()`. It decides relevance and
  calls `load_past_session(id)` to pull a full transcript **only when needed**.
- Tools: `list_past_sessions`, `load_past_session`, `search_past_sessions`.
- `task_agent.chat_once` captures turns into `_current_turns`;
  `end_current_session()` summarizes (deterministic from user msgs — no extra LLM
  call) + archives + clears `agent_state.json`; `start_new_session()` resets the
  shared agent.
- Server: `POST /api/agent/new-session`, `GET /api/agent/sessions`. UI: **＋ New
  session** button + recent-session titles on boot. CLI: `/new`.
- Verified end-to-end (chat → archive → index → fresh session sees only one-liner).

---

## 7. Key AgentScope 2.0.0 gotchas (hard-won — will save hours)

- Main class `agentscope.agent.Agent` (ReAct). Build with `name, system_prompt,
  model, toolkit, state, react_config=ReActConfig(max_iters=...)`.
- Messages: `Msg(name=..., content=[TextBlock(type="text", text=...)], role=...)`
  — **content must be a list of blocks, not a str**. `agent.reply(msg)` is async,
  returns `Msg`; read with `.get_text_content()`.
- **Tools**: plain function + Google-style docstring → `FunctionTool(f)` →
  `Toolkit(tools=[...])`. Tool functions must return **`ToolChunk(content=[
  TextBlock(...)])`**, NOT `ToolResponse` (else DeveloperOrientedException).
- Do **NOT** put `from __future__ import annotations` in a tools module —
  stringized `Optional[...]` breaks pydantic schema gen
  (`_StructuredOutputDynamicClass is not fully defined`).
- Non-read-only tools pause for confirmation by default ("Waiting for tool calls
  to be confirmed…"). We set `state.permission_context.mode =
  PermissionMode.BYPASS` for autonomous execution.
- Persistence: serialize `agent.state` (AgentState) via `model_dump_json()`,
  reload with `AgentState.model_validate_json()` → `Agent(state=...)`.
- Models: `OpenAIChatModel`/`AnthropicChatModel`/`GeminiChatModel` with matching
  `*Credential(api_key=..., base_url=...)`. `stream=False` for reliability.

---

## 8. Current state & how to run

1. Mock: `cd openai-mock` → `npm start` (must be logged in; port 8000). **It's
   flaky — if the agent says "Connection error" / `:8000` not listening, restart it.**
2. App: `cd agent` → `.\start.bat` (port 8800). Keep window open.
3. Open http://127.0.0.1:8800. Chat panel: pick brain, ＋ New session, ask it to
   manage tasks / set reminders / browse the web.
4. **Phone (#5):** instead of `start.bat`, run `.\start_mobile.bat` → it prints a
   phone URL + QR; open `…/m` on the phone (same WiFi, or **Tailscale** for any
   network) and **Install** the PWA. Set `MOBILE_TOKEN` before exposing remotely.
   Ask Paimon to "send me the report" → it arrives in the phone chat. See
   `MOBILE_APP.md`.
- Browser one-time setup (already done): `npm install -g agent-browser`;
  `pip install -e ..\agent-browser-middleware\agent-browser-middleware` into the venv.

---

## 9. Known issues / constraints

- **Latency**: ~10–120 s per agent reply; browser tasks are many steps → minutes.
  Use a real-API brain (dropdown) for long/important runs.
- **Mock reliability**: crashes; restart `npm start`. Big context → character
  breaks (sessions feature mitigates this).
- Session summaries are **deterministic** (title = first user msg). Could be
  LLM-generated for nicer titles (deferred).
- Screenshot preview tool was glitchy during dev; verification was done via DOM
  inspection + API + action logs.

---

## 10. Roadmap / what's next (discussed with user)

1. **Multi-agent** — the stated direction. Will need: multiple concurrent agents
   (each its own AgentState/session + browser session), and on the mock side raise
   `MAX_SESSIONS` and size `WARM_POOL_SIZE` = concurrent agents
   (keep `WARM_POOL_SIZE ≤ MAX_SESSIONS`).
2. **Task templating (deferred, design agreed):** every browser run is already
   logged (`data/browser_runs/`). When a run finishes **in one shot with no
   errors**, the agent asks "make this repeatable?"; on yes, **the system
   deterministically compiles the recorded action log into a Playwright script**
   (plain codegen, NOT AI-written) and registers it as a fast tool, so repeat
   tasks skip the slow reasoning loop. The action logging needed for this exists.
3. **Warm-pool right-sizing** — pending user decision (drop to 1–2 now; scale for
   multi-agent).
4. Optional niceties offered but not built: LLM-written session summaries;
   auto-archive a session after idle; OS-level desktop notifications from the
   server; richer browser "activity" view.

---

## 11. Approved plan doc

The browser-automation plan (with the locked decisions) is at
`%USERPROFILE%\.claude\plans\prancy-stirring-pony.md`.
