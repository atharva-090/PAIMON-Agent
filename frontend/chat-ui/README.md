# chat-ui — the focused chat surface

A React/Vite/shadcn app served by FastAPI at **`/chat/`**. The task board at `/`
is unchanged and still owns every setting; this page is only for talking to the
swarm and watching it work.

It is pinned to **`llm: openai`** and **`mode: swarm`** (see the constants at the
top of `src/App.tsx`) — the real API and the full orchestrator + 4 children, not
the mocker and not single-agent mode.

## Building

The built output is committed to `../chat/`, which `server.py` mounts. You only
need to rebuild after changing something in `src/`:

```bash
cd agent/frontend/chat-ui
npm install      # first time only
npm run build    # writes ../chat/
```

Running the API is unchanged (`uvicorn server:app`, or `start.bat`). If
`../chat/` does not exist, `/chat/` simply 404s and the rest of the app is
unaffected — the mount in `server.py` is guarded.

For UI work with hot reload:

```bash
npm run dev      # http://localhost:5173, proxies /api to 127.0.0.1:8800
```

## How the live view works

`POST /api/agent/chat/stream` runs the same turn as `/api/agent/chat` but
streams server-sent events while it does. The events come from
`agent/agent_events.py`: a `StreamEmitter` middleware sits in every agent's
chain (alongside the snapshot pruner) and copies reasoning text, tool calls and
tool results onto an in-process bus. It is a pure observer — the tests in
`agent/test_agent_events.py` pin that everything flowing through it comes back
out identical, in order, with exceptions still propagating.

Event kinds, all tagged with the agent that produced them:

| kind | meaning |
| --- | --- |
| `start` | the turn opened (echoes mode + llm) |
| `step_start` / `step_end` | one reasoning step of one agent |
| `text` / `thinking` | model output deltas |
| `tool_call` | the model chose a tool |
| `tool_start` / `tool_end` | the tool ran; `tool_end` carries `ok`, `ms`, `preview` |
| `delegation` | the orchestrator handed work to a child |
| `report` | a child finished, with its report and `tools_used` |
| `done` | terminal; carries the same reply the plain endpoint returns |
| `error` | terminal; the run failed |

### A note on "thinking"

gpt-4o does not emit reasoning tokens, so the veiled band is **not** chain of
thought. It shows what actually happened: delegations, tool calls with their
arguments, results, and the interim text the agents produced. If a model that
does emit thinking blocks is used later, `ThinkingBlockDeltaEvent` is already
handled and will appear in the same band.
