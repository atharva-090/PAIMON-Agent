"""FastAPI backend for the task manager.

Exposes:
  * a REST API over store.py for the web frontend (CRUD on groups/subgroups/
    tasks, reminders, follow-ups, search, stats),
  * the static frontend at ``/``,
  * a chat endpoint ``POST /api/agent/chat`` that talks to the Hermes-PM agent
    (task_agent.py), which itself uses the SAME store -- so the agent and the UI
    operate on one shared set of JSON files.

Run:
    .venv/Scripts/python.exe -m uvicorn server:app --port 8800 --reload
or simply:
    .venv/Scripts/python.exe server.py
"""

from __future__ import annotations

import asyncio
import hmac
import mimetypes
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, JSONResponse, RedirectResponse,
                               Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import store
import scheduler

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
MOBILE_DIR = os.path.join(os.path.dirname(__file__), "mobile")

# Make sure the PWA manifest + JS are served with sane content types (Starlette
# StaticFiles relies on the mimetypes db, which doesn't know .webmanifest).
mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("text/javascript", ".js")

app = FastAPI(title="BA Task Manager", version="1.0")


# ---------------------------------------------------------------------------
#  Remote-access auth (Roadmap #5 security gate). NO-OP unless MOBILE_TOKEN is
#  set. When it is, every /api/* request from a NON-loopback client (i.e. your
#  phone over Tailscale/LAN) must carry the token (Authorization: Bearer <tok>,
#  X-Paimon-Token, or ?token=<tok> for media URLs). Loopback (the desktop UI on
#  the PC itself) stays tokenless so enabling auth never breaks local use. The
#  static mobile shell (/m/...) is intentionally NOT gated -- it holds no data
#  and the phone must load it to type the token; only the API behind it is.
#  If you front the server with a localhost tunnel (cloudflared/ngrok), the
#  client appears as loopback -- set MOBILE_AUTH_ALLOW_LOOPBACK=0 to force the
#  token even then.
# ---------------------------------------------------------------------------
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


def _request_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.headers.get("x-paimon-token")
            or request.query_params.get("token") or "").strip()


@app.middleware("http")
async def _mobile_auth(request: Request, call_next):
    token = config.MOBILE_TOKEN
    if token and request.url.path.startswith("/api/"):
        host = (request.client.host if request.client else "") or ""
        loopback = host in _LOOPBACK or host.startswith("127.")
        allow_loopback = os.environ.get(
            "MOBILE_AUTH_ALLOW_LOOPBACK", "1") not in ("0", "", "false")
        if not (loopback and allow_loopback):
            if not hmac.compare_digest(_request_token(request), token):
                return JSONResponse({"detail": "unauthorized — set your access "
                                     "token in the app settings"}, status_code=401)
    return await call_next(request)

# Poll interval for the background scheduler that fires due reminders.
SCHEDULER_TICK_SECONDS = 10


@app.on_event("startup")
async def _reset_agent_live_state() -> None:
    """Start each server run lean: drop any leftover live AgentState (an
    abandoned, possibly browser-bloated turn) so the first chat message doesn't
    re-send a huge blob. Knowledge persists in sessions/ + memory/, not here."""
    try:
        from task_agent import discard_live_state
        discard_live_state()
    except Exception as exc:  # never block startup on this
        print(f"[startup] could not reset agent live state: {exc}")


@app.on_event("startup")
async def _start_scheduler() -> None:
    """Launch the background loop that fires due scheduled reminders."""
    async def loop() -> None:
        while True:
            try:
                fired = await asyncio.to_thread(scheduler.fire_due)
                for e in fired:
                    print(f"[scheduler] fired: {e['message']} (was due {e['when']})")
            except Exception as exc:  # never let the loop die
                print(f"[scheduler] error: {exc}")
            await asyncio.sleep(SCHEDULER_TICK_SECONDS)

    app.state._scheduler_task = asyncio.create_task(loop())


@app.on_event("startup")
async def _start_memory_loops() -> None:
    """Launch the self-healing-memory background work (Roadmap #2): a periodic
    INGEST scan that folds new/changed files in uploads/ + knowledge/ into the
    wiki, and a nightly CONSOLIDATION pass that merges/prunes the wiki. Both are
    off the hot path — the user never waits on them. Gated by MEMORY_BG_ENABLED."""
    import config
    if not config.MEMORY_BG_ENABLED:
        print("[memory] background loops disabled (MEMORY_BG_ENABLED=0)")
        return

    async def loop() -> None:
        import memory_ingest
        import memory_learn
        import memory_nudge
        # Let the server (and the backend) settle before touching the brain.
        await asyncio.sleep(20)
        # Both are cheap, idempotent and local. Seeding first means a fresh
        # install never runs with an empty "About them" digest section, and the
        # repair pass heals any fact stored before clean_fact_text existed.
        try:
            seeded = memory_learn.ensure_profile_seeded()
            if seeded.get("seeded"):
                print(f"[memory] profile seeded: {seeded['seeded']} fact(s)")
            fixed = memory_learn.repair_fact_text()
            if fixed.get("facts_changed"):
                print(f"[memory] repaired {fixed['facts_changed']} fact(s) "
                      f"across {len(fixed['pages_changed'])} page(s)")
        except Exception as exc:
            print(f"[memory] profile seed/repair error: {exc}")
        try:
            rep = await memory_ingest.scan_and_ingest()
            if rep.get("ingested"):
                print(f"[memory] initial ingest: {rep['ingested']} file(s)")
                memory_nudge.emit_file_nudge(rep)
        except Exception as exc:
            print(f"[memory] initial ingest error: {exc}")
        try:
            memory_nudge.run_nudge_scan()
        except Exception as exc:
            print(f"[memory] nudge scan error: {exc}")

        last_consolidate_day: Optional[str] = None
        while True:
            try:
                now = scheduler.now()
                day = now.strftime("%Y-%m-%d")
                if (now.hour == config.MEMORY_CONSOLIDATE_HOUR
                        and last_consolidate_day != day):
                    last_consolidate_day = day
                    print("[memory] nightly consolidation starting…")
                    await memory_learn.consolidate()
            except Exception as exc:
                print(f"[memory] consolidation error: {exc}")
            try:
                rep = await memory_ingest.scan_and_ingest()
                if rep.get("ingested"):
                    memory_nudge.emit_file_nudge(rep)
            except Exception as exc:
                print(f"[memory] ingest error: {exc}")
            try:
                memory_nudge.run_nudge_scan()      # proactive heads-ups
            except Exception as exc:
                print(f"[memory] nudge scan error: {exc}")
            try:
                import people_store                 # learn people off the board
                people_store.sync_from_board()
            except Exception as exc:
                print(f"[memory] people sync error: {exc}")
            await asyncio.sleep(max(60, config.MEMORY_INGEST_MINUTES * 60))

    app.state._memory_task = asyncio.create_task(loop())


# ---------------------------------------------------------------------------
#  Request models
# ---------------------------------------------------------------------------
class GroupIn(BaseModel):
    name: str
    description: str = ""


class SubgroupIn(BaseModel):
    name: str
    description: str = ""


class TaskIn(BaseModel):
    title: str
    group_id: str
    subgroup_id: Optional[str] = None
    description: str = ""
    status: str = "todo"
    priority: str = "medium"
    due_date: Optional[str] = None
    person: Optional[str] = None
    tags: Optional[list[str]] = None


class TaskPatch(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[str] = None
    person: Optional[str] = None
    tags: Optional[list[str]] = None


class MoveIn(BaseModel):
    group_id: str
    subgroup_id: Optional[str] = None


class NoteIn(BaseModel):
    text: str


class ReminderIn(BaseModel):
    remind_at: str
    message: str = ""


class FollowUpIn(BaseModel):
    person: str
    note: str = ""
    due_date: Optional[str] = None


class ChatIn(BaseModel):
    message: str
    llm: Optional[str] = None       # 'mock' (default) | 'local' | 'openai' | 'anthropic' | 'gemini'
    mode: Optional[str] = None      # 'single' (default) | 'swarm' -- swarm uses the orchestrator + 4 children


class TTSIn(BaseModel):
    text: str                        # agent reply to speak (footer stripped server-side)


class ScheduleIn(BaseModel):
    message: str
    when: str                      # natural language ("in 20 mins") or ISO
    recurrence: str = "none"       # none | daily | weekly | monthly
    task_id: Optional[str] = None


class AckIn(BaseModel):
    ids: list[str]


# ---------------------------------------------------------------------------
#  Tree / groups
# ---------------------------------------------------------------------------
@app.get("/api/tree")
def api_tree():
    return store.get_tree()


@app.get("/api/stats")
def api_stats():
    return store.stats()


@app.post("/api/groups")
def api_create_group(body: GroupIn):
    return store.create_group(body.name, body.description)


@app.patch("/api/groups/{group_id}")
def api_rename_group(group_id: str, body: GroupIn):
    try:
        return store.rename_group(group_id, body.name)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/groups/{group_id}")
def api_delete_group(group_id: str):
    store.delete_group(group_id)
    return {"ok": True}


@app.post("/api/groups/{group_id}/subgroups")
def api_create_subgroup(group_id: str, body: SubgroupIn):
    try:
        return store.create_subgroup(group_id, body.name, body.description)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/groups/{group_id}/subgroups/{subgroup_id}")
def api_delete_subgroup(group_id: str, subgroup_id: str):
    store.delete_subgroup(group_id, subgroup_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
#  Tasks
# ---------------------------------------------------------------------------
@app.post("/api/tasks")
def api_create_task(body: TaskIn):
    try:
        return store.create_task(**body.model_dump())
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/tasks/{task_id}")
def api_get_task(task_id: str):
    t = store.get_task(task_id)
    if t is None:
        raise HTTPException(404, "task not found")
    return t


@app.patch("/api/tasks/{task_id}")
def api_update_task(task_id: str, body: TaskPatch):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return store.update_task(task_id, **fields)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/tasks/{task_id}")
def api_delete_task(task_id: str):
    store.delete_task(task_id)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/move")
def api_move_task(task_id: str, body: MoveIn):
    try:
        return store.move_task(task_id, body.group_id, body.subgroup_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/tasks/{task_id}/notes")
def api_add_note(task_id: str, body: NoteIn):
    try:
        return store.add_note(task_id, body.text)
    except KeyError as e:
        raise HTTPException(404, str(e))


# ---------------------------------------------------------------------------
#  Reminders
# ---------------------------------------------------------------------------
@app.post("/api/tasks/{task_id}/reminders")
def api_add_reminder(task_id: str, body: ReminderIn):
    try:
        return store.add_reminder(task_id, body.remind_at, body.message)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/tasks/{task_id}/reminders/{reminder_id}/complete")
def api_complete_reminder(task_id: str, reminder_id: str):
    try:
        return store.complete_reminder(task_id, reminder_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/reminders/due")
def api_due_reminders(before: Optional[str] = None):
    return store.due_reminders(before)


# ---------------------------------------------------------------------------
#  Follow-ups
# ---------------------------------------------------------------------------
@app.post("/api/tasks/{task_id}/followups")
def api_add_follow_up(task_id: str, body: FollowUpIn):
    try:
        return store.add_follow_up(task_id, body.person, body.note, body.due_date)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/tasks/{task_id}/followups/{follow_up_id}/complete")
def api_complete_follow_up(task_id: str, follow_up_id: str):
    try:
        return store.complete_follow_up(task_id, follow_up_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/followups/pending")
def api_pending_follow_ups():
    return store.pending_follow_ups()


@app.get("/api/followups/summary")
def api_follow_up_summary():
    return store.follow_up_counts()


# ---------------------------------------------------------------------------
#  Search
# ---------------------------------------------------------------------------
@app.get("/api/tasks/completed/list")
def api_completed_tasks():
    return store.completed_tasks()


@app.get("/api/audit")
def api_audit(limit: int = 200, task_id: Optional[str] = None):
    return store.read_audit(limit=limit, task_id=task_id)


# ---------------------------------------------------------------------------
#  Run history -- every LLM call and tool call, grouped by conversation.
#  The viewer that draws it lives at /runs.
# ---------------------------------------------------------------------------
@app.get("/api/runs")
def api_runs(limit: int = 50):
    import run_history
    return {"sessions": run_history.sessions(limit=limit)}


@app.get("/api/runs/{session}")
def api_run_detail(session: str):
    import run_history
    d = run_history.detail(session)
    if not d:
        raise HTTPException(status_code=404, detail="no such session")
    return d


@app.get("/api/calls/{call_id}")
def api_call_payload(call_id: str):
    """The complete request and response for one LLM or tool call."""
    import run_history
    d = run_history.payload(call_id)
    if not d:
        raise HTTPException(status_code=404,
                            detail="no payload stored for this call")
    return d


@app.get("/runs")
def runs_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "runs.html"))


# ---------------------------------------------------------------------------
#  Scheduler / calendar
# ---------------------------------------------------------------------------
@app.post("/api/schedule")
def api_create_schedule(body: ScheduleIn):
    try:
        return scheduler.create_event(
            message=body.message, when=body.when,
            recurrence=body.recurrence, task_id=body.task_id,
        )
    except scheduler.WhenParseError as e:
        raise HTTPException(400, str(e))


@app.get("/api/schedule")
def api_list_schedule(frm: Optional[str] = None, to: Optional[str] = None):
    """Events whose time falls in [frm, to] (ISO bounds). Used by the calendar."""
    return scheduler.list_events(frm=frm, to=to)


@app.get("/api/schedule/upcoming")
def api_upcoming(limit: int = 20):
    return scheduler.upcoming(limit=limit)


@app.delete("/api/schedule/{event_id}")
def api_cancel_schedule(event_id: str):
    return {"ok": scheduler.cancel_event(event_id)}


@app.get("/api/schedule/notifications")
def api_notifications():
    """Fired-but-unacknowledged reminders for the UI to pop as toasts."""
    return scheduler.notifications()


@app.post("/api/schedule/notifications/ack")
def api_ack(body: AckIn):
    return {"acknowledged": scheduler.acknowledge(body.ids)}


@app.get("/api/now")
def api_now():
    return {"now": scheduler.now_iso()}


@app.get("/api/search")
def api_search(
    query: str = "",
    status: Optional[str] = None,
    person: Optional[str] = None,
    priority: Optional[str] = None,
):
    return store.search_tasks(query, status, person, priority)


# ---------------------------------------------------------------------------
#  Agent chat
# ---------------------------------------------------------------------------
@app.post("/api/agent/chat")
async def api_agent_chat(body: ChatIn):
    # Imported lazily so the API/UI still work even if the LLM backend (the
    # openai-mocker) isn't running.
    # mode='swarm' routes to the orchestrator+children swarm (multi_agent/);
    # the default 'single' is the original Hermes-PM agent.
    if body.mode == "swarm":
        try:
            from multi_agent.swarm import server_swarm_chat
        except Exception as e:  # pragma: no cover
            raise HTTPException(500, f"swarm unavailable: {e}")
        try:
            reply = await server_swarm_chat(body.message, profile=body.llm)
        except Exception as e:
            raise HTTPException(
                502,
                f"swarm/backend error: {e}. Is the openai-mocker running on "
                f"{__import__('config').BASE_URL}?",
            )
        return {"reply": reply, "mode": "swarm"}

    try:
        from task_agent import server_chat
    except Exception as e:  # pragma: no cover
        raise HTTPException(500, f"agent unavailable: {e}")
    try:
        reply = await server_chat(body.message, profile=body.llm)
    except Exception as e:
        raise HTTPException(
            502,
            f"agent/backend error: {e}. Is the openai-mocker running on "
            f"{__import__('config').BASE_URL}?",
        )
    return {"reply": reply, "mode": "single"}


@app.post("/api/agent/chat/stream")
async def api_agent_chat_stream(body: ChatIn):
    """Same turn as /api/agent/chat, but streamed as server-sent events so the
    UI can show the run happening instead of a spinner.

    POST rather than GET (so EventSource is out, the client reads the body with
    fetch) because a chat message does not belong in a query string.

    The turn runs as a task while we drain the event bus; the bus is drained a
    final time AFTER the task finishes so nothing published in the last
    moments is lost. The reply arrives as a terminal 'done' event, which is the
    same string the non-streaming endpoint returns -- this endpoint adds
    visibility, it does not change behaviour.
    """
    import json as _json

    import agent_events

    async def run_turn() -> str:
        if body.mode == "single":
            from task_agent import server_chat
            return await server_chat(body.message, profile=body.llm)
        from multi_agent.swarm import server_swarm_chat
        return await server_swarm_chat(body.message, profile=body.llm)

    async def frames():
        def frame(payload: dict) -> str:
            return f"data: {_json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

        with agent_events.subscribe() as q:
            yield frame({"kind": "start", "mode": body.mode or "swarm",
                         "llm": body.llm or config.LLM_PROFILE})
            task = asyncio.create_task(run_turn())
            try:
                while True:
                    # Wake on either a new event or the turn ending. The
                    # timeout doubles as an idle keepalive so proxies and
                    # browsers don't close a quiet stream mid-run.
                    getter = asyncio.create_task(q.get())
                    done, _ = await asyncio.wait(
                        {getter, task},
                        timeout=15,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if getter in done:
                        yield frame(getter.result())
                        continue
                    getter.cancel()
                    if task in done:
                        break
                    yield ": keepalive\n\n"

                # Drain whatever landed while the turn was finishing.
                while not q.empty():
                    yield frame(q.get_nowait())

                try:
                    reply = task.result()
                    yield frame({"kind": "done", "reply": reply})
                except Exception as exc:
                    yield frame({"kind": "error",
                                 "message": f"{type(exc).__name__}: {exc}"})
            finally:
                if not task.done():
                    task.cancel()

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.get("/api/agent/swarm/status")
def api_swarm_status():
    """Live view of the orchestrator's delegation board: who's running, who's
    finished, what they reported. Used by the UI to render a small status
    strip while a swarm turn is in flight."""
    try:
        from multi_agent.swarm import status_snapshot
    except Exception:
        return {"running": False, "tasks": []}
    return status_snapshot()


@app.get("/api/agent/swarm/specialists")
def api_swarm_specialists():
    """Static roster: name + title + capabilities for each specialist. The UI
    uses this to render the swarm panel even before any task runs."""
    try:
        from multi_agent.roster import ROSTER
    except Exception:
        return []
    return [
        {"name": s.name, "title": s.title, "capabilities": s.capabilities,
         "has_browser": s.has_browser}
        for s in ROSTER
    ]


@app.get("/api/contacts")
def api_contacts():
    """The self-learning people directory (most recently seen first). Read-only;
    contacts grow automatically from conversations, emails sent, and follow-ups."""
    try:
        import people_store
        return {"contacts": people_store.list_contacts()}
    except Exception as exc:  # pragma: no cover
        return {"contacts": [], "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/agent/sessions")
def api_agent_sessions(limit: int = 8):
    """Recent past conversations (title + one-line summary) for the chat UI."""
    import sessions
    return sessions.recent(limit)


@app.post("/api/agent/new-session")
async def api_agent_new_session():
    """Archive the current conversation (summarised) and start a fresh session.
    The next message will see only the recent-sessions index, not the old
    transcript."""
    from task_agent import start_new_session
    rec = await start_new_session()
    # Also clear the swarm's conversation memory so a fresh session starts the
    # orchestrator + children clean (best-effort; ignored if swarm unused).
    try:
        from multi_agent.swarm import reset_swarm
        reset_swarm()
    except Exception:
        pass
    return {"archived": rec}


# ---------------------------------------------------------------------------
#  Voice -- local STT (whisper.cpp) + TTS (piper). Optional: endpoints 503 when
#  the binaries/models aren't installed, and the UI hides the mic accordingly.
# ---------------------------------------------------------------------------
@app.get("/api/voice/status")
def api_voice_status():
    """Whether voice is available + which model/voice/accel is active. The UI
    polls this on load to decide whether to show the mic button."""
    try:
        import voice
        return voice.status()
    except Exception as exc:  # pragma: no cover
        return {"enabled": False, "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/voice/stt")
async def api_voice_stt(request: Request):
    """Transcribe a 16 kHz mono WAV (raw request body) to text via whisper.cpp."""
    import voice
    if not voice.status()["enabled"]:
        raise HTTPException(503, "voice not installed (run setup_voice.py)")
    audio = await request.body()
    if not audio:
        raise HTTPException(400, "empty audio body")
    try:
        text = await asyncio.to_thread(voice.transcribe, audio)
    except Exception as exc:
        raise HTTPException(500, f"transcription failed: {type(exc).__name__}: {exc}")
    return {"text": text}


@app.post("/api/voice/tts")
async def api_voice_tts(body: TTSIn):
    """Speak ``text`` (agent reply) -> WAV bytes via piper. Strips the agent's
    '[verified actions...]' footer and caps length before synthesis."""
    import voice
    if not voice.status()["enabled"]:
        raise HTTPException(503, "voice not installed (run setup_voice.py)")
    speech = voice.clean_for_speech(body.text)
    if not speech:
        raise HTTPException(400, "nothing to speak")
    try:
        wav = await asyncio.to_thread(voice.synthesize, speech)
    except Exception as exc:
        raise HTTPException(500, f"synthesis failed: {type(exc).__name__}: {exc}")
    return Response(content=wav, media_type="audio/wav")


# ---------------------------------------------------------------------------
#  Reports -- files the scripter GENERATES (DOCX/PDF/XLSX/PPTX/HTML/PNG). The
#  agent writes them into data/reports/ via report_tools; these endpoints list
#  them and serve them for download. (The frontend renders a small Reports list.)
# ---------------------------------------------------------------------------
@app.get("/api/reports")
def api_list_reports():
    """List generated report files (newest first) with download URLs."""
    try:
        import report_tools
        return {"reports": report_tools._scan_reports()}
    except Exception as exc:  # pragma: no cover
        return {"reports": [], "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/reports/{name}")
def api_get_report(name: str):
    """Download one generated report. Sandboxed: only a plain filename inside the
    flat reports folder is served (basename + realpath guard reject traversal)."""
    import config
    base = os.path.realpath(config.REPORTS_DIR)
    safe = os.path.basename(name)               # strip any path components
    full = os.path.realpath(os.path.join(base, safe))
    if (full != base and not full.startswith(base + os.sep)) \
            or not os.path.isfile(full):
        raise HTTPException(404, "no such report")
    # filename= sets a Content-Disposition: attachment so the browser downloads it.
    return FileResponse(full, filename=safe)


# ---------------------------------------------------------------------------
#  Files -- the scoped multi-root explorer (Roadmap #4). Read-only browsing of
#  the allowlisted roots (project / Downloads / Documents); the agent's fs_*
#  tools do the writing. Every path is root-qualified and confined by the same
#  allowlist + realpath guard the tools use (fs_tools._resolve).
# ---------------------------------------------------------------------------
@app.get("/api/files/roots")
def api_files_roots():
    """The filesystem roots the agent (and this explorer) may touch + perms."""
    try:
        import fs_tools
        return {"roots": fs_tools.api_roots()}
    except Exception as exc:  # pragma: no cover
        return {"roots": [], "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/files")
def api_files_list(path: str = ""):
    """List a folder inside an allowed root (empty path -> the roots). The path is
    root-qualified, e.g. 'downloads' or 'documents/contracts'."""
    import fs_tools
    try:
        return fs_tools.api_list(path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/files/content")
def api_files_content(path: str):
    """Download one file from a readable root. Confined by fs_tools.resolve_readable
    (allowlist + realpath guard), so traversal / disallowed roots 404."""
    import fs_tools
    try:
        full = fs_tools.resolve_readable(path)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return FileResponse(full, filename=os.path.basename(full))


# ---------------------------------------------------------------------------
#  Deliveries -- the OUTBOX the agent pushes files/media into so they reach the
#  user's phone (Roadmap #5). The mobile app polls /api/deliveries?since=<seq>
#  and renders each new delivery inline (image/audio/video) or as a download
#  chip; /api/outbox/<name> serves the bytes (sandboxed to data/outbox/).
# ---------------------------------------------------------------------------
@app.get("/api/deliveries")
def api_deliveries(since: int = 0):
    """Files the agent has sent to the phone with seq > ``since`` (oldest-first).
    The client tracks the last seq it has seen so it only fetches what's new."""
    try:
        import deliver_tools
        return deliver_tools.api_deliveries(since)
    except Exception as exc:  # pragma: no cover
        return {"deliveries": [], "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/outbox/{name}")
def api_outbox(name: str):
    """Download one delivered file. Sandboxed to data/outbox/ (basename + realpath
    guard); the download keeps the file's original display name."""
    import deliver_tools
    try:
        full, display = deliver_tools.resolve_outbox(name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return FileResponse(full, filename=display)


# ---------------------------------------------------------------------------
#  Frontend (mounted last so /api/* wins)
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


CHAT_UI_DIR = os.path.join(FRONTEND_DIR, "chat")


@app.get("/chat")
def chat_index_redirect():
    """The focused chat UI (React build) lives under /chat/. Bare /chat -> /chat/."""
    return RedirectResponse(url="/chat/")


@app.get("/m")
def mobile_index_redirect():
    """The installable phone app lives under /m/ (its PWA scope). Bare /m -> /m/."""
    return RedirectResponse(url="/m/")


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# The minimalist mobile chat PWA (Roadmap #5): index.html + app.js + style.css +
# manifest + service worker + icons, all under /m/ so the SW scope is clean.
# html=True makes /m/ serve index.html. Mounted after the API routes so /api/*
# always wins.
if os.path.isdir(MOBILE_DIR):
    app.mount("/m", StaticFiles(directory=MOBILE_DIR, html=True), name="mobile")

# The focused chat UI: a Vite/React build (source in frontend/chat-ui/, built
# with `npm run build`). Absent until it has been built once, which is why this
# is guarded -- the rest of the app must keep working without it.
if os.path.isdir(CHAT_UI_DIR):
    app.mount("/chat", StaticFiles(directory=CHAT_UI_DIR, html=True),
              name="chat")


if __name__ == "__main__":
    import uvicorn

    # Bind to MOBILE_HOST (default loopback). Set MOBILE_HOST=0.0.0.0 to let your
    # phone reach the server over the LAN / Tailscale (do that together with
    # MOBILE_TOKEN so the API isn't open). mobile_serve.py is the convenience
    # launcher that prints the phone URL + a QR code.
    host = os.environ.get("MOBILE_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8800"))
    uvicorn.run(app, host=host, port=port)
