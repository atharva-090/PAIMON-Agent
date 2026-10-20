"""An OpenAI-compatible HTTP shim in front of local K2-Horizon weights.

The agent already speaks OpenAI (AgentScope's OpenAIChatModel), and every
`build_model` in this repo dispatches on `provider == "openai"` + a base_url.
So rather than teach the agent a new provider, we make the local model look
like OpenAI and just point a profile at it:

    LLM_PROFILES["local"] -> http://127.0.0.1:8100/v1

That is also why switching back to the openai-mocker stays a one-word change.

Run:
    agent/.venv-k2/Scripts/python.exe -m uvicorn local_model.serve_k2:app \
        --host 127.0.0.1 --port 8100
or just:
    agent/start_k2.bat
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import k2_runtime
from .k2_runtime import RUNTIME

PORT = int(os.environ.get("K2_PORT", "8100"))
# Load the weights at startup by default; set K2_LAZY=1 to defer to first call.
LAZY = os.environ.get("K2_LAZY", "0") not in ("0", "", "false")


# ===========================================================================
#  Request / response models
# ===========================================================================
class ChatRequest(BaseModel):
    model: str = k2_runtime.MODEL_ID
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    # Non-standard passthrough so callers can dial reasoning per request.
    reasoning_effort: str | None = None
    # The openai-mocker uses this; accept and ignore so the same client works.
    session_id: str | None = None

    model_config = {"extra": "allow"}


def _ensure_loaded() -> None:
    if RUNTIME.model is None:
        RUNTIME.load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not LAZY:
        # Loading is slow and blocking; keep the event loop free.
        await anyio.to_thread.run_sync(_ensure_loaded)
    yield


app = FastAPI(title="K2-Horizon local shim", lifespan=lifespan)


# ===========================================================================
#  Endpoints
# ===========================================================================
@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if RUNTIME.model is not None else "loading",
        "model": k2_runtime.MODEL_ID,
        "model_path": k2_runtime.MODEL_PATH,
        "quant": k2_runtime.QUANT,
        "device": RUNTIME.device,
        "dtype": RUNTIME.dtype_name,
        "max_context": k2_runtime.MAX_CONTEXT,
        "reasoning_effort": k2_runtime.REASONING_EFFORT,
        "kv_quant": k2_runtime.KV_QUANT if RUNTIME._kv_quant_ok else "none",
    }


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [{
            "id": k2_runtime.MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "IFM",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    stop = [req.stop] if isinstance(req.stop, str) else req.stop
    budget = req.max_completion_tokens or req.max_tokens

    def _run() -> k2_runtime.Generation:
        _ensure_loaded()
        return RUNTIME.generate(
            messages=req.messages,
            tools=req.tools,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=budget,
            stop=stop,
            reasoning_effort=req.reasoning_effort,
        )

    try:
        gen = await anyio.to_thread.run_sync(_run)
    except Exception as exc:                       # surface real errors to the agent
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _sse(completion_id, created, req.model, gen),
            media_type="text/event-stream",
        )

    message: dict[str, Any] = {"role": "assistant", "content": gen.content or None}
    if gen.reasoning:
        message["reasoning_content"] = gen.reasoning
    if gen.tool_calls:
        message["tool_calls"] = gen.tool_calls

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": gen.finish_reason,
        }],
        "usage": {
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "total_tokens": gen.prompt_tokens + gen.completion_tokens,
        },
    }


def _sse(completion_id: str, created: int, model: str,
         gen: k2_runtime.Generation):
    """Emit the finished generation as SSE chunks.

    We generate fully before streaming rather than streaming raw tokens: tool
    calls arrive inside <ifm|tool_calls> markup that can only be parsed once the
    block is complete. The agent runs stream=False anyway, so this path exists
    for curl/other clients, not for latency.
    """
    def chunk(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return "data: " + json.dumps(payload) + "\n\n"

    yield chunk({"role": "assistant"})
    if gen.reasoning:
        yield chunk({"reasoning_content": gen.reasoning})
    if gen.content:
        yield chunk({"content": gen.content})
    if gen.tool_calls:
        for i, call in enumerate(gen.tool_calls):
            yield chunk({"tool_calls": [{"index": i, **call}]})
    yield chunk({}, gen.finish_reason)
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)
