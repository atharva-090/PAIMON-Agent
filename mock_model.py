"""A thin OpenAIChatModel subclass that pins a stable mock session id.

The openai-mocker keys its ChatGPT browser to a `session_id` in the request
body. By sending the SAME id on every call, the whole agent lives in ONE browser:
the mock keeps that window and starts a NEW ChatGPT chat at each task boundary
(detected as a non-continuation) instead of spawning a fresh browser per task.

We inject the id via the OpenAI SDK's `extra_body`, which is merged verbatim into
the request JSON. This is only used for the local mock backend — real OpenAI
would reject an unknown body field, so build_model only pins it for the 'mock'
profile.
"""

from __future__ import annotations

from agentscope.model import OpenAIChatModel


class SessionPinnedOpenAIModel(OpenAIChatModel):
    """OpenAIChatModel that adds `session_id` to every request body so the
    openai-mocker reuses one browser across tasks (new chat per task)."""

    def __init__(self, *args, session_id: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mock_session_id = session_id

    async def _call_api(
        self, model_name, messages, tools=None, tool_choice=None,
        **generate_kwargs,
    ):
        extra_body = dict(generate_kwargs.pop("extra_body", None) or {})
        extra_body.setdefault("session_id", self._mock_session_id)
        generate_kwargs["extra_body"] = extra_body
        return await super()._call_api(
            model_name, messages, tools=tools, tool_choice=tool_choice,
            **generate_kwargs,
        )
