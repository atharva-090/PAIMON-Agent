"""Hermes — a conversational AgentScope agent backed by the openai-mocker.

Run an interactive terminal chat. The agent keeps a persistent memory: the
full conversation context (plus an auto-generated rolling summary once the
history gets long) is saved to disk and reloaded on the next run, so it
remembers you across sessions.

Usage:
    python chat_agent.py            # start / resume chatting
    python chat_agent.py --fresh    # wipe memory and start clean

In-chat commands:
    /exit  /quit   end the session (memory is saved)
    /reset         forget everything and start a new memory
    /memory        show how much the agent currently remembers
"""

import argparse
import asyncio
import json
import os
import sys

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg, TextBlock
from agentscope.state import AgentState

import config


def build_model() -> OpenAIChatModel:
    """An OpenAI-compatible model pointed at the local openai-mocker."""
    credential = OpenAICredential(
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
    )
    return OpenAIChatModel(
        credential=credential,
        model=config.MODEL,
        # stream=False is more robust against the mock and the latency is
        # dominated by ChatGPT's own generation anyway.
        stream=False,
    )


def load_state() -> AgentState | None:
    """Reload persisted memory if we have any."""
    if not os.path.exists(config.STATE_FILE):
        return None
    try:
        with open(config.STATE_FILE, "r", encoding="utf-8") as fh:
            data = fh.read()
        if not data.strip():
            return None
        return AgentState.model_validate_json(data)
    except Exception as exc:  # corrupt/old state shouldn't brick the agent
        print(f"[warn] couldn't load saved memory ({exc}); starting fresh.")
        return None


def save_state(agent: Agent) -> None:
    os.makedirs(os.path.dirname(config.STATE_FILE), exist_ok=True)
    with open(config.STATE_FILE, "w", encoding="utf-8") as fh:
        fh.write(agent.state.model_dump_json())


def memory_summary(agent: Agent) -> str:
    ctx = agent.state.context or []
    turns = len(ctx)
    summary = (agent.state.summary or "").strip()
    line = f"{turns} message(s) held in context."
    if summary:
        line += f"\n  rolling summary: {summary[:300]}"
    return line


def build_agent() -> Agent:
    return Agent(
        name=config.AGENT_NAME,
        system_prompt=config.SYSTEM_PROMPT,
        model=build_model(),
        state=load_state(),
    )


async def chat_loop(agent: Agent) -> None:
    print(f"\n  {config.AGENT_NAME} is ready. Backend: {config.BASE_URL} "
          f"(model {config.MODEL})")
    print(f"  Memory: {memory_summary(agent)}")
    print("  Type a message, or /exit to quit, /reset to forget, "
          "/memory to inspect.\n")

    while True:
        try:
            user_text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue

        low = user_text.lower()
        if low in ("/exit", "/quit"):
            break
        if low == "/memory":
            print(f"  {memory_summary(agent)}\n")
            continue
        if low == "/reset":
            agent.state = AgentState()
            if os.path.exists(config.STATE_FILE):
                os.remove(config.STATE_FILE)
            print("  Memory wiped. Fresh start.\n")
            continue

        msg = Msg(
            name="user",
            content=[TextBlock(type="text", text=user_text)],
            role="user",
        )
        print(f"{config.AGENT_NAME} > ", end="", flush=True)
        reply = None
        # The mock's first call can fail while a cold Chrome warms up; retry once.
        for attempt in range(2):
            try:
                reply = await agent.reply(msg)
                break
            except Exception as exc:
                if attempt == 0:
                    print("(backend warming up, retrying...) ",
                          end="", flush=True)
                    continue
                print(f"\n  [error talking to backend: {exc}]\n")
        if reply is None:
            continue

        print(reply.get_text_content() or "(no text)")
        print()
        # Persist after every turn so a crash never loses memory.
        save_state(agent)

    save_state(agent)
    print(f"  Memory saved to {config.STATE_FILE}. Bye!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Hermes.")
    parser.add_argument(
        "--fresh", action="store_true",
        help="ignore and overwrite any saved memory",
    )
    args = parser.parse_args()

    if args.fresh and os.path.exists(config.STATE_FILE):
        os.remove(config.STATE_FILE)

    agent = build_agent()
    try:
        asyncio.run(chat_loop(agent))
    except KeyboardInterrupt:
        print("\n  interrupted.")


if __name__ == "__main__":
    main()
