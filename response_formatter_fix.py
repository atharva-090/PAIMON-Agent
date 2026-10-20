"""Three repairs to AgentScope's OpenAI Responses formatter.

Each one makes every tool-using turn fail with a 400, and each was found by
reading the request we actually sent (data/call_payloads.jsonl) rather than
the source.


The Responses API puts TWO identifiers on a function call:

    id       fc_xxx    the item id, echoed when you replay the item
    call_id  call_xxx  the id the matching function_call_output must carry

AgentScope keeps both on ``ToolCallBlock`` and emits them correctly. But
``ToolResultBlock`` has only ``id``, and the ReAct loop fills it from
``tool_call.id`` -- the *item* id -- so the formatter emits

    {"type": "function_call_output", "call_id": "fc_xxx", ...}

OpenAI resolves the echoed fc_xxx item to its real call_xxx, finds no output
under that id, and rejects the whole request:

    No tool output found for function call call_D7qvHk9I9bbUVl99fcDWCDzO.

Every tool-using turn fails, so the agent cannot get past its first tool call.

The repair is a post-pass over the formatted items: a function_call_output
whose call_id matches some function_call's *item* id is rewritten to that
call's real call_id. If AgentScope fixes this upstream the ids already agree,
nothing matches, and this becomes a no-op.

SECOND, related problem. A reasoning model returns the function_call next to a
``reasoning`` item (rs_xxx), and the API refuses to accept the one without the
other:

    Item 'fc_0ae7...' of type 'function_call' was provided without its
    required 'reasoning' item: 'rs_0ae7...'.

AgentScope only keeps the rs_xxx id when the reasoning SUMMARY is non-empty
(``if combined_summary:`` in the Responses model), and summaries are empty
unless you ask for them -- so the id is usually discarded while the fc_xxx id
is still echoed. Echoing the item id is what creates the obligation, so when
no reasoning item is present we drop ``id`` from the function_call and let
``call_id`` alone pair the call with its output. The call still replays
correctly; what is lost is reasoning CONTINUITY across turns -- the model
re-reasons instead of resuming. That is the cost of AgentScope not retaining
the id, and it is cheaper than requesting summaries on every call and still
depending on them being non-empty.
"""
from __future__ import annotations

from agentscope.formatter import OpenAIResponseFormatter


def repair_call_ids(items: list) -> list:
    """Make function_call / function_call_output items acceptable to the API."""
    # item id -> real call id, for the calls in this request
    by_item: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict) or it.get("type") != "function_call":
            continue
        item_id, call_id = it.get("id"), it.get("call_id")
        if item_id and call_id and item_id != call_id:
            by_item[item_id] = call_id

    # No early return when by_item is empty: the reasoning-item strip below
    # still has to run, and it did not when the two ids happened to agree.
    for it in items:
        if not isinstance(it, dict) or it.get("type") != "function_call_output":
            continue
        fixed = by_item.get(it.get("call_id"))
        if fixed:
            it["call_id"] = fixed

    # THIRD problem. The Responses API splits content part types by direction:
    # a user or system message carries "input_text", an assistant message must
    # carry "output_text". The formatter writes "input_text" for every role, so
    # the first turn (no assistant text in history yet) succeeds and the
    # SECOND one fails:
    #
    #     Invalid value: 'input_text'. Supported values are: 'output_text'
    #     and 'refusal'.   param: input[4].content[0]
    # Collapsing to a plain string rather than rewriting the part type to
    # "output_text": a structured assistant message is a ResponseOutputMessage,
    # which REQUIRES id + status + type="message" (see the SDK's
    # ResponseOutputMessageParam) and AgentScope keeps no msg_xxx id to put
    # there. EasyInputMessageParam accepts a bare string for any role, so the
    # string sidesteps the whole input_text/output_text distinction and needs
    # no ids.
    for it in items:
        if not isinstance(it, dict) or it.get("role") != "assistant":
            continue
        parts = it.get("content")
        if not isinstance(parts, list) or not parts:
            continue
        if all(isinstance(p, dict) and p.get("type") in ("input_text",
                                                         "output_text")
               for p in parts):
            it["content"] = "".join(p.get("text", "") for p in parts)
        else:
            # Something non-textual is in there; keep the structure and fix
            # only the direction of the text parts.
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "input_text":
                    part["type"] = "output_text"

    # Echoing a stored fc_xxx item obliges us to echo its rs_xxx sibling too.
    # When no reasoning item made it into the request, drop the item id rather
    # than send an obligation we cannot meet. If reasoning IS present the ids
    # are kept, so proper continuity still works when AgentScope provides it.
    if not any(isinstance(i, dict) and i.get("type") == "reasoning"
               for i in items):
        for it in items:
            if isinstance(it, dict) and it.get("type") == "function_call":
                it.pop("id", None)
    return items


def validate_items(items: list) -> list:
    """Items that the OpenAI SDK's own request schema rejects, as (i, why).

    The installed SDK carries the Responses request types, so this catches a
    malformed item offline instead of as a 400 three restarts later. It is a
    development aid, not a runtime gate: the server is STRICTER than these
    types (it rejects input_text on an assistant message, which the union
    happily accepts), so an empty result means "nothing provably wrong", not
    "guaranteed to be accepted".
    """
    try:
        from pydantic import TypeAdapter
        from openai.types.responses.response_input_param import (
            ResponseInputItemParam)
    except Exception:
        return []
    ta = TypeAdapter(ResponseInputItemParam)
    bad = []
    for i, it in enumerate(items):
        try:
            ta.validate_python(it)
        except Exception as exc:
            bad.append((i, str(exc).splitlines()[0][:120]))
    return bad


class PatchedOpenAIResponseFormatter(OpenAIResponseFormatter):
    """OpenAIResponseFormatter with the output call_id corrected."""

    async def format(self, *args, **kwargs) -> list:
        return repair_call_ids(await super().format(*args, **kwargs))


if __name__ == "__main__":
    # The exact shape from data/call_payloads.jsonl on the failing run.
    FC, CALL = "fc_0229c38b50ddd5ec006", "call_D7qvHk9I9bbUVl99fcDWCDzO"
    items = [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "function_call", "id": FC, "call_id": CALL,
         "name": "check_status", "arguments": "{}"},
        {"type": "function_call_output", "call_id": FC, "output": "{}"},
    ]
    out = repair_call_ids(items)
    assert out[2]["call_id"] == CALL, out[2]
    assert out[1]["call_id"] == CALL, out[1]
    # No reasoning item in the request -> the fc_ id is dropped, so the API
    # does not demand the rs_ sibling AgentScope threw away.
    assert "id" not in out[1], out[1]
    assert out[0] == {"role": "user",
                      "content": [{"type": "input_text", "text": "hi"}]}

    # With a reasoning item present the obligation CAN be met, so the item id
    # is kept and reasoning continuity is preserved.
    with_rs = [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "function_call", "id": FC, "call_id": CALL,
         "name": "check_status", "arguments": "{}"},
        {"type": "function_call_output", "call_id": FC, "output": "{}"},
    ]
    kept = repair_call_ids(with_rs)
    assert kept[1]["id"] == FC, kept[1]
    assert kept[2]["call_id"] == CALL, kept[2]

    # Already correct (upstream fixed, or a non-Responses API with one id):
    # nothing matches, nothing is touched.
    same = [{"type": "function_call", "id": CALL, "call_id": CALL,
             "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": CALL, "output": "{}"}]
    fixed_same = repair_call_ids([dict(x) for x in same])
    assert fixed_same[0]["call_id"] == CALL and "id" not in fixed_same[0]
    assert fixed_same[1] == same[1]

    # An output with no matching call is left alone rather than guessed at.
    orphan = [{"type": "function_call_output", "call_id": "fc_unknown",
               "output": "{}"}]
    assert repair_call_ids([dict(x) for x in orphan]) == orphan

    # Two calls in one turn must not cross-wire.
    two = [{"type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "a", "arguments": "{}"},
           {"type": "function_call_output", "call_id": "fc_1", "output": "1"},
           {"type": "function_call", "id": "fc_2", "call_id": "call_2",
            "name": "b", "arguments": "{}"},
           {"type": "function_call_output", "call_id": "fc_2", "output": "2"}]
    fixed = repair_call_ids(two)
    assert [x["call_id"] for x in fixed] == ["call_1", "call_1",
                                             "call_2", "call_2"], fixed
    assert all("id" not in x for x in fixed
               if x["type"] == "function_call"), fixed
    # An assistant text part must be output_text; user/system stay input_text.
    roles = [
        {"role": "system", "content": [{"type": "input_text", "text": "s"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "u"}]},
        {"role": "assistant", "content": [{"type": "input_text", "text": "a"},
                                          {"type": "input_text", "text": "b"}]},
    ]
    r = repair_call_ids(roles)
    assert r[0]["content"][0]["type"] == "input_text", r[0]
    assert r[1]["content"][0]["type"] == "input_text", r[1]
    assert r[2]["content"] == "ab", r[2]
    # A mixed assistant message keeps its structure; only the direction of the
    # text parts is corrected.
    mixed = [{"role": "assistant",
              "content": [{"type": "input_text", "text": "a"},
                          {"type": "refusal", "refusal": "no"}]}]
    m = repair_call_ids(mixed)
    assert m[0]["content"][0]["type"] == "output_text", m[0]
    assert m[0]["content"][1] == {"type": "refusal", "refusal": "no"}, m[0]

    # Everything this module emits must satisfy the SDK's own request schema.
    assert validate_items(r) == [], validate_items(r)

    # ---------------------------------------------------------------- sweep
    # The unit checks above use hand-written items. This runs the REAL
    # formatter over the message shapes these agents actually produce and
    # audits the result, so a fourth bug shows up here instead of as a 400
    # three restarts later.
    import asyncio
    from agentscope.message import (TextBlock, ToolCallBlock, ToolResultBlock,
                                    ThinkingBlock, AssistantMsg, SystemMsg,
                                    UserMsg)

    def _tc(fc, call, name="t"):
        b = ToolCallBlock(type="tool_call", id=fc, name=name, input="{}")
        try:
            b["call_id"] = call
        except Exception:
            setattr(b, "call_id", call)
        return b

    def _tr(fc, out="{}", name="t"):
        return ToolResultBlock(type="tool_result", id=fc, name=name, output=out)

    _SYS = SystemMsg(name="system", content=[TextBlock(type="text", text="P")])

    _LONG = "URL: x" + chr(10) + ('cell "v"' + chr(10)) * 50

    def _u(t):
        return UserMsg(name="user", content=[TextBlock(type="text", text=t)])

    scenarios = {
        "plain Q&A": [_SYS, _u("hi")],
        "one tool call": [_SYS, _u("q"), AssistantMsg(
            name="a", content=[_tc("fc_1", "call_1"), _tr("fc_1")])],
        "tool call then text": [_SYS, _u("q"), AssistantMsg(
            name="a", content=[_tc("fc_1", "call_1"), _tr("fc_1"),
                               TextBlock(type="text", text="answer")]),
            _u("follow up")],
        "two tools in one turn": [_SYS, _u("q"), AssistantMsg(
            name="a", content=[_tc("fc_1", "call_1", "a"), _tr("fc_1", "1", "a"),
                               _tc("fc_2", "call_2", "b"), _tr("fc_2", "2", "b")])],
        "three turns of tools": [_SYS, _u("q"), AssistantMsg(
            name="a", content=[_tc("fc_1", "call_1"), _tr("fc_1", "1"),
                               TextBlock(type="text", text="r1")]), _u("more"),
            AssistantMsg(name="a", content=[_tc("fc_2", "call_2"),
                                            _tr("fc_2", "2"),
                                            TextBlock(type="text", text="r2")]),
            _u("again")],
        "with a thinking block": [_SYS, _u("q"), AssistantMsg(
            name="a", content=[ThinkingBlock(type="thinking", thinking="hmm"),
                               _tc("fc_1", "call_1"), _tr("fc_1")])],
        "failed tool result": [_SYS, _u("q"), AssistantMsg(
            name="a", content=[_tc("fc_1", "call_1"),
                               _tr("fc_1", "Error: element not found")])],
        "long multiline output": [_SYS, _u("q"), AssistantMsg(
            name="a", content=[_tc("fc_1", "call_1"),
                               _tr("fc_1", _LONG)])],
    }

    def _audit(items):
        probs = []
        has_reasoning = any(isinstance(x, dict) and x.get("type") == "reasoning"
                            for x in items)
        for i, m in enumerate(items):
            if not isinstance(m, dict):
                continue
            if m.get("type") == "function_call" and "id" in m and not has_reasoning:
                probs.append(f"[{i}] function_call id echoed with no reasoning")
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for part in m["content"]:
                    if isinstance(part, dict) and part.get("type") == "input_text":
                        probs.append(f"[{i}] input_text on an assistant message")
        calls = {m["call_id"] for m in items
                 if isinstance(m, dict) and m.get("type") == "function_call"}
        outs = {m["call_id"] for m in items
                if isinstance(m, dict) and m.get("type") == "function_call_output"}
        probs += [f"call {c} has no output" for c in calls - outs]
        probs += [f"output {o} has no call" for o in outs - calls]
        probs += [f"[{i}] schema: {why}" for i, why in validate_items(items)]
        return probs

    fmt = PatchedOpenAIResponseFormatter()
    for _name, _msgs in scenarios.items():
        _items = asyncio.run(fmt.format(_msgs))
        _probs = _audit(_items)
        assert not _probs, f"{_name}: {_probs}"
        # The stock formatter must still be broken on the tool scenarios --
        # otherwise this module is dead weight and should be deleted.
        if any(isinstance(b, ToolCallBlock)
               for m in _msgs for b in (m.content if isinstance(m.content, list) else [])):
            _stock = _audit(asyncio.run(OpenAIResponseFormatter().format(_msgs)))
            assert _stock, f"{_name}: stock formatter is clean -- drop this shim"
    print(f"  swept {len(scenarios)} real message shapes, all clean")

    print("response_formatter_fix: all checks passed")
