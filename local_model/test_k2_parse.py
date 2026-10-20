"""Fast, deterministic tests for the K2-Horizon markup parser -- NO model load.

parse_output() is the riskiest part of the local backend: everything the agent
sees (visible text, reasoning, tool calls) depends on correctly splitting IFM's
chat-template markup. These tests pin the literal wire formats from
chat_template.jinja so a template change fails loudly here instead of silently
breaking tool use. They import only stdlib -- no torch, no transformers.

Run:  agent/.venv/Scripts/python.exe -m local_model.test_k2_parse
"""

from __future__ import annotations

import json
import sys

from local_model.k2_runtime import (
    _normalise_message,
    _normalise_tools,
    parse_output,
)

_FAILURES: list[str] = []


def check(label: str, got, want) -> None:
    if got != want:
        _FAILURES.append(f"{label}\n    got : {got!r}\n    want: {want!r}")


def main() -> int:
    # --- plain answer, nothing to strip ---------------------------------
    g = parse_output("Hello there.<|ifm|im_end|>")
    check("plain content", g.content, "Hello there.")
    check("plain no reasoning", g.reasoning, "")
    check("plain no calls", g.tool_calls, [])

    # --- reasoning blocks, all three efforts -----------------------------
    g = parse_output("<ifm|think>weighing it up</ifm|think>The answer is 4.")
    check("think content", g.content, "The answer is 4.")
    check("think reasoning", g.reasoning, "weighing it up")

    g = parse_output("<ifm|think_faster>quick</ifm|think_faster>Done.")
    check("think_faster content", g.content, "Done.")
    check("think_faster reasoning", g.reasoning, "quick")

    # Budget ran out mid-reasoning: keep it as reasoning, not as the answer.
    g = parse_output("<ifm|think>ran out of tok")
    check("unterminated content", g.content, "")
    check("unterminated reasoning", g.reasoning, "ran out of tok")

    # --- the shape the model ACTUALLY produces ---------------------------
    # The generation prompt ends with an open <ifm|think_faster>, so output
    # starts inside reasoning and only the CLOSING tag is generated.
    g = parse_output("user wants a listing</ifm|think_faster>\nHere they are.",
                     starts_in_think=True)
    check("open-think reasoning", g.reasoning, "user wants a listing")
    check("open-think content", g.content, "Here they are.")

    # Same, but the turn is a tool call instead of prose.
    raw = (
        "need to list files</ifm|think_faster>"
        "<ifm|tool_calls><ifm|tool_call>ls\n"
        "<ifm|arg_key>path</ifm|arg_key><ifm|arg_value>.</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )
    g = parse_output(raw, starts_in_think=True)
    check("open-think call reasoning", g.reasoning, "need to list files")
    check("open-think call count", len(g.tool_calls), 1)
    if g.tool_calls:
        check("open-think call name", g.tool_calls[0]["function"]["name"], "ls")
    check("open-think call content", g.content, "")

    # Budget exhausted before the model closed the block: all reasoning.
    g = parse_output("still thinking and never finished", starts_in_think=True)
    check("open-think truncated reasoning", g.reasoning,
          "still thinking and never finished")
    check("open-think truncated content", g.content, "")

    # High effort uses <ifm|think> rather than <ifm|think_faster>.
    g = parse_output("pondered</ifm|think>Answer.", starts_in_think=True)
    check("open-think high effort", g.reasoning, "pondered")
    check("open-think high content", g.content, "Answer.")

    # --- json tool call --------------------------------------------------
    raw = (
        '<ifm|tool_calls><ifm|tool_call>'
        '{"name": "read_file", "arguments": {"path": "a.txt", "lines": 10}}'
        '</ifm|tool_call></ifm|tool_calls>'
    )
    g = parse_output(raw)
    check("json call count", len(g.tool_calls), 1)
    if g.tool_calls:
        fn = g.tool_calls[0]["function"]
        check("json call name", fn["name"], "read_file")
        check("json call args", json.loads(fn["arguments"]),
              {"path": "a.txt", "lines": 10})
        check("json call type", g.tool_calls[0]["type"], "function")
        check("json call id prefix", g.tool_calls[0]["id"][:5], "call_")
    check("json call strips markup", g.content, "")

    # --- two calls in one block -----------------------------------------
    raw = (
        '<ifm|tool_calls>'
        '<ifm|tool_call>{"name": "a", "arguments": {}}</ifm|tool_call>'
        '<ifm|tool_call>{"name": "b", "arguments": {"x": 1}}</ifm|tool_call>'
        '</ifm|tool_calls>'
    )
    g = parse_output(raw)
    check("multi call count", len(g.tool_calls), 2)
    check("multi call names", [c["function"]["name"] for c in g.tool_calls],
          ["a", "b"])

    # --- reasoning AND a call AND trailing prose -------------------------
    raw = (
        "<ifm|think>need the file</ifm|think>"
        '<ifm|tool_calls><ifm|tool_call>{"name":"ls","arguments":{}}'
        "</ifm|tool_call></ifm|tool_calls>"
        "Checking now."
    )
    g = parse_output(raw)
    check("combo reasoning", g.reasoning, "need the file")
    check("combo content", g.content, "Checking now.")
    check("combo calls", len(g.tool_calls), 1)

    # --- xml format, untyped ---------------------------------------------
    raw = (
        "<ifm|tool_calls><ifm|tool_call>search\n"
        "<ifm|arg_key>query</ifm|arg_key>"
        "<ifm|arg_value>red shoes</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )
    g = parse_output(raw)
    check("xml call count", len(g.tool_calls), 1)
    if g.tool_calls:
        fn = g.tool_calls[0]["function"]
        check("xml call name", fn["name"], "search")
        check("xml call args", json.loads(fn["arguments"]), {"query": "red shoes"})

    # --- xml_typed coerces via arg_type ----------------------------------
    raw = (
        "<ifm|tool_calls><ifm|tool_call>page\n"
        "<ifm|arg_key>n</ifm|arg_key><ifm|arg_type>int</ifm|arg_type>"
        "<ifm|arg_value>7</ifm|arg_value>"
        "<ifm|arg_key>deep</ifm|arg_key><ifm|arg_type>bool</ifm|arg_type>"
        "<ifm|arg_value>true</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )
    g = parse_output(raw)
    if g.tool_calls:
        args = json.loads(g.tool_calls[0]["function"]["arguments"])
        check("xml_typed int", args.get("n"), 7)
        check("xml_typed bool", args.get("deep"), True)

    # --- malformed json call is dropped, not crashed on ------------------
    g = parse_output("<ifm|tool_calls><ifm|tool_call>{bad json"
                     "</ifm|tool_call></ifm|tool_calls>tail")
    check("malformed dropped", g.tool_calls, [])
    check("malformed keeps tail", g.content, "tail")

    # --- message normalisation -------------------------------------------
    m = _normalise_message({
        "role": "user",
        "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
    })
    check("content blocks flattened", m["content"], "one\ntwo")

    m = _normalise_message({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "f", "arguments": '{"k": "v"}'},
        }],
    })
    check("args string -> dict",
          m["tool_calls"][0]["function"]["arguments"], {"k": "v"})
    check("null content -> empty str", m["content"], "")

    # --- assistant turns MUST carry a thinking field ----------------------
    # The template raises without one, and the ReAct loop replays assistant
    # turns on every step, so this guards every multi-turn tool conversation.
    m = _normalise_message({"role": "assistant", "content": "hi"}, "high")
    check("assistant gets think (high)", m.get("think"), "")
    m = _normalise_message({"role": "assistant", "content": "hi"}, "medium")
    check("assistant gets think_fast (medium)", m.get("think_fast"), "")
    m = _normalise_message({"role": "assistant", "content": "hi"}, "low")
    check("assistant gets think_faster (low)", m.get("think_faster"), "")

    # Real reasoning echoed back by the caller is preserved, not overwritten.
    m = _normalise_message(
        {"role": "assistant", "content": "hi", "reasoning_content": "because"},
        "high")
    check("echoed reasoning kept", m.get("reasoning_content"), "because")
    check("no duplicate think key", "think" in m, False)

    # A null thinking field trips a DIFFERENT template raise, so it must be
    # dropped and replaced rather than passed through.
    m = _normalise_message(
        {"role": "assistant", "content": "hi", "reasoning_content": None},
        "high")
    check("null reasoning dropped", m.get("reasoning_content"), None)
    check("null reasoning replaced", m.get("think"), "")

    # Non-assistant roles must NOT get a thinking field.
    m = _normalise_message({"role": "user", "content": "hi"}, "high")
    check("user has no think", any(k in m for k in
                                   ("think", "think_fast", "think_faster")), False)

    m = _normalise_message({"role": "tool", "tool_call_id": "call_1",
                            "content": "result"})
    check("tool role kept", m["role"], "tool")
    check("tool_call_id kept", m["tool_call_id"], "call_1")

    # --- tool schema normalisation ---------------------------------------
    wrapped = {"type": "function", "function": {"name": "f", "parameters": {}}}
    check("wrapped tool passthrough", _normalise_tools([wrapped]), [wrapped])
    check("bare tool wrapped", _normalise_tools([{"name": "f"}]),
          [{"type": "function", "function": {"name": "f"}}])
    check("no tools -> empty", _normalise_tools(None), [])

    # --- schema sanitising (the template hard-raises on these) -----------
    def params(tool_list):
        return tool_list[0]["function"]["parameters"]

    # required names a property that was never defined -> drop that name.
    got = _normalise_tools([{
        "name": "f",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", "ghost"],
        },
    }])
    check("drops undefined required", params(got)["required"], ["a"])

    # required with no properties at all -> drop the key entirely.
    got = _normalise_tools([{
        "name": "f",
        "parameters": {"type": "object", "required": ["x"]},
    }])
    check("drops orphan required", "required" in params(got), False)

    # required that isn't a list -> drop it.
    got = _normalise_tools([{
        "name": "f",
        "parameters": {"type": "object", "properties": {"a": {}},
                       "required": "a"},
    }])
    check("drops non-list required", "required" in params(got), False)

    # nested objects are sanitised too.
    got = _normalise_tools([{
        "name": "f",
        "parameters": {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"b": {"type": "string"}},
                    "required": ["b", "missing"],
                },
            },
        },
    }])
    check("sanitises nested", params(got)["properties"]["outer"]["required"],
          ["b"])

    # array items are sanitised too.
    got = _normalise_tools([{
        "name": "f",
        "parameters": {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"c": {}},
                              "required": ["c", "nope"]},
                },
            },
        },
    }])
    check("sanitises array items",
          params(got)["properties"]["rows"]["items"]["required"], ["c"])

    # a valid schema is left exactly as it was.
    clean = {"type": "object", "properties": {"a": {"type": "string"}},
             "required": ["a"]}
    got = _normalise_tools([{"name": "f", "parameters": clean}])
    check("valid schema untouched", params(got), clean)

    # --- report -----------------------------------------------------------
    if _FAILURES:
        print(f"FAIL ({len(_FAILURES)}):", file=sys.stderr)
        for f in _FAILURES:
            print("  - " + f, file=sys.stderr)
        return 1
    print("all K2 parser tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
