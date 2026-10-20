"""Reasoning models need /v1/responses, not /v1/chat/completions.

gpt-6-luna has reasoning ON by default server-side, and chat/completions
refuses function tools while it is:

    Function tools with reasoning_effort are not supported for gpt-6-luna in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

We never send reasoning_effort -- AgentScope only sends it when you set BOTH
thinking_enable and reasoning_effort, and we set neither. So the fix is the
endpoint, not the parameter.

    .venv/Scripts/python.exe test_model_endpoint.py
"""
import importlib
import os
import sys

FAILS = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}"
          + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILS.append(label)


print("OPENAI_RESPONSES picks the endpoint")

for value, want in (("", False), ("1", True), ("true", True),
                    ("yes", True), ("0", False), ("no", False)):
    os.environ["OPENAI_RESPONSES"] = value
    import config
    importlib.reload(config)
    check(f"OPENAI_RESPONSES={value!r}",
          config.LLM_PROFILES["openai"]["responses"], want)

# No "unset" case here: config re-reads .env on every import, so on a machine
# whose .env sets the flag there is no way to simulate unset without stubbing
# a loader the reload re-creates. The empty-string case above covers the
# property that matters -- anything that is not 1/true/yes leaves gpt-4o on
# chat/completions.

# The flag is only useful if the key the builders read is the key config
# writes. A typo on either side fails silently as "no change at all", which
# is indistinguishable from the bug it is meant to fix.
import multi_agent.swarm                                           # noqa: E402
src = open("multi_agent/swarm.py", encoding="utf-8").read()
check("swarm reads the same key", 'prof.get("responses")' in src, True)
src2 = open("task_agent.py", encoding="utf-8").read()
check("task_agent reads the same key", 'prof.get("responses")' in src2, True)

from agentscope.model import OpenAIChatModel, OpenAIResponseModel  # noqa: E402
# Same constructor shape, or swapping the class breaks the call site.
for arg in ("credential", "model", "stream"):
    import inspect
    check(f"both accept {arg}",
          arg in inspect.signature(OpenAIResponseModel.__init__).parameters
          and arg in inspect.signature(OpenAIChatModel.__init__).parameters,
          True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
    sys.exit(1)
print("all endpoint checks passed")
