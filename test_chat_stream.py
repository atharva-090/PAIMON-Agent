"""End-to-end test of POST /api/agent/chat/stream, with a stubbed turn so it
costs nothing. Proves the SSE frames a real run would produce."""
import asyncio
import json
import sys

sys.path.insert(0, r"E:\KIRO projects\Open AI mocker (hermes agent try)\agent")

import agent_events as E  # noqa: E402
import multi_agent.swarm as swarm  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402


async def fake_turn(text, profile=None):
    """Stands in for a swarm run: publishes the same events, returns a reply."""
    E.publish("delegation", "orchestrator", task_id="t1",
              child="browser_agent", description="research term loans")
    E.publish("tool_start", "browser_agent", id="c1", name="browse_open",
              args='{"url": "https://google.com"}')
    await asyncio.sleep(0.01)
    E.publish("tool_end", "browser_agent", id="c1", name="browse_open",
              ok=True, ms=1200, preview="URL: https://google.com")
    E.publish("text", "orchestrator", text="Handing off to the browser.")
    E.publish("report", "browser_agent", task_id="t1", status="done",
              report="Found 6 sources on term loans.",
              tools_used=["browse_open", "browse_snapshot"])
    return "Here is the summary of term loans."


swarm.server_swarm_chat = fake_turn

client = TestClient(server.app)
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
    print(("   ok   " if cond else "   FAIL ") + msg)


print("--- POST /api/agent/chat/stream ---")
with client.stream("POST", "/api/agent/chat/stream",
                   json={"message": "research term loans", "mode": "swarm",
                         "llm": "openai"}) as r:
    check(r.status_code == 200, f"status 200 (got {r.status_code})")
    check("text/event-stream" in r.headers.get("content-type", ""),
          "content-type is text/event-stream")
    events = []
    for line in r.iter_lines():
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))

kinds = [e["kind"] for e in events]
print("   frames:", kinds)
check(kinds[0] == "start", "first frame is 'start'")
check(kinds[-1] == "done", "last frame is 'done'")
check("delegation" in kinds, "the delegation was streamed")
check("tool_start" in kinds and "tool_end" in kinds, "tool bracket streamed")
check("report" in kinds, "the child's report was streamed")

done = events[-1]
check(done["reply"] == "Here is the summary of term loans.",
      "the terminal frame carries the same reply the plain endpoint returns")

deleg = next(e for e in events if e["kind"] == "delegation")
check(deleg["child"] == "browser_agent", "delegation names the child")

end = next(e for e in events if e["kind"] == "tool_end")
check(end["ms"] == 1200 and end["ok"] is True, "tool_end carries timing + ok")

print("\n--- an agent-side failure arrives as an event, not a 500 ---")


async def boom(text, profile=None):
    raise RuntimeError("mocker is not running")


swarm.server_swarm_chat = boom
with client.stream("POST", "/api/agent/chat/stream",
                   json={"message": "hi", "mode": "swarm"}) as r:
    check(r.status_code == 200, "still HTTP 200 (the stream had already opened)")
    evs = [json.loads(l[5:].strip()) for l in r.iter_lines()
           if l.startswith("data:")]
print("   frames:", [e["kind"] for e in evs])
check(evs[-1]["kind"] == "error", "terminal frame is 'error'")
check("mocker is not running" in evs[-1]["message"],
      "the real reason is passed through")

print("\n--- /chat/ serves the built React app ---")
r = client.get("/chat/")
check(r.status_code == 200, f"/chat/ -> 200 (got {r.status_code})")
check("<div id=\"root\">" in r.text, "the SPA root element is served")
check("/chat/assets/" in r.text, "asset URLs are rooted at /chat/")

r = client.get("/chat", follow_redirects=False)
check(r.status_code in (301, 302, 307), "bare /chat redirects to /chat/")

print("\n" + "=" * 60)
if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
