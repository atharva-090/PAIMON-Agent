/** Client for POST /api/agent/chat/stream.
 *
 *  The endpoint speaks server-sent events but over a POST body, so EventSource
 *  (GET-only) is out and we read the response stream by hand. That is a few
 *  more lines here and avoids putting the user's message in a query string.
 */

export type ServerEvent =
  | { kind: "start"; mode: string; llm: string }
  | { kind: "step_start"; agent: string }
  | { kind: "step_end"; agent: string }
  | { kind: "text"; agent: string; text: string }
  | { kind: "thinking"; agent: string; text: string }
  | { kind: "tool_call"; agent: string; id: string; name: string; args: string }
  | { kind: "tool_start"; agent: string; id: string; name: string; args: string }
  | {
      kind: "tool_end";
      agent: string;
      id: string;
      name: string;
      ok: boolean;
      ms: number;
      preview: string;
    }
  | {
      kind: "delegation";
      agent: string;
      task_id: string;
      child: string;
      description: string;
    }
  | {
      kind: "report";
      agent: string;
      task_id: string;
      status: "done" | "failed";
      report: string;
      tools_used: string[];
    }
  | { kind: "done"; reply: string }
  | { kind: "error"; message: string };

export interface StreamOptions {
  message: string;
  llm: string;
  mode: string;
  signal?: AbortSignal;
  onEvent: (event: ServerEvent) => void;
}

/** Runs one turn, calling `onEvent` as the server publishes. Resolves when the
 *  stream closes. Throws only on transport failure -- an agent-side failure
 *  arrives as an `error` event so the UI can render it in place. */
export async function streamChat({
  message,
  llm,
  mode,
  signal,
  onEvent,
}: StreamOptions): Promise<void> {
  const res = await fetch("/api/agent/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, llm, mode }),
    signal,
  });

  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* a non-JSON error body is not worth a second failure */
    }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. Keep the trailing partial
    // frame in the buffer for the next chunk.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue; // ": keepalive" and friends
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          onEvent(JSON.parse(payload) as ServerEvent);
        } catch {
          /* a malformed frame must not kill the rest of the stream */
        }
      }
    }
  }
}
