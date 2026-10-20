import { useCallback, useRef, useState } from "react";
import { streamChat, type ServerEvent } from "./stream";

export type Activity =
  | {
      type: "delegation";
      id: string;
      agent: string;
      child: string;
      description: string;
    }
  | {
      type: "tool";
      id: string;
      agent: string;
      name: string;
      args: string;
      status: "running" | "ok" | "fail";
      ms?: number;
      preview?: string;
    }
  | {
      type: "report";
      id: string;
      agent: string;
      status: "done" | "failed";
      report: string;
      tools: string[];
    }
  | { type: "narration"; id: string; agent: string; text: string };

export interface Turn {
  id: string;
  role: "user" | "assistant";
  text: string;
  activities: Activity[];
  status: "streaming" | "done" | "error";
  startedAt: number;
  endedAt?: number;
}

let counter = 0;
const nextId = (prefix: string) => `${prefix}-${Date.now()}-${counter++}`;

export function useChat(getSettings: () => { llm: string; mode: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // One narration bubble per agent per reasoning step, so streamed text deltas
  // append to a paragraph instead of creating an entry per token.
  const narrationRef = useRef<Record<string, string>>({});

  const patch = useCallback(
    (turnId: string, fn: (turn: Turn) => Turn) => {
      setTurns((prev) =>
        prev.map((t) => (t.id === turnId ? fn(t) : t))
      );
    },
    []
  );

  const addActivity = useCallback(
    (turnId: string, activity: Activity) => {
      patch(turnId, (t) => ({ ...t, activities: [...t.activities, activity] }));
    },
    [patch]
  );

  const appendNarration = useCallback(
    (turnId: string, agent: string, text: string) => {
      const key = agent;
      const existing = narrationRef.current[key];
      if (existing) {
        patch(turnId, (t) => ({
          ...t,
          activities: t.activities.map((a) =>
            a.id === existing && a.type === "narration"
              ? { ...a, text: a.text + text }
              : a
          ),
        }));
        return;
      }
      const id = nextId("narr");
      narrationRef.current[key] = id;
      patch(turnId, (t) => ({
        ...t,
        activities: [...t.activities, { type: "narration", id, agent, text }],
      }));
    },
    [patch]
  );

  const handle = useCallback(
    (turnId: string, ev: ServerEvent) => {
      switch (ev.kind) {
        case "step_end":
          // Close this agent's open narration paragraph.
          delete narrationRef.current[ev.agent];
          break;

        case "text":
        case "thinking":
          appendNarration(turnId, ev.agent, ev.text);
          break;

        case "delegation":
          addActivity(turnId, {
            type: "delegation",
            id: ev.task_id || nextId("del"),
            agent: ev.agent,
            child: ev.child,
            description: ev.description,
          });
          break;

        case "tool_start":
          addActivity(turnId, {
            type: "tool",
            id: ev.id || nextId("tool"),
            agent: ev.agent,
            name: ev.name,
            args: ev.args,
            status: "running",
          });
          break;

        case "tool_end": {
          // A tool that returns an error STRING while raising nothing is this
          // project's most common failure, so trust the preview, not `ok`.
          const failed =
            !ev.ok || /^\s*(error|failed|could not|unable)\b/i.test(ev.preview ?? "");
          patch(turnId, (t) => ({
            ...t,
            activities: t.activities.map((a) =>
              a.id === ev.id && a.type === "tool"
                ? {
                    ...a,
                    status: failed ? "fail" : "ok",
                    ms: ev.ms,
                    preview: ev.preview,
                  }
                : a
            ),
          }));
          break;
        }

        case "report":
          addActivity(turnId, {
            type: "report",
            id: nextId("rep"),
            agent: ev.agent,
            status: ev.status,
            report: ev.report,
            tools: ev.tools_used ?? [],
          });
          break;

        case "done":
          patch(turnId, (t) => ({
            ...t,
            text: ev.reply,
            status: "done",
            endedAt: Date.now(),
          }));
          break;

        case "error":
          patch(turnId, (t) => ({
            ...t,
            text: ev.message,
            status: "error",
            endedAt: Date.now(),
          }));
          break;
      }
    },
    [addActivity, appendNarration, patch]
  );

  const send = useCallback(
    async (message: string) => {
      const trimmed = message.trim();
      if (!trimmed || busy) return;

      const { llm, mode } = getSettings();
      narrationRef.current = {};
      const assistantId = nextId("a");

      setTurns((prev) => [
        ...prev,
        {
          id: nextId("u"),
          role: "user",
          text: trimmed,
          activities: [],
          status: "done",
          startedAt: Date.now(),
        },
        {
          id: assistantId,
          role: "assistant",
          text: "",
          activities: [],
          status: "streaming",
          startedAt: Date.now(),
        },
      ]);

      setBusy(true);
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamChat({
          message: trimmed,
          llm,
          mode,
          signal: controller.signal,
          onEvent: (ev) => handle(assistantId, ev),
        });
        // A stream that closes without a terminal event (server restart, or
        // the user hitting stop) must not leave a turn spinning forever.
        patch(assistantId, (t) =>
          t.status === "streaming"
            ? {
                ...t,
                status: t.text ? "done" : "error",
                text: t.text || "The run stopped before it produced an answer.",
                endedAt: Date.now(),
              }
            : t
        );
      } catch (err) {
        const aborted = controller.signal.aborted;
        patch(assistantId, (t) => ({
          ...t,
          status: aborted ? "done" : "error",
          text: aborted
            ? t.text || "Stopped."
            : `Could not reach the agent: ${
                err instanceof Error ? err.message : String(err)
              }`,
          endedAt: Date.now(),
        }));
      } finally {
        setBusy(false);
        abortRef.current = null;
      }
    },
    [busy, getSettings, handle, patch]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  // "New chat" has to END the conversation on the SERVER, not just blank the
  // screen. Clearing only the local turns left the swarm's board, orchestrator
  // and children holding the previous conversation forever -- and, worse, the
  // archive path never ran, so nothing was ever written to the session store
  // and the memory system never learned anything at all. This endpoint
  // archives the transcript, fires fact extraction, and resets the swarm.
  const clear = useCallback(async () => {
    setTurns([]);
    try {
      await fetch("/api/agent/new-session", { method: "POST" });
    } catch {
      // The screen is already cleared; a failed reset must not strand the UI.
      // The next turn simply continues the old server-side conversation.
    }
  }, []);

  return { turns, busy, send, stop, clear };
}
