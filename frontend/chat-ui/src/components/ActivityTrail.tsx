import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ChevronRight,
  CornerDownRight,
  Check,
  X,
  Loader2,
  Sparkles,
} from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import { agentMeta, toolLabel } from "@/lib/agents";
import { cn } from "@/lib/utils";
import type { Activity } from "@/lib/useChat";

function elapsed(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

function AgentChip({ name }: { name: string }) {
  const meta = agentMeta(name);
  const Icon = meta.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10.5px] font-medium ring-1 ring-inset",
        meta.chip,
        meta.tone
      )}
    >
      <Icon className="size-2.5" />
      {meta.label}
    </span>
  );
}

function Row({ activity }: { activity: Activity }) {
  const meta = agentMeta(activity.agent);

  if (activity.type === "delegation") {
    const child = agentMeta(activity.child);
    const ChildIcon = child.icon;
    return (
      <div className="flex items-start gap-2 py-1">
        <CornerDownRight className="mt-0.5 size-3 shrink-0 text-muted-foreground/60" />
        <div className="min-w-0 flex-1">
          <span className="text-muted-foreground">delegated to </span>
          <span className={cn("inline-flex items-center gap-1 font-medium", child.tone)}>
            <ChildIcon className="size-3" />
            {child.label}
          </span>
          <p className="mt-0.5 line-clamp-2 text-[12px] leading-snug text-muted-foreground/80">
            {activity.description}
          </p>
        </div>
      </div>
    );
  }

  if (activity.type === "tool") {
    const running = activity.status === "running";
    const failed = activity.status === "fail";
    return (
      <div className="flex items-start gap-2 py-[3px]">
        <span className="mt-[3px] shrink-0">
          {running ? (
            <Loader2 className="size-3 animate-spin text-primary" />
          ) : failed ? (
            <X className="size-3 text-destructive" />
          ) : (
            <Check className="size-3 text-[var(--success)]" />
          )}
        </span>
        {/* break-words so an unbroken JSON blob or long URL wraps instead of
            being clipped by the pane's overflow-x-hidden. */}
        <div className="min-w-0 flex-1 break-words">
          <span className={cn("font-medium", meta.tone)}>
            {toolLabel(activity.name)}
          </span>
          {activity.args && (
            <span className="ml-1.5 font-mono text-[11px] text-muted-foreground/70">
              {activity.args}
            </span>
          )}
          {activity.ms !== undefined && activity.ms > 900 && (
            <span className="ml-1.5 text-[10.5px] tabular-nums text-muted-foreground/50">
              {elapsed(activity.ms)}
            </span>
          )}
          {failed && activity.preview && (
            <p className="mt-0.5 line-clamp-2 text-[11.5px] leading-snug text-destructive/85">
              {activity.preview}
            </p>
          )}
        </div>
      </div>
    );
  }

  if (activity.type === "report") {
    const failed = activity.status === "failed";
    return (
      <div className="my-1 rounded-md border border-border/60 bg-card/50 px-2.5 py-2">
        <div className="mb-1 flex items-center gap-1.5">
          <AgentChip name={activity.agent} />
          <Badge variant={failed ? "destructive" : "success"}>
            {failed ? "failed" : "reported"}
          </Badge>
          {activity.tools.length > 0 && (
            <span className="text-[10.5px] text-muted-foreground/60">
              {activity.tools.length} tool
              {activity.tools.length === 1 ? "" : "s"}
            </span>
          )}
        </div>
        <p className="line-clamp-4 whitespace-pre-wrap text-[12px] leading-snug text-muted-foreground/85">
          {activity.report}
        </p>
      </div>
    );
  }

  // narration
  if (!activity.text.trim()) return null;
  return (
    <div className="flex items-start gap-2 py-1">
      <Sparkles className="mt-[3px] size-3 shrink-0 text-muted-foreground/45" />
      <p className="min-w-0 flex-1 whitespace-pre-wrap text-[12px] leading-relaxed text-muted-foreground/80">
        {activity.text}
      </p>
    </div>
  );
}

export function ActivityTrail({
  activities,
  streaming,
  startedAt,
  endedAt,
}: {
  activities: Activity[];
  streaming: boolean;
  startedAt: number;
  endedAt?: number;
}) {
  // Open while the run is live, then fold away on its own so the finished
  // answer is what remains on screen.
  const [open, setOpen] = useState(streaming);
  const [now, setNow] = useState(Date.now());
  const scrollRef = useRef<HTMLDivElement>(null);
  const userToggled = useRef(false);

  useEffect(() => {
    if (!streaming) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [streaming]);

  useEffect(() => {
    if (!userToggled.current) setOpen(streaming);
  }, [streaming]);

  useEffect(() => {
    if (open && streaming && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [activities.length, open, streaming]);

  const stats = useMemo(() => {
    const tools = activities.filter((a) => a.type === "tool");
    const agents = new Set(activities.map((a) => a.agent));
    return {
      tools: tools.length,
      failed: tools.filter((a) => a.type === "tool" && a.status === "fail").length,
      agents: agents.size,
    };
  }, [activities]);

  const current = activities.length
    ? activities[activities.length - 1].agent
    : "orchestrator";
  const duration = (endedAt ?? now) - startedAt;

  if (!activities.length && !streaming) return null;

  return (
    <Collapsible
      open={open}
      onOpenChange={(v) => {
        userToggled.current = true;
        setOpen(v);
      }}
      className="mb-2"
    >
      <CollapsibleTrigger className="group flex w-full items-center gap-2 rounded-md px-1 py-1 text-left text-xs text-muted-foreground transition-colors hover:text-foreground">
        <ChevronRight
          className={cn(
            "size-3.5 shrink-0 transition-transform duration-200",
            open && "rotate-90"
          )}
        />
        {streaming ? (
          <>
            <span className="relative flex size-1.5 shrink-0">
              <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-70" />
              <span className="relative inline-flex size-1.5 rounded-full bg-primary" />
            </span>
            <span className="font-medium text-foreground/90">
              {agentMeta(current).label}
            </span>
            <span className="text-muted-foreground/70">working</span>
            <span className="tabular-nums text-muted-foreground/50">
              {elapsed(duration)}
            </span>
            <span className="ml-1 h-px flex-1 shimmer" />
          </>
        ) : (
          <>
            <span className="font-medium text-foreground/80">
              Worked for {elapsed(duration)}
            </span>
            <span className="text-muted-foreground/60">
              · {stats.agents} agent{stats.agents === 1 ? "" : "s"} ·{" "}
              {stats.tools} tool call{stats.tools === 1 ? "" : "s"}
            </span>
            {stats.failed > 0 && (
              <Badge variant="destructive">{stats.failed} failed</Badge>
            )}
          </>
        )}
      </CollapsibleTrigger>

      <CollapsibleContent
        className={cn(
          "overflow-hidden",
          "data-[state=closed]:animate-out data-[state=closed]:fade-out-0",
          "data-[state=open]:animate-in data-[state=open]:fade-in-0"
        )}
      >
        <div className="relative mt-1 ml-[6px] border-l border-border/70 pl-3.5">
          <div
            ref={scrollRef}
            className="scrollbar-slim max-h-[19rem] overflow-y-auto overflow-x-hidden pr-2 text-xs"
          >
            <AnimatePresence initial={false}>
              {activities.map((a) => (
                <motion.div
                  key={a.id}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18, ease: "easeOut" }}
                >
                  <Row activity={a} />
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
