import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowDown, Settings2, SquarePen, Layers, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Composer } from "@/components/Composer";
import { Message } from "@/components/Message";
import { useChat } from "@/lib/useChat";
import { cn } from "@/lib/utils";

// This surface is deliberately opinionated: the real API and the full swarm.
// The task board at / keeps the switches; this page is for talking to it.
const LLM = "openai";
const MODE = "swarm";

const SUGGESTIONS = [
  "Research cash credit loans, write a Word summary, and build an approval-rules workbook",
  "What's on the task board that's overdue?",
  "Find the latest report I generated and email it to Anand",
];

export default function App() {
  const [draft, setDraft] = useState("");
  const [atBottom, setAtBottom] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const getSettings = useCallback(() => ({ llm: LLM, mode: MODE }), []);
  const { turns, busy, send, stop, clear } = useChat(getSettings);

  const scrollToBottom = useCallback((smooth = true) => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({
      top: el.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  }, []);

  // Follow the stream, but stop following the moment the user scrolls up --
  // yanking someone back to the bottom while they are reading is hostile.
  useEffect(() => {
    if (atBottom) scrollToBottom(false);
  }, [turns, atBottom, scrollToBottom]);

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
  }, []);

  const submit = useCallback(() => {
    const text = draft;
    setDraft("");
    setAtBottom(true);
    void send(text);
  }, [draft, send]);

  const empty = turns.length === 0;

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex h-full flex-col bg-background">
        {/* ---- header ---------------------------------------------------- */}
        <header className="z-20 flex h-12 shrink-0 items-center gap-2 border-b border-border/60 bg-background/80 px-3 backdrop-blur-xl">
          <div className="flex items-center gap-2">
            <div className="grid size-6 place-items-center rounded-md bg-primary/15 ring-1 ring-inset ring-primary/25">
              <span className="text-[11px] font-bold text-primary">P</span>
            </div>
            <span className="text-sm font-semibold tracking-tight">PAIMON</span>
          </div>

          <div className="ml-1 flex items-center gap-1">
            <Pill icon={Layers} text="swarm" />
            <Pill icon={Cpu} text="openai" />
          </div>

          <div className="ml-auto flex items-center gap-1">
            {!empty && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => void clear()}
                    disabled={busy}
                    aria-label="New chat"
                    className="text-muted-foreground"
                  >
                    <SquarePen className="size-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>New chat</TooltipContent>
              </Tooltip>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  asChild
                  className="text-muted-foreground"
                >
                  <a href="/" aria-label="Board and settings">
                    <Settings2 className="size-4" />
                  </a>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Board &amp; settings</TooltipContent>
            </Tooltip>
          </div>
        </header>

        {/* ---- conversation ---------------------------------------------- */}
        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="scrollbar-slim relative flex-1 overflow-y-auto"
        >
          {empty ? (
            <EmptyState
              onPick={(s) => {
                setDraft(s);
              }}
            />
          ) : (
            <div className="mx-auto w-full max-w-3xl px-2 py-6">
              {turns.map((turn) => (
                <Message key={turn.id} turn={turn} />
              ))}
              <div className="h-8" />
            </div>
          )}
        </div>

        {/* ---- composer --------------------------------------------------- */}
        <div className="bloom relative shrink-0 px-4 pb-5 pt-2">
          <div className="relative z-10 mx-auto w-full max-w-3xl">
            <AnimatePresence>
              {!atBottom && !empty && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 6 }}
                  className="pointer-events-none absolute -top-11 left-1/2 -translate-x-1/2"
                >
                  <Button
                    variant="secondary"
                    size="icon"
                    onClick={() => {
                      setAtBottom(true);
                      scrollToBottom();
                    }}
                    aria-label="Scroll to latest"
                    className="pointer-events-auto rounded-full shadow-lg"
                  >
                    <ArrowDown className="size-4" />
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>

            <Composer
              value={draft}
              onChange={setDraft}
              onSend={submit}
              onStop={stop}
              busy={busy}
              autoFocus
            />
            <p className="mt-2 text-center text-[11px] text-muted-foreground/45">
              Runs against the real OpenAI API and can send email and browse the
              web. Check anything that matters.
            </p>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}

function Pill({
  icon: Icon,
  text,
}: {
  icon: typeof Layers;
  text: string;
}) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md bg-muted/70 px-1.5 py-0.5 text-[10.5px] font-medium text-muted-foreground ring-1 ring-inset ring-border/60">
      <Icon className="size-2.5" />
      {text}
    </span>
  );
}

function EmptyState({ onPick }: { onPick: (s: string) => void }) {
  return (
    <div className="bloom relative flex h-full flex-col items-center justify-center px-6">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative z-10 w-full max-w-2xl text-center"
      >
        <h1 className="bg-gradient-to-b from-foreground to-foreground/55 bg-clip-text text-[1.75rem] font-semibold tracking-tight text-transparent">
          What should the swarm do?
        </h1>
        <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground/75">
          An orchestrator plus four specialists — browser, scripter, comms and
          task board. You'll see each step as it happens.
        </p>

        <div className="mt-7 grid gap-2 text-left sm:grid-cols-3">
          {SUGGESTIONS.map((s, i) => (
            <motion.button
              key={s}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.08 + i * 0.05 }}
              onClick={() => onPick(s)}
              className={cn(
                "rounded-xl border border-border/70 bg-card/50 p-3 text-[12.5px] leading-snug text-muted-foreground",
                "transition-all duration-200 hover:border-primary/35 hover:bg-card hover:text-foreground hover:shadow-md"
              )}
            >
              {s}
            </motion.button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
