import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { motion } from "framer-motion";
import { Copy, Check, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ActivityTrail } from "./ActivityTrail";
import { cn } from "@/lib/utils";
import type { Turn } from "@/lib/useChat";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label="Copy reply"
      className="text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1600);
        } catch {
          /* clipboard blocked (insecure origin) -- selection still works */
        }
      }}
    >
      {copied ? (
        <Check className="size-3.5 text-[var(--success)]" />
      ) : (
        <Copy className="size-3.5" />
      )}
    </Button>
  );
}

export function Message({ turn }: { turn: Turn }) {
  if (turn.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className="flex justify-end px-4 py-2.5"
      >
        <div className="max-w-[min(42rem,85%)] break-words rounded-2xl rounded-br-md bg-secondary px-4 py-2.5 text-[0.9375rem] leading-relaxed whitespace-pre-wrap">
          {turn.text}
        </div>
      </motion.div>
    );
  }

  const failed = turn.status === "error";
  const waiting = turn.status === "streaming" && !turn.activities.length;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="group px-4 py-2.5"
    >
      <div className="max-w-3xl">
        <ActivityTrail
          activities={turn.activities}
          streaming={turn.status === "streaming"}
          startedAt={turn.startedAt}
          endedAt={turn.endedAt}
        />

        {waiting && (
          <div className="flex items-center gap-1.5 py-1 pl-1">
            {[0, 1, 2].map((i) => (
              <motion.span
                key={i}
                className="size-1.5 rounded-full bg-muted-foreground/50"
                animate={{ opacity: [0.25, 1, 0.25] }}
                transition={{
                  duration: 1.1,
                  repeat: Infinity,
                  delay: i * 0.16,
                }}
              />
            ))}
          </div>
        )}

        {turn.text && (
          <div
            className={cn(
              "prose-reply",
              failed &&
                "flex items-start gap-2 rounded-lg border border-destructive/35 bg-destructive/8 px-3 py-2.5 text-destructive"
            )}
          >
            {failed && <TriangleAlert className="mt-0.5 size-4 shrink-0" />}
            {failed ? (
              <span className="text-[0.9rem]">{turn.text}</span>
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {turn.text}
              </ReactMarkdown>
            )}
          </div>
        )}

        {turn.text && !failed && turn.status === "done" && (
          <div className="-ml-1 mt-1">
            <CopyButton text={turn.text} />
          </div>
        )}
      </div>
    </motion.div>
  );
}
