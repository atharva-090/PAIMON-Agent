import { useEffect, useRef } from "react";
import { ArrowUp, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  busy,
  autoFocus,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  busy: boolean;
  autoFocus?: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to a ceiling, then scroll inside.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 224)}px`;
  }, [value]);

  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

  // Refocus when a turn ends so the next message can just be typed.
  useEffect(() => {
    if (!busy) ref.current?.focus();
  }, [busy]);

  return (
    <div
      className={cn(
        "relative flex items-end gap-2 rounded-[1.4rem] border border-border bg-card/85 p-2 pl-4 shadow-xl backdrop-blur-xl",
        "transition-shadow duration-200",
        "focus-within:border-primary/45 focus-within:shadow-[0_0_0_4px_color-mix(in_oklch,var(--primary)_13%,transparent)]"
      )}
    >
      <textarea
        ref={ref}
        rows={1}
        value={value}
        disabled={busy}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          // Enter sends; Shift+Enter is a newline. Standard for this shape of
          // input and what anyone coming from ChatGPT/Claude will expect.
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            onSend();
          }
        }}
        placeholder={busy ? "Running…" : "Ask PAIMON anything…"}
        className="scrollbar-slim max-h-56 flex-1 resize-none bg-transparent py-2.5 text-[0.9375rem] leading-relaxed outline-none placeholder:text-muted-foreground/55 disabled:opacity-60"
      />
      {busy ? (
        <Button
          size="icon"
          variant="secondary"
          onClick={onStop}
          aria-label="Stop"
          className="mb-0.5 rounded-full"
        >
          <Square className="size-3.5 fill-current" />
        </Button>
      ) : (
        <Button
          size="icon"
          onClick={onSend}
          disabled={!value.trim()}
          aria-label="Send"
          className="mb-0.5 rounded-full"
        >
          <ArrowUp className="size-4" />
        </Button>
      )}
    </div>
  );
}
