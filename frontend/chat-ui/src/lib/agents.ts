import {
  Bot,
  Globe,
  Mail,
  ListChecks,
  FileSpreadsheet,
  type LucideIcon,
} from "lucide-react";

export interface AgentMeta {
  label: string;
  icon: LucideIcon;
  /** Tailwind text colour for the icon + name. */
  tone: string;
  /** Matching translucent background for chips. */
  chip: string;
}

/** Names match multi_agent/roster.py exactly; "orchestrator" is the label the
 *  swarm's StreamEmitter uses for the top-level agent. Anything unrecognised
 *  falls back to a neutral look rather than breaking the render. */
export const AGENTS: Record<string, AgentMeta> = {
  orchestrator: {
    label: "Orchestrator",
    icon: Bot,
    tone: "text-violet-300",
    chip: "bg-violet-400/10 ring-violet-400/25",
  },
  browser_agent: {
    label: "Browser",
    icon: Globe,
    tone: "text-sky-300",
    chip: "bg-sky-400/10 ring-sky-400/25",
  },
  comms_agent: {
    label: "Comms",
    icon: Mail,
    tone: "text-emerald-300",
    chip: "bg-emerald-400/10 ring-emerald-400/25",
  },
  task_manager: {
    label: "Tasks",
    icon: ListChecks,
    tone: "text-rose-300",
    chip: "bg-rose-400/10 ring-rose-400/25",
  },
  scripter: {
    label: "Scripter",
    icon: FileSpreadsheet,
    tone: "text-amber-300",
    chip: "bg-amber-400/10 ring-amber-400/25",
  },
};

export const FALLBACK_AGENT: AgentMeta = {
  label: "Agent",
  icon: Bot,
  tone: "text-muted-foreground",
  chip: "bg-muted ring-border",
};

export function agentMeta(name: string): AgentMeta {
  return AGENTS[name] ?? FALLBACK_AGENT;
}

/** Tool names -> a short human phrase. Anything absent renders as its raw name,
 *  which is still unambiguous -- this is polish, not a lookup requirement. */
export const TOOL_LABELS: Record<string, string> = {
  delegate_task: "delegating",
  wait_for_results: "waiting for children",
  check_status: "checking the board",
  list_specialists: "listing specialists",
  browse_open: "opening page",
  browse_snapshot: "reading page",
  browse_click: "clicking",
  browse_click_text: "clicking",
  browse_type: "typing",
  browse_press: "pressing key",
  browse_scroll: "scrolling",
  browse_back: "going back",
  browse_get_text: "extracting text",
  browse_attach_file: "attaching file",
  browse_download: "downloading",
  send_email: "sending email",
  lookup_contact: "looking up contact",
  add_contact: "saving contact",
  create_task: "creating task",
  update_task: "updating task",
  create_group: "creating group",
  add_reminder: "setting reminder",
  fs_find: "finding file",
  fs_read: "reading file",
  fs_list: "listing folder",
  make_docx: "writing Word doc",
  make_xlsx: "building spreadsheet",
  make_pdf: "writing PDF",
  make_pptx: "building deck",
};

export function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? name;
}
