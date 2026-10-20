"""AgentScope tools that let the AI agent operate the task manager.

Each function below is wrapped into a ``FunctionTool`` and handed to the agent
via a ``Toolkit`` (see task_agent.py). They call the SAME store.py / wiki_store
modules the REST API uses, so anything the agent does shows up in the web UI and
vice-versa.

Tool authoring rules (AgentScope 2.0):
    * plain function with a Google-style docstring (the ``Args:`` block becomes
      the JSON schema the model sees), and
    * return a ``ToolResponse(content=[TextBlock(type="text", text=...)])``.

We keep tool *output* compact and text-based (JSON snippets), which is what the
openai-mocker / ChatGPT backend handles best.
"""

import json
import re
from typing import Optional

from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, Toolkit, ToolChunk

import config
import store
import wiki_store          # the self-healing memory wiki (replaces flat notes)
import people_store        # the self-learning contacts directory
import scheduler
import sessions
import task_scripts
import mailer
import calendar_tools    # real Google Calendar + Meet (distinct from scheduler.py)
import deliver_tools       # send files/media to the user's phone (Roadmap #5)


def _ok(obj) -> ToolChunk:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return ToolChunk(content=[TextBlock(type="text", text=text)])


def _err(msg: str) -> ToolChunk:
    return ToolChunk(content=[TextBlock(type="text", text=f"ERROR: {msg}")])


# ===========================================================================
#  TASK / STRUCTURE TOOLS
# ===========================================================================
def view_board() -> ToolChunk:
    """Get a cheap overview of the whole task board: every group, subgroup,
    and a lightweight stub of each task (title, status, priority, due date,
    open follow-up/reminder counts). Call this FIRST to orient yourself, then
    drill into a specific task with get_task. This is the navigable index --
    it does not include full task descriptions/notes."""
    return _ok(store.get_tree())


def get_task(task_id: str) -> ToolChunk:
    """Load the full details of one task: description, notes, all reminders and
    all follow-ups.

    Args:
        task_id (str): The task id (e.g. 'tsk_ab12cd34ef').
    """
    t = store.get_task(task_id)
    if t is None:
        return _err(f"no task with id {task_id}")
    return _ok(t)


def create_group(name: str, description: str = "") -> ToolChunk:
    """Create a new top-level group (an area or project, e.g. 'Vendor
    Onboarding').

    Args:
        name (str): The group name.
        description (str): Optional description of what the group is for.
    """
    return _ok(store.create_group(name, description))


def create_subgroup(
    group_id: str, name: str, description: str = ""
) -> ToolChunk:
    """Create a subgroup nested under an existing group (e.g. a phase or
    workstream).

    Args:
        group_id (str): The parent group's id.
        name (str): The subgroup name.
        description (str): Optional description.
    """
    try:
        return _ok(store.create_subgroup(group_id, name, description))
    except KeyError as e:
        return _err(str(e))


def create_task(
    title: str,
    group_id: str,
    subgroup_id: Optional[str] = None,
    description: str = "",
    priority: str = "medium",
    due_date: Optional[str] = None,
    person: Optional[str] = None,
) -> ToolChunk:
    """Create an individual task inside a group (and optionally a subgroup).

    Args:
        title (str): Short title of the task.
        group_id (str): The group the task belongs to.
        subgroup_id (str): Optional subgroup id within that group.
        description (str): Optional longer description.
        priority (str): One of 'low', 'medium', 'high', 'urgent'.
        due_date (str): Optional ISO date, e.g. '2026-06-10'.
        person (str): Optional person this task is associated with / owned by.
    """
    try:
        return _ok(
            store.create_task(
                title=title,
                group_id=group_id,
                subgroup_id=subgroup_id,
                description=description,
                priority=priority,
                due_date=due_date,
                person=person,
            )
        )
    except (KeyError, ValueError) as e:
        return _err(str(e))


def update_task(
    task_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    due_date: Optional[str] = None,
    person: Optional[str] = None,
) -> ToolChunk:
    """Update fields on a task. Only pass the fields you want to change.

    Args:
        task_id (str): The task id.
        title (str): New title.
        description (str): New description.
        status (str): One of 'todo', 'in_progress', 'blocked', 'done'.
        priority (str): One of 'low', 'medium', 'high', 'urgent'.
        due_date (str): New ISO due date.
        person (str): New associated person.
    """
    fields = {
        k: v for k, v in {
            "title": title, "description": description, "status": status,
            "priority": priority, "due_date": due_date, "person": person,
        }.items() if v is not None
    }
    if not fields:
        return _err("nothing to update")
    try:
        return _ok(store.update_task(task_id, **fields))
    except (KeyError, ValueError) as e:
        return _err(str(e))


def move_task(
    task_id: str, group_id: str, subgroup_id: Optional[str] = None
) -> ToolChunk:
    """Move a task into a different group/subgroup.

    Args:
        task_id (str): The task to move.
        group_id (str): Destination group id.
        subgroup_id (str): Optional destination subgroup id.
    """
    try:
        return _ok(store.move_task(task_id, group_id, subgroup_id))
    except KeyError as e:
        return _err(str(e))


def delete_task(task_id: str) -> ToolChunk:
    """Permanently delete a task.

    Args:
        task_id (str): The task id to delete.
    """
    store.delete_task(task_id)
    return _ok({"deleted": task_id})


def add_note(task_id: str, text: str) -> ToolChunk:
    """Append a timestamped note/comment to a task.

    Args:
        task_id (str): The task id.
        text (str): The note text.
    """
    try:
        return _ok(store.add_note(task_id, text))
    except KeyError as e:
        return _err(str(e))


# ===========================================================================
#  REMINDER TOOLS
# ===========================================================================
def add_reminder(task_id: str, remind_at: str, message: str = "") -> ToolChunk:
    """Set a reminder on a task.

    Args:
        task_id (str): The task id.
        remind_at (str): When to be reminded -- ISO date or datetime, e.g.
            '2026-06-05' or '2026-06-05T14:00'.
        message (str): Optional reminder message (defaults to the task title).
    """
    try:
        return _ok(store.add_reminder(task_id, remind_at, message))
    except KeyError as e:
        return _err(str(e))


def complete_reminder(task_id: str, reminder_id: str) -> ToolChunk:
    """Mark a reminder as done.

    Args:
        task_id (str): The task id.
        reminder_id (str): The reminder id.
    """
    try:
        return _ok(store.complete_reminder(task_id, reminder_id))
    except KeyError as e:
        return _err(str(e))


def list_due_reminders(before: Optional[str] = None) -> ToolChunk:
    """List open reminders that are due. Use this to tell the user what they
    need to act on.

    Args:
        before (str): Optional ISO cutoff; defaults to now. Pass '9999' to get
            ALL open reminders regardless of date.
    """
    return _ok(store.due_reminders(before))


# ===========================================================================
#  FOLLOW-UP TOOLS
# ===========================================================================
def add_follow_up(
    task_id: str, person: str, note: str = "", due_date: Optional[str] = None
) -> ToolChunk:
    """Log a follow-up needed with a person on a task. Each call increments the
    'attempt' count for that person on that task, so you can track how many
    times the same issue has been chased.

    Args:
        task_id (str): The task id.
        person (str): The person to follow up with.
        note (str): Optional context (what the follow-up is about).
        due_date (str): Optional ISO date by which to follow up.
    """
    try:
        return _ok(store.add_follow_up(task_id, person, note, due_date))
    except KeyError as e:
        return _err(str(e))


def complete_follow_up(task_id: str, follow_up_id: str) -> ToolChunk:
    """Mark a follow-up as resolved.

    Args:
        task_id (str): The task id.
        follow_up_id (str): The follow-up id.
    """
    try:
        return _ok(store.complete_follow_up(task_id, follow_up_id))
    except KeyError as e:
        return _err(str(e))


def list_pending_follow_ups() -> ToolChunk:
    """List every open follow-up across all tasks, grouped sensibly so you can
    tell the user who they still need to chase."""
    return _ok(store.pending_follow_ups())


def follow_up_summary() -> ToolChunk:
    """Summarise how many times each person has been followed up with per task
    (total and still-open). Answers 'how many follow-ups have I taken for this
    issue / this person'."""
    return _ok(store.follow_up_counts())


# ===========================================================================
#  SEARCH / STATS
# ===========================================================================
def search_tasks(
    query: str = "",
    status: Optional[str] = None,
    person: Optional[str] = None,
    priority: Optional[str] = None,
) -> ToolChunk:
    """Search tasks by free text and/or filters.

    Args:
        query (str): Text to match in title, description, notes, or follow-up
            people.
        status (str): Optional status filter.
        person (str): Optional person filter.
        priority (str): Optional priority filter.
    """
    return _ok(store.search_tasks(query, status, person, priority))


def board_stats() -> ToolChunk:
    """Get high-level counts: total tasks, tasks per status, open reminders,
    open follow-ups."""
    return _ok(store.stats())


def list_completed_tasks() -> ToolChunk:
    """List all tasks marked done (newest completion first). Completed tasks
    are never deleted -- this reads them back with their full history."""
    return _ok(store.completed_tasks())


def audit_log(limit: int = 50, task_id: Optional[str] = None) -> ToolChunk:
    """Read the append-only audit log of every change (creates, updates,
    completions, deletions, follow-ups).

    Args:
        limit (int): Max number of recent events to return.
        task_id (str): Optional -- restrict to one task's history.
    """
    return _ok(store.read_audit(limit=limit, task_id=task_id))


# ===========================================================================
#  SCHEDULER / CALENDAR TOOLS
# ===========================================================================
def current_datetime() -> ToolChunk:
    """Get the current local date and time (and weekday). Call this when you
    need to reason about 'when' relative to now."""
    n = scheduler.now()
    return _ok({"now": scheduler.now_iso(), "weekday": n.strftime("%A")})


def schedule_reminder(
    message: str,
    when: str,
    recurrence: str = "none",
    task_id: Optional[str] = None,
) -> ToolChunk:
    """Schedule a time-based reminder that will actively fire (pop a
    notification) at the given time. Use this for 'remind me in 20 minutes',
    'remind me in 2 days', 'ping me tomorrow at 3pm', etc.

    Args:
        message (str): What to remind the user about.
        when (str): When to fire -- natural language ('in 20 minutes', 'in 2
            days', 'tomorrow at 3pm', 'next monday 09:30') OR an ISO datetime.
            It is parsed against the real current time, so just pass the user's
            phrasing.
        recurrence (str): One of 'none', 'daily', 'weekly', 'monthly'.
        task_id (str): Optional task id to attach the reminder to.
    """
    try:
        return _ok(scheduler.create_event(message, when, recurrence, task_id))
    except scheduler.WhenParseError as e:
        return _err(f"couldn't understand the time '{when}': {e}")


def list_scheduled() -> ToolChunk:
    """List upcoming scheduled reminders (not yet fired, not canceled)."""
    return _ok(scheduler.upcoming(limit=50))


def cancel_scheduled(event_id: str) -> ToolChunk:
    """Cancel a scheduled reminder by its id.

    Args:
        event_id (str): The scheduled event id (e.g. 'evt_...').
    """
    ok = scheduler.cancel_event(event_id)
    return _ok({"canceled": event_id}) if ok else _err("no such event")


# ===========================================================================
#  PAST-SESSION TOOLS (continue an earlier conversation on demand)
# ===========================================================================
def list_past_sessions() -> ToolChunk:
    """List your recent past conversations (id, title, one-line summary). You
    already see the most recent few in your instructions; use this to see them
    again. Only load one if the current request clearly relates to it."""
    return _ok(sessions.recent(8))


def load_past_session(session_id: str) -> ToolChunk:
    """Pull the full transcript of an earlier session into context so you can
    continue it. Call this ONLY when the user's current task clearly relates to
    that past session.

    Args:
        session_id (str): The session id from the recent-sessions list (ses_...).
    """
    text = sessions.transcript_text(session_id)
    if not text:
        return _err(f"no past session '{session_id}'")
    return _ok(text)


def search_past_sessions(query: str) -> ToolChunk:
    """Search your stored past conversations by keyword (returns matching
    session ids + titles).

    Args:
        query (str): Keyword/phrase to look for across past sessions.
    """
    return _ok(sessions.search(query))


def get_messages(message_ids: list) -> ToolChunk:
    """Fetch the full text of specific past messages by their ids, without
    loading a whole session. A message id (mid) looks like 'ses_..._.2' and is
    shown next to each turn in a session transcript and in the CONTINUATION block
    of your instructions. Pass one or more.

    Args:
        message_ids (list): Message ids to fetch (e.g. ['ses_..._.0']).
    """
    if isinstance(message_ids, str):
        message_ids = [message_ids]
    hits = sessions.get_messages(message_ids)
    if not hits:
        return _err("no messages found for those ids")
    return _ok(hits)


# ===========================================================================
#  LONG-TERM MEMORY TOOLS (PageIndex-style)
# ===========================================================================
def memory_index() -> ToolChunk:
    """List the table-of-contents of your long-term memory wiki (slug, title,
    one-line description, tags, confidence). Read this FIRST, then load the
    specific page you need with memory_read. Each page is one entity/topic — a
    person, project, or recurring subject — built from dated, sourced facts."""
    return _ok(wiki_store.index())


def memory_read(slug: str) -> ToolChunk:
    """Read the full body of one memory-wiki page (its dated fact bullets).

    Args:
        slug (str): The page slug from memory_index (e.g. 'profile', 'umesh-patil').
    """
    page = wiki_store.read_page(slug)
    if page is None:
        return _err(f"no memory page '{slug}'")
    return _ok(page)


def memory_write(
    title: str, body: str, description: str = "", slug: Optional[str] = None
) -> ToolChunk:
    """Create or overwrite a whole long-term memory page (a Markdown file that
    survives across sessions). Prefer remember_fact for a single fact; use this
    only when you want to (re)write an entire page's body.

    Args:
        title (str): Page title (the entity/topic, e.g. 'Vendor Onboarding').
        body (str): The page content (Markdown).
        description (str): One-line summary used in the memory index.
        slug (str): Optional explicit slug; otherwise derived from the title.
    """
    return _ok(wiki_store.write_page(title, body, description=description,
                                     source="agent", slug=slug))


def memory_append(slug: str, text: str) -> ToolChunk:
    """Add a fact to an existing memory page (creating it if absent). Idempotent:
    a near-identical fact already on the page is refreshed, not duplicated.

    Args:
        slug (str): The page slug.
        text (str): The fact to record.
    """
    title = slug.replace("-", " ").title()
    return _ok(wiki_store.upsert_fact(title, text, source="agent",
                                      confidence=0.8, slug=slug))


def remember_fact(text: str, topic: str = "profile") -> ToolChunk:
    """Durably remember a fact about the user or their work. Defaults to the
    user-profile page. The user telling you something is strong evidence, so it's
    saved at high confidence.

    Args:
        text (str): The fact to remember.
        topic (str): Optional page slug to attach it to (default 'profile' — use
            a person's or project's slug for entity-specific facts).
    """
    return _ok(wiki_store.remember(text, topic, source="user", confidence=0.9))


def memory_search(query: str) -> ToolChunk:
    """Search your long-term memory wiki for a keyword/entity and get matching
    pages with snippets (full-text, most relevant first).

    Args:
        query (str): The keyword/phrase/name to look for.
    """
    return _ok(wiki_store.search(query))


# ===========================================================================
#  PEOPLE DIRECTORY (self-learning contacts: resolve a name → email/channels)
# ===========================================================================
def lookup_contact(query: str) -> ToolChunk:
    """Resolve a name/nickname/email to a known contact BEFORE you message or
    email someone. Returns {status, contact, candidates}: status 'resolved' (use
    contact.email / contact.channels), 'ambiguous' (several match — show the
    candidates and ask which), or 'none' (unknown — ask the user for the address,
    then add_contact it). Always call this first for 'email/message <name>'.

    Args:
        query (str): The person to resolve (e.g. 'Umesh', 'Umesh Patil', an email).
    """
    return _ok(people_store.resolve(query))


def add_contact(full_name: str, email: str = "", aliases: Optional[list] = None,
                notes: str = "") -> ToolChunk:
    """Add or update a contact (high confidence — the user told you). Use after
    the user gives you someone's email/handle so you remember it next time.

    Args:
        full_name (str): The person's name.
        email (str): Their email address (optional).
        aliases (list): Other names/nicknames they go by (optional).
        notes (str): Any note about them (role, team, context).
    """
    return _ok(people_store.add_contact(full_name, email=email,
                                        aliases=aliases, notes=notes))


def update_contact(contact_id: str, email: Optional[str] = None,
                   full_name: Optional[str] = None,
                   notes: Optional[str] = None) -> ToolChunk:
    """Update fields on an existing contact (by id from lookup_contact/list_contacts).

    Args:
        contact_id (str): The contact id (ct_...).
        email (str): New email.
        full_name (str): Corrected name.
        notes (str): Note to set.
    """
    fields = {k: v for k, v in {"email": email, "full_name": full_name,
                                "notes": notes}.items() if v is not None}
    res = people_store.update_contact(contact_id, **fields)
    return _ok(res) if res else _err(f"no contact '{contact_id}'")


def list_contacts() -> ToolChunk:
    """List everyone in the contacts directory (most recently seen first), with
    their email/channels/aliases. Use to answer 'who do I know' or to find a
    contact id."""
    return _ok(people_store.list_contacts())


# ===========================================================================
#  REPEATABLE BROWSER TASKS (record once → run fast by name)
# ===========================================================================
def save_repeatable_task(task_name: str, description: str = "") -> ToolChunk:
    """Save the browser task you JUST completed as a reusable, parameterised
    automation, so next time you can run it INSTANTLY by name instead of driving
    the browser click-by-click. Call this ONLY after a web task succeeded
    cleanly in one go, and only once the user confirms it's something they'll
    repeat.

    The text you typed during the task becomes the task's inputs (parameters),
    each defaulting to what you typed. Returns the saved name + parameter names.

    Args:
        task_name (str): Short friendly name, e.g. 'teams reminder to person'.
        description (str): What the task does and when to use it.
    """
    try:
        m = task_scripts.save_task(task_name, description)
    except Exception as exc:
        return _err(str(exc))
    return _ok({"saved": m["name"],
                "params": [p["name"] for p in m["params"]],
                "from_run": m["source_run"]})


def list_repeatable_tasks() -> ToolChunk:
    """List the saved repeatable browser tasks you can run by name (with their
    input parameter names). Check this BEFORE driving the browser manually -- if
    a saved task already covers the request, run it instead (much faster)."""
    return _ok(task_scripts.list_tasks())


def run_repeatable_task(task_name: str, params: Optional[dict] = None) -> ToolChunk:
    """Run a previously saved browser task by name -- fast and deterministic, no
    step-by-step browser driving. Pass only the inputs you want to change from
    their recorded defaults.

    Args:
        task_name (str): The saved task's name (see list_repeatable_tasks).
        params (dict): Input overrides, e.g. {"type_a_message": "Hi Ravi"}. Use
            the parameter names from the saved task; omitted ones use defaults.
    """
    return _ok(task_scripts.run_task(task_name, params or {}))


# ===========================================================================
#  EMAIL (Gmail SMTP via app password — no browser, no creds through the LLM)
# ===========================================================================
# --- pre-send body checks --------------------------------------------------
# Defects seen in real sends, each of which reached a human inbox:
#   * "[Your Name]" -- the model had no name to sign, so it left the template.
#   * "/api/reports/X.docx" -- a server route pasted where a file should have
#     been attached; meaningless to the recipient.
#   * "/mnt/data/..." -- ChatGPT's own sandbox path, for a file that never
#     existed on this machine.
#   * "**bold**" / "## heading" -- markdown, which Gmail renders literally in a
#     plain-text body.
# A persona rule is a suggestion; this is the part that actually holds, because
# it runs between the model and SMTP.
_BODY_PLACEHOLDERS = (
    "[your name]", "[name]", "[recipient]", "[insert", "[company]",
    "[date]", "[subject]", "your name here", "xxx",
)
_BODY_BAD_PATHS = ("/api/reports/", "/mnt/data", "sandbox:/", "/home/sandbox")
_BODY_MARKDOWN = (
    re.compile(r"\*\*\S"),              # **bold**
    re.compile(r"^#{1,6}\s", re.M),     # ## heading
    re.compile(r"^\s*[-*]\s+\*\*"),     # - **item**
    re.compile(r"\[[^\]]+\]\(https?://"),  # [text](url)
)


def _email_body_defects(body: str) -> list[str]:
    """Reasons this body should not be sent as-is. Empty list = fine."""
    text = body or ""
    low = text.lower()
    out = []

    for ph in _BODY_PLACEHOLDERS:
        if ph in low:
            out.append(
                f"the body still contains the placeholder '{ph}' — replace it "
                f"with the real value (you sign as {config.USER_NAME or 'the user'})")
            break
    for bad in _BODY_BAD_PATHS:
        if bad in low:
            out.append(
                f"the body contains '{bad}', which is a path on this machine, "
                "not something the recipient can open — pass the file in "
                "'attachments' instead and refer to it by filename")
            break
    for pat in _BODY_MARKDOWN:
        if pat.search(text):
            out.append(
                "the body contains markdown (**bold**, ## headings or "
                "[links](url)) — Gmail shows a plain-text body literally, so "
                "write plain sentences and use blank lines and '-' bullets")
            break
    return out


def create_meeting(title: str, when: str, attendees: Optional[str] = None,
                   duration_minutes: int = 30,
                   description: str = "") -> ToolChunk:
    """Create a REAL Google Calendar event with a Google Meet link, and invite
    people to it.

    This is the ONLY way to get a working Meet link. Do NOT try to make one by
    opening meet.google.com in the browser and clicking — that does not produce
    a shareable link and has failed every time it was attempted. Do NOT use
    schedule_reminder for a meeting with other people: that is a private local
    nudge, it creates no event and no link and nobody else ever sees it.

    The invitees are emailed by Google automatically. The returned meet_link can
    then be put in the body of a separate email if you were asked to send one.

    Args:
        title (str): Event title, e.g. "LMS functional stories review".
        when (str): Start time — "tomorrow 5 PM", "Friday 14:30" or an ISO
            timestamp like "2026-09-19T17:00".
        attendees (str): Comma-separated email addresses to invite. Resolve
            names to addresses with lookup_contact first; never invent one.
        duration_minutes (int): Length in minutes (default 30).
        description (str): Optional agenda/notes shown in the invite.
    """
    return _ok(calendar_tools.create_meeting(
        title, when, attendees, duration_minutes, description))


def list_meetings(days: int = 7) -> ToolChunk:
    """List upcoming REAL Google Calendar events (soonest first), with their
    Meet links and attendees.

    This reads the actual Google calendar other people can see — distinct from
    list_scheduled, which shows this agent's own private reminders.

    Args:
        days (int): How many days ahead to look (default 7).
    """
    return _ok(calendar_tools.list_meetings(days))


def cancel_meeting(event_id: str) -> ToolChunk:
    """Cancel a Google Calendar event and notify its guests.

    Args:
        event_id (str): The event_id returned by create_meeting or
            list_meetings.
    """
    return _ok(calendar_tools.cancel_meeting(event_id))


_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def _resolve_recipient(to: str) -> tuple[str, str]:
    """Turn a bare name into an address using the people directory.

    A real run did this:

        lookup_contact(query=Lesley)   -> resolved, lesley@example.com
        send_email(to=Lesley)          -> SMTPRecipientsRefused 553

    The model had ALREADY resolved the contact and then passed the display name
    anyway. Rather than rely on it remembering, resolve here -- the directory
    exists precisely so a casual name becomes an address, and failing at the
    SMTP server is the worst place to discover it did not.

    Returns (address, note). An unresolvable name yields ("", explanation) so
    the caller can tell the model what to fix instead of bouncing off Gmail.
    """
    raw = (to or "").strip()
    if not raw:
        return "", "NOT SENT — no recipient given."
    # Already one or more real addresses: leave completely alone.
    if _EMAIL_RE.search(raw):
        return raw, ""
    try:
        import people_store
        hit = people_store.resolve(raw)
    except Exception:
        hit = None
    email = ""
    if isinstance(hit, dict):
        email = (hit.get("email")
                 or (hit.get("contact") or {}).get("email") or "").strip()
    if email:
        return email, f"resolved '{raw}' to {email} via the people directory"
    return "", (f"NOT SENT — '{raw}' is not an email address and no contact "
                f"matches it. Use lookup_contact to find them, or add_contact "
                f"to record their address, then call send_email with the "
                f"actual address.")


def send_email(to: str, subject: str, body: str, cc: Optional[str] = None,
               attachments: Optional[list] = None) -> ToolChunk:
    """Send an email from the user's own Gmail account, optionally with files
    attached.

    WORKFLOW (always): first DRAFT the email and show the user the exact
    recipient, subject, and full body, then WAIT for their explicit 'yes/send'.
    Only call this tool AFTER they approve. Keep the tone firm, clear, and
    professional — you may flag blockers, delays, or who a dependency is on, but
    never write a personal attack or insult. You do NOT need a password and must
    never ask for or include credentials.

    ATTACHMENTS: pass root-qualified paths, e.g.
    ["project/reports/Term_Loan_Overview.docx"]. Reports you generated live in
    'project/reports'; use fs_find if you are unsure of the exact name. Never
    paste a /api/reports/... link into the body — attach the file instead.

    The body must be PLAIN TEXT: no markdown, no placeholders like [Your Name].

    Args:
        to (str): Recipient email address.
        subject (str): Subject line.
        body (str): Plain-text email body (the approved draft).
        cc (str): Optional CC recipient(s), comma-separated.
        attachments (list): Optional root-qualified file paths to attach.
    """
    defects = _email_body_defects(body)
    if defects:
        return _ok({
            "ok": False,
            "sent": False,
            "error": "NOT SENT — fix the body and call send_email again: "
                     + "; ".join(defects),
            "defects": defects,
        })
    to, note = _resolve_recipient(to)
    if note and not to:
        return _ok({"ok": False, "sent": False, "error": note})
    out = mailer.send_email(to, subject, body, cc, attachments)
    if note and isinstance(out, dict):
        out["resolved_recipient"] = note
    return _ok(out)


# ===========================================================================
#  TOOLKIT FACTORY
# ===========================================================================
_TOOL_FUNCS = [
    # structure / tasks
    view_board, get_task, create_group, create_subgroup, create_task,
    update_task, move_task, delete_task, add_note,
    # reminders
    add_reminder, complete_reminder, list_due_reminders,
    # follow-ups
    add_follow_up, complete_follow_up, list_pending_follow_ups,
    follow_up_summary,
    # search / stats / history
    search_tasks, board_stats, list_completed_tasks, audit_log,
    # scheduler / calendar
    current_datetime, schedule_reminder, list_scheduled, cancel_scheduled,
    # past sessions
    list_past_sessions, load_past_session, search_past_sessions, get_messages,
    # long-term memory
    memory_index, memory_read, memory_write, memory_append, remember_fact,
    memory_search,
    # people directory (self-learning contacts)
    lookup_contact, add_contact, update_contact, list_contacts,
    # repeatable browser tasks (record once → run by name)
    save_repeatable_task, list_repeatable_tasks, run_repeatable_task,
    # email (Gmail SMTP)
    send_email,
    # real Google Calendar + Meet links (scheduler.* is local reminders only)
    create_meeting, list_meetings, cancel_meeting,
    # phone delivery (send a file/media to the user's mobile app)
    deliver_tools.send_file, deliver_tools.list_deliveries,
]


def build_toolkit(include_browser: bool = True) -> Toolkit:
    """Build the Toolkit the agent uses to operate the task manager (and,
    optionally, to control a real browser)."""
    tools = [FunctionTool(f, is_read_only=False) for f in _TOOL_FUNCS]
    if include_browser:
        try:
            import browser_tools
            tools += browser_tools.build_tools()
        except Exception as exc:  # browser stack optional; don't break tasks
            print(f"[agent] browser tools unavailable: {exc}")
    return Toolkit(tools=tools)
