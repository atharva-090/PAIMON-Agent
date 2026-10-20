// BA Task Manager frontend — vanilla JS, talks to the FastAPI backend.

const API = "";              // same origin
let TREE = { groups: [] };
let SEL = { groupId: null, subId: null, completed: false };   // current selection
let SEARCHING = false;

// ---- helpers --------------------------------------------------------------
async function api(method, path, body) {
  const opt = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opt.body = JSON.stringify(body);
  const r = await fetch(API + path, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}
const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };
const esc = (s) => (s == null ? "" : String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])));

// ---- load + render top-level ---------------------------------------------
async function refresh() {
  TREE = await api("GET", "/api/tree");
  renderTree();
  if (SEL.calendar) renderCalendar();
  else if (SEL.completed) renderCompleted();
  else if (!SEARCHING) renderTasks();
  loadStats(); loadReminders(); loadFollowups(); loadReports();
}

async function loadStats() {
  const s = await api("GET", "/api/stats");
  document.getElementById("stats").innerHTML =
    `<span><b>${s.total_tasks}</b> tasks</span>
     <span><b>${s.by_status.in_progress || 0}</b> in progress</span>
     <span><b>${s.by_status.done || 0}</b> done</span>
     <span><b>${s.open_reminders}</b> reminders</span>
     <span><b>${s.open_follow_ups}</b> follow-ups</span>`;
}

function findGroup(id) { return TREE.groups.find(g => g.id === id); }
function findSub(g, id) { return (g.subgroups || []).find(s => s.id === id); }

// ---- sidebar tree ---------------------------------------------------------
function renderTree() {
  const root = document.getElementById("tree");
  root.innerHTML = "";
  if (!TREE.groups.length) {
    root.appendChild(el(`<div class="muted small">No groups yet. Click "+ Group".</div>`));
  }
  for (const g of TREE.groups) {
    const openCount = (g.tasks || []).filter(t => t.status !== "done").length;
    const gEl = el(`<div class="group"></div>`);
    const active = SEL.groupId === g.id && !SEL.subId && !SEL.completed ? "active" : "";
    const row = el(
      `<div class="row ${active}">
        <span>📁</span><span class="name">${esc(g.name)}</span>
        <span class="count">${openCount || ""}</span>
      </div>`);
    row.onclick = () => select(g.id, null);
    gEl.appendChild(row);
    for (const s of g.subgroups || []) {
      const sa = SEL.subId === s.id && !SEL.completed ? "active" : "";
      const sOpen = (s.tasks || []).filter(t => t.status !== "done").length;
      const sRow = el(
        `<div class="sub"><div class="row ${sa}">
           <span>↳</span><span class="name">${esc(s.name)}</span>
           <span class="count">${sOpen || ""}</span>
        </div></div>`);
      sRow.firstChild.onclick = (e) => { e.stopPropagation(); select(g.id, s.id); };
      gEl.appendChild(sRow);
    }
    root.appendChild(gEl);
  }

  // "Calendar" pseudo-view: the scheduler / mini google-calendar.
  const calActive = SEL.calendar ? "active" : "";
  const calRow = el(
    `<div class="group"><div class="row ${calActive}">
       <span>📅</span><span class="name">Calendar</span>
    </div></div>`);
  calRow.firstChild.onclick = () => selectCalendar();
  root.appendChild(calRow);

  // "Completed" pseudo-view: shows finished tasks across the whole board.
  const cActive = SEL.completed ? "active" : "";
  const cRow = el(
    `<div class="group"><div class="row ${cActive}">
       <span>✅</span><span class="name">Completed</span>
    </div></div>`);
  cRow.firstChild.onclick = () => selectCompleted();
  root.appendChild(cRow);
}

function select(groupId, subId) {
  SEL = { groupId, subId, completed: false, calendar: false };
  SEARCHING = false;
  document.getElementById("search").style.display = "";
  document.getElementById("search").value = "";
  renderTree();
  renderTasks();
}

function selectCompleted() {
  SEL = { groupId: null, subId: null, completed: true, calendar: false };
  SEARCHING = false;
  document.getElementById("search").style.display = "none";
  renderTree();
  renderCompleted();
}

function selectCalendar() {
  SEL = { groupId: null, subId: null, completed: false, calendar: true };
  SEARCHING = false;
  document.getElementById("search").style.display = "none";
  renderTree();
  renderCalendar();
}

// ---- main task list -------------------------------------------------------
// Open tasks only -- finished ones live in the Completed view (but are never
// deleted from data; status is just flipped to "done").
function currentStubs() {
  const g = findGroup(SEL.groupId);
  if (!g) return { title: "Select a group", stubs: null };
  const openOnly = (arr) => (arr || []).filter(t => t.status !== "done");
  if (SEL.subId) {
    const s = findSub(g, SEL.subId);
    return { title: `${g.name} › ${s ? s.name : "?"}`, stubs: s ? openOnly(s.tasks) : [] };
  }
  return { title: g.name, stubs: openOnly(g.tasks) };
}

function taskCard(stub) {
  const warn = (stub.open_followups || stub.open_reminders)
    ? `<span class="badge warn">${stub.open_followups || 0}🔁 ${stub.open_reminders || 0}⏰</span>` : "";
  const due = stub.due_date ? `<span class="badge due">📅 ${esc(stub.due_date)}</span>` : "";
  const c = el(
    `<div class="task ${stub.status === "done" ? "done" : ""}">
      <span class="check" title="toggle done">${stub.status === "done" ? "☑️" : "⬜"}</span>
      <div class="body">
        <div class="title">${esc(stub.title)}</div>
        <div class="meta">
          <span class="badge status s-${stub.status}">${stub.status.replace("_", " ")}</span>
          <span class="badge p-${stub.priority}">${stub.priority}</span>
          ${due}${warn}
        </div>
      </div>
    </div>`);
  c.querySelector(".check").onclick = async (e) => {
    e.stopPropagation();
    const ns = stub.status === "done" ? "todo" : "done";
    await api("PATCH", `/api/tasks/${stub.id}`, { status: ns });
    refresh();
  };
  c.querySelector(".body").onclick = () => openTask(stub.id);
  return c;
}

function renderTasks(stubsOverride, titleOverride) {
  const box = document.getElementById("tasks");
  box.innerHTML = "";
  const sel = stubsOverride ? { title: titleOverride, stubs: stubsOverride } : currentStubs();
  document.getElementById("main-title").textContent = sel.title;
  document.getElementById("main-actions").style.display = SEL.groupId && !stubsOverride ? "flex" : "none";
  if (sel.stubs == null) {
    box.appendChild(el(`<div class="muted">Pick a group on the left, or ask Hermes-PM to create one.</div>`));
    return;
  }
  if (!sel.stubs.length) {
    box.appendChild(el(`<div class="muted">No tasks here yet.</div>`));
    return;
  }
  for (const stub of sel.stubs) box.appendChild(taskCard(stub));
}

// ---- completed view -------------------------------------------------------
function groupNameOf(t) {
  const g = findGroup(t.group_id);
  if (!g) return "";
  const s = t.subgroup_id ? findSub(g, t.subgroup_id) : null;
  return s ? `${g.name} › ${s.name}` : g.name;
}

function completedCard(t) {
  const when = t.completed_at ? t.completed_at.slice(0, 16).replace("T", " ") : "";
  const c = el(
    `<div class="task done">
      <span class="check" title="reopen task">☑️</span>
      <div class="body">
        <div class="title">${esc(t.title)}</div>
        <div class="meta">
          <span class="badge status s-done">done</span>
          <span class="badge p-${t.priority}">${t.priority}</span>
          ${t.person ? `<span class="badge person">👤 ${esc(t.person)}</span>` : ""}
          <span class="badge due">📁 ${esc(groupNameOf(t))}</span>
          ${when ? `<span class="badge due">✅ ${esc(when)} UTC</span>` : ""}
          <span class="badge due">${(t.follow_ups || []).length}🔁 · ${(t.notes || []).length}📝</span>
        </div>
      </div>
    </div>`);
  c.querySelector(".check").onclick = async (e) => {
    e.stopPropagation();
    await api("PATCH", `/api/tasks/${t.id}`, { status: "todo" });  // reopen
    refresh();
  };
  c.querySelector(".body").onclick = () => openTask(t.id);
  return c;
}

async function renderCompleted() {
  const box = document.getElementById("tasks");
  document.getElementById("main-title").textContent = "✅ Completed tasks";
  document.getElementById("main-actions").style.display = "none";
  box.innerHTML = `<div class="muted small">loading…</div>`;
  const tasks = await api("GET", "/api/tasks/completed/list");
  box.innerHTML = "";
  if (!tasks.length) {
    box.appendChild(el(`<div class="muted">Nothing completed yet. Tick a task's box to finish it — it moves here, the data is kept.</div>`));
    return;
  }
  box.appendChild(el(`<div class="muted small" style="margin-bottom:6px">${tasks.length} completed · click the box to reopen, click a card for full history</div>`));
  for (const t of tasks) box.appendChild(completedCard(t));
}

// ---- calendar / scheduler -------------------------------------------------
let CAL_MONTH = new Date(); CAL_MONTH.setDate(1);

function pad(n) { return String(n).padStart(2, "0"); }
function ymd(d) { return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`; }
function fmtTime(iso) { return iso ? iso.slice(11, 16) : ""; }

// gather task due-dates from the loaded tree -> { 'YYYY-MM-DD': [titles] }
function taskDueMap() {
  const m = {};
  for (const g of TREE.groups || []) {
    const add = (t) => { if (t.due_date && t.status !== "done") (m[t.due_date.slice(0, 10)] ||= []).push(t.title); };
    (g.tasks || []).forEach(add);
    (g.subgroups || []).forEach(s => (s.tasks || []).forEach(add));
  }
  return m;
}

async function renderCalendar() {
  const box = document.getElementById("tasks");
  document.getElementById("main-title").textContent = "📅 Calendar & reminders";
  document.getElementById("main-actions").style.display = "none";

  const monthStart = new Date(CAL_MONTH);
  const monthEnd = new Date(CAL_MONTH.getFullYear(), CAL_MONTH.getMonth() + 1, 0);
  const events = await api("GET",
    `/api/schedule?frm=${ymd(monthStart)}T00:00&to=${ymd(monthEnd)}T23:59`);
  const byDay = {};
  for (const e of events) (byDay[e.when.slice(0, 10)] ||= []).push(e);
  const dueMap = taskDueMap();

  const monthName = CAL_MONTH.toLocaleString(undefined, { month: "long", year: "numeric" });
  box.innerHTML = "";

  // quick-add reminder
  const add = el(`<div>
    <div class="sched-add">
      <input class="msg" id="sa-msg" placeholder="Remind me to… (e.g. call vendor)">
      <input id="sa-when" placeholder="when — 'in 20 mins', 'in 2 days', 'tomorrow 3pm'">
      <select id="sa-rec">
        <option value="none">once</option><option value="daily">daily</option>
        <option value="weekly">weekly</option><option value="monthly">monthly</option>
      </select>
      <button class="primary" id="sa-add">Set reminder</button>
    </div>
    <div class="sched-hint">⏰ Reminders fire live with a pop-up + sound while this tab is open.
      <span id="notif-state"></span></div>
  </div>`);
  add.querySelector("#sa-add").onclick = async () => {
    const message = add.querySelector("#sa-msg").value.trim();
    const when = add.querySelector("#sa-when").value.trim();
    if (!message || !when) return alert("Enter both what to remind and when.");
    try {
      await api("POST", "/api/schedule", { message, when, recurrence: add.querySelector("#sa-rec").value });
      add.querySelector("#sa-msg").value = ""; add.querySelector("#sa-when").value = "";
      ensureNotifyPermission();
      renderCalendar();
    } catch (e) { alert("Couldn't schedule: " + e.message); }
  };
  box.appendChild(add);

  // header / month nav
  const head = el(`<div class="cal-head">
    <button class="mini" id="cal-prev">‹</button>
    <h3>${monthName}</h3>
    <button class="mini" id="cal-next">›</button>
    <button class="mini" id="cal-today">today</button>
  </div>`);
  head.querySelector("#cal-prev").onclick = () => { CAL_MONTH.setMonth(CAL_MONTH.getMonth() - 1); renderCalendar(); };
  head.querySelector("#cal-next").onclick = () => { CAL_MONTH.setMonth(CAL_MONTH.getMonth() + 1); renderCalendar(); };
  head.querySelector("#cal-today").onclick = () => { CAL_MONTH = new Date(); CAL_MONTH.setDate(1); renderCalendar(); };
  box.appendChild(head);

  // grid
  const grid = el(`<div class="cal-grid"></div>`);
  for (const d of ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    grid.appendChild(el(`<div class="cal-dow">${d}</div>`));

  const firstDow = new Date(CAL_MONTH.getFullYear(), CAL_MONTH.getMonth(), 1).getDay();
  const start = new Date(CAL_MONTH); start.setDate(1 - firstDow);
  const todayStr = ymd(new Date());
  for (let i = 0; i < 42; i++) {
    const day = new Date(start); day.setDate(start.getDate() + i);
    const key = ymd(day);
    const other = day.getMonth() !== CAL_MONTH.getMonth();
    const cell = el(`<div class="cal-cell ${other ? "other" : ""} ${key === todayStr ? "today" : ""}">
      <div class="dn">${day.getDate()}</div></div>`);
    const evs = byDay[key] || [];
    evs.slice(0, 3).forEach(e => cell.appendChild(
      el(`<div class="cal-ev ${e.fired ? "fired" : ""}" title="${esc(e.message)}">${fmtTime(e.when)} ${esc(e.message)}</div>`)));
    (dueMap[key] || []).slice(0, 2).forEach(t => cell.appendChild(
      el(`<div class="cal-ev due" title="due: ${esc(t)}">📌 ${esc(t)}</div>`)));
    const extra = evs.length - 3;
    if (extra > 0) cell.appendChild(el(`<div class="cal-more">+${extra} more</div>`));
    cell.onclick = () => openDay(day, evs);
    grid.appendChild(cell);
  }
  box.appendChild(grid);
}

function openDay(day, events) {
  const card = el(`<div>
    <h3>${day.toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" })}</h3>
    <div id="day-list"></div>
    <div class="detail-section"><h4>＋ Add reminder this day</h4>
      <div class="inline-form">
        <input id="d-msg" placeholder="message"><input id="d-time" type="time" value="09:00">
        <button class="mini" id="d-add">add</button>
      </div>
    </div>
    <div class="modal-actions"><span></span><button class="primary" id="d-close">Close</button></div>
  </div>`);
  const list = card.querySelector("#day-list");
  const draw = (evs) => {
    list.innerHTML = evs.length ? "" : `<div class="muted small">No reminders this day.</div>`;
    for (const e of evs) {
      const row = el(`<div class="list-row">
        <span class="grow">${e.fired ? "✓ " : "⏰ "}<b>${fmtTime(e.when)}</b> · ${esc(e.message)}
          ${e.recurrence !== "none" ? `<span class="attempt">${e.recurrence}</span>` : ""}</span>
        <button class="mini" data-cancel="${e.id}">cancel</button></div>`);
      row.querySelector("[data-cancel]").onclick = async () => {
        await api("DELETE", `/api/schedule/${e.id}`);
        const left = evs.filter(x => x.id !== e.id); draw(left); renderCalendar();
      };
      list.appendChild(row);
    }
  };
  draw(events);
  card.querySelector("#d-add").onclick = async () => {
    const message = card.querySelector("#d-msg").value.trim();
    if (!message) return alert("message?");
    const when = `${ymd(day)}T${card.querySelector("#d-time").value}`;
    await api("POST", "/api/schedule", { message, when });
    closeModal(); renderCalendar();
  };
  card.querySelector("#d-close").onclick = closeModal;
  showModal(card);
}

// ---- live reminder notifications -----------------------------------------
const _shownToasts = new Set();

function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.connect(g); g.connect(ctx.destination);
    o.type = "sine"; o.frequency.value = 880;
    g.gain.setValueAtTime(0.001, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.02);
    g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
    o.start(); o.stop(ctx.currentTime + 0.5);
  } catch (e) {}
}

function ensureNotifyPermission() {
  if ("Notification" in window && Notification.permission === "default")
    Notification.requestPermission();
}

function showToast(ev) {
  const box = document.getElementById("toasts");
  const t = el(`<div class="toast">
    <span class="t-close">✕</span>
    <div class="t-title">⏰ Reminder</div>
    <div class="t-body">${esc(ev.message)}</div>
  </div>`);
  t.querySelector(".t-close").onclick = () => t.remove();
  box.appendChild(t);
  setTimeout(() => t.remove(), 30000);
  beep();
  if ("Notification" in window && Notification.permission === "granted")
    try { new Notification("⏰ Reminder", { body: ev.message }); } catch (e) {}
}

async function pollNotifications() {
  let fired = [];
  try { fired = await api("GET", "/api/schedule/notifications"); } catch (e) { return; }
  const fresh = fired.filter(e => !_shownToasts.has(e.id));
  for (const e of fresh) { _shownToasts.add(e.id); showToast(e); }
  if (fired.length) {
    try { await api("POST", "/api/schedule/notifications/ack", { ids: fired.map(e => e.id) }); } catch (e) {}
    if (SEL.calendar) renderCalendar();
    loadStats();
  }
}

// ---- search ---------------------------------------------------------------
let searchTimer = null;
function onSearch(q) {
  clearTimeout(searchTimer);
  if (!q.trim()) { SEARCHING = false; renderTasks(); return; }
  searchTimer = setTimeout(async () => {
    SEARCHING = true;
    const res = await api("GET", `/api/search?query=${encodeURIComponent(q)}`);
    const stubs = res.map(t => ({
      id: t.id, title: t.title, status: t.status, priority: t.priority,
      due_date: t.due_date,
      open_followups: (t.follow_ups || []).filter(f => !f.done).length,
      open_reminders: (t.reminders || []).filter(r => !r.done).length,
    }));
    renderTasks(stubs, `🔍 ${res.length} result(s) for "${q}"`);
  }, 250);
}

// ---- side panels ----------------------------------------------------------
async function loadReminders() {
  const r = await api("GET", "/api/reminders/due?before=9999");
  const box = document.getElementById("reminders");
  box.innerHTML = r.length ? "" : `<div class="muted">Nothing pending.</div>`;
  for (const x of r) {
    const row = el(`<div class="item"><b>${esc(x.remind_at)}</b> · ${esc(x.message)}
      <div class="muted">${esc(x.task_title)}</div></div>`);
    row.onclick = () => openTask(x.task_id);
    row.style.cursor = "pointer";
    box.appendChild(row);
  }
}
async function loadFollowups() {
  const r = await api("GET", "/api/followups/pending");
  const box = document.getElementById("followups");
  box.innerHTML = r.length ? "" : `<div class="muted">All caught up.</div>`;
  for (const x of r) {
    const row = el(`<div class="item">🔁 <b>${esc(x.person)}</b>
      <span class="attempt">attempt #${x.attempt}</span>
      <div class="muted">${esc(x.task_title)}${x.note ? " · " + esc(x.note) : ""}</div></div>`);
    row.onclick = () => openTask(x.task_id);
    row.style.cursor = "pointer";
    box.appendChild(row);
  }
}

// Generated deliverables (DOCX/PDF/XLSX/PPTX/HTML/PNG the scripter produced).
// Each row is a download link to /api/reports/<name> (served as an attachment).
const REPORT_ICONS = { docx: "📝", pdf: "📕", xlsx: "📊", pptx: "📑",
                       html: "🌐", png: "🖼️", md: "📄" };
async function loadReports() {
  const box = document.getElementById("reports");
  if (!box) return;
  let r;
  try { r = await api("GET", "/api/reports"); }
  catch { box.innerHTML = `<div class="muted">—</div>`; return; }
  const reports = (r && r.reports) || [];
  box.innerHTML = reports.length ? ""
    : `<div class="muted">None yet. Ask Hermes (swarm mode) to "generate a report".</div>`;
  for (const x of reports) {
    const icon = REPORT_ICONS[x.type] || "📄";
    const row = el(`<div class="item"><a href="${esc(x.url)}" download
        title="Download ${esc(x.filename)}">${icon} ${esc(x.filename)}</a>
      <div class="muted">${esc(String(x.type).toUpperCase())} · ${x.size_kb} KB</div></div>`);
    box.appendChild(row);
  }
}

// Scoped filesystem explorer (Roadmap #4): browse the allowed roots
// (project / Downloads / Documents). Read-only here -- the agent's fs_* tools do
// the writing. Folders drill in; files are download links to /api/files/content.
let FILES_PATH = "";
async function loadFiles(path) {
  if (path !== undefined) FILES_PATH = path;
  const box = document.getElementById("files");
  if (!box) return;
  let r;
  const q = FILES_PATH ? "?path=" + encodeURIComponent(FILES_PATH) : "";
  try { r = await api("GET", "/api/files" + q); }
  catch { box.innerHTML = `<div class="muted">—</div>`; return; }
  box.innerHTML = "";
  if (r.roots) {                                   // top level: the roots
    if (!r.roots.length) { box.innerHTML = `<div class="muted">filesystem disabled</div>`; return; }
    for (const root of r.roots) {
      const perms = `${root.read ? "r" : ""}${root.write ? "w" : ""}${root.delete ? "d" : ""}`;
      const lock = root.confirm_required ? " 🔒" : "";
      const row = el(`<div class="item" style="cursor:pointer"><a>📁 ${esc(root.name)}${lock}</a>
        <div class="muted">${esc(perms)}${root.confirm_required ? " · confirm to write" : " · free"}</div></div>`);
      row.querySelector("a").onclick = () => loadFiles(root.name);
      box.appendChild(row);
    }
    return;
  }
  const up = FILES_PATH.includes("/") ? FILES_PATH.replace(/\/[^/]*$/, "") : "";
  const crumb = el(`<div class="item" style="cursor:pointer"><a title="up">⬅️ ${esc(FILES_PATH)}</a></div>`);
  crumb.querySelector("a").onclick = () => loadFiles(up);
  box.appendChild(crumb);
  for (const d of (r.folders || [])) {
    const row = el(`<div class="item" style="cursor:pointer"><a>📁 ${esc(d.name)}/</a></div>`);
    row.querySelector("a").onclick = () => loadFiles(d.path);
    box.appendChild(row);
  }
  for (const f of (r.files || [])) {
    const row = el(`<div class="item"><a href="/api/files/content?path=${encodeURIComponent(f.path)}"
        download title="Download ${esc(f.name)}">📄 ${esc(f.name)}</a>
      <div class="muted">${esc(String(f.type).toUpperCase())} · ${f.size_kb} KB</div></div>`);
    box.appendChild(row);
  }
  if (!(r.folders || []).length && !(r.files || []).length)
    box.appendChild(el(`<div class="muted">empty</div>`));
}

// ---- creating things ------------------------------------------------------
async function newGroup() {
  const name = prompt("New group name:");
  if (!name) return;
  await api("POST", "/api/groups", { name });
  refresh();
}
async function newSubgroup() {
  if (!SEL.groupId) return alert("Select a group first.");
  const name = prompt("New subgroup name:");
  if (!name) return;
  await api("POST", `/api/groups/${SEL.groupId}/subgroups`, { name });
  refresh();
}
function newTask() {
  if (!SEL.groupId) return alert("Select a group first.");
  openTaskEditor(null);
}

// ---- task modal (view/edit) ----------------------------------------------
function closeModal() { document.getElementById("modal").style.display = "none"; }
function showModal(node) {
  const card = document.getElementById("modal-card");
  card.innerHTML = ""; card.appendChild(node);
  document.getElementById("modal").style.display = "flex";
}

function openTaskEditor(task) {
  const isNew = !task;
  task = task || { title: "", description: "", priority: "medium", status: "todo", due_date: "", person: "" };
  const card = el(`<div>
    <h3>${isNew ? "New task" : "Edit task"}</h3>
    <div class="field"><label>Title</label><input id="f-title" value="${esc(task.title)}"></div>
    <div class="field"><label>Description</label><textarea id="f-desc" rows="3">${esc(task.description || "")}</textarea></div>
    <div class="row2">
      <div class="field"><label>Status</label><select id="f-status">
        ${["todo","in_progress","blocked","done"].map(s => `<option ${task.status===s?"selected":""}>${s}</option>`).join("")}
      </select></div>
      <div class="field"><label>Priority</label><select id="f-prio">
        ${["low","medium","high","urgent"].map(p => `<option ${task.priority===p?"selected":""}>${p}</option>`).join("")}
      </select></div>
    </div>
    <div class="row2">
      <div class="field"><label>Due date</label><input id="f-due" type="date" value="${esc(task.due_date || "")}"></div>
      <div class="field"><label>Person</label><input id="f-person" value="${esc(task.person || "")}"></div>
    </div>
    <div class="modal-actions">
      <button class="danger" id="f-cancel">Cancel</button>
      <button class="primary" id="f-save">${isNew ? "Create" : "Save"}</button>
    </div>
  </div>`);
  card.querySelector("#f-cancel").onclick = closeModal;
  card.querySelector("#f-save").onclick = async () => {
    const payload = {
      title: card.querySelector("#f-title").value.trim(),
      description: card.querySelector("#f-desc").value,
      status: card.querySelector("#f-status").value,
      priority: card.querySelector("#f-prio").value,
      due_date: card.querySelector("#f-due").value || null,
      person: card.querySelector("#f-person").value || null,
    };
    if (!payload.title) return alert("Title required.");
    try {
      if (isNew) {
        payload.group_id = SEL.groupId;
        payload.subgroup_id = SEL.subId;
        await api("POST", "/api/tasks", payload);
      } else {
        await api("PATCH", `/api/tasks/${task.id}`, payload);
      }
      closeModal(); refresh();
    } catch (e) { alert(e.message); }
  };
  showModal(card);
}

function auditLine(e) {
  const ts = (e.ts || "").slice(0, 16).replace("T", " ");
  let what = e.action;
  if (e.action === "task_created") what = "created";
  else if (e.action === "task_completed") what = "✅ completed";
  else if (e.action === "task_deleted") what = "deleted";
  else if (e.action === "follow_up_added") what = `🔁 follow-up with ${e.person} (attempt #${e.attempt})`;
  else if (e.action === "task_updated") what = "changed " + Object.keys(e.changes || {}).join(", ");
  return `<div class="list-row"><span class="muted small">${esc(ts)}</span><span class="grow">${esc(what)}</span></div>`;
}

async function openTask(taskId) {
  const t = await api("GET", `/api/tasks/${taskId}`);
  let history = [];
  try { history = await api("GET", `/api/audit?task_id=${taskId}&limit=100`); } catch (e) {}
  const reminders = (t.reminders || []).map(r => `
    <div class="list-row">
      <span class="grow">${r.done ? "✓ " : "⏰ "}<b>${esc(r.remind_at)}</b> · ${esc(r.message)}</span>
      ${r.done ? '<span class="done-tag">done</span>'
               : `<button class="mini" data-done-rem="${r.id}">done</button>`}
    </div>`).join("") || `<div class="muted small">none</div>`;

  const followups = (t.follow_ups || []).map(f => `
    <div class="list-row">
      <span class="grow">${f.done ? "✓ " : "🔁 "}<b>${esc(f.person)}</b>
        <span class="attempt">attempt #${f.attempt}</span>
        ${f.note ? "· " + esc(f.note) : ""}${f.due_date ? " · by " + esc(f.due_date) : ""}</span>
      ${f.done ? '<span class="done-tag">done</span>'
               : `<button class="mini" data-done-fu="${f.id}">done</button>`}
    </div>`).join("") || `<div class="muted small">none</div>`;

  const notes = (t.notes || []).map(n =>
    `<div class="list-row"><span class="grow">📝 ${esc(n.text)}</span></div>`).join("")
    || `<div class="muted small">none</div>`;

  const card = el(`<div>
    <h3>${esc(t.title)}</h3>
    <div class="meta" style="margin-bottom:6px">
      <span class="badge status s-${t.status}">${t.status.replace("_"," ")}</span>
      <span class="badge p-${t.priority}">${t.priority}</span>
      ${t.due_date ? `<span class="badge due">📅 ${esc(t.due_date)}</span>` : ""}
      ${t.person ? `<span class="badge person">👤 ${esc(t.person)}</span>` : ""}
      ${t.completed_at ? `<span class="badge s-done status">✅ ${esc(t.completed_at.slice(0,16).replace("T"," "))} UTC</span>` : ""}
    </div>
    ${t.description ? `<div class="muted">${esc(t.description)}</div>` : ""}

    <div class="detail-section"><h4>⏰ Reminders</h4>${reminders}
      <div class="inline-form">
        <input id="r-date" type="date"><input id="r-msg" placeholder="message (optional)">
        <button class="mini" id="r-add">add</button>
      </div>
    </div>

    <div class="detail-section"><h4>🔁 Follow-ups <span class="muted small">(each add = one more chase)</span></h4>${followups}
      <div class="inline-form">
        <input id="fu-person" placeholder="person"><input id="fu-note" placeholder="note (optional)">
        <button class="mini" id="fu-add">add</button>
      </div>
    </div>

    <div class="detail-section"><h4>📝 Notes</h4>${notes}
      <div class="inline-form">
        <input id="n-text" placeholder="add a note…"><button class="mini" id="n-add">add</button>
      </div>
    </div>

    <div class="detail-section"><h4>🕘 History <span class="muted small">(audit log — append-only)</span></h4>
      ${history.length ? history.map(auditLine).join("") : '<div class="muted small">no events</div>'}
    </div>

    <div class="modal-actions">
      <button class="danger" id="t-del">Delete task</button>
      <div style="display:flex;gap:8px">
        <button id="t-edit">Edit</button>
        <button class="primary" id="t-close">Close</button>
      </div>
    </div>
  </div>`);

  card.querySelectorAll("[data-done-rem]").forEach(b => b.onclick = async () => {
    await api("POST", `/api/tasks/${t.id}/reminders/${b.dataset.doneRem}/complete`); openTask(t.id); refresh();
  });
  card.querySelectorAll("[data-done-fu]").forEach(b => b.onclick = async () => {
    await api("POST", `/api/tasks/${t.id}/followups/${b.dataset.doneFu}/complete`); openTask(t.id); refresh();
  });
  card.querySelector("#r-add").onclick = async () => {
    const d = card.querySelector("#r-date").value;
    if (!d) return alert("pick a date");
    await api("POST", `/api/tasks/${t.id}/reminders`, { remind_at: d, message: card.querySelector("#r-msg").value });
    openTask(t.id); refresh();
  };
  card.querySelector("#fu-add").onclick = async () => {
    const p = card.querySelector("#fu-person").value.trim();
    if (!p) return alert("who?");
    await api("POST", `/api/tasks/${t.id}/followups`, { person: p, note: card.querySelector("#fu-note").value });
    openTask(t.id); refresh();
  };
  card.querySelector("#n-add").onclick = async () => {
    const x = card.querySelector("#n-text").value.trim();
    if (!x) return;
    await api("POST", `/api/tasks/${t.id}/notes`, { text: x });
    openTask(t.id); refresh();
  };
  card.querySelector("#t-del").onclick = async () => {
    if (!confirm("Delete this task?")) return;
    await api("DELETE", `/api/tasks/${t.id}`); closeModal(); refresh();
  };
  card.querySelector("#t-edit").onclick = () => openTaskEditor(t);
  card.querySelector("#t-close").onclick = closeModal;
  showModal(card);
}

// ---- agent chat -----------------------------------------------------------
function addMsg(cls, text) {
  const log = document.getElementById("chat-log");
  const m = el(`<div class="msg ${cls}">${esc(text)}</div>`);
  log.appendChild(m); log.scrollTop = log.scrollHeight;
  return m;
}
// Run one agent turn for `text` in the currently-selected mode/brain, render it,
// and RETURN the reply string (or null on error). Shared by the typed form and
// the voice flow so both ride the same machinery.
async function runAgentTurn(text) {
  addMsg("user", text);
  const mode = (document.getElementById("chat-mode") || {}).value || "single";
  const thinking = addMsg("bot",
    mode === "swarm"
      ? "…orchestrator is delegating to specialists (watch the live status strip above)"
      : "…thinking (browser tasks can take a few minutes)");
  const llm = (document.getElementById("llm-profile") || {}).value || "mock";
  // In swarm mode, poll the status board so the user sees children move
  // through pending → in_progress → done in real time.
  let stopPoll = mode === "swarm" ? startSwarmPolling() : null;
  let reply = null;
  try {
    const res = await api("POST", "/api/agent/chat", { message: text, llm, mode });
    thinking.textContent = res.reply;
    reply = res.reply;
  } catch (err) {
    thinking.textContent = "⚠️ " + err.message;
    thinking.classList.add("sys");
  } finally {
    if (stopPoll) stopPoll();
  }
  if (mode === "swarm") renderSwarmStrip();  // one last refresh to capture final reports
  refresh();   // the agent (or a child) may have changed the board
  return reply;
}

async function sendChat(e) {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) return false;
  input.value = "";
  await runAgentTurn(text);
  return false;
}

// ---- swarm mode -----------------------------------------------------------
let SWARM_SPECIALISTS = null;

function onModeChange() {
  const mode = (document.getElementById("chat-mode") || {}).value || "single";
  const strip = document.getElementById("swarm-strip");
  if (mode === "swarm") {
    strip.style.display = "";
    renderSwarmStrip();
    if (!SWARM_SPECIALISTS) {
      api("GET", "/api/agent/swarm/specialists")
        .then(r => { SWARM_SPECIALISTS = r; renderSwarmStrip(); })
        .catch(() => {});
    }
  } else {
    strip.style.display = "none";
  }
}

function renderSwarmStrip(snap) {
  const strip = document.getElementById("swarm-strip");
  if (!strip) return;
  const tasks = (snap && snap.tasks) || [];
  // Build the row: one chip per specialist showing its latest task's status,
  // plus task count. Order matches the static roster.
  const order = SWARM_SPECIALISTS || [];
  const latestBy = {};
  for (const t of tasks) latestBy[t.child] = t;
  const chips = order.length ? order.map(s => {
    const t = latestBy[s.name];
    const status = t ? t.status : "idle";
    const label = t ? `${s.name} · ${status}${t.elapsed_s ? " · " + t.elapsed_s + "s" : ""}` : `${s.name} · idle`;
    const title = t
      ? `${s.title}\n\nLatest task: ${t.description}\n\nReport: ${t.report || (t.error || "(none yet)")}`
      : `${s.title}\n\n${s.capabilities}`;
    return `<span class="swarm-chip swarm-${status}" title="${esc(title)}">${esc(label)}</span>`;
  }).join("") : `<span class="muted small">loading specialists…</span>`;
  strip.innerHTML = chips;
}

function startSwarmPolling() {
  let stopped = false;
  (async function loop() {
    while (!stopped) {
      try {
        const snap = await api("GET", "/api/agent/swarm/status");
        renderSwarmStrip(snap);
      } catch (e) {}
      await new Promise(r => setTimeout(r, 1500));
    }
  })();
  return () => { stopped = true; };
}
document.getElementById("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); document.getElementById("chat-form").requestSubmit(); }
});

// ---- sessions -------------------------------------------------------------
async function showRecentSessions() {
  try {
    const recent = await api("GET", "/api/agent/sessions?limit=5");
    if (recent && recent.length) {
      addMsg("sys", "Recent sessions: " +
        recent.map(s => "“" + s.title + "”").join(" · ") +
        " — I'll pull one up if your next request relates to it.");
    }
  } catch (e) {}
}

async function newSession() {
  try {
    const res = await api("POST", "/api/agent/new-session");
    document.getElementById("chat-log").innerHTML = "";
    if (res.archived)
      addMsg("sys", "Archived “" + res.archived.title + "”. Fresh session started.");
    else
      addMsg("sys", "Fresh session started.");
    await showRecentSessions();
  } catch (e) { addMsg("sys", "⚠️ " + e.message); }
}

// ---- voice (whisper STT + piper TTS) + "Paimon" wake word -----------------
// Browser captures mic audio, encodes a 16 kHz mono WAV (no ffmpeg), posts it to
// /api/voice/stt, runs the SAME agent turn, then plays /api/voice/tts. A simple
// RMS VAD auto-stops on silence. In Paimon wake mode the loop listens
// continuously but only ACTS on an utterance that starts with the wake word.
let VOICE = { enabled: false, wake_word: "paimon" };
let VOICE_BUSY = false;        // a full voice turn is in flight
let RECORDING = null;          // { stop() } while actively recording
let MIC = null;                // shared { stream, ctx } reused across the wake loop

// Paimon flavour — perky companion acks + light spoken lead-ins (the on-screen
// reply text stays the agent's accurate words; only the SPOKEN version is wrapped).
const PAIMON_ACKS = ["Hmm? Paimon's here!", "Ehe? What is it, Traveler?",
  "Paimon's listening!", "Yes, Traveler?", "Paimon's right here!"];
const PAIMON_LEADS = ["Okay, Traveler! ", "Got it, Traveler! ", "Ehe! ",
  "Sure thing, Traveler! ", "Mhm! ", "", ""];
const rnd = (a) => a[Math.floor(Math.random() * a.length)];

async function initVoice() {
  try { VOICE = await api("GET", "/api/voice/status"); } catch (e) { VOICE = { enabled: false }; }
  if (!VOICE.enabled) return;
  document.getElementById("mic-btn").style.display = "";
  document.getElementById("voice-bar").style.display = "";
  setVoiceStatus(`🧚 Paimon ready · ${VOICE.model} / ${VOICE.voice}`);
}

function setVoiceStatus(s) {
  const e = document.getElementById("voice-status");
  if (e) e.textContent = s;
}
function speakOn() { return (document.getElementById("speak-toggle") || {}).checked; }
function handsFree() { return (document.getElementById("handsfree-toggle") || {}).checked; }
function wakeMode() { return (document.getElementById("wake-toggle") || {}).checked; }
function continuous() { return handsFree() || wakeMode(); }

// Shared mic kept open across the wake loop, so we don't re-prompt / blink the
// mic on every utterance. Released when continuous listening stops.
async function ensureMic() {
  if (MIC) { try { await MIC.ctx.resume(); } catch (e) {} return MIC; }
  const stream = await navigator.mediaDevices.getUserMedia(
    { audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  try { await ctx.resume(); } catch (e) {}
  MIC = { stream, ctx };
  return MIC;
}
function releaseMic() {
  if (!MIC) return;
  try { MIC.stream.getTracks().forEach(t => t.stop()); MIC.ctx.close(); } catch (e) {}
  MIC = null;
}

// Capture one utterance from the shared mic → 16 kHz mono 16-bit WAV Blob.
// Auto-stops after ~silenceMs of quiet (once speech started), or maxMs; in the
// wake loop noSpeechMs recycles fast when nobody is talking. RECORDING.stop()
// forces an early finish (the manual-stop click).
function recordWavUtterance({ maxMs = 15000, silenceMs = 1000, noSpeechMs = 0 } = {}) {
  return new Promise(async (resolve, reject) => {
    let mic;
    try { mic = await ensureMic(); }
    catch (e) { return reject(new Error("mic permission denied")); }
    const { ctx, stream } = mic;
    const source = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const sink = ctx.createGain(); sink.gain.value = 0;   // silent sink (no mic echo)
    const chunks = [];
    const inRate = ctx.sampleRate;
    const cutNoSpeech = noSpeechMs || maxMs;
    let lastVoice = performance.now(), started = performance.now(), spoke = false, done = false;

    function finish() {
      if (done) return; done = true;
      try { proc.disconnect(); source.disconnect(); sink.disconnect(); } catch (e) {}
      // Keep the shared stream/ctx OPEN (released by releaseMic when listening stops).
      resolve(encodeWav(mergeChunks(chunks), inRate, 16000));
    }
    RECORDING = { stop: finish };
    source.connect(proc); proc.connect(sink); sink.connect(ctx.destination);
    proc.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(data));
      let sum = 0; for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      const rms = Math.sqrt(sum / data.length);
      const now = performance.now();
      if (rms > 0.015) { lastVoice = now; spoke = true; }
      if (spoke && now - lastVoice > silenceMs) finish();
      else if (!spoke && now - started > cutNoSpeech) finish();
      else if (now - started > maxMs) finish();
    };
  });
}

function mergeChunks(chunks) {
  let len = 0; for (const c of chunks) len += c.length;
  const out = new Float32Array(len); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// nearest-neighbour downsample float32[-1,1] inRate→outRate, encode 16-bit PCM WAV.
function encodeWav(samples, inRate, outRate) {
  const ratio = inRate / outRate;
  const outLen = Math.max(0, Math.floor(samples.length / ratio));
  const pcm = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const s = Math.max(-1, Math.min(1, samples[Math.floor(i * ratio)] || 0));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  const buf = new ArrayBuffer(44 + pcm.length * 2), dv = new DataView(buf);
  const wr = (off, str) => { for (let i = 0; i < str.length; i++) dv.setUint8(off + i, str.charCodeAt(i)); };
  wr(0, "RIFF"); dv.setUint32(4, 36 + pcm.length * 2, true); wr(8, "WAVE");
  wr(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, outRate, true); dv.setUint32(28, outRate * 2, true);
  dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  wr(36, "data"); dv.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) dv.setInt16(44 + i * 2, pcm[i], true);
  return new Blob([buf], { type: "audio/wav" });
}

async function sttFetch(wav) {
  const r = await fetch("/api/voice/stt", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav });
  if (!r.ok) throw new Error(await r.text() || r.statusText);
  return (await r.json()).text || "";
}

// --- "Paimon" wake-word matching (fuzzy, tolerant of whisper mishears) ------
function lev(a, b) {
  const m = a.length, n = b.length;
  const d = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) d[i][0] = i;
  for (let j = 0; j <= n; j++) d[0][j] = j;
  for (let i = 1; i <= m; i++) for (let j = 1; j <= n; j++)
    d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[m][n];
}
function fuzzyWake(w, wake) {
  if (!w) return false;
  if (w === wake || w.startsWith("paim")) return true;   // paimon / paimon's
  return lev(w, wake) <= 2;                               // payman, pamon, pimon, paymon…
}
// Returns { hit, rest } — rest = the command spoken AFTER the wake word (may be "").
function matchWake(text) {
  const wake = (VOICE.wake_word || "paimon").toLowerCase();
  const words = text.toLowerCase().replace(/[^a-z\s]/g, " ").replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
  if (!words.length) return { hit: false, rest: "" };
  if (fuzzyWake(words[0], wake)) return { hit: true, rest: words.slice(1).join(" ") };
  if (words.length > 1 && fuzzyWake(words[0] + words[1], wake))   // "pay mon"
    return { hit: true, rest: words.slice(2).join(" ") };
  return { hit: false, rest: "" };
}
function paimonize(t) { return rnd(PAIMON_LEADS) + t; }
async function paimonAck() { try { await speak(rnd(PAIMON_ACKS)); } catch (e) {} }

function onMicClick() {
  if (!VOICE.enabled) return;
  if (RECORDING) { RECORDING.stop(); return; }   // click while recording = stop now
  voiceTurn();
}

// Toggling Paimon wake mode on: greet + kick off the continuous listen loop.
function onWakeToggle() {
  if (wakeMode()) {
    setVoiceStatus(`🧚 Paimon is listening… say “${VOICE.wake_word || "Paimon"}”`);
    if (speakOn()) speak("Paimon's here to help, Traveler! Just say my name.").catch(() => {});
    if (!VOICE_BUSY) voiceTurn();
  } else {
    setVoiceStatus("🧚 Paimon ready");
    if (!handsFree()) releaseMic();
  }
}

async function voiceTurn() {
  if (VOICE_BUSY) return;
  VOICE_BUSY = true;
  const mic = document.getElementById("mic-btn");
  const wake = wakeMode();

  function endTurn() {
    mic.classList.remove("recording");
    RECORDING = null;
    setVoiceStatus(wakeMode() ? `👂 say “${VOICE.wake_word || "Paimon"}”…` : "🧚 Paimon ready");
    VOICE_BUSY = false;
    if (continuous()) setTimeout(voiceTurn, 500);
    else releaseMic();
  }

  // 1. Listen. In wake mode recycle quickly while nobody's talking.
  mic.classList.add("recording");
  setVoiceStatus(wake ? "👂 listening for “Paimon”…" : "🎙️ listening… (click mic to stop)");
  let wav;
  try { wav = await recordWavUtterance(wake ? { noSpeechMs: 2800 } : {}); }
  catch (e) { setVoiceStatus("⚠️ " + e.message); return endTurn(); }
  RECORDING = null; mic.classList.remove("recording");

  // 2. Transcribe.
  let text = "";
  try { text = await sttFetch(wav); }
  catch (e) { setVoiceStatus("⚠️ STT: " + e.message); return endTurn(); }
  let command = text.trim();

  // 3. Wake gating (Paimon mode only): act only if she was named.
  if (wake) {
    if (!command) return endTurn();                 // silence → keep listening
    const w = matchWake(text);
    if (!w.hit) { setVoiceStatus(`😴 (say “${VOICE.wake_word || "Paimon"}” to wake me)`); return endTurn(); }
    if (w.rest) {
      command = w.rest;                              // "Paimon, do X" → "do X"
    } else {
      await paimonAck();                             // bare "Paimon" → ack, then take the next utterance
      setVoiceStatus("🎙️ listening, Traveler…");
      try { command = (await recordWavUtterance({ noSpeechMs: 4000 }).then(sttFetch)).trim(); }
      catch (e) { setVoiceStatus("⚠️ " + e.message); return endTurn(); }
      if (!command) { setVoiceStatus("🤷 didn't catch that, Traveler"); return endTurn(); }
    }
  } else if (!command) {
    setVoiceStatus("🤷 didn't catch that — click the mic to try again");
    return endTurn();
  }

  // 4. Run the agent + speak the reply (Paimon-wrapped when in companion mode).
  setVoiceStatus("🤖 thinking…");
  const reply = await runAgentTurn(command);
  if (reply && speakOn()) {
    setVoiceStatus("🔊 speaking…");
    try { await speak(continuous() ? paimonize(reply) : reply); }
    catch (e) { setVoiceStatus("⚠️ TTS: " + e.message); }
  }
  return endTurn();
}

function speak(text) {
  return new Promise(async (resolve, reject) => {
    try {
      const r = await fetch("/api/voice/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
      if (!r.ok) throw new Error(await r.text() || r.statusText);
      const url = URL.createObjectURL(await r.blob());
      const audio = new Audio(url);
      audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
      audio.onerror = () => { URL.revokeObjectURL(url); reject(new Error("playback failed")); };
      audio.play().catch(reject);
    } catch (e) { reject(e); }
  });
}

// ---- boot -----------------------------------------------------------------
addMsg("sys", "Ask me to organise tasks, set reminders ('remind me in 20 mins'), browse the web, or log follow-ups.");
initVoice();
showRecentSessions();
refresh();
loadFiles();                 // file-explorer roots (once; navigation is user-driven)
setInterval(loadReminders, 60000);

// Live reminder notifications: ask permission, then poll every 15s.
ensureNotifyPermission();
pollNotifications();
setInterval(pollNotifications, 15000);
