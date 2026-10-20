// Paimon mobile PWA — minimalist phone chat client for the Hermes backend.
//
// Talks to the SAME FastAPI endpoints the desktop UI uses (/api/agent/chat,
// /api/voice/*, /api/agent/new-session) plus the new delivery channel
// (/api/deliveries + /api/outbox/<file>) so the agent can push files/media to
// the phone. Everything is configurable for the "installed app pointing at my
// PC" case: a Server URL + an optional access token, stored on the device.

// ===========================================================================
//  settings / storage
// ===========================================================================
const LS = {
  get: (k, d) => { try { const v = localStorage.getItem("paimon." + k); return v === null ? d : v; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem("paimon." + k, v); } catch {} },
};
// Server base: empty = same origin (the normal "opened from the PC URL" case).
// Only set when running as a bundled APK that must reach a specific PC address.
let API_BASE = (LS.get("server", "") || "").replace(/\/+$/, "");
let TOKEN = LS.get("token", "");

const $ = (id) => document.getElementById(id);
const esc = (s) => (s == null ? "" : String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])));

// ===========================================================================
//  fetch helpers (auth-aware)
// ===========================================================================
function abs(path) {
  if (/^https?:\/\//i.test(path)) return path;
  return API_BASE + path;
}
// For <img>/<audio>/<video>/<a download> that can't send an Authorization
// header: carry the token (and base) in the URL instead.
function mediaUrl(path) {
  let u = abs(path);
  if (TOKEN) u += (u.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(TOKEN);
  return u;
}
async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  if (TOKEN) opt.headers["Authorization"] = "Bearer " + TOKEN;
  const r = await fetch(abs(path), opt);
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    const e = new Error(msg); e.status = r.status; throw e;
  }
  return r.status === 204 ? null : r.json();
}

// ===========================================================================
//  chat rendering
// ===========================================================================
const log = $("chat-log");
function dropEmpty() { const e = $("empty-state"); if (e) e.remove(); }
function scrollDown() { log.scrollTop = log.scrollHeight; }

// Minimal, safe rich text: escape, then linkify [md](links), bare URLs, **bold**.
function rich(text) {
  let h = esc(text);
  h = h.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|\/[^\s)]+)\)/g,
    (_m, label, url) => `<a href="${esc(linkHref(url))}" target="_blank" rel="noopener">${esc(label)}</a>`);
  h = h.replace(/(^|[\s(])((?:https?:\/\/|\/api\/)[^\s)]+)(?=$|[\s).,!?])/g,
    (_m, pre, url) => `${pre}<a href="${esc(linkHref(url))}" target="_blank" rel="noopener">${esc(url)}</a>`);
  h = h.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  return h;
}
function linkHref(url) { return /^https?:\/\//i.test(url) ? url : mediaUrl(url); }

function addMsg(cls, text) {
  dropEmpty();
  const m = document.createElement("div");
  m.className = "msg " + cls;
  if (cls === "bot" || cls === "sys") m.innerHTML = rich(text); else m.textContent = text;
  log.appendChild(m); scrollDown();
  return m;
}
function thinkingBubble(label) {
  dropEmpty();
  const m = document.createElement("div");
  m.className = "msg bot thinking";
  m.innerHTML = esc(label || "") + ' <span class="dots"><i></i><i></i><i></i></span>';
  log.appendChild(m); scrollDown();
  return m;
}

// ===========================================================================
//  delivered files / media (the agent "sends me a file")
// ===========================================================================
const FILE_ICON = { image: "🖼️", audio: "🎵", video: "🎬", pdf: "📕", doc: "📝",
  sheet: "📊", slides: "📑", archive: "🗜️", text: "📄", other: "📎" };
let lastDeliverySeq = 0;
let deliveriesPrimed = false;

function renderDelivery(d) {
  dropEmpty();
  const wrap = document.createElement("div");
  wrap.className = "media";
  const src = mediaUrl(d.url);
  if (d.kind === "image") {
    wrap.innerHTML = `<a href="${esc(src)}" target="_blank" rel="noopener"><img src="${esc(src)}" alt="${esc(d.filename)}"/></a>`;
  } else if (d.kind === "audio") {
    wrap.innerHTML = `<audio controls preload="none" src="${esc(src)}"></audio>`;
  } else if (d.kind === "video") {
    wrap.innerHTML = `<video controls preload="metadata" src="${esc(src)}"></video>`;
  } else {
    const ic = FILE_ICON[d.kind] || FILE_ICON.other;
    wrap.innerHTML =
      `<a class="filecard" href="${esc(src)}" target="_blank" rel="noopener" download="${esc(d.filename)}">
         <span class="ic">${ic}</span>
         <span><span class="fn">${esc(d.filename)}</span><br>
           <span class="meta">${esc((d.kind || "file").toUpperCase())} · ${esc(d.size_human || (d.size_kb + " KB"))}</span></span>
         <span class="dl">⬇︎ Save</span>
       </a>`;
  }
  if (d.caption) {
    const cap = document.createElement("div");
    cap.className = "cap"; cap.textContent = d.caption;
    wrap.appendChild(cap);
  }
  log.appendChild(wrap); scrollDown();
}

async function pollDeliveries() {
  let res;
  try { res = await api("GET", "/api/deliveries?since=" + lastDeliverySeq); }
  catch { return; }
  const items = (res && res.deliveries) || [];
  if (!deliveriesPrimed) {
    // First sight: don't replay history — just set the baseline.
    for (const d of items) lastDeliverySeq = Math.max(lastDeliverySeq, d.seq);
    deliveriesPrimed = true;
    return;
  }
  for (const d of items) {
    if (d.seq > lastDeliverySeq) { renderDelivery(d); lastDeliverySeq = d.seq; }
  }
}

// ===========================================================================
//  connection status
// ===========================================================================
async function ping() {
  const c = $("conn");
  try {
    await api("GET", "/api/now");
    c.textContent = "online"; c.className = "conn ok";
    return true;
  } catch (e) {
    c.textContent = e.status === 401 ? "needs token" : "offline";
    c.className = "conn err";
    return false;
  }
}

// ===========================================================================
//  agent turn (shared by typed + voice)
// ===========================================================================
let BUSY = false;
async function runAgentTurn(text) {
  addMsg("user", text);
  const mode = $("mode").value || "single";
  const brain = $("brain").value || "mock";
  const think = thinkingBubble(mode === "swarm" ? "orchestrating specialists" : "thinking");
  let reply = null;
  try {
    const res = await api("POST", "/api/agent/chat", { message: text, llm: brain, mode });
    think.classList.remove("thinking");
    think.innerHTML = rich(res.reply);
    reply = res.reply;
  } catch (err) {
    think.classList.remove("thinking");
    think.classList.add("sys");
    think.textContent = "⚠️ " + (err.message || err);
  }
  scrollDown();
  pollDeliveries();   // surface any file the agent just sent, right away
  return reply;
}

// ===========================================================================
//  composer (typed)
// ===========================================================================
const input = $("input");
function autogrow() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; }
input.addEventListener("input", autogrow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
});
$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || BUSY) return;
  input.value = ""; autogrow();
  BUSY = true; $("send").disabled = true;
  try { await runAgentTurn(text); }
  finally { BUSY = false; $("send").disabled = false; }
});
log.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip) { input.value = chip.dataset.say; $("composer").requestSubmit(); }
});

// ===========================================================================
//  settings sheet
// ===========================================================================
function openSheet() { $("sheet-backdrop").style.display = "block"; $("sheet").style.display = "block"; }
function closeSheet() { $("sheet-backdrop").style.display = "none"; $("sheet").style.display = "none"; }
$("menu-btn").onclick = openSheet;
$("sheet-close").onclick = closeSheet;
$("sheet-backdrop").onclick = closeSheet;

// hydrate controls from storage
$("server-url").value = LS.get("server", "");
$("token").value = LS.get("token", "");
$("brain").value = LS.get("brain", "mock");
$("mode").value = LS.get("mode", "single");
$("speak").checked = LS.get("speak", "1") === "1";
$("handsfree").checked = LS.get("handsfree", "0") === "1";
$("wake").checked = LS.get("wake", "0") === "1";

function bindPersist(id, key, isCheck) {
  const el = $(id);
  el.addEventListener("change", () => {
    LS.set(key, isCheck ? (el.checked ? "1" : "0") : el.value);
    if (key === "server") { API_BASE = (el.value || "").replace(/\/+$/, ""); ping(); }
    if (key === "token") { TOKEN = el.value; ping(); }
  });
}
bindPersist("server-url", "server"); bindPersist("token", "token");
bindPersist("brain", "brain"); bindPersist("mode", "mode");
bindPersist("speak", "speak", true); bindPersist("handsfree", "handsfree", true);
bindPersist("wake", "wake", true);
$("wake").addEventListener("change", onWakeToggle);
$("handsfree").addEventListener("change", () => { if (handsFree() && !VOICE_BUSY) voiceTurn(); });

$("test-conn").onclick = async () => {
  const r = $("test-result"); r.textContent = "…";
  const ok = await ping();
  r.textContent = ok ? "✓ connected" : "✗ " + $("conn").textContent;
};

$("new-btn").onclick = async () => {
  try {
    const res = await api("POST", "/api/agent/new-session");
    log.innerHTML = "";
    addMsg("sys", res.archived ? "Archived “" + res.archived.title + "”. Fresh chat." : "Fresh chat started.");
  } catch (e) { addMsg("sys", "⚠️ " + e.message); }
};

// ===========================================================================
//  PWA install + service worker
// ===========================================================================
let deferredInstall = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault(); deferredInstall = e; $("install-row").style.display = "flex";
});
$("install-btn").onclick = async () => {
  if (!deferredInstall) return;
  deferredInstall.prompt(); await deferredInstall.userChoice;
  deferredInstall = null; $("install-row").style.display = "none";
};
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

// ===========================================================================
//  VOICE — whisper STT + piper TTS + "Paimon" wake word (ported from desktop)
// ===========================================================================
let VOICE = { enabled: false, wake_word: "paimon" };
let VOICE_BUSY = false, RECORDING = null, MIC = null;

const PAIMON_ACKS = ["Hmm? Paimon's here!", "Yes, Traveler?", "Paimon's listening!", "Ehe? What is it?"];
const PAIMON_LEADS = ["Okay, Traveler! ", "Got it! ", "Ehe! ", "Sure thing! ", "Mhm! ", "", ""];
const rnd = (a) => a[Math.floor(Math.random() * a.length)];

function speakOn() { return $("speak").checked; }
function handsFree() { return $("handsfree").checked; }
function wakeMode() { return $("wake").checked; }
function continuous() { return handsFree() || wakeMode(); }
function setPill(s) { const p = $("voice-pill"); if (!s) { p.style.display = "none"; return; } p.style.display = "block"; p.textContent = s; }
function micState(cls) { const m = $("mic"); m.className = "mic" + (cls ? " " + cls : ""); }

async function initVoice() {
  try { VOICE = await api("GET", "/api/voice/status"); } catch { VOICE = { enabled: false }; }
  if (!VOICE.enabled) {
    $("mic").classList.add("hidden");
    $("voice-unavail").style.display = "block";
    return;
  }
  if (wakeMode()) onWakeToggle();
}

async function ensureMic() {
  if (MIC) { try { await MIC.ctx.resume(); } catch {} return MIC; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  try { await ctx.resume(); } catch {}
  MIC = { stream, ctx };
  return MIC;
}
function releaseMic() { if (!MIC) return; try { MIC.stream.getTracks().forEach((t) => t.stop()); MIC.ctx.close(); } catch {} MIC = null; }

function recordWavUtterance({ maxMs = 15000, silenceMs = 1000, noSpeechMs = 0 } = {}) {
  return new Promise(async (resolve, reject) => {
    let mic;
    try { mic = await ensureMic(); } catch { return reject(new Error("mic permission denied")); }
    const { ctx, stream } = mic;
    const source = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const sink = ctx.createGain(); sink.gain.value = 0;
    const chunks = []; const inRate = ctx.sampleRate;
    const cutNoSpeech = noSpeechMs || maxMs;
    let lastVoice = performance.now(), started = performance.now(), spoke = false, done = false;
    function finish() {
      if (done) return; done = true;
      try { proc.disconnect(); source.disconnect(); sink.disconnect(); } catch {}
      resolve(encodeWav(mergeChunks(chunks), inRate, 16000));
    }
    RECORDING = { stop: finish };
    source.connect(proc); proc.connect(sink); sink.connect(ctx.destination);
    proc.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(data));
      let sum = 0; for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      const rms = Math.sqrt(sum / data.length), now = performance.now();
      if (rms > 0.015) { lastVoice = now; spoke = true; }
      if (spoke && now - lastVoice > silenceMs) finish();
      else if (!spoke && now - started > cutNoSpeech) finish();
      else if (now - started > maxMs) finish();
    };
  });
}
function mergeChunks(chunks) { let len = 0; for (const c of chunks) len += c.length; const out = new Float32Array(len); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out; }
function encodeWav(samples, inRate, outRate) {
  const ratio = inRate / outRate, outLen = Math.max(0, Math.floor(samples.length / ratio));
  const pcm = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) { const s = Math.max(-1, Math.min(1, samples[Math.floor(i * ratio)] || 0)); pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF; }
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
  const headers = { "Content-Type": "audio/wav" };
  if (TOKEN) headers["Authorization"] = "Bearer " + TOKEN;
  const r = await fetch(abs("/api/voice/stt"), { method: "POST", headers, body: wav });
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return (await r.json()).text || "";
}
function speak(text) {
  return new Promise(async (resolve, reject) => {
    try {
      const headers = { "Content-Type": "application/json" };
      if (TOKEN) headers["Authorization"] = "Bearer " + TOKEN;
      const r = await fetch(abs("/api/voice/tts"), { method: "POST", headers, body: JSON.stringify({ text }) });
      if (!r.ok) throw new Error((await r.text()) || r.statusText);
      const url = URL.createObjectURL(await r.blob());
      const audio = new Audio(url);
      audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
      audio.onerror = () => { URL.revokeObjectURL(url); reject(new Error("playback failed")); };
      audio.play().catch(reject);
    } catch (e) { reject(e); }
  });
}

// fuzzy "Paimon" wake matching
function lev(a, b) {
  const m = a.length, n = b.length, d = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) d[i][0] = i; for (let j = 0; j <= n; j++) d[0][j] = j;
  for (let i = 1; i <= m; i++) for (let j = 1; j <= n; j++) d[i][j] = Math.min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1] + (a[i-1] === b[j-1] ? 0 : 1));
  return d[m][n];
}
function fuzzyWake(w, wake) { if (!w) return false; if (w === wake || w.startsWith("paim")) return true; return lev(w, wake) <= 2; }
function matchWake(text) {
  const wake = (VOICE.wake_word || "paimon").toLowerCase();
  const words = text.toLowerCase().replace(/[^a-z\s]/g, " ").replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
  if (!words.length) return { hit: false, rest: "" };
  if (fuzzyWake(words[0], wake)) return { hit: true, rest: words.slice(1).join(" ") };
  if (words.length > 1 && fuzzyWake(words[0] + words[1], wake)) return { hit: true, rest: words.slice(2).join(" ") };
  return { hit: false, rest: "" };
}
function paimonize(t) { return rnd(PAIMON_LEADS) + t; }
async function paimonAck() { try { await speak(rnd(PAIMON_ACKS)); } catch {} }

$("mic").onclick = () => {
  if (!VOICE.enabled) return;
  if (RECORDING) { RECORDING.stop(); return; }
  voiceTurn();
};
function onWakeToggle() {
  if (!VOICE.enabled) return;
  if (wakeMode()) {
    setPill(`🧚 listening… say “${VOICE.wake_word || "Paimon"}”`);
    if (speakOn()) speak("Paimon's here, Traveler! Just say my name.").catch(() => {});
    if (!VOICE_BUSY) voiceTurn();
  } else { setPill(null); if (!handsFree()) releaseMic(); }
}

async function voiceTurn() {
  if (VOICE_BUSY || !VOICE.enabled) return;
  VOICE_BUSY = true;
  const wake = wakeMode();
  function endTurn() {
    micState(""); RECORDING = null; VOICE_BUSY = false;
    if (continuous()) { setPill(wakeMode() ? `👂 say “${VOICE.wake_word || "Paimon"}”…` : "🎙️ listening…"); setTimeout(voiceTurn, 400); }
    else { setPill(null); releaseMic(); }
  }
  micState("recording");
  setPill(wake ? "👂 listening for “Paimon”…" : "🎙️ listening… (tap mic to stop)");
  let wav;
  try { wav = await recordWavUtterance(wake ? { noSpeechMs: 2800 } : {}); }
  catch (e) { setPill("⚠️ " + e.message); return endTurn(); }
  RECORDING = null;
  let text = "";
  try { text = await sttFetch(wav); } catch (e) { setPill("⚠️ STT: " + e.message); return endTurn(); }
  let command = text.trim();
  if (wake) {
    if (!command) return endTurn();
    const w = matchWake(text);
    if (!w.hit) { setPill(`😴 (say “${VOICE.wake_word || "Paimon"}”)`); return endTurn(); }
    if (w.rest) { command = w.rest; }
    else {
      await paimonAck(); setPill("🎙️ listening, Traveler…");
      try { command = (await recordWavUtterance({ noSpeechMs: 4000 }).then(sttFetch)).trim(); }
      catch (e) { setPill("⚠️ " + e.message); return endTurn(); }
      if (!command) { setPill("🤷 didn't catch that"); return endTurn(); }
    }
  } else if (!command) { setPill("🤷 didn't catch that — tap the mic"); return endTurn(); }

  micState("thinking"); setPill("🤖 thinking…");
  const reply = await runAgentTurn(command);
  if (reply && speakOn()) {
    micState("speaking"); setPill("🔊 speaking…");
    try { await speak(continuous() ? paimonize(reply) : reply); } catch (e) { setPill("⚠️ TTS: " + e.message); }
  }
  return endTurn();
}

// ===========================================================================
//  boot
// ===========================================================================
(async function boot() {
  await ping();
  await pollDeliveries();         // prime the baseline (don't replay history)
  initVoice();
  setInterval(pollDeliveries, 6000);
  setInterval(ping, 20000);
})();
