"""Central configuration for the chat agent.

Everything points at your local openai-mocker, which speaks the OpenAI
Chat Completions API but is actually driving chatgpt.com under the hood.
"""

import os


def _load_dotenv() -> None:
    """Minimal, dependency-free .env loader: read agent/.env (KEY=VALUE lines)
    into the environment WITHOUT overriding real env vars. Lets you keep secrets
    like the Gmail app password in a local file instead of the shell/chat."""
    path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

# --- LLM backend (the openai-mocker) ---------------------------------------
# The mock ignores the API key, but the OpenAI SDK requires a non-empty string.
BASE_URL = os.environ.get("MOCK_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("MOCK_API_KEY", "sk-mock-ignored")
MODEL = os.environ.get("MOCK_MODEL", "gpt-4o")

# --- Local model backend (K2-Horizon) --------------------------------------
# agent/local_model/serve_k2.py serves IFM/K2-Horizon-3.7B from local weights
# behind the same OpenAI wire protocol, so it reuses the 'openai' provider
# branch in every build_model. Start it with agent/start_k2.bat, then pick it
# with LLM_PROFILE=local (or `llm: "local"` per request in the web chat).
LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8100/v1")
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY", "sk-local-ignored")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "k2-horizon-3.7b")

# --- Agent identity --------------------------------------------------------
AGENT_NAME = "Hermes"

SYSTEM_PROMPT = (
    "You are Hermes, a friendly, sharp personal assistant chatting with "
    "your user in a terminal. Keep replies concise and conversational unless "
    "asked for depth. You have a persistent memory of past conversations: use "
    "what you remember about the user naturally, and don't pretend to forget "
    "things you were told earlier. If you genuinely don't know something, say "
    "so instead of inventing it."
)

# --- Persistence -----------------------------------------------------------
# The agent's full state (conversation context + rolling summary) is saved
# here so memory survives across restarts.
STATE_FILE = os.path.join(os.path.dirname(__file__), "memory", "agent_state.json")

# The mock is slow (~7-13s/call because it's really ChatGPT), so be patient.
REQUEST_TIMEOUT = float(os.environ.get("MOCK_TIMEOUT", "180"))


# ===========================================================================
#  LLM PROFILES (hybrid brain) -- used by task_agent.build_model(profile)
# ===========================================================================
# The default "mock" profile drives ChatGPT via the openai-mocker (free, slow).
# For long browser tasks you can switch to a real provider for speed/reliability
# by setting OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY and choosing the
# profile (env LLM_PROFILE, or per-request `llm` in the web chat).
LLM_PROFILE = os.environ.get("LLM_PROFILE", "mock")

LLM_PROFILES = {
    "mock": {
        "provider": "openai",
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "model": MODEL,
    },
    "local": {
        # K2-Horizon-3.7B running on this machine. No API key needed (the shim
        # ignores it), but the OpenAI SDK still demands a non-empty string.
        "provider": "openai",
        "base_url": LOCAL_BASE_URL,
        "api_key": LOCAL_API_KEY,
        "model": LOCAL_MODEL,
    },
    "openai": {
        "provider": "openai",
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
        # Reasoning models reject function tools on /v1/chat/completions
        # ("Function tools with reasoning_effort are not supported ... use
        # /v1/responses or set reasoning_effort to 'none'") because reasoning
        # is on by DEFAULT server-side -- we never send the parameter. Set
        # OPENAI_RESPONSES=1 to call /v1/responses instead, which supports
        # both. gpt-4o and friends are unaffected, so this stays off.
        "responses": os.environ.get(
            "OPENAI_RESPONSES", "").strip().lower() in ("1", "true", "yes"),
    },
    "anthropic": {
        "provider": "anthropic",
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    },
    "gemini": {
        "provider": "gemini",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
    },
}


_WARNED_NO_KEY: set = set()


def resolve_profile(profile: str | None = None) -> str:
    """Resolve a profile name to the one ACTUALLY used, falling back to the
    mock when the chosen profile has no API key. Returns the effective name.

    The key check alone is enough to tell the profiles apart: 'mock' and
    'local' both carry provider 'openai' but ship dummy keys, while the real
    provider profiles read theirs from the environment and are empty when
    unset. Before, only NON-openai providers were checked, so LLM_PROFILE=openai
    with no OPENAI_API_KEY sailed through and hit api.openai.com with an empty
    credential -- a 401 from the SDK rather than anything explaining itself.
    """
    name = profile or LLM_PROFILE
    prof = LLM_PROFILES.get(name) or LLM_PROFILES["mock"]
    if not prof.get("api_key"):
        if name not in _WARNED_NO_KEY:
            _WARNED_NO_KEY.add(name)
            print(f"[config] LLM profile '{name}' has no API key set - falling "
                  f"back to the openai-mocker. Set the provider's key (see "
                  f"agent/.env.example) to use it.", flush=True)
        return "mock"
    return name


def is_mock_backend(profile: str | None = None) -> bool:
    """True when the effective backend is the session-pinned openai-mocker
    (ChatGPT-web). The mock keeps the prompt small by sending only the DELTA each
    turn, recognising a continuation via prefix-matching the message history —
    so a middleware that REWRITES already-delivered messages (the browser
    snapshot pruner) breaks that match and makes the mock open a fresh chat every
    step. Callers use this to DISABLE such middleware on the mock (it's needed
    only for real, stateless providers that re-send the full context each call)."""
    return resolve_profile(profile) == "mock"


# ===========================================================================
#  BROWSER (agent-browser via the middleware)
# ===========================================================================
# We drive the user's installed Chrome (no Chrome-for-Testing download needed).
# These are pushed into the environment by browser_tools.py at import, so the
# agent-browser subprocess the middleware spawns inherits them.
BROWSER_SESSION = os.environ.get("AGENT_BROWSER_SESSION", "hermes")
BROWSER_HEADED = os.environ.get("BROWSER_HEADED", "1") not in ("0", "", "false")
BROWSER_CHROME_PATH = os.environ.get(
    "BROWSER_CHROME_PATH",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
)
# Persistent profile dir so logins/cookies survive across runs.
BROWSER_PROFILE_DIR = os.environ.get(
    "BROWSER_PROFILE_DIR",
    os.path.join(os.path.dirname(__file__), ".browser-profile"),
)
# Truncate page/snapshot output so big accessibility trees don't blow up the
# LLM context (ChatGPT-web rejects very large prompts).
BROWSER_MAX_OUTPUT = int(os.environ.get("BROWSER_MAX_OUTPUT", "6000"))

# A2: render a page as an element table (one line per control) rather than the
# raw accessibility tree. ~2.3 KB where the tree was 7.5 KB, same refs.
#
# This one flag also decides whether BrowserSnapshotPruner is installed, and
# the two belong together. The pruner exists because a 7.5 KB snapshot per step
# is unaffordable; it pays for that by REWRITING an already-sent message, which
# breaks the provider's cached prefix from that point on. Measured over the 96
# browser calls of the 13 Oct run: 48% cache hit on a call that followed a
# snapshot tool, 85% on one that did not. With the element table the saving is
# no longer worth the cache, so the pruner comes off. Set
# BROWSER_ELEMENT_TABLE=0 and both revert together.
BROWSER_ELEMENT_TABLE = os.environ.get(
    "BROWSER_ELEMENT_TABLE", "1").strip().lower() not in ("0", "false", "no")
# Optional comma-separated navigation allowlist (empty = allow all).
BROWSER_ALLOWED_DOMAINS = os.environ.get("BROWSER_ALLOWED_DOMAINS", "")
# Where per-run browser action logs go (raw material for the script generator).
BROWSER_LOG_DIR = os.path.join(os.path.dirname(__file__), "data", "browser_runs")
# Where compiled Playwright scripts (generated from a run log) are written.
BROWSER_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "data", "browser_scripts")

# --- The agent's own working area (data/) ----------------------------------
# Everything the agent reads/writes by default lives under here: uploads/ (read
# drop-folder), reports/ (generated deliverables), knowledge/ (memory ingest),
# browser_runs/, etc. This is also the "project" root of the scoped filesystem
# (fs_tools.py) -- the one place fs_* may write/delete WITHOUT a confirm gate.
DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

# --- Files the agent can read (drop-folder) --------------------------------
# Users drop Excel / PDF / Word / CSV / PPT files here; the scripter specialist
# reads and queries them via file_tools.py (spreadsheets through pandas, docs
# converted to markdown via markitdown). Converted markdown is cached under
# <uploads>/.cache/ so repeat searches don't re-convert the same file.
UPLOADS_DIR = os.environ.get(
    "UPLOADS_DIR", os.path.join(DATA_DIR, "uploads"))
# Cap on a single file-tool result so a huge sheet/doc can't overflow the
# ChatGPT-web prompt (mirrors BROWSER_MAX_OUTPUT).
FILE_MAX_OUTPUT = int(os.environ.get("FILE_MAX_OUTPUT", "8000"))

# --- Self-healing memory: background ingest + consolidation ----------------
# A "drop folder" for reference material the agent should fold into its memory
# wiki (separate from uploads/, which is for files it ANALYSES on demand). The
# background ingest loop watches both and turns new/changed files into wiki pages.
KNOWLEDGE_DIR = os.environ.get(
    "KNOWLEDGE_DIR", os.path.join(os.path.dirname(__file__), "data", "knowledge"))
# How often the background ingest scan runs (minutes). The nightly consolidation
# (self-healing merge/prune) runs once per day at MEMORY_CONSOLIDATE_HOUR (local).
MEMORY_INGEST_MINUTES = int(os.environ.get("MEMORY_INGEST_MINUTES", "45"))
MEMORY_CONSOLIDATE_HOUR = int(os.environ.get("MEMORY_CONSOLIDATE_HOUR", "3"))
# Master switch for ALL background memory work (ingest + consolidation). The hot
# path (digest/relevant injection) is always on; this only gates the loops.
MEMORY_BG_ENABLED = os.environ.get("MEMORY_BG_ENABLED", "1") not in ("0", "", "false")

# --- Reports the agent can GENERATE (write area, sandboxed) ----------------
# The scripter writes Markdown and deterministic renderers (pandoc, openpyxl,
# matplotlib, docx2pdf) turn it into real DOCX / PDF / XLSX / PPTX / HTML / PNG
# deliverables, dropped here. This is the ONE write-capable file area -- the
# mirror image of UPLOADS_DIR (the read area) -- and every report tool is
# path-sandboxed to it (flat folder, no traversal). Files are served for
# download at /api/reports/<name>.
REPORTS_DIR = os.environ.get(
    "REPORTS_DIR", os.path.join(os.path.dirname(__file__), "data", "reports"))
# Markdown -> DOCX/PPTX/HTML goes through pandoc (a CLI on PATH). Override the
# binary path here if it isn't on PATH. PDF is produced DOCX->PDF via docx2pdf
# (Microsoft Word COM on Windows); if Word is absent, PDF degrades to a clear
# error and the other formats still work.
PANDOC_BIN = os.environ.get("PANDOC_BIN", "pandoc")

# ===========================================================================
#  SCOPED FILESYSTEM (Roadmap #4) -- broader file access beyond data/
# ===========================================================================
# fs_tools.py lets the agent work with files BEYOND its own data/ folder --
# Downloads, Documents -- but only inside an ALLOWLIST of roots, each with
# explicit per-root permissions. The model:
#   * read   -- list + read files under the root,
#   * write  -- create / overwrite / append / move-into / download-into,
#   * delete -- remove files,
#   * confirm-- writes/deletes/moves/downloads need an explicit confirm=True
#               (the human's "yes"); this is the hard gate for anything OUTSIDE
#               the project workspace.
# The "project" root (DATA_DIR) is the agent's own sandbox -- full access, no
# confirm. Downloads/Documents are read+write but confirm-gated and never
# delete by default. Tighten or widen via env (FS_ALLOW_DOWNLOADS=0, etc.);
# the whole feature is off if FS_ENABLED=0.
FS_ENABLED = os.environ.get("FS_ENABLED", "1") not in ("0", "", "false")
# Per-tool output budget (mirrors FILE_MAX_OUTPUT) and a hard cap on how many
# bytes fs_read will pull / fs_download will fetch, so a huge file can't blow
# the prompt or fill a disk.
FS_MAX_OUTPUT = int(os.environ.get("FS_MAX_OUTPUT", str(FILE_MAX_OUTPUT)))
FS_DOWNLOAD_MAX_MB = float(os.environ.get("FS_DOWNLOAD_MAX_MB", "50"))


def _user_dir(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def fs_roots() -> list:
    """The resolved filesystem allowlist: a list of root specs the fs_* tools
    operate within. Each spec is a dict {name, path, read, write, delete,
    confirm}. Built fresh (cheap) so tests can monkeypatch DATA_DIR/env. A root
    is included only if its directory exists (project is always created).

    Override the standard roots with env flags:
      FS_ALLOW_DOWNLOADS=0 / FS_ALLOW_DOCUMENTS=0  -- drop that root,
      FS_EXTRA_ROOTS="name=C:\\path[:rw];name2=..."  -- add custom roots
        (suffix :ro read-only, :rw read+write confirm-gated, :rwd adds delete).
    """
    roots: list = []
    # The project workspace -- full access, NO confirm (the agent's own sandbox).
    project = os.path.realpath(DATA_DIR)
    os.makedirs(project, exist_ok=True)
    roots.append({"name": "project", "path": project,
                  "read": True, "write": True, "delete": True, "confirm": False})

    # Standard user folders -- read+write but confirm-gated and no delete.
    standard = [
        ("downloads", _user_dir("Downloads"), "FS_ALLOW_DOWNLOADS"),
        ("documents", _user_dir("Documents"), "FS_ALLOW_DOCUMENTS"),
    ]
    for name, path, flag in standard:
        if os.environ.get(flag, "1") in ("0", "", "false"):
            continue
        if os.path.isdir(path):
            roots.append({"name": name, "path": os.path.realpath(path),
                          "read": True, "write": True, "delete": False,
                          "confirm": True})

    # Custom extra roots from env, e.g. "shared=D:\\team:rw;archive=E:\\old:ro".
    for spec in (os.environ.get("FS_EXTRA_ROOTS", "") or "").split(";"):
        spec = spec.strip()
        if not spec or "=" not in spec:
            continue
        name, _, rest = spec.partition("=")
        path, _, mode = rest.rpartition(":")
        if mode not in ("ro", "rw", "rwd"):  # no mode suffix -> rest was the path
            path, mode = rest, "rw"
        name, path = name.strip(), os.path.expanduser(path.strip())
        if not name or not os.path.isdir(path):
            continue
        roots.append({
            "name": name, "path": os.path.realpath(path),
            "read": True, "write": mode in ("rw", "rwd"),
            "delete": mode == "rwd", "confirm": True,
        })
    return roots


# ===========================================================================
#  DELIVERIES / OUTBOX (Roadmap #5) -- files the agent SENDS to the phone
# ===========================================================================
# When the agent "sends me a file" (a generated report, a doc it found in
# Downloads, a chart, an audio clip), it copies the file into this outbox and
# records a delivery; the mobile app polls /api/deliveries and renders each as
# an inline image / audio / video / download chip. This is the WRITE-OUT mirror
# of the read-only file explorer -- the one folder served to the phone.
OUTBOX_DIR = os.environ.get(
    "OUTBOX_DIR", os.path.join(os.path.dirname(__file__), "data", "outbox"))
# Cap on a single delivered file (MB) so the agent can't push a giant blob to a
# phone on mobile data.
DELIVER_MAX_MB = float(os.environ.get("DELIVER_MAX_MB", "75"))

# ===========================================================================
#  MOBILE / REMOTE ACCESS AUTH (Roadmap #5 -- security gate for exposure)
# ===========================================================================
# The API has no login. For LOCAL use that's fine. The moment you expose the
# server beyond localhost (Tailscale / LAN / a tunnel) so your phone can reach
# it, set MOBILE_TOKEN to a long random secret: requests from any NON-loopback
# client must then present it (Authorization: Bearer <token>, or ?token=<token>
# for media/download URLs). Loopback (127.0.0.1/::1 -- the desktop UI on the PC)
# is always allowed WITHOUT a token, so setting one never breaks local use.
# Empty (default) = no auth = localhost-only is your safety boundary.
MOBILE_TOKEN = os.environ.get("MOBILE_TOKEN", "").strip()


def mobile_auth_enabled() -> bool:
    """True when a token is configured (so remote clients must authenticate)."""
    return bool(MOBILE_TOKEN)


# --- Email (Gmail SMTP via an App Password) --------------------------------
# Set these in agent/.env (see .env.example). The app password is generated at
# Google Account -> Security -> App passwords (needs 2-Step Verification on).
# Sending uses smtplib directly -- no browser, no credentials through the LLM.
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))

# Who the agent signs outgoing mail as. Without this the model has no idea whose
# assistant it is, so it either leaves a literal "[Your Name]" in the body or
# invents one -- both observed. Falls back to the local part of the Gmail
# address, which is at least not a placeholder; set USER_NAME in .env to fix it
# properly.
USER_NAME = os.environ.get("USER_NAME", "").strip()
if not USER_NAME and GMAIL_ADDRESS:
    USER_NAME = GMAIL_ADDRESS.split("@", 1)[0].replace(".", " ").title()

# IANA timezone name (e.g. "Asia/Kolkata") for Google Calendar events. Optional:
# when unset, calendar_tools sends each time with this machine's real UTC offset
# instead, which is correct for one-off meetings. Set it if you want recurring
# events to survive a DST change properly.
TIMEZONE = os.environ.get("TIMEZONE", "").strip()


# ===========================================================================
#  VOICE -- local STT (whisper.cpp) + TTS (piper)
# ===========================================================================
# Binaries + models live under agent/voice/ (fetched by setup_voice.py). The
# whole feature is OPTIONAL: if the files aren't present, VOICE_ENABLED is False
# and the server/UI simply hide the mic. Everything is path-overridable via env
# so you can point at a custom build or model.
_VOICE_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "voice")
VOICE_DIR = os.environ.get("VOICE_DIR", _VOICE_DIR_DEFAULT)

# Which whisper build to run: 'cpu' (whisper-blas) or 'cuda' (whisper-cublas).
# Flip this env var to switch -- both builds can be installed side by side.
WHISPER_ACCEL = os.environ.get("WHISPER_ACCEL", "cpu").strip().lower()

# Wake word for the hands-free voice companion ("Paimon"). The browser listens
# continuously and only acts on an utterance that starts with this word; the
# matching itself (fuzzy, to tolerate whisper mishears) lives in the frontend.
WAKE_WORD = os.environ.get("WAKE_WORD", "paimon").strip().lower()


def _find_exe(root: str, names: tuple) -> str:
    """First matching exe under ``root`` (binaries ship inside a versioned subdir
    with their DLL siblings, so we search rather than hard-code the path)."""
    if not os.path.isdir(root):
        return ""
    for base, _dirs, files in os.walk(root):
        for n in names:
            if n in files:
                return os.path.join(base, n)
    return ""


# whisper-cli.exe (v1.8.6) -- fall back to the older main.exe name just in case.
WHISPER_BIN = os.environ.get("WHISPER_BIN") or _find_exe(
    os.path.join(VOICE_DIR, "bin", f"whisper-{WHISPER_ACCEL}"),
    ("whisper-cli.exe", "main.exe", "whisper-cli", "main"))
WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL", os.path.join(VOICE_DIR, "models", "ggml-base.en.bin"))

PIPER_BIN = os.environ.get("PIPER_BIN") or _find_exe(
    os.path.join(VOICE_DIR, "bin"), ("piper.exe", "piper"))
PIPER_VOICE = os.environ.get(
    "PIPER_VOICE", os.path.join(VOICE_DIR, "voices", "en_US-amy-medium.onnx"))

# Seconds before we give up on a whisper/piper subprocess (utterances are short).
VOICE_TIMEOUT = float(os.environ.get("VOICE_TIMEOUT", "60"))


def voice_enabled() -> bool:
    """True only when every piece needed for a full STT+TTS round-trip exists."""
    return all(p and os.path.isfile(p) for p in
               (WHISPER_BIN, WHISPER_MODEL, PIPER_BIN, PIPER_VOICE))
