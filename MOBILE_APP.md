# Paimon on your phone — the mobile app (Roadmap #5)

A minimalist, ChatGPT-style phone client for the Hermes/Paimon agent: a clean
chat screen, a big **talk** button (voice in + spoken replies), and a **file
delivery** channel so the agent can pull any document / image / audio / video
and **send it to your phone**, where it renders inline or as a download.

It is a **PWA** (installable web app) served by the existing FastAPI backend at
**`/m`**. The STT/TTS/swarm all stay on the PC; the phone is a thin client.

```
 Phone (PWA / APK)  ──HTTP──►  FastAPI  ──►  swarm / voice / tools
   chat + 🎤 + files            (server.py, :8800)        │
        ▲                                                 ▼
        └──────────  /api/deliveries + /api/outbox  ◄── agent send_file()
```

---

## 1) Run the server so your phone can reach it

```bat
cd agent
:: optional but recommended once you go beyond your own PC (see Security):
set MOBILE_TOKEN=some-long-random-secret
start_mobile.bat
```

`start_mobile.bat` runs `mobile_serve.py`, which binds to `0.0.0.0` and prints
the **phone URL** plus a **QR code** to scan. (Prerequisite, as always: the
`openai-mocker` is running on port 8000.)

You'll see something like:

```
  Open this on your phone:
      http://192.168.1.20:8800/m
  [QR code]
```

---

## 2) Install it on the phone (3 routes — pick one)

### Route A — Install the PWA  ✅ *works right now, recommended*
1. On the phone, open the printed URL (`http://<pc-ip>:8800/m`) in **Chrome**.
2. Chrome menu **⋮ → Install app** (or **Add to Home screen**).
3. Done — Paimon gets its own icon, opens **full-screen with no browser bar**,
   and behaves like a native app. This is exactly how installable apps work on
   Android (the same mechanism the ChatGPT web app uses).

This needs nothing but the phone and the PC on the same network (or Tailscale).

### Route B — A real signed `.apk` with **PWABuilder** (no Android tooling)
Best when you want an actual installable `.apk` file and don't want to install
Android Studio.
1. Expose the PWA over **HTTPS** (PWABuilder/TWA requires it). Easiest:
   `cloudflared tunnel --url http://localhost:8800` → gives a public
   `https://….trycloudflare.com` URL. **Set `MOBILE_TOKEN` first** — this URL is
   public.
2. Go to **https://www.pwabuilder.com**, paste `https://….trycloudflare.com/m`.
3. Package for **Android** → download the generated **signed APK**.
4. Copy the `.apk` to the phone and install (enable "install unknown apps").

### Route C — A real `.apk` with **Capacitor** (needs Android Studio / JDK)
A thin native WebView shell that points at your PC. Scaffold is in
[`mobile/apk/`](mobile/apk/).
```bash
cd agent/mobile/apk
#   edit capacitor.config.json -> server.url = http://<your-pc-TAILSCALE-ip>:8800/m
#   (use the Tailscale 100.x IP so the APK works from any network, not just home WiFi)
npm install
npx cap add android
npx cap sync android
npx cap open android      # opens Android Studio -> Build > Build APK(s)
# (or headless:)  cd android && gradlew.bat assembleDebug
#   -> android/app/build/outputs/apk/debug/app-debug.apk
```
> Requires the **Android SDK + JDK** (Android Studio installs both). This machine
> doesn't have them, so the APK is built on a machine that does — the scaffold,
> config, and scripts are all here and ready.

> **Why Route A is the default here:** building a binary `.apk` needs the Android
> SDK/JDK, which isn't installed on this PC. The PWA install (Route A) gives you
> the installed-app experience today with zero toolchain; Routes B and C are the
> paths to an actual `.apk` file when you want one.

---

## 3) Reach it from a DIFFERENT network (cellular / other WiFi / anywhere)

**Use Tailscale. It's free, works from any network, and is the most secure
option** — your PC is never exposed to the public internet; only your own
signed-in devices can reach it over an encrypted private mesh. No port-
forwarding, no router config, no monthly cost.

| Option | Cost | Works on any network? | Exposure | Use when |
|--------|------|----------------------|----------|----------|
| **Tailscale** ⭐ | **Free** | ✅ yes | Private (your devices only) | **Default — the PWA from anywhere** |
| Cloudflare Tunnel | Free | ✅ yes | Public HTTPS URL (set a token!) | You need a public **HTTPS** URL (e.g. the PWABuilder APK, Route B) |
| ngrok | Free tier (random URL) | ✅ yes | Public HTTPS URL | Quick one-off public test |
| Same-WiFi only | Free | ❌ home only | LAN | You're always on home WiFi |

### Tailscale setup (≈5 min, one time)
1. Install **Tailscale** on the **PC** and the **phone**; sign into both with the
   **same account** (free "Personal" plan).
2. On the PC, run `start_mobile.bat`. It auto-detects the Tailscale IP and prints
   the **"works anywhere"** URL + QR (e.g. `http://100.101.102.103:8800/m`).
3. (Recommended) `set MOBILE_TOKEN=<random-secret>` before launching.
4. On the phone (Tailscale toggled **on**), open that URL, enter the token once
   in **Settings**, and **Install**. It now works on cellular, at the office,
   anywhere — for ₹0/month.

### Cloudflare Tunnel (free) — when you specifically need a public HTTPS URL
```bash
cloudflared tunnel --url http://localhost:8800     # prints https://….trycloudflare.com
```
**Set `MOBILE_TOKEN` first** — this URL is reachable by anyone who has it. This is
the HTTPS URL you'd feed to PWABuilder in Route B.

---

## 4) Security (read before exposing it)
The API has **no login by default** — fine for localhost. The moment the phone
can reach it, set a token:

```bat
set MOBILE_TOKEN=a-long-random-secret-string
```

- Every `/api/*` call from a **non-loopback** client must then send the token
  (the app stores it and adds `Authorization: Bearer …`, or `?token=…` for media
  URLs). Enter it once in the app's **Settings**.
- **Loopback (the desktop UI on the PC) stays tokenless**, so turning auth on
  never breaks local use.
- If you front the server with a **localhost tunnel** (cloudflared/ngrok), the
  client looks like loopback — set `MOBILE_AUTH_ALLOW_LOOPBACK=0` to force the
  token even then.

---

## 5) The "send me a file" feature
Ask Paimon things like *"send me the Q2 report you just made"*, *"send me the
invoice in my Downloads"*, *"make a chart of regional sales and send it"*. The
agent calls **`send_file(...)`**, which copies the file into `data/outbox/` and
records a delivery; the phone polls `/api/deliveries` and shows it:

- **images** → inline preview, **audio** → a player, **video** → a player,
- **pdf / docx / xlsx / pptx / anything else** → a tappable **download chip**.

It can only send files it's already allowed to read (the same allowlist as the
file tools: generated reports, uploads, Downloads, Documents) — no new access.

---

## Config knobs (env)
| var | default | meaning |
|-----|---------|---------|
| `MOBILE_HOST` | `127.0.0.1` (server.py) / `0.0.0.0` (mobile_serve.py) | bind address |
| `PORT` | `8800` | server port |
| `MOBILE_TOKEN` | *(empty = no auth)* | required for non-loopback `/api/*` |
| `MOBILE_AUTH_ALLOW_LOOPBACK` | `1` | exempt loopback from the token |
| `OUTBOX_DIR` | `data/outbox` | where delivered files are staged |
| `DELIVER_MAX_MB` | `75` | per-file delivery cap |

## Files
- `mobile/` — the PWA: `index.html`, `app.js`, `style.css`,
  `manifest.webmanifest`, `sw.js`, `icons/`, `make_icons.py`.
- `mobile/apk/` — the Capacitor scaffold (Route C).
- `deliver_tools.py` — `send_file` / `list_deliveries` + the outbox API.
- `mobile_serve.py` / `start_mobile.bat` — the phone-mode launcher.
- `server.py` — `/m` mount, `/api/deliveries`, `/api/outbox/<f>`, the auth middleware.
