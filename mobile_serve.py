"""Launch the Hermes/Paimon server so your PHONE can reach it (Roadmap #5).

This is the normal server (server.py) bound to 0.0.0.0 instead of localhost,
plus a friendly banner: it prints the exact URL to open on your phone and a
QR code you can scan, and reminds you to set an access token before exposing
the API beyond your own machine.

Usage:
    agent/.venv/Scripts/python.exe mobile_serve.py
    agent/.venv/Scripts/python.exe mobile_serve.py --port 8800 --host 0.0.0.0

Make it reachable from anywhere (not just the same WiFi) with Tailscale:
    1) install Tailscale on the PC AND the phone, sign into both,
    2) set a token:  set MOBILE_TOKEN=<a-long-random-secret>
    3) run this; open  http://<your-PC-tailscale-ip>:8800/m  on the phone,
       enter the token once in Settings, and Install the app.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess


def _tailscale_ip() -> str:
    """The PC's Tailscale IPv4 (100.x.y.z) if Tailscale is installed + up, else
    "". This is the address that works from ANY network (cellular, other WiFi)
    with no port-forwarding — the recommended "Paimon anywhere" path."""
    exe = shutil.which("tailscale") or r"C:\Program Files\Tailscale\tailscale.exe"
    if not os.path.exists(exe) and not shutil.which("tailscale"):
        return ""
    try:
        out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True,
                             timeout=5)
        for line in (out.stdout or "").splitlines():
            ip = line.strip()
            if ip.startswith("100."):
                return ip
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def _lan_ips() -> list[str]:
    """Best-effort list of this machine's LAN/VPN IPv4 addresses, primary first."""
    ips: list[str] = []
    # The address used to reach the internet = the primary LAN interface.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    # Everything else the host resolves to (Tailscale 100.x, other NICs).
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass
    return ips or ["127.0.0.1"]


def _print_qr(url: str) -> None:
    try:
        import qrcode  # optional
    except ImportError:
        print("   (install 'qrcode' for a scannable code:  "
              ".venv\\Scripts\\python.exe -m pip install qrcode)")
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve Paimon for your phone.")
    ap.add_argument("--host", default=os.environ.get("MOBILE_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8800")))
    args = ap.parse_args()

    os.environ["MOBILE_HOST"] = args.host
    os.environ["PORT"] = str(args.port)

    ts_ip = _tailscale_ip()
    lan_ips = _lan_ips()
    # The "anywhere" URL (Tailscale) is the one to scan when present; otherwise
    # fall back to the LAN address (same-WiFi only).
    primary = ts_ip or lan_ips[0]
    phone_url = f"http://{primary}:{args.port}/m"

    print("=" * 64)
    print("  📱  Paimon — phone access")
    print("=" * 64)
    if ts_ip:
        print("  ✅ Tailscale detected — this URL works from ANY network")
        print("     (cellular, other WiFi), as long as the phone has Tailscale on:\n")
        print(f"      http://{ts_ip}:{args.port}/m   ← scan / open this\n")
        print("  Same-WiFi-only fallback addresses:")
        for ip in lan_ips:
            print(f"      http://{ip}:{args.port}/m")
        print()
    else:
        print("  Open this on your phone (must be on the SAME WiFi), then use the")
        print("  browser menu → 'Install app' / 'Add to Home screen':\n")
        print(f"      {phone_url}\n")
        for ip in lan_ips[1:]:
            print(f"      (also) http://{ip}:{args.port}/m")
        print("  ▶ To use it from a DIFFERENT network (cellular, anywhere), install")
        print("    Tailscale (free) on this PC and the phone, then rerun this — the")
        print("    'works anywhere' URL appears here automatically. See MOBILE_APP.md.")
        print()
    _print_qr(phone_url)
    print()
    if os.environ.get("MOBILE_TOKEN", "").strip():
        print("  🔒 Auth ON — enter the token once in the app's Settings.")
    else:
        print("  ⚠️  No MOBILE_TOKEN set: the API is OPEN to anyone who can reach")
        print("      this machine. On Tailscale only your own devices can; still,")
        print("      before any public tunnel set:  set MOBILE_TOKEN=<random-secret>")
    print("  Prerequisite: the openai-mocker must be running on port 8000.")
    print("=" * 64)

    import uvicorn
    uvicorn.run("server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
