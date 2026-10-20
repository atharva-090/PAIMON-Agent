"""One-time browser login helper — sign into a site by hand, no creds via the LLM.

IMPORTANT — why this launches PLAIN Chrome (not agent-browser):
Google (and some other sites) BLOCK the login flow in automation-controlled
browsers ("Couldn't sign you in — this browser or app may not be secure"),
because Playwright/agent-browser add automation flags. But an already-logged-in
session is fine. So we log in here using your NORMAL Chrome, pointed at the SAME
persistent profile folder the agent uses. After you sign in, the cookies live in
that profile, and the agent's automated browser opens the site ALREADY signed in.
Your password never touches the LLM (no transcript/log leak, no safety refusal).

Run with the agent venv's Python (for config), e.g. on Windows:
    .venv\\Scripts\\python.exe setup_browser_login.py                 # opens Gmail
    .venv\\Scripts\\python.exe setup_browser_login.py https://...      # any site

Then: log in in the Chrome window that opens, finish 2FA, and CLOSE it.
Close it BEFORE running the agent or saved Playwright tasks (one Chrome process
per profile folder at a time).
"""

import os
import subprocess
import sys

import config

CHROME = config.BROWSER_CHROME_PATH
PROFILE = config.BROWSER_PROFILE_DIR


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "https://mail.google.com"

    if not (CHROME and os.path.exists(CHROME)):
        sys.exit(
            f"Chrome not found at {CHROME!r}. Set BROWSER_CHROME_PATH, or run "
            f"Chrome yourself with:\n"
            f'  chrome.exe --user-data-dir="{PROFILE}" {url}'
        )

    os.makedirs(PROFILE, exist_ok=True)
    # Plain Chrome on the agent's profile dir — NO automation flags, so Google
    # treats it as a normal browser and lets you log in.
    cmd = [CHROME, f"--user-data-dir={PROFILE}", "--no-first-run",
           "--no-default-browser-check", url]
    print(f"Opening a normal Chrome window at {url}")
    print(f"  profile (shared with the agent): {PROFILE}")
    print("  → Log in MANUALLY, finish any 2FA, then CLOSE the window.")
    print("  → Keep it closed while the agent / saved tasks run.\n")
    subprocess.Popen(cmd, close_fds=True)
    print("Launched. After you sign in and close it, the agent will be logged in "
          "for that site.")


if __name__ == "__main__":
    main()
