@echo off
REM Start Paimon so your PHONE can reach it (Roadmap #5).
REM Binds to 0.0.0.0 and prints a QR / phone URL. Prerequisite: the
REM openai-mocker must already be running on port 8000 (..\openai-mock: npm start).
REM
REM Before exposing beyond a trusted LAN, set an access token first, e.g.:
REM     set MOBILE_TOKEN=change-me-to-a-long-random-secret
cd /d "%~dp0"
echo Starting Paimon (phone mode) — keep this window open. Ctrl+C to stop.
".venv\Scripts\python.exe" mobile_serve.py
