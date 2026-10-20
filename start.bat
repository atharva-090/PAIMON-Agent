@echo off
REM Start the BA Task Manager web app (frontend + REST API + AI agent).
REM Prerequisite: the openai-mocker must already be running on port 8000
REM (in ..\openai-mock:  npm start).
cd /d "%~dp0"
echo Starting BA Task Manager on http://127.0.0.1:8800
echo (Keep this window open. Press Ctrl+C to stop.)
".venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8800
