@echo off
rem One-click launcher: starts the local server and opens the site in the default browser.
rem The server window stays open while you browse - close it (or Ctrl+C) when done.
cd /d "%~dp0"
start "" "http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website/index.html"
python serve.py
