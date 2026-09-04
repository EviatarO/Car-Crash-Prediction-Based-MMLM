@echo off
rem Starts the server DETACHED, with no console window attached to it.
rem
rem Why this exists: the plain launcher ties the server's lifetime to its console, so
rem closing that window - or the session that opened it - takes the site down, which
rem looks like a crash but is not one. pythonw.exe has no console to close, so the
rem server keeps serving until it is stopped deliberately.
rem
rem Stop it with stop_website.bat.
cd /d "%~dp0"
tasklist /fi "imagename eq pythonw.exe" | find /i "pythonw.exe" >nul && (
  echo A pythonw.exe is already running - if the site is up, nothing to do.
  echo Use stop_website.bat first if you want a clean restart.
)
start "" /b pythonw serve.py
echo Server started detached on http://localhost:8765
echo Opening the site...
timeout /t 2 /nobreak >nul
start "" "http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website/index.html"
