@echo off
rem One-click launcher: starts the local server and opens the site.
rem
rem The server runs in a RESTART LOOP. serve.py should not exit on its own - client
rem disconnects are handled internally - so if the python process ever does die
rem (unhandled error, port hiccup), this brings it straight back instead of leaving
rem the site silently unreachable until someone notices.
rem
rem Closing this window still stops the server. For a server that survives the window
rem closing, use start_website_background.bat instead.
cd /d "%~dp0"
start "" "http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website/index.html"
:loop
python serve.py
echo.
echo [server exited - restarting in 2s.  Press Ctrl+C twice to stop for good]
timeout /t 2 /nobreak >nul
goto loop
