@echo off
rem Stops a server started by start_website_background.bat.
rem Only targets pythonw.exe (the detached launcher's process); a foreground
rem `python serve.py` is left alone - stop that one with Ctrl+C in its own window.
taskkill /f /im pythonw.exe >nul 2>&1 && (
  echo Stopped the background server.
) || (
  echo No background server was running.
)
