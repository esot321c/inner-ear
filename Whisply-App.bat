@echo off
REM Double-click me to launch the Whisply browser GUI (drag-and-drop).
set "PATH=%~dp0bin;%PATH%"
set "HF_HUB_DISABLE_SYMLINKS=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONPATH=%~dp0pylib"
"%~dp0.venv\Scripts\whisply.exe" app
