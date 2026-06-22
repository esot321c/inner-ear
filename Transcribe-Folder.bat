@echo off
REM Double-click me. Transcribes everything in the "in" folder into "out".
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Transcribe-Folder.ps1"
