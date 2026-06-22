@echo off
REM Double-click to transcribe everything in ..\in into ..\out (GPU + speakers).
REM Builds the image automatically the FIRST time only; after that it just runs.
setlocal
cd /d "%~dp0"

REM Reuse the same token file as the native setup (one line, the HF token).
set "HF_TOKEN="
if exist "..\hf_token.txt" set /p HF_TOKEN=<..\hf_token.txt
if "%HF_TOKEN%"=="" (
  echo NOTE: no token in ..\hf_token.txt - transcribing WITHOUT speaker labels.
)

REM Build only if the image doesn't exist yet (one-time, ~10 min).
docker image inspect morley-transcribe:latest >nul 2>&1
if errorlevel 1 (
  echo First run: building the image once. This takes a while...
  docker compose build
)

echo Transcribing...
docker compose run --rm transcribe
echo.
pause
