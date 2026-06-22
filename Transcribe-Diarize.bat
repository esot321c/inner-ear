@echo off
REM Drop audio/video in in\, double-click this. Output (speaker-labeled, smoothed)
REM lands in out\<name>\, the source is archived to archive\<timestamp>\.
REM Engine: Whisper large-v3 + streaming Sortformer diarization, on the GPU.
setlocal
set "ROOT=%~dp0"

REM timestamp for the archive folder (local time)
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set "STAMP=%%i"

if not exist "%ROOT%in" mkdir "%ROOT%in"
if not exist "%ROOT%out" mkdir "%ROOT%out"
if not exist "%ROOT%archive" mkdir "%ROOT%archive"

REM build the image once if it's not present yet
docker image inspect nemo-sortformer:latest >nul 2>&1
if errorlevel 1 (
  echo First run: building the image once ^(heavy, ~20 min^)...
  docker build -t nemo-sortformer "%ROOT%docker\nemo"
)

echo.
echo Transcribing + diarizing everything in in\  (GPU: large-v3 + Sortformer)...
echo This is slow on purpose - large-v3 for accuracy. A 1-hour file takes a while.
echo.
docker run --rm --gpus all ^
  -e STAMP=%STAMP% ^
  -v "%ROOT%in:/in" ^
  -v "%ROOT%out:/out" ^
  -v "%ROOT%archive:/archive" ^
  -v "%ROOT%docker\nemo\diarize_transcribe.py:/work/diarize_transcribe.py" ^
  -v "nemo-cache:/root/.cache" ^
  nemo-sortformer

echo.
echo Done. Transcripts are in out\  ^|  sources archived to archive\%STAMP%\
pause
