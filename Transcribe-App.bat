@echo off
REM Double-click: starts the Meeting Transcriber web app and opens your browser.
REM Drop meeting files in in\ (or upload in the app). Output -> out\, source -> archive\.
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%in" mkdir "%ROOT%in"
if not exist "%ROOT%out" mkdir "%ROOT%out"
if not exist "%ROOT%archive" mkdir "%ROOT%archive"

docker image inspect nemo-sortformer:latest >nul 2>&1
if errorlevel 1 (
  echo First run: building the image once ^(heavy, ~20 min^)...
  docker build -t nemo-sortformer "%ROOT%docker\nemo"
)

REM Optional config from .env (HF_TOKEN, WHISPER_MODEL, ...). App needs none.
set "ENVOPT="
if exist "%ROOT%.env" set ENVOPT=--env-file "%ROOT%.env"

start "" http://localhost:7860
echo.
echo Meeting Transcriber starting at http://localhost:7860
echo (Give it a few seconds. CLOSE THIS WINDOW to stop the app.)
echo.
docker run --rm --gpus all %ENVOPT% -p 7860:7860 ^
  -v "%ROOT%in:/in" ^
  -v "%ROOT%out:/out" ^
  -v "%ROOT%archive:/archive" ^
  -v "%ROOT%docker\nemo\app.py:/work/app.py" ^
  -v "%ROOT%docker\nemo\pipeline.py:/work/pipeline.py" ^
  -v "nemo-cache:/root/.cache" ^
  --entrypoint python3 nemo-sortformer /work/app.py
