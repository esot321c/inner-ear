============================================================
  TRANSCRIPTION IN DOCKER  (GPU, speaker labels)
============================================================

Why Docker: it runs the whole thing on a clean Linux image, so none of the
Windows-specific breakage applies. It's the recommended way to run this.

Requirements (already confirmed working on this machine):
  - Docker Desktop running, with GPU support (NVIDIA runtime).
  - The RTX 3070 is visible to containers.

------------------------------------------------------------
EVERYDAY USE
------------------------------------------------------------
1. Drop audio/video into the   in   folder (one level up: Transcribe\in).
2. Double-click            Transcribe-Docker.bat
3. Results appear in       Transcribe\out

First run builds the image and downloads models (~5-6 GB image + ~2-3 GB
models). After that it's cached and fast. Models persist in a Docker volume
("hf-cache"), so they are downloaded only once.

Speaker labels reuse the same   ..\hf_token.txt   file as the native setup.
No token = transcript only (no "who said what").

------------------------------------------------------------
COMMAND LINE (optional)
------------------------------------------------------------
From this folder:

  # build (only needed once, or after Dockerfile changes)
  docker compose build

  # run a batch over ..\in -> ..\out
  $env:HF_TOKEN = (Get-Content ..\hf_token.txt -Raw).Trim()
  docker compose run --rm transcribe

  # use the full (more accurate, heavier) model instead of turbo
  $env:WHISPER_MODEL = "large-v3"
  docker compose run --rm transcribe

------------------------------------------------------------
NATIVE vs DOCKER
------------------------------------------------------------
- Native (..\Transcribe-Folder.bat): no Docker needed, has the GUI app.
- Docker (this folder): cleaner, more reproducible, no Windows workarounds.
Both read the same in\, out\, and hf_token.txt. Use whichever you prefer.

------------------------------------------------------------
HOUSEKEEPING
------------------------------------------------------------
  docker images                       # see the image
  docker compose down                 # stop (nothing persistent runs)
  docker volume rm docker_hf-cache    # delete cached models (frees disk)
  docker image rm morley-transcribe   # delete the image
============================================================
