# Meeting Transcriber

> Local, GPU-accelerated meeting transcription with **accurate speaker separation** — even on a single mixed track with similar voices talking over each other. Nothing leaves your machine.

*(Working title — rename to whatever you like.)*

Transcribe a recording, see the voices it detected, **play a clip of each and name them**, and save a clean labeled transcript. Built for messy real-world audio: crosstalk, interruptions, multiple speakers.

---

## Why this exists

Most "Whisper + diarization" setups use **pyannote**, which collapses similar-sounding voices into one speaker on hard audio. This project uses **NVIDIA NeMo streaming Sortformer** instead, which is overlap-aware and separates voices that pyannote can't — validated on a fast, crosstalk-heavy podcast clip where pyannote tagged 88% of words as a single speaker and Sortformer got the speakers right.

## What it does

- **Transcribes** with OpenAI Whisper `large-v3` (via faster-whisper) — accurate, captures fast/quiet speech.
- **Separates speakers** with NeMo streaming Sortformer — handles long files within 8 GB VRAM.
- **Sentence-level smoothing** so boundary words don't hang off the wrong speaker.
- **Two-channel meetings** (your mic + the call) are auto-detected and split: your channel is labeled as you; the far side (which may have several people) is diarized.
- **Naming UI** — plays a sample clip of each detected voice so you can name them.
- **Single** (interactive) and **batch** (whole-folder) modes.
- Fully local. No cloud, no uploads.

## Requirements

- **NVIDIA GPU with ≥8 GB VRAM** (developed on an RTX 3070).
- **Docker** with GPU support (NVIDIA Container Toolkit; on Windows, Docker Desktop + WSL2).
- ~13 GB disk for the image, plus ~3–4 GB of models downloaded on first run (cached after).
- The `.bat` launchers are Windows; on Linux/macOS run the equivalent `docker run` (see [Manual run](#manual-run)).

## Quick start (Windows)

1. Build/launch in one step — double-click **`Transcribe-App.bat`** (first run builds the image, ~20 min).
2. Your browser opens to **http://localhost:7860**.
3. Upload a meeting → **Transcribe** → play each voice, type names → **Save**.
4. Transcript lands in `out/<name>/`, the source moves to `archive/`.

## Usage

**Single meeting:** set your name + which channel is you (for two-channel files), upload, transcribe, name the voices, save.

**Batch:** drop files in `in/`, open the **Batch folder** tab, **Process all**, then pick each meeting to name and save.

**Best capture (recommended):** OBS multi-track — **Mic → Track 1**, **Desktop audio → Track 2** (Settings → Output → Recording). Recorded as separate tracks in one MKV; the pipeline pulls each independently. A stereo file (you = left, call = right) also works. A single mixed track is diarized as-is.

## How it works

```
audio file
  ├─ channel detection (multi-track / stereo / mono)         pipeline.plan_channels
  ├─ transcribe each channel        (Whisper large-v3)        pipeline.transcribe
  ├─ diarize the mixed/far channel  (NeMo Sortformer)         pipeline.diarize
  ├─ assign speaker per word + sentence-level smoothing        pipeline.smooth_sentences
  ├─ merge channels chronologically, group into turns          pipeline.build_turns
  └─ you name the speakers → render labeled transcript          pipeline.render
```

Models load sequentially (transcribe → free VRAM → diarize) to fit 8 GB.

## Project layout

| Path | What |
|------|------|
| `docker/nemo/pipeline.py` | Shared engine (pure helpers + model helpers) |
| `docker/nemo/app.py` | Gradio web app |
| `docker/nemo/diarize_transcribe.py` | Batch CLI (same engine) |
| `docker/nemo/Dockerfile` | CUDA + NeMo + Whisper + Gradio image |
| `docker/nemo/tests/` | Unit tests (run on CPU, no GPU) |
| `Transcribe-App.bat` | Launch the web app |
| `Transcribe-Diarize.bat` | Batch folder, no UI |
| `docs/` | Design spec + implementation plan |

## Manual run

```bash
# build
docker build -t nemo-sortformer docker/nemo

# web app
docker run --rm --gpus all -p 7860:7860 \
  -v "$PWD/in:/in" -v "$PWD/out:/out" -v "$PWD/archive:/archive" \
  -v nemo-cache:/root/.cache \
  --entrypoint python3 nemo-sortformer /work/app.py

# batch (no UI)
docker run --rm --gpus all \
  -v "$PWD/in:/in" -v "$PWD/out:/out" -v "$PWD/archive:/archive" \
  -v nemo-cache:/root/.cache nemo-sortformer
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Unit tests run on CPU without the models:

```bash
python -m pytest docker/nemo/tests -v
```

## License & models

Code is **MIT** (see [LICENSE](LICENSE)). The models downloaded at runtime have their **own** licenses you accept by using them — notably **NVIDIA's Sortformer license** (check it before commercial redistribution). Whisper is MIT. Your MIT code license does not override the model licenses.

## Acknowledgements

[OpenAI Whisper](https://github.com/openai/whisper) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Sortformer) · [Gradio](https://github.com/gradio-app/gradio).
