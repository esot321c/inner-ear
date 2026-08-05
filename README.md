# Inner Ear

> Local, GPU-accelerated meeting transcription with **accurate speaker separation** — even on a single mixed track with similar voices talking over each other. Nothing leaves your machine.

Transcribe a recording, see the voices it detected, **play a clip of each and name them**, and save a clean labeled transcript. Built for messy real-world audio: crosstalk, interruptions, multiple speakers.

---

## Why this exists

Most "Whisper + diarization" setups use **pyannote**, which collapses similar-sounding voices into one speaker on hard audio. This project uses **NVIDIA NeMo streaming Sortformer** instead, which is overlap-aware and separates voices that pyannote can't — validated on a fast, crosstalk-heavy podcast clip where pyannote tagged 88% of words as a single speaker and Sortformer got the speakers right.

## What it does

- **Transcribes** with OpenAI Whisper `large-v3` (via faster-whisper) — accurate, captures fast/quiet speech.
- **Skips silence** (voice-activity detection), so dead air isn't transcribed as filler or hallucinated text.
- **Separates speakers** with NeMo streaming Sortformer — handles long files within 8 GB VRAM.
- **Boundary fixes** snap mis-timed back-channels ("yeah", "okay") and stranded turn-openers back to the right speaker — a measured 10→2 mistakes on a rapid two-person call.
- **Auto-escalates to DiCoW** (Diarization-Conditioned Whisper) when **3+ speakers** are detected on a track: it decodes each speaker separately, so overlapping crosstalk is attributed inside the model. Measured to cut 3-speaker mistakes by ~63% vs the standard cascade. Two-speaker tracks stay on the fast path. Threshold via `DICOW_MIN_SPK`.
- **Every track is diarized.** Multi-track and stereo recordings are split and each track is diarized on its own, so a track with several people on it is separated rather than lumped together. Speakers are pooled across tracks and you name all of them.
- **Cross-track bleed suppression** — when you and the call are on separate tracks, each mic still faintly hears the other. Words that are a quieter duplicate of what another track heard louder are dropped, so bleed doesn't show up as a phantom extra speaker.
- **Naming UI** — plays a clean solo clip of each detected voice so you can name them.
- **Single** (interactive) and **batch** (whole-folder) modes.
- Fully local. No cloud, no uploads.

## Requirements

**Inner Ear runs in Docker — always.** There is no host-Python install path for the app: the GPU stack (CUDA, NeMo, faster-whisper, DiCoW) is pinned inside the image, and the only supported way to run it is Docker Compose. The one thing you can run on the host is the CPU unit-test suite (see [Development](#development)).

- **NVIDIA GPU with ≥8 GB VRAM** (developed on an RTX 3070).
- **Docker** with GPU support (NVIDIA Container Toolkit; on Windows, Docker Desktop + WSL2) and **Docker Compose**.
- ~12 GB disk for the image, plus ~6–10 GB of models downloaded on first run (Whisper, Sortformer, and DiCoW for 3+ speakers), cached after in the `nemo-cache` volume.

## Quick start

```bash
cd docker
docker compose up
```

First run builds the image (~20 min) and downloads models. Then:

1. Open **http://localhost:7860**.
2. Upload a meeting → **Transcribe** → play each voice, type names → **Save**.
3. Transcript lands in `out/<name>/`, the source moves to `archive/`.
4. **Ctrl+C** to stop, then `docker compose down` to remove the container.

That's the whole thing. `docker compose up` starts just the web app — the stage
bench sits behind a profile, so it won't come up unless you ask for it.

## Usage

**Single meeting:** upload, transcribe, name each voice it found, save. No channel setup — every track is diarized and every speaker is named the same way.

**Batch:** drop files in `in/`, open the **Batch folder** tab, **Process all**, then pick each meeting to name and save.

**Best capture (recommended):** OBS multi-track — **Mic → Track 1**, **Desktop audio → Track 2** (Settings → Output → Recording). Recorded as separate tracks in one MKV; the pipeline pulls each independently. A stereo file (mic left, call right) also works. A single mixed track is diarized as-is.

## Running

Everything runs through `docker/docker-compose.yml`. The image is defined in
`docker/nemo/Dockerfile` with dependencies pinned in `requirements.lock.txt`, so
the environment is reproducible.

```bash
cd docker

docker compose up          # web app -> http://localhost:7860
docker compose up -d       # same, in the background
docker compose down        # stop and remove the container

# experiment stage bench -> http://localhost:7861
docker compose --profile bench up bench
```

Host paths `in/`, `out/`, and `archive/` are mounted into the container, and model
weights persist in the `nemo-cache` volume so they download only once.

## Configuration

Optional. Copy `.env.example` to `.env` (gitignored) if you want to change anything:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

- **The main app needs no configuration and no token** — Whisper, NeMo
  Sortformer, and DiCoW are all open models (DiCoW is CC-BY-4.0).
- `WHISPER_MODEL` (default `large-v3`) — or `large-v3-turbo` for speed.
- `DICOW_MIN_SPK` (default `3`) — auto-escalate to DiCoW at this many detected
  speakers on a track. Set to a large number (e.g. `99`) to always use the fast
  cascade.
- `LLM_BASE_URL` / `LLM_MODEL` — optional OpenAI-compatible endpoint for an extra
  boundary-cleanup pass on cascade words.
- `HF_TOKEN` is **only** needed for the optional **pyannote** research backend
  (used in the stage bench, not the main app). Get a free token at
  [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and
  accept the terms for
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

## How it works

```
audio file
  ├─ channel detection (multi-track / stereo / mono)          pipeline.plan_channels
  │
  ├─ FOR EACH TRACK (no track is special):
  │    ├─ diarize (NeMo Sortformer) -> soft posterior         pipeline.diarize_soft
  │    ├─ count speakers -> pick this track's engine          pipeline.count_speakers
  │    │
  │    ├─ ≤2 speakers — CASCADE (fast path)
  │    │    ├─ transcribe (Whisper large-v3)                  pipeline.transcribe
  │    │    ├─ attribute per word from the soft posterior     pipeline.resolve_speakers
  │    │    └─ boundary fixes (transition-fix, onset-fix)     pipeline.fix_transition_leaks
  │    │
  │    └─ ≥3 speakers — DiCoW (diarization-conditioned Whisper)
  │         └─ decode each speaker separately, conditioned    dicow_backend.transcribe_dicow
  │            on the Sortformer diarization → overlap in-model
  │
  ├─ BARRIER: all tracks transcribed
  │    └─ drop cross-track bleed (quieter duplicate of a      pipeline.suppress_cross_track_bleed
  │       word another track heard louder)
  │
  └─ namespace speaker ids per track, pool, group into        pipeline.namespace_speakers
     turns, cut a solo clip per speaker, you name them        pipeline.build_turns_overlap
```

Models load sequentially (diarize → free VRAM → transcribe/DiCoW) to fit 8 GB.
Per-stage artifacts are written to `out/<name>/stages/` for diagnostics.

## Project layout

| Path | What |
|------|------|
| `docker/nemo/pipeline.py` | Shared engine (transcribe, diarize, attribute, fixes, bleed suppression) |
| `docker/nemo/app.py` | Gradio web app — per-track diarize, cascade/DiCoW branch, pooling |
| `docker/nemo/dicow_backend.py` | DiCoW (diarization-conditioned Whisper) for 3+ speakers |
| `docker/nemo/pyannote_backend.py` | pyannote backend (research/bench only) |
| `docker/nemo/scoring.py` | Word-level speaker-accuracy scorer (boundary mistakes) |
| `docker/nemo/stagebench_app.py` | Experiment bench: run stages, label truth, score runs |
| `docker/nemo/Dockerfile` + `requirements.lock.txt` | Reproducible image (pinned) |
| `docker/docker-compose.yml` | `app` (production) + `bench` services |
| `docker/nemo/tests/` | Unit tests (run on CPU, no GPU) |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Pure logic lives in `pipeline.py` as
stdlib-only functions, so the unit tests run on the host with plain Python — no
GPU, no Docker, no models:

```bash
python -m pytest docker/nemo/tests -v
```

Anything that touches a model or the GPU runs in the container.

## License & models

Code is **MIT** (see [LICENSE](LICENSE)). The models downloaded at runtime have their **own** licenses you accept by using them — notably **NVIDIA's Sortformer license** (check it before commercial redistribution). Whisper is MIT. Your MIT code license does not override the model licenses.

## Acknowledgements

[OpenAI Whisper](https://github.com/openai/whisper) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Sortformer) · [Gradio](https://github.com/gradio-app/gradio).
