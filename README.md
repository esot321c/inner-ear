# Inner Ear

> Local, GPU-accelerated meeting transcription with accurate speaker separation, even on a single mixed track where similar voices talk over each other. Nothing leaves your machine.

Transcribe a recording, see the voices it detected, play a clip of each and name them, then save a labeled transcript. Built for messy real-world audio: crosstalk, interruptions, several people at once.

---

## Why this exists

Most "Whisper + diarization" setups use pyannote, which collapses similar-sounding voices into one speaker on hard audio. Inner Ear uses NVIDIA NeMo streaming Sortformer instead. It is overlap-aware and separates voices pyannote can't. On a fast, crosstalk-heavy podcast clip, pyannote tagged 88% of words as a single speaker; Sortformer got the speakers right.

## What it does

- Transcribes with OpenAI Whisper `large-v3` (via faster-whisper), which handles fast and quiet speech well.
- Skips silence using voice-activity detection, so dead air doesn't turn into filler or hallucinated text.
- Separates speakers with NeMo streaming Sortformer, overlap-aware and within 8 GB of VRAM.
- Diarizes every track independently, then pools the speakers. A track carrying several people gets separated instead of lumped together, and you name everyone it found.
- Suppresses cross-track bleed. When each participant has their own track, every mic still faintly hears the others. A word that is a quieter duplicate of what another track heard louder gets dropped, so bleed doesn't become a phantom speaker.
- Escalates to DiCoW (Diarization-Conditioned Whisper) on tracks with 3 or more speakers. It decodes each speaker separately, so overlapping crosstalk is attributed inside the model: about 63% fewer 3-speaker mistakes than the cascade.
- Fixes boundaries, so mis-timed back-channels ("yeah", "okay") and stranded turn-openers snap back to the right speaker. On a rapid two-person call that took mistakes from 10 down to 2.
- Runs single meetings interactively or a whole folder in batch.
- Stays local. No cloud, no uploads.

## Requirements

Inner Ear runs in Docker, always. The GPU stack (CUDA, NeMo, faster-whisper, DiCoW) is pinned inside the image, and Docker Compose is the only supported way to run it. There is no host-Python install path for the app. The one thing that runs on the host is the CPU test suite.

- An NVIDIA GPU with 8 GB of VRAM or more (developed on an RTX 3070).
- Docker with GPU support (NVIDIA Container Toolkit; on Windows, Docker Desktop plus WSL2) and Docker Compose.
- About 12 GB of disk for the image, plus 6 to 10 GB of models fetched on first run and cached in the `nemo-cache` volume.

## Quick start

```bash
cd docker
docker compose up
```

The first run builds the image (roughly 20 minutes) and downloads models. Then:

1. Open http://localhost:7860.
2. Upload a meeting, hit Transcribe, play each voice, type names, then Save.
3. Your transcript goes to `out/<name>/` and the source file moves to `archive/`.
4. Press Ctrl+C to stop, then run `docker compose down` to remove the container.

Other commands:

```bash
docker compose up -d                      # run in the background
docker compose --profile bench up bench   # experiment stage bench on :7861
```

The `in/`, `out/`, and `archive/` folders are mounted into the container. Model weights persist in the `nemo-cache` volume, so they only download once.

## Usage

For a single meeting, upload it, transcribe, name each voice, and save. There is no channel setup because every track goes through the same path.

For a batch, drop files in `in/`, open the Batch folder tab, click Process all, then pick each meeting to name and save.

For the best results, record with OBS multi-track: mic to Track 1, desktop audio to Track 2 (Settings, then Output, then Recording). That writes separate tracks into one MKV and the pipeline pulls each one independently. A stereo file with your mic left and the call right also works, and a single mixed track gets diarized as-is.

## Configuration

You don't need any. Whisper, Sortformer, and DiCoW are open models that download without a token. To change a default, copy `.env.example` to `.env` (which is gitignored):

| Variable | Default | What it does |
|----------|---------|--------------|
| `WHISPER_MODEL` | `large-v3` | Use `large-v3-turbo` to trade accuracy for speed |
| `DICOW_MIN_SPK` | `3` | Speakers on a track before escalating to DiCoW. Set it high (`99`) to always use the cascade |
| `LLM_BASE_URL` / `LLM_MODEL` | unset | Optional OpenAI-compatible endpoint for an extra boundary-cleanup pass |
| `HF_TOKEN` | unset | Only for the optional [pyannote](https://huggingface.co/pyannote/speaker-diarization-3.1) research backend in the stage bench |

## How it works

```
audio file
  ├─ channel detection (multi-track / stereo / mono)          pipeline.plan_channels
  │
  ├─ FOR EACH TRACK (no track is special):
  │    ├─ diarize (NeMo Sortformer) -> soft posterior         pipeline.diarize_soft
  │    ├─ count speakers -> pick this track's engine          pipeline.count_speakers
  │    │
  │    ├─ 2 or fewer speakers: CASCADE (fast path)
  │    │    ├─ transcribe (Whisper large-v3)                  pipeline.transcribe
  │    │    ├─ attribute per word from the soft posterior     pipeline.resolve_speakers
  │    │    └─ boundary fixes (transition-fix, onset-fix)     pipeline.fix_transition_leaks
  │    │
  │    └─ 3 or more speakers: DiCoW
  │         └─ decode each speaker separately, conditioned    dicow_backend.transcribe_dicow
  │            on the Sortformer diarization, overlap in-model
  │
  ├─ BARRIER: all tracks transcribed
  │    └─ drop cross-track bleed (quieter duplicate of a      pipeline.suppress_cross_track_bleed
  │       word another track heard louder)
  │
  └─ namespace speaker ids per track, pool, group into        pipeline.namespace_speakers
     turns, cut a solo clip per speaker, you name them        pipeline.build_turns_overlap
```

Models load one at a time (diarize, free the VRAM, then transcribe or run DiCoW) to fit in 8 GB. Per-stage artifacts go to `out/<name>/stages/` if you need to diagnose a bad run.

## Project layout

| Path | What |
|------|------|
| `docker/nemo/pipeline.py` | Shared engine: transcribe, diarize, attribute, fixes, bleed suppression |
| `docker/nemo/app.py` | Gradio web app: per-track diarize, cascade/DiCoW branch, pooling |
| `docker/nemo/dicow_backend.py` | DiCoW backend |
| `docker/nemo/pyannote_backend.py` | pyannote backend, research and bench only |
| `docker/nemo/scoring.py` | Word-level speaker-accuracy scorer (boundary mistakes) |
| `docker/nemo/stagebench_app.py` | Experiment bench: run stages, label truth, score runs |
| `docker/nemo/Dockerfile` + `requirements.lock.txt` | Reproducible image, fully pinned |
| `docker/docker-compose.yml` | `app` (production) and `bench` services |
| `docker/nemo/tests/` | Unit tests, CPU only |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Pure logic lives in `pipeline.py` as stdlib-only functions, so the tests need nothing but Python:

```bash
python -m pytest docker/nemo/tests -v
```

Anything that touches a model or the GPU runs in the container.

## License and models

The code is MIT (see [LICENSE](LICENSE)). Models downloaded at runtime carry their own licenses, which you accept by using them. NVIDIA's Sortformer license is the one to read before any commercial redistribution. Whisper is MIT and DiCoW is CC-BY-4.0. Your MIT code license does not override any of them.

## Acknowledgements

[OpenAI Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Sortformer), [DiCoW](https://huggingface.co/BUT-FIT/DiCoW_v3_3), and [Gradio](https://github.com/gradio-app/gradio).
