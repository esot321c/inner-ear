# Transcription + Diarization App — Design

Date: 2026-06-22
Status: Approved (design), pending implementation plan

## Goal

Turn the local transcribe + speaker-diarization pipeline into a usable **local web
app** with a speaker-naming UI, so Graham can transcribe meetings, see the voices
it detected, play a clip of each, name them, and save a clean named transcript —
without touching the command line or the Docker folder.

Everything runs locally on the RTX 3070 (8 GB) via Docker. No audio leaves the
machine.

## Requirements (from brainstorming)

- **Two modes:** single-meeting (interactive) AND batch-folder (process many, then
  name each).
- **Naming:** fresh per meeting (type names each time). Structured so voice
  *enrollment* (auto-recognising returning voices) can be added later — NOT built now.
- **Input:** one combined file per meeting. **Recommended: OBS multi-track** — mic
  on Track 1, desktop audio on Track 2, recorded as separate independent audio
  streams in one MKV/MP4 (true isolation, no mono-in-stereo compromise). The
  pipeline pulls each track independently. A stereo file (you on L, far side on R)
  is supported as a fallback. Mono / dual-mono is diarized as a single mixed track.
  All auto-detected — no filename/grouping convention needed.
  - OBS setup: Settings → Output → Recording, enable audio tracks; assign
    Mic/Aux → Track 1 only, Desktop Audio → Track 2 only (the default mixes all
    sources into Track 1, which defeats the isolation).
- **Frontend:** Gradio (Python) web UI, in the Docker image. Not an exe (the
  CUDA/NeMo/Whisper stack does not package into a Windows exe cleanly).

## Architecture

- **Engine (shared):** refactor `docker/nemo/diarize_transcribe.py` into a
  `pipeline.py` module of importable functions. Both the batch CLI and the Gradio
  app call the same functions — one engine, no duplicated logic.
- **App:** `docker/nemo/app.py`, a Gradio app, added to the existing
  `nemo-sortformer` image (Gradio added to the Dockerfile).
- **Launcher:** `Transcribe-App.bat` starts the container with `--gpus all`, mounts
  `in\` `out\` `archive\` and the model cache volume, maps port 7860, and opens the
  browser to `http://localhost:7860`.
- **Models:** loaded sequentially per job (Whisper → free → Sortformer) to stay
  within 8 GB VRAM. First job of a session is slower (model load).

### Shared `pipeline.py` surface (functions)

- `plan_channels(file) -> (mode_label, [(role, wav)])` — ffprobe + L/R-difference
  detection; returns `single` (mono/dual-mono) or `dual` (stereo or multi-track,
  splitting `you` / `far`).
- `transcribe(wav) -> [word]` — faster-whisper large-v3, word timestamps, VAD off.
- `diarize(wav) -> [(start,end,speaker)]` — streaming Sortformer.
- `assign_and_smooth(words, segs)` — per-word speaker by midpoint, then
  sentence-level majority smoothing.
- `build_turns(words) -> [turn]` — group consecutive same-speaker words.
- `extract_speaker_samples(wav, turns) -> {speaker: clip_path}` — a representative
  audio clip per detected speaker (their longest clean turn) for the play button.
- `render(turns, names) -> (txt, json)` — apply the speaker→name map and produce
  the final transcript text + json.

## Two modes (Gradio tabs)

### Single meeting (interactive)
1. Upload one combined file.
2. Process → run pipeline.
3. If dual-channel: play a few seconds of **each channel**, ask "which is you?".
   The chosen channel = "You" (or a set name). (This also resolves which-channel
   ambiguity.)
4. For each **other** detected voice: a row with a play button (sample clip) + a
   name input box.
5. Type names → Save → write named transcript to `out\<meeting>\`, archive source.

### Batch folder
1. Process everything in `in\` unattended (one file = one meeting).
2. A queue lists finished meetings. Generic `Speaker 1/2/3` until named.
3. Click a meeting → same naming UI as above → Save updates that transcript.

## Data flow

```
upload / in\ file
  -> plan_channels  (ffmpeg split if dual)
  -> transcribe (Whisper large-v3)          [load, run, free]
  -> diarize    (Sortformer, far/mono only) [load, run, free]
  -> assign_and_smooth (sentence majority)
  -> build_turns + extract_speaker_samples
  -> [user names speakers in UI]
  -> render(named) -> out\<meeting>\<meeting>.txt + .json
  -> move source -> archive\<timestamp>\
```

## Outputs

- `out\<meeting>\<meeting>.txt` — named, smoothed, `[hh:mm:ss] Name: text` turns.
- `out\<meeting>\<meeting>.json` — turns + metadata (mode, speakers).
- Source moved to `archive\<timestamp>\`. (Converted wavs are temp, never archived.)

## Error handling

- A file that fails ffmpeg/transcription/diarization is skipped with a visible
  message; the batch continues. No crash.
- Single-user, one GPU job at a time; Gradio's queue serialises requests.

## Designed-for-later (NOT in v1)

- Voice **enrollment** / cross-meeting recognition. v1 keeps per-speaker sample
  clips + (optionally) embeddings in the json so enrollment can hook in later.
- Prettier custom HTML/JS frontend (Gradio first; rebuild later if wanted).

## Out of scope (YAGNI)

- Windows exe packaging.
- Cloud/remote anything.
- Real-time/streaming transcription (offline files only).
- Editing transcript text in the UI (name speakers only).

## Open implementation notes

- Keeping both models resident would exceed 8 GB; sequential load/free is the
  constraint that drives the per-job flow.
- Gradio + the GPU pipeline run **in the same container**, so the app calls the
  pipeline functions in-process (no shelling out to `docker run`).
