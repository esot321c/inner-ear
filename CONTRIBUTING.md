# Contributing

Thanks for your interest! This is a local, GPU-based meeting transcriber. Contributions — bug reports, fixes, features, docs — are welcome.

## Ground rules

- Keep it **local-first and private** — no feature should send audio off the user's machine by default.
- Match the existing style: small focused functions, the shared `pipeline.py` engine, no unnecessary dependencies.
- Pure logic goes in `pipeline.py` as stdlib-only functions so it stays unit-testable without a GPU.

## Dev setup

You need an NVIDIA GPU (≥8 GB), Docker with GPU support, and Python 3.10–3.12 for running the unit tests.

```bash
git clone <your-fork>
cd <repo>

# build the image (engine + app)
docker build -t nemo-sortformer docker/nemo
```

For unit tests you only need a plain Python env (no GPU, no NeMo):

```bash
python -m venv .venv && . .venv/Scripts/activate   # or source .venv/bin/activate
pip install pytest
python -m pytest docker/nemo/tests -v
```

## Architecture

- **`docker/nemo/pipeline.py`** — the single source of truth for logic. Two kinds of functions:
  - *Pure* (stdlib-only, unit-tested): `decide_mode`, `assign_speakers`, `smooth_sentences`, `build_turns`, `render`, `pick_sample_turns`, `display_name`.
  - *Model/IO* (lazy-import torch/faster-whisper/NeMo inside the body): `plan_channels`, `transcribe`, `diarize`, `extract_clip`.
- **`docker/nemo/app.py`** — Gradio UI. Calls `pipeline.py` only; no logic of its own beyond wiring.
- **`docker/nemo/diarize_transcribe.py`** — headless batch CLI. Also calls `pipeline.py`.

If you add logic, put it in `pipeline.py` and write a test. The app and CLI both benefit.

## Testing changes

- **Pure logic:** add a test in `docker/nemo/tests/test_pipeline.py` and run pytest (fast, CPU).
- **Model/GPU paths:** there's a headless integration check pattern — run the app's `process_file` / `save_named` inside the container on a sample clip and verify the output. Don't claim a model-path change works without running it on the GPU.

## Pull requests

1. Branch from `main`.
2. Keep PRs focused; one concern per PR.
3. Make sure `python -m pytest docker/nemo/tests` passes.
4. If you touched a model/GPU path, describe how you verified it (the sample you ran, what you checked).
5. Update `README.md` / docs if behavior changed.

## Reporting bugs

Open an issue with: what you ran, what you expected, what happened, your GPU/VRAM, and the relevant console output. For diarization-quality issues, a short description of the audio (number of speakers, crosstalk, channel layout) helps a lot.

## License

By contributing you agree your contributions are licensed under the project's [MIT License](LICENSE).
