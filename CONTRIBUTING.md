# Contributing

Thanks for your interest. This is a local, GPU-based meeting transcriber, and contributions of any kind (bug reports, fixes, features, docs) are welcome.

## Ground rules

- Keep it local-first and private. No feature should send audio off the user's machine by default.
- Match the existing style: small focused functions, the shared `pipeline.py` engine, no unnecessary dependencies.
- Pure logic goes in `pipeline.py` as stdlib-only functions so it stays unit-testable without a GPU.

## Dev setup

The app runs in Docker only. There is no host-Python path for it. You need an NVIDIA GPU with 8 GB of VRAM or more and Docker with GPU support to run it, plus Python 3.10 to 3.12 on the host if you want to run the unit tests.

```bash
git clone <your-fork>
cd <repo>/docker

# build and run the app on http://localhost:7860
docker compose up
```

For unit tests you only need a plain Python env, no GPU and no NeMo:

```bash
python -m venv .venv && . .venv/Scripts/activate   # or source .venv/bin/activate
pip install pytest
python -m pytest docker/nemo/tests -v
```

## Architecture

`docker/nemo/pipeline.py` is the single source of truth for logic. It holds two kinds of functions:

- Pure (stdlib-only, unit-tested): `decide_mode`, `assign_speakers`, `smooth_sentences`, `build_turns`, `build_turns_overlap`, `namespace_speakers`, `render`, `pick_sample_turns`, `display_name`.
- Model or IO (lazy-import torch, faster-whisper, or NeMo inside the body): `plan_channels`, `transcribe`, `diarize`, `pick_solo_windows`, `suppress_cross_track_bleed`, `extract_clip`.

`docker/nemo/app.py` is the Gradio UI. It calls `pipeline.py` and holds no logic of its own beyond wiring. `docker/nemo/stagebench_app.py` is the experiment bench for running stages, labeling ground truth, and scoring runs, and it calls `pipeline.py` too.

If you add logic, put it in `pipeline.py` and write a test. The app and the bench both benefit.

## Testing changes

For pure logic, add a test in `docker/nemo/tests/test_pipeline.py` and run pytest. It is fast and CPU-only.

For model and GPU paths, there is a headless integration check pattern: run the app's `process_file` and `save_named` inside the container on a sample clip, then verify the output. Don't claim a model-path change works without running it on the GPU.

## Pull requests

1. Branch from `main`.
2. Keep PRs focused, one concern per PR.
3. Make sure `python -m pytest docker/nemo/tests` passes.
4. If you touched a model or GPU path, describe how you verified it: the sample you ran and what you checked.
5. Update `README.md` and the changelog if behavior changed.

## Reporting bugs

Open an issue with what you ran, what you expected, what happened, your GPU and VRAM, and the relevant console output. For diarization-quality issues, a short description of the audio helps a lot: how many speakers, how much crosstalk, and the channel layout.

## License

By contributing you agree your contributions are licensed under the project's [MIT License](LICENSE).
