# Changelog

All notable changes to Inner Ear are documented here. This project adheres to
[Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.3.0] - 2026-08-05

### Added
- **Every track is diarized.** Multi-track and stereo recordings no longer treat
  one track as "you" and diarize only the other. Each track is diarized on its
  own and picks its own engine by its own speaker count (cascade for ≤2, DiCoW
  for ≥3), so a track carrying several people is separated instead of being
  labeled as one person. Speaker ids are namespaced per track (`t0:speaker_1`)
  so they stay unique once pooled, and you name every speaker found.
- **Cross-track bleed suppression.** When each participant is on their own track,
  every mic still faintly picks up the others, which the per-track diarizer
  would otherwise report as an extra speaker. A pooling barrier now compares all
  tracks against each other and drops bleed in two passes: stage 1 removes a word
  only when all three signals agree (time overlap, ≥6 dB louder on another track,
  same normalized text), so genuine simultaneous speech is kept; stage 2 drops a
  whole (track, speaker) cluster whose surviving words are mostly energy-dominated
  by another track, catching faint bleed that mis-transcribed to a different token.
  A cluster needs ≥3 measured words before it can be dropped.
- **Clean solo naming clips.** Each speaker's sample clip is cut from a window
  where that speaker is talking alone on their own track (`pick_solo_windows`),
  instead of a slice that may contain crosstalk.
- **Per-stage artifacts** written to `out/<name>/stages/` for diagnosing a bad
  run: channel detection, raw and post-suppression per-track transcripts, pooled
  words, and final turns.

### Changed
- **The "Which channel is YOU?" controls are gone** from the UI, along with the
  `YOU_NAME` / `YOU_CH` settings. Every speaker is named the same way, so there
  is nothing to configure per recording. Naming rows now scale to the pooled
  speaker count across tracks rather than a single track's cap.
- **LLM boundary cleanup is scoped to cascade words only.** DiCoW output is
  already per-speaker and the no-diarization fallback has no posterior to reason
  over, so both are left untouched.
- **README and CONTRIBUTING rewritten.** The README described the retired
  two-channel flow and advertised `YOU_NAME` / `YOU_CH`, which the code no longer
  reads. It now documents what the pipeline actually does, states that Docker
  Compose is the only run path, and drops the duplicated run instructions that
  had accumulated across three sections. CONTRIBUTING lists the current
  `pipeline.py` functions.

### Removed
- **The last of the legacy whisply stack.** The 0.2.0 release retired the `.bat`
  launchers but left the old path's files behind; they were orphaned (nothing in
  the Compose path referenced them) and are now gone: `Transcribe-Folder.ps1`,
  `docker/Transcribe-Docker.bat`, the old whisply `docker/Dockerfile` +
  `entrypoint.sh` + `README-docker.txt`, the whisply `config.json`, and
  `pylib/sitecustomize.py` (a speechbrain patch for the host venv, unused in the
  container). **Docker Compose is the only way to run Inner Ear.**
- `YOU_NAME` / `YOU_CH` settings, along with the stale `diarize_transcribe.py`
  reference in the contributor docs.

### Fixed
- **Turn merging no longer swallows interruptions.** `build_turns_overlap` is
  boundary-aware, so an interjection inside another speaker's turn breaks the
  turn instead of being absorbed into it, while genuinely uninterrupted speech
  still merges.
- **`rms_db` is bounds-safe past the end of a track's audio.** A window beyond
  the samples returns "no evidence" rather than reading out of range, so tracks
  of unequal length can't produce a bad loudness comparison.

## [0.2.0] - 2026-06-25

### Added
- **DiCoW auto-escalation for 3+ speakers.** When the diarizer detects ≥3
  speakers, the pipeline switches from the cascade to DiCoW (Diarization-Conditioned
  Whisper), which decodes each speaker separately so overlapping crosstalk is
  attributed inside the model. Measured to cut 3-speaker boundary mistakes by ~63%
  vs the cascade on a hand-labeled podcast. Threshold via `DICOW_MIN_SPK` (default
  3; set high to disable). Two-speaker meetings stay on the fast path.
- **Speaker-accuracy harness** (stage bench): run the pipeline stage by stage,
  correct the transcript in-browser and **save it as ground truth**, and every run
  is scored live against it. Reports **boundary mistakes** (the honest metric —
  adjacent wrong words count once) plus a separate back-channel number; logs every
  run with its settings for side-by-side comparison; partial labeling via an
  `---END---` marker; a clip tool to cut test sections from long files.
- **pyannote** available as an alternative diarizer backend (research/bench only).
- **Reproducible image + Docker Compose.** The exact validated environment is
  pinned in `requirements.lock.txt`; `docker compose run --rm --service-ports app`
  builds and runs the UI (ephemeral), `--profile bench` runs the stage bench.
- **Lab notebook** (`docs/experiments/diarization-quality-log.md`) documenting every
  experiment, dataset, and result — including the ideas that didn't work.

### Changed
- **Pipeline restructured to diarize-first**: count speakers, then pick the engine
  (cascade for ≤2, DiCoW for ≥3). DiCoW output gets per-speaker punctuation/casing
  and an overlap-aware turn builder so simultaneous speech isn't shattered into
  one-word turns.
- **Docker Compose is the only run path.** Retired the `.bat` launchers and the
  superseded standalone scripts; the image rebuilds from scratch via the lock file.

### Fixed
- **Rapid back-channels stuck to the wrong speaker.** A new transition-fix
  reassigns words whose diarization posterior shifts speaker mid-word (a "yeah"
  smeared across a turn boundary) to the speaker at the word's onset — 10→2
  boundary mistakes on a rapid two-person call, back-channels 50%→100%.

## [0.1.0] - 2026-06-24

### Added
- **Live progress** while processing: per-file status and a disabled "working"
  button so the app no longer looks frozen during a long run.
- `in/` ships in fresh clones (drop audio there); its contents stay gitignored.

### Changed
- Renamed the app to **Inner Ear**.
- Standardized the model env var on `WHISPER_MODEL` (`WHISPLY_MODEL` kept as a
  backward-compatible alias).

### Fixed
- **"Nothing to save yet" after long jobs.** Processed results are now persisted to
  disk and reloaded at save time, so saving works even after the session's
  in-memory state is evicted (idle/backgrounded tab) or the server restarts.
- **Silence transcribed as "Thank you." loops.** Voice-activity detection is on, so
  trailing dead air is skipped instead of hallucinated — which also stops it from
  poisoning diarization by inventing a speaker for the silence.
- **Speaker attribution rebuilt to work across meeting types.** The old step
  reassigned each sentence to its loudest speaker, which flattened rapid
  interrupt-heavy talk. Attribution now uses Sortformer's soft per-speaker
  posteriors with a sentence-aware Viterbi resolver plus a grammar-based
  onset-leak fix ("did you", "how much" no longer stick to the wrong person).
  It handles both a rapid two-person mono call and 4-speaker crosstalk -
  verified turn-for-turn against real ground truth. (A lightweight despeckle
  remains as a fallback when soft posteriors aren't available.)
- **Giant voice-naming clips.** Each voice's sample is now a short (~12 s) slice
  instead of the speaker's entire longest turn.
