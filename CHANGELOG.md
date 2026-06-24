# Changelog

All notable changes to Inner Ear are documented here.

## [Unreleased]

### Fixed
- **"Nothing to save yet" after long jobs.** Processed results are now persisted to
  disk and reloaded at save time, so saving works even after the session's
  in-memory state is evicted (idle/backgrounded tab) or the server restarts.
- **Silence transcribed as "Thank you." loops.** Voice-activity detection is on, so
  trailing dead air is skipped instead of hallucinated — which also stops it from
  poisoning diarization by inventing a speaker for the silence.
- **Rapid conversations flattened onto one speaker.** Replaced the
  reassign-each-sentence-to-its-loudest-speaker smoothing (which erased real
  interrupt-heavy back-and-forth) with a conservative despeckle that only fixes
  isolated single-word jitter.
- **Giant voice-naming clips.** Each voice's sample is now a short (~12 s) slice
  instead of the speaker's entire longest turn.

### Added
- **Live progress** while processing: per-file status and a disabled "working"
  button so the app no longer looks frozen during a long run.
- `in/` ships in fresh clones (drop audio there); its contents stay gitignored.

### Changed
- Renamed the app to **Inner Ear**.
- Standardized the model env var on `WHISPER_MODEL` (`WHISPLY_MODEL` kept as a
  backward-compatible alias).
