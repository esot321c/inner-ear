# Diarization pipeline redesign — enriched data + hardware-aware quality

**Date:** 2026-06-23
**Status:** Design, pending review
**Goal:** Make single-mixed-track speaker attribution actually good, by stopping the
pipeline from throwing away information at each step and by spending whatever the
machine has to run the best models. Quality first; slower is acceptable.

## Problem

On a real two-person mono call (`2026-06-23 12-59-22.mp4`) the current pipeline
badly mis-attributes speech, while a clean studio podcast (also single mono track,
4 speakers) works well. Investigation this session established:

- It is **not** the channel (podcast is mono too), **not** audio quality (both
  ~110–128 kbps), and **not** the diarizer model: streaming vs offline Sortformer
  produce the *same* result.
- The diarizer's **raw** output is actually correct (first 138 s: speaker_0 142
  words / speaker_1 89 words, properly alternating, matches user ground truth
  turn-for-turn).
- Two of **our own steps** were destroying that good data:
  1. `smooth_sentences` (dominance-flatten) collapsed rapid interrupt-heavy
     back-and-forth — five real turns flattened to one. (Already replaced this
     session with a conservative despeckle, but that loses the legitimate
     sentence-completion / "trailing off" behaviour we actually want.)
  2. `assign_speakers` tags each word by its **midpoint** against **hard**
     diarizer segments. The new speaker's onset word ("did", "how", "i") sits in
     the fade zone and lands on the previous speaker — a consistent one-word lag a
     global time offset cannot fix without breaking the other direction.

Root cause: we **collapse rich signals into bare tokens too early**. Sortformer
emits a *soft posterior ramp* at each handoff; we threshold it to a line and
discard the ramp. Whisper knows word confidence and sentence structure; we drop
both. The step that should reconcile speakers (sentence smoothing) is left with
nothing to reason from.

## Goals / non-goals

**Goals**
- Correctly attribute fast, overlapping, interrupt-heavy mono conversations.
- Preserve sentence-completion: a speaker trailing off keeps their tail; do not
  flatten genuine rapid turns.
- Use the best models the hardware can run; chunk and go slow rather than degrade.
- Be auditable and regression-tested against real ground truth.

**Non-goals**
- Audio source separation (un-mixing two voices from one channel).
- Real-time / streaming output.
- More than 4 speakers (Sortformer's cap).
- Changing the dual-track / stereo path — it already short-circuits to near-perfect
  separation and stays as-is (the "you" channel is forced; only the far side runs
  the soft pipeline).

## Architecture

New pipeline shape (replaces steps 2–6 of `process_file`):

```
0. probe machine        -> Profile (models, chunk size, overlap, batch, compute type)
1. plan_channels        -> wav(s)              [dual-track unchanged]
2. transcribe (Whisper) -> [Word] + Sentence grouping (+ asr_conf)
3. diarize_soft         -> global per-speaker POSTERIOR timeline (+ seam records)
4. enrich               -> each Word gets speaker_probs[] + sentence + seam flags
5. resolve (Viterbi)    -> per-word speaker (trailing-off / onset / interjection)
6. build_turns -> render -> sample clips
```

### Data model (what persists end-to-end)

```python
Word    = { text, start, end, asr_conf,
            sentence_id, is_sentence_start, ends_sentence,
            crosses_seam, source_chunks: [int],
            speaker_probs: {spk: float},   # filled in step 4
            speaker: str }                 # filled in step 5
Sentence= { id, start, end, word_idxs: [int], crosses_seam }
Seam    = { time, chunk_a, chunk_b,
            straddling_sentence_ids: [int], straddling_word_idxs: [int] }
PosteriorTimeline = { frame_dur, frames: [[p_spk0, p_spk1, ...]], t0 }
```

Nothing collapses to a bare `(speaker, text)` token until `build_turns`.

## Pillar 1 — hardware-aware profile

A probe (`capabilities()`) reads `torch.cuda.is_available()`,
`torch.cuda.get_device_properties(0).total_memory`, and `os.cpu_count()` and
selects a `Profile`. Quality-first: always reach for the **offline** Sortformer
(`nvidia/diar_sortformer_4spk-v1`, better than streaming on recordings) and chunk
to fit.

| VRAM | diarizer | diar window / overlap | whisper |
|------|----------|----------------------|---------|
| ≥16 GB | offline | ≤600 s (else chunk) / 15 s | large-v3 fp16 |
| 8–16 GB (this box) | offline | 90 s / 15 s | large-v3 fp16 |
| <8 GB | offline | 45 s / 10 s | large-v3 fp16/int8 |
| CPU only | streaming (fallback) | n/a | large-v3 int8 |

Offline @ 90 s windows is proven to fit 8 GB this session (the 16-min whole-file is
what OOM'd). Env overrides: `WHISPER_MODEL`, `DIAR_MODEL`, `INNER_EAR_PROFILE`.

## Pillar 2a — soft diarization, chunking, seams

**Soft posteriors.** Call below NeMo's `.diarize()` to obtain frame-level sigmoid
posteriors `[frames, num_spk]`. Frame stride = `window_stride * subsampling_factor`
= `0.01 * 8` = **0.08 s**. *(Risk: this uses NeMo internals — see Risks. De-risked
by a spike before anything else is built.)*

**Chunking that respects sentences:**
- **Cut on silence, not the clock.** Target the profile window, but snap each cut
  to the nearest VAD pause within tolerance, so seams land between utterances.
- **Overlap** (per profile, ~15 s) does double duty: label alignment + posterior
  splicing.
- **Seam records** capture every join and which sentences/words straddle it.

**Label stitching.** Across an overlap region, chunk B's speaker slots are matched
to chunk A's by maximising posterior agreement (Hungarian assignment on per-speaker
probability curves over the overlap). Apply the mapping so "speaker 0" is one person
end-to-end, then **splice** the per-frame posteriors into one global timeline.

**Seam reconciliation.** A word flagged `crosses_seam` reads its posterior from the
spliced global timeline (frames from both chunks) — never half-missing. The stitcher
enforces that a seam artifact alone cannot flip a speaker inside a straddling
sentence. Auditable: log "N sentences straddled seams, all reconciled."

## Pillar 2b — sentences + the resolver

**Sentence grouping (step 2).** New sentence when the previous word ends with
`.?!` **or** the gap to the next word exceeds a pause threshold (~0.6 s). The pause
rule is essential — Whisper emits long unpunctuated runs, and punctuation-only
grouping made a 2-minute "sentence" (the original flatten bug).

**Enrichment (step 4).** For each word, average the global posterior over the
word's `[start, end]` span → normalised `speaker_probs`. Attach `sentence_id`,
sentence flags, `crosses_seam`.

**Resolve (step 5) — sentence-aware Viterbi.** Decode the per-word speaker sequence:
- **Emission**(word, spk) = `log(speaker_probs[word][spk] + eps)`.
- **Transition**(spk_prev → spk_cur): `0` if same speaker; if different,
  `P_switch_boundary` when `word.is_sentence_start`, else `P_switch_mid`
  (`P_switch_mid >> P_switch_boundary`).
- Viterbi maximises `Σ emission − Σ transition`.

This single mechanism yields all three behaviours:
- **Trailing off** ("…because of the price.") → mid-sentence, prior speaker still
  has emission mass → stays (switching is expensive).
- **New-speaker onset** ("did", "how") → new sentence + posterior crossed → switch
  is cheap at the boundary → onset word lands correctly (lag gone).
- **Real interjection** ("yeah") → posterior flips hard → strong emission beats the
  mid-sentence penalty → kept as the interjector.

Parameters (`P_switch_mid`, `P_switch_boundary`, pause threshold, eps) are tuned
against the validation fixture.

## Error handling / fallbacks

Degrade, never hard-fail:
- Offline chunk OOMs → shrink window and retry → ultimately the streaming model.
- Posterior extraction fails → fall back to hard segments + **max-overlap**
  assignment (today's path minus the flattening).
- Probe failure → conservative CPU profile.
- Zero or one speaker detected → single-speaker transcript, no crash.

## Testing & validation

**Unit (native venv, no GPU):**
- Sentence grouping: punctuation breaks, pause breaks, unpunctuated run.
- Posterior enrichment: a word spanning frames averages correctly.
- Label stitching: synthetic 2-chunk with swapped slots → re-aligned.
- Seam reconciliation: word straddling a seam gets a continuous posterior.
- Viterbi: synthetic posteriors for trailing-off (stays), onset (switches at
  sentence start), interjection (flips mid-sentence on strong posterior).

**Validation fixture — the real call.** Encode the user's ground-truth attribution
(provided this session) as a fixture. Run the full pipeline, align output words to
ground truth, compute **word-level speaker accuracy** (target ≥90 %). This becomes
a permanent regression test so the pipeline can't silently rot again. Also run the
podcast to confirm no regression on the case that already worked.

## Risks

1. **NeMo posterior extraction** uses internals, not the public `.diarize()`.
   Mitigation: spike first on a 90 s clip; if infeasible, the lighter
   hard-segment + max-overlap path is the documented fallback.
2. **Label stitching** across chunks can mis-map on hard audio. Mitigation:
   sufficient overlap, Hungarian on full posterior curves, seam logging.
3. **Viterbi tuning** overfitting one fixture. Mitigation: validate on podcast +
   the call; keep parameters few and interpretable.
4. **Speed**: offline + many chunks is slow. Accepted per requirement
   (quality-first); surfaced in the existing progress UI.

## Implementation phases (for the plan)

1. Spike: extract Sortformer posteriors on a 90 s clip; confirm shape + frame
   timing. Decide go / fallback.
2. Hardware probe + `Profile`.
3. Soft diarize, single chunk → enrichment → posteriors-on-words.
4. Chunking + silence-aware cuts + label stitch + seam records + splice.
5. Sentence grouping + Viterbi resolver.
6. Wire into `process_file`; fallbacks; keep dual-track path intact.
7. Validation fixture + unit tests; tune parameters.
