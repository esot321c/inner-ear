"""Shared transcription + diarization engine.

Pure helpers (decide_mode, assign_speakers, smooth_sentences, build_turns,
render, pick_sample_turns, display_name) are stdlib-only and import without a
GPU - so they're unit-tested in the native venv. Model helpers lazy-import
torch / faster_whisper / nemo inside the function bodies.
"""
import os
import re
import json
import subprocess
import tempfile
import shutil
from collections import defaultdict

DUAL_THRESHOLD_DB = -40  # L-R louder than this => true stereo, not dual-mono


# ----------------------------- pure helpers -----------------------------

def decide_mode(streams, lr_diff_db):
    """streams = list of channel-counts per audio stream.
    Returns 'dual_track' | 'dual_stereo' | 'single'."""
    if len(streams) >= 2:
        return "dual_track"
    if streams and streams[0] >= 2 and lr_diff_db > DUAL_THRESHOLD_DB:
        return "dual_stereo"
    return "single"


def assign_speakers(words, dsegs):
    """Tag each word with the diarization speaker active at its midpoint."""
    def spk_at(t):
        for st, en, sp in dsegs:
            if st <= t <= en:
                return sp
        return min(dsegs, key=lambda x: min(abs(x[0] - t), abs(x[1] - t)))[2] if dsegs else "speaker_0"
    for w in words:
        w["speaker"] = spk_at((w["start"] + w["end"]) / 2)


def smooth_sentences(words, max_blip_dur=0.4, max_blip_gap=0.4):
    """Remove isolated single-word speaker 'blips' - a lone, short word tagged as
    a different speaker than BOTH its neighbours, with no real pause around it.
    That pattern is almost always diarizer jitter, not a real one-word turn.

    Crucially this does NOT reassign whole sentences to a 'dominant' speaker. The
    diarizer is good; in fast, interrupt-heavy conversation each ~few-second span
    genuinely contains both people, and forcing the span to one speaker flattens
    real back-and-forth (five turns become one). Trust the per-word diarization;
    only clean obvious sub-word-scale noise."""
    orig = [w["speaker"] for w in words]
    for i in range(1, len(words) - 1):
        if orig[i - 1] == orig[i + 1] != orig[i] \
                and (words[i]["end"] - words[i]["start"]) <= max_blip_dur \
                and (words[i]["start"] - words[i - 1]["end"]) <= max_blip_gap \
                and (words[i + 1]["start"] - words[i]["end"]) <= max_blip_gap:
            words[i]["speaker"] = orig[i - 1]


def build_turns(words):
    """Group consecutive same-speaker words into turns."""
    turns, c = [], None
    for w in words:
        if c and c["speaker"] == w["speaker"]:
            c["text"] += w["word"]; c["end"] = w["end"]
        else:
            if c:
                turns.append(c)
            c = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": w["word"]}
    if c:
        turns.append(c)
    return turns


def display_name(spk, names, you_name="You"):
    if spk == "__you__":
        return names.get("__you__", you_name)
    return names.get(spk, spk.replace("speaker_", "Speaker "))


def _ts(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{s%60:05.2f}"


def render(turns, names, you_name="You", mode_label=""):
    lines = [f"[{_ts(t['start'])}] {display_name(t['speaker'], names, you_name)}: {t['text'].strip()}"
             for t in turns]
    header = f"# mode: {mode_label}\n\n" if mode_label else ""
    txt = header + "\n\n".join(lines)
    data = {"mode": mode_label, "turns": turns,
            "speakers": sorted({t["speaker"] for t in turns})}
    return txt, data


def pick_sample_turns(turns):
    """Longest turn per (non-you) speaker, used for the play-clip button."""
    best = {}
    for t in turns:
        if t["speaker"] == "__you__":
            continue
        dur = t["end"] - t["start"]
        if t["speaker"] not in best or dur > (best[t["speaker"]]["end"] - best[t["speaker"]]["start"]):
            best[t["speaker"]] = t
    return best


# --------------------------- model / ffmpeg helpers ---------------------------

def ffprobe_audio_streams(f):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=channels", "-of", "csv=p=0", f],
                         capture_output=True, text=True).stdout.strip()
    return [int(r) if r.strip().isdigit() else 1 for r in out.splitlines() if r.strip()]


def lr_diff_db(f):
    out = subprocess.run(["ffmpeg", "-i", f, "-af", "pan=mono|c0=c0-c1,volumedetect",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", out)
    return float(m.group(1)) if m else -99.0


def _extract(f, args, dst):
    subprocess.run(["ffmpeg", "-y", "-i", f] + args + ["-ac", "1", "-ar", "16000", dst],
                   check=True, capture_output=True)


def plan_channels(f, tmp, you_ch=0):
    """Returns (mode_label, [(role, wav)]). role in {'mono','you','far'}."""
    base = os.path.splitext(os.path.basename(f))[0]
    streams = ffprobe_audio_streams(f)
    diff = lr_diff_db(f) if streams[:1] == [2] else -99.0
    mode = decide_mode(streams, diff)
    you = os.path.join(tmp, base + ".you.wav")
    far = os.path.join(tmp, base + ".far.wav")
    mono = os.path.join(tmp, base + ".mono.wav")
    if mode == "dual_track":
        _extract(f, ["-map", f"0:a:{you_ch}"], you)
        _extract(f, ["-map", f"0:a:{1 - you_ch}"], far)
        return "dual (separate tracks)", [("you", you), ("far", far)]
    if mode == "dual_stereo":
        _extract(f, ["-af", f"pan=mono|c0=c{you_ch}"], you)
        _extract(f, ["-af", f"pan=mono|c0=c{1 - you_ch}"], far)
        return "dual (stereo L/R)", [("you", you), ("far", far)]
    _extract(f, [], mono)
    return "single mixed track", [("mono", mono)]


def load_whisper(model_name="large-v3", compute_type="float16"):
    import torch
    from faster_whisper import WhisperModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel(model_name, device=dev,
                        compute_type=compute_type if dev == "cuda" else "int8")


def transcribe(wm, wav, vad_filter=True):
    # VAD on by default: skip silence so Whisper doesn't hallucinate ("Thank
    # you." loops) over dead air, which also poisons diarization by inventing a
    # speaker for the silent stretch. condition_on_previous_text off stops a
    # stray hallucination from snowballing into a repeat loop.
    segs, info = wm.transcribe(wav, word_timestamps=True, vad_filter=vad_filter,
                               condition_on_previous_text=False, beam_size=5)
    return [{"start": w.start, "end": w.end, "word": w.word}
            for s in segs for w in (s.words or [])]


def load_diarizer(model_name="nvidia/diar_streaming_sortformer_4spk-v2"):
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(model_name)
    m.eval()
    if torch.cuda.is_available():
        m = m.to("cuda")
    return m


def diarize(dm, wav):
    pred = dm.diarize(audio=[wav], batch_size=1)
    out = []
    for s in pred[0]:
        p = s.split()
        if len(p) >= 3:
            out.append((float(p[0]), float(p[1]), p[2]))
    return out


FRAME_DUR = 0.08  # Sortformer posterior frame = window_stride(0.01) * subsampling(8)


def diarize_soft(dm, wav):
    """Sortformer's frame-level soft posteriors instead of hard segments.
    Returns (post, frame_dur): post is a [frames, n_spk] array of per-speaker
    probabilities in [0,1] (independent sigmoids - they can both be high during
    overlap). This is the 'ramp' the hard-segment path threw away."""
    import numpy as np
    _segs, tensors = dm.diarize(audio=[wav], batch_size=1, include_tensor_outputs=True)
    arr = tensors[0]
    arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
    return arr, FRAME_DUR


def words_speaker_probs(words, post, frame_dur, n_spk=4):
    """Attach to each word a normalised per-speaker probability vector, averaged
    over the posterior frames spanning the word. This is the Viterbi emission."""
    T = post.shape[0]
    for w in words:
        a = max(0, min(int(w["start"] / frame_dur), T - 1))
        b = max(a + 1, min(int(round(w["end"] / frame_dur)), T))
        m = post[a:b].mean(axis=0)
        tot = float(m.sum()) or 1.0
        w["speaker_probs"] = [float(m[s] / tot) if s < len(m) else 0.0
                              for s in range(n_spk)]


def group_sentences(words, max_gap=0.6):
    """Tag each word with sent_id and sent_start. A sentence ends on .?! OR a
    pause longer than max_gap (Whisper emits long unpunctuated runs, so pauses
    are the reliable boundary). The resolver lets speaker switch cheaply at a
    sentence start and dearly mid-sentence."""
    sid = 0
    for i, w in enumerate(words):
        if i == 0:
            start = True
        else:
            prev = words[i - 1]
            tail = prev["word"].strip()
            start = (bool(tail) and tail[-1] in ".?!") or (w["start"] - prev["end"] > max_gap)
        if start:
            sid += 1
        w["sent_start"] = start
        w["sent_id"] = sid


def resolve_speakers(words, n_spk=4, switch_mid=3.0, switch_boundary=0.4, eps=1e-6):
    """Sentence-aware Viterbi over words. Emission = log soft-posterior per
    speaker; switching speaker costs `switch_mid` within a sentence but only
    `switch_boundary` at a sentence start. Net effect:
      - trailing off (mid-sentence, prior speaker still has signal) -> stays
      - new-speaker onset (sentence start, posterior crossed) -> switches
      - real interjection (posterior flips hard) -> emission beats the penalty
    Sets each word's 'speaker'. Operates only on speaker_probs + sent_start, so
    it is pure-Python and unit-testable without a GPU."""
    import math
    if not words:
        return
    S = list(range(n_spk))

    def emit(w, s):
        p = w.get("speaker_probs") or []
        return math.log((p[s] if s < len(p) else 0.0) + eps)

    dp = [emit(words[0], s) for s in S]
    back = [[0] * n_spk for _ in range(len(words))]
    for i in range(1, len(words)):
        pen = switch_boundary if words[i].get("sent_start") else switch_mid
        ndp = [-math.inf] * n_spk
        for s in S:
            e = emit(words[i], s)
            best, bs = -math.inf, 0
            for ps in S:
                sc = dp[ps] + (0.0 if ps == s else -pen) + e
                if sc > best:
                    best, bs = sc, ps
            ndp[s], back[i][s] = best, bs
        dp = ndp
    last = max(S, key=lambda s: dp[s])
    path = [last] * len(words)
    for i in range(len(words) - 1, 0, -1):
        path[i - 1] = back[i][path[i]]
    for w, s in zip(words, path):
        w["speaker"] = f"speaker_{s}"


# Sentence-opener words: these almost always START an utterance, so when one is
# stranded at the end of a turn (its sound landed in the previous speaker via a
# word-timing quirk) it really belongs to the next, different speaker's turn.
ONSET_WORDS = {"how", "what", "why", "when", "where", "who", "which",
               "did", "do", "does", "is", "are", "was", "were",
               "can", "could", "would", "should", "i", "you"}


def fix_onset_leaks(words):
    """Final correction pass. Runs after resolve_speakers, with every prior
    signal available. Moves a turn-final sentence-opener word onto the following
    turn when that turn is a different speaker - this is the 'did you / how much'
    leak, where the word's text opens the next person's sentence but its audio
    glued to the previous turn. A lone opener that is its own one-word turn (a
    real interjection like 'what?') is left untouched, because its neighbours on
    BOTH sides are the other speaker."""
    n = len(words)
    for i in range(n - 1):
        w, nxt = words[i], words[i + 1]
        if w["speaker"] == nxt["speaker"]:
            continue                      # not a turn boundary
        prev_same = (i == 0) or words[i - 1]["speaker"] == w["speaker"]
        if not prev_same:
            continue                      # w is its own turn (interjection) - keep
        tok = w["word"].strip().lower().strip(".,!?")
        if tok in ONSET_WORDS:
            w["speaker"] = nxt["speaker"]


def extract_clip(src_wav, start, end, dst, pad=0.0):
    _extract(src_wav, ["-ss", str(max(0, start - pad)), "-to", str(end + pad)], dst)
    return dst
