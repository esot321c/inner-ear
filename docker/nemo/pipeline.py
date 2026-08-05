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


def build_turns_overlap(words, max_gap=1.2):
    """Turn builder for pooled/overlapping multi-speaker output. Strict
    time-ordered grouping shatters simultaneous speech into one-word fragments
    because two speakers' words interleave; instead we group each speaker's OWN
    consecutive words into an utterance. But we do NOT merge across another
    speaker's interjection: if a different speaker has a word starting between
    this speaker's previous word and the current one, we start a new utterance
    even when the within-speaker gap is small. That keeps genuine overlap whole
    while preserving natural back-and-forth (A / B interjects / A resumes ->
    three turns, not one run-on A turn wrapping B)."""
    from collections import defaultdict
    # starts of every word, for detecting an interjection by a different speaker
    starts = sorted(((w["start"], w["speaker"]) for w in words), key=lambda x: x[0])

    def other_speaker_between(spk, t0, t1):
        # is there a word by a speaker != spk with start in (t0, t1)?
        import bisect
        lo = bisect.bisect_right(starts, (t0, chr(0x10FFFF)))
        for i in range(lo, len(starts)):
            s, sp = starts[i]
            if s >= t1:
                break
            if sp != spk:
                return True
        return False

    by_spk = defaultdict(list)
    for w in words:
        by_spk[w["speaker"]].append(w)
    turns = []
    for spk, ws in by_spk.items():
        ws.sort(key=lambda x: x["start"])
        cur = None
        for w in ws:
            if (cur and w["start"] - cur["end"] <= max_gap
                    and not other_speaker_between(spk, cur["end"], w["start"])):
                cur["text"] += w["word"]
                cur["end"] = w["end"]
            else:
                if cur:
                    turns.append(cur)
                cur = {"speaker": spk, "start": w["start"], "end": w["end"], "text": w["word"]}
        if cur:
            turns.append(cur)
    turns.sort(key=lambda t: t["start"])
    return turns


def namespace_speakers(words, track_index):
    """Rewrite each word's speaker id to a globally-unique, track-scoped id
    't{track_index}:{raw}', so speaker_0 from track 0 and track 1 don't collide
    once words from every track are pooled. Idempotent."""
    prefix = f"t{track_index}:"
    for w in words:
        spk = w.get("speaker")
        if spk is None or str(spk).startswith(prefix):
            continue
        w["speaker"] = prefix + str(spk)


def _norm_text(s):
    return "".join(c for c in s.lower() if c.isalnum())


def suppress_cross_track_bleed(tracks, margin_db=6.0, overlap_tol=0.3, cluster_frac=0.6, _decode=None):
    """Drop words that are a fainter duplicate of the same utterance heard louder
    on another track (cross-track bleed). A word is dropped ONLY if another track
    is (a) time-overlapping, (b) louder by >= margin_db in that window, and (c)
    carrying the same normalized text. Genuine simultaneous speech (loud on both,
    different text) is preserved. Mutates each track's 'words' list in place."""
    import numpy as np
    if len(tracks) < 2:
        return
    if _decode is None:
        from faster_whisper.audio import decode_audio as _decode
    SR = 16000

    # Per-track prefix sum of squared samples -> O(1) window RMS. None if no audio.
    tables = []
    for tr in tracks:
        audio = tr.get("audio")
        if audio is None:
            try:
                audio = _decode(tr["wav"])
            except Exception:
                audio = None
        if audio is None:
            tables.append(None)
            continue
        a = np.asarray(audio, dtype="float64")
        tables.append((np.concatenate([[0.0], np.cumsum(a * a)]), len(a)))

    def rms_db(ti, s, e):
        tab = tables[ti]
        if tab is None:
            return None
        csum, n = tab
        i0 = max(0, min(int(s * SR), n))
        i1 = min(max(i0 + 1, int(e * SR)), n)
        if i1 <= i0:                      # window past end of this track's audio -> no evidence
            return None
        mean_sq = (csum[i1] - csum[i0]) / (i1 - i0)
        return 10.0 * np.log10(mean_sq + 1e-10)

    def overlaps(w, x):
        return w["start"] < x["end"] + overlap_tol and w["end"] > x["start"] - overlap_tol

    for ai, A in enumerate(tracks):
        if tables[ai] is None:
            continue  # can't measure A's own loudness -> never drop A's words
        keep = []
        for w in A["words"]:
            wt = _norm_text(w["word"])
            dbA = rms_db(ai, w["start"], w["end"])
            bleed = False
            if wt and dbA is not None:
                for bi, B in enumerate(tracks):
                    if bi == ai or tables[bi] is None:
                        continue
                    dbB = rms_db(bi, w["start"], w["end"])
                    if dbB is None or dbB - dbA < margin_db:
                        continue
                    if any(overlaps(w, x) and _norm_text(x["word"]) == wt for x in B["words"]):
                        bleed = True
                        break
            if not bleed:
                keep.append(w)
        A["words"][:] = keep

    # Stage 2 — cluster pass (energy-only). Faint bleed sometimes mis-transcribes
    # to a different token, so the text-gated stage 1 leaves a residue. Any
    # (track, speaker) cluster whose SURVIVING words are mostly louder on another
    # track is bleed as a whole -> drop it. Energy-only, but aggregated over the
    # cluster, so a lone real overlapping word can't trip it and a real speaker on
    # their own track (~0% louder-elsewhere) is never flagged.
    from collections import defaultdict
    for ai, A in enumerate(tracks):
        if tables[ai] is None:
            continue
        groups = defaultdict(list)
        for w in A["words"]:
            spk = w.get("speaker")
            if spk is not None:
                groups[spk].append(w)
        drop_ids = set()
        for spk, ws in groups.items():
            dominated = measured = 0
            for w in ws:
                dbA = rms_db(ai, w["start"], w["end"])
                if dbA is None:
                    continue
                others = [rms_db(bi, w["start"], w["end"])
                          for bi in range(len(tracks)) if bi != ai and tables[bi] is not None]
                others = [d for d in others if d is not None]
                if not others:
                    continue
                measured += 1
                if max(others) - dbA >= margin_db:
                    dominated += 1
            # require a non-trivial cluster: dropping a whole speaker on 1-2 words
            # is too thin. A real cluster of bleed has many energy-dominated words.
            if measured >= 3 and dominated / measured >= cluster_frac:
                drop_ids.add(spk)
        if drop_ids:
            A["words"][:] = [w for w in A["words"] if w.get("speaker") not in drop_ids]


def display_name(spk, names):
    if spk in names:
        return names[spk]
    raw = spk.split(":", 1)[1] if ":" in spk else spk
    return raw.replace("speaker_", "Speaker ")


def _ts(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{s%60:05.2f}"


def render(turns, names, mode_label=""):
    lines = [f"[{_ts(t['start'])}] {display_name(t['speaker'], names)}: {t['text'].strip()}"
             for t in turns]
    header = f"# mode: {mode_label}\n\n" if mode_label else ""
    txt = header + "\n\n".join(lines)
    data = {"mode": mode_label, "turns": turns,
            "speakers": sorted({t["speaker"] for t in turns})}
    return txt, data


def pick_sample_turns(turns):
    """Longest turn per speaker, used for the play-clip button."""
    best = {}
    for t in turns:
        dur = t["end"] - t["start"]
        if t["speaker"] not in best or dur > (best[t["speaker"]]["end"] - best[t["speaker"]]["start"]):
            best[t["speaker"]] = t
    return best


def pick_solo_windows(post, frame_dur, spk_cols, clip_secs=12.0):
    """For each speaker column, find the clip_secs-long window where that speaker
    is loudest while every other speaker is quietest - a clean solo sample for
    the naming UI. Score of a window = mean(target) - mean(max over others).
    Returns {col: (start_sec, end_sec)}. Always returns a window per requested
    col (best-available even if no true solo exists)."""
    import numpy as np
    post = np.asarray(post, dtype=float)
    T, ncol = post.shape
    win = max(1, int(round(clip_secs / frame_dur)))
    win = min(win, T)
    out = {}
    for c in spk_cols:
        if c >= ncol:
            out[c] = (0.0, min(clip_secs, T * frame_dur))
            continue
        others = [j for j in range(ncol) if j != c]
        target = post[:, c]
        other_max = post[:, others].max(axis=1) if others else np.zeros(T)
        score = target - other_max
        # sliding-window sum via cumulative sum
        csum = np.concatenate([[0.0], np.cumsum(score)])
        best_i, best_v = 0, -1e18
        for i in range(0, T - win + 1):
            v = csum[i + win] - csum[i]
            if v > best_v:
                best_v, best_i = v, i
        s = best_i * frame_dur
        e = (best_i + win) * frame_dur
        out[c] = (s, e)
    return out


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


def plan_channels(f, tmp):
    """Returns (mode_label, [(role, wav)]). Every track gets a neutral,
    index-based role ('track0', 'track1', ...) or 'mono' for a single mixed
    track. No track is privileged as 'you' - each is diarized uniformly and the
    user names the speakers found across all of them."""
    base = os.path.splitext(os.path.basename(f))[0]
    streams = ffprobe_audio_streams(f)
    diff = lr_diff_db(f) if streams[:1] == [2] else -99.0
    mode = decide_mode(streams, diff)
    if mode == "dual_track":
        jobs = []
        for i in range(len(streams)):
            wav = os.path.join(tmp, f"{base}.track{i}.wav")
            _extract(f, ["-map", f"0:a:{i}"], wav)
            jobs.append((f"track{i}", wav))
        return "dual (separate tracks)", jobs
    if mode == "dual_stereo":
        jobs = []
        for i in range(2):
            wav = os.path.join(tmp, f"{base}.track{i}.wav")
            _extract(f, ["-af", f"pan=mono|c0=c{i}"], wav)
            jobs.append((f"track{i}", wav))
        return "dual (stereo L/R)", jobs
    mono = os.path.join(tmp, base + ".mono.wav")
    _extract(f, [], mono)
    return "single mixed track", [("mono", mono)]


def load_whisper(model_name="large-v3", compute_type="float16"):
    import torch
    from faster_whisper import WhisperModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel(model_name, device=dev,
                        compute_type=compute_type if dev == "cuda" else "int8")


def transcribe(wm, wav, vad_filter=True):
    """Transcribe each VAD speech chunk as a SEPARATE Whisper call.

    Why not one call with vad_filter=True: faster-whisper's word timestamps come
    from cross-attention over the whole decoded window, and it refuses to put a
    gap *before* a word - so a word spoken after a long pause (your "did" after a
    2.5s silence) gets glued onto the previous speaker's phrase, then handed to
    the diarizer in the wrong voice. Slicing the audio at the silences and
    decoding each chunk independently makes that bridge physically impossible:
    "did" starts its own chunk, so it lands in your voice. It also lets Whisper
    punctuate each chunk natively (audio-informed, better placed than a post-hoc
    text model). condition_on_previous_text stays off to avoid repeat loops.
    """
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps
    audio = decode_audio(wav)
    sr = 16000
    chunks = get_speech_timestamps(audio) if vad_filter else [{"start": 0, "end": len(audio)}]
    words = []
    for ch in chunks:
        off = ch["start"] / sr
        segs, _info = wm.transcribe(audio[ch["start"]:ch["end"]], word_timestamps=True,
                                    vad_filter=False, condition_on_previous_text=False,
                                    beam_size=5)
        for s in segs:
            for w in (s.words or []):
                words.append({"start": w.start + off, "end": w.end + off, "word": w.word})
    return words


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


def count_speakers(post, thr=0.5, min_secs=2.0, frame_dur=FRAME_DUR):
    """How many speakers are genuinely present: columns active for at least
    min_secs total. Used to pick the pipeline (cascade for <=2, DiCoW for >=3).
    Ignores phantom speakers the diarizer barely lit up."""
    import numpy as np
    on = (np.asarray(post) >= thr).sum(axis=0) * frame_dur
    return int((on >= min_secs).sum())


def words_speaker_probs(words, post, frame_dur, n_spk=4, normalize=True):
    """Attach to each word a per-speaker probability vector, averaged over the
    posterior frames spanning the word. This is the Viterbi emission.

    Sortformer's posteriors are INDEPENDENT per-speaker sigmoids, so two speakers
    can both read ~0.9 (= genuine overlap). normalize=True divides by the sum
    (legacy behaviour) which collapses 0.9/0.9 -> 0.5/0.5 and discards the overlap
    signal; normalize=False keeps the raw activations so overlap survives."""
    T = post.shape[0]
    for w in words:
        a = max(0, min(int(w["start"] / frame_dur), T - 1))
        b = max(a + 1, min(int(round(w["end"] / frame_dur)), T))
        m = post[a:b].mean(axis=0)
        if normalize:
            tot = float(m.sum()) or 1.0
            w["speaker_probs"] = [float(m[s] / tot) if s < len(m) else 0.0
                                  for s in range(n_spk)]
        else:
            w["speaker_probs"] = [float(m[s]) if s < len(m) else 0.0
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


def fix_transition_leaks(words, post, frame_dur, n_spk=4, edge=0.12, conf=0.6):
    """Re-attribute boundary-straddling words using the soft posterior's shape
    WITHIN each word, not just its average.

    Rapid back-channels ('yeah', 'okay') get a too-wide Whisper timestamp that
    starts in the previous speaker's tail/silence and ends on the actual spoken
    token at the new speaker's onset. The averaged emission used by the resolver
    is then dominated by the wrong (earlier, longer) region. Here we look at the
    speaker dominant in the word's first `edge` seconds vs its last `edge`
    seconds: if they differ and the END region confidently belongs to one
    speaker, the spoken token is theirs, so we move the word to that end speaker.

    Returns the number of words moved. Conservative by design (requires a clear
    start!=end split and a confident end region) so clean words are untouched."""
    T = post.shape[0]
    ef = max(1, int(round(edge / frame_dur)))
    moved = 0
    for w in words:
        a = max(0, min(int(w["start"] / frame_dur), T - 1))
        b = max(a + 1, min(int(round(w["end"] / frame_dur)), T))
        if b - a < 2 * ef:
            continue                          # word too short to have two regions
        s_reg = post[a:a + ef].mean(axis=0)
        e_reg = post[b - ef:b].mean(axis=0)
        ss, es = int(s_reg.argmax()), int(e_reg.argmax())
        if ss == es or es >= n_spk:
            continue                          # no transition across the word
        etot = float(e_reg.sum()) or 1.0
        if e_reg[es] / etot < conf:
            continue                          # end region not confidently one speaker
        tgt = f"speaker_{es}"
        if w.get("speaker") != tgt:
            w["speaker"] = tgt
            moved += 1
    return moved


# --------------------------- punctuation + truecasing ---------------------------

def load_punctuator():
    from deepmultilingualpunctuation import PunctuationModel
    return PunctuationModel()


_PUNCT_LABELS = {".": ".", ",": ",", "?": "?", ":": ":"}


def restore_punctuation(words, pm):
    """Add sentence punctuation and capitalisation to a flat word list. Whisper
    leaves fast/overlapping speech as an unpunctuated lowercase run; this makes
    it readable AND gives group_sentences real sentence boundaries. Each word
    keeps its leading space; punctuation is appended, then sentence starts and
    'I' are capitalised."""
    if not words:
        return
    toks = [w["word"].strip() for w in words]
    labels = pm.predict(toks)
    for w, lab in zip(words, labels):
        mark = _PUNCT_LABELS.get(lab[1] if len(lab) > 1 else "0", "")
        lead = " " if w["word"][:1] == " " else ""
        tok = w["word"].strip()
        if mark and tok[-1:] not in ".,?!:":
            tok += mark
        w["word"] = lead + tok
    _truecase(words)


def _truecase(words):
    cap_next = True
    for w in words:
        lead = " " if w["word"][:1] == " " else ""
        tok = w["word"].strip()
        if not tok:
            continue
        if cap_next and tok[0].isalpha():
            tok = tok[0].upper() + tok[1:]
        low = tok.lower()
        if low == "i" or low[:2] == "i'":          # standalone I, I'm, I'll, ...
            tok = "I" + tok[1:]
        w["word"] = lead + tok
        cap_next = tok[-1:] in ".?!"


# ------------------------------- optional LLM pass -------------------------------

_LLM_BOUNDARY_SYSTEM = ("You judge automatic-transcription speaker boundaries. "
                        "Reply with ONLY one word: YES or NO.")


def _llm_chat(base_url, model, system, user, api_key, timeout):
    import json
    import urllib.request
    body = json.dumps({
        "model": model, "temperature": 0,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=body, headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]["content"]


def llm_cleanup(words, base_url, model, api_key=None, context=8, timeout=60):
    """Optional LLM reattribution pass. Diarization word timestamps are imprecise
    at turn boundaries, so the first word of a new speaker's sentence sometimes
    sticks to the end of the previous turn. For each boundary we ask the model ONE
    yes/no question - does this turn's last word actually begin the next speaker's
    sentence? - and move it only on YES. The model never reproduces the transcript,
    so it cannot rewrite or hallucinate; it only votes on single words.

    Hard guard against the model's over-eagerness: a word ending in '. ? !' ends a
    sentence and therefore cannot begin the next one, so it is never moved (this is
    what blocks the false positives a small model produces). Single-word turns are
    left alone (real interjections). Mutates `words`; returns the number of moves."""
    turns = []
    for i, w in enumerate(words):
        if turns and turns[-1][0] == w["speaker"]:
            turns[-1][1].append(i)
        else:
            turns.append((w["speaker"], [i]))
    moved = 0
    for n in range(len(turns) - 1):
        spk_a, wi_a = turns[n]
        spk_b, wi_b = turns[n + 1]
        if spk_a == spk_b or len(wi_a) < 2:
            continue
        last = words[wi_a[-1]]["word"].strip()
        if not last or last[-1] in ".?!":
            continue
        prev = "".join(words[k]["word"] for k in wi_a[-context:]).strip()
        nxt = "".join(words[k]["word"] for k in wi_b[:context]).strip()
        q = (f'One speaker said: "...{prev}"\nThen a different speaker said: '
             f'"{nxt}..."\n\nDoes the final word "{last}" actually begin the SECOND '
             f"speaker's sentence (mis-attached to the first)? Answer YES or NO.")
        try:
            ans = _llm_chat(base_url, model, _LLM_BOUNDARY_SYSTEM, q, api_key, timeout)
        except Exception as e:
            print("llm boundary skipped:", e)
            continue
        if ans.strip().upper().startswith("Y"):
            words[wi_a[-1]]["speaker"] = spk_b
            moved += 1
    return moved


def extract_clip(src_wav, start, end, dst, pad=0.0):
    _extract(src_wav, ["-ss", str(max(0, start - pad)), "-to", str(end + pad)], dst)
    return dst
