"""Batch pipeline: every audio/video file in /in -> speaker-labeled transcript
in /out, source archived to /archive.

AUTO-DETECTS two-channel meeting recordings:
  - 2 separate audio tracks (e.g. OBS: mic + desktop), OR
  - true stereo where L and R differ (mic left, call-audio right)
  => channel/track 0 = "You" (your mic, no diarization needed),
     channel/track 1 = the far side (diarized with Sortformer, may be many people).
  Both are transcribed independently, so nothing is lost even when you and the
  other side talk at once. Merged into one chronological transcript.

  A normal single mic / dual-mono file (e.g. a downloaded podcast) is diarized
  as one mixed track (the original behavior).

Env: WHISPER_MODEL, DIAR_MODEL, COMPUTE_TYPE, VAD_FILTER, STAMP,
     YOU_CH (0|1, which channel is you; default 0),
     YOU_NAME (default 'You'),
     SPEAKER_NAMES ('speaker_1=Client').
"""
import os, re, json, glob, subprocess, tempfile, shutil, torch
from collections import defaultdict

IN, OUT, ARCHIVE = "/in", "/out", "/archive"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DIAR_MODEL = os.environ.get("DIAR_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
VAD_FILTER = os.environ.get("VAD_FILTER", "0") == "1"
STAMP = os.environ.get("STAMP", "")
YOU_CH = int(os.environ.get("YOU_CH", "0"))
YOU_NAME = os.environ.get("YOU_NAME", "You")
EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
device = "cuda" if torch.cuda.is_available() else "cpu"


def ffprobe_audio_streams(f):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=index,channels", "-of", "csv=p=0", f],
                         capture_output=True, text=True).stdout.strip()
    rows = [r for r in out.splitlines() if r.strip()]
    chans = []
    for r in rows:
        parts = r.split(",")
        chans.append(int(parts[-1]) if parts[-1].isdigit() else 1)
    return chans  # list of channel-counts per audio stream


def lr_diff_db(f):
    out = subprocess.run(["ffmpeg", "-i", f, "-af", "pan=mono|c0=c0-c1,volumedetect",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", out)
    return float(m.group(1)) if m else -99.0


def extract(f, args, dst):
    subprocess.run(["ffmpeg", "-y", "-i", f] + args + ["-ac", "1", "-ar", "16000", dst],
                   check=True, capture_output=True)


def plan(f, tmp):
    """Return list of (role, wav). role: 'mono' | 'you' | 'far'."""
    base = os.path.splitext(os.path.basename(f))[0]
    streams = ffprobe_audio_streams(f)
    you = os.path.join(tmp, base + ".you.wav")
    far = os.path.join(tmp, base + ".far.wav")
    mono = os.path.join(tmp, base + ".mono.wav")
    if len(streams) >= 2:                       # multi-track (e.g. OBS)
        extract(f, ["-map", f"0:a:{YOU_CH}"], you)
        extract(f, ["-map", f"0:a:{1 - YOU_CH}"], far)
        return "dual (separate tracks)", [("you", you), ("far", far)]
    if streams and streams[0] >= 2 and lr_diff_db(f) > -40:   # true stereo
        extract(f, ["-af", f"pan=mono|c0=c{YOU_CH}"], you)
        extract(f, ["-af", f"pan=mono|c0=c{1 - YOU_CH}"], far)
        return "dual (stereo L/R)", [("you", you), ("far", far)]
    extract(f, [], mono)                         # mono or dual-mono mix
    return "single mixed track", [("mono", mono)]


files = [f for f in sorted(glob.glob(os.path.join(IN, "*")))
         if os.path.isfile(f) and f.lower().endswith(EXTS)]
if not files:
    print(">> No audio/video files in ./in - drop files there and run again.")
    raise SystemExit(0)
print(f">> {len(files)} file(s): {[os.path.basename(f) for f in files]}", flush=True)

tmp = tempfile.mkdtemp()
jobs_by = {}   # file -> (mode_label, [(role, wav), ...])
for f in files:
    try:
        jobs_by[f] = plan(f, tmp)
        print(f">> {os.path.basename(f)}: {jobs_by[f][0]}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"!! ffmpeg failed on {os.path.basename(f)}: {e.stderr.decode()[-300:]}")
files = [f for f in files if f in jobs_by]

# ---- PASS 1: transcribe every wav (one Whisper load) ----
from faster_whisper import WhisperModel
print(f">> loading Whisper {WHISPER_MODEL} ...", flush=True)
wm = WhisperModel(WHISPER_MODEL, device=device,
                  compute_type=COMPUTE_TYPE if device == "cuda" else "int8")
words_by = {}   # wav -> [words]
for f in files:
    for role, wav in jobs_by[f][1]:
        segs, info = wm.transcribe(wav, word_timestamps=True, vad_filter=VAD_FILTER, beam_size=5)
        words_by[wav] = [{"start": w.start, "end": w.end, "word": w.word}
                         for s in segs for w in (s.words or [])]
        print(f"   {os.path.basename(f)} [{role}]: {len(words_by[wav])} words", flush=True)
del wm
if device == "cuda":
    torch.cuda.empty_cache()

# ---- PASS 2: diarize only the wavs that need it ('far' / 'mono') ----
from nemo.collections.asr.models import SortformerEncLabelModel
print(f">> loading diarizer {DIAR_MODEL} ...", flush=True)
dm = SortformerEncLabelModel.from_pretrained(DIAR_MODEL)
dm.eval()
if device == "cuda":
    dm = dm.to("cuda")
segs_by = {}   # wav -> [(start,end,spk)]
for f in files:
    for role, wav in jobs_by[f][1]:
        if role == "you":
            continue
        pred = dm.diarize(audio=[wav], batch_size=1)
        ds = []
        for s in pred[0]:
            p = s.split()
            if len(p) >= 3:
                ds.append((float(p[0]), float(p[1]), p[2]))
        segs_by[wav] = ds
del dm
if device == "cuda":
    torch.cuda.empty_cache()

# ---- PASS 3: assign, smooth, merge, write, archive ----
names = {}
for pair in os.environ.get("SPEAKER_NAMES", "").split(","):
    if "=" in pair:
        k, v = pair.split("=", 1)
        names[k.strip()] = v.strip()
disp = lambda spk: names.get(spk, YOU_NAME if spk == "__you__"
                             else spk.replace("speaker_", "Speaker "))
ts = lambda s: f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{s%60:05.2f}"


def assign_and_smooth(words, dsegs):
    def spk_at(t):
        for st, en, sp in dsegs:
            if st <= t <= en:
                return sp
        return min(dsegs, key=lambda x: min(abs(x[0]-t), abs(x[1]-t)))[2] if dsegs else "speaker_0"
    for w in words:
        w["speaker"] = spk_at((w["start"] + w["end"]) / 2)
    # sentence-level majority smoothing
    sent, cur = [], []
    for w in words:
        cur.append(w)
        t = w["word"].strip()
        if t and t[-1] in ".?!":
            sent.append(cur); cur = []
    if cur:
        sent.append(cur)
    for s in sent:
        dur = defaultdict(float)
        for w in s:
            dur[w["speaker"]] += w["end"] - w["start"]
        dom = max(dur, key=dur.get)
        for w in s:
            w["speaker"] = dom


for f in files:
    mode_label, jobs = jobs_by[f]
    all_words = []
    for role, wav in jobs:
        ws = words_by[wav]
        if role == "you":
            for w in ws:
                w["speaker"] = "__you__"
        else:
            assign_and_smooth(ws, segs_by.get(wav, []))
        all_words.extend(ws)
    all_words.sort(key=lambda w: w["start"])

    # group into turns
    turns, c = [], None
    for w in all_words:
        if c and c["speaker"] == w["speaker"]:
            c["text"] += w["word"]; c["end"] = w["end"]
        else:
            if c:
                turns.append(c)
            c = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": w["word"]}
    if c:
        turns.append(c)

    base = os.path.splitext(os.path.basename(f))[0]
    od = os.path.join(OUT, base)
    os.makedirs(od, exist_ok=True)
    lines = [f"[{ts(t['start'])}] {disp(t['speaker'])}: {t['text'].strip()}" for t in turns]
    with open(os.path.join(od, base + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# mode: {mode_label}\n\n" + "\n\n".join(lines))
    with open(os.path.join(od, base + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"mode": mode_label, "turns": turns}, fh, indent=2)
    print(f">> wrote out/{base}/{base}.txt  ({len(turns)} turns, {mode_label})", flush=True)

    if STAMP and os.path.isdir(ARCHIVE):
        dest = os.path.join(ARCHIVE, STAMP)
        os.makedirs(dest, exist_ok=True)
        target = os.path.join(dest, os.path.basename(f))
        if not os.path.exists(target):
            shutil.move(f, target)
            print(f">> archived source -> archive/{STAMP}/{os.path.basename(f)}", flush=True)

shutil.rmtree(tmp, ignore_errors=True)
print(">> all done")
