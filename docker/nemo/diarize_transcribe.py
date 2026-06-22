"""Batch CLI: every audio/video file in /in -> speaker-labeled transcript in
/out, source archived to /archive. All logic lives in pipeline.py.

Env: WHISPER_MODEL, DIAR_MODEL, COMPUTE_TYPE, VAD_FILTER, STAMP,
     YOU_CH (0|1), YOU_NAME, SPEAKER_NAMES ('speaker_1=Client,...')."""
import os
import sys
import glob
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P

IN, OUT, ARCHIVE = "/in", "/out", "/archive"
EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
STAMP = os.environ.get("STAMP", "")
YOU_CH = int(os.environ.get("YOU_CH", "0"))
YOU_NAME = os.environ.get("YOU_NAME", "You")
VAD = os.environ.get("VAD_FILTER", "0") == "1"

names = {}
for pair in os.environ.get("SPEAKER_NAMES", "").split(","):
    if "=" in pair:
        k, v = pair.split("=", 1)
        names[k.strip()] = v.strip()

files = [f for f in sorted(glob.glob(os.path.join(IN, "*")))
         if os.path.isfile(f) and f.lower().endswith(EXTS)]
if not files:
    print(">> No audio/video in ./in - drop files there and run again.")
    raise SystemExit(0)
print(f">> {len(files)} file(s): {[os.path.basename(f) for f in files]}", flush=True)

tmp = tempfile.mkdtemp()
plans = {}
for f in files:
    try:
        plans[f] = P.plan_channels(f, tmp, YOU_CH)
        print(f">> {os.path.basename(f)}: {plans[f][0]}", flush=True)
    except Exception as e:
        print(f"!! skipping {os.path.basename(f)}: {e}")
files = [f for f in files if f in plans]

# PASS 1: transcribe everything with one Whisper load
import torch
wm = P.load_whisper(os.environ.get("WHISPER_MODEL", "large-v3"),
                    os.environ.get("COMPUTE_TYPE", "float16"))
words = {}
for f in files:
    for role, wav in plans[f][1]:
        words[wav] = P.transcribe(wm, wav, VAD)
        print(f"   {os.path.basename(f)} [{role}]: {len(words[wav])} words", flush=True)
del wm
torch.cuda.empty_cache()

# PASS 2: diarize the channels that need it (one Sortformer load)
dm = P.load_diarizer(os.environ.get("DIAR_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2"))
dsegs = {}
for f in files:
    for role, wav in plans[f][1]:
        if role != "you":
            dsegs[wav] = P.diarize(dm, wav)
del dm
torch.cuda.empty_cache()

# PASS 3: assign + smooth + merge + write + archive
for f in files:
    mode_label, jobs = plans[f]
    allw = []
    for role, wav in jobs:
        ws = words[wav]
        if role == "you":
            for w in ws:
                w["speaker"] = "__you__"
        else:
            P.assign_speakers(ws, dsegs.get(wav, []))
            P.smooth_sentences(ws)
        allw.extend(ws)
    allw.sort(key=lambda w: w["start"])
    turns = P.build_turns(allw)
    base = os.path.splitext(os.path.basename(f))[0]
    od = os.path.join(OUT, base)
    os.makedirs(od, exist_ok=True)
    txt, data = P.render(turns, names, YOU_NAME, mode_label)
    with open(os.path.join(od, base + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(od, base + ".json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f">> wrote out/{base}/{base}.txt ({len(turns)} turns, {mode_label})", flush=True)
    if STAMP and os.path.isdir(ARCHIVE):
        dest = os.path.join(ARCHIVE, STAMP)
        os.makedirs(dest, exist_ok=True)
        tgt = os.path.join(dest, os.path.basename(f))
        if not os.path.exists(tgt):
            shutil.move(f, tgt)
            print(f">> archived -> archive/{STAMP}/{os.path.basename(f)}", flush=True)

shutil.rmtree(tmp, ignore_errors=True)
print(">> all done")
