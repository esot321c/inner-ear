"""Run NeMo Sortformer diarization on /work/audio.wav (mono 16k) and report.
Validates speaker separation against Graham's ground-truth timestamps.

Env:
  GT_OFFSET  seconds to add to model-relative times (if audio is a slice).
"""
import os, sys, torch
from collections import defaultdict

AUDIO = "/work/audio.wav"
OFFSET = float(os.environ.get("GT_OFFSET", "0"))

# (absolute second in the ORIGINAL file, who really said it)
GROUND_TRUTH = [
    (193.9, "Shane"), (196.5, "Joe"), (203.1, "Shane"),
    (207.1, "Joe"), (250.5, "Shane"), (254.0, "Joe"),
]

from nemo.collections.asr.models import SortformerEncLabelModel

MODEL = os.environ.get("MODEL_NAME", "nvidia/diar_sortformer_4spk-v1")
print(f">> loading Sortformer: {MODEL} (downloads on first run)...", flush=True)
m = SortformerEncLabelModel.from_pretrained(MODEL)
m.eval()
if torch.cuda.is_available():
    m = m.to("cuda")
    print(">> on CUDA:", torch.cuda.get_device_name(0), flush=True)

print(">> diarizing...", flush=True)
pred = m.diarize(audio=[AUDIO], batch_size=1)

# Be defensive about the return shape - dump it, then parse.
print(">> raw prediction type:", type(pred), flush=True)
segs = pred[0] if isinstance(pred, (list, tuple)) else pred
print(">> first few segment entries:", segs[:5], flush=True)

parsed = []
for s in segs:
    if isinstance(s, str):
        p = s.split()
        if len(p) >= 3:
            parsed.append((float(p[0]), float(p[1]), p[2]))
    elif isinstance(s, (list, tuple)) and len(s) >= 3:
        parsed.append((float(s[0]), float(s[1]), str(s[2])))

if not parsed:
    print(">> could not parse segments - inspect raw output above.")
    sys.exit(0)

dur = defaultdict(float)
for st, en, spk in parsed:
    dur[spk] += en - st
total = sum(dur.values()) or 1
print("\n=== speaker time distribution ===")
for spk, d in sorted(dur.items(), key=lambda x: -x[1]):
    print(f"   {spk}: {d:.1f}s ({100*d/total:.0f}%)  turns: {sum(1 for x in parsed if x[2]==spk)}")

def spk_at(t_abs):
    t = t_abs - OFFSET
    hits = [spk for st, en, spk in parsed if st <= t <= en]
    return hits[0] if hits else "(silence/none)"

print("\n=== ground-truth check (does it FLIP speakers correctly?) ===")
for t, who in GROUND_TRUTH:
    if t - OFFSET < 0:
        continue
    print(f"   t={t:7.1f}s  predicted={spk_at(t):>14}  actual={who}")
