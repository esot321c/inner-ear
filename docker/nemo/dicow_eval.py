"""Evaluate DiCoW (diarization-conditioned, fused ASR+attribution) against the
Sortformer cascade on every labeled file, scored on the same ground truth.

DiCoW is conditioned on Sortformer's per-frame posterior (the diarization mask),
so it shares Sortformer's notion of WHO, but decodes each speaker's words
independently — the test is whether that recovers overlap/boundary words the
cascade mis-attributes.

Run: docker exec inner-ear python3 /work/dicow_eval.py
"""
import os
import glob
import json
import logging
import warnings
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
import numpy as np
from faster_whisper.audio import decode_audio
import scoring
import dicow_backend as D

BENCH = "/out/bench"


def labeled():
    out = []
    for cdir in sorted(glob.glob(os.path.join(BENCH, "*"))):
        if all(os.path.exists(os.path.join(cdir, f)) for f in
               ("_truth.json", "_post.npy")) and glob.glob(os.path.join(cdir, "_wav", "*.wav")):
            out.append(cdir)
    return out


def main():
    print(f"{'file':<24} {'mist':>5} {'words':>6} {'cover':>6} {'backch':>7}")
    print("-" * 52)
    for cdir in labeled():
        name = os.path.basename(cdir)[:22]
        truth = json.load(open(os.path.join(cdir, "_truth.json"), encoding="utf-8"))
        post = np.load(os.path.join(cdir, "_post.npy"))
        wav = glob.glob(os.path.join(cdir, "_wav", "*.wav"))[0]
        audio = decode_audio(wav)
        # only score the labeled span to bound runtime
        end_s = float(truth[-1].get("end", 0)) + 5
        audio = audio[:int(end_s * 16000)]
        post = post[:int(end_s / 0.08)]
        words = D.transcribe_dicow(audio, post)
        # cache DiCoW output for inspection
        json.dump(words, open(os.path.join(cdir, "_pred_dicow.json"), "w", encoding="utf-8"))
        s = scoring.score(words, truth)
        bc = "n/a" if s["backchannel_acc"] is None else f"{s['backchannel_acc']*100:.0f}%"
        print(f"{name:<24} {s['mistakes']:>5} {s['aligned']-s['correct']:>6} "
              f"{s['coverage']*100:>5.0f}% {bc:>7}", flush=True)


if __name__ == "__main__":
    main()
