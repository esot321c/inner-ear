"""Backend axis of the matrix: Sortformer vs pyannote, each x post-processing,
scored on every labeled file. pyannote posteriors are computed once and cached
to _post_pyannote.npy. Needs HF_TOKEN for the gated pyannote download.

Run: docker exec -e HF_TOKEN=... inner-ear python3 /work/matrix_backends.py
"""
import os
import glob
import json
import numpy as np
import pipeline as P
import scoring

BENCH = "/out/bench"


def labeled():
    out = []
    for cdir in sorted(glob.glob(os.path.join(BENCH, "*"))):
        if all(os.path.exists(os.path.join(cdir, f)) for f in
               ("_truth.json", "_words.json", "_post.npy")) and \
           glob.glob(os.path.join(cdir, "_wav", "*.wav")):
            out.append(cdir)
    return out


def pyannote_post(cdir):
    f = os.path.join(cdir, "_post_pyannote.npy")
    if os.path.exists(f):
        return np.load(f)
    import pyannote_backend as PB
    pipe = PB.load_pyannote()
    wav = glob.glob(os.path.join(cdir, "_wav", "*.wav"))[0]
    post, _ = PB.diarize_soft_pyannote(pipe, wav)
    np.save(f, post)
    return post


def run(words0, post, truth, tfix, onset=True):
    w = [dict(x) for x in words0]
    P.words_speaker_probs(w, post, P.FRAME_DUR)
    P.group_sentences(w)
    P.resolve_speakers(w)
    if tfix:
        P.fix_transition_leaks(w, post, P.FRAME_DUR)
    if onset:
        P.fix_onset_leaks(w)
    pred = [{"word": x["word"], "start": x["start"], "end": x["end"],
             "speaker": x["speaker"]} for x in w]
    return scoring.score(pred, truth)


def main():
    files = labeled()
    print(f"{'file':<24} {'backend':<11} {'tfix':<5} {'mist':>5} {'words':>6} {'backch':>7}")
    print("-" * 64)
    for cdir in files:
        name = os.path.basename(cdir)[:22]
        truth = json.load(open(os.path.join(cdir, "_truth.json"), encoding="utf-8"))
        words0 = json.load(open(os.path.join(cdir, "_words.json"), encoding="utf-8"))
        sf = np.load(os.path.join(cdir, "_post.npy"))
        pa = pyannote_post(cdir)
        for bname, post in [("Sortformer", sf), ("pyannote", pa)]:
            for tfix in (False, True):
                s = run(words0, post, truth, tfix)
                bc = "n/a" if s["backchannel_acc"] is None else f"{s['backchannel_acc']*100:.0f}%"
                print(f"{name:<24} {bname:<11} {str(tfix):<5} {s['mistakes']:>5} "
                      f"{s['aligned']-s['correct']:>6} {bc:>7}")
        print()


if __name__ == "__main__":
    main()
