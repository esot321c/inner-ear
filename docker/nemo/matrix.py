"""Sweep the stackable post-processing steps across every labeled file and print
a results matrix (boundary mistakes per file per combination). Uses the bench's
cached Whisper words + Sortformer posterior + saved ground truth, so it's a fast,
pure-CPU sweep with no re-transcription.

Run: docker exec inner-ear python3 /work/matrix.py
"""
import os
import glob
import json
import itertools
import numpy as np
import pipeline as P
import scoring

BENCH = "/out/bench"


def labeled_files():
    out = []
    for cdir in sorted(glob.glob(os.path.join(BENCH, "*"))):
        tf = os.path.join(cdir, "_truth.json")
        wf = os.path.join(cdir, "_words.json")
        pf = os.path.join(cdir, "_post.npy")
        if all(os.path.exists(x) for x in (tf, wf, pf)):
            out.append(cdir)
    return out


def run_combo(words0, post, fd, truth, normalize, tfix, onset):
    w = [dict(x) for x in words0]
    P.words_speaker_probs(w, post, fd, normalize=normalize)
    P.group_sentences(w)
    P.resolve_speakers(w)
    if tfix:
        P.fix_transition_leaks(w, post, fd)
    if onset:
        P.fix_onset_leaks(w)
    pred = [{"word": x["word"], "start": x["start"], "end": x["end"],
             "speaker": x["speaker"]} for x in w]
    return scoring.score(pred, truth)


def main():
    files = labeled_files()
    if not files:
        print("no labeled files")
        return
    short = {f: os.path.basename(f)[:22] for f in files}
    combos = list(itertools.product([True, False], repeat=3))  # normalize, tfix, onset

    # header
    cols = "  ".join(f"{short[f]:>22}" for f in files)
    print(f"{'norm tfix onset':<16} | {cols}")
    print("-" * (16 + 3 + len(files) * 24))

    data = {}
    for normalize, tfix, onset in combos:
        cells = []
        for f in files:
            truth = json.load(open(os.path.join(f, "_truth.json"), encoding="utf-8"))
            words0 = json.load(open(os.path.join(f, "_words.json"), encoding="utf-8"))
            post = np.load(os.path.join(f, "_post.npy"))
            s = run_combo(words0, post, P.FRAME_DUR, truth, normalize, tfix, onset)
            data[(normalize, tfix, onset, f)] = s
            cells.append(f"{s['mistakes']:>3}m {s['aligned']-s['correct']:>3}w")
        tag = f"{'Y' if normalize else 'n'}    {'Y' if tfix else 'n'}    {'Y' if onset else 'n'}   "
        print(f"{tag:<16} | " + "  ".join(f"{c:>22}" for c in cells))

    print("\n(cells = boundary mistakes 'm' / wrong words 'w'; lower is better)")
    # best per file by mistakes
    print("\nbest combo per file (fewest mistakes):")
    for f in files:
        best = min(combos, key=lambda c: data[(c[0], c[1], c[2], f)]["mistakes"])
        s = data[(best[0], best[1], best[2], f)]
        print(f"  {short[f]:>22}: norm={best[0]} tfix={best[1]} onset={best[2]} "
              f"-> {s['mistakes']} mistakes, {s['aligned']-s['correct']} words")


if __name__ == "__main__":
    main()
