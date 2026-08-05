"""Per-word speaker-accuracy scorer for diarization output.

Speaker labels are arbitrary (`speaker_0` vs "Graham"), so scoring is
permutation-invariant: we find the label mapping that best agrees with the
ground truth, then report accuracy under it. This is the standard way to score
diarization — without it, a perfect transcript with swapped names scores 0%.

Pred and truth are lists of word dicts with at least {"word", "speaker"} (and
optionally "start"). Truth is normally the predicted word stream with corrected
"speaker" fields, so alignment is 1:1; we still align by sequence (difflib) so an
occasional transcription difference does not derail speaker scoring.
"""
import re
import difflib
import itertools
from collections import defaultdict

# Short discourse/back-channel tokens — the hard sub-population. Scored and
# reported separately so the headline number never hides them.
BACKCHANNEL = {"yeah", "yep", "yup", "ya", "ok", "okay", "mm", "mmhm", "mhm",
               "hmm", "right", "uhhuh", "huh", "no", "nah", "sure", "yes", "yup",
               "oh", "wow", "exactly", "totally", "gotcha"}


def _norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def _align(pred, truth):
    """Return matched (pred_idx, truth_idx) pairs via longest-common-subsequence
    over normalized tokens. Only 'equal' blocks are scored."""
    a = [_norm(w["word"]) for w in pred]
    b = [_norm(w["word"]) for w in truth]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs.extend((i1 + k, j1 + k) for k in range(i2 - i1))
    return pairs


def _best_mapping(pairs, pred, truth):
    """Optimal 1-to-1 map from pred speaker labels to truth labels, maximizing
    word agreement. Brute force over permutations (<=4 speakers => trivial)."""
    cont = defaultdict(lambda: defaultdict(int))
    plabels, tlabels = set(), set()
    for pi, ti in pairs:
        pl, tl = pred[pi]["speaker"], truth[ti]["speaker"]
        cont[pl][tl] += 1
        plabels.add(pl)
        tlabels.add(tl)
    plabels, tlabels = sorted(plabels), sorted(tlabels)
    # pad truth options with None so every pred label gets an assignment slot
    cand = tlabels + [None] * max(0, len(plabels) - len(tlabels))
    best_agree, best_map = -1, {pl: None for pl in plabels}
    for perm in set(itertools.permutations(cand, len(plabels))):
        m = dict(zip(plabels, perm))
        agree = sum(cont[pl][m[pl]] for pl in plabels if m[pl] is not None)
        if agree > best_agree:
            best_agree, best_map = agree, m
    return best_map, best_agree


def score(pred, truth):
    """Score pred speaker labels against truth. Returns a dict with overall
    per-word accuracy, back-channel-subset accuracy, the chosen label mapping,
    and the explicit list of mis-attributed words for debugging."""
    pairs = _align(pred, truth)
    mapping, agree = _best_mapping(pairs, pred, truth)
    total = len(pairs)
    errors, bc_total, bc_ok, err_pos = [], 0, 0, []
    for pos, (pi, ti) in enumerate(pairs):
        pl, tl = pred[pi]["speaker"], truth[ti]["speaker"]
        ok = mapping.get(pl) == tl
        if _norm(pred[pi]["word"]) in BACKCHANNEL:
            bc_total += 1
            bc_ok += int(ok)
        if not ok:
            errors.append({"time": round(float(pred[pi].get("start", 0.0)), 2),
                           "word": pred[pi]["word"].strip(),
                           "pred": pl, "pred_mapped": mapping.get(pl), "truth": tl})
            err_pos.append(pos)
    # Boundary mistakes: adjacent wrong words (gap <= 2) are ONE mistake, not N.
    # This is the metric that matters — a 18-word goodbye crosstalk is 1 bad
    # boundary, not 18 failures diluting a 99% word score.
    mistakes, last = 0, None
    for pos in err_pos:
        if last is None or pos - last > 2:
            mistakes += 1
        last = pos
    # True speaker turns (context for "is N mistakes a lot?").
    tsp = [w["speaker"] for w in truth]
    turns = (1 + sum(1 for i in range(1, len(tsp)) if tsp[i] != tsp[i - 1])) if tsp else 0
    return {
        "accuracy": (agree / total) if total else 0.0,
        "correct": agree,
        "aligned": total,
        "pred_words": len(pred),
        "truth_words": len(truth),
        "coverage": (total / len(truth)) if truth else 0.0,
        "backchannel_acc": (bc_ok / bc_total) if bc_total else None,
        "backchannel_n": bc_total,
        "mistakes": mistakes,
        "turns": turns,
        "mapping": mapping,
        "errors": errors,
    }


def _is_end_marker(line):
    """A line like '---END---', '=== STOP ===' marks where correction stopped;
    everything below it is ignored when building ground truth."""
    a = "".join(c for c in line if c.isalpha()).upper()
    return a in ("END", "STOP", "ENDTRUTH", "ENDHERE")


def parse_turns(text):
    """Parse the bench's editable turn box ('Speaker: words' per line) into a
    flat list of (normalized_token, speaker). A line with no colon continues the
    previous speaker. Stops at an END/STOP marker line (partial labeling)."""
    toks, spks = [], []
    last = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_end_marker(line):
            break
        if ":" in line:
            spk, rest = line.split(":", 1)
            spk = spk.strip()
        else:
            spk, rest = (last or "Speaker"), line
        last = spk
        for t in rest.split():
            n = _norm(t)
            if n:
                toks.append(n)
                spks.append(spk)
    return toks, spks


def build_truth(pred_words, edited_text):
    """Turn the user's corrected turn text into word-level ground truth: the
    predicted word stream with each word's speaker taken from the edited turns
    (aligned by token sequence; gaps filled from neighbours)."""
    t_toks, t_spks = parse_turns(edited_text)
    a = [_norm(w["word"]) for w in pred_words]
    sm = difflib.SequenceMatcher(a=a, b=t_toks, autojunk=False)
    assigned = [None] * len(pred_words)
    last_matched = -1
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                assigned[i1 + k] = t_spks[j1 + k]
            last_matched = max(last_matched, i2 - 1)
    # Truth only covers what the edited text actually reached — drop pred words
    # past the last corrected token (partial labeling / END marker / deleted tail).
    cut = (last_matched + 1) if last_matched >= 0 else len(pred_words)
    last = None                                   # forward-fill gaps within [0,cut)
    for i in range(cut):
        if assigned[i] is None:
            assigned[i] = last
        else:
            last = assigned[i]
    nxt = None                                    # back-fill any leading gap
    for i in range(cut - 1, -1, -1):
        if assigned[i] is None:
            assigned[i] = nxt
        else:
            nxt = assigned[i]
    truth = [dict(w) for w in pred_words[:cut]]
    for i, w in enumerate(truth):
        w["speaker"] = assigned[i] if assigned[i] is not None else w.get("speaker")
    return truth


def format_report(s, title=""):
    """One-glance human summary for the bench."""
    bc = ("n/a" if s["backchannel_acc"] is None
          else f"{s['backchannel_acc']*100:.1f}% ({s['backchannel_n']})")
    lines = [f"### {title}".rstrip(),
             f"**{s.get('mistakes', '?')} boundary mistakes** across {s.get('turns', '?')} turns  "
             f"· word acc {s['accuracy']*100:.2f}% ({s['aligned']-s['correct']} words off)",
             f"Back-channel accuracy: {bc}",
             f"Label map: {s['mapping']}"]
    if s["errors"]:
        lines.append(f"\n**{len(s['errors'])} mis-attributed:**")
        lines += [f"  {e['time']:>7.2f}s  '{e['word']}'  -> got {e['pred_mapped']} "
                  f"(was {e['pred']}), truth {e['truth']}" for e in s["errors"][:60]]
        if len(s["errors"]) > 60:
            lines.append(f"  ... +{len(s['errors'])-60} more")
    return "\n".join(lines)


def aggregate(scores):
    """Micro-average across files (pool all words): the scoreboard number we
    optimize, so one short file can't dominate one long one."""
    corr = sum(s["correct"] for s in scores)
    tot = sum(s["aligned"] for s in scores)
    bc_ok = sum(round((s["backchannel_acc"] or 0) * s["backchannel_n"]) for s in scores)
    bc_n = sum(s["backchannel_n"] for s in scores)
    return {"accuracy": (corr / tot) if tot else 0.0, "aligned": tot,
            "backchannel_acc": (bc_ok / bc_n) if bc_n else None, "backchannel_n": bc_n,
            "files": len(scores)}
