"""Gradio stage bench: run the transcription pipeline stage by stage and inspect
every stage's output in the browser. The slow stages (Whisper transcription,
Sortformer diarization) are cached per file, so re-runs and option tweaks are
instant.

Runs are persisted to disk (out/bench/_runs/), so the run counter is reliable
across restarts and you can reload any past run from the History dropdown to
compare. Reference feedback as "run N, stage M"."""
import os
import json
import subprocess
import numpy as np
import torch
import gradio as gr
import pipeline as P
import scoring

BENCH = "/out/bench"
RUNS_DIR = "/out/bench/_runs"
EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
        ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")
STAGE_KEYS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]


def list_files():
    out = []
    for base in ("/in", "/archive"):
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f.lower().endswith(EXTS):
                    out.append(os.path.join(root, f))
    return sorted(out)


def _key(p):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in os.path.basename(p))


def _next_run_num():
    os.makedirs(RUNS_DIR, exist_ok=True)
    nums = [int(f[4:-5]) for f in os.listdir(RUNS_DIR)
            if f.startswith("run_") and f.endswith(".json") and f[4:-5].isdigit()]
    return (max(nums) + 1) if nums else 1


def _save_run(rec):
    os.makedirs(RUNS_DIR, exist_ok=True)
    json.dump(rec, open(os.path.join(RUNS_DIR, f"run_{rec['num']:04d}.json"), "w", encoding="utf-8"))


def _run_choices():
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for f in sorted(os.listdir(RUNS_DIR)):
        if f.startswith("run_") and f.endswith(".json"):
            try:
                r = json.load(open(os.path.join(RUNS_DIR, f), encoding="utf-8"))
                out.append((f"Run {r['num']} · {r['file']} · punct={r['punct']} llm={r['llm']}", r["num"]))
            except Exception:
                pass
    return out[::-1]  # newest first


def _transcribe(wav, cdir, fresh):
    f = os.path.join(cdir, "_words.json")
    if not fresh and os.path.exists(f):
        return json.load(open(f, encoding="utf-8"))
    wm = P.load_whisper(os.environ.get("WHISPER_MODEL", "large-v3"))
    w = P.transcribe(wm, wav)
    del wm
    torch.cuda.empty_cache()
    json.dump(w, open(f, "w", encoding="utf-8"))
    return w


def _diarize(wav, cdir, fresh):
    f = os.path.join(cdir, "_post.npy")
    if not fresh and os.path.exists(f):
        return np.load(f), P.FRAME_DUR
    dm = P.load_diarizer(os.environ.get("DIAR_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2"))
    post, fd = P.diarize_soft(dm, wav)
    del dm
    torch.cuda.empty_cache()
    np.save(f, post)
    return post, fd


def _turns(words):
    return "\n".join(f"[{t['start']:6.1f}s] {t['speaker']}: {t['text'].strip()}"
                     for t in P.build_turns([dict(w) for w in words]))


def _parse_time(t):
    """Accept '90', '1:30', or '1:02:03' -> seconds."""
    t = str(t).strip()
    if not t:
        return 0.0
    parts = t.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return 0.0
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def make_clip(src, start, length):
    """Cut a section out of a long file into /in (re-encoded audio only, channels
    preserved so stereo/dual-track detection still works), add it to the picker."""
    if not src:
        return gr.update(), "Pick a source file first."
    s = _parse_time(start)
    L = int(_parse_time(length) or 300)
    base = os.path.splitext(os.path.basename(src))[0]
    name = f"{base}__clip_{int(s)}s+{L}s.m4a"
    out = os.path.join("/in", name)
    cmd = ["ffmpeg", "-y", "-ss", str(s), "-t", str(L), "-i", src,
           "-vn", "-c:a", "aac", "-b:a", "192k", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(out):
        return gr.update(), f"ffmpeg failed: {p.stderr.strip()[-300:]}"
    return (gr.update(choices=list_files(), value=out),
            f"Made **{name}** ({L}s from {start or '0'}). Selected it — hit Run.")


def _editable_turns(words):
    """Clean 'speaker: text' per turn — what the user edits into ground truth."""
    return "\n".join(f"{t['speaker']}: {t['text'].strip()}"
                     for t in P.build_turns([dict(w) for w in words]))


def _pred_words(words):
    return [{"word": w["word"], "start": w.get("start", 0.0),
             "end": w.get("end", 0.0), "speaker": w["speaker"]} for w in words]


def save_truth(src, edited):
    """Build word-level ground truth from the edited turns and persist it."""
    if not src:
        return "Pick and run a file first."
    cdir = os.path.join(BENCH, _key(src))
    pf = os.path.join(cdir, "_last_pred.json")
    if not os.path.exists(pf):
        return "Run this file first, then edit + save."
    pred = json.load(open(pf, encoding="utf-8"))
    truth = scoring.build_truth(pred, edited)
    json.dump(truth, open(os.path.join(cdir, "_truth.json"), "w", encoding="utf-8"))
    s = scoring.score(pred, truth)
    return (f"Saved ground truth ({len(truth)} words) for `{os.path.basename(src)}`. "
            f"This run scored **{s['accuracy']*100:.2f}%** against it.")


def diff_runs(num_a, num_b):
    """Exact word-level diff between two runs of the same file: every word whose
    speaker changed from run A to run B, and (if truth exists) whether the change
    moved it toward or away from the truth. Callable from the CLI for analysis."""
    runs = {r["num"]: r for r in _all_runs()}
    a, b = runs.get(int(num_a)), runs.get(int(num_b))
    if not a or not b:
        return f"run {num_a} or {num_b} not found"
    if a["file"] != b["file"]:
        return f"different files: {a['file']} vs {b['file']}"
    pa, pb = a.get("pred") or [], b.get("pred") or []
    truth = _truth_for(a["file"])
    tmap = {}
    if truth:
        for w in truth:
            tmap[(round(float(w.get("start", 0)), 2), scoring._norm(w["word"]))] = w["speaker"]
    out = [f"diff run {num_a} ({a.get('punct')},llm={a.get('llm')},tfix={a.get('tfix')}) "
           f"-> run {num_b} ({b.get('punct')},llm={b.get('llm')},tfix={b.get('tfix')})  [{a['file']}]"]
    changed = better = worse = 0
    for wa, wb in zip(pa, pb):
        if wa["speaker"] == wb["speaker"]:
            continue
        changed += 1
        key = (round(float(wb.get("start", 0)), 2), scoring._norm(wb["word"]))
        t = tmap.get(key)
        verdict = ""
        if t is not None:
            a_ok, b_ok = wa["speaker"] == t, wb["speaker"] == t
            if b_ok and not a_ok:
                verdict = "FIXED"; better += 1
            elif a_ok and not b_ok:
                verdict = "BROKE"; worse += 1
            else:
                verdict = "(both wrong)" if not b_ok else "(both right)"
        out.append(f"  {wb.get('start',0):7.2f}s '{wb['word'].strip()}'  "
                   f"{wa['speaker']} -> {wb['speaker']}  truth={t}  {verdict}")
    out.append(f"\n{changed} words changed | FIXED {better} | BROKE {worse} | net {better-worse:+d}")
    return "\n".join(out)


def _all_runs():
    """Every persisted run record, oldest first."""
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for f in sorted(os.listdir(RUNS_DIR)):
        if f.startswith("run_") and f.endswith(".json"):
            try:
                out.append(json.load(open(os.path.join(RUNS_DIR, f), encoding="utf-8")))
            except Exception:
                pass
    return out


def _truth_for(file_basename):
    tf = os.path.join(BENCH, _key(file_basename), "_truth.json")
    return json.load(open(tf, encoding="utf-8")) if os.path.exists(tf) else None


def scoreboard():
    """Experiment log: every run, grouped by file, scored LIVE against that
    file's locked ground truth (so saving/editing truth re-scores all past runs),
    with the per-run delta. This is the data we mine to compare settings."""
    from collections import defaultdict
    runs = _all_runs()
    if not runs:
        return "No runs yet. Run a file, correct the transcript, **Save as ground truth**."
    byfile = defaultdict(list)
    for r in runs:
        byfile[r["file"]].append(r)
    blocks = []
    for file in sorted(byfile):
        rs = sorted(byfile[file], key=lambda r: r["num"])
        truth = _truth_for(file)
        if truth:
            end = float(truth[-1].get("end", truth[-1].get("start", 0)))
            tag = f"  _(labeled to {int(end)//60}:{int(end)%60:02d}, {len(truth)} words)_"
        else:
            tag = "  _(no ground truth yet — save one to score these)_"
        lines = [f"### {file}{tag}",
                 "| run | punct | llm | tfix | **mistakes** | turns | back-ch | words off | word acc | Δ mistakes |",
                 "|--|--|--|--|--|--|--|--|--|--|"]
        prev = None
        best_num, best_mis = None, 10 ** 9
        for r in rs:
            st = (r.get("punct", "?"), "on" if r.get("llm") else "off",
                  "on" if r.get("tfix") else "off")
            if truth is not None and r.get("pred"):
                s = scoring.score(r["pred"], truth)
                mis = s["mistakes"]
                bc = "n/a" if s["backchannel_acc"] is None else f"{s['backchannel_acc']*100:.0f}%"
                d = "" if prev is None else f"{mis - prev:+d}"
                prev = mis
                if mis < best_mis:
                    best_mis, best_num = mis, r["num"]
                lines.append(f"| {r['num']} | {st[0]} | {st[1]} | {st[2]} "
                             f"| **{mis}** | {s['turns']} | {bc} | {s['aligned']-s['correct']} "
                             f"| {s['accuracy']*100:.2f}% | {d} |")
            else:
                lines.append(f"| {r['num']} | {st[0]} | {st[1]} | {st[2]} | — | — | — | — | — | |")
        if best_num is not None:
            lines.append(f"\nFewest mistakes: **run {best_num} = {best_mis}**")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def run(src, punct, use_llm, llm_url, model, fresh, tfix, progress=gr.Progress()):
    if not src:
        return ["**Pick a file first.**", "", "", "", "", "", "", "",
                gr.update(), gr.update(), scoreboard()]
    num = _next_run_num()
    cdir = os.path.join(BENCH, _key(src))
    os.makedirs(cdir, exist_ok=True)
    wavdir = os.path.join(cdir, "_wav")
    os.makedirs(wavdir, exist_ok=True)

    progress(0.05, desc="channels")
    mode, jobs = P.plan_channels(src, wavdir)
    wav = jobs[-1][1]

    progress(0.2, desc="transcribe")
    words = _transcribe(wav, cdir, fresh)
    s1 = "".join(w["word"] for w in words).strip()

    s2 = "(punctuation not set to 'before')"
    if punct == "before":
        progress(0.35, desc="punctuation")
        pm = P.load_punctuator(); P.restore_punctuation(words, pm); del pm; torch.cuda.empty_cache()
        s2 = "".join(w["word"] for w in words).strip()

    progress(0.5, desc="diarize")
    post, fd = _diarize(wav, cdir, fresh)
    win = max(1, int(5 / fd))
    s3 = "\n".join(
        f"[{s0*fd:6.1f}s] dominant=speaker_{int(post[s0:s0+win].mean(axis=0).argmax())}"
        f"   probs={[round(float(x),2) for x in post[s0:s0+win].mean(axis=0)]}"
        for s0 in range(0, post.shape[0], win))

    progress(0.6, desc="resolve")
    P.words_speaker_probs(words, post, fd); P.group_sentences(words); P.resolve_speakers(words)
    s4 = _turns(words)
    tmoved = P.fix_transition_leaks(words, post, fd) if tfix else 0
    P.fix_onset_leaks(words)
    s5 = ((f"# transition-fix: {tmoved} word(s) moved by posterior shape\n\n" if tfix
           else "# (transition-fix off)\n\n") + _turns(words))

    s6 = "(only runs when Punctuation = 'after')"
    if punct == "after":
        progress(0.7, desc="punctuation (after)")
        pm = P.load_punctuator(); P.restore_punctuation(words, pm); del pm; torch.cuda.empty_cache()
        s6 = _turns(words)

    s7 = "(LLM pass off)"
    if use_llm and llm_url:
        progress(0.85, desc="LLM boundary pass")
        n = P.llm_cleanup(words, llm_url, model)
        s7 = f"# {n} boundary word(s) moved\n\n" + _turns(words)

    # persist this run's final words so ground-truth save + scoring + diffing work
    pred = _pred_words(words)
    json.dump(pred, open(os.path.join(cdir, "_last_pred.json"), "w", encoding="utf-8"))

    rec = {"num": num, "file": os.path.basename(src), "mode": mode, "punct": punct,
           "llm": bool(use_llm), "model": model, "tfix": bool(tfix),
           "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7,
           "pred": pred}
    _save_run(rec)

    score_md = ""
    tf = os.path.join(cdir, "_truth.json")
    if os.path.exists(tf):
        truth = json.load(open(tf, encoding="utf-8"))
        score_md = "\n\n" + scoring.format_report(scoring.score(pred, truth),
                                                   "Score vs saved ground truth")
    head = (f"### Run {num} — `{os.path.basename(src)}`\n"
            f"mode: **{mode}** · punctuation: **{punct}** · LLM: **{use_llm}** ({model}) · "
            f"saved (reload from History)" + score_md)
    return [head, s1, s2, s3, s4, s5, s6, s7,
            gr.update(choices=_run_choices(), value=num),
            _editable_turns(words), scoreboard()]


def load_run(num):
    if num is None:
        return [gr.update()] * 9
    r = json.load(open(os.path.join(RUNS_DIR, f"run_{int(num):04d}.json"), encoding="utf-8"))
    head = (f"### Run {r['num']} (reloaded) — `{r['file']}`\n"
            f"mode: **{r.get('mode','?')}** · punctuation: **{r['punct']}** · LLM: **{r['llm']}**")
    return [head] + [r[k] for k in STAGE_KEYS]


with gr.Blocks(title="Stage Bench") as demo:
    gr.Markdown("# Pipeline Stage Bench\n"
                "Run the pipeline stage by stage and inspect each output. Whisper + "
                "Sortformer are cached per file. Runs are saved — reload any from "
                "**History** to compare. Reference feedback as **run N, stage M**.")
    with gr.Row():
        f = gr.Dropdown(list_files(), label="File (from in/ and archive/)", scale=4)
        refresh = gr.Button("↻", scale=0)
    with gr.Accordion("✂ Make a test clip from a long file", open=False):
        with gr.Row():
            clip_start = gr.Textbox("0:00", label="Start (s or m:ss)", scale=1)
            clip_len = gr.Textbox("300", label="Length (s or m:ss)", scale=1)
            clip_btn = gr.Button("✂ Make clip", scale=1)
        clip_status = gr.Markdown()
    with gr.Row():
        punct = gr.Radio(["before", "after", "off"], value="before", label="Punctuation timing")
        use_llm = gr.Checkbox(True, label="LLM boundary pass")
        tfix = gr.Checkbox(True, label="Transition-fix (boundary straddle)")
        fresh = gr.Checkbox(False, label="Recompute cache")
    with gr.Row():
        llm_url = gr.Textbox("http://host.docker.internal:11434/v1", label="LLM base URL")
        model = gr.Textbox("qwen2.5:7b", label="LLM model")
    with gr.Row():
        run_btn = gr.Button("▶ Run pipeline", variant="primary", scale=3)
        history = gr.Dropdown(_run_choices(), label="History (reload a past run)", scale=3)
    head = gr.Markdown()

    boxes = []
    stages = [("1. Raw transcript — Whisper output, no speakers yet", 8, False),
              ("2. Punctuated (before diarization)", 8, False),
              ("3. Diarization timeline — Sortformer dominant speaker / 5 s", 10, False),
              ("4. Resolved — sentence-aware Viterbi assigns speakers", 16, True),
              ("5. Onset-fix — deterministic opener cleanup", 16, False),
              ("6. Punctuated (after resolution)", 16, False),
              ("7. LLM boundary pass — final", 16, True)]
    for title, lines, open_ in stages:
        with gr.Accordion(title, open=open_):
            boxes.append(gr.Textbox(lines=lines, show_label=False))

    with gr.Accordion("✓ Ground truth — correct the transcript, then save it as truth", open=True):
        gr.Markdown("Fix speaker labels / move stray words to the right speaker "
                    "(keep the `speaker_N:` prefix or rename it — names are matched "
                    "by best overlap). Then **Save as ground truth**. Every future run "
                    "of this file auto-scores against it.")
        gt_box = gr.Textbox(lines=16, show_label=False)
        with gr.Row():
            save_btn = gr.Button("💾 Save as ground truth", variant="primary", scale=2)
            gt_status = gr.Markdown(scale=4)

    with gr.Accordion("📊 Run log — every run vs ground truth, by file (compare settings here)", open=True):
        board = gr.Markdown(scoreboard())
        board_refresh = gr.Button("↻ refresh", scale=0)

    refresh.click(lambda: gr.update(choices=list_files()), None, f)
    clip_btn.click(make_clip, [f, clip_start, clip_len], [f, clip_status])
    run_btn.click(run, [f, punct, use_llm, llm_url, model, fresh, tfix],
                  [head] + boxes + [history, gt_box, board])
    history.change(load_run, [history], [head] + boxes)
    save_btn.click(save_truth, [f, gt_box], gt_status).then(
        lambda: scoreboard(), None, board)
    board_refresh.click(lambda: scoreboard(), None, board)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
