"""Gradio web app: transcribe + diarize a meeting on the GPU, play a clip of
each detected voice, name them, save a labeled transcript. Runs inside the
nemo-sortformer container. Models load/unload per job to fit 8 GB VRAM."""
import os
import glob
import json
import tempfile
import shutil
import datetime
import gradio as gr
import pipeline as P

IN, OUT, ARCHIVE = "/in", "/out", "/archive"
MAX_SPK = 4  # Sortformer caps at 4 speakers
EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def process_file(path, you_ch):
    """Full pipeline on one file. Returns a result dict (pre-naming)."""
    import torch
    tmp = tempfile.mkdtemp()
    mode_label, jobs = P.plan_channels(path, tmp, int(you_ch))

    wm = P.load_whisper(os.environ.get("WHISPER_MODEL", "large-v3"))
    words = {wav: P.transcribe(wm, wav) for _, wav in jobs}
    del wm
    torch.cuda.empty_cache()

    dm = P.load_diarizer(os.environ.get("DIAR_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2"))
    allw, diar_wav = [], None
    for role, wav in jobs:
        ws = words[wav]
        if role == "you":
            for w in ws:
                w["speaker"] = "__you__"
        else:
            diar_wav = wav
            P.assign_speakers(ws, P.diarize(dm, wav))
            P.smooth_sentences(ws)
        allw.extend(ws)
    del dm
    torch.cuda.empty_cache()

    allw.sort(key=lambda w: w["start"])
    turns = P.build_turns(allw)
    samples = P.pick_sample_turns(turns)
    clip_dir = tempfile.mkdtemp()
    clips = {}
    for spk, t in samples.items():
        dst = os.path.join(clip_dir, f"{spk}.wav")
        clips[spk] = P.extract_clip(diar_wav, t["start"], t["end"], dst)
    return {"path": path, "mode": mode_label, "turns": turns,
            "speakers": sorted(samples.keys()), "clips": clips,
            "tmp": tmp, "clip_dir": clip_dir}


def save_named(result, names, you_name):
    base = os.path.splitext(os.path.basename(result["path"]))[0]
    od = os.path.join(OUT, base)
    os.makedirs(od, exist_ok=True)
    txt, data = P.render(result["turns"], names, you_name, result["mode"])
    with open(os.path.join(od, base + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(od, base + ".json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if os.path.isdir(ARCHIVE) and os.path.exists(result["path"]):
        dest = os.path.join(ARCHIVE, stamp)
        os.makedirs(dest, exist_ok=True)
        tgt = os.path.join(dest, os.path.basename(result["path"]))
        if not os.path.exists(tgt):
            shutil.move(result["path"], tgt)
    shutil.rmtree(result.get("tmp", ""), ignore_errors=True)
    shutil.rmtree(result.get("clip_dir", ""), ignore_errors=True)
    return txt


def build_ui():
    with gr.Blocks(title="Meeting Transcriber") as demo:
        gr.Markdown("# Meeting Transcriber\n"
                    "Transcribe + diarize on your GPU, name the voices, save.")
        with gr.Row():
            you_name = gr.Textbox("You", label="Your name (your own mic channel)")
            you_ch = gr.Radio(["0", "1"], value="0",
                              label="Which channel/track is YOU? (dual-channel files only)")

        with gr.Tab("Single meeting"):
            up = gr.File(label="Meeting file (multi-track MKV, stereo, or mono)",
                        file_count="single")
            go = gr.Button("Transcribe", variant="primary")
        with gr.Tab("Batch folder (in\\)"):
            gr.Markdown("Process every file in **in\\**, then pick one to name.")
            run = gr.Button("Process all in in\\", variant="primary")
            picker = gr.Dropdown(label="Processed meetings", choices=[], visible=False)

        status = gr.Markdown(visible=False)
        rows, clip_c, name_c, spk_c = [], [], [], []
        for i in range(MAX_SPK):
            with gr.Row(visible=False) as r:
                clip_c.append(gr.Audio(label=f"Voice {i + 1}", interactive=False))
                name_c.append(gr.Textbox(label="Name this voice", scale=1))
                spk_c.append(gr.State())
            rows.append(r)
        save = gr.Button("Save transcript", variant="primary", visible=False)
        out_txt = gr.Textbox(label="Saved transcript", lines=20, visible=False)

        current = gr.State()
        batch_results = gr.State({})

        def rows_for(res):
            u = []
            for i in range(MAX_SPK):
                if i < len(res["speakers"]):
                    spk = res["speakers"][i]
                    u += [gr.update(visible=True), gr.update(value=res["clips"][spk]),
                          gr.update(value=""), spk]
                else:
                    u += [gr.update(visible=False), gr.update(value=None),
                          gr.update(value=""), None]
            return u

        row_outputs = []
        for i in range(MAX_SPK):
            row_outputs += [rows[i], clip_c[i], name_c[i], spk_c[i]]

        def on_single(file, ych):
            if not file:
                return [None, gr.update(value="Upload a file first.", visible=True),
                        gr.update(visible=False)] + rows_for({"speakers": [], "clips": {}})
            res = process_file(file.name, ych)
            msg = f"Done - {res['mode']}, {len(res['speakers'])} other voice(s) detected."
            return [res, gr.update(value=msg, visible=True),
                    gr.update(visible=True)] + rows_for(res)

        go.click(on_single, [up, you_ch],
                 [current, status, save] + row_outputs)

        def on_batch_run(ych):
            files = [f for f in sorted(glob.glob(os.path.join(IN, "*")))
                     if os.path.isfile(f) and f.lower().endswith(EXTS)]
            done = {}
            for f in files:
                try:
                    done[os.path.basename(f)] = process_file(f, ych)
                except Exception as e:
                    print("skip", f, e)
            choices = list(done.keys())
            return done, gr.update(choices=choices, visible=True,
                                   value=choices[0] if choices else None)

        run.click(on_batch_run, [you_ch], [batch_results, picker])

        def on_pick(name, results):
            if not name or name not in results:
                return [None, gr.update(visible=False),
                        gr.update(visible=False)] + rows_for({"speakers": [], "clips": {}})
            res = results[name]
            msg = f"{name}: {res['mode']}, {len(res['speakers'])} other voice(s)."
            return [res, gr.update(value=msg, visible=True),
                    gr.update(visible=True)] + rows_for(res)

        picker.change(on_pick, [picker, batch_results],
                      [current, status, save] + row_outputs)

        def on_save(res, yname, *vals):
            if not res:
                return gr.update(value="Nothing to save yet.", visible=True)
            names = {}
            for i in range(MAX_SPK):
                nm = vals[i]
                spk = vals[MAX_SPK + i]
                if spk and nm and nm.strip():
                    names[spk] = nm.strip()
            txt = save_named(res, names, yname)
            return gr.update(value=txt, visible=True)

        save.click(on_save, [current, you_name] + name_c + spk_c, [out_txt])
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="0.0.0.0", server_port=7860)
