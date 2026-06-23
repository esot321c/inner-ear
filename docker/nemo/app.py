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

# Processed results are persisted here, keyed by meeting, so that "Save
# transcript" still works after the long processing gap - even if the Gradio
# session's in-memory gr.State was evicted (idle/backgrounded tab dropping the
# heartbeat) or the server was restarted. The browser keeps showing the result;
# this is where the data it needs to actually save lives.
CACHE = os.path.join(OUT, ".cache")


def _cache_path(key):
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in (key or ""))
    return os.path.join(CACHE, safe + ".json")


def store_result(key, res):
    os.makedirs(CACHE, exist_ok=True)
    with open(_cache_path(key), "w", encoding="utf-8") as fh:
        json.dump(res, fh)


def load_result(key):
    path = _cache_path(key)
    if key and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def drop_result(key):
    try:
        os.remove(_cache_path(key))
    except OSError:
        pass


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
    with gr.Blocks(title="Inner Ear") as demo:
        gr.Markdown("# Inner Ear\n"
                    "Local meeting transcriber - transcribe + diarize on your "
                    "GPU, name the voices, save.")
        with gr.Row():
            you_name = gr.Textbox("You", label="Your name (your own mic channel)")
            you_ch = gr.Radio(["0", "1"], value="0",
                              label="Which channel/track is YOU? (dual-channel files only)")

        with gr.Tab("Single meeting"):
            up = gr.File(label="Meeting file (multi-track MKV, stereo, or mono)",
                        file_count="single")
            go = gr.Button("Transcribe", variant="primary")
        with gr.Tab("Batch folder (in\\)"):
            gr.Markdown("Transcribe **every** file in **in\\** on the GPU. When it "
                        "finishes, open each meeting from the dropdown to name its "
                        "speakers and save - one meeting at a time.")
            run = gr.Button("Process all in in\\", variant="primary")
            picker = gr.Dropdown(label="Open a processed meeting to name + save",
                                 choices=[], visible=False)

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

        # Hidden carrier for the meeting key. It's a real component value (lives
        # in the browser DOM and is re-sent with the Save click), so it survives
        # the long gap that wipes gr.State. on_save uses it to reload the result
        # from disk - never trusting in-memory state to still be there.
        save_key = gr.Textbox(visible=False)

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

        def show(res, key, prefix=""):
            """Common 'a meeting is loaded' UI update, plus the durable key."""
            msg = f"{prefix}{res['mode']}, {len(res['speakers'])} other voice(s) detected."
            return ([gr.update(value=msg, visible=True), gr.update(visible=True), key]
                    + rows_for(res))

        def show_none(msg):
            return ([gr.update(value=msg, visible=True), gr.update(visible=False), ""]
                    + rows_for({"speakers": [], "clips": {}}))

        def busy(msg):
            """A 'still working' status update that leaves the voice rows alone."""
            return ([gr.update(value=msg, visible=True), gr.update(visible=False), ""]
                    + [gr.update() for _ in row_outputs])

        # outputs shared by the "show a meeting" handlers
        show_outputs = [status, save, save_key] + row_outputs

        GO_LABEL, RUN_LABEL = "Transcribe", "Process all in in\\"

        def working(label):
            """Disabled button showing a spinner + 'Working…' while a job runs."""
            return gr.update(value=f"⏳ {label}…", interactive=False)

        def idle(label):
            return gr.update(value=label, interactive=True)

        def on_single(file, ych):
            if not file:
                yield [gr.update()] + show_none("Upload a file first.")
                return
            key = os.path.basename(file.name)
            yield [working("Transcribing")] + busy(
                f"⏳ Transcribing **{key}** on the GPU. This takes a few "
                "minutes - leave this tab open.")
            try:
                res = process_file(file.name, ych)
                store_result(key, res)
                yield [idle(GO_LABEL)] + show(res, key, "Done - ")
            except Exception as e:
                print("single failed", file.name, e)
                yield [idle(GO_LABEL)] + show_none(f"Failed: {e}")

        go.click(on_single, [up, you_ch], [go] + show_outputs)

        def on_batch_run(ych):
            files = [f for f in sorted(glob.glob(os.path.join(IN, "*")))
                     if os.path.isfile(f) and f.lower().endswith(EXTS)]
            if not files:
                yield (gr.update(), gr.update(choices=[], visible=False, value=None),
                       gr.update(value="No audio/video files found in **in\\**.",
                                 visible=True))
                return
            n = len(files)
            yield (working("Processing"), gr.update(),
                   gr.update(value=f"⏳ Processing {n} file(s) on the GPU. A few "
                             "minutes each - leave this tab open.", visible=True))
            done, errs = [], []
            for idx, f in enumerate(files, 1):
                key = os.path.basename(f)
                yield (gr.update(), gr.update(),
                       gr.update(value=f"⏳ Processing {idx}/{n}: **{key}** …",
                                 visible=True))
                try:
                    store_result(key, process_file(f, ych))
                    done.append(key)
                except Exception as e:
                    errs.append(f"{key}: {e}")
                    print("skip", f, e)
            msg = (f"✓ Processed {len(done)}/{n} file(s). Use the dropdown to open "
                   "each meeting, name its speakers, and save.")
            if errs:
                msg += "  ⚠ Skipped: " + "; ".join(errs)
            yield (idle(RUN_LABEL),
                   gr.update(choices=done, visible=bool(done),
                             value=done[0] if done else None),
                   gr.update(value=msg, visible=True))

        run.click(on_batch_run, [you_ch], [run, picker, status])

        def on_pick(name):
            res = load_result(name)
            if not res:
                return show_none("That meeting's data is no longer available - reprocess it.")
            return show(res, name, f"{name}: ")

        picker.change(on_pick, [picker], show_outputs)

        def on_save(key, yname, *name_vals):
            res = load_result(key)
            if not res:
                return gr.update(
                    value="Nothing to save - the meeting data expired. Re-run processing.",
                    visible=True)
            names, speakers = {}, res["speakers"]
            for i in range(MAX_SPK):
                nm = name_vals[i]
                if i < len(speakers) and nm and nm.strip():
                    names[speakers[i]] = nm.strip()
            txt = save_named(res, names, yname)
            drop_result(key)
            return gr.update(value=txt, visible=True)

        save.click(on_save, [save_key, you_name] + name_c, [out_txt])
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="0.0.0.0", server_port=7860)
