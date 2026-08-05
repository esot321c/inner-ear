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
MAX_ROWS = 8  # naming rows: pooled speakers across all tracks may exceed 4
SAMPLE_SECS = 12  # length of each voice's naming clip (a short, recognizable sample)
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


def _diarize_channel(diar_wav):
    """Sortformer soft posterior + speaker count for one channel. Loads/unloads
    the diarizer so its VRAM is free before transcription/DiCoW."""
    import torch
    dm = P.load_diarizer(os.environ.get("DIAR_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2"))
    try:
        post, fd = P.diarize_soft(dm, diar_wav)
        n_spk = P.count_speakers(post)
        return post, fd, n_spk
    finally:
        del dm
        torch.cuda.empty_cache()


def _cascade(ws, post, fd):
    """<=2-speaker path: faster-whisper words attributed via the soft posterior +
    validated post-processing (transition-fix then onset-fix)."""
    P.words_speaker_probs(ws, post, fd)
    P.group_sentences(ws)
    P.resolve_speakers(ws)
    P.fix_transition_leaks(ws, post, fd)   # boundary-straddle back-channels
    P.fix_onset_leaks(ws)                   # stranded turn-openers
    return ws


def _dicow(diar_wav, post, fd):
    """>=3-speaker path: diarization-conditioned Whisper. Decodes each speaker
    separately so overlapping speech is attributed inside the model. DiCoW decodes
    lowercase/unpunctuated, so we restore casing+punctuation per speaker stream."""
    import torch
    from collections import defaultdict
    import dicow_backend as DI
    from faster_whisper.audio import decode_audio
    try:
        audio = decode_audio(diar_wav)
        ws = DI.transcribe_dicow(audio, post, frame_dur_post=fd)
        try:
            pm = P.load_punctuator()
            by_spk = defaultdict(list)
            for w in ws:
                by_spk[w["speaker"]].append(w)
            for sub in by_spk.values():     # punctuate each speaker's own sequence
                P.restore_punctuation(sub, pm)
            del pm
        except Exception as e:
            print("DiCoW punctuation skipped:", e)
        return ws
    finally:
        torch.cuda.empty_cache()


def process_file(path):
    """Full pipeline on one file. Returns a result dict (pre-naming).

    Every audio track is diarized uniformly (no track is 'you'). Per track we
    pick the engine by that track's speaker count - the fast cascade
    (faster-whisper + Sortformer attribution + fixes) for <=2 speakers, DiCoW
    for >=3 (DICOW_MIN_SPK, default 3). Speaker ids are namespaced per track so
    they stay unique once pooled; the user names every speaker found."""
    import torch
    tmp = tempfile.mkdtemp()
    mode_label, jobs = P.plan_channels(path, tmp)

    stages_dir = os.path.join("/out", os.path.splitext(os.path.basename(path))[0], "stages")
    os.makedirs(stages_dir, exist_ok=True)

    def _stage(name, content):
        try:
            with open(os.path.join(stages_dir, name), "w", encoding="utf-8") as fh:
                fh.write(content if isinstance(content, str) else json.dumps(content, indent=2))
        except Exception as e:
            print(f"stage save {name} skipped:", e)

    min_spk = int(os.environ.get("DICOW_MIN_SPK", "3"))
    clip_dir = tempfile.mkdtemp()
    allw, clips, per_track = [], {}, []
    detection_lines = [f"input: {path}", f"channel mode: {mode_label}",
                       f"tracks: {[(r, os.path.basename(w)) for r, w in jobs]}"]

    for ti, (role, wav) in enumerate(jobs):
        # 1) diarize this track -> soft posterior + speaker count picks the engine
        post = fd = None
        n_spk = 0
        try:
            post, fd, n_spk = _diarize_channel(wav)
        except Exception as e:
            print(f"diarization failed on {role}:", e)
        use_dicow = post is not None and n_spk >= min_spk
        engine = "DiCoW" if use_dicow else ("cascade" if post is not None else "hard-segment fallback")
        print(f"[{role}] detected {n_spk} speaker(s) -> {engine}")
        detection_lines.append(f"  {role}: {n_spk} speaker(s) -> {engine}")

        # 2) transcribe + attribute
        if use_dicow:
            ws = _dicow(wav, post, fd)
        elif post is not None:
            wm = P.load_whisper(os.environ.get("WHISPER_MODEL", "large-v3"))
            ws = P.transcribe(wm, wav)
            del wm
            torch.cuda.empty_cache()
            try:
                pm = P.load_punctuator()
                P.restore_punctuation(ws, pm)
                del pm
                torch.cuda.empty_cache()
            except Exception as e:
                print("punctuation restoration skipped:", e)
            ws = _cascade(ws, post, fd)
        else:  # diarization unavailable -> hard-segment fallback
            wm = P.load_whisper(os.environ.get("WHISPER_MODEL", "large-v3"))
            ws = P.transcribe(wm, wav)
            del wm
            torch.cuda.empty_cache()
            dm = P.load_diarizer(os.environ.get("DIAR_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2"))
            P.assign_speakers(ws, P.diarize(dm, wav))
            P.smooth_sentences(ws)
            del dm
            torch.cuda.empty_cache()

        # tag provenance so the post-loop LLM pass touches only cascade words
        is_cascade = (not use_dicow) and (post is not None)
        for w in ws:
            w["_cascade"] = is_cascade

        # raw per-track transcript (pre-suppression view, for diagnostics)
        _stage(f"02_transcript_{role}.txt", "".join(w["word"] for w in ws).strip())

        # defer namespacing/pooling/clips until every track is transcribed, so
        # cross-track bleed suppression can compare tracks against each other.
        per_track.append({"ti": ti, "role": role, "wav": wav,
                          "post": post, "fd": fd, "words": ws})

    # BARRIER: every track transcribed. Drop cross-track bleed (a faint duplicate
    # of the same word heard louder on another track) before pooling, so one
    # speaker bleeding into another's mic doesn't appear as a second speaker.
    P.suppress_cross_track_bleed(
        [{"words": t["words"], "wav": t["wav"], "audio": None} for t in per_track])

    for t in per_track:
        ti, role, wav, post, fd, ws = (t["ti"], t["role"], t["wav"],
                                       t["post"], t["fd"], t["words"])
        P.namespace_speakers(ws, ti)
        _stage(f"02b_transcript_{role}_clean.txt", "".join(w["word"] for w in ws).strip())
        allw.extend(ws)
        # clean solo naming clip per surviving speaker on THIS track
        if post is not None:
            present_cols = sorted({int(str(w["speaker"]).rsplit("_", 1)[-1])
                                   for w in ws if "_" in str(w["speaker"])})
            windows = P.pick_solo_windows(post, fd, present_cols, clip_secs=SAMPLE_SECS)
            for col, (s, e) in windows.items():
                spk = f"t{ti}:speaker_{col}"
                dst = os.path.join(clip_dir, f"{spk.replace(':', '_')}.wav")
                clips[spk] = P.extract_clip(wav, s, e, dst)
        else:
            for spk, tt in P.pick_sample_turns(P.build_turns(ws)).items():
                s, e = tt["start"], tt["end"]
                if e - s > SAMPLE_SECS:
                    mid = (s + e) / 2
                    s, e = mid - SAMPLE_SECS / 2, mid + SAMPLE_SECS / 2
                dst = os.path.join(clip_dir, f"{spk.replace(':', '_')}.wav")
                clips[spk] = P.extract_clip(wav, s, e, dst)

    _stage("01_detection.txt", "\n".join(detection_lines) +
           f"\n  (DICOW_MIN_SPK={min_spk}, whisper={os.environ.get('WHISPER_MODEL', 'large-v3')})\n")

    allw.sort(key=lambda w: w["start"])

    # Optional LLM boundary reattribution — cascade words only. DiCoW output is
    # already per-speaker, and the fallback path has no soft posterior to reason
    # over, so both are left untouched.
    llm_url = os.environ.get("LLM_BASE_URL")
    if llm_url:
        cascade_words = [w for w in allw if w.get("_cascade")]
        if cascade_words:
            try:
                n = P.llm_cleanup(cascade_words, llm_url,
                                  os.environ.get("LLM_MODEL", "qwen2.5:7b"),
                                  os.environ.get("LLM_API_KEY"))
                print(f"LLM reattributed {n} boundary word(s)")
            except Exception as e:
                print("LLM cleanup skipped:", e)

    turns = P.build_turns_overlap(allw)
    _stage("03_words.json", [{k: w.get(k) for k in ("word", "start", "end", "speaker")} for w in allw])
    _stage("04_turns.txt", "\n".join(
        f"[{P._ts(t['start'])}] {t['speaker']}: {t['text'].strip()}" for t in turns))

    speakers = sorted({t["speaker"] for t in turns})
    clips = {spk: clips[spk] for spk in speakers if spk in clips}
    return {"path": path, "mode": mode_label, "turns": turns,
            "speakers": speakers, "clips": clips,
            "tmp": tmp, "clip_dir": clip_dir}


def save_named(result, names):
    base = os.path.splitext(os.path.basename(result["path"]))[0]
    od = os.path.join(OUT, base)
    os.makedirs(od, exist_ok=True)
    txt, data = P.render(result["turns"], names, result["mode"])
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
            gr.Markdown("Upload a meeting, transcribe, then play each detected "
                        "voice and give it a name - including yourself.")

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
        for i in range(MAX_ROWS):
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
            for i in range(MAX_ROWS):
                if i < len(res["speakers"]):
                    spk = res["speakers"][i]
                    u += [gr.update(visible=True), gr.update(value=res["clips"][spk]),
                          gr.update(value=""), spk]
                else:
                    u += [gr.update(visible=False), gr.update(value=None),
                          gr.update(value=""), None]
            return u

        row_outputs = []
        for i in range(MAX_ROWS):
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

        def on_single(file):
            if not file:
                yield [gr.update()] + show_none("Upload a file first.")
                return
            key = os.path.basename(file.name)
            yield [working("Transcribing")] + busy(
                f"⏳ Transcribing **{key}** on the GPU. This takes a few "
                "minutes - leave this tab open.")
            try:
                res = process_file(file.name)
                store_result(key, res)
                yield [idle(GO_LABEL)] + show(res, key, "Done - ")
            except Exception as e:
                print("single failed", file.name, e)
                yield [idle(GO_LABEL)] + show_none(f"Failed: {e}")

        go.click(on_single, [up], [go] + show_outputs)

        def on_batch_run():
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
                    store_result(key, process_file(f))
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

        run.click(on_batch_run, [], [run, picker, status])

        def on_pick(name):
            res = load_result(name)
            if not res:
                return show_none("That meeting's data is no longer available - reprocess it.")
            return show(res, name, f"{name}: ")

        picker.change(on_pick, [picker], show_outputs)

        def on_save(key, *name_vals):
            res = load_result(key)
            if not res:
                return gr.update(
                    value="Nothing to save - the meeting data expired. Re-run processing.",
                    visible=True)
            names, speakers = {}, res["speakers"]
            for i in range(MAX_ROWS):
                nm = name_vals[i]
                if i < len(speakers) and nm and nm.strip():
                    names[speakers[i]] = nm.strip()
            txt = save_named(res, names)
            drop_result(key)
            return gr.update(value=txt, visible=True)

        save.click(on_save, [save_key] + name_c, [out_txt])
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="0.0.0.0", server_port=7860)
