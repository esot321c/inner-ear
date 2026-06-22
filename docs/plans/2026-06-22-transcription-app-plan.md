# Transcription + Diarization App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Gradio web app that transcribes a meeting, detects the voices, lets the user play a clip of each and name them, and saves a clean named transcript — running on the GPU inside the existing Docker image.

**Architecture:** Refactor the existing one-shot script into a shared, importable `pipeline.py` (pure helpers are stdlib-only and unit-tested; model-bound helpers lazy-import torch/NeMo/faster-whisper). A Gradio `app.py` and the batch CLI both call `pipeline.py`. App + engine run in the same `nemo-sortformer` container so models load in-process. Spec: `docs/2026-06-22-transcription-app-design.md`.

**Tech Stack:** Python 3.10 (container) / 3.12 (venv for unit tests), Gradio, faster-whisper (large-v3), NeMo streaming Sortformer, ffmpeg, Docker (GPU), pytest.

---

## Conventions

- **Container files** live in `docker/nemo/`. The image already has faster-whisper + NeMo; we add Gradio + soundfile.
- **Unit tests** run in the native venv (no GPU/NeMo needed) because pure helpers are stdlib-only:
  `C:\Users\djgra\Transcribe\.venv\Scripts\python.exe -m pytest docker/nemo/tests -v`
- **Integration tests** run the container (GPU) on the 80s slice `docker/nemo/audio.wav`.
- **Commit** after each task. Repo root: `C:\Users\djgra\Transcribe` (branch `main`).
- **No heavy imports at module top of `pipeline.py`** — `import torch` / `faster_whisper` / `nemo` go *inside* the model functions, so pure helpers import without a GPU.
- Sortformer caps at **4 speakers**; the naming UI pre-creates 4 rows and shows N.

## File Structure

- Create `docker/nemo/pipeline.py` — shared engine (pure helpers + model helpers).
- Create `docker/nemo/tests/test_pipeline.py` — unit tests for pure helpers.
- Create `docker/nemo/tests/__init__.py` — empty.
- Create `docker/nemo/app.py` — Gradio app (single + batch tabs, naming UI).
- Modify `docker/nemo/diarize_transcribe.py` — become a thin CLI over `pipeline.py`.
- Modify `docker/nemo/Dockerfile` — `pip install gradio soundfile`, COPY new files.
- Create `Transcribe-App.bat` — launch container with port 7860, open browser.
- Create `LICENSE` — MIT.
- Modify `README.txt` — usage + Model Licenses section.
- Create `docker/nemo/tests/fixtures.py` — sample word/segment dicts for tests.

---

## Task 1: Channel-mode decision (pure, TDD)

**Files:**
- Create: `docker/nemo/pipeline.py`
- Create: `docker/nemo/tests/__init__.py` (empty)
- Test: `docker/nemo/tests/test_pipeline.py`

- [ ] **Step 1: Install pytest in the venv**

Run: `C:\Users\djgra\Transcribe\.venv\Scripts\python.exe -m pip install pytest`
Expected: pytest installed.

- [ ] **Step 2: Write the failing test**

```python
# docker/nemo/tests/test_pipeline.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import decide_mode

def test_two_tracks_is_dual_track():
    assert decide_mode(streams=[1, 1], lr_diff_db=-99) == "dual_track"

def test_stereo_with_difference_is_dual_stereo():
    assert decide_mode(streams=[2], lr_diff_db=-20) == "dual_stereo"

def test_dual_mono_is_single():
    assert decide_mode(streams=[2], lr_diff_db=-59) == "single"

def test_mono_is_single():
    assert decide_mode(streams=[1], lr_diff_db=-99) == "single"
```

- [ ] **Step 3: Run test, verify it fails**

Run: `C:\Users\djgra\Transcribe\.venv\Scripts\python.exe -m pytest docker/nemo/tests/test_pipeline.py -v`
Expected: FAIL (ImportError: cannot import name 'decide_mode').

- [ ] **Step 4: Implement `decide_mode` in `pipeline.py`**

```python
# docker/nemo/pipeline.py
"""Shared transcription + diarization engine. Pure helpers are stdlib-only and
import without a GPU; model helpers lazy-import torch/faster_whisper/nemo."""
import os, re, json, glob, subprocess, tempfile, shutil
from collections import defaultdict

DUAL_THRESHOLD_DB = -40  # L-R louder than this => true stereo, not dual-mono

def decide_mode(streams, lr_diff_db):
    """streams = list of channel-counts per audio stream. Returns
    'dual_track' | 'dual_stereo' | 'single'."""
    if len(streams) >= 2:
        return "dual_track"
    if streams and streams[0] >= 2 and lr_diff_db > DUAL_THRESHOLD_DB:
        return "dual_stereo"
    return "single"
```

- [ ] **Step 5: Run test, verify it passes**

Run: `C:\Users\djgra\Transcribe\.venv\Scripts\python.exe -m pytest docker/nemo/tests/test_pipeline.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add docker/nemo/pipeline.py docker/nemo/tests
git commit -m "feat(pipeline): channel-mode decision with tests"
```

---

## Task 2: Sentence smoothing + turn grouping (pure, TDD)

**Files:**
- Modify: `docker/nemo/pipeline.py`
- Test: `docker/nemo/tests/test_pipeline.py`

- [ ] **Step 1: Add failing tests**

```python
# append to test_pipeline.py
from pipeline import assign_speakers, smooth_sentences, build_turns

def _w(start, end, word, speaker=None):
    return {"start": start, "end": end, "word": word, "speaker": speaker}

def test_assign_speakers_by_midpoint():
    words = [_w(0.0, 1.0, " Hi"), _w(1.0, 2.0, " there")]
    segs = [(0.0, 0.9, "speaker_0"), (0.9, 2.0, "speaker_1")]
    assign_speakers(words, segs)
    assert words[0]["speaker"] == "speaker_0"
    assert words[1]["speaker"] == "speaker_1"

def test_smooth_moves_boundary_word_to_sentence_majority():
    # "Was" tagged spk0 but the sentence "Was it good?" is mostly spk1 -> all spk1
    words = [_w(0.0, 0.2, " Was", "speaker_0"),
             _w(0.2, 0.6, " it", "speaker_1"),
             _w(0.6, 1.4, " good?", "speaker_1")]
    smooth_sentences(words)
    assert [w["speaker"] for w in words] == ["speaker_1"] * 3

def test_build_turns_groups_consecutive():
    words = [_w(0.0, 1.0, " Hi", "A"), _w(1.0, 2.0, " there", "A"),
             _w(2.0, 3.0, " Yo", "B")]
    turns = build_turns(words)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "A" and turns[0]["text"].strip() == "Hi there"
    assert turns[1]["speaker"] == "B"
```

- [ ] **Step 2: Run, verify fail**

Run: `...python.exe -m pytest docker/nemo/tests/test_pipeline.py -v`
Expected: FAIL (ImportError for assign_speakers/smooth_sentences/build_turns).

- [ ] **Step 3: Implement in `pipeline.py`**

```python
def assign_speakers(words, dsegs):
    def spk_at(t):
        for st, en, sp in dsegs:
            if st <= t <= en:
                return sp
        return min(dsegs, key=lambda x: min(abs(x[0]-t), abs(x[1]-t)))[2] if dsegs else "speaker_0"
    for w in words:
        w["speaker"] = spk_at((w["start"] + w["end"]) / 2)

def smooth_sentences(words):
    sent, cur = [], []
    for w in words:
        cur.append(w)
        t = w["word"].strip()
        if t and t[-1] in ".?!":
            sent.append(cur); cur = []
    if cur:
        sent.append(cur)
    for s in sent:
        dur = defaultdict(float)
        for w in s:
            dur[w["speaker"]] += w["end"] - w["start"]
        dom = max(dur, key=dur.get)
        for w in s:
            w["speaker"] = dom

def build_turns(words):
    turns, c = [], None
    for w in words:
        if c and c["speaker"] == w["speaker"]:
            c["text"] += w["word"]; c["end"] = w["end"]
        else:
            if c:
                turns.append(c)
            c = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": w["word"]}
    if c:
        turns.append(c)
    return turns
```

- [ ] **Step 4: Run, verify pass**

Run: `...python.exe -m pytest docker/nemo/tests/test_pipeline.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add docker/nemo/pipeline.py docker/nemo/tests/test_pipeline.py
git commit -m "feat(pipeline): speaker assignment, sentence smoothing, turn grouping"
```

---

## Task 3: Render + speaker-sample selection (pure, TDD)

**Files:**
- Modify: `docker/nemo/pipeline.py`
- Test: `docker/nemo/tests/test_pipeline.py`

- [ ] **Step 1: Add failing tests**

```python
# append to test_pipeline.py
from pipeline import render, pick_sample_turns, display_name

def test_display_name_maps_and_defaults():
    assert display_name("speaker_1", {"speaker_1": "Joe"}) == "Joe"
    assert display_name("speaker_2", {}) == "Speaker 2"
    assert display_name("__you__", {}, you_name="Graham") == "Graham"

def test_render_applies_names():
    turns = [{"speaker": "speaker_1", "start": 0.0, "end": 1.0, "text": " Hi"}]
    txt, data = render(turns, {"speaker_1": "Joe"})
    assert "Joe: Hi" in txt
    assert data["turns"][0]["speaker"] == "speaker_1"  # raw id preserved in json

def test_pick_sample_turns_returns_longest_per_speaker():
    turns = [{"speaker": "A", "start": 0, "end": 1, "text": "x"},
             {"speaker": "A", "start": 5, "end": 9, "text": "longer"},
             {"speaker": "B", "start": 9, "end": 11, "text": "y"}]
    picks = pick_sample_turns(turns)
    assert picks["A"]["start"] == 5 and picks["A"]["end"] == 9
    assert picks["B"]["start"] == 9
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement in `pipeline.py`**

```python
def display_name(spk, names, you_name="You"):
    if spk == "__you__":
        return names.get("__you__", you_name)
    return names.get(spk, spk.replace("speaker_", "Speaker "))

def _ts(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{s%60:05.2f}"

def render(turns, names, you_name="You", mode_label=""):
    lines = [f"[{_ts(t['start'])}] {display_name(t['speaker'], names, you_name)}: {t['text'].strip()}"
             for t in turns]
    header = f"# mode: {mode_label}\n\n" if mode_label else ""
    txt = header + "\n\n".join(lines)
    data = {"mode": mode_label, "turns": turns,
            "speakers": sorted({t["speaker"] for t in turns})}
    return txt, data

def pick_sample_turns(turns):
    """Longest turn per (non-you) speaker, for the play-clip button."""
    best = {}
    for t in turns:
        if t["speaker"] == "__you__":
            continue
        dur = t["end"] - t["start"]
        if t["speaker"] not in best or dur > (best[t["speaker"]]["end"] - best[t["speaker"]]["start"]):
            best[t["speaker"]] = t
    return best
```

- [ ] **Step 4: Run, verify pass**

- [ ] **Step 5: Commit**

```bash
git add docker/nemo/pipeline.py docker/nemo/tests/test_pipeline.py
git commit -m "feat(pipeline): render + sample-turn selection"
```

---

## Task 4: Model-bound helpers + channel extraction (port existing logic)

**Files:**
- Modify: `docker/nemo/pipeline.py`

These need the GPU/models, so they're validated in Task 9 (integration), not unit-tested. Port the proven code from `diarize_transcribe.py`.

- [ ] **Step 1: Add ffprobe + extraction + model functions to `pipeline.py`**

```python
def ffprobe_audio_streams(f):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=channels", "-of", "csv=p=0", f],
                         capture_output=True, text=True).stdout.strip()
    return [int(r) if r.strip().isdigit() else 1 for r in out.splitlines() if r.strip()]

def lr_diff_db(f):
    out = subprocess.run(["ffmpeg", "-i", f, "-af", "pan=mono|c0=c0-c1,volumedetect",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", out)
    return float(m.group(1)) if m else -99.0

def _extract(f, args, dst):
    subprocess.run(["ffmpeg", "-y", "-i", f] + args + ["-ac", "1", "-ar", "16000", dst],
                   check=True, capture_output=True)

def plan_channels(f, tmp, you_ch=0):
    """Returns (mode_label, [(role, wav)]). role in {'mono','you','far'}."""
    base = os.path.splitext(os.path.basename(f))[0]
    mode = decide_mode(ffprobe_audio_streams(f),
                       lr_diff_db(f) if ffprobe_audio_streams(f)[:1] == [2] else -99)
    you = os.path.join(tmp, base + ".you.wav")
    far = os.path.join(tmp, base + ".far.wav")
    mono = os.path.join(tmp, base + ".mono.wav")
    if mode == "dual_track":
        _extract(f, ["-map", f"0:a:{you_ch}"], you)
        _extract(f, ["-map", f"0:a:{1-you_ch}"], far)
        return "dual (separate tracks)", [("you", you), ("far", far)]
    if mode == "dual_stereo":
        _extract(f, ["-af", f"pan=mono|c0=c{you_ch}"], you)
        _extract(f, ["-af", f"pan=mono|c0=c{1-you_ch}"], far)
        return "dual (stereo L/R)", [("you", you), ("far", far)]
    _extract(f, [], mono)
    return "single mixed track", [("mono", mono)]

def load_whisper(model_name="large-v3", compute_type="float16"):
    import torch
    from faster_whisper import WhisperModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel(model_name, device=dev,
                        compute_type=compute_type if dev == "cuda" else "int8")

def transcribe(wm, wav, vad_filter=False):
    segs, info = wm.transcribe(wav, word_timestamps=True, vad_filter=vad_filter, beam_size=5)
    return [{"start": w.start, "end": w.end, "word": w.word}
            for s in segs for w in (s.words or [])]

def load_diarizer(model_name="nvidia/diar_streaming_sortformer_4spk-v2"):
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(model_name)
    m.eval()
    if torch.cuda.is_available():
        m = m.to("cuda")
    return m

def diarize(dm, wav):
    pred = dm.diarize(audio=[wav], batch_size=1)
    out = []
    for s in pred[0]:
        p = s.split()
        if len(p) >= 3:
            out.append((float(p[0]), float(p[1]), p[2]))
    return out

def extract_clip(src_wav, start, end, dst, pad=0.0):
    _extract(src_wav, ["-ss", str(max(0, start - pad)), "-to", str(end + pad)], dst)
    return dst
```

- [ ] **Step 2: Smoke-import in the container (no model download)**

Run: `docker run --rm -v "%cd%/docker/nemo/pipeline.py:/work/pipeline.py" nemo-sortformer python3 -c "import sys; sys.path.insert(0,'/work'); import pipeline; print('ok', pipeline.decide_mode([1,1], -99))"`
Expected: `ok dual_track`

- [ ] **Step 3: Commit**

```bash
git add docker/nemo/pipeline.py
git commit -m "feat(pipeline): ffprobe/channel-extract + model helpers"
```

---

## Task 5: Refactor the batch CLI onto `pipeline.py`

**Files:**
- Modify: `docker/nemo/diarize_transcribe.py`

- [ ] **Step 1: Rewrite `diarize_transcribe.py` as a thin batch driver**

```python
"""Batch CLI: every file in /in -> transcript in /out, source archived. Uses
pipeline.py for all logic. Env: WHISPER_MODEL, DIAR_MODEL, COMPUTE_TYPE,
VAD_FILTER, STAMP, YOU_CH, YOU_NAME, SPEAKER_NAMES."""
import os, glob, json, tempfile, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P

IN, OUT, ARCHIVE = "/in", "/out", "/archive"
EXTS = (".mp3",".wav",".m4a",".aac",".flac",".ogg",".opus",".wma",".mp4",".mov",".mkv",".webm",".avi",".m4v")
STAMP = os.environ.get("STAMP", "")
YOU_CH = int(os.environ.get("YOU_CH", "0"))
YOU_NAME = os.environ.get("YOU_NAME", "You")
VAD = os.environ.get("VAD_FILTER", "0") == "1"
names = {}
for pair in os.environ.get("SPEAKER_NAMES", "").split(","):
    if "=" in pair:
        k, v = pair.split("=", 1); names[k.strip()] = v.strip()

files = [f for f in sorted(glob.glob(os.path.join(IN, "*")))
         if os.path.isfile(f) and f.lower().endswith(EXTS)]
if not files:
    print(">> No audio/video in ./in"); raise SystemExit(0)

tmp = tempfile.mkdtemp()
plans = {}
for f in files:
    plans[f] = P.plan_channels(f, tmp, YOU_CH)
    print(f">> {os.path.basename(f)}: {plans[f][0]}", flush=True)

wm = P.load_whisper(os.environ.get("WHISPER_MODEL", "large-v3"),
                    os.environ.get("COMPUTE_TYPE", "float16"))
words = {}
for f in files:
    for role, wav in plans[f][1]:
        words[wav] = P.transcribe(wm, wav, VAD)
        print(f"   {os.path.basename(f)} [{role}]: {len(words[wav])} words", flush=True)
del wm
import torch; torch.cuda.empty_cache()

dm = P.load_diarizer(os.environ.get("DIAR_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2"))
dsegs = {}
for f in files:
    for role, wav in plans[f][1]:
        if role != "you":
            dsegs[wav] = P.diarize(dm, wav)
del dm; torch.cuda.empty_cache()

for f in files:
    mode_label, jobs = plans[f]
    allw = []
    for role, wav in jobs:
        ws = words[wav]
        if role == "you":
            for w in ws: w["speaker"] = "__you__"
        else:
            P.assign_speakers(ws, dsegs.get(wav, []))
            P.smooth_sentences(ws)
        allw.extend(ws)
    allw.sort(key=lambda w: w["start"])
    turns = P.build_turns(allw)
    base = os.path.splitext(os.path.basename(f))[0]
    od = os.path.join(OUT, base); os.makedirs(od, exist_ok=True)
    txt, data = P.render(turns, names, YOU_NAME, mode_label)
    open(os.path.join(od, base + ".txt"), "w", encoding="utf-8").write(txt)
    json.dump(data, open(os.path.join(od, base + ".json"), "w", encoding="utf-8"), indent=2)
    print(f">> wrote out/{base}/{base}.txt ({len(turns)} turns)", flush=True)
    if STAMP and os.path.isdir(ARCHIVE):
        dest = os.path.join(ARCHIVE, STAMP); os.makedirs(dest, exist_ok=True)
        tgt = os.path.join(dest, os.path.basename(f))
        if not os.path.exists(tgt):
            shutil.move(f, tgt)
shutil.rmtree(tmp, ignore_errors=True)
print(">> all done")
```

- [ ] **Step 2: Integration test the batch CLI on the slice**

Run:
```
copy docker\nemo\audio.wav in\itest.wav
docker run --rm --gpus all -e STAMP=itest -v "%cd%/in:/in" -v "%cd%/out:/out" -v "%cd%/archive:/archive" -v "%cd%/docker/nemo/pipeline.py:/work/pipeline.py" -v "%cd%/docker/nemo/diarize_transcribe.py:/work/diarize_transcribe.py" -v "nemo-cache:/root/.cache" nemo-sortformer
```
Expected: `out/itest/itest.txt` exists with `Speaker N:` turns; `archive/itest/itest.wav` exists; `in/` no longer has itest.wav.

- [ ] **Step 3: Commit**

```bash
git add docker/nemo/diarize_transcribe.py
git commit -m "refactor(cli): batch driver uses shared pipeline.py"
```

---

## Task 6: Add Gradio to the image

**Files:**
- Modify: `docker/nemo/Dockerfile`

- [ ] **Step 1: Add deps + COPY new files**

In `docker/nemo/Dockerfile`, after the `faster-whisper` line add:
```dockerfile
RUN python3 -m pip install gradio soundfile
```
Change the COPY line to:
```dockerfile
COPY sortformer_diar.py diarize_transcribe.py pipeline.py app.py /work/
```
(Leave ENTRYPOINT as the batch CLI; the app launcher overrides the command.)

- [ ] **Step 2: Build**

Run: `docker build -t nemo-sortformer docker/nemo`
Expected: build succeeds (NeMo/whisper layers cached; only gradio/soundfile layer is new).

- [ ] **Step 3: Commit**

```bash
git add docker/nemo/Dockerfile
git commit -m "build: add gradio + soundfile to image"
```

---

## Task 7: Gradio app — single-meeting tab

**Files:**
- Create: `docker/nemo/app.py`

The app keeps models loaded lazily and serialises jobs (one GPU). Variable speakers handled by 4 pre-built rows shown/hidden by count.

- [ ] **Step 1: Implement core app + single tab**

```python
"""Gradio app for transcribe + diarize + name. Runs inside the GPU container."""
import os, glob, json, tempfile, shutil, datetime
import gradio as gr
import pipeline as P

IN, OUT, ARCHIVE = "/in", "/out", "/archive"
MAX_SPK = 4
_W = {"wm": None, "dm": None}

def _whisper():
    if _W["wm"] is None:
        _W["wm"] = P.load_whisper(os.environ.get("WHISPER_MODEL", "large-v3"))
    return _W["wm"]

def process_file(path, you_ch):
    """Run the full pipeline on one file. Returns a result dict (no naming yet)."""
    tmp = tempfile.mkdtemp()
    mode_label, jobs = P.plan_channels(path, tmp, you_ch)
    wm = _whisper()
    words = {wav: P.transcribe(wm, wav) for _, wav in jobs}
    import torch
    if _W["dm"] is None:
        # free whisper first to fit 8GB, then load diarizer
        _W["wm"] = None; torch.cuda.empty_cache()
        _W["dm"] = P.load_diarizer(os.environ.get("DIAR_MODEL",
                    "nvidia/diar_streaming_sortformer_4spk-v2"))
    dm = _W["dm"]
    allw = []
    far_wav = None
    for role, wav in jobs:
        ws = words[wav]
        if role == "you":
            for w in ws: w["speaker"] = "__you__"
        else:
            far_wav = wav
            P.assign_speakers(ws, P.diarize(dm, wav))
            P.smooth_sentences(ws)
        allw.extend(ws)
    allw.sort(key=lambda w: w["start"])
    turns = P.build_turns(allw)
    samples = P.pick_sample_turns(turns)
    clip_dir = tempfile.mkdtemp()
    clips = {}
    for spk, t in samples.items():
        dst = os.path.join(clip_dir, f"{spk}.wav")
        clips[spk] = P.extract_clip(far_wav, t["start"], t["end"], dst)
    return {"path": path, "mode": mode_label, "turns": turns,
            "speakers": sorted(samples.keys()), "clips": clips, "tmp": tmp}

def save_named(result, names, you_name):
    base = os.path.splitext(os.path.basename(result["path"]))[0]
    od = os.path.join(OUT, base); os.makedirs(od, exist_ok=True)
    txt, data = P.render(result["turns"], names, you_name, result["mode"])
    open(os.path.join(od, base + ".txt"), "w", encoding="utf-8").write(txt)
    json.dump(data, open(os.path.join(od, base + ".json"), "w", encoding="utf-8"), indent=2)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if os.path.isdir(ARCHIVE) and os.path.exists(result["path"]):
        dest = os.path.join(ARCHIVE, stamp); os.makedirs(dest, exist_ok=True)
        tgt = os.path.join(dest, os.path.basename(result["path"]))
        if not os.path.exists(tgt):
            shutil.move(result["path"], tgt)
    shutil.rmtree(result.get("tmp", ""), ignore_errors=True)
    return txt

def build_ui():
    with gr.Blocks(title="Meeting Transcriber") as demo:
        gr.Markdown("# Meeting Transcriber\nTranscribe + diarize on your GPU. Name the voices, save.")
        with gr.Tab("Single meeting"):
            up = gr.File(label="Meeting file (stereo or multi-track)", file_count="single")
            you_ch = gr.Radio([0, 1], value=0, label="Which channel/track is YOU? (dual only)")
            you_name = gr.Textbox("You", label="Your name")
            go = gr.Button("Transcribe", variant="primary")
            state = gr.State()
            status = gr.Markdown(visible=False)
            rows, clip_c, name_c, spk_c = [], [], [], []
            for i in range(MAX_SPK):
                with gr.Row(visible=False) as r:
                    a = gr.Audio(label=f"Voice {i+1}", interactive=False)
                    n = gr.Textbox(label="Name this voice")
                    s = gr.State()
                rows.append(r); clip_c.append(a); name_c.append(n); spk_c.append(s)
            save = gr.Button("Save transcript", variant="primary", visible=False)
            out_txt = gr.Textbox(label="Transcript", lines=20, visible=False)

            def on_go(file, ych):
                res = process_file(file.name, ych)
                updates = []
                for i in range(MAX_SPK):
                    if i < len(res["speakers"]):
                        spk = res["speakers"][i]
                        updates += [gr.update(visible=True), gr.update(value=res["clips"][spk]),
                                    gr.update(value=""), spk]
                    else:
                        updates += [gr.update(visible=False), gr.update(),
                                    gr.update(), None]
                return [res, gr.update(value=f"Done — {res['mode']}, {len(res['speakers'])} other voice(s).", visible=True),
                        gr.update(visible=True)] + updates

            go.click(on_go, [up, you_ch],
                     [state, status, save] + sum([[rows[i], clip_c[i], name_c[i], spk_c[i]] for i in range(MAX_SPK)], []))

            def on_save(res, yname, *vals):
                names = {}
                for i in range(MAX_SPK):
                    nm = vals[i]; spk = vals[MAX_SPK + i]
                    if spk and nm and nm.strip():
                        names[spk] = nm.strip()
                txt = save_named(res, names, yname)
                return gr.update(value=txt, visible=True)

            save.click(on_save, [state, you_name] + name_c + spk_c, [out_txt])
    return demo

if __name__ == "__main__":
    build_ui().queue().launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: Rebuild image (picks up app.py via COPY)**

Run: `docker build -t nemo-sortformer docker/nemo`
Expected: success.

- [ ] **Step 3: Manual smoke (no GPU needed for UI to render)**

Run: `docker run --rm -p 7860:7860 -v "%cd%/docker/nemo/app.py:/work/app.py" -v "%cd%/docker/nemo/pipeline.py:/work/pipeline.py" nemo-sortformer python3 /work/app.py`
Open `http://localhost:7860`. Expected: the Single-meeting tab renders (upload, button). Ctrl-C to stop.

- [ ] **Step 4: Commit**

```bash
git add docker/nemo/app.py
git commit -m "feat(app): gradio single-meeting tab with naming"
```

---

## Task 8: Gradio app — batch tab

**Files:**
- Modify: `docker/nemo/app.py`

- [ ] **Step 1: Add a Batch tab to `build_ui()`** (inside the Blocks, after the Single tab)

```python
        with gr.Tab("Batch folder (in\\)"):
            gr.Markdown("Process every file in **in\\**. Then pick a meeting to name its voices.")
            run = gr.Button("Process all in in\\", variant="primary")
            picker = gr.Dropdown(label="Processed meetings", choices=[], visible=False)
            results = gr.State({})

            def on_run(ych):
                files = [f for f in sorted(glob.glob(os.path.join(IN, "*")))
                         if os.path.isfile(f)]
                done = {}
                for f in files:
                    try:
                        done[os.path.basename(f)] = process_file(f, ych)
                    except Exception as e:
                        print("skip", f, e)
                return done, gr.update(choices=list(done.keys()), visible=True)

            run.click(on_run, [you_ch], [results, picker])
```

Naming for the batch picker reuses the same 4-row mechanism: wire `picker.change` to populate the rows from `results[selected]` (same update shape as `on_go`), and `save` writes the selected result. (Implementer: factor the row-update builder from Task 7 `on_go` into a helper `rows_for(res)` and call it from both.)

- [ ] **Step 2: Refactor the row-update logic into `rows_for(res)`** and call it in both `on_go` and `picker.change`.

```python
def rows_for(res):
    u = []
    for i in range(MAX_SPK):
        if i < len(res["speakers"]):
            spk = res["speakers"][i]
            u += [gr.update(visible=True), gr.update(value=res["clips"][spk]), gr.update(value=""), spk]
        else:
            u += [gr.update(visible=False), gr.update(), gr.update(), None]
    return u
```

- [ ] **Step 3: Manual smoke**

Drop two short clips in `in\`, run the app, click "Process all", pick each from the dropdown, confirm voices + clips populate.

- [ ] **Step 4: Commit**

```bash
git add docker/nemo/app.py
git commit -m "feat(app): batch folder tab"
```

---

## Task 9: Launcher + end-to-end integration

**Files:**
- Create: `Transcribe-App.bat`

- [ ] **Step 1: Write the launcher**

```bat
@echo off
REM Double-click: starts the transcription web app, opens your browser.
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%in" mkdir "%ROOT%in"
if not exist "%ROOT%out" mkdir "%ROOT%out"
if not exist "%ROOT%archive" mkdir "%ROOT%archive"
docker image inspect nemo-sortformer:latest >nul 2>&1
if errorlevel 1 ( echo Building image once... & docker build -t nemo-sortformer "%ROOT%docker\nemo" )
start "" http://localhost:7860
echo App starting at http://localhost:7860  (close this window to stop)
docker run --rm --gpus all -p 7860:7860 ^
  -v "%ROOT%in:/in" -v "%ROOT%out:/out" -v "%ROOT%archive:/archive" ^
  -v "%ROOT%docker\nemo\app.py:/work/app.py" ^
  -v "%ROOT%docker\nemo\pipeline.py:/work/pipeline.py" ^
  -v "nemo-cache:/root/.cache" ^
  nemo-sortformer python3 /work/app.py
```

- [ ] **Step 2: End-to-end test (the real acceptance test)**

1. `copy docker\nemo\audio.wav in\meeting_test.wav`
2. Double-click `Transcribe-App.bat` (or run its docker line).
3. Browser → Single meeting → upload `in\meeting_test.wav` → Transcribe.
4. Expected: 2 voice rows appear with playable clips. Play each — they should be different voices (Joe / Shane).
5. Name them "Joe" and "Shane", Save.
6. Expected: `out\meeting_test\meeting_test.txt` shows `[..] Joe:` / `[..] Shane:` lines; source moved to `archive\<stamp>\`.

- [ ] **Step 3: Commit**

```bash
git add Transcribe-App.bat
git commit -m "feat: Transcribe-App.bat launcher"
```

---

## Task 10: Open-source — LICENSE + README

**Files:**
- Create: `LICENSE`
- Modify: `README.txt`

- [ ] **Step 1: Add MIT LICENSE**

Create `LICENSE` with the standard MIT text, `Copyright (c) 2026 Graham Morley`.

- [ ] **Step 2: Add a "License & Models" section to `README.txt`**

```
------------------------------------------------------------
LICENSE & MODEL NOTES
------------------------------------------------------------
This project's CODE is MIT licensed (see LICENSE) - free to use, modify,
redistribute, keep the copyright notice.

The AI MODELS it downloads at runtime have their OWN licenses, which you accept
when you use them:
  - OpenAI Whisper (large-v3): MIT
  - NVIDIA NeMo Sortformer diarizer: NVIDIA's model license - CHECK IT before any
    commercial redistribution.
  - pyannote (only in the older whisply path): MIT code, gated model terms.
Your MIT code license does not override the model licenses.
```

- [ ] **Step 3: Commit**

```bash
git add LICENSE README.txt
git commit -m "docs: MIT license + model-license notes"
```

---

## Self-Review Checklist (completed by plan author)

- **Spec coverage:** single + batch (Tasks 7,8); naming w/ play-clip (Tasks 3,7); which-channel-is-you (Task 7 `you_ch`); auto-detect multi-track/stereo/mono (Tasks 1,4); shared engine (Tasks 1-5); outputs + archive (Tasks 5,7); error skip (Task 8 try/except); MIT + model notes (Task 10). Enrollment intentionally absent (out of scope). ✓
- **Placeholders:** none — every code step has full code. The one "factor into helper" instruction (Task 8) includes the helper code. ✓
- **Type consistency:** `speaker` ids (`speaker_N` / `__you__`), `turn` dict shape `{speaker,start,end,text}`, `names` map, `clips` map — consistent across render/build_turns/pick_sample_turns/app. ✓
