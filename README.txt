============================================================
  LOCAL TRANSCRIPTION + SPEAKER LABELS  (Whisper, on your GPU)
============================================================

Everything runs locally on your RTX 3070. Nothing is uploaded.
Engine: Whisply -> faster-whisper (transcription) + WhisperX/pyannote (speakers).
Model: large-v3-turbo (chosen to fit your 8GB card comfortably).

************************************************************
  RECOMMENDED: the Meeting Transcriber app
************************************************************
The best tool here is now the web app. Double-click:

    Transcribe-App.bat

It opens http://localhost:7860 in your browser. There you can:
  - Upload a meeting (single mode) or process the whole in\ folder (batch).
  - It transcribes (Whisper large-v3) + separates speakers (NeMo Sortformer).
  - It plays a clip of each detected voice and lets you NAME them.
  - Save -> labeled transcript in out\, source moved to archive\.

Two-channel meetings (your mic + the call) are auto-detected and split, so
your own voice is labeled perfectly and the far side is diarized.
Best capture: OBS multi-track (Mic -> Track 1, Desktop -> Track 2).

The folder-only tools below (Transcribe-Folder.bat etc.) still work, but the
app is the one to use.


------------------------------------------------------------
EVERYDAY USE  (folder -> folder)
------------------------------------------------------------
1. Drop your audio/video files into the   in   folder.
2. Double-click            Transcribe-Folder.bat
3. Results appear in the   out   folder.

You get, per file: .txt, .json, .srt, .vtt, .rttm and an .html viewer.
Speaker turns are labelled (SPEAKER_00, SPEAKER_01, ...).

Supported inputs: mp3, wav, m4a, aac, flac, ogg, opus, wma,
                  mp4, mov, mkv, webm, avi, m4v.

------------------------------------------------------------
DRAG-AND-DROP GUI  (optional)
------------------------------------------------------------
Double-click   Whisply-App.bat   -> opens a page in your browser
where you can drop files and pick options. Close the black window
when you're done to shut it down.

------------------------------------------------------------
ONE-TIME SETUP FOR SPEAKER LABELS  (required, free)
------------------------------------------------------------
Speaker separation uses pyannote models that need a free
HuggingFace token. Without it you still get a transcript, just
no "who said what".

  1. Make a free account at https://huggingface.co/join
  2. Accept the model terms (click "Agree" on each page while
     logged in):
        https://huggingface.co/pyannote/speaker-diarization-3.1
        https://huggingface.co/pyannote/segmentation-3.0
  3. Create a token (type: Read):
        https://huggingface.co/settings/tokens
  4. Copy   .env.example   to   .env   and set   HF_TOKEN=<your token>
     (the old hf_token.txt still works as a fallback).

That's it. Run Transcribe-Folder.bat again and speakers will be labelled.
NOTE: the new app (Transcribe-App.bat) needs NO token at all.

------------------------------------------------------------
NOTES
------------------------------------------------------------
- First run downloads the models (~2-3 GB total) once; after that
  it's offline and fast.
- Speed: roughly 5-15x faster than real time on your 3070
  (a 1-hour recording in a few minutes).
- "Not dropping anything": large-v3-turbo + the default voice
  detection captures speech well. If a specific recording is very
  quiet or has long silences and you suspect missed words, tell
  Graham/Claude - we can switch that job to the full large-v3
  model or tune the voice-detection threshold.
- To label a known number of speakers (more accurate), use the GUI
  and set the speaker count, or ask Claude to add -num N.

------------------------------------------------------------
WHAT'S INSTALLED  (for reference / if something breaks)
------------------------------------------------------------
- Python 3.12 virtual env:   .venv
- ffmpeg/ffprobe:            bin
- Settings:                  config.json
- GPU: torch 2.8 + CUDA 12.8, ctranslate2 (faster-whisper backend)

------------------------------------------------------------
LICENSE & MODEL NOTES
------------------------------------------------------------
This project's CODE is MIT licensed (see LICENSE) - free to use, modify, and
redistribute; keep the copyright notice.

The AI MODELS it downloads at runtime have their OWN licenses, which you accept
when you use them:
  - OpenAI Whisper (large-v3): MIT
  - NVIDIA NeMo Sortformer diarizer: NVIDIA model license - CHECK IT before any
    commercial redistribution.
  - pyannote (older whisply path only): MIT code, gated model terms.
Your MIT code license does not override the model licenses.
============================================================
