# NeMo Sortformer speaker diarization (overlap-aware, better on similar voices /
# crosstalk than pyannote). Linux base - NeMo installs cleanly here.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg git libsndfile1 sox build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "nemo_toolkit[asr]"

# Confirm the Sortformer model class is importable at build time.
RUN python3 -c "from nemo.collections.asr.models import SortformerEncLabelModel; print('NeMo Sortformer import OK')"

# Whisper (faster-whisper) for the transcription half of the pipeline.
# Separate layer so the heavy NeMo layer stays cached.
RUN python3 -m pip install faster-whisper

# Punctuation + truecasing restoration (Whisper leaves fast speech unpunctuated).
RUN python3 -m pip install deepmultilingualpunctuation

# Web app (Gradio) + audio I/O. Separate layer again to keep NeMo/whisper cached.
# Gradio pulls huggingface-hub>=1.0, but faster-whisper needs <1.0 - pin it back.
RUN python3 -m pip install gradio soundfile && \
    python3 -m pip install "huggingface-hub>=0.34.0,<1.0"

# Verify the whole stack imports together (catches dep conflicts at build time).
RUN python3 -c "import gradio, faster_whisper; from nemo.collections.asr.models import SortformerEncLabelModel; print('stack import OK')"

WORKDIR /work
COPY sortformer_diar.py diarize_transcribe.py pipeline.py app.py /work/
ENTRYPOINT ["python3", "/work/diarize_transcribe.py"]
