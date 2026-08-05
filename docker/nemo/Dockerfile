# Inner Ear — local meeting transcriber + speaker diarization.
#   - faster-whisper (large-v3) ASR
#   - NeMo Sortformer diarization
#   - DiCoW (diarization-conditioned Whisper) auto-escalation for 3+ speakers
#   - pyannote available as a research backend
# Reproducible: the full validated dependency set is pinned in requirements.lock.txt
# (torch 2.12.1+cu130 / NeMo 2.7.3 / pyannote 3.1.1 / transformers 4.57.6, ...).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_VERBOSITY=error

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg git libsndfile1 sox build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Heavy, pinned dependency layer — copied before code so it stays cached across
# code edits. The lock is a complete frozen closure (every transitive dep pinned),
# so install it with --no-deps: faster-whisper and gradio declare contradictory
# huggingface-hub ranges that a fresh resolver rejects, but the single pinned
# version they actually run on is in the lock. --no-deps reproduces it exactly.
# torch 2.12.1 pulls its bundled CUDA 13 wheels by default.
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN python3 -m pip install --no-deps -r /tmp/requirements.lock.txt

# Fail the build now if the stack doesn't import together.
RUN python3 -c "import gradio, faster_whisper, transformers, torch, torchaudio; \
from nemo.collections.asr.models import SortformerEncLabelModel; \
print('stack import OK — torch', torch.__version__)"

WORKDIR /work
COPY pipeline.py app.py dicow_backend.py scoring.py pyannote_backend.py stagebench_app.py /work/

# Models (Whisper, Sortformer, DiCoW) download on first run into the mounted HF
# cache, not baked — keeps the image lean. Production UI on :7860; override CMD
# with /work/stagebench_app.py to run the experiment bench.
EXPOSE 7860
ENTRYPOINT ["python3"]
CMD ["/work/app.py"]
