# Local Whisper transcription + speaker diarization, GPU-accelerated.
# Linux base => none of the Windows-specific workarounds are needed
# (symlinks work, speechbrain's path bug doesn't trigger, pip ships CUDA torch).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Python 3.10 + ffmpeg + git (some deps build from git).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# On Linux, `pip install torch` is the CUDA build by default - no CPU-trap.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "whisply[app]"

# Sanity-check at build time that CUDA torch + ctranslate2 GPU are wired up.
RUN python3 -c "import torch, ctranslate2; assert torch.version.cuda, 'CPU torch!'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ct2', ctranslate2.__version__)"

WORKDIR /work
COPY entrypoint.sh /usr/local/bin/transcribe
# Strip any CRLF (file may be authored on Windows) and make executable.
RUN sed -i 's/\r$//' /usr/local/bin/transcribe && chmod +x /usr/local/bin/transcribe

ENTRYPOINT ["/usr/local/bin/transcribe"]
