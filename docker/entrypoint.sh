#!/usr/bin/env bash
# Transcribe everything in /in -> /out, on the GPU, with speaker labels.
set -euo pipefail

IN=/in
OUT=/out
MODEL="${WHISPER_MODEL:-${WHISPLY_MODEL:-large-v3-turbo}}"  # WHISPLY_MODEL: legacy alias

# Help ctranslate2 (faster-whisper) find torch's bundled cuDNN if the
# system one isn't picked up.
CUDNN_DIR="$(python3 -c 'import os,nvidia.cudnn.lib as l; print(os.path.dirname(l.__file__))' 2>/dev/null || true)"
if [ -n "$CUDNN_DIR" ]; then
  export LD_LIBRARY_PATH="$CUDNN_DIR:${LD_LIBRARY_PATH:-}"
fi

# Any audio/video to process? (top level of /in only - never the archive)
shopt -s nullglob nocaseglob
mapfile -t FILES < <(find "$IN" -maxdepth 1 -type f \( \
  -iname '*.mp3' -o -iname '*.wav' -o -iname '*.m4a' -o -iname '*.aac' -o \
  -iname '*.flac' -o -iname '*.ogg' -o -iname '*.opus' -o -iname '*.wma' -o \
  -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.webm' -o \
  -iname '*.avi' -o -iname '*.m4v' \) 2>/dev/null)
if [ "${#FILES[@]}" -eq 0 ]; then
  echo ">> No audio/video files found in ./in - drop files there and run again."
  exit 0
fi

# Speaker labels need a HuggingFace token (passed in as HF_TOKEN).
ANNOTATE=()
if [ -n "${HF_TOKEN:-}" ]; then
  echo ">> Speaker labels: ON"
  ANNOTATE=(-a -hf "$HF_TOKEN")
else
  echo ">> Speaker labels: OFF (no HF_TOKEN set). Transcribing only."
fi

echo ">> Transcribing ${#FILES[@]} file(s) on GPU with model '$MODEL'..."
whisply run -f "$IN" -o "$OUT" -d gpu -m "$MODEL" "${ANNOTATE[@]}" -e all
echo ">> Done. Results are in ./out"

# Archive source files so ./in is clear for next time, WITHOUT deleting anything.
# Only runs if whisply succeeded (set -e would have exited otherwise). Files are
# MOVED into archive/<timestamp>/. mv -n never overwrites. /archive is a separate
# mount, so archived files are never re-processed.
if [ -d /archive ]; then
  STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
  DEST="/archive/$STAMP"
  mkdir -p "$DEST"
  # whisply's *_converted.wav is a throwaway (regenerated from source) - delete it.
  find "$IN" -maxdepth 1 -type f -name '*_converted.wav' -delete
  # archive only the original source files.
  find "$IN" -maxdepth 1 -type f -exec mv -n -t "$DEST" {} +
  echo ">> Source files archived to ./archive/$STAMP (converted .wav discarded)"
else
  echo ">> NOTE: ./archive not mounted; left source files in ./in (nothing deleted)."
fi
