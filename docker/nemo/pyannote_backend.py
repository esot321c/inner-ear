"""pyannote diarizer backend — an alternative to Sortformer that produces a
GLOBAL, overlap-aware per-frame speaker matrix on the same 0.08s grid our
resolver/word-attribution already consumes.

Why the full Pipeline and not raw segmentation: pyannote's segmentation model
emits *chunk-local* speaker columns (speaker 2 in one window != speaker 2 in
another). Word attribution needs globally consistent identities, which only the
pipeline's embedding+clustering stage resolves. We run the pipeline, then rasterize
its (overlap-capable) Annotation onto our frame grid. Overlapping segments set two
columns in the same frame, so genuine overlap survives — the whole point of testing
pyannote.

Models are gated; needs HF_TOKEN + accepted terms for pyannote/speaker-diarization-3.1
and pyannote/segmentation-3.0. Cached under /root/.cache after first download.
"""
import os
import sys
import types
import numpy as np

# --- compat shims so the 2023-era pyannote 3.1.1 imports under our modern stack ---
# (1) torchaudio >=2.1 removed the global backend setter; modern torchaudio
#     dispatches per-call, so stub it as a no-op.
import torchaudio as _ta
if not hasattr(_ta, "set_audio_backend"):
    _ta.set_audio_backend = lambda *a, **k: None
if not hasattr(_ta, "get_audio_backend"):
    _ta.get_audio_backend = lambda *a, **k: "soundfile"
# (1b) torchaudio.backend submodule was removed; recreate the AudioMetaData path.
if "torchaudio.backend.common" not in sys.modules:
    _meta = getattr(_ta, "AudioMetaData", None)
    if _meta is None:
        try:
            from torchaudio import AudioMetaData as _meta
        except Exception:
            class _meta:  # last-resort placeholder
                pass
    _backend = types.ModuleType("torchaudio.backend")
    _common = types.ModuleType("torchaudio.backend.common")
    _common.AudioMetaData = _meta
    _backend.common = _common
    sys.modules["torchaudio.backend"] = _backend
    sys.modules["torchaudio.backend.common"] = _common
    _ta.backend = _backend
# (1c) torch >=2.6 flipped torch.load to weights_only=True; pyannote's Lightning
#      checkpoint carries non-allowlisted globals (TorchVersion, omegaconf...).
#      Force weights_only=False — the weights come from the trusted pyannote repo.
import torch as _torch
if not getattr(_torch.load, "_pyannote_patched", False):
    _orig_load = _torch.load

    def _patched_load(*a, **k):
        k["weights_only"] = False          # force-override Lightning's explicit True
        return _orig_load(*a, **k)
    _patched_load._pyannote_patched = True
    _torch.load = _patched_load
# (2) NumPy 2.0 removed deprecated aliases pyannote references at import time.
for _name, _val in {"NaN": np.nan, "NAN": np.nan, "Inf": np.inf, "Infinity": np.inf,
                    "PINF": np.inf, "NINF": -np.inf, "float_": np.float64,
                    "int_": np.int64, "bool_": np.bool_, "complex_": np.complex128,
                    "object_": np.object_, "str_": np.str_, "unicode_": np.str_}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _val)

FRAME_DUR = 0.08  # match Sortformer's grid so downstream code is identical

_PIPELINE = None


def load_pyannote(model_name="pyannote/speaker-diarization-3.1", token=None):
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    import torch
    from pyannote.audio import Pipeline
    token = token or os.environ.get("HF_TOKEN")
    pipe = Pipeline.from_pretrained(model_name, use_auth_token=token)
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    _PIPELINE = pipe
    return pipe


def diarize_soft_pyannote(pipe, wav, n_spk=4, frame_dur=FRAME_DUR):
    """Run pyannote and rasterize the global diarization onto a (frames, n_spk)
    binary matrix at frame_dur. Speaker labels are mapped to columns in order of
    first appearance; overlap-capable. Returns (post, frame_dur).

    We decode the audio ourselves (torchaudio 2.11 routes file loading through
    torchcodec, which we don't install) and pass pyannote an in-memory waveform."""
    import torch
    from faster_whisper.audio import decode_audio
    audio = decode_audio(wav)                    # 16 kHz mono float32
    wf = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)  # (1, samples)
    ann = pipe({"waveform": wf, "sample_rate": 16000})
    # collect (start, end, speaker) and assign global columns by first appearance
    tracks = list(ann.itertracks(yield_label=True))
    if not tracks:
        return np.zeros((1, n_spk), dtype=np.float32), frame_dur
    order, col = [], {}
    for seg, _trk, spk in tracks:
        if spk not in col:
            col[spk] = len(order)
            order.append(spk)
    end_t = max(seg.end for seg, _t, _s in tracks)
    T = max(1, int(np.ceil(end_t / frame_dur)))
    post = np.zeros((T, n_spk), dtype=np.float32)
    for seg, _trk, spk in tracks:
        c = col[spk]
        if c >= n_spk:
            continue                       # more speakers than columns: drop extras
        a = max(0, int(seg.start / frame_dur))
        b = min(T, int(np.ceil(seg.end / frame_dur)))
        post[a:b, c] = 1.0
    return post, frame_dur
