"""DiCoW (Diarization-Conditioned Whisper) backend — Tier 3.

Fuses ASR + speaker attribution: instead of transcribe-then-attribute, it
conditions Whisper on a per-speaker diarization mask and decodes one transcript
per target speaker, so overlapping speech is handled inside the model.

Consumes the SAME (frames, n_spk) posterior our Sortformer path produces, builds
DiCoW's 4-channel STNO mask [silence, target, non-target, overlap] at the 50 fps
encoder grid, and runs the model once per active speaker. Returns a flat word
list [{word,start,end,speaker}] so the existing scorer can grade it.

Recipe (model API, get_stno_mask, 50fps grid, per-speaker batching) extracted
verbatim from BUTSpeechFIT/DiCoW. Model BUT-FIT/DiCoW_v3_3 (large-v3-turbo,
use_enrollments=false). NOTE: repo pins transformers==4.55.0 / torch==2.5.1; we
run newer — generate() compatibility is verified empirically.
"""
import re
import numpy as np
import torch

ENC_FPS = 50              # encoder frames/sec (0.02s) — mel_len // 2
WIN_ENC = 30 * ENC_FPS    # 1500 encoder frames per 30s window
_MODEL = _FE = _TOK = None


def load_dicow(model_id="BUT-FIT/DiCoW_v3_3", fp16=True):
    global _MODEL, _FE, _TOK
    if _MODEL is not None:
        return _MODEL, _FE, _TOK
    from transformers import (AutoModelForSpeechSeq2Seq, AutoFeatureExtractor,
                              AutoTokenizer)
    dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32
    m = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype)
    m = m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    m.set_tokenizer(tok)                       # mandatory — builds SoftLabelCreator
    _MODEL, _FE, _TOK = m, fe, tok
    return m, fe, tok


def active_speakers(post, thr=0.5, min_secs=1.0, frame_dur=0.02):
    """Columns that are active for at least min_secs total — avoids running DiCoW
    for phantom speakers the diarizer barely lit up."""
    on = (post >= thr).sum(axis=0) * frame_dur
    return [c for c in range(post.shape[1]) if on[c] >= min_secs]


def _diar_mask(post, cols, n_enc_frames, thr=0.5):
    """Resample the (frames, n_spk) posterior to the 50fps encoder grid and
    binarize → (len(cols), n_enc_frames)."""
    f_in = post.shape[0]
    idx = np.minimum((np.arange(n_enc_frames) * f_in / max(1, n_enc_frames)).astype(int),
                     f_in - 1)
    return torch.from_numpy((post[idx][:, cols] >= thr).T.astype(np.float32))


def get_stno_mask(diar_mask, s_index):
    """[silence, target, non-target, overlap] for speaker s_index. Verbatim from
    BUTSpeechFIT/DiCoW pipeline.py."""
    non_target = torch.ones((diar_mask.shape[0],), dtype=torch.bool)
    non_target[s_index] = False
    sil = (1 - diar_mask).prod(axis=0)
    anyone_else = (1 - diar_mask[non_target]).prod(axis=0)
    target = diar_mask[s_index] * anyone_else
    non_target_spk = (1 - diar_mask[s_index]) * (1 - anyone_else)
    overlap = diar_mask[s_index] - target
    return torch.stack([sil, target, non_target_spk, overlap], axis=0)


_TS = re.compile(r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>", flags=re.S)


def _seg_words(text, win_start, speaker):
    """Split a decoded '<|t|>words<|t|>' string into per-word dicts with linearly
    interpolated timestamps."""
    out = []
    for a, body, b in _TS.findall(text):
        st, en = win_start + float(a), win_start + float(b)
        toks = body.split()
        if not toks:
            continue
        dur = max(en - st, 1e-3)
        for k, w in enumerate(toks):
            ws = st + dur * k / len(toks)
            we = st + dur * (k + 1) / len(toks)
            out.append({"word": " " + w, "start": ws, "end": we, "speaker": speaker})
    return out


def transcribe_dicow(audio, post, frame_dur_post=0.08, thr=0.5):
    """audio: 16k mono float32 np. post: (frames, n_spk) diarization posterior.
    Returns flat word list with speaker_{col} labels."""
    m, fe, tok = load_dicow()
    cols = active_speakers(post, thr=thr, frame_dur=frame_dur_post) or list(range(post.shape[1]))
    total_enc = int(np.ceil(len(audio) / 16000 * ENC_FPS))
    diar = _diar_mask(post, cols, total_enc, thr=thr)            # (n_active, total_enc)
    words = []
    for w0 in range(0, total_enc, WIN_ENC):
        w1 = min(w0 + WIN_ENC, total_enc)
        s0, s1 = int(w0 / ENC_FPS * 16000), int(w1 / ENC_FPS * 16000)
        chunk = audio[s0:s1]
        if len(chunk) < 16000 * 0.1:
            continue
        feats = fe(chunk, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(m.device, dtype=m.dtype)
        enc_T = feats.shape[-1] // 2
        win_diar = torch.zeros(len(cols), enc_T)
        seg = diar[:, w0:w1]
        win_diar[:, :seg.shape[1]] = seg
        stno = torch.stack([get_stno_mask(win_diar, i) for i in range(len(cols))], dim=0)
        stno = stno.to(feats.device, dtype=feats.dtype)
        inp = feats.repeat(len(cols), 1, 1)
        attn = torch.ones(inp.shape[0], inp.shape[2], dtype=torch.bool, device=feats.device)
        with torch.no_grad():
            tokens = m.generate(input_features=inp, attention_mask=attn,
                                stno_mask=stno, return_timestamps=True)
        texts = tok.batch_decode(tokens, decode_with_timestamps=True, skip_special_tokens=True)
        for ci, text in enumerate(texts):
            words.extend(_seg_words(text, w0 / ENC_FPS, f"speaker_{cols[ci]}"))
    words.sort(key=lambda x: x["start"])
    return words
