import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import (decide_mode, assign_speakers, smooth_sentences, build_turns,
                      render, pick_sample_turns, display_name,
                      group_sentences, resolve_speakers, fix_onset_leaks)


def _wp(probs, sent_start=False):
    return {"speaker_probs": probs, "sent_start": sent_start}


def _w(start, end, word, speaker=None):
    return {"start": start, "end": end, "word": word, "speaker": speaker}


# --- decide_mode ---
def test_two_tracks_is_dual_track():
    assert decide_mode(streams=[1, 1], lr_diff_db=-99) == "dual_track"

def test_stereo_with_difference_is_dual_stereo():
    assert decide_mode(streams=[2], lr_diff_db=-20) == "dual_stereo"

def test_dual_mono_is_single():
    assert decide_mode(streams=[2], lr_diff_db=-59) == "single"

def test_mono_is_single():
    assert decide_mode(streams=[1], lr_diff_db=-99) == "single"


# --- plan_channels roles (no you/far) ---
def test_plan_channels_signature_has_no_you_ch():
    import inspect
    from pipeline import plan_channels
    params = list(inspect.signature(plan_channels).parameters)
    assert params == ["f", "tmp"], params


# --- assign / smooth / turns ---
def test_assign_speakers_by_midpoint():
    words = [_w(0.0, 1.0, " Hi"), _w(1.0, 2.0, " there")]
    segs = [(0.0, 0.9, "speaker_0"), (0.9, 2.0, "speaker_1")]
    assign_speakers(words, segs)
    assert words[0]["speaker"] == "speaker_0"
    assert words[1]["speaker"] == "speaker_1"

def test_smooth_despeckles_isolated_blip():
    # A lone short word tagged speaker_1 between speaker_0 neighbours, no pauses:
    # diarizer jitter -> snap it back to speaker_0.
    words = [_w(0.0, 0.5, " hello there", "speaker_0"),
             _w(0.5, 0.7, " um", "speaker_1"),       # 0.2s, no gaps
             _w(0.7, 1.2, " how are you", "speaker_0")]
    smooth_sentences(words)
    assert [w["speaker"] for w in words] == ["speaker_0"] * 3

def test_smooth_preserves_rapid_back_and_forth():
    # Real (not blip-length) alternating turns must NOT be flattened to one
    # speaker - this is the fast-conversation case that dominance-smoothing broke.
    words = [_w(0.0, 1.0, " long thing from me", "speaker_0"),
             _w(1.0, 2.0, " a real reply", "speaker_1"),
             _w(2.0, 3.0, " back to me now", "speaker_0")]
    smooth_sentences(words)
    assert [w["speaker"] for w in words] == ["speaker_0", "speaker_1", "speaker_0"]

def test_smooth_keeps_short_word_with_a_real_pause():
    # A short word IS its own turn when separated by a pause - don't despeckle it.
    words = [_w(0.0, 1.0, " talking here", "speaker_0"),
             _w(2.0, 2.3, " yeah", "speaker_1"),      # 1s pause before it
             _w(3.5, 4.5, " continuing on", "speaker_0")]
    smooth_sentences(words)
    assert words[1]["speaker"] == "speaker_1"

def test_build_turns_groups_consecutive():
    words = [_w(0.0, 1.0, " Hi", "A"), _w(1.0, 2.0, " there", "A"),
             _w(2.0, 3.0, " Yo", "B")]
    turns = build_turns(words)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "A" and turns[0]["text"].strip() == "Hi there"
    assert turns[1]["speaker"] == "B"


# --- build_turns_overlap ---
def test_build_turns_overlap_breaks_on_interjection():
    from pipeline import build_turns_overlap
    # A speaks, B interjects, A resumes 0.3s later. Must be 3 turns, not a merged A.
    words = [
        {"start": 0.0, "end": 1.0, "word": " so I think", "speaker": "A"},
        {"start": 1.1, "end": 1.4, "word": " yeah", "speaker": "B"},
        {"start": 1.5, "end": 2.5, "word": " we should go", "speaker": "A"},
    ]
    turns = build_turns_overlap(words)
    assert [t["speaker"] for t in turns] == ["A", "B", "A"], turns
    assert turns[0]["text"].strip() == "so I think"
    assert turns[2]["text"].strip() == "we should go"

def test_build_turns_overlap_still_merges_uninterrupted():
    from pipeline import build_turns_overlap
    # Same speaker, small gaps, NO other speaker between -> one merged turn.
    words = [
        {"start": 0.0, "end": 1.0, "word": " hello", "speaker": "A"},
        {"start": 1.2, "end": 2.0, "word": " there", "speaker": "A"},
        {"start": 2.1, "end": 3.0, "word": " friend", "speaker": "A"},
    ]
    turns = build_turns_overlap(words)
    assert len(turns) == 1
    assert turns[0]["text"].strip() == "hello there friend"

def test_build_turns_overlap_keeps_true_simultaneous_whole():
    from pipeline import build_turns_overlap
    # Two speakers genuinely overlapping in time: each stays whole (not shattered),
    # ordered by start. A's words are contiguous with no B word BETWEEN consecutive
    # A words (B starts after A's span), so A stays one turn.
    words = [
        {"start": 0.0, "end": 0.5, "word": " one", "speaker": "A"},
        {"start": 0.5, "end": 1.0, "word": " two", "speaker": "A"},
        {"start": 1.2, "end": 2.5, "word": " meanwhile B talks", "speaker": "B"},
    ]
    turns = build_turns_overlap(words)
    assert [t["speaker"] for t in turns] == ["A", "B"]
    assert turns[0]["text"].strip() == "one two"


# --- display_name / render without __you__ ---
def test_display_name_humanizes_namespaced_id():
    assert display_name("t0:speaker_1", {}) == "Speaker 1"

def test_display_name_uses_provided_name():
    assert display_name("t1:speaker_0", {"t1:speaker_0": "Ezra"}) == "Ezra"

def test_render_names_every_speaker():
    turns = [{"speaker": "t0:speaker_0", "start": 0.0, "end": 1.0, "text": "hi"},
             {"speaker": "t1:speaker_0", "start": 1.0, "end": 2.0, "text": "yo"}]
    txt, data = render(turns, {"t0:speaker_0": "Me", "t1:speaker_0": "Ezra"}, "dual")
    assert "Me: hi" in txt and "Ezra: yo" in txt
    assert data["mode"] == "dual"

# --- render / samples / names ---
def test_display_name_maps_and_defaults():
    assert display_name("speaker_1", {"speaker_1": "Joe"}) == "Joe"
    assert display_name("speaker_2", {}) == "Speaker 2"

def test_render_applies_names():
    turns = [{"speaker": "speaker_1", "start": 0.0, "end": 1.0, "text": " Hi"}]
    txt, data = render(turns, {"speaker_1": "Joe"})
    assert "Joe: Hi" in txt
    assert data["turns"][0]["speaker"] == "speaker_1"

def test_pick_sample_turns_returns_longest_per_speaker():
    turns = [{"speaker": "A", "start": 0, "end": 1, "text": "x"},
             {"speaker": "A", "start": 5, "end": 9, "text": "longer"},
             {"speaker": "B", "start": 9, "end": 11, "text": "y"}]
    picks = pick_sample_turns(turns)
    assert picks["A"]["start"] == 5 and picks["A"]["end"] == 9
    assert picks["B"]["start"] == 9


# --- pick_solo_windows ---
def test_pick_solo_windows_finds_clean_stretch():
    import numpy as np
    from pipeline import pick_solo_windows
    # 20s @ 0.5s frames = 40 frames, 2 speakers.
    fd = 0.5
    frames = 40
    post = np.zeros((frames, 2), dtype=float)
    # spk0 solo in first half, spk1 solo in second half, a little overlap at the seam.
    post[0:20, 0] = 0.9
    post[20:40, 1] = 0.9
    post[19:21, :] = 0.9  # overlap seam - should be avoided
    win = pick_solo_windows(post, fd, spk_cols=[0, 1], clip_secs=6.0)
    (s0, e0), (s1, e1) = win[0], win[1]
    # spk0 window lands in its solo first half, away from the seam at 10s
    assert e0 <= 10.0 + 1e-6
    # spk1 window lands in its solo second half, away from the seam
    assert s1 >= 10.0 - 1e-6
    assert abs((e0 - s0) - 6.0) < fd
    assert abs((e1 - s1) - 6.0) < fd


# --- soft-posterior resolver pipeline ---
def test_group_sentences_breaks_on_punctuation_and_pause():
    words = [_w(0.0, 0.5, " Hi."), _w(0.6, 1.0, " There"), _w(3.0, 3.5, " Later")]
    group_sentences(words)
    assert [w["sent_id"] for w in words] == [1, 2, 3]
    assert all(w["sent_start"] for w in words)

def test_group_sentences_keeps_unpunctuated_run_together():
    words = [_w(0.0, 0.4, " hey"), _w(0.4, 0.8, " there"), _w(0.8, 1.2, " you")]
    group_sentences(words)
    assert [w["sent_id"] for w in words] == [1, 1, 1]

def test_resolve_follows_posteriors():
    words = [_wp([0.9, 0.1], True), _wp([0.1, 0.9], True)]
    resolve_speakers(words, n_spk=2)
    assert [w["speaker"] for w in words] == ["speaker_0", "speaker_1"]

def test_resolve_keeps_speaker_through_weak_midsentence_dip():
    # a brief weak dip toward spk1 mid-sentence must not flip the speaker
    words = [_wp([0.9, 0.1], True), _wp([0.45, 0.55], False), _wp([0.9, 0.1], False)]
    resolve_speakers(words, n_spk=2, switch_mid=3.0, switch_boundary=0.4)
    assert [w["speaker"] for w in words] == ["speaker_0"] * 3

def test_resolve_switches_for_interjection_at_sentence_start():
    # a real interjection begins its own sentence (pause before it); the cheap
    # sentence-boundary switch lets a strong posterior flip it
    words = [_wp([0.95, 0.05], True), _wp([0.02, 0.98], True), _wp([0.95, 0.05], True)]
    resolve_speakers(words, n_spk=2, switch_mid=3.0, switch_boundary=0.4)
    assert words[1]["speaker"] == "speaker_1"

def test_resolve_does_not_fragment_on_midsentence_blip():
    # a strong but isolated MID-sentence frame stays with the turn's speaker -
    # this is what stops fast crosstalk (the podcast) from fragmenting
    words = [_wp([0.95, 0.05], True), _wp([0.05, 0.95], False), _wp([0.95, 0.05], False)]
    resolve_speakers(words, n_spk=2, switch_mid=3.0, switch_boundary=0.4)
    assert [w["speaker"] for w in words] == ["speaker_0"] * 3

def test_fix_onset_moves_trailing_opener_to_next_speaker():
    words = [{"word": " make", "speaker": "speaker_0"},
             {"word": " sense", "speaker": "speaker_0"},
             {"word": " did", "speaker": "speaker_0"},   # stranded opener
             {"word": " you", "speaker": "speaker_1"},
             {"word": " send", "speaker": "speaker_1"}]
    fix_onset_leaks(words)
    assert words[2]["speaker"] == "speaker_1"   # "did" moved to Graham
    assert words[1]["speaker"] == "speaker_0"   # "sense" stays

def test_fix_onset_keeps_lone_interjection():
    words = [{"word": " much", "speaker": "speaker_1"},
             {"word": " what", "speaker": "speaker_0"},   # its own one-word turn
             {"word": " how", "speaker": "speaker_1"}]
    fix_onset_leaks(words)
    assert words[1]["speaker"] == "speaker_0"

def test_fix_onset_leaves_turn_initial_opener():
    words = [{"word": " okay", "speaker": "speaker_0"},
             {"word": " i", "speaker": "speaker_1"},      # starts speaker_1's turn
             {"word": " get", "speaker": "speaker_1"}]
    fix_onset_leaks(words)
    assert words[1]["speaker"] == "speaker_1"

def test_truecase_capitalizes_sentence_starts_and_I():
    from pipeline import _truecase
    words = [{"word": " hello."}, {"word": " i"}, {"word": " said"}, {"word": " yes."}]
    _truecase(words)
    assert [w["word"] for w in words] == [" Hello.", " I", " said", " yes."]


# --- LLM boundary reattribution (LLM mocked; tests the guards, not the model) ---
def _wsp(word, speaker):
    return {"word": word, "speaker": speaker}

def test_llm_cleanup_moves_opener_on_yes(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, "_llm_chat", lambda *a, **k: "YES")
    words = [_wsp(" make", "speaker_0"), _wsp(" sense", "speaker_0"),
             _wsp(" did", "speaker_0"),   # turn-final, no terminal punctuation
             _wsp(" you", "speaker_1"), _wsp(" go", "speaker_1")]
    moved = pipeline.llm_cleanup(words, "http://x/v1", "m")
    assert words[2]["speaker"] == "speaker_1" and moved == 1

def test_llm_cleanup_never_moves_sentence_ender(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, "_llm_chat", lambda *a, **k: "YES")  # even on YES
    words = [_wsp(" how", "speaker_0"), _wsp(" much", "speaker_0"),
             _wsp(" coupon?", "speaker_0"),   # ends a sentence -> can't start the next
             _wsp(" what", "speaker_1")]
    pipeline.llm_cleanup(words, "http://x/v1", "m")
    assert words[2]["speaker"] == "speaker_0"

def test_llm_cleanup_keeps_single_word_turn(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, "_llm_chat", lambda *a, **k: "YES")
    words = [_wsp(" much", "speaker_1"),
             _wsp(" what", "speaker_0"),   # lone interjection between speaker_1
             _wsp(" how", "speaker_1")]
    pipeline.llm_cleanup(words, "http://x/v1", "m")
    assert words[1]["speaker"] == "speaker_0"

def test_llm_cleanup_respects_no(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, "_llm_chat", lambda *a, **k: "NO")
    words = [_wsp(" yes", "speaker_0"), _wsp(" sir", "speaker_0"),
             _wsp(" really", "speaker_0"), _wsp(" okay", "speaker_1")]
    moved = pipeline.llm_cleanup(words, "http://x/v1", "m")
    assert moved == 0 and words[2]["speaker"] == "speaker_0"


# --- namespace_speakers ---
def test_namespace_speakers_prefixes_track_index():
    from pipeline import namespace_speakers
    words = [_w(0, 1, "hi", "speaker_0"), _w(1, 2, "yo", "speaker_1")]
    namespace_speakers(words, 2)
    assert [w["speaker"] for w in words] == ["t2:speaker_0", "t2:speaker_1"]

def test_namespace_speakers_is_idempotent():
    from pipeline import namespace_speakers
    words = [_w(0, 1, "hi", "speaker_0")]
    namespace_speakers(words, 0)
    namespace_speakers(words, 0)
    assert words[0]["speaker"] == "t0:speaker_0"


# --- suppress_cross_track_bleed ---
import numpy as np
from pipeline import suppress_cross_track_bleed

SR = 16000

def _tone(dur, amp):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype("float32")

def _mk(words, audio):
    return {"words": words, "wav": "ignored", "audio": audio}

def test_bleed_faint_same_text_word_is_dropped():
    # Track A (loud) and track B (faint copy) both have the word "hello" at 1.0-1.4s.
    audioA = np.concatenate([_tone(1.0, 0.0), _tone(0.4, 0.5), _tone(0.6, 0.0)])   # loud burst 1.0-1.4
    audioB = audioA * 0.05                                                          # ~26 dB quieter
    A = _mk([{"word": " hello", "start": 1.0, "end": 1.4}], audioA)
    B = _mk([{"word": " hello", "start": 1.02, "end": 1.42}], audioB)
    suppress_cross_track_bleed([A, B], _decode=lambda p: None)
    assert len(A["words"]) == 1          # loud copy kept
    assert len(B["words"]) == 0          # faint duplicate dropped

def test_simultaneous_different_text_both_kept():
    # Both loud in the same window but DIFFERENT words -> genuine overlap, keep both.
    audioA = np.concatenate([_tone(1.0, 0.0), _tone(0.4, 0.5), _tone(0.6, 0.0)])
    audioB = np.concatenate([_tone(1.0, 0.0), _tone(0.4, 0.5), _tone(0.6, 0.0)])   # equally loud
    A = _mk([{"word": " yes", "start": 1.0, "end": 1.4}], audioA)
    B = _mk([{"word": " no", "start": 1.0, "end": 1.4}], audioB)
    suppress_cross_track_bleed([A, B], _decode=lambda p: None)
    assert len(A["words"]) == 1 and len(B["words"]) == 1

def test_same_text_but_not_louder_kept():
    # Same text, overlapping, but B is only ~3 dB quieter (< margin) -> not clearly bleed, keep.
    audioA = np.concatenate([_tone(1.0, 0.0), _tone(0.4, 0.5), _tone(0.6, 0.0)])
    audioB = audioA * 0.7                                                           # ~3 dB down
    A = _mk([{"word": " hello", "start": 1.0, "end": 1.4}], audioA)
    B = _mk([{"word": " hello", "start": 1.0, "end": 1.4}], audioB)
    suppress_cross_track_bleed([A, B], margin_db=6.0, _decode=lambda p: None)
    assert len(A["words"]) == 1 and len(B["words"]) == 1

def test_single_track_is_noop():
    A = _mk([{"word": " hi", "start": 0.0, "end": 0.5}], _tone(0.5, 0.5))
    suppress_cross_track_bleed([A], _decode=lambda p: None)
    assert len(A["words"]) == 1

def test_missing_audio_track_never_drops():
    audioA = np.concatenate([_tone(1.0, 0.0), _tone(0.4, 0.5)])
    A = _mk([{"word": " hello", "start": 1.0, "end": 1.4}], audioA)
    B = _mk([{"word": " hello", "start": 1.0, "end": 1.4}], None)   # undecodable
    suppress_cross_track_bleed([A, B], _decode=lambda p: None)      # decode returns None too
    assert len(A["words"]) == 1 and len(B["words"]) == 1            # no evidence -> keep all

def test_same_text_louder_but_not_time_overlapping_kept():
    # Same text, B louder, but the two words do NOT overlap in time (gap > overlap_tol)
    # -> overlap leg of the AND fails -> keep both. Proves overlap is load-bearing.
    audioA = np.concatenate([_tone(1.0, 0.0), _tone(0.4, 0.5), _tone(2.0, 0.0)])   # A word 1.0-1.4
    audioB = np.concatenate([_tone(3.0, 0.5)])                                      # B loud everywhere
    A = _mk([{"word": " hello", "start": 1.0, "end": 1.4}], audioA)
    B = _mk([{"word": " hello", "start": 2.5, "end": 2.9}], audioB)                # 1.1s later, no overlap
    suppress_cross_track_bleed([A, B], overlap_tol=0.3, _decode=lambda p: None)
    assert len(A["words"]) == 1 and len(B["words"]) == 1

def test_word_window_past_audio_end_does_not_crash():
    # A word whose window lies beyond track B's shorter audio must not crash;
    # with no measurable B energy there, the word is kept.
    audioA = np.concatenate([_tone(1.0, 0.0), _tone(0.4, 0.5)])                     # ~1.4s of audio
    audioB = _tone(0.5, 0.5)                                                        # only 0.5s of audio
    A = _mk([{"word": " hello", "start": 1.0, "end": 1.4}], audioA)                # window past B's end
    B = _mk([{"word": " hello", "start": 1.0, "end": 1.4}], audioB)
    suppress_cross_track_bleed([A, B], _decode=lambda p: None)                      # must not raise
    assert len(A["words"]) == 1                                                     # kept (no B evidence)

def test_cluster_drop_removes_text_mismatch_residue():
    # Track B has a cluster that is ENERGY-dominated by A everywhere, but the text
    # never matches A (mis-transcription), so stage-1 keeps it. Stage-2 must drop it.
    # A: loud 0-3s. B: faint copy, words tagged speaker 'b1', DIFFERENT text than A.
    # B has >=3 words so the cluster clears the minimum-size floor (thin-evidence guard).
    audioA = _tone(3.0, 0.5)
    audioB = audioA * 0.05                       # ~26 dB quieter across the board
    A = _mk([{"word": " hello", "start": 0.2, "end": 0.6, "speaker": "speaker_0"},
             {"word": " world", "start": 1.2, "end": 1.6, "speaker": "speaker_0"},
             {"word": " again", "start": 2.2, "end": 2.6, "speaker": "speaker_0"}], audioA)
    B = _mk([{"word": " zzz",  "start": 0.25, "end": 0.65, "speaker": "speaker_1"},
             {"word": " qqq",  "start": 1.25, "end": 1.65, "speaker": "speaker_1"},
             {"word": " www",  "start": 2.25, "end": 2.65, "speaker": "speaker_1"}], audioB)
    suppress_cross_track_bleed([A, B], cluster_frac=0.6, _decode=lambda p: None)
    assert len(A["words"]) == 3                  # loud real speaker kept
    assert len(B["words"]) == 0                  # faint mis-transcribed cluster dropped whole

def test_cluster_thin_evidence_not_dropped():
    # A 1-word cluster on track B, energy-dominated by A, must NOT be dropped by
    # stage 2 (too few words = thin evidence). Different text so stage 1 keeps it.
    audioA = _tone(2.0, 0.5)
    audioB = audioA * 0.05                       # ~26 dB quieter (dominated)
    A = _mk([{"word": " hello", "start": 0.2, "end": 0.6, "speaker": "speaker_0"},
             {"word": " there", "start": 0.8, "end": 1.2, "speaker": "speaker_0"},
             {"word": " world", "start": 1.4, "end": 1.8, "speaker": "speaker_0"}], audioA)
    B = _mk([{"word": " zzz", "start": 0.25, "end": 0.65, "speaker": "speaker_1"}], audioB)
    suppress_cross_track_bleed([A, B], cluster_frac=0.6, _decode=lambda p: None)
    assert len(B["words"]) == 1                  # single-word cluster kept (thin evidence)
    assert len(A["words"]) == 3                  # loud real speaker kept

def test_cluster_keeps_real_speaker_not_energy_dominated():
    # B's cluster is the LOUD source in its own windows -> 0% dominated -> kept,
    # even though A is loud at other times.
    audioA = np.concatenate([_tone(1.0, 0.5), _tone(2.0, 0.0)])     # A loud 0-1s only
    audioB = np.concatenate([_tone(1.0, 0.0), _tone(2.0, 0.5)])     # B loud 1-3s only
    A = _mk([{"word": " hi", "start": 0.2, "end": 0.6, "speaker": "speaker_0"}], audioA)
    B = _mk([{"word": " bye", "start": 1.5, "end": 2.5, "speaker": "speaker_0"}], audioB)
    suppress_cross_track_bleed([A, B], cluster_frac=0.6, _decode=lambda p: None)
    assert len(A["words"]) == 1 and len(B["words"]) == 1            # both real, kept

def test_cluster_pass_ignores_words_without_speaker():
    audioA = _tone(2.0, 0.5); audioB = audioA * 0.05
    A = _mk([{"word": " x", "start": 0.2, "end": 0.6, "speaker": "speaker_0"}], audioA)
    B = _mk([{"word": " y", "start": 0.2, "end": 0.6}], audioB)     # no speaker key
    suppress_cross_track_bleed([A, B], cluster_frac=0.6, _decode=lambda p: None)
    assert len(B["words"]) == 1                                     # untouched (not grouped)
