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


# --- render / samples / names ---
def test_display_name_maps_and_defaults():
    assert display_name("speaker_1", {"speaker_1": "Joe"}) == "Joe"
    assert display_name("speaker_2", {}) == "Speaker 2"
    assert display_name("__you__", {}, you_name="Graham") == "Graham"

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

def test_pick_sample_turns_skips_you():
    turns = [{"speaker": "__you__", "start": 0, "end": 5, "text": "me"},
             {"speaker": "A", "start": 5, "end": 7, "text": "them"}]
    picks = pick_sample_turns(turns)
    assert "__you__" not in picks and "A" in picks


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
