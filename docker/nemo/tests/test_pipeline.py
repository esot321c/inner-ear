import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import (decide_mode, assign_speakers, smooth_sentences, build_turns,
                      render, pick_sample_turns, display_name)


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

def test_smooth_moves_boundary_word_to_sentence_majority():
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
