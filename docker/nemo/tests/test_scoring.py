"""Tests for the per-word speaker-accuracy scorer."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scoring  # noqa: E402


def _w(word, speaker, start=0.0):
    return {"word": word, "speaker": speaker, "start": start}


def test_perfect_with_swapped_labels_scores_100():
    # pred uses speaker_0/1; truth uses Tina/Graham; mapping must recover 100%.
    pred = [_w("hello", "speaker_0"), _w("there", "speaker_0"),
            _w("hi", "speaker_1"), _w("back", "speaker_1")]
    truth = [_w("hello", "Tina"), _w("there", "Tina"),
             _w("hi", "Graham"), _w("back", "Graham")]
    s = scoring.score(pred, truth)
    assert s["accuracy"] == 1.0
    assert s["mapping"] == {"speaker_0": "Tina", "speaker_1": "Graham"}
    assert s["errors"] == []


def test_one_wrong_word_is_reported():
    pred = [_w("a", "speaker_0"), _w("yeah", "speaker_0", 1.0), _w("b", "speaker_1")]
    truth = [_w("a", "Tina"), _w("yeah", "Graham", 1.0), _w("b", "Graham")]
    s = scoring.score(pred, truth)
    # best mapping: speaker_0->Tina, speaker_1->Graham. "yeah" is pred Tina, truth Graham.
    assert s["accuracy"] == 2 / 3
    assert len(s["errors"]) == 1
    e = s["errors"][0]
    assert e["word"] == "yeah" and e["truth"] == "Graham" and e["time"] == 1.0


def test_backchannel_subset_tracked_separately():
    pred = [_w("okay", "speaker_0"), _w("right", "speaker_0"), _w("word", "speaker_0")]
    truth = [_w("okay", "A"), _w("right", "B"), _w("word", "A")]
    s = scoring.score(pred, truth)
    # mapping speaker_0->A: okay ok, right wrong, word ok => 2/3 overall.
    assert s["backchannel_n"] == 2  # okay, right
    assert s["backchannel_acc"] == 0.5  # okay ok, right wrong


def test_alignment_tolerates_a_transcription_diff():
    pred = [_w("the", "speaker_0"), _w("kat", "speaker_0"), _w("sat", "speaker_0")]
    truth = [_w("the", "A"), _w("cat", "A"), _w("sat", "A")]
    s = scoring.score(pred, truth)
    # "kat" vs "cat" doesn't align; "the" and "sat" do, both correct.
    assert s["aligned"] == 2
    assert s["accuracy"] == 1.0


def test_build_truth_reassigns_from_edited_turns():
    # pred glued "yeah" to speaker_0; user moved it to speaker_1 in the turn text.
    pred = [_w("okay", "speaker_0", 0.0), _w("yeah", "speaker_0", 1.0),
            _w("then", "speaker_1", 1.6)]
    edited = "speaker_0: okay\nspeaker_1: yeah then"
    truth = scoring.build_truth(pred, edited)
    assert [w["speaker"] for w in truth] == ["speaker_0", "speaker_1", "speaker_1"]
    # words/times preserved
    assert [w["word"] for w in truth] == ["okay", "yeah", "then"]
    assert truth[1]["start"] == 1.0


def test_build_truth_then_score_roundtrips():
    pred = [_w("a", "speaker_0"), _w("yeah", "speaker_0", 1.0), _w("b", "speaker_1")]
    edited = "speaker_0: a\nspeaker_1: yeah b"
    truth = scoring.build_truth(pred, edited)
    s = scoring.score(pred, truth)
    assert len(s["errors"]) == 1 and s["errors"][0]["word"] == "yeah"


def test_partial_label_with_end_marker_truncates_truth():
    pred = [_w("one", "speaker_0", 0.0), _w("two", "speaker_0", 1.0),
            _w("three", "speaker_1", 2.0), _w("four", "speaker_1", 3.0)]
    # user corrected first two words, then stopped with an END marker
    edited = "speaker_0: one\nspeaker_1: two\n---END---\nspeaker_0: three four"
    truth = scoring.build_truth(pred, edited)
    assert len(truth) == 2                      # tail dropped, not scored
    assert [w["speaker"] for w in truth] == ["speaker_0", "speaker_1"]


def test_deleting_tail_also_truncates():
    pred = [_w("a", "s0", 0.0), _w("b", "s0", 1.0), _w("c", "s1", 2.0)]
    truth = scoring.build_truth(pred, "s0: a b")   # 'c' line deleted
    assert len(truth) == 2


def test_aggregate_micro_averages():
    s1 = scoring.score([_w("a", "s0")], [_w("a", "A")])               # 1/1
    s2 = scoring.score([_w("a", "s0"), _w("b", "s1")],                # 1/2
                       [_w("a", "A"), _w("b", "A")])
    agg = scoring.aggregate([s1, s2])
    assert agg["aligned"] == 3
    assert abs(agg["accuracy"] - 2 / 3) < 1e-9


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
