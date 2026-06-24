"""video/segment.py の文節境界マッピング単体テスト (LLM 呼び出しは除く)。"""

from __future__ import annotations

from src.models import KeptWord, WhisperWord
from src.video.segment import (
    SEGMENT_MARKER,
    _normalize,
    _split_into_chunks,
    break_after_indices_for_kept,
    break_after_indices_from_marked,
    break_after_times_from_marked,
)


def _ww(word: str, start: float, end: float) -> WhisperWord:
    return WhisperWord(word=word, start=start, end=end)


def _kw(word: str, s: float, e: float) -> KeptWord:
    return KeptWord(word=word, old_start=s, old_end=e, new_start=s, new_end=e)


M = SEGMENT_MARKER


def test_break_indices_basic():
    # body "本日は質問" with a marker after は (index 2)
    words = [_ww("本", 0.0, 0.1), _ww("日", 0.1, 0.2), _ww("は", 0.2, 0.3),
             _ww("質", 0.3, 0.4), _ww("問", 0.4, 0.5)]
    marked = f"本日は{M}質問"
    idxs = break_after_indices_from_marked(marked, words)
    assert idxs == {2}  # break after は (the 3rd word, index 2)


def test_break_indices_multiple():
    words = [_ww(c, i * 0.1, (i + 1) * 0.1) for i, c in enumerate("本日は質問です")]
    marked = f"本日は{M}質問{M}です"
    idxs = break_after_indices_from_marked(marked, words)
    assert idxs == {2, 4}  # after は(2), after 問(4)


def test_break_indices_no_marker():
    words = [_ww("本", 0.0, 0.1), _ww("日", 0.1, 0.2)]
    assert break_after_indices_from_marked("本日", words) == set()


def test_break_indices_ignores_whitespace_in_words():
    words = [_ww(" 本 ", 0.0, 0.1), _ww("日", 0.1, 0.2), _ww("は", 0.2, 0.3)]
    marked = f"本日{M}は"
    assert break_after_indices_from_marked(marked, words) == {1}


def test_break_times_uses_word_end():
    words = [_ww("本", 0.0, 0.1), _ww("日", 0.1, 0.25), _ww("は", 0.25, 0.4)]
    marked = f"本日{M}は"
    times = break_after_times_from_marked(marked, words)
    assert times == {0.25}  # end of 日 (the boundary word)


def test_break_times_map_to_kept_after_filler_removal():
    # aligned words: 本 日 は [filler あの] 質 問  (with end times)
    aligned = [_ww("本", 0.0, 0.1), _ww("日", 0.1, 0.25), _ww("は", 0.25, 0.4),
               _ww("あ", 0.4, 0.5), _ww("の", 0.5, 0.6),
               _ww("質", 0.6, 0.7), _ww("問", 0.7, 0.8)]
    marked = f"本日は{M}質問"  # boundary after は (end=0.4)
    times = break_after_times_from_marked(marked, aligned)
    assert times == {0.4}

    # JetCut dropped あの → kept_words = 本 日 は 質 問 (old_end preserved)
    kept = [_kw("本", 0.0, 0.1), _kw("日", 0.1, 0.25), _kw("は", 0.25, 0.4),
            _kw("質", 0.6, 0.7), _kw("問", 0.7, 0.8)]
    idxs = break_after_indices_for_kept(kept, times)
    assert idxs == {2}  # break after は (index 2 in kept) — correct despite removal


def test_break_indices_for_kept_tolerance():
    kept = [_kw("a", 0.0, 0.1), _kw("b", 0.1, 0.2)]
    # break time 0.103 should match old_end 0.1 within tol
    assert break_after_indices_for_kept(kept, {0.103}, tol=0.05) == {0}


def test_normalize_strips_whitespace():
    assert _normalize("本 日　は") == "本日は"


def test_split_into_chunks_keeps_sentences():
    text = "一文目です。" * 100  # ~600 chars
    chunks = _split_into_chunks(text, limit=200)
    assert len(chunks) > 1
    assert "".join(chunks) == text  # no content lost
    # every chunk ends at a sentence boundary (。)
    assert all(c.endswith("。") for c in chunks)


def test_split_into_chunks_short_text_single():
    assert _split_into_chunks("短い。", limit=200) == ["短い。"]
