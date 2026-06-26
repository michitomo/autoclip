"""video/disfluency.py の単体テスト (純ロジック, ネットなし)。

LLM 呼び出しは伴わず、マーカー → 時刻スパン写像・char展開・スパン結合を検証する。
"""

from __future__ import annotations

from src.models import WhisperWord
from src.video.disfluency import (
    MARK_CLOSE,
    MARK_OPEN,
    _chars_with_times,
    _spans_from_marked,
    merge_spans,
)


def _w(word: str, start: float, end: float) -> WhisperWord:
    return WhisperWord(word=word, start=start, end=end)


def _O(s: str) -> str:  # noqa: N802 - 読みやすさ優先 (open でラップ)
    return MARK_OPEN + s + MARK_CLOSE


# ---------------------------------------------------------------------------
# _chars_with_times: 語 → 文字+時刻 (線形補間, 空白除去)
# ---------------------------------------------------------------------------


def test_chars_with_times_interpolates():
    raw, times = _chars_with_times([_w("ab", 0.0, 1.0)])
    assert raw == "ab"
    assert times == [(0.0, 0.5), (0.5, 1.0)]


def test_chars_with_times_skips_blank_tokens():
    raw, times = _chars_with_times([_w(" ", 0.0, 0.2), _w("x", 0.2, 0.4)])
    assert raw == "x"
    assert times == [(0.2, 0.4)]


# ---------------------------------------------------------------------------
# _spans_from_marked: ⟪ ⟫ で囲まれた本文範囲 → member-WAV 時刻スパン
# ---------------------------------------------------------------------------


def _budget_times() -> list[tuple[float, float]]:
    # 予算いや補正予算 の 8 文字に時刻を割り当てる
    words = [_w("予算", 0.0, 0.6), _w("いや", 0.6, 1.0), _w("補正予算", 1.0, 2.0)]
    _, times = _chars_with_times(words)
    return times


def test_spans_from_marked_basic():
    times = _budget_times()  # 8 文字
    # 「予算いや」(0..4 文字) を囲む → [予の開始, や の終了] = (0.0, 1.0)
    spans = _spans_from_marked(_O("予算いや") + "補正予算", times, 0)
    assert spans == [(0.0, 1.0)]


def test_spans_from_marked_respects_base_offset():
    times = _budget_times()
    # base=4 のチャンク "補正予算" の先頭 2 文字「補正」を囲む → 文字 index 4,5
    spans = _spans_from_marked(_O("補正") + "予算", times, 4)
    assert spans == [(times[4][0], times[5][1])]


def test_spans_from_marked_multiple_and_unbalanced():
    times = _budget_times()
    # 2 箇所 + 末尾に閉じ無し open (無視される)
    marked = _O("予算") + "いや" + _O("補正") + "予" + MARK_OPEN + "算"
    spans = _spans_from_marked(marked, times, 0)
    assert spans == [(times[0][0], times[1][1]), (times[4][0], times[5][1])]


def test_spans_from_marked_empty_wrap_ignored():
    times = _budget_times()
    spans = _spans_from_marked(MARK_OPEN + MARK_CLOSE + "予算", times, 0)
    assert spans == []


def test_spans_from_marked_too_long_span_discarded():
    # 26 文字を 1 スパンで囲む → _MAX_SPAN_CHARS(25) 超で破棄
    words = [_w("あ", i * 0.1, i * 0.1 + 0.1) for i in range(30)]
    _, times = _chars_with_times(words)
    body = "あ" * 30
    spans = _spans_from_marked(_O("あ" * 26) + body[26:], times, 0)
    assert spans == []


def test_spans_from_marked_ignores_added_whitespace():
    times = _budget_times()
    # LLM が空白を足しても本文位置はズレない (空白は body 計数に含めない)
    spans = _spans_from_marked(MARK_OPEN + "予 算" + MARK_CLOSE + "いや補正予算", times, 0)
    assert spans == [(times[0][0], times[1][1])]


# ---------------------------------------------------------------------------
# merge_spans
# ---------------------------------------------------------------------------


def test_merge_spans_overlap_and_sort():
    assert merge_spans([(3.0, 4.0), (0.0, 1.0), (0.5, 2.0)]) == [(0.0, 2.0), (3.0, 4.0)]


def test_merge_spans_empty():
    assert merge_spans([]) == []
