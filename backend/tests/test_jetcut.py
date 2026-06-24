"""video/jetcut.py の単体テスト (純ロジック)。

重点: 日本語サブワード分割でのフィラー検出、無音分割、旧→新タイムライン写像。
"""

from __future__ import annotations

from src.models import WhisperWord
from src.video.jetcut import (
    DEFAULT_FILLERS,
    _mark_filler_words,
    build_edl,
)


def _w(word: str, start: float, end: float) -> WhisperWord:
    return WhisperWord(word=word, start=start, end=end)


def _words(seq: list[tuple[str, float, float]]) -> list[WhisperWord]:
    return [_w(*t) for t in seq]


# ---------------------------------------------------------------------------
# フィラー検出 (サブワード連結) — 最重要 (spike 由来)
# ---------------------------------------------------------------------------


def test_filler_single_token():
    ws = _words([("あの", 0.0, 0.3), ("予算", 0.3, 0.8)])
    drop = _mark_filler_words(ws, DEFAULT_FILLERS)
    assert drop == [True, False]


def test_filler_split_across_subword_tokens():
    # 「えーと」が え / ー / と に割れていても検出する
    ws = _words([("え", 0.0, 0.1), ("ー", 0.1, 0.2), ("と", 0.2, 0.4), ("質問", 0.4, 0.9)])
    drop = _mark_filler_words(ws, DEFAULT_FILLERS)
    assert drop == [True, True, True, False]


def test_filler_does_not_remove_real_word_containing_kana():
    # 「映画」を構成する仮名がフィラーに化けない (連結してもフィラー集合に無い)
    ws = _words([("映", 0.0, 0.2), ("画", 0.2, 0.5)])
    drop = _mark_filler_words(ws, DEFAULT_FILLERS)
    assert drop == [False, False]


def test_filler_longest_match_preferred():
    # 「えー」も「えーと」もフィラー。え/ー/と は最長一致で 3 トークンまとめて落とす
    ws = _words([("えー", 0.0, 0.2), ("と", 0.2, 0.4), ("本日", 0.4, 0.9)])
    drop = _mark_filler_words(ws, DEFAULT_FILLERS)
    assert drop == [True, True, False]


# ---------------------------------------------------------------------------
# build_edl — dead air 分割 + 連結
# ---------------------------------------------------------------------------


def test_build_edl_empty():
    edl = build_edl([])
    assert edl.keep_ranges == []
    assert edl.kept_words == []


def test_build_edl_removes_dead_air_gap():
    # 0.0-1.0 話す, 3.0 まで無音 (2s gap > 0.6), 3.0-4.0 話す → 2 区間に割れる
    ws = _words([("あ", 0.0, 0.5), ("い", 0.5, 1.0), ("う", 3.0, 3.5), ("え", 3.5, 4.0)])
    edl = build_edl(ws, remove_fillers=False, min_segment=0.1, edge_pad=0.0, silence_pad=0.1)
    assert len(edl.keep_ranges) == 2
    # 区間1は ~[0,1.1], 区間2は ~[2.9,4.1] (pad込み); 大きな無音は含まれない
    r1, r2 = edl.keep_ranges
    assert r1.start < 0.2 and 1.0 <= r1.end <= 1.2
    assert 2.8 <= r2.start <= 3.0 and 4.0 <= r2.end <= 4.2
    # 大きな無音 (1.1〜2.9) は除去されている
    assert r2.start - r1.end > 1.0


def test_build_edl_keeps_small_gaps_in_one_range():
    # 語間 gap 0.2s (< 0.6) は同一区間に残す
    ws = _words([("あ", 0.0, 0.5), ("い", 0.7, 1.2), ("う", 1.4, 1.9)])
    edl = build_edl(ws, remove_fillers=False, min_segment=0.1, edge_pad=0.0)
    assert len(edl.keep_ranges) == 1


def test_build_edl_removes_fillers_and_remaps_time():
    # フィラー「あの」を挟む。除去後、後続語の new 時間は前詰めされる。
    ws = _words([
        ("本日", 0.0, 0.5),
        ("あの", 0.5, 0.9),   # filler
        ("予算", 0.9, 1.4),
    ])
    edl = build_edl(ws, min_segment=0.1, edge_pad=0.0, silence_pad=0.05, merge_gap=1.0)
    kept_texts = [k.word for k in edl.kept_words]
    assert "あの" not in kept_texts
    assert kept_texts == ["本日", "予算"]
    # new timeline monotonic and starts at ~0
    assert edl.kept_words[0].new_start <= 0.05
    news = [k.new_start for k in edl.kept_words]
    assert news == sorted(news)


def test_build_edl_new_timeline_shorter_than_old_when_cutting():
    ws = _words([
        ("あ", 0.0, 0.5),
        ("い", 0.5, 1.0),
        ("う", 5.0, 5.5),   # big gap → dead air removed
        ("え", 5.5, 6.0),
    ])
    edl = build_edl(ws, remove_fillers=False, min_segment=0.1, edge_pad=0.0)
    old_span = ws[-1].end - ws[0].start  # 6.0
    new_span = edl.kept_words[-1].new_end
    assert new_span < old_span  # dead air removed → compressed
    # kept_duration roughly equals sum of ranges
    assert abs(edl.kept_duration - new_span) < 0.3


def test_build_edl_kept_words_within_kept_duration():
    ws = _words([("あ", 0.0, 0.5), ("い", 0.6, 1.1), ("う", 1.2, 1.7)])
    edl = build_edl(ws, remove_fillers=False, min_segment=0.1, edge_pad=0.0)
    for k in edl.kept_words:
        assert 0.0 <= k.new_start <= edl.kept_duration + 0.01
        assert k.new_end <= edl.kept_duration + 0.01
        assert k.new_start <= k.new_end
