"""video/align.py の単体テスト: 補正文を word タイムスタンプに再アライン。"""

from __future__ import annotations

from src.models import WhisperWord
from src.video.align import align_corrected_to_words


def _w(word: str, start: float, end: float) -> WhisperWord:
    return WhisperWord(word=word, start=start, end=end)


def _text(words: list[WhisperWord]) -> str:
    return "".join(w.word for w in words)


def test_align_identity_preserves_text_and_time():
    raw = [_w("本", 0.0, 0.3), _w("日", 0.3, 0.6), _w("は", 0.6, 0.9)]
    out = align_corrected_to_words(raw, "本日は")
    assert _text(out) == "本日は"
    # times monotonic and within original span
    assert out[0].start == 0.0
    assert all(out[i].start <= out[i + 1].start for i in range(len(out) - 1))


def test_align_name_substitution_keeps_timing():
    # raw misheard "高山智", corrected to "高山聡史" (智→聡史)
    raw = [_w("高", 1.0, 1.3), _w("山", 1.3, 1.6), _w("智", 1.6, 2.0)]
    out = align_corrected_to_words(raw, "高山聡史")
    assert _text(out) == "高山聡史"
    # 高山 keep their original times
    assert out[0].start == 1.0  # 高
    assert out[1].start == 1.3  # 山
    # 聡 and 史 share the replaced 智 span [1.6, 2.0]
    assert out[2].start >= 1.6
    assert out[-1].end <= 2.01


def test_align_party_name_substitution():
    # 賛成党 → 参政党 (same length replace)
    raw = [_w("賛", 0.0, 0.2), _w("成", 0.2, 0.4), _w("党", 0.4, 0.6)]
    out = align_corrected_to_words(raw, "参政党")
    assert _text(out) == "参政党"
    assert out[0].start == 0.0
    assert out[2].end <= 0.61


def test_align_punctuation_insertion_inherits_time():
    # corrector inserts 。 after は
    raw = [_w("本", 0.0, 0.3), _w("日", 0.3, 0.6), _w("は", 0.6, 0.9)]
    out = align_corrected_to_words(raw, "本日は。")
    assert _text(out) == "本日は。"
    # the inserted 。 should take the trailing time (~0.9), zero-ish length
    assert out[-1].word == "。"
    assert out[-1].start >= 0.9 - 1e-6


def test_align_deletion_loop_removed():
    # raw has a repeated loop "ありがとうありがとう", corrected drops the dup
    raw = [_w(c, i * 0.1, (i + 1) * 0.1) for i, c in enumerate("ありがとうありがとう")]
    out = align_corrected_to_words(raw, "ありがとう")
    assert _text(out) == "ありがとう"
    assert all(out[i].start <= out[i + 1].start for i in range(len(out) - 1))


def test_align_empty_inputs():
    assert align_corrected_to_words([], "本日は") == []
    assert align_corrected_to_words([_w("あ", 0.0, 0.1)], "") == []


def test_align_ignores_whitespace_in_raw_words():
    raw = [_w(" 本 ", 0.0, 0.3), _w("日", 0.3, 0.6)]
    out = align_corrected_to_words(raw, "本日")
    assert _text(out) == "本日"


def test_align_monotonic_after_complex_edit():
    raw = [_w(c, i * 0.1, (i + 1) * 0.1) for i, c in enumerate("あいうえお")]
    # replace middle + insert punctuation
    out = align_corrected_to_words(raw, "あいXYZお。")
    starts = [w.start for w in out]
    assert starts == sorted(starts)


# --- 案1: 捨てすぎガード (校正が実発話を丸ごと落としたら raw を復元) ---

def test_align_guard_restores_large_unique_deletion():
    """校正が長い実発話を落とし、その内容が校正後に再出現しない場合は復元する。"""
    raw_text = (
        "発注プロセスについて契約の進め方について"
        "関係省庁がこのように関与していくべきだということについて"
        "制度上の根拠があるのかということ"
    )
    raw = [_w(c, i * 0.1, (i + 1) * 0.1) for i, c in enumerate(raw_text)]
    # 校正が中盤の塊を丸ごと落としたケース
    corrected = "発注プロセスについて契約の進め方について、どうあるべきか。"
    out = _text(align_corrected_to_words(raw, corrected))
    assert "関係省庁" in out
    assert "制度上の根拠" in out
    assert "関与していくべき" in out


def test_align_guard_keeps_loop_dedup_even_when_long():
    """長い削除でも、それが校正後に重複として残る (Whisperループ) なら捨てる。"""
    # 「制度上の根拠があるのかということ」を2回言ったループを校正が1回に畳む
    phrase = "制度上の根拠があるのかということ"
    raw = [_w(c, i * 0.1, (i + 1) * 0.1) for i, c in enumerate(phrase + phrase)]
    out = _text(align_corrected_to_words(raw, phrase))
    # 二重に復活していない (1回だけ)
    assert out.count("制度上の根拠") == 1


def test_align_guard_short_deletion_still_dropped():
    """短い削除 (フィラー等) は従来どおり捨てる。"""
    raw = [_w(c, i * 0.1, (i + 1) * 0.1) for i, c in enumerate("えーと本日はあのご質問")]
    out = _text(align_corrected_to_words(raw, "本日はご質問"))
    assert out == "本日はご質問"
