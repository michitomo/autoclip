"""語単位タイムスタンプ (word-level) アップグレードの単体テスト。

JetCut が依存する WhisperSegment.words の充填・オフセット・整合性を検証する。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.models import WhisperWord
from src.transcriber import (
    _bucket_words_into_segments,
    _call_whisper,
    _drop_leading_words,
    _merge_window_results,
)


def _seg(start: float, end: float, text: str, *, id: int = 0, words=None) -> dict:
    return {
        "id": id,
        "seek": 0,
        "start": start,
        "end": end,
        "text": text,
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": -0.2,
        "compression_ratio": 1.5,
        "no_speech_prob": 0.01,
        "words": words or [],
    }


def _w(word: str, start: float, end: float) -> dict:
    return {"word": word, "start": start, "end": end}


# ---------------------------------------------------------------------------
# _bucket_words_into_segments
# ---------------------------------------------------------------------------


def test_bucket_words_assigns_by_midpoint():
    segments = [_seg(0.0, 1.0, "あ"), _seg(1.0, 2.0, "い")]
    words = [_w("あ", 0.1, 0.4), _w("い", 1.2, 1.6)]
    _bucket_words_into_segments(segments, words)
    assert [w["word"] for w in segments[0]["words"]] == ["あ"]
    assert [w["word"] for w in segments[1]["words"]] == ["い"]


def test_bucket_words_boundary_goes_to_containing_segment():
    # midpoint exactly inside the second segment
    segments = [_seg(0.0, 1.0, "x"), _seg(1.0, 2.0, "y")]
    words = [_w("y", 1.0, 1.4)]  # midpoint 1.2 → second segment
    _bucket_words_into_segments(segments, words)
    assert segments[0]["words"] == []
    assert len(segments[1]["words"]) == 1


def test_bucket_words_empty_inputs_noop():
    segments = [_seg(0.0, 1.0, "x")]
    _bucket_words_into_segments(segments, [])
    assert segments[0]["words"] == []
    _bucket_words_into_segments([], [_w("x", 0.0, 0.1)])  # no crash


def test_bucket_words_in_gap_is_dropped():
    # a word whose midpoint falls in a gap between segments is dropped (not mis-assigned)
    segments = [_seg(0.0, 1.0, "x"), _seg(2.0, 3.0, "y")]
    words = [_w("gap", 1.4, 1.6)]  # midpoint 1.5, in the 1.0–2.0 gap
    _bucket_words_into_segments(segments, words)
    assert segments[0]["words"] == []
    assert segments[1]["words"] == []


# ---------------------------------------------------------------------------
# _merge_window_results — words inherit +w_start offset and stay monotonic
# ---------------------------------------------------------------------------


def test_merge_window_offsets_words_to_absolute_time():
    # single window starting at t=100; word at window-relative 0.2 → absolute 100.2
    windows = [(100.0, 125.0)]
    seg = _seg(0.0, 2.0, "テスト", words=[_w("テ", 0.0, 0.5), _w("スト", 0.5, 2.0)])
    merged = _merge_window_results(windows, [[seg]], overlap_seconds=5.0)
    assert len(merged) == 1
    ws = merged[0].words
    assert [round(w.start, 2) for w in ws] == [100.0, 100.5]
    assert [round(w.end, 2) for w in ws] == [100.5, 102.0]
    # words fall within their segment's absolute span
    assert merged[0].start <= ws[0].start
    assert ws[-1].end <= merged[0].end


def test_merge_window_words_globally_monotonic_across_windows():
    # two overlapping windows; after merge, word starts must be non-decreasing
    windows = [(0.0, 25.0), (20.0, 45.0)]
    segs0 = [_seg(2.0, 8.0, "あいう", words=[_w("あ", 2.0, 4.0), _w("いう", 4.0, 8.0)])]
    segs1 = [_seg(2.0, 8.0, "えお", words=[_w("え", 2.0, 5.0), _w("お", 5.0, 8.0)])]
    merged = _merge_window_results(windows, [segs0, segs1], overlap_seconds=5.0)
    all_words = [w for s in merged for w in s.words]
    starts = [w.start for w in all_words]
    assert starts == sorted(starts), f"word starts not monotonic: {starts}"


# ---------------------------------------------------------------------------
# _drop_leading_words
# ---------------------------------------------------------------------------


def test_drop_leading_words_removes_prefix_by_char_count():
    words = [
        WhisperWord(word="宿", start=0.0, end=0.1),
        WhisperWord(word="泊", start=0.1, end=0.2),
        WhisperWord(word="する", start=0.2, end=0.5),
    ]
    # trim 2 chars ("宿泊") → drop first two words
    out = _drop_leading_words(words, 2)
    assert [w.word for w in out] == ["する"]


def test_drop_leading_words_zero_or_empty():
    words = [WhisperWord(word="あ", start=0.0, end=0.1)]
    assert _drop_leading_words(words, 0) == words
    assert _drop_leading_words([], 5) == []


# ---------------------------------------------------------------------------
# _call_whisper — parses top-level words and buckets into segments
# ---------------------------------------------------------------------------


def test_call_whisper_parses_and_buckets_top_level_words():
    fake_result = MagicMock()
    fake_result.text = "宿泊する"
    fake_result.segments = [
        MagicMock(
            id=0, seek=0, start=0.0, end=1.0, text="宿泊", tokens=[],
            temperature=0.0, avg_logprob=-0.2, compression_ratio=1.5, no_speech_prob=0.01,
        ),
        MagicMock(
            id=1, seek=0, start=1.0, end=2.0, text="する", tokens=[],
            temperature=0.0, avg_logprob=-0.2, compression_ratio=1.5, no_speech_prob=0.01,
        ),
    ]
    fake_result.words = [
        {"word": "宿", "start": 0.0, "end": 0.3},
        {"word": "泊", "start": 0.3, "end": 0.9},
        {"word": "する", "start": 1.1, "end": 1.8},
    ]
    client = MagicMock()
    client.audio.transcriptions.create.return_value = fake_result

    text, segments = _call_whisper(client, b"fakewav", "prompt")
    assert text == "宿泊する"
    assert [w["word"] for w in segments[0]["words"]] == ["宿", "泊"]
    assert [w["word"] for w in segments[1]["words"]] == ["する"]
    # confirm we requested word+segment granularity
    _, kwargs = client.audio.transcriptions.create.call_args
    assert kwargs["timestamp_granularities"] == ["word", "segment"]


# ---------------------------------------------------------------------------
# strip_prompt_echo — remove leaked Whisper prompt from start of words
# ---------------------------------------------------------------------------

from src.models import SpeakerInfo  # noqa: E402
from src.transcriber import strip_prompt_echo  # noqa: E402


def _spk_echo() -> SpeakerInfo:
    return SpeakerInfo(
        name="高山聡史", affiliation="チームみらい", role="質疑者",
        start_seconds=0.0, start_time="", duration_minutes=1,
    )


def _chars(text: str) -> list:
    return [WhisperWord(word=c, start=i * 0.2, end=(i + 1) * 0.2)
            for i, c in enumerate(text)]


def test_strip_prompt_echo_removes_leaked_header():
    words = _chars("高山聡史（チームみらい）：チームみらいの高山です")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == "チームみらいの高山です"


def test_strip_prompt_echo_committee_form():
    words = _chars("内閣委員会。高山聡史（チームみらい）：本日は")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == "本日は"


def test_strip_prompt_echo_no_leak_untouched():
    words = _chars("本日は質問します")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert len(out) == len(words)


def test_strip_prompt_echo_colon_without_name_not_stripped():
    words = _chars("理由は次の通りです")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert len(out) == len(words)


def test_strip_prompt_echo_empty():
    assert strip_prompt_echo([], _spk_echo(), "内閣委員会") == []


def test_strip_prompt_echo_double_header():
    leak = "山下貴司委員長：次、高山聡史君。"
    words = _chars(leak + "高山聡史（チームみらい）：チームみらいの高山です")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == "チームみらいの高山です"


def test_strip_prompt_echo_keeps_real_colon_speech():
    words = _chars("これは重要です：第一に予算")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == "これは重要です：第一に予算"


def test_strip_prompt_echo_colonless_paren_label():
    # 「氏名（所属）」label WITHOUT a colon must be stripped
    words = _chars("高山聡史（チームみらい）チームみらいの高山聡史です。")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == "チームみらいの高山聡史です。"


def test_strip_prompt_echo_keeps_real_parenthetical():
    # real parenthetical speech (no speaker-name match) must be kept
    words = _chars("予算（令和8年度）について質問します")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == "予算（令和8年度）について質問します"


def test_strip_prompt_echo_label_with_trailing_open_quote():
    # 議事録形式「氏名（所属）「発話…」: 閉じ括弧の後に残る開き鉤括弧も剥がす。
    # (実音声には無い飾り。これが残ると字幕が音声から前倒しにズレる)
    words = _chars("高山聡史（チームみらい）「チームみらいの高山です")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == "チームみらいの高山です"


def test_strip_prompt_echo_keeps_leading_quote_when_no_echo():
    # echo を剥がしていない (drop_chars==0) なら文頭の「は本物の発話 → 温存。
    words = _chars("「重要なお知らせ」と書いてあります")
    out = strip_prompt_echo(words, _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == "「重要なお知らせ」と書いてあります"


def test_strip_leading_hallucination_and_boundary_junk():
    # lead_pad pulls in prev-speaker tail + Whisper hallucination + mangled call
    junk = ("ありがとうございました。次に高山聡史君。"
            "高山聡史大きな、大和尾文静。チームみらいの高山聡史です。先ほど")
    out = strip_prompt_echo(_chars(junk), _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == "チームみらいの高山聡史です。先ほど"


def test_strip_keeps_member_intro_with_affiliation():
    # member self-intro (contains affiliation) must NOT be stripped
    text = "チームみらいの高山聡史です。質問します。"
    out = strip_prompt_echo(_chars(text), _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == text


def test_strip_does_not_overstrip_plain_speech():
    text = "本日は質問します。よろしくお願いします。"
    out = strip_prompt_echo(_chars(text), _spk_echo(), "内閣委員会")
    assert "".join(w.word for w in out) == text
