"""video/subtitles.py + renderer.py の単体テスト (純ロジック)。"""

from __future__ import annotations

from src.models import EDL, KeepRange, KeptWord
from src.video.renderer import _build_concat_filtergraph, _escape_filter_path
from src.video.subtitles import (
    _ass_time,
    _srt_time,
    build_ass,
    build_srt,
    group_captions,
)


def _kw(word: str, ns: float, ne: float) -> KeptWord:
    # old times irrelevant for subtitle tests; set equal to new for simplicity
    return KeptWord(word=word, old_start=ns, old_end=ne, new_start=ns, new_end=ne)


# ---------------------------------------------------------------------------
# caption grouping (new timeline)
# ---------------------------------------------------------------------------


def test_group_captions_breaks_on_sentence_punctuation():
    # 文節単位: each clause becomes its own caption (both >= MIN_CAPTION_CHARS)
    words = [_kw("本日は晴天", 0.0, 0.5), _kw("なり。", 0.5, 1.0),
             _kw("質問します", 1.0, 1.5)]
    caps = group_captions(words, max_chars=99, line_break_gap=99)
    assert len(caps) == 2
    assert caps[0].text == "本日は晴天なり。"
    assert caps[1].text == "質問します"


def test_group_captions_breaks_on_clause_comma():
    # 読点でも改行 (文節単位)
    words = [_kw("先ほど委員", 0.0, 0.5), _kw("から、", 0.5, 1.0),
             _kw("お話があり", 1.0, 1.5)]
    caps = group_captions(words, max_chars=99, line_break_gap=99)
    assert len(caps) == 2
    assert caps[0].text == "先ほど委員から、"


def test_group_captions_breaks_on_max_chars():
    words = [_kw("あ" * 12, 0.0, 0.5), _kw("い" * 12, 0.5, 1.0)]
    caps = group_captions(words, max_chars=10, line_break_gap=99)
    assert len(caps) == 2


def test_group_captions_breaks_on_gap():
    words = [_kw("先ほど委員", 0.0, 0.5), _kw("お話があり", 3.0, 3.5)]  # 2.5s gap
    caps = group_captions(words, max_chars=99, line_break_gap=0.6)
    assert len(caps) == 2


def test_group_captions_uses_new_timeline_times():
    words = [_kw("本日は晴", 1.0, 1.5), _kw("質問です", 1.5, 2.0)]
    caps = group_captions(words, max_chars=99, line_break_gap=99)
    assert caps[0].start == 1.0


def test_group_captions_merges_punctuation_only_fragment():
    # a lone 。 (from alignment at a cut boundary) must NOT be its own caption
    words = [_kw("質問します", 0.0, 1.0), _kw("。", 1.0, 1.0)]
    caps = group_captions(words, max_chars=99, line_break_gap=99)
    assert len(caps) == 1
    assert caps[0].text == "質問します。"


def test_group_captions_merges_short_fragment_into_previous():
    words = [_kw("本日は晴天", 0.0, 1.0), _kw("なり。", 1.0, 1.5), _kw("が", 1.5, 1.7)]
    caps = group_captions(words, max_chars=99, line_break_gap=99)
    # "が" (1 char) merges into previous rather than forming its own caption
    assert all(_visible_len_helper(c.text) >= 3 for c in caps)


def _visible_len_helper(text: str) -> int:
    return len("".join(text.split()))


def test_group_captions_no_overlap():
    words = [_kw("本日は晴天。", 0.0, 1.0), _kw("質問します", 0.9, 1.5)]
    caps = group_captions(words, max_chars=99, line_break_gap=99)
    assert len(caps) == 2
    assert caps[0].end <= caps[1].start


# ---------------------------------------------------------------------------
# time formatting
# ---------------------------------------------------------------------------


def test_ass_time_format():
    assert _ass_time(0.0) == "0:00:00.00"
    assert _ass_time(65.5) == "0:01:05.50"
    assert _ass_time(3661.25) == "1:01:01.25"


def test_srt_time_format():
    assert _srt_time(0.0) == "00:00:00,000"
    assert _srt_time(65.5) == "00:01:05,500"


def test_ass_and_srt_contain_caption_text():
    words = [_kw("本日", 0.0, 0.5), _kw("は質問", 0.5, 1.2)]
    ass = build_ass(words, max_chars=99)
    srt = build_srt(words, max_chars=99)
    assert "本日は質問" in ass
    assert "Dialogue:" in ass
    assert "[V4+ Styles]" in ass
    assert "本日は質問" in srt
    assert "-->" in srt


def test_ass_events_format_has_name_and_marginv():
    # Events Format MUST include Name + MarginV so the 9-field Dialogue line
    # (Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect) aligns and the
    # caption text isn't prefixed with stray "0,," (regression guard).
    ass = build_ass([_kw("あ", 0.0, 0.5)], max_chars=99)
    fmt = next(line for line in ass.splitlines() if line.startswith("Format:") and "Start" in line)
    fields = [f.strip() for f in fmt[len("Format:"):].split(",")]
    assert fields == [
        "Layer", "Start", "End", "Style", "Name",
        "MarginL", "MarginR", "MarginV", "Effect", "Text",
    ]


def test_ass_dialogue_text_not_prefixed_with_numbers():
    ass = build_ass([_kw("テスト字幕", 0.0, 1.0)], max_chars=99)
    dlg = next(line for line in ass.splitlines() if line.startswith("Dialogue:"))
    # everything after the 9th comma is the text; it must start with the caption,
    # not a leftover margin/effect number.
    text = dlg.split(",", 9)[-1]
    assert text == "テスト字幕"


def test_ass_escapes_braces():
    words = [_kw("{evil}", 0.0, 0.5)]
    ass = build_ass(words, max_chars=99)
    assert "{evil}" not in ass  # braces neutralized
    assert "(evil)" in ass


# ---------------------------------------------------------------------------
# renderer trim+concat filtergraph + path escaping
# ---------------------------------------------------------------------------


def test_concat_filtergraph_single_range():
    edl = EDL(keep_ranges=[KeepRange(start=1.0, end=2.5)])
    fg = _build_concat_filtergraph(edl, member_start=0.0, post_video="fps=30")
    assert "[0:v]trim=start=1.000:end=2.500,setpts=PTS-STARTPTS[v0]" in fg
    assert "[0:a]atrim=start=1.000:end=2.500,asetpts=PTS-STARTPTS[a0]" in fg
    # 連結と後段の出力ラベル
    assert "[v0][a0]concat=n=1:v=1:a=1[vc][ac]" in fg
    assert fg.endswith("[vc]fps=30[v]")


def test_concat_filtergraph_applies_member_offset():
    edl = EDL(keep_ranges=[KeepRange(start=0.0, end=1.0)])
    fg = _build_concat_filtergraph(edl, member_start=100.0, post_video="fps=30")
    assert "trim=start=100.000:end=101.000" in fg
    assert "atrim=start=100.000:end=101.000" in fg


def test_concat_filtergraph_multiple_ranges():
    edl = EDL(keep_ranges=[KeepRange(start=0.0, end=1.0), KeepRange(start=3.0, end=4.0)])
    fg = _build_concat_filtergraph(edl, member_start=0.0, post_video="fps=30")
    assert "[v0]" in fg and "[v1]" in fg
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[vc][ac]" in fg


def test_concat_filtergraph_time_base_shift():
    # 入力シーク併用: trim 時刻 = start + member_start - shift。
    # range(member-WAV) 100-110, member_start=5 → 絶対 105-115。shift=100 → trim 5-15。
    edl = EDL(keep_ranges=[KeepRange(start=100.0, end=110.0)])
    fg = _build_concat_filtergraph(
        edl, member_start=5.0, post_video="fps=30", time_base_shift=100.0
    )
    assert "trim=start=5.000:end=15.000" in fg
    assert "atrim=start=5.000:end=15.000" in fg


def test_concat_filtergraph_shift_clamps_negative_to_zero():
    # shift がトピック先頭より大きい (マージン取り) 場合でも負時刻にならない
    edl = EDL(keep_ranges=[KeepRange(start=10.0, end=12.0)])
    fg = _build_concat_filtergraph(
        edl, member_start=0.0, post_video="fps=30", time_base_shift=10.5
    )
    # start 10 - 10.5 = -0.5 → 0 にクランプ、end 12 - 10.5 = 1.5
    assert "trim=start=0.000:end=1.500" in fg


def test_concat_filtergraph_scales_to_many_ranges():
    # 旧 select 式は ~100 区間で ffmpeg パーサが破綻 (exit 244)。
    # trim+concat は区間数に比例した素直なグラフで耐える。
    n = 150
    edl = EDL(keep_ranges=[
        KeepRange(start=float(i * 2), end=float(i * 2 + 1)) for i in range(n)
    ])
    fg = _build_concat_filtergraph(edl, member_start=0.0, post_video="fps=30")
    assert f"concat=n={n}:v=1:a=1[vc][ac]" in fg
    # 各区間で v/a の trim が 1 つずつ
    assert fg.count("]trim=start=") == n
    assert fg.count("]atrim=start=") == n


def test_escape_filter_path_escapes_colon():
    from pathlib import Path
    out = _escape_filter_path(Path("/tmp/a:b/clip.ass"))
    assert "\\:" in out


# ---------------------------------------------------------------------------
# rolling subtitles (shusantv-style)
# ---------------------------------------------------------------------------

from src.video.subtitles import _rolling_windows, build_ass_rolling  # noqa: E402


def test_rolling_one_window_per_word():
    words = [_kw(c, i * 0.5, (i + 1) * 0.5) for i, c in enumerate("本日は質問")]
    wins = _rolling_windows(words, cap_chars=100, break_after=None)
    assert len(wins) == len(words)


def test_rolling_bright_dim_split_at_break():
    words = [_kw(c, i * 0.5, (i + 1) * 0.5) for i, c in enumerate("本日は質問")]
    # break after index 2 (は): は-and-before become dim once past it
    wins = _rolling_windows(words, cap_chars=100, break_after={2})
    # last window (index 4): dim should be 本日は, bright 質問
    s, e, lo, i, b = wins[-1]
    dim = "".join(words[j].word for j in range(lo, max(lo, b)))
    bright = "".join(words[j].word for j in range(max(lo, b), i + 1))
    assert dim == "本日は"
    assert bright == "質問"


def test_rolling_window_respects_char_cap():
    words = [_kw(c, i * 0.3, (i + 1) * 0.3) for i, c in enumerate("あ" * 20)]
    wins = _rolling_windows(words, cap_chars=5, break_after=None)
    # final window shows at most ~5 chars (win_lo advanced)
    s, e, lo, i, b = wins[-1]
    assert (i - lo + 1) <= 5


def test_rolling_resets_at_sentence_end():
    words = [_kw(c, i * 0.4, (i + 1) * 0.4) for i, c in enumerate("あい。うえ")]
    wins = _rolling_windows(words, cap_chars=100, break_after=None)
    # after 。 (index 2), window restarts → last window only has うえ
    s, e, lo, i, b = wins[-1]
    assert lo == 3  # seg_start moved past 。


def test_build_ass_rolling_has_color_tags_and_dialogues():
    words = [_kw(c, i * 0.5, (i + 1) * 0.5) for i, c in enumerate("本日は質問")]
    ass = build_ass_rolling(words, break_after={2})
    assert ass.count("Dialogue:") == len(words)
    from src.video.subtitles import _TEXT_COLOUR as _TC
    assert f"{_TC}&" in ass  # bright color tag present
    assert "1080" in ass and "1920" in ass  # vertical res


# ---------------------------------------------------------------------------
# karaoke + sentence-first pagination
# ---------------------------------------------------------------------------

from src.video.subtitles import _paginate, build_ass_karaoke  # noqa: E402


def test_paginate_sentence_first_ignores_clause_when_fits():
    words = [_kw(c, i * 0.3, (i + 1) * 0.3) for i, c in enumerate("本日は晴れ。明日は雨。")]
    # clause break after は (idx 2) must be ignored — sentence fits
    pages = _paginate(words, page_chars=100, break_after={2})
    assert pages == [(0, 6), (6, 11)]  # one page per sentence


def test_paginate_falls_back_to_clause_when_over_budget():
    text = "あ" * 10 + "、" + "い" * 10 + "。"
    words = [_kw(c, i * 0.3, (i + 1) * 0.3) for i, c in enumerate(text)]
    breaks = {i for i, c in enumerate(text) if c == "、"}
    pages = _paginate(words, page_chars=12, break_after=breaks)
    # over budget → split at the 、 clause boundary
    assert len(pages) >= 2
    assert "".join(words[j].word for j in range(*pages[0])).endswith("、")


def test_karaoke_one_dialogue_per_unit_full_text_shown():
    from src.video.subtitles import _TEXT_COLOUR as _TC2
    # 5 chars + 。, break after idx 1 and 4 → units [本日][は質問][。]
    words = [_kw(c, i * 0.4, (i + 1) * 0.4) for i, c in enumerate("本日は質問。")]
    ass = build_ass_karaoke(words, break_after={1, 4}, max_chars=99)
    dlgs = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert len(dlgs) == 3  # one per UNIT, not per char
    # the full sentence text appears in each dialogue (whole page visible)
    for d in dlgs:
        body = d.split(",", 9)[-1]
        plain = (
            body.replace("{\\c" + _TC2 + "&}", "")
            .replace("{\\c&H00A2A9A9&}", "")
            .replace("\\N", "")
        )
        assert plain == "本日は質問。"
    # every dialogue has a bright run (the current unit) and the others dim
    assert all(f"{_TC2}&" in d for d in dlgs)


def test_karaoke_big_font_vertical():
    ass = build_ass_karaoke([_kw("あ", 0.0, 0.5)])
    style = next(ln for ln in ass.splitlines() if ln.startswith("Style:"))
    fontsize = int(style.split(",")[2])
    assert fontsize >= 88  # big for mobile


# ---------------------------------------------------------------------------
# kinsoku (禁則) + 文節-unit highlight
# ---------------------------------------------------------------------------

from src.video.subtitles import _highlight_units, _wrap_break_positions  # noqa: E402


def test_highlight_units_split_at_breaks():
    assert _highlight_units(0, 7, {2, 5}) == [[0, 1, 2], [3, 4, 5], [6]]


def test_highlight_units_no_breaks_single_unit():
    assert _highlight_units(0, 4, set()) == [[0, 1, 2, 3]]


def test_wrap_no_punctuation_at_line_start():
    chars = list("これはテストです。次の文です。")
    breaks = _wrap_break_positions(chars, max_chars=6)
    for i in breaks:
        assert chars[i] not in "。、）」", f"punctuation {chars[i]} at line start"


def test_wrap_no_open_bracket_at_line_end():
    chars = list("これは（重要）です")
    breaks = _wrap_break_positions(chars, max_chars=4)
    for i in breaks:
        # char before break must not be an opening bracket
        assert chars[i - 1] not in "（「『", "open bracket at line end"


def test_karaoke_highlights_whole_clause_unit():
    # 5 chars, break after idx 1 → 2 units: [本日],[は質問]
    words = [_kw(c, i * 0.4, (i + 1) * 0.4) for i, c in enumerate("本日は質問")]
    ass = build_ass_karaoke(words, break_after={1}, max_chars=99)
    dlgs = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    # one dialogue per UNIT (2), not per char (5)
    assert len(dlgs) == 2


def test_wrap_breaks_only_at_unit_boundaries():
    # units: 先ほど(0) 直前(1) 吉田委員から(2) 児童扶養手当の(3)
    texts = ["先ほど", "直前", "吉田委員から", "児童扶養手当の"]
    chars, char_unit = [], []
    for ui, t in enumerate(texts):
        for c in t:
            chars.append(c)
            char_unit.append(ui)
    breaks = _wrap_break_positions(chars, max_chars=10, char_unit=char_unit)
    # no break may fall inside a unit
    for b in breaks:
        assert char_unit[b] != char_unit[b - 1], f"break inside unit at {b}"


def test_wrap_single_long_unit_force_splits():
    # one unit longer than max_chars must still split (fallback)
    chars = list("あ" * 15)
    char_unit = [0] * 15
    breaks = _wrap_break_positions(chars, max_chars=6, char_unit=char_unit)
    assert len(breaks) >= 2  # 15 chars / 6 → at least 2 breaks


def test_karaoke_kinsoku_no_punct_at_line_start():
    import re
    # 文節境界の直後に句点 unit が来るケース → 。が行頭に来てはいけない
    text = "内閣委員会でもさせていただきました。"
    words = [_kw(c, i * 0.3, (i + 1) * 0.3) for i, c in enumerate(text)]
    ass = build_ass_karaoke(words, break_after={5, len(words) - 2}, max_chars=8)
    dlg = next(
        ln for ln in ass.splitlines() if ln.startswith("Dialogue: 0,")
    )
    plain = re.sub(r"\{[^}]*\}", "", dlg.split(",,", 1)[-1])
    for line in plain.split("\\N")[1:]:
        assert not (line and line[0] in "。、，．！？）」"), \
            f"punctuation at line head: {plain}"


def test_kinsoku_does_not_split_latin_run():
    from src.video.subtitles import _apply_kinsoku
    chars = list("AIの悪用による")
    # a candidate break between A and I (index 1) must be removed/moved
    assert 1 not in _apply_kinsoku(chars, {1})
    # break between Claude letters too
    chars2 = list("Claude3を使う")
    for b in range(1, 6):  # inside "Claude3"
        assert b not in _apply_kinsoku(chars2, {b})


# ---------------------------------------------------------------------------
# build_ass_from_captions (edited captions → ASS)
# ---------------------------------------------------------------------------

from src.models import EditCaption  # noqa: E402
from src.video.subtitles import build_ass_from_captions  # noqa: E402


def test_build_ass_from_captions_basic():
    caps = [
        EditCaption(start=0.0, end=1.5, text="本日は質問します。"),
        EditCaption(start=1.5, end=3.0, text="よろしくお願いします。"),
    ]
    ass = build_ass_from_captions(caps, title="テスト｜タイトル")
    # 2 captions + 白パネルタイトル (Plate ドローイング + Big 要旨; 見出し header 無し)
    assert ass.count("Dialogue:") == 4
    assert "本日は質問します。" in ass
    assert "よろしくお願いします。" in ass
    # 白パネルタイトルのスタイル群が存在する
    assert "Style: Plate" in ass
    assert "Style: Big" in ass
    # 白パネル: \p ドローイング矩形が出る
    assert "\\p1" in ass


def test_build_ass_from_captions_skips_empty():
    caps = [EditCaption(start=0, end=1, text="  "), EditCaption(start=1, end=2, text="あり")]
    ass = build_ass_from_captions(caps)
    assert ass.count("Dialogue:") == 1


# ---------------------------------------------------------------------------
# 白パネルタイトル (Plate + Head + Big) の幾何
# ---------------------------------------------------------------------------

from src.video.subtitles import _build_title_layout  # noqa: E402


def test_title_header_emits_head_and_big():
    caps = [EditCaption(start=0.0, end=1.5, text="本文。")]
    ass = build_ass_from_captions(
        caps, title="研究開発への影響と｜制度設計の透明性",
        title_header=["2026年6月12日（金）", "厚生労働委員会 古川あおい"],
    )
    # 見出しあり → Plate + Head + Big の 3 タイトル Dialogue + 1 caption
    assert ass.count("Dialogue:") == 4
    assert "Style: Head" in ass
    assert ",Head,," in ass and ",Big,," in ass and ",Plate,," in ass
    # 見出し文字列が出る
    assert "2026年6月12日" in ass and "古川あおい" in ass


def test_title_panel_height_grows_with_lines():
    # 要旨の行数が増えるとパネル下端 (= 高さ) が増える
    hdr = ["2026年6月12日（金）", "厚生労働委員会 古川あおい"]
    one = _build_title_layout("短い要旨", hdr, res_x=1080, res_y=1920,
                              margin_h=40, title_fontsize=100)
    three = _build_title_layout("研究開発への影響と｜制度設計の透明性と｜国民への説明責任",
                                hdr, res_x=1080, res_y=1920,
                                margin_h=40, title_fontsize=100)
    h1 = one.plate_bottom - one.plate_top
    h3 = three.plate_bottom - three.plate_top
    assert h3 > h1
    # 見出しが折返されず想定行数に収まる (「い」だけ孤立しない)
    assert all(line.count("\\N") == 0 for line in one.head_lines)


def test_title_no_header_starts_at_top():
    # 見出し無しだと要旨がパネル上端側から始まる (big_top == head_top)
    lay = _build_title_layout("研究開発への影響と｜制度設計の透明性", None,
                              res_x=1080, res_y=1920, margin_h=40, title_fontsize=100)
    assert lay.head_lines == []
    assert lay.big_top == lay.head_top


# ---------------------------------------------------------------------------
# 話者色分け (質疑=ミント, 答弁=オレンジ)
# ---------------------------------------------------------------------------

from src.video.subtitles import (  # noqa: E402
    _ANSWER_COLOUR,
    _TEXT_COLOUR,
    _role_colour,
    _role_colour_at,
)


def test_role_colour_answer_vs_question():
    assert _role_colour("質疑者") == _TEXT_COLOUR
    assert _role_colour("答弁者") == _ANSWER_COLOUR
    assert _role_colour("政府参考人") == _ANSWER_COLOUR
    assert _role_colour("参考人") == _ANSWER_COLOUR
    assert _role_colour("委員長") == _TEXT_COLOUR  # 既定はミント
    assert _role_colour(None) == _TEXT_COLOUR


def test_role_colour_at_uses_spans():
    spans = [(0.0, 5.0, "質疑者"), (5.0, 10.0, "答弁者")]
    assert _role_colour_at(2.0, spans) == _TEXT_COLOUR
    assert _role_colour_at(7.0, spans) == _ANSWER_COLOUR
    assert _role_colour_at(99.0, spans) == _TEXT_COLOUR  # span 外は既定
    assert _role_colour_at(2.0, None) == _TEXT_COLOUR


def test_captions_coloured_by_role_span():
    caps = [
        EditCaption(start=1.0, end=2.0, text="質問です"),
        EditCaption(start=6.0, end=7.0, text="答弁です"),
    ]
    spans = [(0.0, 5.0, "質疑者"), (5.0, 10.0, "答弁者")]
    ass = build_ass_from_captions(caps, role_spans=spans)
    dlgs = [l for l in ass.splitlines() if l.startswith("Dialogue: 0,")]
    assert _TEXT_COLOUR in dlgs[0]    # 質疑=ミント
    assert _ANSWER_COLOUR in dlgs[1]  # 答弁=オレンジ


def test_karaoke_coloured_by_role_span():
    # 質疑区間と答弁区間で明色が変わる。break_after で文節 (ハイライト単位) を分け、
    # 各単位の発話時刻で色が引かれる。「質問だ」(0-1.5)=質疑, 「答弁だ」(1.5-3)=答弁。
    words = [_kw(c, i * 0.5, (i + 1) * 0.5) for i, c in enumerate("質問だ答弁だ")]
    spans = [(0.0, 1.5, "質疑者"), (1.5, 3.0, "答弁者")]
    ass = build_ass_karaoke(words, role_spans=spans, max_chars=99, break_after={2})
    assert _ANSWER_COLOUR in ass  # 答弁単位にオレンジが出る
    assert _TEXT_COLOUR in ass    # 質疑単位にミントが出る


# ---------------------------------------------------------------------------
# 常時ヘッダー (Banner)
# ---------------------------------------------------------------------------

from src.video.subtitles import _banner_event  # noqa: E402


def test_banner_event_spans_title_to_end():
    b = _banner_event(["2026年6月19日 厚生労働委員会", "辰巳孝太郎 質疑"], 2.0, 60.0)
    assert b.startswith("Dialogue: 0,0:00:02.00,0:01:00.00,Banner,,")
    assert "辰巳孝太郎 質疑" in b
    assert "\\N" in b  # 2 行が \N 連結


def test_banner_event_empty_when_no_header_or_bad_range():
    assert _banner_event(None, 2.0, 60.0) == ""
    assert _banner_event([], 2.0, 60.0) == ""
    assert _banner_event(["x"], 5.0, 3.0) == ""  # end <= start


def test_karaoke_emits_banner_after_title():
    words = [_kw(c, i * 0.5, (i + 1) * 0.5)
             for i, c in enumerate("質問ですよろしくね今日も")]  # ~6s > title 2s
    ass = build_ass_karaoke(
        words, title="t", title_header=["日付 委員会", "議員 質疑"], title_seconds=2.0
    )
    assert "Style: Banner," in ass
    assert ",Banner,," in ass
    # title_header 無しなら banner 出ない
    ass2 = build_ass_karaoke(words, title="t", title_seconds=2.0)
    assert ",Banner,," not in ass2


def test_captions_emit_banner():
    caps = [EditCaption(start=0, end=10, text="本文")]
    ass = build_ass_from_captions(caps, title="t", title_header=["日付", "議員"])
    assert ",Banner,," in ass
