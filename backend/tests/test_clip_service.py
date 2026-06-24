"""clip_service.py の単体テスト (純ロジック、ネットワーク不要)。"""

from __future__ import annotations

import pytest

from src.clip_service import _resolve_member, _safe_name
from src.models import SessionDetail


def _detail(speakers) -> SessionDetail:
    return SessionDetail(
        chamber="shugiin", session_id="56325", date="2026-06-12",
        committee="内閣委員会", hls_url="https://x/y.m3u8", source_url="https://x",
        speakers=speakers,
    )


def _spk(name, start, role="質疑者", aff="自由民主党"):
    from src.models import SpeakerInfo
    return SpeakerInfo(
        name=name, affiliation=aff, role=role,
        start_seconds=start, start_time="", duration_minutes=5,
    )


def test_resolve_member_exact():
    d = _detail([_spk("山下貴司", 100.0), _spk("高市早苗", 500.0)])
    idx, spk = _resolve_member(d, "高市早苗")
    assert idx == 1
    assert spk.name == "高市早苗"


def test_resolve_member_fuzzy_surname():
    d = _detail([_spk("山下貴司", 100.0), _spk("高市早苗", 500.0)])
    idx, spk = _resolve_member(d, "高市")  # 2-char surname prefix
    assert spk.name == "高市早苗"
    assert idx == 1


def test_resolve_member_not_found_raises():
    d = _detail([_spk("山下貴司", 100.0)])
    with pytest.raises(ValueError, match="not found"):
        _resolve_member(d, "存在しない議員")


def test_resolve_member_picks_correct_index_distinct_names():
    d = _detail([_spk("田中一郎", 100.0), _spk("田村二郎", 300.0), _spk("佐藤花子", 600.0)])
    idx, spk = _resolve_member(d, "佐藤花子")
    assert idx == 2




def test_safe_name():
    assert _safe_name("山下 貴司") == "山下_貴司"
    assert _safe_name("a/b") == "a_b"
    assert _safe_name("全角　空白") == "全角_空白"


from src.clip_service import _remap_caption_time  # noqa: E402
from src.models import EditRange  # noqa: E402


def test_remap_caption_time_kept_range_unchanged():
    orig = [EditRange(start=0, end=10), EditRange(start=10, end=20)]
    enabled = orig
    assert abs(_remap_caption_time(5, orig, enabled) - 5.0) < 1e-6


def test_remap_caption_time_after_disabled_shifts_earlier():
    orig = [EditRange(start=0, end=10), EditRange(start=10, end=20), EditRange(start=20, end=30)]
    enabled = [EditRange(start=0, end=10), EditRange(start=20, end=30)]  # middle off
    # full-time 25 → source 25 (range3) → new 15
    assert abs(_remap_caption_time(25, orig, enabled) - 15.0) < 1e-6


def test_remap_caption_time_in_disabled_region_clamps():
    orig = [EditRange(start=0, end=10), EditRange(start=10, end=20), EditRange(start=20, end=30)]
    enabled = [EditRange(start=0, end=10), EditRange(start=20, end=30)]
    # full-time 15 is in disabled middle → clamps to boundary (10)
    assert _remap_caption_time(15, orig, enabled) == 10.0


# ---------------------------------------------------------------------------
# apply_tree_to_ranges: ツリー leaf enabled → ranges.enabled 再計算
# ---------------------------------------------------------------------------

from src.clip_service import (  # noqa: E402
    _enabled_leaf_spans,
    _filter_edl_by_spans,
    apply_tree_to_ranges,
)
from src.models import (  # noqa: E402
    EDL,
    ClipProject,
    KeepRange,
    KeptWord,
    QASentence,
    QATopic,
    QATree,
    QATurn,
)


def _proj(ranges: list[tuple[float, float]], tree: QATree | None) -> ClipProject:
    return ClipProject(
        session_id="s", member="m", source_video="m_src.mp4", member_start=0.0,
        ranges=[EditRange(start=s, end=e, enabled=True) for s, e in ranges],
        qa_tree=tree,
    )


def _tree_one_topic(sents: list[tuple[float, float, bool]]) -> QATree:
    return QATree(topics=[QATopic(index=0, question_speaker="A", turns=[
        QATurn(speaker="A", role="質疑者", sentences=[
            QASentence(text=f"s{i}", start=s, end=e, enabled=en)
            for i, (s, e, en) in enumerate(sents)
        ])
    ])])


def test_apply_tree_none_leaves_ranges_untouched():
    p = _proj([(0, 5), (5, 10)], None)
    apply_tree_to_ranges(p)
    assert all(r.enabled for r in p.ranges)  # 変化なし


def test_apply_tree_all_enabled_is_identity():
    # 文 span が全 range を覆う → 全 range 有効のまま
    p = _proj([(0, 5), (5, 10), (10, 15)], _tree_one_topic([(0, 15, True)]))
    apply_tree_to_ranges(p)
    assert [r.enabled for r in p.ranges] == [True, True, True]


def test_apply_tree_disables_ranges_under_low_leaf():
    # 文0 (0-10) 有効、文1 (10-20) 無効 → 中点が 10-20 の range だけ off
    p = _proj([(0, 4), (5, 9), (12, 18), (20, 24)],
              _tree_one_topic([(0, 10, True), (10, 20, False)]))
    apply_tree_to_ranges(p)
    # mids: 2,7 (in 0-10 → on), 15 (in 10-20 disabled → off), 22 (no span → off)
    assert [r.enabled for r in p.ranges] == [True, True, False, False]


def test_apply_tree_range_in_gap_is_off():
    # 有効文は 0-5 のみ。10-15 の range は中点 12.5 がどの span にも入らず off
    p = _proj([(0, 5), (10, 15)], _tree_one_topic([(0, 5, True)]))
    apply_tree_to_ranges(p)
    assert [r.enabled for r in p.ranges] == [True, False]


def test_apply_tree_straddle_decided_by_midpoint():
    # range 8-12 は文 (0-10 有効) と (10-20 無効) を跨ぐ。中点 10 は境界 → 有効側
    p = _proj([(8, 12)], _tree_one_topic([(0, 10, True), (10, 20, False)]))
    apply_tree_to_ranges(p)
    assert p.ranges[0].enabled is True  # mid=10 <= 10+eps → in enabled span


def test_apply_tree_all_off_raises():
    p = _proj([(0, 5)], _tree_one_topic([(0, 5, False)]))
    with pytest.raises(ValueError, match="選択が空"):
        apply_tree_to_ranges(p)


def test_enabled_leaf_spans_merges_adjacent():
    # 隣接/重なる span は結合される
    tree = _tree_one_topic([(0, 5, True), (5, 10, True), (20, 25, True)])
    assert _enabled_leaf_spans(tree) == [(0.0, 10.0), (20.0, 25.0)]


# ---------------------------------------------------------------------------
# _filter_edl_by_spans: option b 用 EDL 絞り込み + new 時刻再計算
# ---------------------------------------------------------------------------


def _kw(word, os_, oe, ns, ne) -> KeptWord:
    return KeptWord(word=word, old_start=os_, old_end=oe, new_start=ns, new_end=ne)


def test_member_to_post_cut_maps_through_enabled_ranges():
    from src.clip_service import _member_to_post_cut
    er = [EditRange(start=0, end=10), EditRange(start=20, end=30)]  # 10-20 cut
    assert _member_to_post_cut(5, er) == 5.0       # range0
    assert _member_to_post_cut(25, er) == 15.0     # range1 → 10+(25-20)
    assert _member_to_post_cut(15, er) == 10.0     # cut gap → next range head


def test_role_spans_post_cut_maps_turns():
    from src.clip_service import _role_spans_post_cut
    er = [EditRange(start=0, end=10), EditRange(start=20, end=30)]
    tree = QATree(topics=[QATopic(index=0, question_speaker="A", turns=[
        QATurn(speaker="A", role="質疑者",
               sentences=[QASentence(text="q", start=0, end=10)]),
        QATurn(speaker="B", role="答弁者",
               sentences=[QASentence(text="a", start=20, end=30)]),
    ])])
    spans = _role_spans_post_cut(tree, er)
    # 答弁ターンは member-WAV 20-30 → post-cut 10-20 に前詰め
    assert spans == [(0.0, 10.0, "質疑者"), (10.0, 20.0, "答弁者")]


def test_role_spans_post_cut_none_tree():
    from src.clip_service import _role_spans_post_cut
    assert _role_spans_post_cut(None, [EditRange(start=0, end=5)]) == []


def test_filter_edl_keeps_only_enabled_and_remaps():
    # 3 range, 中央を落とす。残った 2 range で new 時刻が前詰めされる
    edl = EDL(
        keep_ranges=[KeepRange(start=0, end=10), KeepRange(start=10, end=20),
                     KeepRange(start=20, end=30)],
        kept_words=[
            _kw("a", 1, 2, 1, 2),      # range0
            _kw("b", 12, 13, 12, 13),  # range1 (落とす)
            _kw("c", 22, 23, 22, 23),  # range2 → new は 10+2=12 付近に前詰め
        ],
    )
    spans = [(0.0, 10.0), (20.0, 30.0)]  # 中央 10-20 を除外
    out = _filter_edl_by_spans(edl, spans)
    assert len(out.keep_ranges) == 2
    words = {w.word: w for w in out.kept_words}
    assert "b" not in words  # 落ちた
    assert "a" in words and "c" in words
    # c の new_start = range0長(10) + (22-20) = 12
    assert abs(words["c"].new_start - 12.0) < 1e-6


# ---------------------------------------------------------------------------
# render_topic_clips: トピック別クリップ (render_clip はモック)
# ---------------------------------------------------------------------------

import src.clip_service as clip_service_mod  # noqa: E402
from src.models import EditCaption  # noqa: E402


def _two_topic_project(tmp_path):
    # トピック0: 質疑0-10 + 答弁10-20 / トピック1: 質疑30-40
    # ranges は member-WAV で文に対応。caption も全 range タイムライン (=この例では等価)。
    tree = QATree(topics=[
        QATopic(index=0, label="トピックゼロ", question_speaker="A", turns=[
            QATurn(speaker="A", role="質疑者",
                   sentences=[QASentence(text="q0", start=0, end=10)]),
            QATurn(speaker="B", role="答弁者",
                   sentences=[QASentence(text="a0", start=10, end=20)]),
        ]),
        QATopic(index=1, label="トピックイチ", question_speaker="A", turns=[
            QATurn(speaker="A", role="質疑者",
                   sentences=[QASentence(text="q1", start=30, end=40)]),
        ]),
    ])
    proj = ClipProject(
        session_id="s", member="議員", source_video="src.mp4", member_start=0.0,
        ranges=[EditRange(start=0, end=10), EditRange(start=10, end=20),
                EditRange(start=30, end=40)],
        captions=[EditCaption(start=0, end=5, text="q0です"),
                  EditCaption(start=10, end=15, text="a0です"),
                  EditCaption(start=20, end=25, text="q1です")],
        title="登壇全体の要旨", title_header=["2026年6月19日", "厚生労働委員会 議員 質疑"],
        qa_tree=tree,
    )
    (tmp_path / "src.mp4").write_bytes(b"fake")  # source 存在チェック用
    return proj


def test_render_topic_clips_windows_and_titles(tmp_path, monkeypatch):
    captured = []

    def fake_render(source, edl, out_path, **kw):
        captured.append((out_path.name, [(r.start, r.end) for r in edl.keep_ranges]))
        out_path.write_bytes(b"clip")
        return out_path

    monkeypatch.setattr(clip_service_mod, "render_clip", fake_render)
    proj = _two_topic_project(tmp_path)
    out = clip_service_mod.render_topic_clips(proj, tmp_path)

    assert len(out) == 2
    # 出力ファイル名はトピック index 付き
    names = {c["clip_path"].split("/")[-1] for c in out}
    assert names == {"議員_topic0_clip.mp4", "議員_topic1_clip.mp4"}
    # トピック0 の window は range 0-10,10-20 (質疑+答弁)、トピック1 は 30-40
    by_name = dict(captured)
    assert by_name["議員_topic0_clip.mp4"] == [(0.0, 10.0), (10.0, 20.0)]
    assert by_name["議員_topic1_clip.mp4"] == [(30.0, 40.0)]
    # ラベルが返る
    labels = {c["topic_index"]: c["topic_label"] for c in out}
    assert labels == {0: "トピックゼロ", 1: "トピックイチ"}


def test_render_topic_clips_requires_tree(tmp_path):
    proj = ClipProject(
        session_id="s", member="議員", source_video="src.mp4", member_start=0.0,
        ranges=[EditRange(start=0, end=10)], qa_tree=None,
    )
    with pytest.raises(ValueError, match="qa_tree"):
        clip_service_mod.render_topic_clips(proj, tmp_path)


def test_caption_in_ranges_filters_by_window():
    from src.clip_service import _caption_in_ranges
    orig = [EditRange(start=0, end=10), EditRange(start=10, end=20),
            EditRange(start=30, end=40)]
    # window = トピック1 (30-40 が post-cut 20-30 に来る)
    window = [EditRange(start=30, end=40)]
    # caption (post-cut 20-25) は source 30-35 → window 内
    assert _caption_in_ranges(EditCaption(start=20, end=25, text="x"), orig, window)
    # caption (post-cut 2-5) は source 2-5 → window 外
    assert not _caption_in_ranges(EditCaption(start=2, end=5, text="y"), orig, window)


# ---------------------------------------------------------------------------
# トピックプレビュー: window ローカル時間 + オフ文スキップ区間
# ---------------------------------------------------------------------------

from src.clip_service import (  # noqa: E402
    _disabled_spans_in_window,
    _window_local_time,
)


def test_window_local_time_maps_through_window():
    # window: 10-20, 30-40 (member-WAV)。連結プレビューは 0-10(=10-20), 10-20(=30-40)。
    w = [EditRange(start=10, end=20), EditRange(start=30, end=40)]
    assert _window_local_time(10, w) == 0.0
    assert _window_local_time(15, w) == 5.0
    assert _window_local_time(30, w) == 10.0      # 2つ目 range 先頭 = 累積 10
    assert _window_local_time(35, w) == 15.0
    assert _window_local_time(25, w) is None      # window 外 (gap)


def test_disabled_spans_in_window_maps_off_sentences():
    # window 10-40 (全 range)。オフ文 (15-20 member-WAV) → ローカル 5-10。
    tree = QATree(topics=[QATopic(index=0, question_speaker="A", turns=[
        QATurn(speaker="A", role="質疑者", sentences=[
            QASentence(text="on", start=10, end=15, enabled=True),
            QASentence(text="off", start=15, end=20, enabled=False),
            QASentence(text="on2", start=20, end=30, enabled=True),
        ])])])
    w = [EditRange(start=10, end=30)]
    spans = _disabled_spans_in_window(tree, w)
    assert spans == [[5.0, 10.0]]  # 15-20 member → local 5-10


def test_disabled_spans_empty_when_all_enabled():
    tree = QATree(topics=[QATopic(index=0, question_speaker="A", turns=[
        QATurn(speaker="A", role="質疑者", sentences=[
            QASentence(text="a", start=0, end=5, enabled=True),
        ])])])
    assert _disabled_spans_in_window(tree, [EditRange(start=0, end=5)]) == []


# ---------------------------------------------------------------------------
# _karaoke_for_window: _edl.json から window-local カラオケ語列 + break_after
# ---------------------------------------------------------------------------

from src.clip_service import _karaoke_for_window  # noqa: E402


def test_karaoke_for_window_remaps_words_and_breaks(tmp_path):
    # _edl.json を用意 (語は member-WAV old_*, 全体 new_* 付き)。window=10-30。
    edl = EDL(
        keep_ranges=[KeepRange(start=10, end=30)],
        kept_words=[
            KeptWord(word="あ", old_start=11, old_end=12, new_start=1, new_end=2),
            KeptWord(word="。", old_start=14, old_end=15, new_start=4, new_end=5),
            KeptWord(word="い", old_start=20, old_end=21, new_start=10, new_end=11),
            KeptWord(word="ろ", old_start=40, old_end=41, new_start=30, new_end=31),  # window外
        ],
    )
    (tmp_path / "議員_edl.json").write_text(edl.model_dump_json(), encoding="utf-8")
    # 文末 14.5 (member-WAV) を持つ qa_tree → break_after に「。」の index が入る
    tree = QATree(topics=[QATopic(index=0, question_speaker="A", turns=[
        QATurn(speaker="A", role="質疑者", sentences=[
            QASentence(text="あ。", start=11, end=15, enabled=True),
            QASentence(text="い", start=20, end=21, enabled=True),
        ])])])
    proj = ClipProject(
        session_id="s", member="議員", source_video="議員_src.mp4", member_start=0.0,
        ranges=[EditRange(start=10, end=30)], qa_tree=tree,
    )
    out_path = tmp_path / "議員_topic0_preview.mp4"
    kw, breaks = _karaoke_for_window(proj, [EditRange(start=10, end=30)], out_path)
    # window 外の「ろ」は除外、3語残る
    assert [w.word for w in kw] == ["あ", "。", "い"]
    # window-local 時間: old 11→local 1 (range開始10), old 20→local 10
    assert abs(kw[0].new_start - 1.0) < 1e-6
    assert abs(kw[2].new_start - 10.0) < 1e-6
    # 文末14.5 をまたぐ語 (「。」= old_end 15) が break_after に
    assert breaks is not None and 1 in breaks


def test_karaoke_for_window_no_edl_falls_back(tmp_path):
    # _edl.json が無ければ ([], None) → 呼び出し側はプレーン字幕にフォールバック
    proj = ClipProject(
        session_id="s", member="議員", source_video="議員_src.mp4", member_start=0.0,
        ranges=[EditRange(start=0, end=5)],
    )
    kw, breaks = _karaoke_for_window(proj, [EditRange(start=0, end=5)], tmp_path / "議員_x.mp4")
    assert kw == [] and breaks is None
