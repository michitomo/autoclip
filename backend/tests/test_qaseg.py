"""Q&A 区間分割の単体テスト (LLM 話者タグは除く純ロジック)。"""

from __future__ import annotations

from src.models import QASegment, TimedUtterance, Utterance, WhisperWord
from src.video.qaseg import (
    _split_sentences_keep,
    attach_times_to_utterances,
    build_qa_tree,
    group_into_qa,
)


def _w(word: str, s: float, e: float) -> WhisperWord:
    return WhisperWord(word=word, start=s, end=e)


def _words(text: str, per: float = 0.5) -> list[WhisperWord]:
    return [_w(c, i * per, (i + 1) * per) for i, c in enumerate(text)]


def test_attach_times_partitions_words_in_order():
    words = _words("質問です答弁します")
    utts = [
        Utterance(speaker="A", role="質疑者", text="質問です"),
        Utterance(speaker="B", role="答弁者", text="答弁します"),
    ]
    timed = attach_times_to_utterances(utts, words)
    assert len(timed) == 2
    assert timed[0].start == 0.0
    assert abs(timed[0].end - 2.0) < 1e-6  # 4 chars * 0.5
    assert abs(timed[1].start - 2.0) < 1e-6
    assert timed[1].speaker == "B"


def test_group_into_qa_pairs_question_then_answer():
    words = _words("質問です答弁します再質問です")
    utts = [
        Utterance(speaker="A", role="質疑者", text="質問です"),
        Utterance(speaker="B", role="答弁者", text="答弁します"),
        Utterance(speaker="A", role="質疑者", text="再質問です"),
    ]
    timed = attach_times_to_utterances(utts, words)
    qas = group_into_qa(timed)
    assert len(qas) == 2
    assert qas[0].question_speaker == "A"
    assert qas[0].answer_speakers == ["B"]
    assert qas[1].question_speaker == "A"


def test_group_into_qa_chair_absorbed():
    words = _words("質問です指名答弁します")
    utts = [
        Utterance(speaker="A", role="質疑者", text="質問です"),
        Utterance(speaker="C", role="委員長", text="指名"),
        Utterance(speaker="B", role="答弁者", text="答弁します"),
    ]
    timed = attach_times_to_utterances(utts, words)
    qas = group_into_qa(timed)
    # chair turn absorbed into the single Q&A
    assert len(qas) == 1
    assert qas[0].question_speaker == "A"
    assert qas[0].answer_speakers == ["B"]
    assert len(qas[0].utterances) == 3


def test_group_multiple_qa_rounds():
    words = _words("質問1答弁1質問2答弁2")
    utts = [
        Utterance(speaker="A", role="質疑者", text="質問1"),
        Utterance(speaker="B", role="答弁者", text="答弁1"),
        Utterance(speaker="A", role="質疑者", text="質問2"),
        Utterance(speaker="B", role="答弁者", text="答弁2"),
    ]
    timed = attach_times_to_utterances(utts, words)
    qas = group_into_qa(timed)
    assert len(qas) == 2


def test_attach_times_empty():
    assert attach_times_to_utterances([], _words("あ")) == []
    assert attach_times_to_utterances([Utterance(speaker="A", role="質疑者", text="あ")], []) == []


from src.video.qaseg import extract_answerers  # noqa: E402


def test_extract_answerers_from_naming():
    raw = "まず副大臣からお答えいただければ。二木厚生労働副大臣。お答えします。"
    out = extract_answerers(raw)
    names = {s.name for s in out}
    assert "二木" in names
    a = next(s for s in out if s.name == "二木")
    assert a.role == "答弁者"


def test_extract_answerers_role_for_sankounin():
    raw = "田中太郎政府参考人。お答えします。"
    out = extract_answerers(raw)
    a = next(s for s in out if s.name == "田中太郎")
    assert a.role == "政府参考人"


def test_extract_answerers_filters_garbage():
    # misrecognition fragments ending in 務/総括/官 should be dropped
    raw = "首務大臣について。木総括審議官。"
    out = extract_answerers(raw)
    names = {s.name for s in out}
    assert "首務" not in names
    assert "木総括" not in names


def test_extract_answerers_skips_chair():
    # 委員長 is not an answer title → not extracted as answerer
    raw = "大串厚生労働委員長。古川あおいくん。"
    out = extract_answerers(raw)
    assert all("委員長" not in s.affiliation for s in out)


# ---------------------------------------------------------------------------
# 文分割 (_split_sentences_keep)
# ---------------------------------------------------------------------------


def test_split_sentences_keeps_terminator():
    out = _split_sentences_keep("本日は晴天なり。質問します。")
    assert out == ["本日は晴天なり。", "質問します。"]


def test_split_sentences_no_terminator_single_leaf():
    out = _split_sentences_keep("終止符のない発言")
    assert out == ["終止符のない発言"]


def test_split_sentences_does_not_split_on_comma():
    # 読点では割らない (1 文のまま)
    out = _split_sentences_keep("まず、香料について、伺います。")
    assert out == ["まず、香料について、伺います。"]


def test_split_sentences_merges_short_fragment():
    # 「はい。」は短いので前の文に併合される (単独「。」行を防ぐ)
    out = _split_sentences_keep("お答えいたします。はい。")
    assert out == ["お答えいたします。はい。"]


def test_split_sentences_question_exclamation():
    out = _split_sentences_keep("そうですか？本当ですね！")
    assert out == ["そうですか？", "本当ですね！"]


# ---------------------------------------------------------------------------
# build_qa_tree (語時刻 → トピック>発言者>発言内容)
# ---------------------------------------------------------------------------


def _qa_from(utts: list[TimedUtterance], idx: int = 0) -> QASegment:
    q = next((u.speaker for u in utts if u.role == "質疑者"), utts[0].speaker)
    a = [u.speaker for u in utts if u.role in ("答弁者", "政府参考人")]
    return QASegment(
        index=idx, question_speaker=q, answer_speakers=a,
        start=utts[0].start, end=utts[-1].end, utterances=utts,
    )


def test_build_qa_tree_sentence_spans_tile_within_turn():
    # 2 文のターン: 文 span が連続してターンを覆い、隙間/重複なし
    words = _words("本日は質問です。香料を伺います。")  # 16 chars, 0.5s each
    utts = [TimedUtterance(speaker="A", role="質疑者",
                           text="本日は質問です。香料を伺います。",
                           start=0.0, end=8.0)]
    tree = build_qa_tree([_qa_from(utts)], words)
    assert len(tree.topics) == 1
    turn = tree.topics[0].turns[0]
    assert [s.text for s in turn.sentences] == ["本日は質問です。", "香料を伺います。"]
    s0, s1 = turn.sentences
    assert s0.start == 0.0
    assert abs(s0.end - 4.0) < 1e-6      # 「本日は質問です。」= 8 chars * 0.5
    assert abs(s1.start - 4.0) < 1e-6    # 次文は直後から
    assert abs(s1.end - 8.0) < 1e-6
    # 文 span がターン span (= utterance) 内にネスト
    assert s0.start >= 0.0 and s1.end <= 8.0 + 1e-6


def test_build_qa_tree_two_turns_no_cross_contamination():
    # 質問ターン → 答弁ターン。文 span が次ターンへ食い込まない
    words = _words("質問です。答弁します。")  # 質問です。=5, 答弁します。=6
    utts = [
        TimedUtterance(speaker="A", role="質疑者", text="質問です。",
                       start=0.0, end=2.5),
        TimedUtterance(speaker="B", role="答弁者", text="答弁します。",
                       start=2.5, end=5.5),
    ]
    tree = build_qa_tree([_qa_from(utts)], words)
    turns = tree.topics[0].turns
    assert len(turns) == 2
    assert turns[0].speaker == "A" and turns[1].speaker == "B"
    # ターン A の最後の文は ターン B の最初の文より前で終わる
    assert turns[0].sentences[-1].end <= turns[1].sentences[0].start + 1e-6


def test_build_qa_tree_turn_spans_match_attach_times_regression():
    # build_qa_tree のターン span (= 文 span の端) が
    # 旧 attach_times_to_utterances のターン span と一致する (回帰ガード)
    words = _words("質問です答弁します再質問です")
    utts_plain = [
        Utterance(speaker="A", role="質疑者", text="質問です"),
        Utterance(speaker="B", role="答弁者", text="答弁します"),
        Utterance(speaker="A", role="質疑者", text="再質問です"),
    ]
    timed = attach_times_to_utterances(utts_plain, words)
    qa = _qa_from(timed)
    tree = build_qa_tree([qa], words)
    turns = tree.topics[0].turns
    assert len(turns) == len(timed)
    for t, u in zip(turns, timed):
        assert abs(t.sentences[0].start - u.start) < 1e-6
        assert abs(t.sentences[-1].end - u.end) < 1e-6


def test_build_qa_tree_defaults_mid_enabled():
    # 注釈前は全て importance=mid, enabled=True, summary=""
    words = _words("質問です。")
    utts = [TimedUtterance(speaker="A", role="質疑者", text="質問です。",
                           start=0.0, end=2.5)]
    tree = build_qa_tree([_qa_from(utts)], words)
    s = tree.topics[0].turns[0].sentences[0]
    assert s.importance == "mid" and s.enabled is True and s.summary == ""


def test_build_qa_tree_empty_qas():
    assert build_qa_tree([], _words("あ")).topics == []


def test_response_comment_moved_to_prev_topic():
    # topic1 の質疑者ターン先頭の reply 文 (LLM 判定) を topic0 末尾へ移し、
    # 構成を 質疑→答弁→コメント にする。
    from src.video.qaseg import move_response_comments_to_prev
    from src.models import QASentence, QATopic, QATree, QATurn

    def turn(role, *texts):
        return QATurn(speaker="X", role=role,
                      sentences=[QASentence(text=t, start=0.0, end=1.0) for t in texts])

    t0q = turn("質疑者", "最初の質問です。")
    t0a = turn("答弁者", "答弁します。")
    t1q = turn("質疑者", "ありがとうございました。", "次の質問です。")
    t1a = turn("答弁者", "次の答弁です。")
    tree = QATree(topics=[
        QATopic(index=0, label="", question_speaker="A", answer_speakers=["B"], turns=[t0q, t0a]),
        QATopic(index=1, label="", question_speaker="A", answer_speakers=["B"], turns=[t1q, t1a]),
    ])
    # 「ありがとうございました。」だけ reply=true
    reply_ids = {id(t1q.sentences[0])}
    move_response_comments_to_prev(tree, reply_ids)

    # topic0 末尾に応答コメントの質疑者ターンが付く
    assert [t.role for t in tree.topics[0].turns] == ["質疑者", "答弁者", "質疑者"]
    assert tree.topics[0].turns[-1].sentences[0].text == "ありがとうございました。"
    # topic1 の質疑者ターンは本題から始まる
    assert tree.topics[1].turns[0].sentences[0].text == "次の質問です。"


def test_response_comment_not_moved_when_no_reply():
    # reply 文が無ければ移動しない (誤爆防止)。
    from src.video.qaseg import move_response_comments_to_prev
    from src.models import QASentence, QATopic, QATree, QATurn

    def turn(role, *texts):
        return QATurn(speaker="X", role=role,
                      sentences=[QASentence(text=t, start=0.0, end=1.0) for t in texts])

    tree = QATree(topics=[
        QATopic(index=0, label="", question_speaker="A", answer_speakers=["B"],
                turns=[turn("質疑者", "質問1"), turn("答弁者", "答弁1")]),
        QATopic(index=1, label="", question_speaker="A", answer_speakers=["B"],
                turns=[turn("質疑者", "これは普通の質問の続きです。"), turn("答弁者", "答弁2")]),
    ])
    move_response_comments_to_prev(tree, set())  # reply_ids 空
    assert [t.role for t in tree.topics[0].turns] == ["質疑者", "答弁者"]  # 変化なし


def test_response_comment_all_reply_not_moved():
    # 質疑者ターンが全文 reply (= トピック丸ごと応答) なら移動しない。
    from src.video.qaseg import move_response_comments_to_prev
    from src.models import QASentence, QATopic, QATree, QATurn

    def turn(role, *texts):
        return QATurn(speaker="X", role=role,
                      sentences=[QASentence(text=t, start=0.0, end=1.0) for t in texts])

    t1q = turn("質疑者", "ありがとうございました。", "以上です。")
    tree = QATree(topics=[
        QATopic(index=0, label="", question_speaker="A", answer_speakers=["B"],
                turns=[turn("質疑者", "質問1"), turn("答弁者", "答弁1")]),
        QATopic(index=1, label="", question_speaker="A", answer_speakers=["B"], turns=[t1q]),
    ])
    move_response_comments_to_prev(tree, {id(s) for s in t1q.sentences})
    assert [t.role for t in tree.topics[0].turns] == ["質疑者", "答弁者"]  # 変化なし
    assert len(tree.topics[1].turns[0].sentences) == 2  # topic1 はそのまま
