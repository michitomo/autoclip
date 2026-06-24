"""qa_annotate の単体テスト (LLM はモック; ネットワークなし)。"""

from __future__ import annotations

import src.video.qa_annotate as qa_annotate
from src.models import QASentence, QATopic, QATree, QATurn
from src.video.qa_annotate import _parse_items, annotate_sentences


def _tree(texts: list[str]) -> QATree:
    sents = [QASentence(text=t, start=float(i), end=float(i) + 1.0)
             for i, t in enumerate(texts)]
    return QATree(topics=[QATopic(index=0, question_speaker="A",
                                  turns=[QATurn(speaker="A", role="質疑者",
                                                sentences=sents)])])


def _flat(tree: QATree) -> list[QASentence]:
    return tree.topics[0].turns[0].sentences


# ---------------------------------------------------------------------------
# _parse_items: JSON 堅牢性
# ---------------------------------------------------------------------------


def test_parse_items_object_with_items():
    out = _parse_items('{"items":[{"idx":0,"summary":"要点","importance":"high"}]}')
    assert out == [{"idx": 0, "summary": "要点", "importance": "high"}]


def test_parse_items_bare_array():
    out = _parse_items('[{"idx":1,"importance":"low"}]')
    assert out == [{"idx": 1, "importance": "low"}]


def test_parse_items_malformed_returns_empty():
    assert _parse_items("not json at all") == []


def test_parse_items_repairs_control_chars():
    # 文字列値内に生の改行 (制御文字) → 修復して解釈
    out = _parse_items('{"items":[{"idx":0,"summary":"a\nb","importance":"mid"}]}')
    assert out and out[0]["idx"] == 0


# ---------------------------------------------------------------------------
# annotate_sentences: 整合・フォールバック・enabled 確定
# ---------------------------------------------------------------------------


def test_annotate_applies_importance_and_enabled(monkeypatch):
    monkeypatch.setattr(qa_annotate, "_call_llm", lambda *a, **k: {
        0: {"idx": 0, "summary": "核心の追及", "importance": "high"},
        1: {"idx": 1, "summary": "繋ぎ", "importance": "low"},
        2: {"idx": 2, "summary": "背景説明", "importance": "mid"},
    })
    tree = _tree(["核心です。", "えーと。", "背景は。"])
    annotate_sentences(tree, member="A", committee="厚生労働委員会")
    s = _flat(tree)
    # 重要度は付くが、全文 初期 ON (低も enabled=True; 外すかは人間が決める)
    assert s[0].importance == "high" and s[0].enabled is True
    assert s[0].summary == "核心の追及"
    assert s[1].importance == "low" and s[1].enabled is True
    assert s[2].importance == "mid" and s[2].enabled is True


def test_annotate_missing_idx_defaults_mid_enabled(monkeypatch):
    # idx 1 が欠落 → mid (残す側) になる
    monkeypatch.setattr(qa_annotate, "_call_llm", lambda *a, **k: {
        0: {"idx": 0, "importance": "high"},
    })
    tree = _tree(["A。", "B。"])
    annotate_sentences(tree)
    s = _flat(tree)
    assert s[1].importance == "mid" and s[1].enabled is True and s[1].summary == ""


def test_annotate_bad_importance_coerced_to_mid(monkeypatch):
    monkeypatch.setattr(qa_annotate, "_call_llm", lambda *a, **k: {
        0: {"idx": 0, "importance": "CRITICAL"},  # 不正値
    })
    tree = _tree(["A。"])
    annotate_sentences(tree)
    assert _flat(tree)[0].importance == "mid"


def test_annotate_total_failure_all_mid(monkeypatch):
    # LLM 完全失敗 (空 dict) → 全 mid/enabled、要約空、クラッシュしない
    monkeypatch.setattr(qa_annotate, "_call_llm", lambda *a, **k: {})
    tree = _tree(["A。", "B。", "C。"])
    annotate_sentences(tree)
    for s in _flat(tree):
        assert s.importance == "mid" and s.enabled is True and s.summary == ""


def test_annotate_empty_tree_noop(monkeypatch):
    monkeypatch.setattr(qa_annotate, "_call_llm",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    out = annotate_sentences(QATree(topics=[]))
    assert out.topics == []


def test_annotate_fills_topic_label_from_high_question(monkeypatch):
    monkeypatch.setattr(qa_annotate, "_call_llm", lambda *a, **k: {
        0: {"idx": 0, "summary": "挨拶", "importance": "low"},
        1: {"idx": 1, "summary": "香料規制の追及", "importance": "high"},
    })
    tree = _tree(["どうも。", "香料規制について伺います。"])
    annotate_sentences(tree)
    # 質問ターンの high 文の要約がトピック見出しになる
    assert tree.topics[0].label == "香料規制の追及"


def test_annotate_chunks_preserve_global_idx(monkeypatch):
    # 文数 > _MAX_SENTENCES_PER_CALL でチャンク分割しても idx 整合が崩れない
    monkeypatch.setattr(qa_annotate, "_MAX_SENTENCES_PER_CALL", 2)

    def fake_call(payload, member, committee, model):
        import json
        items = json.loads(payload)
        # payload 内の各 idx を low にして返す (idx 整合の検証用)
        return {it["idx"]: {"idx": it["idx"], "importance": "low"} for it in items}

    monkeypatch.setattr(qa_annotate, "_call_llm", fake_call)
    tree = _tree([f"文{i}。" for i in range(5)])  # 5 文 → 3 チャンク
    annotate_sentences(tree)
    s = _flat(tree)
    assert len(s) == 5
    # idx 整合: 全文に importance が付く。enabled は全文 ON。
    assert all(x.importance == "low" and x.enabled is True for x in s)
