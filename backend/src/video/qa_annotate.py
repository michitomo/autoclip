"""Q&A 編集ツリーの文に LLM で「1行要約＋重要度」を付ける (autoclip 固有)。

編集UIで人間が 100+ 文を読まずに取捨選択できるよう、各文 (QASentence) に
体言止めの 1 行要約と重要度 (high/mid/low) を与える。low (挨拶/定型/脱線) は
初期 OFF にする (= enabled の唯一の確定箇所)。

設計:
- 入力: ツリー全文をグローバル連番 (idx) 付きでフラット化し、{idx, speaker, role,
  text} のリストを 1 回 (大きければトピック単位で並列) の LLM 呼び出しに渡す。
- 出力: {"items":[{"idx","summary","importance"}]} の JSON。idx で元の文に整合。
- 堅牢性: JSON 破損は structurer の _repair_json_string_controls で修復、なお失敗
  なら「全 mid・要約空」にフォールバック (クラッシュさせない)。欠落 idx は mid
  (= 残す側に倒す)、不正 importance も mid、重複 idx は first-wins。
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from src.api_client import LLM_MODEL, get_client, with_retry
from src.models import QASentence, QATree
from src.structurer import _repair_json_string_controls

logger = logging.getLogger(__name__)

# 1 回の呼び出しに含める最大文数。チャンクは並列実行されるので、小さくするほど
# 1 call の出力 (JSON) が短くなり、全体の wall-clock が縮む (大チャンク1個の生成待ちが
# ボトルネックだった)。20 前後 × 多並列が速い。
_MAX_SENTENCES_PER_CALL = 20
_MAX_WORKERS = 12

_VALID_IMPORTANCE = {"high", "mid", "low"}

_SYSTEM_PROMPT = """あなたは国会質疑の編集アシスタントです。各発言(文)に
(1)10〜20文字の体言止め要約 と (2)重要度 high/mid/low を判定してください。

## 重要度の基準
- high: 質問の核心 / 数値・固有名詞・法令名 / 明確な要求・約束 / 争点となる主張。
- mid:  文脈に必要な説明・背景・前置き・具体例。
- low:  挨拶・お礼・自己紹介・定型の議事進行(「はい」「以上です」等)・繋ぎ・脱線。

## 出力 (JSON のみ)
{"items":[{"idx":<入力のidx>,"summary":"<体言止め要約>","importance":"high|mid|low"}, ...]}
- 入力の全 idx を 1 回ずつ含めること。説明や前置きを付けないこと。"""


def _flatten(tree: QATree) -> list[QASentence]:
    """ツリー内の全文リーフを (走査順=グローバル idx 順) で返す。"""
    out: list[QASentence] = []
    for topic in tree.topics:
        for turn in topic.turns:
            out.extend(turn.sentences)
    return out


def _speaker_role_of(tree: QATree) -> list[tuple[str, str]]:
    """_flatten と同順で各文の (speaker, role) を返す。"""
    out: list[tuple[str, str]] = []
    for topic in tree.topics:
        for turn in topic.turns:
            for _ in turn.sentences:
                out.append((turn.speaker, turn.role))
    return out


def _build_payload(
    sentences: list[QASentence], meta: list[tuple[str, str]], idx0: int
) -> str:
    """LLM へ渡す {idx,speaker,role,text} の JSON 配列文字列を作る。"""
    items = [
        {"idx": idx0 + i, "speaker": meta[i][0], "role": meta[i][1],
         "text": s.text}
        for i, s in enumerate(sentences)
    ]
    return json.dumps(items, ensure_ascii=False)


def _call_llm(payload: str, member: str, committee: str, model: str) -> dict:
    """1 チャンク分を LLM に投げ {idx: {summary, importance}} を返す。失敗時 {}。"""
    client = get_client()
    ctx = f"委員会: {committee}\n登壇議員: {member}\n\n発言リスト(JSON):\n{payload}"

    def _do() -> object:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": ctx},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

    try:
        resp = with_retry(_do)
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 - 失敗時は空 (呼び出し側で mid フォールバック)
        logger.warning("QA annotate LLM call failed: %s", e)
        return {}

    data = _parse_items(content)
    by_idx: dict[int, dict] = {}
    for it in data:
        if not isinstance(it, dict):
            continue
        idx = it.get("idx")
        if not isinstance(idx, int) or idx in by_idx:
            continue  # 重複 idx は first-wins
        by_idx[idx] = it
    return by_idx


def _parse_items(content: str) -> list:
    """JSON 文字列から items 配列を取り出す。破損は修復、なお失敗なら []。"""
    for text in (content, _repair_json_string_controls(content)):
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            items = obj.get("items")
            if isinstance(items, list):
                return items
        if isinstance(obj, list):  # モデルが配列を直接返した場合
            return obj
    logger.warning("QA annotate: could not parse JSON (using mid fallback)")
    return []


def annotate_sentences(
    tree: QATree, *, member: str = "", committee: str = "", model: str = LLM_MODEL
) -> QATree:
    """ツリーの各文に要約＋重要度を付け、enabled を確定して同じツリーを返す。

    enabled = (importance != "low")。LLM 失敗・JSON 破損時も全文 mid/enabled の
    まま (要約空) で返し、編集UIは階層表示できる。in-place で更新する。
    """
    sentences = _flatten(tree)
    if not sentences:
        return tree
    meta = _speaker_role_of(tree)

    # チャンク分割 (グローバル idx を維持) → 並列呼び出し → idx でマージ
    chunks: list[tuple[int, list[QASentence], list[tuple[str, str]]]] = []
    for start in range(0, len(sentences), _MAX_SENTENCES_PER_CALL):
        end = start + _MAX_SENTENCES_PER_CALL
        chunks.append((start, sentences[start:end], meta[start:end]))

    by_idx: dict[int, dict] = {}
    if len(chunks) == 1:
        start, sub, submeta = chunks[0]
        by_idx = _call_llm(_build_payload(sub, submeta, start), member, committee, model)
    else:
        def _run(chunk: tuple[int, list[QASentence], list[tuple[str, str]]]) -> dict:
            start, sub, submeta = chunk
            return _call_llm(
                _build_payload(sub, submeta, start), member, committee, model
            )

        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(chunks))) as ex:
            for part in ex.map(_run, chunks):
                for k, v in part.items():
                    by_idx.setdefault(k, v)  # first-wins (チャンクは idx 非重複)

    for i, s in enumerate(sentences):
        it = by_idx.get(i, {})
        summary = it.get("summary")
        s.summary = summary.strip() if isinstance(summary, str) else ""
        imp = it.get("importance")
        s.importance = imp if imp in _VALID_IMPORTANCE else "mid"
        # 全文を初期 ON にする (重要度は判断材料として残すが「勝手に消えない」)。
        # 低を外すかは編集UIで人間が決める。
        s.enabled = True

    # トピック見出しを補完: 質問ターン (質疑者) の最重要文の要約を使う。
    # segment_qa は QASegment.topic を空にするため、ここで埋めると UI が読みやすい。
    for topic in tree.topics:
        if topic.label:
            continue
        topic.label = _derive_topic_label(topic)

    n_low = sum(1 for s in sentences if s.importance == "low")
    logger.info(
        "QA annotate: %d sentences (%d low, all enabled) via %d call(s)",
        len(sentences), n_low, len(chunks),
    )
    return tree


def _derive_topic_label(topic) -> str:
    """トピック見出しを質問ターンの最重要文の要約から作る (無ければ空)。"""
    q_sents = [
        s for turn in topic.turns if turn.role == "質疑者"
        for s in turn.sentences
    ]
    if not q_sents:
        q_sents = [s for turn in topic.turns for s in turn.sentences]
    # high を優先、無ければ要約のある最初の文
    for want_high in (True, False):
        for s in q_sents:
            if s.summary and (s.importance == "high" or not want_high):
                return s.summary
    return ""
