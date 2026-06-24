"""Q&A 区間分割 (autoclip 固有)。

一度の議員登壇には「質問 → 答弁 → 再質問 → …」と複数の質疑応答が含まれる。
これを Q&A 単位に分割し、各 Q&A に動画区間 (start/end) を付ける。

手順:
1. speaker_tagger.tag_speakers で校正済みテキストを話者ターン (Utterance) に分割
   (質疑者 / 答弁者 / 委員長 の交代を検出)。
2. 各 Utterance のテキストを **単語時刻列** に順次対応づけて start/end を付与
   (utterances はテキストを順番に分割したものなので、語を前から消費すれば対応する)。
3. 質問ターン + 直後の答弁ターン群を 1 つの QASegment にまとめる。

字幕の話者色分け・Q&A 単位の切り抜きに使う。
"""

from __future__ import annotations

import logging
import re

from src.models import (
    QASegment,
    QASentence,
    QATopic,
    QATree,
    QATurn,
    SpeakerInfo,
    TimedUtterance,
    Utterance,
    WhisperWord,
)
from src.speaker_tagger import tag_speakers

logger = logging.getLogger(__name__)

# 答弁者の役職語 (委員長の指名文に現れる)。長いものを先にして貪欲一致させる。
_ANSWER_TITLES = (
    "内閣総理大臣", "厚生労働副大臣", "文部科学副大臣", "内閣府副大臣",
    "厚生労働大臣", "文部科学大臣", "内閣府特命担当大臣", "総務大臣",
    "財務大臣", "法務大臣", "経済産業大臣", "環境大臣",
    "副大臣", "大臣政務官", "大臣", "政府参考人", "参考人",
    "総括審議官", "審議官", "局長", "部長", "次長", "長官",
)
# 役職の直前に来る氏名 (2〜5 文字の漢字)。指名文 "二木厚生労働副大臣。" を想定。
_ANSWER_NAME_RE = re.compile(
    r"([一-龥]{2,5})(" + "|".join(_ANSWER_TITLES) + r")"
)


def extract_answerers(raw_text: str) -> list[SpeakerInfo]:
    """raw 文字起こしの指名文から答弁者 (氏名＋役職) を抽出する。

    衆議院TV の発言者リストには質疑者と委員長しか無く、答弁者 (大臣/副大臣/参考人)
    の氏名が欠ける。委員長は答弁前に「○○厚生労働副大臣。」等と指名するので、その
    パターンから氏名を拾い、校正・話者タグの参照に足す。

    校正前 (raw) のテキストに対して使うこと。校正が氏名を落とす前に拾う。
    """
    found: dict[str, str] = {}  # name -> title(role 推定用)
    for m in _ANSWER_NAME_RE.finditer(raw_text):
        name, title = m.group(1), m.group(2)
        if len(name) < 2:
            continue
        # 役職語の断片や明らかな誤認識を弾く: 氏名に役職語が含まれる/役職的な語で
        # 終わる場合は呼称の途切れの可能性が高い (例: 「首務」「総括」)。
        if any(t in name for t in _ANSWER_TITLES):
            continue
        if name[-1] in "務総括官長相党会":
            continue
        found.setdefault(name, title)

    speakers: list[SpeakerInfo] = []
    for name, title in found.items():
        role = "政府参考人" if ("参考人" in title or "審議官" in title
                                or "局長" in title) else "答弁者"
        speakers.append(
            SpeakerInfo(
                name=name, affiliation=title, role=role,
                start_seconds=0.0, start_time="", duration_minutes=0,
            )
        )
    if speakers:
        logger.info(
            "Extracted %d answerer(s) from 指名文: %s",
            len(speakers), ", ".join(f"{s.name}({s.affiliation})" for s in speakers),
        )
    return speakers

# 質問側・答弁側の role 分類
_QUESTION_ROLES = {"質疑者"}
_ANSWER_ROLES = {"答弁者", "政府参考人", "参考人"}
_CHAIR_ROLES = {"委員長", "議長"}


def _norm(s: str) -> str:
    return "".join(s.split())


def _consume_span(
    words: list[WhisperWord],
    wi: int,
    n: int,
    wlens: list[int],
    target_chars: int,
) -> tuple[float, float, int]:
    """語カーソル wi から可視文字数 target_chars ぶん消費し (start, end, 次wi) を返す。

    語を前から消費してテキスト断片に時刻を割り当てる共通処理。
    attach_times_to_utterances と build_qa_tree が同じカーソルロジックを共有する
    ことで、ターン span と文 span が同一の語クロックから生成され、ズレない。
    """
    start_i = wi
    consumed = 0
    while wi < n and consumed < target_chars:
        consumed += wlens[wi]
        wi += 1
    end_i = wi - 1 if wi > start_i else start_i
    start_i = min(start_i, n - 1)
    return words[start_i].start, words[min(end_i, n - 1)].end, wi


def attach_times_to_utterances(
    utterances: list[Utterance], words: list[WhisperWord]
) -> list[TimedUtterance]:
    """各 Utterance を単語時刻に対応づけて TimedUtterance 列を返す。

    utterances はテキストを順番に分割したものなので、語列を前から消費して各
    utterance の可視文字数ぶんを割り当てる。ズレ吸収のため、割当は文字数ベース。
    """
    timed: list[TimedUtterance] = []
    wi = 0
    n = len(words)
    # 各語の可視文字数を前計算
    wlens = [len(_norm(w.word)) for w in words]

    for u in utterances:
        target = len(_norm(u.text))
        if target == 0 or wi >= n:
            continue
        start, end, wi = _consume_span(words, wi, n, wlens, target)
        timed.append(
            TimedUtterance(
                speaker=u.speaker,
                role=u.role,
                text=u.text,
                start=start,
                end=end,
            )
        )
    return timed


# 文末記号 (文単位分割用)。読点 (、，) では割らない。
_SENT_END_RE = re.compile(r"(?<=[。．？！?!])")
# これ未満の可視文字数の文断片は前の文に併合 (「。」だけの行を防ぐ)
_MIN_SENTENCE_CHARS = 4


def _split_sentences_keep(text: str) -> list[str]:
    """テキストを文単位に分割する (文末記号は各文の末尾に残す)。

    - 句点 (。．？！?!) の直後で割る。読点では割らない。
    - 文末記号が無ければ 1 文として全体を返す。
    - 極端に短い断片 (< _MIN_SENTENCE_CHARS 可視文字) は前の文に併合する
      (「はい。」「。」だけのリーフを避ける)。
    """
    raw = [s.strip() for s in _SENT_END_RE.split(text) if s and s.strip()]
    if not raw:
        return [text.strip()] if text.strip() else []
    merged: list[str] = []
    for s in raw:
        if merged and len(_norm(s)) < _MIN_SENTENCE_CHARS:
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)
    return merged


def build_qa_tree(
    qas: list[QASegment], words: list[WhisperWord]
) -> QATree:
    """QASegment 列 + 語時刻から編集ツリー (トピック>発言者>発言内容) を構築する。

    語を **1 本のカーソル**で前から走査し、QASegment 時系列順・ターン順に各ターンの
    テキストを文単位へ分割して文ごとに span を割り当てる。これで文 span は必ずその
    ターン span の内側にネストし、次のターンへ食い込まない (attach_times_to_utterances
    と同じ消費順)。importance/summary/enabled は後段の annotate_sentences で確定する
    (ここでは既定 mid / 空 / enabled=True)。
    """
    n = len(words)
    wlens = [len(_norm(w.word)) for w in words]
    wi = 0
    topics: list[QATopic] = []

    for qa in qas:
        turns: list[QATurn] = []
        for u in qa.utterances:
            if wi >= n:
                break
            sentences: list[QASentence] = []
            for sent in _split_sentences_keep(u.text):
                target = len(_norm(sent))
                if target == 0 or wi >= n:
                    continue
                start, end, wi = _consume_span(words, wi, n, wlens, target)
                sentences.append(
                    QASentence(text=sent, start=start, end=end)
                )
            if sentences:
                turns.append(
                    QATurn(speaker=u.speaker, role=u.role, sentences=sentences)
                )
        topics.append(
            QATopic(
                index=qa.index,
                label=qa.topic or "",
                question_speaker=qa.question_speaker,
                answer_speakers=list(qa.answer_speakers),
                turns=turns,
            )
        )
    # 「応答コメント→質疑→答弁」を「質疑→答弁→コメント」に並べ替えるのは、応答コメント
    # 判定 (LLM) が必要なので annotate 後に move_response_comments_to_prev で行う。
    return QATree(topics=topics)


def move_response_comments_to_prev(
    tree: QATree, reply_ids: set[int]
) -> QATree:
    """各トピックの質疑者ターン先頭にある「前答弁への応答コメント」を前トピック末尾へ移す。

    backend の Q&A 分割は質疑者の発話開始でトピックを切るため、1 つの質疑者ターンに
    「前答弁への応答コメント + 本題の質問」が混在し、構成が『コメント→質疑→答弁』に
    なってしまう。LLM が reply=true と判定した文 (id() が reply_ids にある) が
    質疑者ターン先頭に連続している分を、直前トピック (= その答弁の直後) の末尾へ移し、
    『質疑→答弁→コメント』にする。reply 判定は qa_annotate が行う (LLM)。
    (in-place で topics を書き換え、同じ tree を返す。topic0 は前が無いので対象外。)
    """
    topics = tree.topics
    for i in range(1, len(topics)):
        topic = topics[i]
        if not topic.turns or topic.turns[0].role not in _QUESTION_ROLES:
            continue
        qturn = topic.turns[0]
        sents = qturn.sentences
        if not sents:
            continue
        # 先頭から連続する reply 文の数を数える (本題に入った時点で打ち切り)。
        split = 0
        while split < len(sents) and id(sents[split]) in reply_ids:
            split += 1
        # 全文が reply、または1文も reply でないなら移動しない
        # (全文 reply = トピック丸ごと応答 → そのまま残す方が安全)。
        if split == 0 or split >= len(sents):
            continue
        comment = sents[:split]
        qturn.sentences = sents[split:]
        topics[i - 1].turns.append(
            QATurn(speaker=qturn.speaker, role=qturn.role, sentences=comment)
        )
    return tree


def group_into_qa(timed: list[TimedUtterance]) -> list[QASegment]:
    """時刻付き発話を Q&A 単位にまとめる。

    質問ターン (質疑者) を起点に、直後の答弁ターン群 (答弁者/参考人) までを 1 Q&A に。
    委員長の指名などは前後の Q&A に吸収 (区間には含めるが話者としては chair)。
    最初の質問より前のターン (冒頭の委員長指名等) は最初の Q&A 開始に含める。
    """
    qas: list[QASegment] = []
    cur: list[TimedUtterance] = []
    seen_question = False
    seen_answer_after_q = False

    def flush() -> None:
        nonlocal cur, seen_question, seen_answer_after_q
        if not cur:
            return
        q_spk = next(
            (u.speaker for u in cur if u.role in _QUESTION_ROLES), cur[0].speaker
        )
        a_spk = [u.speaker for u in cur if u.role in _ANSWER_ROLES]
        # 重複除去 (順序維持)
        a_spk = list(dict.fromkeys(a_spk))
        qas.append(
            QASegment(
                index=len(qas),
                question_speaker=q_spk,
                answer_speakers=a_spk,
                start=cur[0].start,
                end=cur[-1].end,
                utterances=list(cur),
            )
        )
        cur = []
        seen_question = False
        seen_answer_after_q = False

    for u in timed:
        is_q = u.role in _QUESTION_ROLES
        is_a = u.role in _ANSWER_ROLES

        # 新しい質問が始まった & 既に「質問→答弁」を1往復見た → 区切る
        if is_q and seen_question and seen_answer_after_q:
            flush()

        cur.append(u)
        if is_q:
            seen_question = True
        if is_a and seen_question:
            seen_answer_after_q = True

    flush()
    return qas


def segment_qa(
    corrected_text: str,
    words: list[WhisperWord],
    member: SpeakerInfo,
    all_speakers: list[SpeakerInfo],
    extra_answerers: list[SpeakerInfo] | None = None,
) -> list[QASegment]:
    """校正済みテキスト + 単語時刻から Q&A 区間を構築する。

    Args:
        corrected_text: 校正後の議員区間テキスト
        words: 対応する WhisperWord 列 (member-WAV 時間, start 昇順)
        member: 登壇議員 (segment_speaker)
        all_speakers: セッションの全発言者 (話者解決用)
        extra_answerers: 指名文から抽出した答弁者 (発言者リストに無い大臣等)。
            話者タグの参照に足すと「政府参考人」汎用名でなく実名になりやすい。

    Returns:
        QASegment のリスト (時系列)。検出できなければ全体を 1 つにまとめて返す。
    """
    if not words:
        return []
    tag_speakers_list = list(all_speakers) + list(extra_answerers or [])
    try:
        utterances = tag_speakers(corrected_text, member, tag_speakers_list)
    except Exception as e:  # noqa: BLE001 - 失敗時は全体を 1 Q&A に
        logger.warning("speaker tagging failed (%s); whole segment as 1 Q&A", e)
        utterances = []

    if not utterances:
        return [
            QASegment(
                index=0,
                question_speaker=member.name,
                answer_speakers=[],
                start=words[0].start,
                end=words[-1].end,
                utterances=[
                    TimedUtterance(
                        speaker=member.name, role="質疑者",
                        text=corrected_text, start=words[0].start, end=words[-1].end,
                    )
                ],
            )
        ]

    timed = attach_times_to_utterances(utterances, words)
    qas = group_into_qa(timed)
    logger.info(
        "Q&A segmentation: %d utterances → %d Q&A segments", len(timed), len(qas)
    )
    return qas
