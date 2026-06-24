"""LLM 話者タグ付け (DeepInfra DeepSeek V3.2)

セグメント内の話者交代を検出し、utterances 配列を生成する。
LLMには話者交代ポイント（文番号）と話者名・役割のみ返させ、
テキスト本体はコード側で元の文を結合する（出力トークン大幅削減）。
"""

from __future__ import annotations

import json
import logging
import re

from src.api_client import LLM_MODEL, with_retry
from src.api_client import get_client as _get_client
from src.models import (
    RawTranscript,
    SegmentTranscript,
    SegmentUtterances,
    SessionDetail,
    SpeakerInfo,
    Utterance,
    UtterancesOutput,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """国会議事録の話者タグ付け専門家。番号付き文リストから話者交代ポイントを検出し、開始文番号・話者名・役割のみ返す（テキスト本体は不要）。

## speaker フィールドのルール
speaker には必ず人物の実名（苗字＋名前）を使う。
- 正しい例: "高市早苗"、"木原稔"、"西村智奈美"、"坂本哲志"、"森英介"
- 誤った例: "内閣総理大臣"、"官房長官"、"委員長"、"議長"（役職名・肩書のみは不可）
テキスト中に「内閣総理大臣 高市早苗君」のように役職＋氏名が現れる場合は、氏名部分（高市早苗）のみを使う。
委員長・議長のsplitでも、"委員長"や"議長"ではなく、発言者一覧から議長・委員長の実名（例: 坂本哲志、森英介）を使う。
発言者一覧から最も近い名前を選ぶ。一覧にない場合は推定した実名を使う。

## 話者交代ルール（優先度順）

**R1【最高優先度】指名文** 「〇〇君。」「〇〇委員。」「〇〇さん。」「〇〇大臣。」「〇〇副大臣。」「〇〇政務官。」だけの短文、役職付き指名文（例:「財務大臣片山さつき君。」）、「続いて〇〇君。」「次に〇〇君。」等 → 指名文は委員長の発言に含める（委員長のsplitに統合）。指名文の**次の文**から指名された人物の新splitを開始する。
  - 例: (0)「坂本哲志（予算委員長）」(1)「次に西村智奈美君。」(2)「中道改革連合の西村智奈美です。」→ start=0: 坂本哲志(委員長), start=2: 西村智奈美(質疑者)

**R2: Whisperラベル行** 「〇〇委員長：text」等コロン付き行、または「〇〇（役職名）」のみの行 → その番号からその話者の発言開始。line 0が非ラベルでline 1がラベルの場合、line 0はラベル話者に帰属（委員長・議長の議事進行文に多い）。

**R3: 答弁冒頭の定型句** 「お答えいたします」「お答え申し上げます」「御指摘の点について」「ただいまの御質問に」「御説明申し上げます」等 → 答弁者または政府参考人。

**R4: 継続** R1〜3非該当文は直前の話者の継続。発言内に他者の名前が登場しても話者交代ではない。

## 役割（role）

| role | 対象 |
|------|------|
| 委員長 | 委員長・副委員長・議長の議事進行・指名発言 |
| 質疑者 | 委員・議員の質問・意見（大臣等には使わない） |
| 答弁者 | 大臣・副大臣・政務官・官房長官の答弁 |
| 政府参考人 | 省庁の局長・部長・審議官等の答弁 |
| 参考人 | 独立機関の官僚・外部有識者・民間人 |
| その他 | 上記以外 |

## 過剰分割の防止
- R1〜3に明確に該当する場合のみsplitを作成。同一話者は1 splitに統合（複数テーマでも分割しない）
- 通常2〜6 splits/セグメント。10超は統合すること
- **連続する同一話者のsplitは必ず1つに統合する（最初のstart番号のみ残す）**
  - 誤: [{"start":0,"speaker":"坂本哲志","role":"委員長"},{"start":1,"speaker":"坂本哲志","role":"委員長"}]
  - 正: [{"start":0,"speaker":"坂本哲志","role":"委員長"}]
- ラベル行と指名文が連続して同一話者（委員長）を示す場合も1 splitにまとめる

## セグメント冒頭の境界ノイズ除去（PR25）
音声録音の切り替えタイミングの都合で、このセグメントの先頭 1〜2文が
**前のセグメントの末尾**に属する内容（委員長の指名文・前の答弁者の結語等）を
含む場合がある。

具体的には:
- 「○○大臣。」「次に○○君。」のような委員長の指名文のみの行
- 「以上です。」「ありがとうございました。」等の前の答弁者の締め発言

これらがセグメントの先頭 (文番号 0〜1) に現れたときは、**start=0 の話者として
セグメント主発言者 (segment_speaker) に帰属させ、その次の文から実際の内容を
開始する**。指名文を別 speaker の split に切り出さない。

## 出力形式
{"splits": [{"start": 0, "speaker": "発言者名", "role": "役割"}, ...]}
start: 0始まりの文番号、昇順、最初は必ず0。
"""


def _split_sentences(text: str) -> list[str]:
    """テキストを文単位に分割する。

    句点「。」、疑問符「？」、改行で分割。
    空文字列は除外する。
    """
    # 句点・疑問符の後、または改行で分割
    parts = re.split(r'(?<=[。？])|(?<=\n)', text)
    return [s.strip() for s in parts if s.strip()]


def _number_sentences(sentences: list[str]) -> str:
    """文リストを番号付きテキストに変換する。"""
    return "\n".join(f"({i}){s}" for i, s in enumerate(sentences))


def tag_speakers(
    raw_text: str,
    segment_speaker: SpeakerInfo,
    all_speakers: list[SpeakerInfo],
) -> list[Utterance]:
    """1セグメントの文字起こしテキストに話者タグを付ける。

    入力テキストを文単位で番号付けし、LLMには話者交代ポイントのみ返させる。
    テキスト本体はコード側で結合する。

    Args:
        raw_text: Whisper 文字起こしテキスト
        segment_speaker: このセグメントの主発言者
        all_speakers: セッション全発言者リスト

    Returns:
        Utterance のリスト

    Raises:
        openai.APIError: API 呼び出しが失敗した場合
        ValueError: LLM の出力が正しい JSON でない場合
    """
    client = _get_client()

    sentences = _split_sentences(raw_text)
    if not sentences:
        return [Utterance(speaker=segment_speaker.name, role="質疑者", text=raw_text)]

    numbered_text = _number_sentences(sentences)

    speaker_list = "\n".join(
        f"- {s.name}（{s.affiliation}）" for s in all_speakers
    )

    user_prompt = f"""セグメントの主発言者: {segment_speaker.name}（{segment_speaker.affiliation}）
役割: {segment_speaker.role or "質疑者"}

このセッションの発言者一覧:
{speaker_list}

以下の番号付き文リストの話者交代ポイントを検出してください（{len(sentences)}文）:

{numbered_text}"""

    logger.info("Tagging speakers for segment: %s", segment_speaker.name)

    response = with_retry(lambda: client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    ))

    content = response.choices[0].message.content
    if not content:
        logger.warning(
            "Empty response from LLM for segment speaker=%s; using whole text as single utterance",
            segment_speaker.name,
        )
        return [Utterance(
            speaker=segment_speaker.name,
            role=segment_speaker.role or "質疑者",
            text=raw_text,
        )]

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse LLM JSON for segment speaker=%s: %s; using whole text as single utterance",
            segment_speaker.name, e,
        )
        return [Utterance(
            speaker=segment_speaker.name,
            role=segment_speaker.role or "質疑者",
            text=raw_text,
        )]
    splits = data.get("splits", [])

    if not splits:
        # フォールバック: 分割なし → 全文を主発言者の発言とする
        logger.warning("No splits returned, using entire text as single utterance")
        return [Utterance(
            speaker=segment_speaker.name,
            role=segment_speaker.role or "質疑者",
            text=raw_text,
        )]

    # start で昇順ソート
    splits.sort(key=lambda s: s.get("start", 0))

    # 最初の split が 0 でなければ補正
    if splits[0].get("start", 0) != 0:
        splits.insert(0, {
            "start": 0,
            "speaker": segment_speaker.name,
            "role": segment_speaker.role or "質疑者",
        })

    # splits → Utterance リストに変換
    result = []
    for i, split in enumerate(splits):
        start_idx = split.get("start", 0)
        end_idx = splits[i + 1].get("start", len(sentences)) if i + 1 < len(splits) else len(sentences)

        # 範囲チェック
        start_idx = max(0, min(start_idx, len(sentences)))
        end_idx = max(start_idx, min(end_idx, len(sentences)))

        if start_idx >= end_idx:
            continue

        text = "".join(sentences[start_idx:end_idx])
        speaker = split.get("speaker", "")
        role = split.get("role", "その他")

        if not speaker:
            logger.warning("Split missing speaker, using segment speaker: %s", segment_speaker.name)
            speaker = segment_speaker.name

        result.append(Utterance(speaker=speaker, role=role, text=text))

    return result


def _build_video_url(chamber: str, session_id: str, start_seconds: float) -> str:
    """院に応じた動画リンクURLを生成する。"""
    if chamber == "shugiin":
        return (
            f"https://www.shugiintv.go.jp/jp/index.php"
            f"?ex=VL&media_type=&deli_id={session_id}&time={start_seconds}"
        )
    elif chamber == "sangiin":
        return (
            f"https://www.webtv.sangiin.go.jp/webtv/detail.php"
            f"?sid={session_id}#{start_seconds}"
        )
    return ""


def tag_all_segments(
    raw_transcript: RawTranscript,
    session_detail: SessionDetail,
    max_workers: int = 16,
) -> UtterancesOutput:
    """全セグメントを並列でLLM話者タグ付けして UtterancesOutput を返す。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    speakers = session_detail.speakers

    def _resolve_speaker(seg: SegmentTranscript) -> SpeakerInfo:
        if seg.segment_index < len(speakers):
            return speakers[seg.segment_index]
        matched = next((s for s in speakers if s.name == seg.speaker_name), None)
        return matched or SpeakerInfo(
            name=seg.speaker_name,
            affiliation="",
            start_seconds=seg.start_seconds,
            start_time="",
            duration_minutes=0,
        )

    def _tag(seg: SegmentTranscript) -> SegmentUtterances:
        segment_speaker = _resolve_speaker(seg)
        utterances = tag_speakers(seg.text, segment_speaker, speakers)
        video_url = _build_video_url(
            session_detail.chamber, session_detail.session_id, seg.start_seconds
        )
        return SegmentUtterances(
            segment_index=seg.segment_index,
            segment_speaker=segment_speaker.name,
            segment_affiliation=segment_speaker.affiliation,
            start_seconds=seg.start_seconds,
            video_url=video_url,
            utterances=utterances,
        )

    results: list[SegmentUtterances] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_tag, seg): seg.segment_index for seg in raw_transcript.segments}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda s: s.segment_index)
    return UtterancesOutput(segments=results)
