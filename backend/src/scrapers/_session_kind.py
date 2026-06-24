"""SessionKind をページテキストから決定論的に判定するモジュール。"""

from __future__ import annotations

from src.models import SessionKind, SpeakerInfo

_FLOOR_SPEECH_KEYWORDS: tuple[str, ...] = (
    "趣旨説明",
    "討論",
    "採決",
    "解任決議",
    "所信表明",
    "弔詞",
)

_PROCEDURAL_KEYWORDS: tuple[str, ...] = (
    "理事会",
    "議事手続",
)


def detect_session_kind(
    page_text: str, committee: str, speakers: list[SpeakerInfo]
) -> SessionKind:
    """ページテキスト・委員会名・発言者一覧から SessionKind を 1 つ決める。

    判定優先順位:
        本会議: 代表質問 > floor_speech キーワード > floor_speech (default)
        委員会: 公聴会・全員参考人なら expert_hearing
                理事会等の事務系なら procedural
                それ以外は regular_qa
    """
    if committee == "本会議":
        if "代表質問" in page_text and any(s.role == "質疑者" for s in speakers):
            return "representative_questions"
        if any(kw in page_text for kw in _FLOOR_SPEECH_KEYWORDS):
            return "floor_speech"
        return "floor_speech"

    if "公聴会" in committee:
        return "expert_hearing"

    if speakers and all(s.role == "参考人" for s in speakers):
        return "expert_hearing"

    if any(kw in page_text for kw in _PROCEDURAL_KEYWORDS):
        return "procedural"

    return "regular_qa"
