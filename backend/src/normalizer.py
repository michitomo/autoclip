"""Step 5.5: utterances.json の speaker / role を metadata.speakers に正規化する。

LLM 出力の表記揺れ（「高市」「高市総理大臣」「高市内閣総理大臣」など）を
metadata.speakers の name に揃え、role を SpeakerRole の値域に強制する。
"""

from __future__ import annotations

import logging

from src.models import (
    SPEAKER_ROLES,
    SpeakerInfo,
    SpeakerRole,
    Utterance,
    UtterancesOutput,
)
from src.scrapers._role import derive_role
from src.speaker_lookup import build_lookup, find_by_name

logger = logging.getLogger(__name__)


def normalize_utterances(
    utterances: UtterancesOutput, speakers: list[SpeakerInfo]
) -> UtterancesOutput:
    """utterances 全体に正規化を適用する（破壊的編集ではなく値を書き換えた同一参照を返す）。"""
    lookup = build_lookup(speakers)
    unmatched_count = 0
    for seg in utterances.segments:
        for u in seg.utterances:
            matched = find_by_name(
                u.speaker,
                lookup,
                allow_single_char=False,
                hint_affiliation=seg.segment_affiliation,
            )
            _apply_normalization(u, matched)
            if u.unmatched:
                unmatched_count += 1
    if unmatched_count:
        logger.info(
            "Step 5.5 normalization: %d/%d utterances unmatched against metadata.speakers",
            unmatched_count,
            sum(len(s.utterances) for s in utterances.segments),
        )
    return utterances


def coerce_role(raw: str, matched: SpeakerInfo | None) -> SpeakerRole:
    """raw role / matched speaker から SpeakerRole の 1 値を返す。

    優先順位: matched.role（scraper 派生済み）→ raw が SpeakerRole 値域内
    → raw 末尾パターン（"〇〇大臣" 等）→ "その他"
    """
    if matched and matched.role and matched.role in SPEAKER_ROLES:
        return matched.role  # type: ignore[return-value]

    if raw in SPEAKER_ROLES:
        return raw  # type: ignore[return-value]

    derived = derive_role(raw)
    return derived


def _apply_normalization(u: Utterance, matched: SpeakerInfo | None) -> None:
    """1 utterance を破壊的に書き換える。"""
    if matched is not None:
        u.speaker = matched.name
        u.unmatched = False
    else:
        u.unmatched = True
    u.role = coerce_role(u.role, matched)


__all__ = ["normalize_utterances", "coerce_role"]
