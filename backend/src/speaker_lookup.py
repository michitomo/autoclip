"""metadata.speakers に対する名前照合ユーティリティ。

structurer (Q&A 抽出時の発言者解決) と normalizer (Step 5.5) の両方で使う。
"""

from __future__ import annotations

import logging

from src.models import SpeakerInfo

logger = logging.getLogger(__name__)

SINGLE_CHAR_SURNAMES: frozenset[str] = frozenset(
    ("林", "森", "原", "関", "堀", "岡", "辻", "塚", "柳", "萩", "菅", "泉", "馬")
)


def build_lookup(speakers: list[SpeakerInfo]) -> dict[str, SpeakerInfo]:
    """name → SpeakerInfo の辞書を作る。同名は最初の登場を残す。"""
    lookup: dict[str, SpeakerInfo] = {}
    for s in speakers:
        if s.name and s.name not in lookup:
            lookup[s.name] = s
    return lookup


def find_by_name(
    name: str,
    lookup: dict[str, SpeakerInfo],
    *,
    allow_single_char: bool = True,
    hint_affiliation: str = "",
) -> SpeakerInfo | None:
    """完全一致 → 2文字姓 → 1文字姓(条件付き) → 3文字姓 で speaker を解決する。

    Args:
        name: 照合対象の発言者名
        lookup: build_lookup の結果
        allow_single_char: True なら 1 文字姓マッチを許可（structurer は許可、normalizer は禁止）
        hint_affiliation: 複数候補がある場合の tie-break ヒント（segment の affiliation 等）

    Returns:
        一致した SpeakerInfo。見つからなければ None。
    """
    if not name or not lookup:
        return None
    if name in lookup:
        return lookup[name]

    for prefix_len in (2, 1, 3):
        if prefix_len > len(name):
            continue
        if prefix_len == 1:
            if not allow_single_char:
                continue
            if name[0] not in SINGLE_CHAR_SURNAMES:
                continue
        prefix = name[:prefix_len]
        candidates = [info for key, info in lookup.items() if key.startswith(prefix)]
        if not candidates:
            continue
        if len(candidates) == 1:
            return candidates[0]
        if hint_affiliation:
            for c in candidates:
                if c.affiliation == hint_affiliation:
                    return c
        logger.debug(
            "Ambiguous %d-char prefix '%s' for '%s' (%d candidates); picking first",
            prefix_len,
            prefix,
            name,
            len(candidates),
        )
        return candidates[0]

    return None
