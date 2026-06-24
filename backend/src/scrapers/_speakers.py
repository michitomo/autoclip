"""speakers リストの fuzzy 重複マージ (PR24)。

衆議院/参議院の `_extract_speakers` で `(name, affiliation)` キー dedup を行った
後段で本モジュールの `merge_fuzzy_duplicates` を呼び、表記揺れによる重複を解消する。

典型ケース (F2 audit より):
- 同一人物が "鈴木" / "鈴木大臣" / "鈴木憲和" として 3 件登録されてしまう
- "片山さつき" と "片山さつき (財務大臣)" が別エントリになる

マージ判定:
- 名前が完全一致、または片方が他方の prefix (短い方が 2 文字以上)
- affiliation が衝突しない (どちらかが空、または片方が他方の substring)

衝突時は別人物として両方残す。affiliation だけでは別人物の判定が完璧にできない
ため、過剰マージを避ける方を優先 (= 安全側)。
"""

from __future__ import annotations

from src.models import SpeakerInfo


def _names_fuzzy_match(a: str, b: str) -> bool:
    """name が fuzzy 一致するか。完全一致または短い方が長い方の prefix。"""
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    if len(short) < 2:
        return False
    return long_.startswith(short)


def _affiliations_compatible(a: str, b: str) -> bool:
    """affiliation が衝突しないか。両方非空かつ substring 関係も無ければ False。"""
    if not a or not b:
        return True
    return a == b or a in b or b in a


def _pick_affiliation(a: str, b: str) -> str:
    """より情報量の多い affiliation を返す (compatible 前提)。"""
    if not a:
        return b
    if not b:
        return a
    if a == b:
        return a
    if a in b:
        return b
    if b in a:
        return a
    return a  # safety net (compatible なら到達しない)


def merge_fuzzy_duplicates(speakers: list[SpeakerInfo]) -> list[SpeakerInfo]:
    """name 部分一致 + affiliation 統合で speakers をマージする (PR24)。

    入力リストの順序は保つ。各エントリについて、それより前のエントリと fuzzy 一致
    しかつ affiliation が衝突しなければ、前のエントリにマージする。

    マージ動作:
        - name: より長い (情報量多い) 方を採用
        - affiliation: 非空 / 長い方を採用 (`_pick_affiliation`)
        - start_seconds / start_time: より小さい (= 早い) 方
        - duration_minutes: 合算

    role は scraper 側で merge 後に `derive_role(affiliation)` で再計算される前提
    のため、本関数では更新しない。
    """
    merged: list[SpeakerInfo] = []
    for sp in speakers:
        target_idx: int | None = None
        for j, existing in enumerate(merged):
            if not _names_fuzzy_match(sp.name, existing.name):
                continue
            if not _affiliations_compatible(sp.affiliation, existing.affiliation):
                continue
            target_idx = j
            break
        if target_idx is None:
            merged.append(sp)
            continue

        target = merged[target_idx]
        if len(sp.name) > len(target.name):
            target.name = sp.name
        target.affiliation = _pick_affiliation(target.affiliation, sp.affiliation)
        if sp.start_seconds < target.start_seconds:
            target.start_seconds = sp.start_seconds
            target.start_time = sp.start_time
        target.duration_minutes += sp.duration_minutes
    return merged


__all__ = ["merge_fuzzy_duplicates"]
