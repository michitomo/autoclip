"""Step 5↔5.5 間の metadata.speakers 逆補完 (PR6, §2.2/2.3)。

scrapers が抽出した metadata.speakers には委員長・質疑者しか含まれず、
答弁者 (大臣・副大臣・政務官) と政府参考人 (局長・審議官等) が登録されていない
ことが多い (TV ページのタイムスタンプリンク仕様の制約)。

本モジュールは speaker_tagger 出力 (UtterancesOutput) から
role∈{答弁者, 政府参考人} の発言者を抽出し、既存 metadata.speakers と
fuzzy 一致しないものを追加する。affiliation は委員長 utterance 内の
「○○大臣 ××君。」等の指名パターンから推定する (推定不可なら空文字)。

役割確定ロジック:
- 推定 affiliation が非空 → derive_role(affiliation) で role 決定
- 推定 affiliation が空 → speaker_tagger が付けた role (答弁者 or 政府参考人) を踏襲
"""

from __future__ import annotations

import logging
import re

from src.models import SpeakerInfo, UtterancesOutput
from src.scrapers._role import derive_role
from src.speaker_lookup import build_lookup, find_by_name

logger = logging.getLogger(__name__)

# 答弁者・政府参考人の役職タイトルキーワード (長い順、re alternation で誤分割回避)
_ANSWERER_TITLE_KEYWORDS: tuple[str, ...] = tuple(
    sorted(
        (
            "内閣総理大臣",
            "総理大臣",
            "国務大臣",
            "大臣政務官",
            "副大臣",
            "大臣",
            "副長官",
            "長官",
            "事務次長",
            "次長",
            "部長",
            "局長",
            "審議官",
            "参事官",
            "課長",
            "総裁",
            "理事長",
            "本部長",
        ),
        key=len,
        reverse=True,
    )
)

_TITLE_KEYWORDS_PAT = "|".join(re.escape(k) for k in _ANSWERER_TITLE_KEYWORDS)

_NAME_CHARS = r"[ぁ-ゟァ-ヿ一-鿿]"
_HONORIFIC = r"(?:君|氏|さん|議員|委員)"

# 役職タイトル + 人名 + 敬称 — non-greedy 20-char prefix で省名等を含めて拾う
# PR43 fix: {0,20} に拡張 (公正取引委員会事務総局官房審議官 等の長い省庁名に対応)
# PR44: 役職と人名の間に「、」等の区切り文字がある場合にも対応
#   「内閣総理大臣、高市早苗君。」「防衛大臣木原稔君。」両パターンを許容
_NOMINATION_PATTERN = re.compile(
    rf"(?P<title>[一-鿿]{{0,20}}?(?:{_TITLE_KEYWORDS_PAT}))"
    r"[、，\s]{0,2}"  # 0〜2 字の区切り (「、」「, 」等)
    rf"(?P<name>{_NAME_CHARS}{{2,8}}){_HONORIFIC}"
)

# PR42: utterance テキスト先頭から「内閣総理大臣の高市でございます」等のパターンで役職を抽出
# 0〜20 字の省名プレフィックス + 役職キーワード、直後は「の/は/で/、/。/：/空白」
# PR42 fix: {0,20} に拡張 + 全角コロン「：」を追加
_UTT_AFFILIATION_RE = re.compile(
    rf"^([一-鿿ぁ-ゟァ-ヿ]{{0,20}}?(?:{_TITLE_KEYWORDS_PAT}))"
    r"(?:の|は|で[ごあ]|でし|、|。|：|\s|$)"
)

_ENRICH_ROLES: frozenset[str] = frozenset(("答弁者", "政府参考人", "参考人"))

# PR33: 質疑者も含め、metadata に未登録なら補完する対象ロール
# 委員長・議長はスクレイパーが通常取得するため除外
_ALL_VALID_ROLES: frozenset[str] = frozenset(
    ("答弁者", "政府参考人", "参考人", "質疑者")
)

# PR26: 既存 speakers の role を utterances から逆引き補完する際に許容する役割
_BACKFILL_ROLES: frozenset[str] = frozenset(
    ("委員長", "議長", "質疑者", "答弁者", "政府参考人", "参考人")
)

# 委員長相当 (進行役) の追加キーワード — 名前末尾から affiliation 抽出時に使う
_CHAIR_LIKE_KEYWORDS: tuple[str, ...] = ("委員長", "事務総長", "副議長", "議長")


def _extract_affiliation_from_name(name: str) -> str:
    """speaker name 自体が役職タイトル混じり文字列のとき affiliation を抽出する。

    speaker_tagger が「松本大臣」「内閣府宇宙開発戦略推進事務局長」のような
    役職込みの speaker 名を返すケースに対応する。

    名前末尾が _ANSWERER_TITLE_KEYWORDS / _CHAIR_LIKE_KEYWORDS のいずれかと
    一致したら、(短い surname + suffix の場合) は suffix だけ、(役職描写型の場合)
    は name 全体を affiliation として返す。

    省/府/院/庁/院/局/部/委員会/会議 等を含む長い prefix は役職描写型と判定する。
    """
    if not name:
        return ""
    keywords = (*_ANSWERER_TITLE_KEYWORDS, *_CHAIR_LIKE_KEYWORDS)
    # 長い順にチェック (重複ある可能性があるが endswith マッチで先勝ち)
    for kw in sorted(keywords, key=len, reverse=True):
        if name.endswith(kw):
            prefix = name[: -len(kw)]
            if not prefix:
                return kw
            # prefix に役職描写キーワードがあれば name 全体を affiliation に
            if any(
                marker in prefix
                for marker in ("省", "府", "院", "庁", "局", "部", "委員会", "会議", "事務")
            ):
                return name
            return kw
    return ""


def _extract_affiliation_from_utterance_text(text: str, speaker_name: str = "") -> str:
    """utterance テキスト先頭から役職タイトルを抽出する (PR42)。

    「内閣総理大臣の高市でございます」→「内閣総理大臣」
    「厚生労働省社会・援護局長の山下です」→「厚生労働省社会・援護局長」
    「高市早苗内閣総理大臣：ご質問にお答えします」→「内閣総理大臣」（speaker_name="高市早苗"）

    speaker_name を渡すと抽出結果の先頭から人名プレフィックスを除去する。
    マッチしない場合は空文字を返す。
    """
    if not text:
        return ""
    m = _UTT_AFFILIATION_RE.match(text.strip())
    if not m:
        return ""
    extracted = m.group(1)
    # speaker_name が抽出結果の先頭と一致する場合は除去して役職部分のみ返す
    # 例: "高市早苗内閣総理大臣" → "内閣総理大臣"（speaker_name="高市早苗" のとき）
    if speaker_name and extracted.startswith(speaker_name):
        extracted = extracted[len(speaker_name):]
    return extracted or ""


def _build_utterance_role_map(utterances: UtterancesOutput) -> dict[str, str]:
    """speaker name → 観測 role のマップを作る (PR26)。

    最初に見つかった有効 role を採用する。「その他」は採用しない。
    """
    role_map: dict[str, str] = {}
    for seg in utterances.segments:
        for u in seg.utterances:
            name = (u.speaker or "").strip()
            if not name or u.role not in _BACKFILL_ROLES:
                continue
            if name not in role_map:
                role_map[name] = u.role
    return role_map


def _backfill_existing_speaker_roles(
    speakers: list[SpeakerInfo],
    role_map: dict[str, str],
) -> int:
    """既存 speakers の role を再計算 (PR26 + PR29/PR30 で更新、in-place)。

    補完優先度:
        1. ``derive_role(affiliation)`` が「その他」以外を返せばそれを採用
           — 既存 role が異なる値だった場合も上書きする (PR29/PR30 の修正版
           derive_role を partial regen でも反映するため)
        2. utterances の観測 role (`role_map`) を採用 (role 空 or その他 のとき)
        3. 最終フォールバック: "その他"

    Returns: 実際に role が更新されたエントリ数。
    """
    updated = 0
    for sp in speakers:
        # まず affiliation 由来で再計算 — derive_role 修正 (PR29/PR30) が
        # 既存データにも適用されるよう、role 値の有無に関わらず実施。
        derived = derive_role(sp.affiliation) if sp.affiliation else "その他"
        if derived != "その他":
            if sp.role != derived:
                sp.role = derived
                updated += 1
            continue
        # affiliation で決まらない場合は既存 role を維持 (空でなければ)
        if sp.role and sp.role != "その他":
            continue
        utt_role = role_map.get(sp.name)
        if utt_role and utt_role in _BACKFILL_ROLES:
            if sp.role != utt_role:
                sp.role = utt_role
                updated += 1
            continue
        # フォールバック (Pydantic デフォルトとの差を防ぐ)
        if not sp.role:
            sp.role = "その他"
            updated += 1
    return updated


def _format_hms(seconds: float) -> str:
    """秒数を `HH:MM` 形式に整形する (PR26.1、SpeakerInfo.start_time 用)。

    seconds <= 0 / NaN は空文字列を返す。HH は 0 詰め 2 桁、ただし >= 100h は
    桁数増加。
    """
    if seconds is None or seconds <= 0:
        return ""
    total_minutes = int(seconds) // 60
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h:02d}:{m:02d}"


def _build_chair_nomination_map(utterances: UtterancesOutput) -> dict[str, str]:
    """委員長 utterance を全 scan して name → 推定タイトル マップを作る。

    最初に検出したタイトルを採用する (重複時は上書きしない)。
    """
    name_to_title: dict[str, str] = {}
    for seg in utterances.segments:
        for u in seg.utterances:
            if u.role != "委員長":
                continue
            for m in _NOMINATION_PATTERN.finditer(u.text):
                title = m.group("title")
                name = m.group("name")
                if not name or not title:
                    continue
                if name not in name_to_title:
                    name_to_title[name] = title
    return name_to_title


def enrich_metadata_from_utterances(
    utterances: UtterancesOutput,
    speakers: list[SpeakerInfo],
) -> list[SpeakerInfo]:
    """utterances から答弁者・政府参考人・質疑者を抽出して speakers に逆補完。

    既存 speakers との fuzzy 重複チェックを行い、新規エントリのみ追加する。
    affiliation は委員長 utterance の「(役職)<名前>君。」パターンから推定する。

    PR32: start_seconds / duration_minutes を utterance 観測ベースで補完。
    PR33: 答弁系ロール (_ENRICH_ROLES) に加え、質疑者・委員長・議長も
    metadata 未登録なら追加対象とする。

    Args:
        utterances: speaker_tagger の出力 (Step 5 後・Step 5.5 前)
        speakers: scrapers が抽出した既存 speakers (metadata.speakers)

    Returns:
        既存 + 補完エントリ を含む新リスト (入力 speakers は変更しない)
    """
    lookup = build_lookup(speakers)
    nomination_map = _build_chair_nomination_map(utterances)
    role_map = _build_utterance_role_map(utterances)

    # PR26: 既存 speakers の role を補完 (古い metadata.json で role="" のケース対応)
    # 入力 speakers を破壊しないよう各 SpeakerInfo を deep-copy する
    enriched = [s.model_copy() for s in speakers]
    backfilled = _backfill_existing_speaker_roles(enriched, role_map)
    if backfilled:
        logger.info(
            "metadata role backfill: filled %d/%d existing speakers",
            backfilled, len(speakers),
        )

    # PR26.1 + PR32: name → (first_seen_at, last_seen_at) で発言区間を記録
    first_seen_at: dict[str, float] = {}
    last_seen_at: dict[str, float] = {}
    for seg in utterances.segments:
        for u in seg.utterances:
            name = (u.speaker or "").strip()
            if not name:
                continue
            if name not in first_seen_at:
                first_seen_at[name] = seg.start_seconds
            last_seen_at[name] = seg.start_seconds

    # PR32: 既存 speakers の duration_minutes が 0 ならば utterance 観測から補完
    for sp in enriched:
        if sp.duration_minutes > 0:
            continue
        t0 = first_seen_at.get(sp.name, 0.0)
        t1 = last_seen_at.get(sp.name, 0.0)
        if t1 > t0:
            sp.duration_minutes = max(1, int((t1 - t0) / 60))

    # name → (affiliation, fallback_role) を蓄積
    # PR33: _ENRICH_ROLES のみではなく _ALL_VALID_ROLES を対象とする
    candidates: dict[str, tuple[str, str]] = {}

    for seg in utterances.segments:
        for u in seg.utterances:
            if u.role not in _ALL_VALID_ROLES:
                continue
            name = (u.speaker or "").strip()
            if not name:
                continue
            if find_by_name(name, lookup, allow_single_char=True) is not None:
                continue
            if name in candidates:
                continue
            # 1. 委員長指名文から推定
            affiliation = nomination_map.get(name, "")
            # 2. 推定不可なら speaker name 自体から末尾役職を抽出
            if not affiliation:
                affiliation = _extract_affiliation_from_name(name)
            # 3. PR42: それでも不明なら utterance テキスト先頭から役職パターンを抽出
            if not affiliation and u.role in _ENRICH_ROLES:
                affiliation = _extract_affiliation_from_utterance_text(u.text, name)
            candidates[name] = (affiliation, u.role)

    if not candidates:
        return enriched

    for name, (affiliation, fallback_role) in candidates.items():
        if affiliation:
            role = derive_role(affiliation)
            # derive_role が想定外 ("質疑者" 等) を返した場合は fallback を優先
            if role not in _ENRICH_ROLES:
                role = fallback_role
        else:
            role = fallback_role
        # PR26.1: affiliation が空なら role 名を最低限の affiliation として使う
        # ("答弁者" / "政府参考人" / "参考人")。ただし 質疑者/委員長/議長 は
        # affiliation 空のままにして party 等は未知扱い。
        if not affiliation and role in _ENRICH_ROLES:
            affiliation = role
        # PR26.1: start_seconds を最初の発言 segment.start_seconds で補完
        start_seconds = first_seen_at.get(name, 0.0)
        # PR32: duration_minutes を観測区間から算出
        t1 = last_seen_at.get(name, start_seconds)
        dur = max(1, int((t1 - start_seconds) / 60)) if t1 > start_seconds else 0
        enriched.append(
            SpeakerInfo(
                name=name,
                affiliation=affiliation,
                role=role,
                start_seconds=start_seconds,
                start_time=_format_hms(start_seconds),
                duration_minutes=dur,
            )
        )

    logger.info(
        "metadata enrichment: existing=%d, added=%d (with affiliation=%d)",
        len(speakers),
        len(candidates),
        sum(1 for _, (a, _r) in candidates.items() if a),
    )
    return enriched


__all__ = ["enrich_metadata_from_utterances", "_extract_affiliation_from_utterance_text"]
