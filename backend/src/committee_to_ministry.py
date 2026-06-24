"""委員会名 → 所管省庁のマッピングと、laws_compact.txt の予選フィルタ。

Step 6c の関連法案タグ付けでプロンプトに渡す候補法案を絞り込むために使う。
75 法案を 10〜15 件程度に予選することで LLM の precision/recall 双方が改善する。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

COMMITTEE_TO_MINISTRY: dict[str, list[str]] = {
    # 本会議・運営
    "本会議": [],
    "議院運営委員会": [],
    "国家基本政策委員会": [],
    "憲法審査会": [],
    "情報監視審査会": [],
    # 予算・決算
    "予算委員会": ["内閣官房", "財務省"],
    "決算行政監視委員会": ["財務省", "会計検査院"],
    # 常任委員会
    "内閣委員会": ["内閣官房", "内閣府", "デジタル庁"],
    "総務委員会": ["総務省"],
    "法務委員会": ["法務省"],
    "外務委員会": ["外務省"],
    "財務金融委員会": ["財務省", "金融庁"],
    "文部科学委員会": ["文部科学省"],
    "厚生労働委員会": ["厚生労働省"],
    "農林水産委員会": ["農林水産省"],
    "経済産業委員会": ["経済産業省"],
    "国土交通委員会": ["国土交通省"],
    "環境委員会": ["環境省"],
    "安全保障委員会": ["防衛省"],
    # 参議院常任委員会（名称が異なるもの）
    "外交防衛委員会": ["外務省", "防衛省"],
    "財政金融委員会": ["財務省", "金融庁"],
    "国民生活・経済及び地方に関する調査会": [],
    # 特別委員会（複数省庁にまたがるため pass-through）
    "災害対策特別委員会": [],
    "原子力問題調査特別委員会": [],
    "原子力特別委員会": [],  # 短縮名エイリアス
    "東日本大震災復興特別委員会": ["復興庁"],
    "地方創生及びデジタル社会の形成等に関する特別委員会": ["デジタル庁"],
    "政治倫理確立及び公職選挙法改正に関する特別委員会": [],
    "政治改革に関する特別委員会": [],
    "政治改革特別委員会": [],  # 短縮名エイリアス
    "消費者問題に関する特別委員会": [],
    "消費者問題特別委員会": [],  # 短縮名エイリアス
    "沖縄及び北方問題に関する特別委員会": [],
    "沖縄北方特別委員会": [],  # 短縮名エイリアス
    "拉致問題等に関する特別委員会": [],
    "拉致問題特別委員会": [],  # 短縮名エイリアス
    # スクレイパーが解決できなかった場合のフォールバック
    "不明": [],
    "特別委員会": [],
}


def filter_laws_for_committee(laws_compact: str, committee: str) -> str:
    """laws_compact.txt の各行を所管省庁でフィルタする。

    フィルタが効かない（マッピング未登録 or マッチ 0 件）場合は全件返す。
    これにより precision を狙いつつ recall は劣化させない。
    """
    if not laws_compact:
        return ""

    ministries = COMMITTEE_TO_MINISTRY.get(committee, [])
    if not ministries:
        return laws_compact

    lines = laws_compact.splitlines()
    matched = [line for line in lines if any(m in line for m in ministries)]
    if not matched:
        logger.info(
            "filter_laws_for_committee: %s (%s) matched 0/%d laws; falling back to full list",
            committee,
            ",".join(ministries),
            len(lines),
        )
        return laws_compact

    logger.info(
        "filter_laws_for_committee: %s narrowed %d laws to %d via %s",
        committee,
        len(lines),
        len(matched),
        ",".join(ministries),
    )
    return "\n".join(matched)


__all__ = ["COMMITTEE_TO_MINISTRY", "filter_laws_for_committee"]
