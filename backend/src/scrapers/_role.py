"""SpeakerInfo.role を affiliation 文字列から決定論的に派生するモジュール。"""

from __future__ import annotations

from src.models import SpeakerRole

PARTY_NAMES: frozenset[str] = frozenset(
    (
        "自由民主党",
        "立憲民主党",
        "公明党",
        "国民民主党",
        "日本共産党",
        "れいわ新選組",
        "日本維新の会",
        "参政党",
        "チームみらい",
        "社会民主党",
        "教育無償化を実現する会",
        "無所属",
        "無所属の会",
    )
)

_GOV_ATTENDEE_SUFFIXES: tuple[str, ...] = (
    "局長",
    "部長",
    "審議官",
    "長官",
    "次長",
    "参事官",
    "課長",
)


_GICHO_SUFFIXES: tuple[str, ...] = ("議長", "副議長")
_GICHO_KEYWORDS: tuple[str, ...] = ("衆議院議長", "参議院議長", "副議長")

_CHAIR_SUFFIXES: tuple[str, ...] = ("委員長", "事務総長")
_CHAIR_SUBSTRINGS: tuple[str, ...] = ("委員長", "事務総長")


def derive_role(affiliation: str) -> SpeakerRole:
    """affiliation から SpeakerRole を 1 つ決定する。マッチしなければ "その他"。

    優先順位:
        参考人 (prefix) > 議長/副議長 > 委員長/事務総長 > 大臣系 >
        政府参考人(局長等) > 政党所属 > その他

    PR29: 議長/副議長 は「委員長」と区別して独立 role に。
    PR30: affiliation が「参考人 」prefix で始まれば、後段の suffix チェック
    (委員長 / 部長 / 局長 等) より優先して「参考人」を返す。これにより
    民間参考人の affiliation に役職名が含まれるケース (例「参考人 ○○委員長」)
    の誤分類を防ぐ。

    委員長相当には「臨時委員長」「事務総長」「副議長」「衆議院事務総長」等を含む。
    複合 affiliation (空白区切り等) でも substring 検出する。
    """
    if not affiliation:
        return "その他"

    # PR30: 参考人 を最優先で判定 (suffix ベースの誤分類を防ぐ)
    if affiliation.startswith("参考人") or affiliation == "参考人":
        return "参考人"

    # PR29: 議長 / 副議長 を委員長と区別
    if (
        affiliation.endswith(_GICHO_SUFFIXES)
        or any(kw in affiliation for kw in _GICHO_KEYWORDS)
        or affiliation in {"議長", "副議長"}
    ):
        return "議長"

    # 進行役 (委員長相当): 委員長/事務総長
    if (
        affiliation.endswith(_CHAIR_SUFFIXES)
        or any(s in affiliation for s in _CHAIR_SUBSTRINGS)
        or affiliation in {"委員長", "事務総長"}
    ):
        return "委員長"

    # 答弁者: 大臣・副大臣・政務官
    if affiliation.endswith(("大臣", "副大臣", "政務官")):
        return "答弁者"

    # Multi-role affiliations e.g. "財務大臣 内閣府特命担当大臣（金融） ..."
    if "大臣" in affiliation or "政務官" in affiliation:
        return "答弁者"

    if affiliation.endswith(_GOV_ATTENDEE_SUFFIXES):
        return "政府参考人"

    # affiliation が末尾 "参考人" だが prefix で捕まらなかった残ケース
    if affiliation.endswith("参考人"):
        return "参考人"

    if affiliation in PARTY_NAMES or any(party in affiliation for party in PARTY_NAMES):
        return "質疑者"

    if affiliation.endswith("会派"):
        return "質疑者"

    return "その他"
