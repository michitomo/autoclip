"""委員会名を 3 段階フォールバックで解決するモジュール。"""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from src.models import SpeakerInfo

logger = logging.getLogger(__name__)

_KANJI = r"[一-鿿ぁ-んァ-ヶ・]"
_COMMITTEE_PATTERN = re.compile(rf"{_KANJI}+委員会")
_SPECIAL_CATEGORIES: tuple[str, ...] = ("審査会", "調査会", "公聴会")


def find_committee_by_label(soup: BeautifulSoup) -> str:
    """「会議名」というラベルの隣にあるテキストを探す。"""
    # 参議院TV: <dt>会議名</dt><dd>...</dd>
    dt = soup.find("dt", string=re.compile(r"会議名"))
    if dt:
        dd = dt.find_next_sibling("dd")
        if dd:
            return _scan_text(dd.get_text(strip=True))

    # 衆議院TV: <td><b>会議名</b></td> ... <td>...</td>
    label = soup.find(["b", "td", "th"], string=re.compile(r"会議名"))
    if label:
        # 親の tr を探して、その中の td を走査
        tr = label.find_parent("tr")
        if tr:
            for td in tr.find_all("td"):
                t = td.get_text(strip=True)
                if t and "会議名" not in t and ":" not in t and "：" not in t:
                    if found := _scan_text(t):
                        return found
    return ""



def find_committee_in_title(soup: BeautifulSoup) -> str:
    """<title> タグ内から委員会名を探す（衆議院ページ向け）。"""
    title_tag = soup.find("title")
    if not title_tag:
        return ""
    text = title_tag.get_text(strip=True)
    # title タグは長くなるが「本会議」「○○委員会」等は文字数に関わらず抽出する。
    # _scan_text の 30 字制限は誤検知防止のため本文段落向けに設けており title には不適切。
    if "本会議" in text:
        return "本会議"
    return _scan_text(text)


def find_committee_in_body(soup: BeautifulSoup) -> str:
    """構造化されたメタデータタグから委員会名を探す。

    見出し (h1〜h3) と表組セル (td/th/dd) のみを対象にする。本文段落
    (p/div/span) は誤検知 (本文に「○○委員会」言及があるだけのケース) の
    温床になるためスコープから外す。「会議名」ラベルや <title> は
    `find_committee_by_label` / `find_committee_in_title` で別途処理される。
    """
    for tag in soup.find_all(["h1", "h2", "h3"]):
        if found := _scan_text(tag.get_text(strip=True)):
            return found
    for tag in soup.find_all(["td", "th", "dd"]):
        if found := _scan_text(tag.get_text(strip=True)):
            return found
    return ""


def derive_committee_from_speakers(speakers: list[SpeakerInfo]) -> str:
    """speakers の affiliation 末尾「委員長」を「委員会」に置換して返す。

    Returns:
        マッチした委員会名（例: "内閣委員長" → "内閣委員会"）。マッチなしは "".
    """
    for s in speakers:
        if s.affiliation.endswith("委員長") and len(s.affiliation) > len("委員長"):
            return s.affiliation[: -len("委員長")] + "委員会"
    return ""


def resolve_committee(soup: BeautifulSoup, speakers: list[SpeakerInfo]) -> str:
    """4 段階フォールバックで委員会名を解決する。

    Stage 1: 「会議名」ラベルの隣から抽出
    Stage 2: title → body から「〇〇委員会」「本会議」「審査会」等を直接抽出
    Stage 3: speakers.affiliation 末尾「委員長」→「委員会」に置換
    Stage 4: "不明" を返す
    """
    if found := find_committee_by_label(soup):
        return found
    if found := find_committee_in_title(soup):
        return found
    if found := find_committee_in_body(soup):
        return found
    if found := derive_committee_from_speakers(speakers):
        logger.info("Committee resolved via speakers fallback: %s", found)
        return found
    return "不明"


def _scan_text(text: str) -> str:
    """1 行のテキストから委員会・本会議・審査会等を見つけて返す。"""
    if not text:
        return ""

    # "本会議" のマッチング。
    # 免責事項などの長い文章（30文字以上）に含まれる場合は誤検知として無視する。
    if "本会議" in text:
        if len(text) < 30:
            return "本会議"

    if match := _COMMITTEE_PATTERN.search(text):
        return match.group(0)
    for category in _SPECIAL_CATEGORIES:
        if category in text:
            if match := re.search(rf"{_KANJI}+{category}", text):
                return match.group(0)
            return category
    return ""
