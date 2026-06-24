"""衆議院TV (shugiintv.go.jp) スクレイパー

EUC-JP エンコーディングを明示的に処理する。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from src.models import SessionDetail, SpeakerInfo
from src.scrapers._committee import resolve_committee
from src.scrapers._role import derive_role
from src.scrapers._session_kind import detect_session_kind
from src.scrapers._speakers import merge_fuzzy_duplicates
from src.scrapers.base import BaseScraper, SessionNotReadyError

logger = logging.getLogger(__name__)

BASE_URL = "https://www.shugiintv.go.jp/jp/index.php"
JST = timezone(timedelta(hours=9))

_NON_SPEAKER_TEXTS = {"はじめから再生", "先頭から再生", "全体再生"}


class ShugiinScraper(BaseScraper):
    """衆議院TV (shugiintv.go.jp) スクレイパー。"""

    chamber = "shugiin"

    def detect_new_sessions(self, date: str) -> list[str]:
        """カレンダーGETで指定日のdeli_idリストを返す。

        Args:
            date: 検索対象日（YYYY-MM-DD形式、例: "2026-04-09"）

        Returns:
            deli_idの文字列リスト（重複なし）
        """
        u_day = date.replace("-", "")
        url = f"{BASE_URL}?{urlencode({'ex': 'VL', 'u_day': u_day})}"
        logger.info("Fetching session list for date=%s: %s", date, url)

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "euc-jp"

        return list(set(re.findall(r"deli_id=(\d+)", response.text)))

    def get_session_calendar(self, date: str) -> list[dict[str, str | None]]:
        """指定日のカレンダーGET 1回で {session_id, committee, duration} を返す (軽量)。

        日付検索結果ページの各行 (<tr>) に deli_id リンク・委員会名 (リンク文字列)・
        審議時間が載っているため、詳細ページを開かずに一覧プレビューを作れる。
        審議が無い日は空リスト。重い話者・HLS 取得は session detail 側に任せる。

        Args:
            date: YYYY-MM-DD

        Returns:
            [{"session_id", "committee", "duration"}] (ページ掲載順、重複なし)
        """
        u_day = date.replace("-", "")
        url = f"{BASE_URL}?{urlencode({'ex': 'VL', 'u_day': u_day})}"
        logger.info("Fetching calendar for date=%s: %s", date, url)

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "euc-jp"

        soup = BeautifulSoup(response.text, "html.parser")
        out: list[dict[str, str | None]] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=re.compile(r"deli_id=\d+")):
            m = re.search(r"deli_id=(\d+)", a["href"])
            if not m:
                continue
            sid = m.group(1)
            if sid in seen:
                continue
            seen.add(sid)
            committee = a.get_text(strip=True)
            row = a.find_parent("tr")
            row_text = row.get_text(" ", strip=True) if row else ""
            dur_m = re.search(r"(?:\d+時間)?\d+分", row_text)
            out.append(
                {
                    "session_id": sid,
                    "committee": committee,
                    "duration": dur_m.group(0) if dur_m else None,
                }
            )
        return out

    def get_session_detail(self, session_id: str) -> SessionDetail:
        """衆議院TVの詳細ページからセッション情報を取得する。"""
        url = f"{BASE_URL}?{urlencode({'ex': 'VL', 'deli_id': session_id})}"
        logger.info("Fetching session detail: %s", url)

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "euc-jp"

        soup = BeautifulSoup(response.text, "html.parser")

        hls_url = _extract_hls_url(soup)
        if not hls_url:
            raise ValueError(f"HLS URL not found for deli_id={session_id}")

        speakers = _extract_speakers(soup, session_id)

        if not speakers:
            raise SessionNotReadyError(
                f"Speaker list not yet published for deli_id={session_id}. "
                "Will retry later."
            )

        for s in speakers:
            s.role = derive_role(s.affiliation)

        committee = resolve_committee(soup, speakers)
        date_str = _extract_date(soup, session_id)
        duration = _extract_duration(soup)
        page_text = soup.get_text()
        session_kind = detect_session_kind(page_text, committee, speakers)

        return SessionDetail(
            chamber="shugiin",
            session_id=session_id,
            date=date_str,
            committee=committee,
            session_kind=session_kind,
            duration=duration,
            hls_url=hls_url,
            source_url=url,
            processed_at=datetime.now(JST).isoformat(),
            speakers=speakers,
        )

    def get_audio_url(self, session_id: str) -> str:
        """HLS URLを返す（詳細ページから軽量取得）。"""
        url = f"{BASE_URL}?{urlencode({'ex': 'VL', 'deli_id': session_id})}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "euc-jp"

        soup = BeautifulSoup(response.text, "html.parser")
        hls_url = _extract_hls_url(soup)
        if not hls_url:
            raise ValueError(f"HLS URL not found for deli_id={session_id}")
        return hls_url


def get_session_detail(deli_id: str) -> SessionDetail:
    """後方互換用: ShugiinScraper().get_session_detail() のエイリアス。"""
    return ShugiinScraper().get_session_detail(deli_id)


def _extract_hls_url(soup: BeautifulSoup) -> str:
    """hidden input #vtag_src_base_vod からHLS URLを取得する。"""
    tag = soup.find("input", {"id": "vtag_src_base_vod"})
    if not tag:
        return ""
    value = tag.get("value", "")
    if not value:
        return ""
    if value.startswith("http://"):
        return value.replace("http://", "https://", 1)
    if value.startswith("https://"):
        return value
    return f"https://hlsvod.shugiintv.go.jp/vod/_definst_/amlst:{value}"


def _extract_date(soup: BeautifulSoup, deli_id: str) -> str:
    """ページから西暦/令和の日付を YYYY-MM-DD 形式で返す。"""
    title_tag = soup.find("title")
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        if found := _parse_reiwa(title_text):
            return found

    for tag in soup.find_all(string=re.compile(r"\d{4}年\d+月\d+日")):
        match = re.search(r"(\d{4})年(\d+)月(\d+)日", str(tag))
        if match and 2000 <= int(match.group(1)) <= 2100:
            return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

    for tag in soup.find_all(string=re.compile(r"令和\d+年\d+月\d+日")):
        if found := _parse_reiwa(str(tag)):
            return found

    raise ValueError(f"Could not extract date for deli_id={deli_id}")


def _parse_reiwa(text: str) -> str:
    """令和X年Y月Z日 を YYYY-MM-DD に変換する。マッチなしは ""。"""
    match = re.search(r"令和(\d+)年(\d+)月(\d+)日", text)
    if not match:
        return ""
    year = 2018 + int(match.group(1))
    return f"{year:04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def _extract_duration(soup: BeautifulSoup) -> str:
    """所要時間（"XX時間XX分"）を返す。見つからなければ ""。"""
    for tag in soup.find_all(string=re.compile(r"^\s*\d+時間\d+分\s*$")):
        return str(tag).strip()
    return ""


def _extract_speakers(soup: BeautifulSoup, deli_id: str) -> list[SpeakerInfo]:
    """発言者リストを抽出する。

    実際のページ構造:
    <tr>
      <td></td>
      <td><a href="...&time=1310.4">森英介(衆議院議長)</a></td>
      <td>13時 02分</td>
      <td>01分</td>
    </tr>

    同一人物の複数スロット (午前/午後など) は (name, affiliation) で dedup し、
    start_seconds は最小、duration_minutes は合算、start_time は若い方を残す。
    """
    seen: dict[tuple[str, str], SpeakerInfo] = {}
    seen_order: list[tuple[str, str]] = []

    anchors = soup.find_all("a", href=re.compile(r"time=[\d.]+"))

    for anchor in anchors:
        href = anchor.get("href", "")
        time_match = re.search(r"time=([\d.]+)", href)
        if not time_match:
            continue

        start_seconds = float(time_match.group(1))
        raw_text = anchor.get_text(strip=True)

        if not raw_text or raw_text in _NON_SPEAKER_TEXTS:
            continue

        name, affiliation = _parse_speaker_text(raw_text)
        if not name:
            continue

        start_time, duration_minutes = _find_speaker_row_data(anchor)

        key = (name, affiliation)
        if key in seen:
            existing = seen[key]
            if start_seconds < existing.start_seconds:
                existing.start_seconds = start_seconds
                existing.start_time = start_time
            existing.duration_minutes += duration_minutes
            continue

        seen[key] = SpeakerInfo(
            name=name,
            affiliation=affiliation,
            start_seconds=start_seconds,
            start_time=start_time,
            duration_minutes=duration_minutes,
        )
        seen_order.append(key)

    speakers = [seen[k] for k in seen_order]
    pre_merge = len(speakers)
    speakers = merge_fuzzy_duplicates(speakers)
    if len(speakers) != pre_merge:
        logger.info(
            "Fuzzy merge dedup'd speakers for deli_id=%s: %d -> %d",
            deli_id, pre_merge, len(speakers),
        )
    logger.info("Extracted %d speakers for deli_id=%s", len(speakers), deli_id)
    return speakers


def _parse_speaker_text(text: str) -> tuple[str, str]:
    """アンカーテキストから名前と所属を分離する。

    例:
        "古川あおい(チームみらい)" -> ("古川あおい", "チームみらい")
        "古川あおい" -> ("古川あおい", "")
    """
    match = re.match(r"^(.+?)\((.+)\)\s*$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text.strip(), ""


def _find_speaker_row_data(anchor_tag: object) -> tuple[str, int]:
    """アンカータグの親 <tr> から開始時刻と所要時間を取得する。"""
    parent = getattr(anchor_tag, "parent", None)
    for _ in range(5):
        if parent is None:
            break
        if getattr(parent, "name", None) == "tr":
            return _parse_table_row(parent)
        parent = getattr(parent, "parent", None)

    logger.warning(
        "Could not find parent <tr> for speaker anchor: %s",
        getattr(anchor_tag, "text", "")[:50],
    )
    return "", 0


def _parse_table_row(tr_tag: object) -> tuple[str, int]:
    """テーブル行のセルから開始時刻と所要時間を取得する。"""
    cells = getattr(tr_tag, "find_all", lambda *a, **k: [])("td")
    cell_texts = [c.get_text(strip=True) for c in cells]

    start_time = ""
    duration_minutes = 0

    for text in cell_texts:
        jtime_match = re.match(r"^(\d{1,2})時\s*(\d{2})分$", text)
        if jtime_match:
            h, m = int(jtime_match.group(1)), int(jtime_match.group(2))
            start_time = f"{h:02d}:{m:02d}"
            continue

        if re.match(r"^\d{1,2}:\d{2}$", text):
            start_time = text
            continue

        duration_match = re.match(r"^(\d+)分$", text)
        if duration_match:
            duration_minutes = int(duration_match.group(1))

    return start_time, duration_minutes
