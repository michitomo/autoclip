"""母集合 (corpus) JSON 生成 CLI — GitHub Pages 統合UI 用 (autoclip 固有)。

衆議院TV を直近 N 営業日ぶんスクレイプし、「その日にどの委員会・どの議員が
質疑したか」の母集合を **単一の corpus.json** に書き出す。生成済みか否かは無関係
(生成済み判定はフロントが catalog.json と clip_id を突合して行う)。

build_site.py との違い:
  - **ASR/LLM/ffmpeg を一切呼ばない** (HTTP スクレイプのみ・API キー不要)。
  - 出力は corpus.json 1 ファイル (日付→委員会→議員)。
  - 既存 corpus.json を読み、**既知 session_id の detail 取得をスキップ** (詳細ページは
    一度公開されたら不変なので再利用)。calendar GET だけは毎回叩く (新規委員会検出)。

使い方:
  python -m src.build_corpus --out ../site/data --days 10
出力: <out>/corpus.json
  {
    "generated_at": "...",
    "days": {
      "2026-06-19": [
        {"session_id","committee","duration",
         "members":[{"name","affiliation","role","duration_minutes","clip_id"}]}
      ]
    }
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.clip_service import _safe_name
from src.scrapers.shugiin import ShugiinScraper, SessionNotReadyError

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# detail GET 間に挟む待ち (shugiintv.go.jp へのレート配慮)。
_REQUEST_PAD_SEC = 0.4

# corpus に含める役割 (質疑者中心。答弁者/参考人も切り抜き対象になり得るので残す)。
# 議長/委員長/その他 は司会・進行なので除外。
_INCLUDE_ROLES = {"質疑者", "答弁者", "政府参考人", "参考人"}


def _recent_weekdays(today: datetime, n: int) -> list[str]:
    """today から遡って直近 n 営業日 (平日) の YYYY-MM-DD を新しい順で返す。"""
    out: list[str] = []
    cur = today
    while len(out) < n:
        if cur.weekday() < 5:  # 0=月 .. 4=金
            out.append(cur.strftime("%Y-%m-%d"))
        cur -= timedelta(days=1)
    return out


def _load_prev_details(corpus_path: Path) -> dict[str, dict]:
    """既存 corpus.json から session_id -> session オブジェクト の辞書を作る。

    detail 再取得をスキップして使い回すための索引。ファイルが無ければ空。
    """
    if not corpus_path.exists():
        return {}
    try:
        prev = json.loads(corpus_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    index: dict[str, dict] = {}
    for sessions in (prev.get("days") or {}).values():
        for s in sessions:
            sid = s.get("session_id")
            if sid:
                index[sid] = s
    return index


def _session_from_detail(scraper: ShugiinScraper, sid: str, committee: str,
                         duration: str | None) -> dict:
    """detail を取得して corpus の session オブジェクトを作る。"""
    detail = scraper.get_session_detail(sid)
    members = []
    for sp in detail.speakers:
        if sp.role not in _INCLUDE_ROLES:
            continue
        members.append({
            "name": sp.name,
            "affiliation": sp.affiliation,
            "role": sp.role,
            "duration_minutes": sp.duration_minutes,
            "clip_id": f"{sid}_{_safe_name(sp.name)}",
        })
    return {
        "session_id": sid,
        # detail 側の委員会名を優先 (calendar はリンク文字列で表記揺れがあり得る)
        "committee": detail.committee or committee,
        "duration": detail.duration or duration,
        "members": members,
    }


def build(out_dir: Path, days: int) -> dict:
    """直近 days 営業日の母集合を out_dir/corpus.json に書き出す。"""
    scraper = ShugiinScraper()
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.json"
    prev = _load_prev_details(corpus_path)

    dates = _recent_weekdays(datetime.now(JST), days)
    logger.info("対象日: %s (既知 session %d 件)", dates, len(prev))

    days_out: dict[str, list[dict]] = {}
    n_detail_fetched = 0
    n_detail_reused = 0

    for date in dates:
        try:
            cal = scraper.get_session_calendar(date)
        except Exception as e:  # noqa: BLE001 - 1 日失敗しても続行
            logger.error("calendar 取得失敗 (%s): %s", date, e)
            continue
        if not cal:
            continue  # 審議なしの日は days に載せない

        sessions: list[dict] = []
        for row in cal:
            sid = str(row["session_id"])
            committee = str(row.get("committee") or "")
            duration = row.get("duration")
            # 既知 session は detail を再取得しない (詳細は不変)。
            if sid in prev and prev[sid].get("members"):
                sessions.append(prev[sid])
                n_detail_reused += 1
                continue
            try:
                sess = _session_from_detail(scraper, sid, committee, duration)
                n_detail_fetched += 1
                time.sleep(_REQUEST_PAD_SEC)
            except SessionNotReadyError:
                # 議員未公開: 委員会は載せ members 空 (次回 cron で埋まる)。
                logger.info("議員未公開 (deli_id=%s) — members 空で載せる", sid)
                sess = {"session_id": sid, "committee": committee,
                        "duration": duration, "members": []}
            except Exception as e:  # noqa: BLE001
                logger.error("detail 取得失敗 (deli_id=%s): %s", sid, e)
                continue
            sessions.append(sess)
        if sessions:
            days_out[date] = sessions

    corpus = {
        "generated_at": datetime.now(JST).isoformat(),
        "days": days_out,
    }
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), "utf-8")
    logger.info(
        "corpus.json 書き出し: %d 日, detail 取得 %d / 再利用 %d",
        len(days_out), n_detail_fetched, n_detail_reused,
    )
    return corpus


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="母集合 corpus.json を生成 (HTTP のみ)")
    ap.add_argument("--out", required=True, type=Path, help="出力先 (site/data 等)")
    ap.add_argument("--days", type=int, default=10, help="直近 N 営業日 (既定 10)")
    args = ap.parse_args()
    build(args.out, args.days)


if __name__ == "__main__":
    main()
