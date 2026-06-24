"""GitHub Pages 用の静的データ生成 CLI (autoclip 固有)。

targets (生成したい議員のリスト) を食い、指定議員だけ generate_clip を実行して
project.json / _edl.json を作り、ブラウザ(WebCodecs)がそのまま読める静的データ
(docs/data/) を書き出す。**動画レンダはしない** (レンダはブラウザ側)。

ブラウザは動画を HLS から直接取得する (CORS: ACAO:* 確認済み) ので、巨大な mp4 は
Pages に置かない。代わりに catalog にHLS URLと「議員の絶対開始秒(member_start)」を記録。
EDL の keep_ranges は議員区間先頭=0 の相対秒なので、ブラウザでは
  source_absolute = keep_range + member_start_absolute
で HLS の絶対時刻に直す。

使い方:
  python -m src.build_site --targets targets.yml --out ../docs/data
targets.yml 例:
  - session_id: "56345"      # または date: "2026-06-19" + committee: 厚生労働委員会
    member: 古川あおい
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from src.clip_service import LEAD_PAD, _resolve_member, _safe_name, generate_clip
from src.scrapers.shugiin import ShugiinScraper
from src.video.downloader import member_window

logger = logging.getLogger(__name__)


@dataclass
class Target:
    member: str
    session_id: str | None = None
    date: str | None = None
    committee: str | None = None


def _load_targets(path: Path) -> list[Target]:
    """targets ファイル (yaml or json) を読む。yaml は簡易パーサで依存を増やさない。"""
    text = path.read_text("utf-8")
    data: list[dict]
    if path.suffix in (".json",):
        data = json.loads(text)
    else:
        data = _parse_simple_yaml_list(text)
    targets: list[Target] = []
    for d in data:
        if "member" not in d:
            raise ValueError(f"target に member がありません: {d}")
        targets.append(
            Target(
                member=str(d["member"]),
                session_id=str(d["session_id"]) if d.get("session_id") else None,
                date=str(d["date"]) if d.get("date") else None,
                committee=str(d["committee"]) if d.get("committee") else None,
            )
        )
    return targets


def _parse_simple_yaml_list(text: str) -> list[dict]:
    """`- key: value` 形式の単純な YAML リストだけを解釈する (PyYAML 非依存)。

    入れ子なし・1階層の `- a: x` / `  b: y` のみ対応 (targets はこの形で足りる)。
    """
    items: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            cur = {}
            items.append(cur)
            stripped = stripped[2:]
        if cur is None:
            continue
        if ":" in stripped:
            k, _, v = stripped.partition(":")
            val = v.strip().strip('"').strip("'")
            if val:
                cur[k.strip()] = val
    return items


def _resolve_session_id(scraper: ShugiinScraper, t: Target) -> str:
    """target から session_id を決める。date+committee 指定なら calendar から引く。"""
    if t.session_id:
        return t.session_id
    if not t.date:
        raise ValueError(f"session_id も date もありません: {t}")
    cal = scraper.get_session_calendar(t.date)
    if t.committee:
        for s in cal:
            if t.committee in s["committee"] or s["committee"] in t.committee:
                return str(s["session_id"])
        raise ValueError(f"{t.date} に委員会 {t.committee!r} が見つかりません")
    if len(cal) == 1:
        return str(cal[0]["session_id"])
    raise ValueError(
        f"{t.date} に複数セッションあり。committee を指定してください: "
        f"{[s['committee'] for s in cal]}"
    )


def _catalog_entry(clip_id: str, sid: str, member_name: str, affiliation: str,
                   committee: str, date: str, project: dict) -> dict:
    """project.json から catalog の 1 エントリを作る。"""
    topics = (project.get("qa_tree") or {}).get("topics", [])
    return {
        "id": clip_id,
        "session_id": sid,
        "member": member_name,
        "affiliation": affiliation,
        "committee": committee,
        "date": date,
        "title": project.get("title", ""),
        "n_topics": len(topics),
        "topics": [
            {
                "index": tp["index"],
                "label": tp.get("label", ""),
                "question_speaker": tp.get("question_speaker", ""),
                "answer_speakers": tp.get("answer_speakers", []),
            }
            for tp in topics
        ],
    }


def build(targets_path: Path, out_dir: Path, work_dir: Path) -> dict:
    """targets を生成し、out_dir に静的データ (catalog.json + 各クリップ) を書く。

    out_dir は **リポ追跡の data/** を想定 (project.json/edl.json は軽量なので main に
    コミットして永続化)。既に out_dir/<clip_id>/project.json があればその議員は
    **再生成しない** (ASR/LLM を毎時走らせない = API 課金ゼロ)。既存 JSON から catalog
    エントリだけ作り直す。
    """
    scraper = ShugiinScraper()
    targets = _load_targets(targets_path)
    logger.info("targets: %d 件", len(targets))

    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []

    for t in targets:
        try:
            sid = _resolve_session_id(scraper, t)
            safe = _safe_name(t.member)
            clip_id = f"{sid}_{safe}"
            clip_dir = out_dir / clip_id
            existing_proj = clip_dir / "project.json"

            # 既に生成済み (JSON が data/ にある) ならスキップ。catalog だけ作り直す。
            if existing_proj.exists() and (clip_dir / "edl.json").exists():
                project = json.loads(existing_proj.read_text("utf-8"))
                clips.append(_catalog_entry(
                    clip_id, sid, project.get("member", t.member),
                    project.get("affiliation", ""),
                    project.get("committee", ""), project.get("date", ""),
                    project,
                ))
                logger.info("  -> %s (既存・スキップ)", clip_id)
                continue

            logger.info("=== 生成: session=%s member=%s ===", sid, t.member)
            detail = scraper.get_session_detail(sid)
            member_index, member_info = _resolve_member(detail, t.member)
            # 議員の絶対開始秒 (clip_begin = start - LEAD_PAD)。ブラウザが HLS を
            # 直接食う際の member_start として使う。
            start, _end = member_window(
                detail.speakers, member_index, video_duration=None
            )
            member_start_absolute = max(0.0, start - LEAD_PAD)

            # 1クリップを生成 (project.json / _edl.json を session out_dir に作る)。
            session_out = work_dir / sid
            session_out.mkdir(parents=True, exist_ok=True)
            generate_clip(sid, t.member, session_out, reuse_source=True)

            safe = _safe_name(member_info.name)
            proj_path = session_out / f"{safe}_project.json"
            edl_path = session_out / f"{safe}_edl.json"
            project = json.loads(proj_path.read_text("utf-8"))
            edl = json.loads(edl_path.read_text("utf-8"))

            # ブラウザ用にHLS URLと絶対member_startを注入 (source_video=ローカルmp4は使わない)
            # catalog/再ビルド用に committee/date/affiliation も project に残す。
            project["hls_url"] = detail.hls_url
            project["member_start"] = member_start_absolute
            project["committee"] = detail.committee
            project["date"] = detail.date
            project["affiliation"] = member_info.affiliation
            project.pop("source_video", None)

            clip_id = f"{sid}_{safe}"
            clip_dir = out_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            (clip_dir / "project.json").write_text(
                json.dumps(project, ensure_ascii=False), "utf-8"
            )
            (clip_dir / "edl.json").write_text(
                json.dumps(edl, ensure_ascii=False), "utf-8"
            )

            clips.append(_catalog_entry(
                clip_id, sid, member_info.name, member_info.affiliation,
                detail.committee, detail.date, project,
            ))
            logger.info("  -> %s (%d topics)", clip_id,
                        len((project.get("qa_tree") or {}).get("topics", [])))
        except Exception as e:  # noqa: BLE001 - 1 件失敗は飛ばして続行
            logger.error("生成失敗 (member=%s): %s", t.member, e)

    catalog = {"clips": clips, "count": len(clips)}
    (out_dir / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), "utf-8"
    )
    logger.info("catalog.json 書き出し: %d クリップ", len(clips))
    return catalog


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="GitHub Pages 用の静的クリップデータを生成")
    ap.add_argument("--targets", required=True, type=Path, help="targets.yml / .json")
    ap.add_argument("--out", required=True, type=Path, help="出力先 (docs/data 等)")
    ap.add_argument(
        "--work", type=Path, default=Path("media/clips"),
        help="中間生成物 (動画/wav) の置き場 (gitignore 対象)",
    )
    args = ap.parse_args()
    cat = build(args.targets, args.out, args.work)
    if cat["count"] == 0:
        logger.error("1件も生成できませんでした")
        sys.exit(1)


if __name__ == "__main__":
    main()
