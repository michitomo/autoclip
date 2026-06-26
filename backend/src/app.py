"""autoclip REST API (FastAPI)。

ブラウザ UI のバックエンド。日付→委員会→議員 を返し、クリップ生成を非同期ジョブで
実行する。この API ペアが「自動化用の自前 API」要件も満たす (ヘッドレス利用可)。

起動:
    cd backend && .venv/bin/uvicorn src.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.api_client import ensure_fd_limit
from src.clip_service import (
    _safe_name,
    apply_tree_to_ranges,
    generate_clip,
    render_topic_clips,
    render_topic_preview,
    rerender_project,
)
from src.jobs import JobRegistry
from src.models import ClipProject
from src.scrapers.base import SessionNotReadyError
from src.scrapers.shugiin import ShugiinScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MEDIA_DIR = Path(__file__).parent.parent.parent / "media" / "clips"


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    ensure_fd_limit()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="autoclip API", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ローカル開発: Vite dev server からのアクセスを許可
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = JobRegistry(max_workers=2)
_scraper = ShugiinScraper()

# session_id -> SessionDetail のメモリキャッシュ (スクレイプ削減)
_detail_cache: dict[str, Any] = {}


def _get_detail(session_id: str):
    """SessionDetail をキャッシュ付きで取得する。"""
    if session_id in _detail_cache:
        return _detail_cache[session_id]
    detail = _scraper.get_session_detail(session_id)
    _detail_cache[session_id] = detail
    return detail


# ---------------------------------------------------------------------------
# カレンダー (日付 → 委員会名の軽量プレビュー。詳細取得なし)
# ---------------------------------------------------------------------------

# date(YYYY-MM-DD) -> [{session_id, committee, duration}] のメモリキャッシュ。
# カレンダーGET 1回ぶんを保持し、日付ピッカーのプレビュー再取得を抑える。
_calendar_cache: dict[str, list[dict[str, Any]]] = {}


@app.get("/api/calendar")
def get_calendar(date: str = Query(..., description="YYYY-MM-DD")) -> dict[str, Any]:
    """指定日の委員会名一覧を軽量に返す (カレンダーGET 1回・詳細取得なし)。

    日付ピッカーのプレビュー用。返り値の sessions が空なら「その日は審議なし」。
    重い話者・HLS 取得は /api/sessions（委員会選択後）に任せる。
    """
    if date in _calendar_cache:
        return {"date": date, "sessions": _calendar_cache[date]}
    try:
        sessions = _scraper.get_session_calendar(date)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Failed to fetch calendar: {e}") from e
    _calendar_cache[date] = sessions
    return {"date": date, "sessions": sessions}


# ---------------------------------------------------------------------------
# セッション一覧 (日付 → 委員会)
# ---------------------------------------------------------------------------


@app.get("/api/sessions")
def list_sessions(date: str = Query(..., description="YYYY-MM-DD")) -> dict[str, Any]:
    """指定日の委員会セッション一覧を返す (deli_id ごとに detail を取得)。"""
    try:
        ids = _scraper.detect_new_sessions(date)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Failed to list sessions: {e}") from e

    sessions: list[dict[str, Any]] = []

    def _fetch(sid: str) -> dict[str, Any] | None:
        try:
            d = _get_detail(sid)
        except SessionNotReadyError:
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("detail %s failed: %s", sid, e)
            return None
        n_members = sum(1 for s in d.speakers if s.role == "質疑者")
        return {
            "session_id": sid,
            "committee": d.committee,
            "date": d.date,
            "duration": d.duration,
            "session_kind": d.session_kind,
            "n_speakers": len(d.speakers),
            "n_members": n_members,
        }

    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(_fetch, s) for s in ids]):
            r = fut.result()
            if r:
                sessions.append(r)

    sessions.sort(key=lambda s: s["committee"])
    return {"date": date, "sessions": sessions}


# ---------------------------------------------------------------------------
# 議員一覧 (セッション → 質疑者)
# ---------------------------------------------------------------------------


@app.get("/api/sessions/{session_id}/members")
def list_members(session_id: str) -> dict[str, Any]:
    """セッションの発言者のうち質疑者を返す。"""
    try:
        d = _get_detail(session_id)
    except SessionNotReadyError as e:
        raise HTTPException(425, f"Session not ready: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Failed to fetch session: {e}") from e

    members = [
        {
            "name": s.name,
            "affiliation": s.affiliation,
            "role": s.role,
            "start_seconds": s.start_seconds,
            "duration_minutes": s.duration_minutes,
        }
        for s in d.speakers
        if s.role == "質疑者"
    ]
    return {
        "session_id": session_id,
        "committee": d.committee,
        "date": d.date,
        "members": members,
    }


# ---------------------------------------------------------------------------
# クリップ生成 (非同期ジョブ)
# ---------------------------------------------------------------------------


class ClipRequest(BaseModel):
    session_id: str
    member: str
    aspect: str = "9:16"
    subtitle_style: str = "karaoke"
    title: str | None = None  # 指定時は LLM 生成せずこれを使う
    preview_seconds: float | None = None
    remove_disfluencies: bool = True  # 言い間違い・言い淀みを LLM 検出して JetCut
    dead_air_gap: float | None = None  # 語間無音の除去しきい値 秒 (None=サーバ既定の積極値)


def _clip_filename(session_id: str, member: str) -> Path:
    return MEDIA_DIR / session_id / f"{_safe_name(member)}_clip.mp4"


def _run_clip(req: ClipRequest):
    out_dir = MEDIA_DIR / req.session_id

    def _job(progress):
        kwargs: dict[str, Any] = dict(
            session_id=req.session_id,
            member=req.member,
            out_dir=out_dir,
            aspect=req.aspect,
            subtitle_style=req.subtitle_style,
            title_override=req.title,
            preview_seconds=req.preview_seconds,
            remove_disfluencies=req.remove_disfluencies,
            progress=progress,
        )
        if req.dead_air_gap is not None:
            kwargs["dead_air_gap"] = req.dead_air_gap
        result = generate_clip(**kwargs)
        # 全体レンダは廃止: 生成完了 = 編集可能 (project.json 準備済み)。
        # プレビューはトピック単位でオンデマンド (/preview-topic)。
        n_topics = 0
        proj_path = out_dir / f"{_safe_name(result.member.name)}_project.json"
        try:
            proj = ClipProject.model_validate_json(proj_path.read_text("utf-8"))
            n_topics = len(proj.qa_tree.topics) if proj.qa_tree else 0
        except Exception:  # noqa: BLE001
            pass
        return {
            "clip_path": None,  # 全体クリップは作らない
            "session_id": req.session_id,
            "member": result.member.name,
            "affiliation": result.member.affiliation,
            "duration": result.edl.kept_duration,
            "n_ranges": len(result.edl.keep_ranges),
            "n_topics": n_topics,
        }

    return _job


@app.post("/api/clips")
def create_clip(req: ClipRequest) -> dict[str, Any]:
    """クリップ生成ジョブを開始し job_id を返す。"""
    job = jobs.create("clip", meta={"session_id": req.session_id, "member": req.member})
    jobs.submit(job, _run_clip(req))
    return {"job_id": job.id}


@app.post("/api/clips/{session_id}/render")
def rerender_clip(session_id: str, req: ClipRequest) -> dict[str, Any]:
    """タイトル等を変えて再生成する (フルパイプライン再実行)。"""
    req.session_id = session_id
    job = jobs.create("render", meta={"session_id": session_id, "member": req.member})
    jobs.submit(job, _run_clip(req))
    return {"job_id": job.id}


# ---------------------------------------------------------------------------
# 編集 (字幕手修正 + カットトグル) — Whisper/LLM 再実行なしの軽量再レンダリング
# ---------------------------------------------------------------------------


def _project_path(session_id: str, member: str) -> Path:
    return MEDIA_DIR / session_id / f"{_safe_name(member)}_project.json"


@app.get("/api/clips/{session_id}/{member}/project")
def get_project(session_id: str, member: str) -> dict[str, Any]:
    """編集プロジェクト (字幕・カット区間) を返す。"""
    p = _project_path(session_id, member)
    if not p.exists():
        raise HTTPException(404, "project not found (clip 未生成?)")
    return ClipProject.model_validate_json(p.read_text(encoding="utf-8")).model_dump()


@app.post("/api/clips/{session_id}/{member}/edit")
def edit_clip(session_id: str, member: str, project: ClipProject) -> dict[str, Any]:
    """編集後の project を受け取り、軽量再レンダリングジョブを開始する。"""
    out_dir = MEDIA_DIR / session_id
    # パス安全性: source_video はファイル名のみ許可
    project.source_video = Path(project.source_video).name
    project.session_id = session_id
    project.member = member

    job = jobs.create("edit", meta={"session_id": session_id, "member": member})

    def _job(progress):
        progress("rendering")
        # qa_tree があれば leaf の選択から ranges.enabled を再計算 (描画契約)。
        # 空選択は apply_tree_to_ranges が ValueError → job.error として UI に出る。
        apply_tree_to_ranges(project)
        clip = rerender_project(project, out_dir)
        return {
            "clip_path": str(clip.relative_to(MEDIA_DIR)),
            "session_id": session_id,
            "member": member,
            "n_ranges": sum(1 for r in project.ranges if r.enabled),
        }

    jobs.submit(job, _job)
    return {"job_id": job.id}


@app.post("/api/clips/{session_id}/{member}/export")
def export_clips(
    session_id: str, member: str, project: ClipProject,
    mode: str = Query("topics", pattern="^(topics|full|both)$"),
    topic: int | None = Query(None, ge=0),
) -> dict[str, Any]:
    """編集後 project をトピック別 / 全体 / 両方のクリップに書き出すジョブを開始する。

    mode=topics: <member>_topicN_clip.mp4 を各トピックに。
    mode=full:   <member>_clip.mp4 (全体)。
    mode=both:   両方。
    topic=N:     mode=topics のとき、その index のトピックだけを書き出す
                 (トピック編集画面の「このトピックを書き出し」)。
    """
    out_dir = MEDIA_DIR / session_id
    project.source_video = Path(project.source_video).name
    project.session_id = session_id
    project.member = member

    job = jobs.create(
        "export",
        meta={"session_id": session_id, "member": member, "mode": mode, "topic": topic},
    )

    def _job(progress):
        progress("rendering")
        apply_tree_to_ranges(project)  # ツリー選択を ranges へ (空なら ValueError)
        clips: list[dict[str, Any]] = []
        if mode in ("full", "both"):
            full = rerender_project(project, out_dir)
            clips.append({
                "clip_path": str(full.relative_to(MEDIA_DIR)),
                "topic_index": None,
                "topic_label": "全体",
                "duration": round(
                    sum(r.end - r.start for r in project.ranges if r.enabled), 1
                ),
            })
        if mode in ("topics", "both"):
            clips.extend(render_topic_clips(project, out_dir, only_index=topic))
        return {"clips": clips, "session_id": session_id, "member": member}

    jobs.submit(job, _job)
    return {"job_id": job.id}


@app.post("/api/clips/{session_id}/{member}/preview-topic")
def preview_topic(
    session_id: str, member: str, project: ClipProject,
    index: int = Query(..., ge=0),
) -> dict[str, Any]:
    """編集中 project のトピック index を軽量・入力シークで焼く (オンデマンドプレビュー)。

    生成時に全体クリップは作らないので、編集画面はこれで 1 トピックだけ素早く確認する。
    job result = {clip_path, topic_index, topic_label, duration}。
    """
    out_dir = MEDIA_DIR / session_id
    project.source_video = Path(project.source_video).name
    project.session_id = session_id
    project.member = member

    job = jobs.create(
        "preview", meta={"session_id": session_id, "member": member, "index": index}
    )

    def _job(progress):
        progress("rendering")
        apply_tree_to_ranges(project)  # 編集中の選択を ranges へ反映
        info = render_topic_preview(project, index, out_dir)
        info["session_id"] = session_id
        info["member"] = member
        return info

    jobs.submit(job, _job)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.public()


# ---------------------------------------------------------------------------
# 生成済みクリップの配信 (range 対応は FileResponse が自動処理)
# ---------------------------------------------------------------------------


@app.get("/api/clips/file/{session_id}/{filename}")
def get_clip_file(session_id: str, filename: str) -> FileResponse:
    # パストラバーサル防止: ファイル名のみ許可
    safe = Path(filename).name
    path = MEDIA_DIR / session_id / safe
    if not path.exists():
        raise HTTPException(404, "clip not found")
    return FileResponse(path, media_type="video/mp4", filename=safe)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
