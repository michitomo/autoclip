"""インプロセス非同期ジョブレジストリ (autoclip API 用)。

クリップ生成は 2〜3 分かかるため、API は即座に job_id を返し、クライアントは
GET /api/jobs/{id} で進捗をポーリングする。外部キュー無し (ローカル・単一プロセス
前提)。ThreadPoolExecutor でジョブを走らせ、進捗 step を job に記録する。
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    """1 件の非同期ジョブの状態。"""

    id: str
    kind: str  # "clip" | "render" など
    state: JobState = "queued"
    step: str = "queued"  # scraping/downloading/transcribing/.../done
    result: dict[str, Any] | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        """API レスポンス用の dict。"""
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "step": self.step,
            "result": self.result,
            "error": self.error,
            "meta": self.meta,
        }


class JobRegistry:
    """スレッドセーフなジョブ管理 + 実行。"""

    def __init__(self, max_workers: int = 2) -> None:
        # クリップ生成は CPU(ffmpeg)+ネットワーク混在。同時実行は控えめに。
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, meta: dict[str, Any] | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, meta=meta or {})
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _set(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in fields.items():
                    setattr(job, k, v)

    def submit(
        self,
        job: Job,
        fn: Callable[[Callable[[str], None]], dict[str, Any]],
    ) -> None:
        """fn(progress_cb) をワーカースレッドで実行する。

        fn は進捗コールバック (step 文字列) を受け取り、結果 dict を返す。
        例外は job.error に記録し state=error にする。
        """

        def _run() -> None:
            self._set(job.id, state="running", step="starting")

            def _progress(step: str) -> None:
                self._set(job.id, step=step)

            try:
                result = fn(_progress)
                self._set(job.id, state="done", step="done", result=result)
            except Exception as e:  # noqa: BLE001 - ジョブ失敗はクライアントに返す
                logger.exception("Job %s failed", job.id)
                self._set(
                    job.id, state="error", step="error",
                    error=f"{type(e).__name__}: {e}",
                    meta={**job.meta, "traceback": traceback.format_exc()[-2000:]},
                )

        self._executor.submit(_run)
