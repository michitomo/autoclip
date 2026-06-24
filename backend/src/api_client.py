"""共通 LLM API クライアントファクトリ & 並列数設定

Whisper (Step 3/4) は DeepInfra (transcriber.py 側で独立クライアント) を使う。
Step 4.5 / 5 / 6 (corrector / speaker_tagger / structurer / metrics) は
本モジュール get_client() 経由で OpenRouter を使う。
"""

from __future__ import annotations

import logging
import os
import random
import resource
import time
from collections.abc import Callable
from typing import TypeVar

import openai

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 既定 LLM モデル。env AUTOCLIP_LLM_MODEL で上書き可能。校正・文節分割で使う。
# gemma-4-26b-a4b-it は MoE (active 4B) で 31b 密モデルより速く、日本語の名前/社名
# 修正の品質も上 (実測: アンソルピック→Anthropic, クロード→Claude まで修正)。
# 検証メモ (2026-06-19): mercury-2 は高速だが長文で幻覚・名前修正失敗で不採用。
# flash 系 (qwen3.6/gemini-3.5/deepseek-v4) は 9〜14s と遅く gemma MoE に劣る。
LLM_MODEL = os.environ.get("AUTOCLIP_LLM_MODEL", "google/gemma-4-26b-a4b-it")

# ---------------------------------------------------------------------------
# ステップごとのデフォルト並列数
# OpenRouter は provider 経由で rate limit が分散されるため、DeepInfra 時代と
# 同じ 80 をそのまま使う (実測で詰まれば下げる)
# ---------------------------------------------------------------------------
MAX_WORKERS_AUDIO = 4        # Step 3: ffmpeg subprocess（fd重い）
MAX_WORKERS_HLS = 8          # Step 3前段: HLSセグメント並列取得（HTTP keep-alive）
MAX_WORKERS_WHISPER = 16     # Step 4: Whisper API (DeepInfra)
MAX_WORKERS_LLM = 80         # Step 4.5/5/6: LLM API (OpenRouter)


def ensure_fd_limit(minimum: int = 2048) -> None:
    """ファイルディスクリプタ上限を引き上げる（macOS対策）。

    macOSのデフォルトは256と低く、ffmpeg並列＋API並列で枯渇しやすい。
    パイプライン起動時に1度呼ぶ。
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < minimum:
            new_soft = min(minimum, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            logging.getLogger(__name__).info(
                "Raised fd limit: %d → %d (hard=%d)", soft, new_soft, hard,
            )
    except (ValueError, OSError) as e:
        logging.getLogger(__name__).warning("Could not raise fd limit: %s", e)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# 429 / 5xx に対するリトライ設定
_MAX_RETRIES = 6
_BASE_DELAY = 2.0   # 秒（指数バックオフのベース）
_MAX_DELAY = 120.0  # 秒（上限）
_JITTER = 0.25      # ±25% のランダムジッター


def with_retry(fn: Callable[[], _T]) -> _T:
    """429 (rate limit) と 5xx (server error) に対してリトライする。

    指数バックオフ + ジッター戦略:
        delay = min(BASE * 2^attempt, MAX) * uniform(1-JITTER, 1+JITTER)

    429 の Retry-After ヘッダがあればその値を優先する。

    Args:
        fn: 呼び出す関数（引数なし）

    Returns:
        fn() の戻り値

    Raises:
        openai.RateLimitError: リトライ上限を超えた場合
        openai.APIStatusError: 5xx でリトライ上限を超えた場合
        その他の例外: リトライせずそのまま再 raise
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except openai.RateLimitError as e:
            if attempt == _MAX_RETRIES:
                logger.error("Rate limit exceeded after %d retries", _MAX_RETRIES)
                raise
            retry_after = _parse_retry_after(e)
            delay = retry_after if retry_after else _backoff(attempt)
            logger.warning(
                "429 rate limit (attempt %d/%d), retrying in %.1fs",
                attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)
        except openai.APIStatusError as e:
            if e.status_code < 500 or attempt == _MAX_RETRIES:
                raise
            delay = _backoff(attempt)
            logger.warning(
                "HTTP %d server error (attempt %d/%d), retrying in %.1fs",
                e.status_code, attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)

    # unreachable — kept for type checker
    raise RuntimeError("with_retry: unexpected exit")  # pragma: no cover


def _backoff(attempt: int) -> float:
    """指数バックオフ + ジッターで待機秒数を返す。"""
    base = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    return base * random.uniform(1 - _JITTER, 1 + _JITTER)


def _parse_retry_after(exc: openai.RateLimitError) -> float | None:
    """Retry-After ヘッダ（秒数）をパースして返す。なければ None。"""
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            value = response.headers.get("retry-after") or response.headers.get("Retry-After")
            if value:
                return float(value)
    except Exception:
        pass
    return None


def get_client() -> openai.OpenAI:
    """OpenRouter APIクライアントを返す (LLM ステップ専用)。"""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OSError("OPENROUTER_API_KEY environment variable is not set")
    return openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
