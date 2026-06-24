"""BaseScraper ABC: 両院共通のスクレイパーインターフェース"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import SessionDetail


class SessionNotReadyError(Exception):
    """セッションページは存在するが発言者リストがまだ公開されていない場合の例外。

    動画アップロード後、発言者表の反映までに時間差がある。これはリトライで解決する
    一時的な状態なので、恒久的なエラーと区別する。
    """


class BaseScraper(ABC):
    """両院共通のScraperインターフェース。

    各院のScraper実装はこの3メソッドを実装する。
    エンコーディング処理など院固有の差異は各実装に閉じ込める。
    """

    chamber: str  # "shugiin" | "sangiin"

    @abstractmethod
    def detect_new_sessions(self, date: str) -> list[str]:
        """指定日（YYYY-MM-DD）の新規セッションIDリストを返す。

        Args:
            date: 検索対象日（YYYY-MM-DD形式）

        Returns:
            セッションIDの文字列リスト（衆議院ならdeli_id、参議院ならsid）
        """
        ...

    @abstractmethod
    def get_session_detail(self, session_id: str) -> SessionDetail:
        """セッション詳細（メタデータ + 発言者リスト）を返す。

        Args:
            session_id: 院固有のセッションID

        Returns:
            SessionDetail Pydanticモデル
        """
        ...

    @abstractmethod
    def get_audio_url(self, session_id: str) -> str:
        """音声ストリームURLを返す。

        Args:
            session_id: 院固有のセッションID

        Returns:
            HLS URLまたはストリームURL文字列
        """
        ...
