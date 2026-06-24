"""HLS音声抽出・WAVセグメント分割の単体テスト (Step 3)"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.audio.extractor import (
    _parse_master_playlist,
    _parse_media_playlist,
    download_full_audio,
    split_segments,
)
from src.models import SpeakerInfo


def _make_dummy_wav(path: Path, duration_seconds: float = 5.0) -> Path:
    """テスト用のダミー WAV ファイルを生成する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    num_frames = int(sample_rate * duration_seconds)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_frames)
    return path


@pytest.fixture
def dummy_wav(tmp_path: Path) -> Path:
    return _make_dummy_wav(tmp_path / "test_audio.wav", duration_seconds=10.0)


@pytest.fixture
def sample_speakers() -> list[SpeakerInfo]:
    return [
        SpeakerInfo(
            name="藤原徹",
            affiliation="自由民主党",
            start_seconds=0.0,
            start_time="13:00",
            duration_minutes=2,
        ),
        SpeakerInfo(
            name="古川あおい",
            affiliation="チームみらい",
            start_seconds=3.0,
            start_time="13:02",
            duration_minutes=3,
        ),
        SpeakerInfo(
            name="山田花子",
            affiliation="立憲民主党",
            start_seconds=7.0,
            start_time="13:05",
            duration_minutes=3,
        ),
    ]


class TestDownloadFullAudio:
    def test_non_hls_url_uses_ffmpeg_direct(self, tmp_path: Path) -> None:
        """非.m3u8 URLは ffmpeg に直接渡される（keep-alive付き）。"""
        output_path = tmp_path / "audio.wav"
        url = "https://public.mediasp.jp/some/video.mp4"

        with patch("src.audio.extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            download_full_audio(url, output_path)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert url in cmd
        assert "-http_persistent" in cmd
        assert "pcm_s16le" in cmd
        assert "16000" in cmd
        assert str(output_path) in cmd

    def test_returns_output_path(self, tmp_path: Path) -> None:
        """戻り値が output_path と一致すること。"""
        output_path = tmp_path / "audio.wav"
        url = "https://example.com/video.mp4"

        with patch("src.audio.extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = download_full_audio(url, output_path)

        assert result == output_path

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """出力先の親ディレクトリが作成されること。"""
        output_path = tmp_path / "nested" / "deep" / "audio.wav"
        url = "https://example.com/video.mp4"

        with patch("src.audio.extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            download_full_audio(url, output_path)

        assert output_path.parent.exists()

    def test_ffmpeg_error_raises(self, tmp_path: Path) -> None:
        """ffmpeg が失敗した場合に CalledProcessError が送出されること。"""
        output_path = tmp_path / "audio.wav"
        url = "https://example.com/video.mp4"

        with patch("src.audio.extractor.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
            with pytest.raises(subprocess.CalledProcessError):
                download_full_audio(url, output_path)

    def test_hls_parallel_path_selects_lowest_bitrate(self, tmp_path: Path) -> None:
        """master playlist→最低帯域variantを選択し、セグメントを並列取得→ffmpegで音声抽出する。"""
        output_path = tmp_path / "audio.wav"
        master_url = "https://hlsvod.example.com/vod/_definst_/amlst:test/playlist.m3u8"

        master_text = (
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=564000,RESOLUTION=640x360\n"
            "chunklist_b564000.m3u8\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=314000,RESOLUTION=480x270\n"
            "chunklist_b314000.m3u8\n"
        )
        media_text = (
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            "#EXT-X-TARGETDURATION:10\n"
            "#EXTINF:10.0,\n"
            "media_0.ts\n"
            "#EXTINF:10.0,\n"
            "media_1.ts\n"
            "#EXT-X-ENDLIST\n"
        )

        fetched_urls: list[str] = []

        def _make_text_resp(text: str) -> MagicMock:
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.text = text
            return r

        def _make_segment_resp() -> MagicMock:
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.iter_content = MagicMock(return_value=[b"\x47" + b"\x00" * 187])
            return r

        def _fake_get(url, *args, **kwargs):  # type: ignore[no-untyped-def]
            fetched_urls.append(url)
            if url.endswith("playlist.m3u8"):
                return _make_text_resp(master_text)
            if url.endswith("chunklist_b314000.m3u8"):
                return _make_text_resp(media_text)
            if url.endswith("chunklist_b564000.m3u8"):
                # 高ビットレートは選ばれてはいけない
                raise AssertionError(f"Should not fetch high-bitrate variant: {url}")
            if url.endswith(".ts"):
                return _make_segment_resp()
            raise AssertionError(f"Unexpected URL: {url}")

        with patch("src.audio.extractor.requests.Session") as mock_session_cls, \
             patch("src.audio.extractor.subprocess.run") as mock_run:
            session = MagicMock()
            session.headers = {}
            session.get = MagicMock(side_effect=_fake_get)
            mock_session_cls.return_value = session
            mock_run.return_value = MagicMock(returncode=0)

            download_full_audio(master_url, output_path)

        # 低ビットレート variant の chunklist と segment のみ取得
        assert any(u.endswith("chunklist_b314000.m3u8") for u in fetched_urls)
        assert sum(1 for u in fetched_urls if u.endswith(".ts")) == 2

        # ffmpeg は1度だけ呼ばれる（連結後の音声抽出）
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        # 入力はローカルTSであり、外部URLではない
        i_idx = cmd.index("-i")
        assert not cmd[i_idx + 1].startswith("http")

    def test_hls_falls_back_to_ffmpeg_on_network_error(self, tmp_path: Path) -> None:
        """並列HLS経路がネットワークエラーで失敗した場合、ffmpeg直経路へフォールバックする。"""
        output_path = tmp_path / "audio.wav"
        master_url = "https://hlsvod.example.com/playlist.m3u8"

        with patch("src.audio.extractor.requests.Session") as mock_session_cls, \
             patch("src.audio.extractor.subprocess.run") as mock_run:
            session = MagicMock()
            session.headers = {}
            session.get = MagicMock(side_effect=requests.ConnectionError("DNS failed"))
            mock_session_cls.return_value = session
            mock_run.return_value = MagicMock(returncode=0)

            download_full_audio(master_url, output_path)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert master_url in cmd
        assert "-http_persistent" in cmd


class TestPlaylistParsers:
    def test_parse_master_playlist_extracts_bandwidth_and_url(self) -> None:
        text = (
            "#EXTM3U\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=564000,RESOLUTION=640x360\n"
            "high.m3u8\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=314000,RESOLUTION=480x270\n"
            "low.m3u8\n"
        )
        result = _parse_master_playlist(text, "https://x.test/a/b/playlist.m3u8")
        assert result == [
            (564000, "https://x.test/a/b/high.m3u8"),
            (314000, "https://x.test/a/b/low.m3u8"),
        ]

    def test_parse_media_playlist_extracts_segments_in_order(self) -> None:
        text = (
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            "#EXTINF:10.0,\n"
            "seg0.ts\n"
            "#EXTINF:10.0,\n"
            "seg1.ts\n"
            "#EXT-X-ENDLIST\n"
        )
        result = _parse_media_playlist(text, "https://x.test/path/list.m3u8")
        assert result == [
            "https://x.test/path/seg0.ts",
            "https://x.test/path/seg1.ts",
        ]

    def test_parse_media_playlist_handles_absolute_urls(self) -> None:
        text = (
            "#EXTM3U\n"
            "https://cdn.test/a.ts\n"
            "https://cdn.test/b.ts\n"
        )
        result = _parse_media_playlist(text, "https://x.test/list.m3u8")
        assert result == ["https://cdn.test/a.ts", "https://cdn.test/b.ts"]


class TestSplitSegments:
    def test_correct_number_of_segments(
        self, dummy_wav: Path, sample_speakers: list[SpeakerInfo], tmp_path: Path
    ) -> None:
        """発言者数と同じ数のセグメントが生成されること。"""
        output_dir = tmp_path / "segments"

        with patch("src.audio.extractor.subprocess.run") as mock_run, \
             patch("src.audio.extractor._get_audio_duration", return_value=10.0):
            mock_run.return_value = MagicMock(returncode=0)
            # ダミーファイルを作成してからsplit
            for i, s in enumerate(sample_speakers):
                safe_name = s.name.replace("/", "_").replace(" ", "_")
                (tmp_path / "segments").mkdir(parents=True, exist_ok=True)
                (tmp_path / "segments" / f"{i:03d}_{safe_name}.wav").touch()

            result = split_segments(dummy_wav, sample_speakers, output_dir)

        assert len(result) == len(sample_speakers)

    def test_ffmpeg_ss_to_args(
        self, dummy_wav: Path, sample_speakers: list[SpeakerInfo], tmp_path: Path
    ) -> None:
        """ffmpeg に正しい -ss / -to 引数が渡されること。"""
        output_dir = tmp_path / "segments"

        with patch("src.audio.extractor.subprocess.run") as mock_run, \
             patch("src.audio.extractor._get_audio_duration", return_value=10.0):
            mock_run.return_value = MagicMock(returncode=0)
            split_segments(dummy_wav, sample_speakers, output_dir)

        # 3つのセグメントに対して3回 ffmpeg が呼ばれること
        assert mock_run.call_count == len(sample_speakers)

        # 並列実行のため順序は不定 → 全呼び出しからコマンドを収集して検証
        all_cmds = [call[0][0] for call in mock_run.call_args_list]

        # -ss 0.0 / -to 3.0 のコマンドが存在すること（最初のセグメント）
        first_seg = [c for c in all_cmds if "0.0" in c and "3.0" in c]
        assert len(first_seg) == 1
        assert "-ss" in first_seg[0]
        assert "-to" in first_seg[0]

        # -to 10.0 のコマンドが存在すること（最後のセグメント）
        last_seg = [c for c in all_cmds if "10.0" in c]
        assert len(last_seg) == 1
        assert "-to" in last_seg[0]

    def test_segment_filename_format(
        self, dummy_wav: Path, sample_speakers: list[SpeakerInfo], tmp_path: Path
    ) -> None:
        """セグメントファイル名が {index:03d}_{speaker_name}.wav 形式であること。"""
        output_dir = tmp_path / "segments"

        with patch("src.audio.extractor.subprocess.run") as mock_run, \
             patch("src.audio.extractor._get_audio_duration", return_value=10.0):
            mock_run.return_value = MagicMock(returncode=0)
            result = split_segments(dummy_wav, sample_speakers, output_dir)

        assert result[0].name == "000_藤原徹.wav"
        assert result[1].name == "001_古川あおい.wav"
        assert result[2].name == "002_山田花子.wav"

    def test_last_segment_uses_total_duration(
        self, dummy_wav: Path, tmp_path: Path
    ) -> None:
        """最後のセグメントの終了時刻が全体の長さであること。"""
        speakers = [
            SpeakerInfo(
                name="話者A",
                affiliation="党A",
                start_seconds=0.0,
                start_time="10:00",
                duration_minutes=5,
            ),
            SpeakerInfo(
                name="話者B",
                affiliation="党B",
                start_seconds=5.0,
                start_time="10:05",
                duration_minutes=5,
            ),
        ]
        output_dir = tmp_path / "segments"
        total_duration = 15.0

        with patch("src.audio.extractor.subprocess.run") as mock_run, \
             patch("src.audio.extractor._get_audio_duration", return_value=total_duration):
            mock_run.return_value = MagicMock(returncode=0)
            split_segments(dummy_wav, speakers, output_dir)

        last_cmd = mock_run.call_args_list[-1][0][0]
        to_idx = last_cmd.index("-to")
        assert float(last_cmd[to_idx + 1]) == total_duration


class TestSubprocessTimeouts:
    """PR17: ffmpeg/ffprobe subprocess.run に必ず timeout を渡すことを検証する。"""

    def test_ffmpeg_direct_download_has_timeout(self, tmp_path: Path) -> None:
        output_path = tmp_path / "audio.wav"
        url = "https://example.com/video.mp4"

        with patch("src.audio.extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            download_full_audio(url, output_path)

        kwargs = mock_run.call_args.kwargs
        assert "timeout" in kwargs and kwargs["timeout"] >= 600

    def test_split_segments_passes_timeout(
        self, dummy_wav: Path, sample_speakers: list[SpeakerInfo], tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "segments"

        with patch("src.audio.extractor.subprocess.run") as mock_run, \
             patch("src.audio.extractor._get_audio_duration", return_value=10.0):
            mock_run.return_value = MagicMock(returncode=0)
            split_segments(dummy_wav, sample_speakers, output_dir)

        for call in mock_run.call_args_list:
            assert "timeout" in call.kwargs, "split_segments must pass timeout="
            assert call.kwargs["timeout"] > 0

    def test_get_audio_duration_passes_timeout(self, dummy_wav: Path) -> None:
        from src.audio.extractor import _get_audio_duration

        with patch("src.audio.extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="42.5\n", returncode=0)
            _get_audio_duration(dummy_wav)

        kwargs = mock_run.call_args.kwargs
        assert kwargs.get("timeout") == 30  # ffprobe メタ取得は 30s

    def test_detect_leading_silence_passes_timeout(self, dummy_wav: Path) -> None:
        from src.audio.extractor import detect_leading_silence

        with patch("src.audio.extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stderr="", returncode=0)
            detect_leading_silence(dummy_wav)

        kwargs = mock_run.call_args.kwargs
        assert "timeout" in kwargs and kwargs["timeout"] >= 60


@pytest.mark.integration
class TestAudioExtractorIntegration:
    def test_real_hls_download(self, tmp_path: Path) -> None:
        """実際のHLSストリームからダウンロードをテストする（結合テスト、要ffmpeg）。"""
        hls_url = "http://hlsvod.shugiintv.go.jp/vod/_definst_/amlst:2026/2026-0409-1300-00/playlist.m3u8"
        output_path = tmp_path / "audio.wav"
        result = download_full_audio(hls_url, output_path)
        assert result.exists()
        assert result.stat().st_size > 0
