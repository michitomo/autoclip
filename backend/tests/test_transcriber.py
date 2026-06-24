"""Whisper 文字起こしの単体テスト (Step 4)"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models import RawTranscript, SegmentTranscript, SpeakerInfo, WhisperSegment
from src.transcriber import (
    _compute_windows,
    _dedupe_overlap_text,
    _merge_window_results,
    _read_wav,
    _slice_wav_window,
    _suffix_prefix_overlap,
    transcribe_all_segments,
    transcribe_segment,
)


def _ws(start: float, end: float, text: str, **kw) -> WhisperSegment:
    """Helper: テスト用 WhisperSegment を作成。"""
    return WhisperSegment(
        id=kw.get("id", 0),
        seek=kw.get("seek", 0),
        start=start,
        end=end,
        text=text,
        tokens=[],
        temperature=0.0,
        avg_logprob=-0.2,
        compression_ratio=1.5,
        no_speech_prob=0.01,
    )


def _write_silent_wav(path: Path, duration_seconds: float, framerate: int = 16000) -> None:
    """Helper: 指定秒数の無音 WAV ファイルを作成する。"""
    nframes = int(duration_seconds * framerate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * nframes)


def _seg(start: float, end: float, text: str, *, id: int = 0) -> dict:
    """Helper: テスト用 whisper segment dict を作成する。"""
    return {
        "id": id,
        "seek": 0,
        "start": start,
        "end": end,
        "text": text,
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": -0.2,
        "compression_ratio": 1.5,
        "no_speech_prob": 0.01,
    }

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def whisper_response_data() -> dict:
    return json.loads((FIXTURES_DIR / "whisper_response.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_speaker() -> SpeakerInfo:
    return SpeakerInfo(
        name="古川あおい",
        affiliation="チームみらい",
        role="質疑者",
        start_seconds=7320.2,
        start_time="14:42",
        duration_minutes=18,
    )


@pytest.fixture
def all_speakers() -> list[SpeakerInfo]:
    return [
        SpeakerInfo(
            name="藤原徹",
            affiliation="自由民主党",
            start_seconds=0.0,
            start_time="13:00",
            duration_minutes=5,
        ),
        SpeakerInfo(
            name="古川あおい",
            affiliation="チームみらい",
            start_seconds=7320.2,
            start_time="14:42",
            duration_minutes=18,
        ),
    ]


def _make_mock_transcription(data: dict) -> MagicMock:
    """Whisper API レスポンスのモックを作成する。"""
    mock = MagicMock()
    mock.text = data["text"]
    mock.segments = data["segments"]
    return mock


class TestTranscribeSegment:
    def test_returns_segment_transcript(
        self,
        tmp_path: Path,
        sample_speaker: SpeakerInfo,
        whisper_response_data: dict,
    ) -> None:
        """SegmentTranscript が返されること。"""
        wav_path = tmp_path / "test.wav"
        wav_path.touch()

        mock_result = _make_mock_transcription(whisper_response_data)

        with patch("src.transcriber._get_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = mock_result
            mock_client_factory.return_value = mock_client

            with patch.dict("os.environ", {"DEEPINFRA_API_KEY": "test-key"}):
                result = transcribe_segment(wav_path, 1, sample_speaker, "内閣委員会")

        assert isinstance(result, SegmentTranscript)
        assert result.segment_index == 1
        assert result.speaker_name == "古川あおい"
        assert result.start_seconds == 7320.2

    def test_text_from_whisper_response(
        self,
        tmp_path: Path,
        sample_speaker: SpeakerInfo,
        whisper_response_data: dict,
    ) -> None:
        """文字起こしテキストが Whisper レスポンスのテキストと一致すること。"""
        wav_path = tmp_path / "test.wav"
        wav_path.touch()

        mock_result = _make_mock_transcription(whisper_response_data)

        with patch("src.transcriber._get_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = mock_result
            mock_client_factory.return_value = mock_client

            with patch.dict("os.environ", {"DEEPINFRA_API_KEY": "test-key"}):
                result = transcribe_segment(wav_path, 1, sample_speaker, "内閣委員会")

        assert result.text == whisper_response_data["text"]

    def test_whisper_segments_parsed(
        self,
        tmp_path: Path,
        sample_speaker: SpeakerInfo,
        whisper_response_data: dict,
    ) -> None:
        """Whisper セグメントが正しくパースされること。"""
        wav_path = tmp_path / "test.wav"
        wav_path.touch()

        mock_result = _make_mock_transcription(whisper_response_data)

        with patch("src.transcriber._get_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = mock_result
            mock_client_factory.return_value = mock_client

            with patch.dict("os.environ", {"DEEPINFRA_API_KEY": "test-key"}):
                result = transcribe_segment(wav_path, 1, sample_speaker, "内閣委員会")

        assert len(result.whisper_segments) == len(whisper_response_data["segments"])
        assert result.whisper_segments[0].text == whisper_response_data["segments"][0]["text"]

    def test_api_call_parameters(
        self,
        tmp_path: Path,
        sample_speaker: SpeakerInfo,
        whisper_response_data: dict,
    ) -> None:
        """API 呼び出しパラメータが正しいこと（言語、モデル、response_format、prompt内容）。"""
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"")

        mock_result = _make_mock_transcription(whisper_response_data)

        with patch("src.transcriber._get_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = mock_result
            mock_client_factory.return_value = mock_client

            with patch.dict("os.environ", {"DEEPINFRA_API_KEY": "test-key"}):
                transcribe_segment(wav_path, 1, sample_speaker, "内閣委員会")

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "openai/whisper-large-v3-turbo"
        assert call_kwargs["language"] == "ja"
        assert call_kwargs["response_format"] == "verbose_json"
        assert "prompt" in call_kwargs
        # prompt に発言者名・委員会名が含まれること（V2サフィックス形式）
        assert "古川あおい" in call_kwargs["prompt"]
        assert "内閣委員会" in call_kwargs["prompt"]
        # ループを誘発した出席議員リストが含まれないこと
        assert "出席議員" not in call_kwargs["prompt"]
        # 削除された石井啓一副議長が含まれないこと
        assert "石井啓一" not in call_kwargs["prompt"]

    def test_missing_api_key_raises(
        self,
        tmp_path: Path,
        sample_speaker: SpeakerInfo,
    ) -> None:
        """DEEPINFRA_API_KEY が未設定の場合に EnvironmentError が送出されること。"""
        wav_path = tmp_path / "test.wav"
        wav_path.touch()

        import os
        env = {k: v for k, v in os.environ.items() if k != "DEEPINFRA_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(OSError):
                transcribe_segment(wav_path, 1, sample_speaker, "内閣委員会")


class TestTranscribeAllSegments:
    def test_returns_raw_transcript(
        self,
        tmp_path: Path,
        all_speakers: list[SpeakerInfo],
        whisper_response_data: dict,
    ) -> None:
        """RawTranscript が返されること。"""
        wav_paths = [tmp_path / f"seg_{i:03d}.wav" for i in range(len(all_speakers))]
        for p in wav_paths:
            p.touch()

        mock_result = _make_mock_transcription(whisper_response_data)

        with patch("src.transcriber._get_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.return_value = mock_result
            mock_client_factory.return_value = mock_client

            with patch.dict("os.environ", {"DEEPINFRA_API_KEY": "test-key"}):
                result = transcribe_all_segments(
                    wav_paths, all_speakers, "56149", committee="内閣委員会",
                )

        assert isinstance(result, RawTranscript)
        assert result.session_id == "56149"
        assert len(result.segments) == len(all_speakers)


class TestComputeWindows:
    def test_short_returns_single_window(self) -> None:
        """ウィンドウサイズ以下なら 1 ウィンドウのみ。"""
        assert _compute_windows(20.0, window_seconds=25.0, overlap_seconds=5.0) == [(0.0, 20.0)]

    def test_exact_window_size(self) -> None:
        """ウィンドウサイズちょうどなら 1 ウィンドウ。"""
        assert _compute_windows(25.0, window_seconds=25.0, overlap_seconds=5.0) == [(0.0, 25.0)]

    def test_long_audio_overlap(self) -> None:
        """長い音声では step=window-overlap で進む。"""
        windows = _compute_windows(60.0, window_seconds=25.0, overlap_seconds=5.0)
        # step=20s: [0,25] [20,45] [40,60]
        assert windows == [(0.0, 25.0), (20.0, 45.0), (40.0, 60.0)]

    def test_zero_duration(self) -> None:
        assert _compute_windows(0.0) == []

    def test_overlap_exceeds_window(self) -> None:
        with pytest.raises(ValueError):
            _compute_windows(60.0, window_seconds=10.0, overlap_seconds=15.0)

    def test_last_window_clips_to_duration(self) -> None:
        """最後のウィンドウは end > duration にならない。"""
        windows = _compute_windows(50.0, window_seconds=25.0, overlap_seconds=5.0)
        assert windows[-1][1] == 50.0
        # ウィンドウは [0,25] [20,45] [40,50] になる想定
        assert windows == [(0.0, 25.0), (20.0, 45.0), (40.0, 50.0)]


class TestSliceWavWindow:
    def test_slice_produces_valid_wav(self, tmp_path: Path) -> None:
        """切り出した WAV が wave モジュールで再読み込みできること。"""
        src = tmp_path / "src.wav"
        _write_silent_wav(src, duration_seconds=10.0)
        wav = _read_wav(src)

        sliced_bytes = _slice_wav_window(wav, 2.0, 5.0)
        with wave.open(io.BytesIO(sliced_bytes), "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 16000
            # 3 秒分のフレーム数 = 48000
            assert w.getnframes() == 16000 * 3


class TestMergeWindowResults:
    def test_single_window_keeps_all_segments(self) -> None:
        windows = [(0.0, 25.0)]
        window_segments = [
            [_seg(0.0, 10.0, "前半"), _seg(10.0, 20.0, "後半", id=1)],
        ]
        merged = _merge_window_results(windows, window_segments, overlap_seconds=5.0)
        assert [m.text for m in merged] == ["前半", "後半"]
        assert merged[0].start == 0.0
        assert merged[1].start == 10.0

    def test_overlap_dedup_by_midpoint(self) -> None:
        """重複領域の同一発言は片方の trust window にのみ採用される (片方は除外)。"""
        # windows: [0,25] [20,45] (last), overlap=5 → half_overlap=2.5
        # window0 trust: [0, 22.5)        ← 末尾の 2.5s は隣に譲る
        # window1 trust: [22.5, 45]       ← 最後の window なので末尾まで採用
        windows = [(0.0, 25.0), (20.0, 45.0)]
        window_segments = [
            [
                _seg(0.0, 10.0, "始まり"),
                # 重複領域 (abs mid=23, window0 trust [0,22.5) OUT)
                _seg(21.0, 25.0, "重複w0", id=1),
            ],
            [
                # 同じ重複領域 (abs mid=23, window1 trust [22.5,45] IN)
                _seg(1.0, 5.0, "重複w1"),
                # window1 中央 (abs mid=32.5, IN)
                _seg(10.0, 15.0, "中央", id=1),
            ],
        ]
        merged = _merge_window_results(windows, window_segments, overlap_seconds=5.0)
        texts = [m.text for m in merged]
        # 始まり (window0 採用) + 重複w1 (window1 採用) + 中央 (window1 採用)
        # 重複w0 は window0 trust 外 → 除外 = dedup 成立
        assert texts == ["始まり", "重複w1", "中央"]

    def test_overlap_boundary_recovered_by_neighbor(self) -> None:
        """ある window が境界で発言を落としても、隣接 window が拾えること。"""
        windows = [(0.0, 25.0), (20.0, 45.0)]
        window_segments = [
            # window0 は境界ロスを模擬 (前半発言のみ)
            [_seg(0.0, 10.0, "前半")],
            [
                # window1 で境界発言が完全に取れる (abs mid=23, window1 trust IN)
                _seg(1.0, 5.0, "境界発言を完全に拾った"),
                _seg(10.0, 15.0, "次の発言", id=1),
            ],
        ]
        merged = _merge_window_results(windows, window_segments, overlap_seconds=5.0)
        texts = [m.text for m in merged]
        assert "境界発言を完全に拾った" in texts

    def test_absolute_timestamps(self) -> None:
        """ウィンドウ相対タイムスタンプが絶対秒に変換されること。"""
        windows = [(0.0, 25.0), (20.0, 45.0)]
        window_segments = [
            [_seg(5.0, 10.0, "A")],
            [_seg(10.0, 15.0, "B")],
        ]
        merged = _merge_window_results(windows, window_segments, overlap_seconds=5.0)
        # A: abs [5,10], mid=7.5, in [0, 22.5)
        # B: abs [30,35], mid=32.5, in [22.5, 45]
        assert merged[0].text == "A"
        assert merged[0].start == 5.0
        assert merged[0].end == 10.0
        assert merged[1].text == "B"
        assert merged[1].start == 30.0
        assert merged[1].end == 35.0

    def test_ids_reassigned_in_order(self) -> None:
        windows = [(0.0, 25.0), (20.0, 45.0)]
        window_segments = [
            [_seg(5.0, 10.0, "A", id=99)],
            [_seg(10.0, 15.0, "B", id=88)],
        ]
        merged = _merge_window_results(windows, window_segments, overlap_seconds=5.0)
        assert [m.id for m in merged] == [0, 1]


class TestSuffixPrefixOverlap:
    def test_simple_overlap(self) -> None:
        assert _suffix_prefix_overlap("ABCDEF", "DEFGHI") == 3

    def test_no_overlap(self) -> None:
        assert _suffix_prefix_overlap("ABCDEF", "XYZGHI") == 0

    def test_full_containment(self) -> None:
        # prev fully matches curr's prefix
        assert _suffix_prefix_overlap("ABCD", "ABCDEFGH") == 4

    def test_japanese(self) -> None:
        n = _suffix_prefix_overlap(
            "実現というものを推進するのでありますけれどもこちらは基本的に",
            "けれどもこちらは基本的に導入経費の補助",
        )
        # 'けれどもこちらは基本的に' = 12 chars
        assert n == 12

    def test_whitespace_normalized(self) -> None:
        """空白文字の有無は無視して overlap を検出する。"""
        # prev は空白あり、curr も別の位置に空白あり
        prev = "進めていただいておりまして 令和7年度の補正予算等々につきましても"
        curr = "まして令和7年度の補正予算等々 につきましても多くのご応募を"
        n = _suffix_prefix_overlap(prev, curr)
        # normalized で 'まして令和7年度の補正予算等々につきましても' (22 chars) が一致
        # curr 内では 22 個目の非空白文字までの位置 + 1 = 23 文字目を返す
        assert n == 23
        # 実際に trim した結果を検証 (期待される残り)
        trimmed = curr[n:]
        assert trimmed == "多くのご応募を"


class TestDedupeOverlapText:
    def test_no_overlap_kept_both(self) -> None:
        segs = [_ws(0.0, 10.0, "前"), _ws(15.0, 25.0, "後")]
        result = _dedupe_overlap_text(segs)
        assert [s.text for s in result] == ["前", "後"]

    def test_identical_overlap_dropped(self) -> None:
        segs = [_ws(0.0, 10.0, "同じ"), _ws(8.0, 12.0, "同じ")]
        result = _dedupe_overlap_text(segs)
        assert [s.text for s in result] == ["同じ"]

    def test_curr_substring_of_prev_dropped(self) -> None:
        segs = [_ws(0.0, 10.0, "前半と後半"), _ws(8.0, 12.0, "後半")]
        result = _dedupe_overlap_text(segs)
        assert [s.text for s in result] == ["前半と後半"]

    def test_prev_substring_of_curr_replaced(self) -> None:
        # 前者が後者に包含 → 後者で置換
        segs = [_ws(0.0, 10.0, "短い"), _ws(8.0, 15.0, "短いより長い")]
        result = _dedupe_overlap_text(segs)
        assert [s.text for s in result] == ["短いより長い"]

    def test_suffix_prefix_overlap_trimmed(self) -> None:
        # prev: '...こちらは基本的に', curr: 'けれどもこちらは基本的に導入...'
        # 重なり 12 chars 'けれどもこちらは基本的に' → curr の prefix を 12 trim
        segs = [
            _ws(0.0, 10.0, "実現というものを推進するのでありますけれどもこちらは基本的に"),
            _ws(8.0, 15.0, "けれどもこちらは基本的に導入経費の補助"),
        ]
        result = _dedupe_overlap_text(segs)
        assert len(result) == 2
        assert result[1].text == "導入経費の補助"

    def test_no_time_overlap_keeps_both(self) -> None:
        # 重複した文字列でも時間が離れていれば trim しない
        segs = [_ws(0.0, 10.0, "同じ"), _ws(20.0, 25.0, "同じ")]
        result = _dedupe_overlap_text(segs)
        assert [s.text for s in result] == ["同じ", "同じ"]


class TestTranscribeSegmentLong:
    """duration > WHISPER_MIN_WINDOWED_DURATION で windowed パスが使われることを確認する。"""

    def test_long_wav_uses_windowed_path(
        self,
        tmp_path: Path,
        sample_speaker: SpeakerInfo,
    ) -> None:
        """40 秒の WAV では _transcribe_windowed が呼ばれ複数の API 呼び出しが行われる。"""
        wav_path = tmp_path / "long.wav"
        _write_silent_wav(wav_path, duration_seconds=40.0)

        # ウィンドウ毎に異なるテキストを返すモック (呼び出し回数を検証)
        call_count = {"n": 0}

        def make_mock_response() -> MagicMock:
            call_count["n"] += 1
            mock = MagicMock()
            mock.text = f"window{call_count['n']}"
            mock.segments = [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 5.0,
                    "end": 10.0,
                    "text": f"text-w{call_count['n']}",
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": -0.2,
                    "compression_ratio": 1.5,
                    "no_speech_prob": 0.01,
                }
            ]
            return mock

        with patch("src.transcriber._get_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.audio.transcriptions.create.side_effect = lambda **_: make_mock_response()
            mock_client_factory.return_value = mock_client

            with patch.dict("os.environ", {"DEEPINFRA_API_KEY": "test-key"}):
                result = transcribe_segment(wav_path, 0, sample_speaker, "厚生労働委員会")

        # 40s の WAV は単一窓 (25s) を超えるので窓分割される。窓数は定数から計算
        # (overlap を調整しても壊れないよう、ハードコードしない)。
        from src.transcriber import _compute_windows

        expected_windows = len(_compute_windows(40.0))
        assert expected_windows >= 2  # 窓分割パスに入っている
        assert call_count["n"] == expected_windows
        # 各窓 1 segment を返すモック。trust window フィルタ後に最低 1 つは残る。
        assert len(result.whisper_segments) >= 1


@pytest.mark.integration
class TestTranscriberIntegration:
    def test_real_api_call(self, tmp_path: Path) -> None:
        """実際の Whisper API を呼び出すテスト（結合テスト、要 API キー）。"""
        import wave
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)  # 1秒の無音

        speaker = SpeakerInfo(
            name="テスト話者",
            affiliation="テスト党",
            start_seconds=0.0,
            start_time="00:00",
            duration_minutes=1,
        )

        result = transcribe_segment(wav_path, 0, speaker, [speaker])
        assert isinstance(result, SegmentTranscript)
