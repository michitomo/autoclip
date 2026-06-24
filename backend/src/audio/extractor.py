"""HLS音声抽出とWAVセグメント分割

衆議院TVのHLSは音声単独トラックを持たず、500kbps映像入りTSが約1500本に分割
配信されている。ffmpegに丸投げするとセグメントを1本ずつシリアル取得するため
GitHub Actions（日本までRTT〜120ms）で20分超かかる。

最適化方針:
1. master playlistから**最低帯域幅variant**（典型的に250k=314kbps）を選択して
   映像分の転送量を削減
2. Pythonの requests.Session で**HTTP keep-alive**を効かせつつ8並列で
   セグメントをfetch
3. 連結したTSをffmpegに1度だけ通して音声抽出（PCM 16kHz mono）

非.m3u8 URL（参議院のmp4直URL等）や上記が失敗したケースでは、`-http_persistent 1`
付きの ffmpeg 直接ダウンロードへフォールバックする。
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from src.api_client import MAX_WORKERS_AUDIO, MAX_WORKERS_HLS
from src.models import SpeakerInfo

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SECONDS = 60
_SEGMENT_RETRIES = 3
_USER_AGENT = "Mozilla/5.0 (compatible; kokkai-transcriber/0.1)"

# ffmpeg/ffprobe サブプロセスの用途別タイムアウト (秒)。
# HLS 配信が途中で停滞してもプロセスがハングしないよう必ず timeout を渡す。
_FFMPEG_TIMEOUT_DOWNLOAD = 1800   # HLS 直接 DL (~30分の音声を想定)
_FFMPEG_TIMEOUT_EXTRACT = 600     # ローカル TS → WAV 抽出
_FFMPEG_TIMEOUT_SPLIT = 300       # 1 セグメント分割 (silenceremove 込み)
_FFMPEG_TIMEOUT_SILENCE = 120     # detect_leading_silence
_FFPROBE_TIMEOUT = 30             # メタ取得 (duration)


def download_full_audio(hls_url: str, output_path: Path) -> Path:
    """HLSストリームをWAVファイルとしてダウンロードする。

    .m3u8 URL の場合は最低帯域variantを選択して並列セグメント取得→ffmpegで
    音声のみ抽出する。失敗時や非HLS URL（mp4直リンク等）は ffmpeg に直接
    URL を渡す経路へフォールバックする。

    Args:
        hls_url: HLS プレイリスト URL もしくはストリーム URL
        output_path: 出力先 WAV ファイルパス

    Returns:
        出力ファイルパス

    Raises:
        subprocess.CalledProcessError: ffmpeg が失敗した場合
        requests.RequestException: ネットワーク取得が全リトライ後も失敗した場合
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if ".m3u8" in hls_url.lower():
        try:
            return _download_via_parallel_hls(hls_url, output_path)
        except (requests.RequestException, ValueError, RuntimeError) as e:
            logger.warning(
                "Parallel HLS download failed (%s); falling back to ffmpeg direct",
                e,
            )

    return _download_via_ffmpeg(hls_url, output_path)


def _download_via_ffmpeg(url: str, output_path: Path) -> Path:
    """ffmpeg に直接 URL を渡してダウンロードする（フォールバック経路）。

    `-http_persistent 1` で HTTP keep-alive を有効化し、TTFB を削減する。
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-http_persistent", "1",
        "-i", url,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]
    logger.info("Downloading audio (ffmpeg direct): %s -> %s", url, output_path)
    subprocess.run(
        cmd, check=True, capture_output=True, text=True,
        timeout=_FFMPEG_TIMEOUT_DOWNLOAD,
    )
    logger.info("Audio downloaded: %s", output_path)
    return output_path


def _download_via_parallel_hls(master_url: str, output_path: Path) -> Path:
    """master m3u8 → 最低帯域variant → セグメント並列DL → ffmpeg音声抽出。"""
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    media_url, media_text = _resolve_media_playlist(session, master_url)
    segment_urls = _parse_media_playlist(media_text, media_url)
    if not segment_urls:
        raise ValueError(f"No segments found in playlist: {media_url}")

    logger.info(
        "HLS segments: %d, parallel workers=%d",
        len(segment_urls), MAX_WORKERS_HLS,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        seg_dir = tmp_path / "segments"
        seg_dir.mkdir()
        concatenated_ts = tmp_path / "concatenated.ts"

        _fetch_segments_parallel(session, segment_urls, seg_dir)
        _concatenate_segments(seg_dir, len(segment_urls), concatenated_ts)
        _extract_audio_with_ffmpeg(concatenated_ts, output_path)

    logger.info("Audio downloaded: %s", output_path)
    return output_path


def _resolve_media_playlist(
    session: requests.Session, url: str,
) -> tuple[str, str]:
    """master playlist なら最低帯域variantを選び、media playlist を返す。

    既に media playlist の場合はそのまま返す。
    """
    text = _fetch_text(session, url)

    if "#EXT-X-STREAM-INF" not in text:
        return url, text

    variants = _parse_master_playlist(text, url)
    if not variants:
        raise ValueError(f"Master playlist contains no STREAM-INF entries: {url}")

    chosen_bw, chosen_url = min(variants, key=lambda v: v[0])
    logger.info(
        "Selected variant: bandwidth=%d bps (%d candidates), url=%s",
        chosen_bw, len(variants), chosen_url,
    )
    return chosen_url, _fetch_text(session, chosen_url)


def _parse_master_playlist(text: str, base_url: str) -> list[tuple[int, str]]:
    """master playlistから (BANDWIDTH, variant_url) を抽出する。"""
    variants: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#EXT-X-STREAM-INF"):
            m = re.search(r"BANDWIDTH=(\d+)", line)
            bandwidth = int(m.group(1)) if m else 0
            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or lines[j].strip().startswith("#")
            ):
                j += 1
            if j < len(lines):
                variant_url = urllib.parse.urljoin(base_url, lines[j].strip())
                variants.append((bandwidth, variant_url))
                i = j + 1
                continue
        i += 1
    return variants


def _parse_media_playlist(text: str, base_url: str) -> list[str]:
    """media playlistからセグメントURLリストを順序保持で抽出する。"""
    segments: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        segments.append(urllib.parse.urljoin(base_url, line))
    return segments


def _fetch_text(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.text


def _fetch_segments_parallel(
    session: requests.Session,
    segment_urls: list[str],
    output_dir: Path,
) -> None:
    """セグメントを並列取得する。完了順にログを出すが順序保持はファイル名で行う。"""

    def _fetch_one(idx: int, url: str) -> int:
        seg_path = output_dir / f"{idx:06d}.ts"
        last_err: Exception | None = None
        for attempt in range(_SEGMENT_RETRIES):
            try:
                resp = session.get(url, timeout=_HTTP_TIMEOUT_SECONDS, stream=True)
                resp.raise_for_status()
                with seg_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
                return idx
            except requests.RequestException as e:
                last_err = e
                if attempt < _SEGMENT_RETRIES - 1:
                    backoff = 2.0 * (attempt + 1)
                    logger.warning(
                        "Segment %d fetch failed (attempt %d/%d): %s; retrying in %.1fs",
                        idx, attempt + 1, _SEGMENT_RETRIES, e, backoff,
                    )
                    time.sleep(backoff)
        raise RuntimeError(f"Segment {idx} failed after {_SEGMENT_RETRIES} attempts: {last_err}")

    completed = 0
    total = len(segment_urls)
    log_step = max(total // 10, 1)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_HLS) as executor:
        futures = [
            executor.submit(_fetch_one, i, url)
            for i, url in enumerate(segment_urls)
        ]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % log_step == 0 or completed == total:
                logger.info("HLS segment progress: %d/%d", completed, total)


def _concatenate_segments(seg_dir: Path, count: int, output_ts: Path) -> None:
    """連番TSをバイト連結する（MPEG-TSは単純連結で結合可能）。"""
    with output_ts.open("wb") as out:
        for i in range(count):
            seg_path = seg_dir / f"{i:06d}.ts"
            with seg_path.open("rb") as f:
                shutil.copyfileobj(f, out)
            seg_path.unlink()
    size_mb = output_ts.stat().st_size / 1024 / 1024
    logger.info("Concatenated %d TS segments: %s (%.1f MB)", count, output_ts, size_mb)


def _extract_audio_with_ffmpeg(input_ts: Path, output_wav: Path) -> None:
    """ローカルTSから音声をWAV(16kHz mono)で抽出する。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_ts),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav),
    ]
    logger.info("Extracting audio from local TS: %s -> %s", input_ts, output_wav)
    subprocess.run(
        cmd, check=True, capture_output=True, text=True,
        timeout=_FFMPEG_TIMEOUT_EXTRACT,
    )


def split_segments(
    full_audio: Path,
    speakers: list[SpeakerInfo],
    output_dir: Path,
    max_workers: int = MAX_WORKERS_AUDIO,
) -> list[Path]:
    """発言者タイムスタンプに基づいて WAV を並列分割する。

    Args:
        full_audio: 分割元の WAV ファイルパス
        speakers: 発言者リスト（start_seconds 昇順であること）
        output_dir: 出力先ディレクトリ
        max_workers: ffmpeg 並列数（デフォルト: MAX_WORKERS_AUDIO）

    Returns:
        生成されたセグメントファイルのパスリスト（speakers と同順）

    Raises:
        subprocess.CalledProcessError: ffmpeg が失敗した場合
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_duration = _get_audio_duration(full_audio)

    def _split_one(i: int, speaker: SpeakerInfo) -> tuple[int, Path]:
        start = speaker.start_seconds
        end = speakers[i + 1].start_seconds if i + 1 < len(speakers) else total_duration

        safe_name = speaker.name.replace("/", "_").replace(" ", "_")
        output_path = output_dir / f"{i:03d}_{safe_name}.wav"

        # -ss/-to は -i の「前」に置く（input options として絶対秒で解釈）。
        # -i の後ろに置くと output options 扱いになり、-af silenceremove と
        # 組み合わさったとき入力全体に無音除去を適用してから圧縮後タイムライン上で
        # 切り出すという挙動になり、各セグメントに別時刻の音声が入る。
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(full_audio),
            "-af", (
                "silenceremove="
                "start_periods=0:"         # 先頭無音は残す（発言開始を保持）
                "stop_periods=-1:"         # 途中・末尾の無音を全て対象
                "stop_duration=2.0:"       # 2秒以上続く無音を除去
                "stop_threshold=-45dB"     # -45dB以下を無音と判定
            ),
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(output_path),
        ]

        logger.info(
            "Splitting segment %d/%d: %s (%.1fs - %.1fs)",
            i + 1, len(speakers), speaker.name, start, end,
        )
        subprocess.run(
            cmd, check=True, capture_output=True, text=True,
            timeout=_FFMPEG_TIMEOUT_SPLIT,
        )

        if output_path.exists():
            original_size = (end - start) * 16000 * 2
            actual_size = output_path.stat().st_size
            reduction_pct = (1 - actual_size / original_size) * 100 if original_size > 0 else 0
            logger.info(
                "Segment %d silence-removed: %.1fMB (%.0f%% reduction)",
                i, actual_size / 1024 / 1024, reduction_pct,
            )

        return i, output_path

    results: list[tuple[int, Path]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_split_one, i, speaker): i
            for i, speaker in enumerate(speakers)
        }
        for future in as_completed(futures):
            results.append(future.result())

    # speakers と同順にソート
    results.sort(key=lambda x: x[0])
    segment_paths = [path for _, path in results]

    logger.info("Split %d segments into %s", len(segment_paths), output_dir)
    return segment_paths


def _get_audio_duration(wav_path: Path) -> float:
    """ffprobe で WAV ファイルの長さ（秒）を取得する。"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True,
        timeout=_FFPROBE_TIMEOUT,
    )
    return float(result.stdout.strip())


def detect_leading_silence(wav_path: Path, threshold_db: float = -60.0) -> float:
    """WAVファイルの先頭無音の長さ（秒）を検出する。

    ffmpegのsilencedetectフィルタを使い、先頭から最初に音声が始まるまでの
    無音区間の長さを返す。無音がなければ0.0を返す。

    Args:
        wav_path: 対象WAVファイル
        threshold_db: 無音と判定する閾値（dB）。デフォルト-60dB。

    Returns:
        先頭無音の長さ（秒）
    """
    cmd = [
        "ffmpeg",
        "-i", str(wav_path),
        "-af", f"silencedetect=noise={threshold_db}dB:d=1.0",
        "-f", "null", "-",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=_FFMPEG_TIMEOUT_SILENCE,
    )
    # silencedetect は stderr に出力する
    # "silence_end: 470.123 | silence_duration: 470.123" の形式
    for line in result.stderr.split("\n"):
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m:
            silence_end = float(m.group(1))
            logger.info("Detected leading silence: %.1fs", silence_end)
            return silence_end

    logger.info("No leading silence detected")
    return 0.0
