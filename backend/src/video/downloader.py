"""映像付き HLS ダウンロードと議員区間の音声抽出 (autoclip 固有)。

kokkaidb の `audio/extractor.download_full_audio` は最低帯域 variant を選び `-vn` で
音声のみ取り出す。クリップ生成では映像が必要なため、本モジュールは:

1. master playlist から **使える画質の variant** (既定 640x360) を RESOLUTION 込みで選ぶ
2. セグメントを並列取得 (extractor の並列フェッチ/連結を再利用) して TS を連結
3. TS → `source.mp4` に remux (`-c copy`, 再エンコードなし)
4. `extract_member_wav` で議員区間 [start,end) を **silenceremove なし** の 16kHz mono WAV に
   切り出す → Whisper の語タイムスタンプが動画時間と線形対応する (JetCut の前提)

非 .m3u8 URL や variant 解決失敗時は ffmpeg 直接ダウンロードにフォールバックする。
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import urllib.parse
from pathlib import Path

import requests

from src.audio.extractor import (
    _USER_AGENT,
    _concatenate_segments,
    _fetch_segments_parallel,
    _fetch_text,
    _parse_media_playlist,
)
from src.video.ffmpeg import ffmpeg_bin

logger = logging.getLogger(__name__)

# 既定で狙う最小の縦解像度。衆議院TV は 640x360 / 480x270 の 2 variant。
# 360p を既定にし、640x360 が無ければ最高解像度にフォールバックする。
DEFAULT_MIN_HEIGHT = 360

_FFMPEG_TIMEOUT_REMUX = 1800       # TS → mp4 remux
_FFMPEG_TIMEOUT_DOWNLOAD = 1800    # ffmpeg 直接 DL フォールバック
_FFMPEG_TIMEOUT_MEMBER = 600       # 議員区間の WAV 切り出し


class VideoVariant:
    """master playlist の 1 variant。"""

    __slots__ = ("bandwidth", "width", "height", "url")

    def __init__(self, bandwidth: int, width: int, height: int, url: str) -> None:
        self.bandwidth = bandwidth
        self.width = width
        self.height = height
        self.url = url

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"VideoVariant(bw={self.bandwidth}, {self.width}x{self.height})"


def _parse_master_playlist_with_resolution(
    text: str, base_url: str
) -> list[VideoVariant]:
    """master playlist から (BANDWIDTH, RESOLUTION, url) を抽出する。

    extractor._parse_master_playlist は BANDWIDTH のみ。映像 variant 選択には
    RESOLUTION が要るため拡張版を別に持つ (共有コードの契約は変えない)。
    """
    variants: list[VideoVariant] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#EXT-X-STREAM-INF"):
            bw_m = re.search(r"BANDWIDTH=(\d+)", line)
            res_m = re.search(r"RESOLUTION=(\d+)x(\d+)", line)
            bandwidth = int(bw_m.group(1)) if bw_m else 0
            width = int(res_m.group(1)) if res_m else 0
            height = int(res_m.group(2)) if res_m else 0
            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or lines[j].strip().startswith("#")
            ):
                j += 1
            if j < len(lines):
                url = urllib.parse.urljoin(base_url, lines[j].strip())
                variants.append(VideoVariant(bandwidth, width, height, url))
                i = j + 1
                continue
        i += 1
    return variants


def _choose_video_variant(
    variants: list[VideoVariant], min_height: int = DEFAULT_MIN_HEIGHT
) -> VideoVariant:
    """min_height 以上で最小の縦解像度の variant を選ぶ。

    なければ最高解像度 (最大 BANDWIDTH) にフォールバックする。映像が要るので
    extractor のように最低帯域を選んではならない。
    """
    if not variants:
        raise ValueError("No variants to choose from")

    eligible = [v for v in variants if v.height >= min_height]
    if eligible:
        chosen = min(eligible, key=lambda v: (v.height, v.bandwidth))
    else:
        chosen = max(variants, key=lambda v: (v.height, v.bandwidth))
    logger.info(
        "Selected video variant: %dx%d (bw=%d) from %d candidates",
        chosen.width, chosen.height, chosen.bandwidth, len(variants),
    )
    return chosen


def _parse_media_playlist_with_durations(
    text: str, base_url: str
) -> list[tuple[str, float]]:
    """media playlist から (segment_url, duration) を順序保持で返す。

    #EXTINF:<dur>, の直後の行がそのセグメント URL。
    """
    segments: list[tuple[str, float]] = []
    lines = text.splitlines()
    pending_dur: float | None = None
    for raw in lines:
        line = raw.strip()
        if line.startswith("#EXTINF:"):
            m = re.search(r"#EXTINF:([\d.]+)", line)
            pending_dur = float(m.group(1)) if m else 0.0
            continue
        if not line or line.startswith("#"):
            continue
        url = urllib.parse.urljoin(base_url, line)
        segments.append((url, pending_dur if pending_dur is not None else 0.0))
        pending_dur = None
    return segments


def _select_segments_for_range(
    segments: list[tuple[str, float]], start: float, end: float
) -> tuple[list[str], float]:
    """[start, end) に重なるセグメント URL 群と、先頭セグメント開始時刻を返す。

    累積時間でセグメント境界を求め、区間に少しでも重なるものを全て含める。
    返す offset (= 先頭採用セグメントの開始絶対秒) は、切り出し後 mp4 内で
    member の start に対応する位置を求めるのに使う (start - offset)。

    Returns:
        (segment_urls, first_segment_start_seconds)
    """
    chosen: list[str] = []
    first_start = 0.0
    t = 0.0
    for url, dur in segments:
        seg_start = t
        seg_end = t + dur
        t = seg_end
        # 区間と重なる: seg_start < end かつ seg_end > start
        if seg_start < end and seg_end > start:
            if not chosen:
                first_start = seg_start
            chosen.append(url)
        elif chosen and seg_start >= end:
            break
    return chosen, first_start


def download_segment_range(
    hls_url: str,
    start: float,
    end: float,
    output_path: Path,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> tuple[Path, float]:
    """HLS から [start, end) に重なるセグメントだけを取得し mp4 に remux する。

    フルセッション (数時間) を落とさず、議員区間ぶんのセグメントのみ取得する。
    区間先頭はセグメント途中になりうるため、返す segment_offset (= 出力 mp4 の
    先頭に対応する絶対秒) を使って呼び出し側が start - segment_offset で
    クリップ内オフセットを求める。

    Args:
        hls_url: master/media playlist URL
        start: 議員区間の開始絶対秒
        end: 終了絶対秒
        output_path: 出力 mp4
        min_height: variant の最小縦解像度

    Returns:
        (output_path, segment_offset)  segment_offset = 出力先頭の絶対秒
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    master_text = _fetch_text(session, hls_url)
    if "#EXT-X-STREAM-INF" in master_text:
        variants = _parse_master_playlist_with_resolution(master_text, hls_url)
        if not variants:
            raise ValueError(f"No STREAM-INF entries in master: {hls_url}")
        chosen = _choose_video_variant(variants, min_height)
        media_url = chosen.url
        media_text = _fetch_text(session, media_url)
    else:
        media_url = hls_url
        media_text = master_text

    seg_durations = _parse_media_playlist_with_durations(media_text, media_url)
    if not seg_durations:
        raise ValueError(f"No segments in playlist: {media_url}")

    segment_urls, segment_offset = _select_segments_for_range(seg_durations, start, end)
    if not segment_urls:
        raise ValueError(
            f"No segments overlap [{start:.1f},{end:.1f}) "
            f"(playlist covers {sum(d for _, d in seg_durations):.0f}s)"
        )

    logger.info(
        "Partial download: %d/%d segments for [%.1f,%.1f) (offset=%.1fs)",
        len(segment_urls), len(seg_durations), start, end, segment_offset,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        seg_dir = tmp_path / "segments"
        seg_dir.mkdir()
        concatenated_ts = tmp_path / "concatenated.ts"

        _fetch_segments_parallel(session, segment_urls, seg_dir)
        _concatenate_segments(seg_dir, len(segment_urls), concatenated_ts)
        _remux_ts_to_mp4(concatenated_ts, output_path)

    return output_path, segment_offset


def download_source_video(
    hls_url: str, output_path: Path, min_height: int = DEFAULT_MIN_HEIGHT
) -> Path:
    """HLS から映像付き source.mp4 を取得する (フル, 互換用)。

    .m3u8 の場合は variant を選んでセグメント並列取得 → TS 連結 → remux。
    失敗 / 非 HLS は ffmpeg 直接ダウンロードにフォールバックする。

    Args:
        hls_url: master/media playlist URL もしくは直接ストリーム URL
        output_path: 出力 mp4 パス
        min_height: 狙う最小縦解像度 (既定 360)

    Returns:
        output_path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if ".m3u8" in hls_url.lower():
        try:
            return _download_via_parallel_hls(hls_url, output_path, min_height)
        except (requests.RequestException, ValueError, RuntimeError) as e:
            logger.warning(
                "Parallel HLS video download failed (%s); falling back to ffmpeg direct",
                e,
            )
    return _download_via_ffmpeg_direct(hls_url, output_path)


def _download_via_parallel_hls(
    master_url: str, output_path: Path, min_height: int
) -> Path:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    master_text = _fetch_text(session, master_url)

    if "#EXT-X-STREAM-INF" in master_text:
        variants = _parse_master_playlist_with_resolution(master_text, master_url)
        if not variants:
            raise ValueError(f"No STREAM-INF entries in master: {master_url}")
        chosen = _choose_video_variant(variants, min_height)
        media_url = chosen.url
        media_text = _fetch_text(session, media_url)
    else:
        # 既に media playlist
        media_url = master_url
        media_text = master_text

    segment_urls = _parse_media_playlist(media_text, media_url)
    if not segment_urls:
        raise ValueError(f"No segments found in playlist: {media_url}")

    logger.info("HLS video segments: %d", len(segment_urls))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        seg_dir = tmp_path / "segments"
        seg_dir.mkdir()
        concatenated_ts = tmp_path / "concatenated.ts"

        _fetch_segments_parallel(session, segment_urls, seg_dir)
        _concatenate_segments(seg_dir, len(segment_urls), concatenated_ts)
        _remux_ts_to_mp4(concatenated_ts, output_path)

    logger.info("Source video downloaded: %s", output_path)
    return output_path


def _remux_ts_to_mp4(input_ts: Path, output_mp4: Path) -> None:
    """連結 TS を再エンコードせず mp4 にコンテナ変換する (-c copy)。"""
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i", str(input_ts),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_mp4),
    ]
    logger.info("Remuxing TS -> mp4: %s -> %s", input_ts, output_mp4)
    subprocess.run(
        cmd, check=True, capture_output=True, text=True,
        timeout=_FFMPEG_TIMEOUT_REMUX,
    )


def _download_via_ffmpeg_direct(url: str, output_path: Path) -> Path:
    """ffmpeg に直接 URL を渡して mp4 を得る (フォールバック)。"""
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-http_persistent", "1",
        "-i", url,
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    logger.info("Downloading video (ffmpeg direct): %s -> %s", url, output_path)
    subprocess.run(
        cmd, check=True, capture_output=True, text=True,
        timeout=_FFMPEG_TIMEOUT_DOWNLOAD,
    )
    return output_path


def extract_member_wav(
    source_mp4: Path, start: float, end: float, output_wav: Path
) -> Path:
    """source.mp4 の [start, end) を 16kHz mono WAV に切り出す (無音除去なし)。

    **silenceremove を一切かけない** のが要点。kokkaidb の split_segments は
    各話者 WAV 内の無音を除去するため transcript 時間が動画時間と非線形になる。
    JetCut は語タイムスタンプを動画時間へ線形写像する必要があるため、ここでは
    無音をそのまま残す。

    `-ss`/`-to` は `-i` の前に置き入力オプションとして絶対秒で解釈させる。

    Args:
        source_mp4: ダウンロード済みソース動画
        start: 開始秒 (動画タイムライン)
        end: 終了秒
        output_wav: 出力 WAV パス

    Returns:
        output_wav
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(source_mp4),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav),
    ]
    logger.info(
        "Extracting member WAV (no silenceremove): %.1fs-%.1fs -> %s",
        start, end, output_wav,
    )
    subprocess.run(
        cmd, check=True, capture_output=True, text=True,
        timeout=_FFMPEG_TIMEOUT_MEMBER,
    )
    return output_wav


def member_window(
    speakers: list, member_index: int, video_duration: float | None = None
) -> tuple[float, float]:
    """speakers[member_index] の発言区間 [start, end) を返す。

    end は次の話者の start_seconds。最後の話者なら video_duration (なければ
    start + その話者の duration_minutes 分)。

    Args:
        speakers: SpeakerInfo のリスト (start_seconds 昇順、補正済み想定)
        member_index: 対象話者の index
        video_duration: 動画全体の長さ (秒)。最後の話者の end に使う。

    Returns:
        (start, end) 秒
    """
    spk = speakers[member_index]
    start = float(spk.start_seconds)
    if member_index + 1 < len(speakers):
        end = float(speakers[member_index + 1].start_seconds)
    elif video_duration is not None:
        end = float(video_duration)
    else:
        end = start + float(getattr(spk, "duration_minutes", 0)) * 60.0
    return start, end
