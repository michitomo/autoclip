"""レンダラ (autoclip 固有): EDL + ASS → 単一パス ffmpeg 再エンコードでクリップ生成。

**必ず再エンコード**する。stream-copy / キーフレームカットは GOP 境界でしか切れず、
多数のマイクロカットで A/V がズレ黒フレームが出る。keep 区間ごとに `trim`/`atrim` で
切り出し `setpts`/`asetpts` で前詰めしてから `concat` で連結し、同一パスで
`subtitles=clip.ass` を焼き込む (filter_complex)。

旧実装は `select='between(t,a,b)+...'` を 1 式で渡していたが、keep_ranges が ~100
区間を超えると ffmpeg の式パーサが破綻し exit 244 で落ちる。trim+concat は区間数に
比例した素直なグラフになり数百区間でも安定する。

EDL の keep_ranges は **member-WAV 時間** (議員区間先頭=0)。入力にフル source.mp4 を
渡す場合は member_start を足して絶対時間に直す。ASS は **カット後** タイムラインなので
そのまま焼ける。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.models import EDL
from src.video.ffmpeg import ffmpeg_bin, has_libass

logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT_RENDER = 1800
DEFAULT_FPS = 30  # 衆議院TV は 29.97。固定 fps で setpts を安定させる

# アスペクト比 → 出力解像度 (縦長/正方/横)
ASPECT_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1280, 720),
}


def _crop_scale_filter(aspect: str, scale_factor: float = 1.0) -> str:
    """入力をセンタークロップして目標アスペクトに合わせ、出力解像度へ scale する。

    元 640x360 を 9:16 等にする。crop は中央基準 (議員が中央に映るため上半身が残る)。
    crop=ow:oh は入力寸法に対する比から算出 (min でフレーム内に収める)。
    scale_factor<1.0 で出力解像度を縮小 (プレビュー高速化)。ASS は PlayRes=フル解像度
    のままなので libass が自動でフレーム寸法に合わせて縮小描画する (見た目は一致)。
    偶数寸法 (yuv420p 要件) に丸める。
    """
    out_w, out_h = ASPECT_RESOLUTIONS[aspect]
    target = out_w / out_h
    sw = max(2, int(round(out_w * scale_factor)) & ~1)  # 偶数化
    sh = max(2, int(round(out_h * scale_factor)) & ~1)
    # 入力アスペクトに対し、目標アスペクトに合うよう幅 or 高さをクロップ。
    # crop=w='min(iw, ih*target)':h='min(ih, iw/target)':中央
    crop = (
        f"crop=w='min(iw,ih*{target:.6f})':h='min(ih,iw/{target:.6f})'"
        f":x='(iw-ow)/2':y='(ih-oh)/2'"
    )
    # フル解像度はlanczos (高品質)、縮小プレビューはbilinear (高速)。
    flags = "lanczos" if scale_factor >= 1.0 else "bilinear"
    scale = f"scale={sw}:{sh}:flags={flags}"
    return f"{crop},{scale}"


def has_subtitles_filter() -> bool:
    """選択中の ffmpeg が subtitles フィルタ (libass) を持つか (後方互換エイリアス)。"""
    return has_libass()


def _build_concat_filtergraph(
    edl: EDL, member_start: float, *, post_video: str, time_base_shift: float = 0.0
) -> str:
    """keep_ranges を trim/atrim+concat で連結する filter_complex を作る。

    旧実装は `select='between(t,a,b)+between(...)+...'` を 1 式で渡していたが、
    keep_ranges が ~100 個を超えると ffmpeg の式パーサが破綻し
    "Cannot allocate memory" (exit 244) で落ちる。trim+concat なら区間数に
    比例した素直なグラフになり、数百区間でも安定する。

    各区間 i について:
      [0:v]trim=start=A:end=B,setpts=PTS-STARTPTS[vi]
      [0:a]atrim=start=A:end=B,asetpts=PTS-STARTPTS[ai]
    を作り、[v0][a0][v1][a1]... concat=n=N:v=1:a=1[vc][ac] で前詰め連結。
    連結後の映像 [vc] に post_video (fps→crop/scale→字幕) を適用し [v] を出す。
    音声 [ac] はそのまま map する。

    trim 時刻 = keep_range (member-WAV) + member_start (= ソース絶対時間) - time_base_shift。
    time_base_shift は **入力シーク** (`-ss <shift>` を `-i` の前) を使う場合の原点補正。
    入力シーク後は ffmpeg のタイムラインがシーク位置=0 になるため、trim もその分戻す。

    Returns:
        filter_complex 文字列。呼び出し側は -map "[v]" -map "[ac]" する。
    """
    segs: list[str] = []
    labels: list[str] = []
    for i, r in enumerate(edl.keep_ranges):
        a = max(0.0, r.start + member_start - time_base_shift)
        b = max(0.0, r.end + member_start - time_base_shift)
        segs.append(
            f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]"
        )
        segs.append(
            f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )
        labels.append(f"[v{i}][a{i}]")
    n = len(edl.keep_ranges)
    concat = "".join(labels) + f"concat=n={n}:v=1:a=1[vc][ac]"
    post = f"[vc]{post_video}[v]"
    return ";".join(segs) + ";" + concat + ";" + post


def render_clip(
    source_mp4: Path,
    edl: EDL,
    output_mp4: Path,
    *,
    ass_path: Path | None = None,
    member_start: float = 0.0,
    fps: int = DEFAULT_FPS,
    crf: int = 20,
    preset: str = "veryfast",
    aspect: str = "9:16",
    scale_factor: float = 1.0,
    input_seek: float | None = None,
    input_end: float | None = None,
) -> Path:
    """source_mp4 から EDL の keep 区間だけを連結し、字幕を焼いて output_mp4 を出す。

    Args:
        source_mp4: 入力動画 (フル session か member 切り出し)
        edl: JetCut の編集判断 (keep_ranges は member-WAV 時間)
        output_mp4: 出力パス
        ass_path: 焼き込む ASS 字幕 (None なら字幕なし)
        member_start: source 内で議員区間が始まる絶対秒。member 切り出し済みなら 0。
        fps: 出力フレームレート
        crf/preset: x264 設定
        aspect: 出力アスペクト ("9:16"|"1:1"|"16:9")。既定 9:16 縦長センタークロップ。
        input_seek: 指定すると入力シーク (`-ss <input_seek>` を `-i` の前) で
            その位置までソースを飛ばしてからデコードする (トピックプレビュー高速化)。
            keep_ranges の trim 時刻もこの分戻す。ソース絶対秒で指定。
        input_end: 入力シーク併用時の終端 (`-to`)。ソース絶対秒。

    Returns:
        output_mp4
    """
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    if not edl.keep_ranges:
        raise ValueError("EDL has no keep_ranges; nothing to render")
    if aspect not in ASPECT_RESOLUTIONS:
        raise ValueError(f"Unsupported aspect {aspect!r} (use {list(ASPECT_RESOLUTIONS)})")

    # 連結後の映像に適用する後段チェーン: fps 固定 → クロップ/scale → 字幕焼き込み。
    # subtitles は最終解像度の後に置くので ASS PlayRes と一致する。
    post_video = f"fps={fps}," + _crop_scale_filter(aspect, scale_factor)
    burn_subs = ass_path is not None and has_subtitles_filter()
    if ass_path is not None and not burn_subs:
        logger.warning(
            "ffmpeg lacks the 'subtitles' filter (no libass) — rendering WITHOUT "
            "burned-in 字幕. Install an ffmpeg built with libass to burn subtitles. "
            "ASS/SRT sidecar files are still produced."
        )
    if burn_subs:
        # crop/scale 後 (= 最終 1080x1920 等) に焼くので ASS の PlayRes と一致させる。
        post_video += f",subtitles='{_escape_filter_path(ass_path)}'"

    # 入力シーク: トピックプレビューでは -ss/-to を -i の前に置き、そのトピック区間だけ
    # デコードする (高速)。キーフレーム手前から取れるよう少し戻して取り、trim で正確に刻む。
    input_args: list[str] = []
    shift = 0.0
    if input_seek is not None:
        shift = max(0.0, input_seek)
        input_args = ["-ss", f"{shift:.3f}"]
        if input_end is not None and input_end > shift:
            input_args += ["-to", f"{input_end:.3f}"]

    # keep_ranges を trim+concat で連結 (大量区間でも式パーサ破綻しない)。
    filtergraph = _build_concat_filtergraph(
        edl, member_start, post_video=post_video, time_base_shift=shift
    )

    # 映像コーデックは x264。HW エンコード (h264_videotoolbox) は実測で x264 veryfast と
    # ほぼ同速 (137s vs 145s) だった — このワークロードのボトルネックはエンコードでなく
    # デコード+crop/scale+字幕焼き込み(libass) のため。画質コントロールが素直な x264 に
    # 一本化する (preset / crf)。
    cmd = [
        ffmpeg_bin(),
        "-y",
        *input_args,
        "-i", str(source_mp4),
        "-filter_complex", filtergraph,
        "-map", "[v]",
        "-map", "[ac]",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_mp4),
    ]
    logger.info(
        "Rendering clip: %d keep-ranges (member_start=%.1fs, seek=%s, libx264 %s crf=%d) -> %s",
        len(edl.keep_ranges), member_start,
        f"{shift:.1f}s" if input_seek is not None else "none",
        preset, crf, output_mp4,
    )
    subprocess.run(
        cmd, check=True, capture_output=True, text=True,
        timeout=_FFMPEG_TIMEOUT_RENDER,
    )
    return output_mp4


def _escape_filter_path(path: Path) -> str:
    r"""ffmpeg filtergraph 内のパスを escape する。

    filtergraph では ':' と '\' と "'" が特別。subtitles= に渡す前に処理する。
    """
    s = str(path)
    s = s.replace("\\", "\\\\")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    return s
