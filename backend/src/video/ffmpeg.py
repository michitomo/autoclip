"""ffmpeg バイナリ解決 (autoclip 固有)。

字幕焼き込みには libass 入りの ffmpeg が要る。Homebrew の標準 `ffmpeg` は
プラットフォームによって libass 無しでビルドされており、その場合は別途
`ffmpeg-full` (keg-only, libass 入り) を使う必要がある。

解決順:
1. 環境変数 AUTOCLIP_FFMPEG が指すパス (明示指定を最優先)
2. PATH 上の `ffmpeg` が subtitles フィルタを持てばそれ
3. 既知の keg-only ロケーション (ffmpeg-full) で subtitles フィルタを持つもの
4. どれも無ければ PATH の `ffmpeg` (字幕焼き込みは renderer 側でスキップされる)
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess

# libass 入り ffmpeg の既知 keg-only ロケーション (Homebrew)
_KNOWN_FULL_FFMPEG = [
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
]


def _has_subtitles_filter(ffmpeg_path: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return any(
        len(parts) >= 2 and parts[1] == "subtitles"
        for parts in (line.split() for line in out.stdout.splitlines())
    )


@functools.lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    """使用する ffmpeg バイナリのパスを返す (libass 入りを優先)。"""
    explicit = os.environ.get("AUTOCLIP_FFMPEG")
    if explicit:
        return explicit

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg and _has_subtitles_filter(path_ffmpeg):
        return path_ffmpeg

    for cand in _KNOWN_FULL_FFMPEG:
        if os.path.exists(cand) and _has_subtitles_filter(cand):
            return cand

    # フォールバック: PATH の ffmpeg (無ければ素の名前)。字幕は焼けない。
    return path_ffmpeg or "ffmpeg"


@functools.lru_cache(maxsize=1)
def ffprobe_bin() -> str:
    """使用する ffprobe バイナリのパスを返す。

    ffmpeg_bin() が keg-only の ffmpeg-full を選んだ場合は同ディレクトリの
    ffprobe を優先する (バージョン整合のため)。
    """
    explicit = os.environ.get("AUTOCLIP_FFPROBE")
    if explicit:
        return explicit

    ffmpeg = ffmpeg_bin()
    if os.path.sep in ffmpeg:
        sibling = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
        if os.path.exists(sibling):
            return sibling

    return shutil.which("ffprobe") or "ffprobe"


@functools.lru_cache(maxsize=1)
def has_libass() -> bool:
    """選択中の ffmpeg が subtitles フィルタ (libass) を持つか。"""
    return _has_subtitles_filter(ffmpeg_bin())
