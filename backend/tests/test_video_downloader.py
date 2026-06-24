"""video/downloader.py の単体テスト (純ロジックのみ、ネットワーク不要)。"""

from __future__ import annotations

import pytest

from src.models import SpeakerInfo
from src.video.downloader import (
    _choose_video_variant,
    _parse_master_playlist_with_resolution,
    _parse_media_playlist_with_durations,
    _select_segments_for_range,
    member_window,
)

MEDIA_PLAYLIST = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:9
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:8.0,
seg_0.ts
#EXTINF:8.0,
seg_1.ts
#EXTINF:8.0,
seg_2.ts
#EXTINF:8.0,
seg_3.ts
#EXTINF:4.0,
seg_4.ts
"""

SHUGIIN_MASTER = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:BANDWIDTH=564000,NAME="500k",RESOLUTION=640x360
chunklist_b564000.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=314000,NAME="250k",RESOLUTION=480x270
chunklist_b314000.m3u8
"""


def test_parse_master_with_resolution():
    variants = _parse_master_playlist_with_resolution(
        SHUGIIN_MASTER, "https://h.example/vod/playlist.m3u8"
    )
    assert len(variants) == 2
    by_h = {v.height: v for v in variants}
    assert by_h[360].width == 640
    assert by_h[360].bandwidth == 564000
    assert by_h[360].url == "https://h.example/vod/chunklist_b564000.m3u8"
    assert by_h[270].width == 480


def test_choose_variant_prefers_min_height_at_or_above_360():
    variants = _parse_master_playlist_with_resolution(SHUGIIN_MASTER, "https://h/p.m3u8")
    chosen = _choose_video_variant(variants, min_height=360)
    assert chosen.height == 360  # 640x360, not 480x270


def test_choose_variant_falls_back_to_highest_when_none_meet_min():
    variants = _parse_master_playlist_with_resolution(SHUGIIN_MASTER, "https://h/p.m3u8")
    # require 1080p — none qualify → highest available (360)
    chosen = _choose_video_variant(variants, min_height=1080)
    assert chosen.height == 360


def test_choose_variant_empty_raises():
    with pytest.raises(ValueError):
        _choose_video_variant([], min_height=360)


def _spk(name: str, start: float, dur_min: int = 5) -> SpeakerInfo:
    return SpeakerInfo(
        name=name, affiliation="", role="質疑者",
        start_seconds=start, start_time="", duration_minutes=dur_min,
    )


def test_member_window_uses_next_speaker_start():
    speakers = [_spk("A", 100.0), _spk("B", 250.0), _spk("C", 400.0)]
    assert member_window(speakers, 0) == (100.0, 250.0)
    assert member_window(speakers, 1) == (250.0, 400.0)


def test_member_window_last_speaker_uses_video_duration():
    speakers = [_spk("A", 100.0), _spk("B", 250.0)]
    assert member_window(speakers, 1, video_duration=900.0) == (250.0, 900.0)


def test_member_window_last_speaker_no_duration_uses_duration_minutes():
    speakers = [_spk("A", 100.0, dur_min=3)]
    # no next speaker, no video_duration → start + 3min
    assert member_window(speakers, 0) == (100.0, 100.0 + 180.0)


# --- partial download: playlist duration parse + range selection ---


def test_parse_media_playlist_with_durations():
    segs = _parse_media_playlist_with_durations(MEDIA_PLAYLIST, "https://h/x.m3u8")
    assert len(segs) == 5
    assert segs[0] == ("https://h/seg_0.ts", 8.0)
    assert segs[4] == ("https://h/seg_4.ts", 4.0)


def test_select_segments_for_range_mid_window():
    # segments: [0-8),[8-16),[16-24),[24-32),[32-36)
    segs = _parse_media_playlist_with_durations(MEDIA_PLAYLIST, "https://h/x.m3u8")
    # member [10, 20) overlaps seg_1 (8-16) and seg_2 (16-24)
    urls, offset = _select_segments_for_range(segs, 10.0, 20.0)
    assert urls == ["https://h/seg_1.ts", "https://h/seg_2.ts"]
    assert offset == 8.0  # first chosen segment starts at 8s


def test_select_segments_for_range_exact_boundary():
    segs = _parse_media_playlist_with_durations(MEDIA_PLAYLIST, "https://h/x.m3u8")
    # [16, 24) lands exactly on seg_2; should not pull seg_1 or seg_3
    urls, offset = _select_segments_for_range(segs, 16.0, 24.0)
    assert urls == ["https://h/seg_2.ts"]
    assert offset == 16.0


def test_select_segments_for_range_spans_many():
    segs = _parse_media_playlist_with_durations(MEDIA_PLAYLIST, "https://h/x.m3u8")
    urls, offset = _select_segments_for_range(segs, 5.0, 30.0)
    # overlaps seg_0..seg_3
    assert urls == [f"https://h/seg_{i}.ts" for i in range(4)]
    assert offset == 0.0


def test_select_segments_offset_lets_caller_compute_clip_start():
    # the offset is what clip_service uses: clip_start = member_start - offset
    segs = _parse_media_playlist_with_durations(MEDIA_PLAYLIST, "https://h/x.m3u8")
    member_start = 10.0
    urls, offset = _select_segments_for_range(segs, member_start, 20.0)
    clip_start = member_start - offset
    assert clip_start == 2.0  # member starts 2s into the downloaded segment range
