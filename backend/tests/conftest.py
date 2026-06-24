"""共通テストフィクスチャ"""

from __future__ import annotations

# .env を読み込む（integration tests 用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pathlib import Path

import pytest

from src.models import (
    QAPair,
    QAPairsOutput,
    RawTranscript,
    SegmentTranscript,
    SegmentUtterances,
    SessionDetail,
    SpeakerInfo,
    Utterance,
    UtterancesOutput,
    WhisperSegment,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_speakers() -> list[SpeakerInfo]:
    return [
        SpeakerInfo(
            name="藤原徹",
            affiliation="自由民主党",
            role="委員長",
            start_seconds=0.0,
            start_time="13:00",
            duration_minutes=5,
        ),
        SpeakerInfo(
            name="古川あおい",
            affiliation="チームみらい",
            role="質疑者",
            start_seconds=7320.2,
            start_time="14:42",
            duration_minutes=18,
        ),
        SpeakerInfo(
            name="山田花子",
            affiliation="立憲民主党",
            role="質疑者",
            start_seconds=8400.0,
            start_time="15:00",
            duration_minutes=20,
        ),
    ]


@pytest.fixture
def sample_session_detail(sample_speakers: list[SpeakerInfo]) -> SessionDetail:
    return SessionDetail(
        chamber="shugiin",
        session_id="56149",
        date="2026-04-09",
        committee="本会議",
        committee_id=1,
        hls_url="http://hlsvod.shugiintv.go.jp/vod/_definst_/amlst:2026/2026-0409-1300-00/playlist.m3u8",
        source_url="https://www.shugiintv.go.jp/jp/index.php?ex=VL&deli_id=56149",
        speakers=sample_speakers,
    )


@pytest.fixture
def sample_raw_transcript() -> RawTranscript:
    return RawTranscript(
        session_id="56149",
        segments=[
            SegmentTranscript(
                segment_index=0,
                speaker_name="藤原徹",
                start_seconds=0.0,
                text="これより会議を開きます。古川あおい君。",
                whisper_segments=[
                    WhisperSegment(
                        id=0,
                        seek=0,
                        start=0.0,
                        end=5.0,
                        text="これより会議を開きます。古川あおい君。",
                    )
                ],
            ),
            SegmentTranscript(
                segment_index=1,
                speaker_name="古川あおい",
                start_seconds=7320.2,
                text="チームみらいの古川あおいです。高額療養費制度について伺います。お答えいたします。問題を認識しております。",
                whisper_segments=[
                    WhisperSegment(
                        id=0,
                        seek=0,
                        start=7320.2,
                        end=7380.0,
                        text="チームみらいの古川あおいです。高額療養費制度について伺います。",
                    ),
                    WhisperSegment(
                        id=1,
                        seek=0,
                        start=7380.0,
                        end=7440.0,
                        text="お答えいたします。問題を認識しております。",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_utterances() -> UtterancesOutput:
    return UtterancesOutput(
        segments=[
            SegmentUtterances(
                segment_index=0,
                segment_speaker="藤原徹",
                segment_affiliation="自由民主党",
                start_seconds=0.0,
                video_url="https://www.shugiintv.go.jp/jp/index.php?ex=VL&media_type=&deli_id=56149&time=0.0",
                utterances=[
                    Utterance(speaker="藤原徹", role="委員長", text="これより会議を開きます。"),
                    Utterance(speaker="藤原徹", role="委員長", text="古川あおい君。"),
                ],
            ),
            SegmentUtterances(
                segment_index=1,
                segment_speaker="古川あおい",
                segment_affiliation="チームみらい",
                start_seconds=7320.2,
                video_url="https://www.shugiintv.go.jp/jp/index.php?ex=VL&media_type=&deli_id=56149&time=7320.2",
                utterances=[
                    Utterance(
                        speaker="古川あおい",
                        role="質疑者",
                        text="チームみらいの古川あおいです。高額療養費制度について伺います。",
                    ),
                    Utterance(
                        speaker="上野賢一郎",
                        role="答弁者",
                        text="お答えいたします。問題を認識しております。",
                    ),
                ],
            ),
        ]
    )


@pytest.fixture
def sample_qa_pairs() -> QAPairsOutput:
    return QAPairsOutput(
        pairs=[
            QAPair(
                id="qa_001",
                segment_index=1,
                topic="高額療養費の多数回該当リセット",
                question={  # type: ignore[arg-type]
                    "speaker": "古川あおい",
                    "party": "チームみらい",
                    "summary": "高額療養費制度の問題点について質問",
                    "full_text": "チームみらいの古川あおいです。高額療養費制度について伺います。",
                    "intent": "fact_check",
                },
                answer={  # type: ignore[arg-type]
                    "speaker": "上野賢一郎",
                    "role": "厚生労働大臣",
                    "summary": "問題を認識しており検討中と答弁",
                    "full_text": "お答えいたします。問題を認識しております。",
                    "evasion_score": 0.3,
                    "has_commitment": True,
                    "commitment_text": "次期制度改正の検討課題として位置づけてまいりたい",
                },
                video_url="https://www.shugiintv.go.jp/jp/index.php?ex=VL&media_type=&deli_id=56149&time=7320.2",
            )
        ]
    )
