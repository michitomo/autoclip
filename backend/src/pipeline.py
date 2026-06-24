"""パイプライン統合エントリポイント

使用方法:
    # 衆議院（セッションID直接指定）
    python -m src.pipeline --chamber shugiin --session-id 56149

    # 参議院
    python -m src.pipeline --chamber sangiin --session-id 7890

    # 後方互換（衆議院のみ）
    python -m src.pipeline --chamber shugiin --deli-id 56149

    # 日付指定の自動巡回
    python -m src.pipeline --chamber shugiin --date 2026-04-09

    # git pushなし（ローカルテスト用）
    python -m src.pipeline --chamber shugiin --session-id 56149 --no-push
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import date as date_type
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.api_client import (
    MAX_WORKERS_LLM,
    MAX_WORKERS_WHISPER,
    ensure_fd_limit,
)
from src.audio.extractor import detect_leading_silence, download_full_audio, split_segments
from src.committee_to_ministry import filter_laws_for_committee
from src.metadata_enricher import enrich_metadata_from_utterances
from src.models import (
    QAPairsOutput,
    RawTranscript,
    SessionDetail,
    SpeakerInfo,
    SummaryOutput,
    TopicsOutput,
    UtterancesOutput,
)
from src.normalizer import normalize_utterances
from src.scrapers.base import BaseScraper, SessionNotReadyError
from src.scrapers.shugiin import ShugiinScraper
from src.speaker_tagger import tag_all_segments
from src.structurer import (
    build_summary_related_laws,
    generate_key_commitments,
    generate_qa_pairs,
    generate_session_summary,
    generate_topics_and_key_topics,
    generate_topics_without_qa,
    score_qa_pairs_metrics,
    tag_related_laws,
)
from src.transcriber import transcribe_all_segments
from src.transcript_corrector import correct_transcript


class EmptySessionError(Exception):
    """音声コンテンツが空またはほぼ無音で、処理をスキップすべきセッション。"""


# パイプライン起動時にfd上限を引き上げ
ensure_fd_limit()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# データ出力先（kokkai-transcriber/ の1つ上の data/）
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_laws_compact() -> str:
    """LLMプロンプト用のコンパクトな法案一覧テキストを読み込む。

    `data/laws/laws_compact.txt`（`src.laws_builder` で事前生成）を読んでそのまま返す。
    見つからない場合は空文字列を返す（法案タグ付けをスキップ）。
    """
    laws_path = DATA_DIR / "laws" / "laws_compact.txt"

    if not laws_path.exists():
        logger.info("laws_compact.txt not found at %s, skipping law tagging", laws_path)
        return ""

    content = laws_path.read_text(encoding="utf-8")
    line_count = sum(1 for line in content.splitlines() if line.strip())
    if line_count:
        logger.info("Loaded %d laws for LLM tagging", line_count)
    return content


def _output_dir_for(chamber: str, date: str, session_id: str, committee: str) -> Path:
    """セッションの出力ディレクトリパスを返す。"""
    year, month, day = date.split("-")
    return DATA_DIR / chamber / year / month / day / f"{session_id}_{committee}"


def correct_speaker_offsets(
    speakers: list[SpeakerInfo], full_wav: Path
) -> float:
    """HLS先頭の無音パディング分だけ speakers[].start_seconds をその場で補正する。

    衆議院TV の time= パラメータは映像トラック基準のため、音声トラック先頭に
    無音パディングがあると start_seconds が実音声/映像位置とズレる。最初の
    speaker の start_seconds と実際の音声開始位置 (leading silence) の差を
    オフセットとして全 speaker から減算する。

    audio パイプライン (split_segments) と video パイプライン
    (extract_member_wav) の両方が同じ補正を適用できるよう関数化している。

    Args:
        speakers: 補正対象の発言者リスト (in-place で start_seconds を更新)
        full_wav: ダウンロード済みフル音声 WAV

    Returns:
        適用したオフセット秒数 (補正なしの場合 0.0)
    """
    if not speakers:
        return 0.0

    leading_silence = detect_leading_silence(full_wav)
    if leading_silence <= 10.0:
        return 0.0

    first_speaker_time = speakers[0].start_seconds
    offset = first_speaker_time - leading_silence
    # 閾値 5.0s: 小さなズレは補正せず誤補正リスクが低い 5s 超のみ積極補正する
    # (kokkaidb docs/STRUCTURER_REWRITE.md §2.13)
    if not (5.0 < offset < leading_silence):
        return 0.0

    logger.info(
        "Applying audio offset correction: %.1fs "
        "(leading_silence=%.1fs, first_speaker=%.1fs)",
        offset, leading_silence, first_speaker_time,
    )
    for speaker in speakers:
        speaker.start_seconds -= offset
    return offset


def run_pipeline(
    chamber: str,
    session_id: str,
    output_dir: Path,
    no_push: bool = False,
) -> None:
    """1セッションのパイプラインを実行する。

    Args:
        chamber: "shugiin" | "sangiin"
        session_id: セッションID
        output_dir: JSON出力先ディレクトリ
        no_push: Trueの場合git pushをスキップ

    Raises:
        SessionNotReadyError: 発言者リスト未公開（呼び出し側で一時エラー扱い）
        RuntimeError: いずれかのステップが失敗した場合
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: スクレイピング
    logger.info("=== Step 2: Scraping session detail (%s %s) ===", chamber, session_id)
    try:
        scraper = _get_scraper(chamber)
        session_detail = scraper.get_session_detail(session_id)
    except SessionNotReadyError:
        # 一時的な失敗はそのまま再raise（呼び出し側でスキップ）
        raise
    except Exception as e:
        raise RuntimeError(f"Step 2 (scraping) failed: {e}") from e

    # バリデーション: 発言者0件はここに到達しないはず（SessionNotReadyErrorで弾かれる）が
    # 念のため boundary check。
    if not session_detail.speakers:
        raise SessionNotReadyError(
            f"Speaker list unexpectedly empty for {chamber} {session_id}"
        )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        session_detail.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved metadata.json (%d speakers)", len(session_detail.speakers))

    # Step 3: HLS音声取得・分割
    logger.info("=== Step 3: Downloading and splitting audio ===")
    try:
        audio_url = session_detail.hls_url

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            full_wav = tmp_path / "full_audio.wav"

            download_full_audio(audio_url, full_wav)

            # 音声DL直後にHTMLを再スクレイプして最新の start_seconds を取得する。
            # 衆議院TV/参議院TVはアーカイブ整備バッチ（会議後数時間）でHLSに冒頭映像を
            # 追加し、その後HTMLのtime=値を更新する。初回スクレイプとHLSダウンロードの
            # 間にこの更新が挟まると start_seconds がHLS内容とズレる。ここで再スクレイプ
            # して、HLSと同じタイミングのtime=値を使う。
            try:
                fresh_detail = scraper.get_session_detail(session_id)
                if fresh_detail.speakers and session_detail.speakers:
                    old_first = session_detail.speakers[0].start_seconds
                    new_first = fresh_detail.speakers[0].start_seconds
                    drift = new_first - old_first
                    if abs(drift) > 30.0:
                        logger.info(
                            "HTML timestamps drifted %.1fs since initial scrape "
                            "(old=%.1f, new=%.1f) — updating start_seconds.",
                            drift, old_first, new_first,
                        )
                        fresh_map = {s.name: s.start_seconds for s in fresh_detail.speakers}
                        for sp in session_detail.speakers:
                            if sp.name in fresh_map:
                                sp.start_seconds = fresh_map[sp.name]
                        metadata_path.write_text(
                            session_detail.model_dump_json(indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
            except Exception as _rescrape_err:
                logger.warning(
                    "Re-scrape for fresh timestamps failed (using original): %s",
                    _rescrape_err,
                )

            # HLS先頭の無音パディング分だけ start_seconds を補正する。
            offset = correct_speaker_offsets(session_detail.speakers, full_wav)
            if offset:
                # metadata.jsonも補正済みタイムスタンプで上書き
                metadata_path.write_text(
                    session_detail.model_dump_json(indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            segment_paths = split_segments(
                full_wav,
                session_detail.speakers,
                output_dir / "segments",
            )
    except Exception as e:
        raise RuntimeError(f"Step 3 (audio extraction) failed: {e}") from e

    logger.info("Created %d audio segments", len(segment_paths))

    # Step 4: Whisper 文字起こし
    logger.info("=== Step 4: Transcribing with Whisper ===")
    try:
        raw_transcript = transcribe_all_segments(
            segment_paths,
            session_detail.speakers,
            session_id,
            committee=session_detail.committee,
            max_workers=MAX_WORKERS_WHISPER,
        )
    except Exception as e:
        raise RuntimeError(f"Step 4 (transcription) failed: {e}") from e

    # Step 4.5: LLM 文字起こし修正（句読点補完・固有名詞修正）
    logger.info("=== Step 4.5: Correcting transcript with LLM ===")
    try:
        raw_transcript = correct_transcript(raw_transcript, session_detail, max_workers=MAX_WORKERS_LLM)
    except Exception as e:
        logger.warning("Transcript correction failed (non-fatal, using original): %s", e)

    transcript_path = output_dir / "raw_transcript.json"
    transcript_path.write_text(
        raw_transcript.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Saved raw_transcript.json (%d segments, corrected=%s)",
        len(raw_transcript.segments),
        raw_transcript.corrected,
    )

    # バリデーション: 文字起こし結果が極端に少ないセッションは空セッションとしてスキップ
    total_chars = sum(len(s.text) for s in raw_transcript.segments)
    if total_chars < 100:
        raise EmptySessionError(
            f"Transcript too short ({total_chars} chars); "
            f"audio may be silent (休会・投票待ちなど)"
        )

    # Step 5: LLM 話者タグ付け
    logger.info("=== Step 5: Tagging speakers with LLM ===")
    try:
        utterances_output = tag_all_segments(raw_transcript, session_detail, max_workers=MAX_WORKERS_LLM)
    except Exception as e:
        raise RuntimeError(f"Step 5 (speaker tagging) failed: {e}") from e

    # Step 5.25: utterances 由来で metadata.speakers を逆補完 (PR6, §2.2/2.3)
    # PR26/PR29/PR30: 既存 speakers の role 再計算も含む。count 不変でも書き出す。
    logger.info("=== Step 5.25: Enriching metadata.speakers from utterances ===")
    pre_count = len(session_detail.speakers)
    pre_signature = tuple(
        (s.name, s.role, s.affiliation) for s in session_detail.speakers
    )
    enriched_speakers = enrich_metadata_from_utterances(
        utterances_output, session_detail.speakers
    )
    post_signature = tuple(
        (s.name, s.role, s.affiliation) for s in enriched_speakers
    )
    if len(enriched_speakers) != pre_count or pre_signature != post_signature:
        added = len(enriched_speakers) - pre_count
        logger.info(
            "Step 5.25: %s",
            f"appended {added} entries"
            if added
            else "backfilled role/affiliation on existing entries",
        )
        session_detail.speakers = enriched_speakers
        metadata_path.write_text(
            session_detail.model_dump_json(indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Step 5.5: speaker / role を metadata.speakers に正規化
    logger.info("=== Step 5.5: Normalizing utterance speakers and roles ===")
    utterances_output = normalize_utterances(utterances_output, session_detail.speakers)

    utterances_path = output_dir / "utterances.json"
    utterances_path.write_text(
        utterances_output.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Saved utterances.json (%d segments)", len(utterances_output.segments)
    )

    # Step 6: Q&Aペア生成 → 要約 → トピック → コミットメント → 関連法案タグ
    logger.info("=== Step 6: Generating Q&A pairs, summary, topics, and law tags ===")
    try:
        qa_pairs, summary, topics = _run_step6(
            session_detail, utterances_output, max_workers=MAX_WORKERS_LLM
        )
    except Exception as e:
        raise RuntimeError(f"Step 6 (structuring) failed: {e}") from e

    (output_dir / "qa_pairs.json").write_text(
        qa_pairs.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "topics.json").write_text(
        topics.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Saved qa_pairs.json (%d pairs), summary.json, topics.json (%d topics)",
        len(qa_pairs.pairs),
        len(topics.topics),
    )

    logger.info("=== Pipeline complete. Output: %s ===", output_dir)

    output_files = list(output_dir.glob("*.json"))
    for f in sorted(output_files):
        size = f.stat().st_size
        logger.info("  %s (%d bytes)", f.name, size)

    # autoclip: no git publish step (kokkaidb published data/ to GitHub Pages;
    # autoclip keeps everything local). `no_push` is retained for CLI compat.
    _ = no_push


_FLOOR_LIKE_KINDS = frozenset(("floor_speech", "procedural"))


def _run_step6(
    session_detail: SessionDetail,
    utterances_output: UtterancesOutput,
    *,
    max_workers: int,
) -> tuple[QAPairsOutput, SummaryOutput, TopicsOutput]:
    """session_kind に応じて Q&A・要約・トピック・法案タグを生成する。"""
    sk = session_detail.session_kind

    if sk in _FLOOR_LIKE_KINDS:
        logger.info("Step 6: skipping Q&A extraction (session_kind=%s)", sk)
        qa_pairs = QAPairsOutput(pairs=[])
    else:
        qa_pairs = generate_qa_pairs(
            utterances_output,
            speakers=session_detail.speakers,
            max_workers=max_workers,
            skip_proposal_segments=(sk == "representative_questions"),
        )

    session_meta = {
        "chamber": session_detail.chamber,
        "committee": session_detail.committee,
        "session_kind": session_detail.session_kind,
    }
    session_summary = generate_session_summary(
        qa_pairs, utterances_output, session_meta=session_meta
    )

    # PR11/§2.10: QA が空でも utterances がある場合 (所信表明・施政方針演説・
    # 全 procedural skip 等) は utterances から topics + key_topics を生成し、
    # content_missing を防ぐ
    if qa_pairs.pairs:
        topics, key_topics = generate_topics_and_key_topics(qa_pairs)
    elif utterances_output.segments:
        logger.info(
            "Step 6: generating topics from utterances (session_kind=%s, no Q&A produced)",
            sk,
        )
        topics, key_topics = generate_topics_without_qa(utterances_output)
    else:
        topics, key_topics = generate_topics_and_key_topics(qa_pairs)
    commitments = generate_key_commitments(qa_pairs)

    if qa_pairs.pairs:
        laws_text = filter_laws_for_committee(_load_laws_compact(), session_detail.committee)
        if laws_text:
            qa_pairs = tag_related_laws(
                qa_pairs,
                chamber=session_detail.chamber,
                committee=session_detail.committee,
                date=session_detail.date,
                laws_text=laws_text,
                max_workers=max_workers,
            )

        # Step 6d: V4品質評価指標の付与（QQ/AS/OC 9軸）
        logger.info("Step 6d: Scoring Q&A pairs with V4 quality metrics")
        qa_pairs = score_qa_pairs_metrics(qa_pairs, max_workers=max_workers)

    summary = SummaryOutput(
        session_summary=session_summary,
        key_topics=key_topics,
        key_commitments=commitments,
        related_laws=build_summary_related_laws(qa_pairs),
    )
    return qa_pairs, summary, topics


def _get_scraper(chamber: str) -> BaseScraper:
    """院に応じたScraperインスタンスを返す。

    autoclip は衆議院のみ対応 (kokkaidb の参議院スクレイパーは未コピー)。
    """
    if chamber == "shugiin":
        return ShugiinScraper()
    raise ValueError(f"Unsupported chamber: {chamber} (autoclip is shugiin-only)")


def run_pipeline_for_session(
    chamber: str,
    session_id: str,
    no_push: bool = False,
) -> None:
    """セッション詳細を取得して output_dir を決めた上で run_pipeline を呼ぶ。"""
    scraper = _get_scraper(chamber)
    detail = scraper.get_session_detail(session_id)
    output_dir = _output_dir_for(chamber, detail.date, session_id, detail.committee)
    run_pipeline(chamber, session_id, output_dir, no_push=no_push)


def run_transcribe_only(
    session_id: str,
    chamber: str = "shugiin",
) -> tuple[SessionDetail, RawTranscript]:
    """Steps 2→4.5 のみ実行し (スクレイプ→音声DL→分割→Whisper→LLM修正)、
    (SessionDetail, RawTranscript) を返す。LLM 構造化 (Q&A/要約/トピック) は行わない。

    autoclip の clip パイプラインは語単位タイムスタンプ付き transcript が必要なだけで、
    Q&A 構造化は不要なため、構造化フェーズを省いた軽量経路を提供する。
    (M4 の highlight 自動選定では別途 run_pipeline で qa_pairs.json を生成する。)

    Args:
        session_id: deli_id
        chamber: "shugiin" 固定 (autoclip は衆議院のみ)

    Returns:
        (補正済み SessionDetail, RawTranscript)
    """
    scraper = _get_scraper(chamber)
    session_detail = scraper.get_session_detail(session_id)
    if not session_detail.speakers:
        raise SessionNotReadyError(
            f"Speaker list unexpectedly empty for {chamber} {session_id}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        full_wav = Path(tmpdir) / "full_audio.wav"
        download_full_audio(session_detail.hls_url, full_wav)
        correct_speaker_offsets(session_detail.speakers, full_wav)

        segment_paths = split_segments(
            full_wav,
            session_detail.speakers,
            Path(tmpdir) / "segments",
        )

        raw_transcript = transcribe_all_segments(
            segment_paths,
            session_detail.speakers,
            session_id,
            committee=session_detail.committee,
            max_workers=MAX_WORKERS_WHISPER,
        )

    try:
        raw_transcript = correct_transcript(
            raw_transcript, session_detail, max_workers=MAX_WORKERS_LLM
        )
    except Exception as e:
        logger.warning("Transcript correction failed (non-fatal, using original): %s", e)

    return session_detail, raw_transcript


def main() -> None:
    parser = argparse.ArgumentParser(
        description="国会TV パイプライン: スクレイピング→音声→文字起こし→構造化"
    )
    parser.add_argument(
        "--chamber",
        default="shugiin",
        choices=["shugiin", "sangiin"],
        help="院（デフォルト: shugiin）",
    )

    # セッション指定モード
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--session-id",
        help="セッションID（衆議院: deli_id、参議院: sid）",
    )
    session_group.add_argument(
        "--deli-id",
        help="衆議院TV の deli_id（後方互換、--session-id を推奨）",
    )
    session_group.add_argument(
        "--date",
        help="指定日の全セッションを自動巡回（YYYY-MM-DD、'today'も可）",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="出力先ディレクトリ（--deli-id使用時のみ。省略時はdata/配下に自動生成）",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="git pushをスキップ（ローカルテスト用）",
    )

    args = parser.parse_args()

    try:
        if args.session_id or args.deli_id:
            session_id = args.session_id or args.deli_id
            if args.output_dir:
                output_dir = args.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                run_pipeline(
                    chamber=args.chamber,
                    session_id=session_id,
                    output_dir=output_dir,
                    no_push=args.no_push,
                )
            else:
                run_pipeline_for_session(
                    chamber=args.chamber,
                    session_id=session_id,
                    no_push=args.no_push,
                )

        elif args.date:
            target_date = (
                str(date_type.today()) if args.date == "today" else args.date
            )
            scraper = _get_scraper(args.chamber)
            session_ids = scraper.detect_new_sessions(target_date)
            logger.info("Found %d sessions for %s", len(session_ids), target_date)

            for sid in session_ids:
                try:
                    run_pipeline_for_session(
                        chamber=args.chamber,
                        session_id=sid,
                        no_push=args.no_push,
                    )
                except SessionNotReadyError as e:
                    logger.warning("Session %s not ready: %s", sid, e)
                except EmptySessionError as e:
                    logger.warning("Session %s skipped (empty audio): %s", sid, e)
                except RuntimeError as e:
                    logger.error("Session %s failed: %s", sid, e)

        else:
            parser.print_help()
            sys.exit(1)

    except SessionNotReadyError as e:
        logger.warning("Session not ready, skipping (will retry later): %s", e)
    except EmptySessionError as e:
        logger.warning("Session skipped (empty audio): %s", e)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
