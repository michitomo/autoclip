"""クリップ生成オーケストレーション + CLI (autoclip MVP)。

M1 のエンドツーエンド経路:
    scrape → 映像DL → 議員区間 WAV (無音除去なし) → Whisper(語) → JetCut → ASS → render

使用例:
    python -m src.clip_service --session-id 56325 --member 高市早苗
    python -m src.clip_service --session-id 56325 --member 高市 --out-dir media/clips

出力 (out_dir):
    source.mp4         ダウンロードしたフルセッション動画 (再利用のため残す)
    <member>.wav       議員区間の音声 (無音除去なし)
    <member>.ass/.srt  字幕サイドカー (new タイムライン)
    <member>_clip.mp4  JetCut + 字幕焼き込み済みクリップ
    <member>_edl.json  編集判断 (レビュー UI / 再レンダリング用)
"""

from __future__ import annotations

import argparse
import bisect
import functools
import logging
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.api_client import MAX_WORKERS_LLM, ensure_fd_limit
from src.models import (
    EDL,
    ClipProject,
    EditCaption,
    EditRange,
    KeepRange,
    KeptWord,
    QATree,
    RawTranscript,
    SegmentTranscript,
    SessionDetail,
    SpeakerInfo,
    WhisperSegment,
    WhisperWord,
)
from src.scrapers.shugiin import ShugiinScraper
from src.speaker_lookup import build_lookup, find_by_name
from src.transcriber import strip_prompt_echo, transcribe_segment
from src.transcript_corrector import correct_transcript
from src.video.align import align_corrected_to_words
from src.video.disfluency import detect_disfluency_spans
from src.video.downloader import (
    download_segment_range,
    download_source_video,
    extract_member_wav,
    member_window,
)
from src.video.ffmpeg import ffprobe_bin, has_libass
from src.video.jetcut import _remap_words, build_edl
from src.video.qa_annotate import annotate_sentences
from src.video.qaseg import build_qa_tree, extract_answerers, segment_qa
from src.video.renderer import ASPECT_RESOLUTIONS, render_clip
from src.video.segment import (
    break_after_indices_for_kept,
    break_after_times_from_marked,
    generate_title,
    insert_phrase_breaks,
)
from src.video.subtitles import (
    build_ass,
    build_ass_from_captions,
    build_ass_karaoke,
    build_ass_rolling,
    build_srt,
    group_captions,
)

logger = logging.getLogger(__name__)

_FFPROBE_TIMEOUT = 30

# 議員区間の冒頭取りこぼし防止: start より手前から切り出す秒数。
# 衆議院TV の time= アンカーが発話開始より遅れることへの対策。
LEAD_PAD = 1.2

# JetCut の dead-air しきい値 (秒)。既定を「積極的」に倒した値 (ユーザー要望:
# 短めの間も詰める)。silence_pad(0.20)+merge_gap(0.25) により実効カットは ~0.65s 超の
# 間からで、文中の自然な息継ぎは pad で残るため極端な詰めにはならない。生成時/API で
# 上書き可能 (大きくすると保守的、小さくするとより積極的)。
AGGRESSIVE_DEAD_AIR_GAP = 0.6


def _detect_disfluency_safe(words: list[WhisperWord]) -> list[tuple[float, float]]:
    """言い間違い・言い淀みスパン検出を例外安全に行う (失敗時 [] = dead-air のみ)。"""
    try:
        return detect_disfluency_spans(words)
    except Exception as e:  # noqa: BLE001 - 検出失敗でクリップ生成は止めない
        logger.warning("Disfluency detection failed (dead-air only): %s", e)
        return []


@dataclass
class ClipResult:
    """生成したクリップの成果物パス群。"""

    source_video: Path  # 部分DLの member セグメント or full source
    member_wav: Path
    clip_mp4: Path | None  # 生成時は全体レンダしないので None (プレビューはトピック単位)
    ass_path: Path
    srt_path: Path
    edl_path: Path
    edl: EDL
    member: SpeakerInfo


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace("　", "_")


_JP_WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]


def _title_header_lines(date: str, committee: str, member: str) -> list[str]:
    """タイトル先頭の定型見出し 3 行を作る。

    例: ["2026年6月19日（金）", "衆議院 厚生労働委員会", "古川あおい 質疑"]
    """
    import datetime

    try:
        y, m, d = (int(x) for x in date.split("-"))
        wd = _JP_WEEKDAYS[datetime.date(y, m, d).weekday()]
        date_line = f"{y}年{m}月{d}日（{wd}）"
    except (ValueError, IndexError):
        date_line = date
    return [date_line, f"衆議院 {committee}", f"{member} 質疑"]


def _seg_from_words(
    seg: SegmentTranscript, words: list[WhisperWord]
) -> SegmentTranscript:
    """strip 後の語列から SegmentTranscript を作り直す。

    text と whisper_segments を除去済み語のみで再構築し、後段の校正に junk が
    渡らないようにする。words の時刻はそのまま 1 つの WhisperSegment に格納する。
    """
    text = "".join(w.word for w in words)
    if words:
        ws = WhisperSegment(
            id=0, seek=0,
            start=words[0].start, end=words[-1].end,
            text=text, words=list(words),
        )
        segments = [ws]
    else:
        segments = []
    return seg.model_copy(update={"text": text, "whisper_segments": segments})


def _truncate_edl(edl: EDL, preview_seconds: float) -> EDL:
    """出力(カット後)タイムラインで先頭 preview_seconds 秒に EDL を切り詰める。

    keep_ranges を累積尺が preview_seconds に達するまで採用し、最後の区間は
    途中で切る。kept_words は new_start が preview_seconds 未満のものだけ残す。
    """
    kept_ranges: list[KeepRange] = []
    acc = 0.0
    for r in edl.keep_ranges:
        dur = r.end - r.start
        if acc + dur <= preview_seconds:
            kept_ranges.append(r)
            acc += dur
        else:
            remain = preview_seconds - acc
            if remain > 0.1:
                kept_ranges.append(KeepRange(start=r.start, end=r.start + remain))
            break
    kept_words = [w for w in edl.kept_words if w.new_start < preview_seconds]
    return EDL(keep_ranges=kept_ranges, kept_words=kept_words, params=edl.params)


def _probe_duration(path: Path) -> float | None:
    """ffprobe で動画/音声の長さ (秒) を返す。失敗時 None。"""
    try:
        result = subprocess.run(
            [
                ffprobe_bin(), "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True, capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        logger.warning("ffprobe duration failed for %s: %s", path, e)
        return None


def _resolve_member(
    session_detail: SessionDetail, member: str
) -> tuple[int, SpeakerInfo]:
    """member 名から (index, SpeakerInfo) を解決する (fuzzy 照合)。"""
    speakers = session_detail.speakers
    lookup = build_lookup(speakers)
    matched = find_by_name(member, lookup, allow_single_char=True)
    if matched is None:
        names = ", ".join(f"{s.name}({s.role})" for s in speakers)
        raise ValueError(
            f"Member {member!r} not found among speakers. Available: {names}"
        )
    # matched は dedup 後の SpeakerInfo。index を引く。
    for i, s in enumerate(speakers):
        if s.name == matched.name and s.start_seconds == matched.start_seconds:
            return i, s
    # start_seconds がズレている場合は name 一致で最初の該当
    for i, s in enumerate(speakers):
        if s.name == matched.name:
            return i, s
    raise ValueError(f"Resolved member {matched.name!r} but could not locate index")


def generate_clip(
    session_id: str,
    member: str,
    out_dir: Path,
    *,
    chamber: str = "shugiin",
    min_height: int = 360,
    reuse_source: bool = True,
    full_source: bool = False,
    correct: bool = True,
    phrase_split: bool = True,
    aspect: str = "9:16",
    preview_seconds: float | None = None,
    subtitle_style: str = "karaoke",
    make_title: bool = True,
    title_override: str | None = None,
    remove_disfluencies: bool = True,
    dead_air_gap: float = AGGRESSIVE_DEAD_AIR_GAP,
    progress: Callable[[str], None] | None = None,
) -> ClipResult:
    """1 議員ぶんの JetCut ハイライトクリップを生成する (フル質疑区間対象)。

    既定では議員区間に重なる HLS セグメントだけを部分ダウンロードする
    (フル数時間動画を落とさない)。`full_source=True` で従来のフル DL に切替。

    Args:
        session_id: deli_id
        member: 議員名 (fuzzy 照合)
        out_dir: 成果物の出力ディレクトリ
        chamber: "shugiin" 固定
        min_height: 動画 variant の最小縦解像度
        reuse_source: full_source 時、既存 source.mp4 があれば DL 省略
        full_source: True ならフルセッションをダウンロード (複数議員を作る場合など)
        correct: True なら LLM 校正 (議員名/政党名修正 + 句読点) を字幕に反映
        phrase_split: True なら LLM で文節境界を厳守して字幕を改行 (correct 前提)
        aspect: 出力アスペクト ("9:16"|"1:1"|"16:9")。既定 9:16 縦長センタークロップ。
        preview_seconds: 指定すると出力を先頭 N 秒に切り詰める (高速確認用)。
        subtitle_style: "karaoke" (3行全文表示・発話語ハイライト, 既定) /
            "rolling" (shusantv 風積み上げ) / "plain" (通常文節キャプション)。
        make_title: True で質疑要旨タイトルを LLM 生成し冒頭 2 秒上部に表示。
        remove_disfluencies: True で言い間違い・言い淀みを LLM 検出し映像から JetCut。
        dead_air_gap: 語間無音の除去しきい値 (秒)。小さいほど積極的に間を詰める。

    Returns:
        ClipResult
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _p(step: str) -> None:
        if progress:
            progress(step)

    # 1. スクレイプ
    _p("scraping")
    logger.info("=== Scraping session %s ===", session_id)
    scraper = ShugiinScraper()
    detail = scraper.get_session_detail(session_id)
    if not detail.speakers:
        raise RuntimeError(f"No speakers found for session {session_id}")

    member_index, member_info = _resolve_member(detail, member)
    logger.info(
        "Resolved member: %s (%s, role=%s, index=%d, start=%.1fs)",
        member_info.name, member_info.affiliation, member_info.role,
        member_index, member_info.start_seconds,
    )
    safe = _safe_name(member_info.name)

    # 2. 議員区間の境界を決める。
    # 部分DLでは全体尺が無いため、最後の話者は duration_minutes から end を推定する
    # (member_window が video_duration=None でそうする)。
    start, end = member_window(detail.speakers, member_index, video_duration=None)
    # 衆議院TV の time= アンカーは発話開始よりわずかに後ろのことがあり、start ちょうどで
    # 切ると冒頭が欠ける。LEAD_PAD ぶん手前から切り出して取りこぼしを防ぐ。
    lead_pad = LEAD_PAD
    # クリップ開始 = start - lead_pad (動画/音声/字幕すべてこの基準に揃える)
    clip_begin = max(0.0, start - lead_pad)
    # DL 範囲はさらに余裕を持たせる。
    dl_pad = 2.0
    win_start = max(0.0, start - dl_pad)
    win_end = end + dl_pad

    # 2.5 映像ダウンロード: 既定は議員区間だけの部分DL。
    member_src = out_dir / f"{safe}_src.mp4"
    if full_source:
        source_mp4 = out_dir / "source.mp4"
        if reuse_source and source_mp4.exists() and source_mp4.stat().st_size > 0:
            logger.info("Reusing existing full source video: %s", source_mp4)
        else:
            _p("downloading")
            logger.info("=== Downloading FULL source video ===")
            download_source_video(detail.hls_url, source_mp4, min_height=min_height)
        member_src = source_mp4
        clip_member_start = clip_begin  # full source: 絶対時間
        extract_start, extract_end = clip_begin, end
    else:
        _p("downloading")
        logger.info("=== Downloading member segment range (partial) ===")
        _, seg_offset = download_segment_range(
            detail.hls_url, win_start, win_end, member_src, min_height=min_height
        )
        # member_src の先頭は seg_offset 秒に対応する。clip_begin をローカル時間に直す。
        clip_member_start = max(0.0, clip_begin - seg_offset)
        extract_start = max(0.0, clip_begin - seg_offset)
        extract_end = end - seg_offset

    # 3. 議員区間の WAV (無音除去なし)
    logger.info(
        "=== Extracting member WAV: clip-local %.1fs - %.1fs ===",
        extract_start, extract_end,
    )
    member_wav = out_dir / f"{safe}.wav"
    extract_member_wav(member_src, extract_start, extract_end, member_wav)

    # 4. Whisper 文字起こし (語タイムスタンプ)
    _p("transcribing")
    logger.info("=== Transcribing (word-level) ===")
    # start_seconds=0 の SpeakerInfo を使う (member_wav は議員区間先頭=0 の線形時間)
    local_speaker = member_info.model_copy(update={"start_seconds": 0.0})
    seg = transcribe_segment(member_wav, 0, local_speaker, detail.committee)
    words = [w for ws in seg.whisper_segments for w in ws.words]
    logger.info("Transcribed %d words", len(words))
    if not words:
        raise RuntimeError(
            "No word timestamps returned; cannot JetCut. "
            "Check ASR_PROVIDER / API key."
        )

    # 4.1 Whisper が冒頭にプロンプト echo / ハルシネーション / 呼名ノイズを出した場合は
    # 除去する (発話ではないので字幕に出さない)。strip 後の語列で seg も作り直し、
    # 後段の校正にも除去済みテキストを渡す (校正経由で junk が復活しないように)。
    n_before = len(words)
    words = strip_prompt_echo(words, local_speaker, detail.committee)
    if len(words) != n_before:
        logger.info("Stripped Whisper prompt echo (%d words)", n_before - len(words))
    seg = _seg_from_words(seg, words)

    # 4.2 答弁者 (大臣/副大臣/参考人) を raw 文字起こしの指名文から抽出する。
    # 衆議院TV の発言者リストには質疑者と委員長しか無いため、これが無いと校正が
    # 答弁者名を落とし、話者タグが「政府参考人」汎用名になる。校正前に拾う。
    raw_text = "".join(w.word for w in words)
    answerers = extract_answerers(raw_text)
    # 校正・話者タグに渡す発言者リスト (member 先頭 + 全員 + 抽出答弁者)
    all_speakers_ext = [local_speaker, *detail.speakers, *answerers]

    # 4.3 言い間違い・言い淀みの検出 (LLM)。**補正前 (align 前) の raw 語列** に対して
    # 行う — 言い直しの「言い直す前の誤った部分」は校正後テキストでは既に消えるため。
    # raw 語の時刻は member-WAV 時間で align 後と同一タイムラインなので、得たスパンを
    # そのまま build_edl(drop_spans=) に渡せば映像から切れる。校正と独立なので並列に
    # 走らせ、JetCut 直前で回収する (直列だと校正待ちに検出待ちが上乗せされる)。
    disfluency_spans: list[tuple[float, float]] = []
    disf_ex: ThreadPoolExecutor | None = None
    disf_fut = None
    if remove_disfluencies:
        raw_words_for_disfluency = list(words)
        disf_ex = ThreadPoolExecutor(max_workers=1)
        disf_fut = disf_ex.submit(_detect_disfluency_safe, raw_words_for_disfluency)

    # 4.5 LLM 校正 (議員名/政党名の誤認識修正 + 句読点補完) → word に再アライン。
    # corrector は segment text のみ直すため、補正文を word タイムスタンプへ
    # 文字レベルでアラインして字幕に反映する (kokkaidb の名前修正を活かす)。
    # さらに LLM で文節境界を厳守させ、その境界を境界語の終了時刻として保持する。
    break_times: set[float] = set()
    title = ""
    # タイトル生成用テキスト (補正前の生テキスト; 補正後があれば差し替える)
    title_source = "".join(w.word for w in words)
    # Q&A 分割用テキスト (補正後があれば差し替える)。補正失敗でも raw で動く。
    corrected_text = title_source
    if correct:
        _p("correcting")
        logger.info("=== Correcting transcript (names/punctuation) ===")
        try:
            # corrector が segment_index=0 で member を引けるよう member を先頭に、
            # 答弁者も足して名前が落ちないようにする。
            corrector_detail = detail.model_copy(
                update={"speakers": all_speakers_ext}
            )
            corrected = correct_transcript(
                RawTranscript(session_id=session_id, segments=[seg]),
                corrector_detail,
                max_workers=MAX_WORKERS_LLM,
            )
            corrected_text = corrected.segments[0].text
            title_source = corrected_text
            aligned = align_corrected_to_words(words, corrected_text)
            if aligned:
                words = aligned
                logger.info("Applied corrections to %d words", len(words))
        except Exception as e:
            logger.warning(
                "Transcript correction failed (using raw words): %s", e
            )

    # 5. JetCut → EDL。カット対象:
    #   (a) dead air: 語間無音 > dead_air_gap (既定は積極的な 0.6s)
    #   (b) フィラー: _mark_filler_words (曖昧フィラーは glue ガードで誤爆抑制)
    #   (c) 言い間違い・言い淀み: 上で並列起動した LLM 検出スパン (drop_spans)
    # 検出スパンを回収してから build_edl に渡す (raw 時刻 = align 後と同一 member-WAV)。
    if disf_fut is not None:
        disfluency_spans = disf_fut.result()
        if disf_ex is not None:
            disf_ex.shutdown(wait=False)
    _p("jetcut")
    logger.info(
        "=== JetCut (dead-air %.2fs + fillers + %d disfluency spans) ===",
        dead_air_gap, len(disfluency_spans),
    )
    edl = build_edl(
        words,
        dead_air_gap=dead_air_gap,
        remove_fillers=True,
        drop_spans=disfluency_spans,
    )

    # 4.7〜5.1 校正後の独立した LLM 段を **並列実行** (直列だと文節+タイトル+Q&A で
    # 24秒かかる)。3 段とも corrected_text/words のみに依存し互いに独立:
    #   (a) 文節分割 insert_phrase_breaks → break_times
    #   (b) タイトル generate_title
    #   (c) Q&A: segment_qa → build_qa_tree → annotate_sentences
    # 各段は例外を内部で握って degrade する (1段失敗が他段を巻き込まない)。
    _p("annotating")
    title = title_override or ""
    qa_tree: QATree | None = None

    def _do_phrase_split() -> set[float]:
        if not (correct and phrase_split):
            return set()
        try:
            logger.info("=== Phrase segmentation (LLM, 文節厳守) ===")
            marked = insert_phrase_breaks(corrected_text)
            bt = break_after_times_from_marked(marked, words)
            logger.info("Got %d phrase boundaries", len(bt))
            return bt
        except Exception as e:  # noqa: BLE001
            logger.warning("Phrase segmentation failed: %s", e)
            return set()

    def _do_title() -> str:
        if title_override is not None:
            return title_override
        if not make_title:
            return ""
        try:
            logger.info("=== Generating title (質疑要旨) ===")
            return generate_title(
                title_source, member=member_info.name, committee=detail.committee
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Title generation failed: %s", e)
            return ""

    def _do_qa_tree() -> QATree | None:
        try:
            qas = segment_qa(
                corrected_text, words, member_info, detail.speakers,
                extra_answerers=answerers,
            )
            tree = build_qa_tree(qas, words)
            try:
                tree = annotate_sentences(
                    tree, member=member_info.name, committee=detail.committee
                )
            except Exception as e:  # noqa: BLE001 - 注釈失敗は未注釈ツリーで続行
                logger.warning("QA annotate failed (tree un-annotated): %s", e)
            return tree
        except Exception as e:  # noqa: BLE001 - Q&A 分割失敗はツリーなしで続行
            logger.warning("Q&A segmentation failed (no edit tree): %s", e)
            return None

    with ThreadPoolExecutor(max_workers=3) as _ex:
        f_phrase = _ex.submit(_do_phrase_split)
        f_title = _ex.submit(_do_title)
        f_qa = _ex.submit(_do_qa_tree)
        break_times = f_phrase.result()
        title = f_title.result()
        qa_tree = f_qa.result()

    # 5.25 プレビュー: 出力(カット後)タイムラインで先頭 preview_seconds に切り詰める。
    if preview_seconds is not None and preview_seconds > 0:
        edl = _truncate_edl(edl, preview_seconds)
        logger.info(
            "Preview mode: truncated to first %.0fs (%d keep-ranges)",
            preview_seconds, len(edl.keep_ranges),
        )

    # 5.5 文節境界 (時刻) を JetCut 後の kept_words index に写す。
    break_after = (
        break_after_indices_for_kept(edl.kept_words, break_times)
        if break_times else None
    )

    # 6. 字幕 (new タイムライン)。ASS PlayRes は最終出力解像度に合わせる。
    #   karaoke: 3行全文を表示し発話中の語だけ明色 (既定・スマホ向け大きめ)
    #   rolling: 語が下から積み上がり最新文節が明色 (shusantv 風)
    #   plain:   通常の文節キャプション
    # タイトル先頭の定型見出し (日付/院/委員会/議員) を作る。
    title_header = _title_header_lines(detail.date, detail.committee, member_info.name)
    res_w, res_h = ASPECT_RESOLUTIONS[aspect]
    # 話者色分け用の role span (カット後時間)。初回は全 keep_range が有効。
    enabled_now = [EditRange(start=r.start, end=r.end) for r in edl.keep_ranges]
    role_spans = _role_spans_post_cut(qa_tree, enabled_now)
    if subtitle_style == "karaoke":
        ass = build_ass_karaoke(
            edl.kept_words, res_x=res_w, res_y=res_h, break_after=break_after,
            title=title, title_header=title_header, role_spans=role_spans,
        )
    elif subtitle_style == "rolling":
        ass = build_ass_rolling(
            edl.kept_words, res_x=res_w, res_y=res_h, break_after=break_after,
            role_spans=role_spans, title_header=title_header,
        )
    else:
        ass = build_ass(
            edl.kept_words, res_x=res_w, res_y=res_h, break_after=break_after
        )
    ass_path = out_dir / f"{safe}.ass"
    srt_path = out_dir / f"{safe}.srt"
    ass_path.write_text(ass, encoding="utf-8")
    # SRT は通常の (非ローリング) キャプションで出す (外部利用しやすいよう)。
    srt_path.write_text(
        build_srt(edl.kept_words, break_after=break_after), encoding="utf-8"
    )

    # 7. (全体レンダは廃止) 生成時は登壇全体クリップを焼かない。
    # 編集画面でトピックを選んだ時に render_topic_preview が入力シークでそのトピックだけ
    # 高速に焼く。これで生成の ~70 秒のレンダ待ちが無くなる。
    clip_mp4: Path | None = None

    # 8. EDL を保存 (レビュー / 再レンダリング用)
    edl_path = out_dir / f"{safe}_edl.json"
    edl_path.write_text(edl.model_dump_json(indent=2), encoding="utf-8")

    # 8.5 編集プロジェクト (project.json) を保存。字幕手修正・カットトグル後に
    # Whisper/LLM を再実行せず render_clip だけで焼き直すための状態。
    # 全文初期 ON なので初回クリップ = 全 keep_range。captions/ranges は edl から直接。
    caps = group_captions(edl.kept_words, break_after=break_after)
    project = ClipProject(
        session_id=session_id,
        member=member_info.name,
        source_video=member_src.name,  # out_dir 内の相対名
        member_start=clip_member_start,
        aspect=aspect,
        subtitle_style=subtitle_style,
        title=title,
        title_header=title_header,
        ranges=[
            EditRange(start=r.start, end=r.end, enabled=True)
            for r in edl.keep_ranges
        ],
        captions=[
            EditCaption(start=c.start, end=c.end, text=c.text)
            for c in caps
        ],
        qa_tree=qa_tree,
    )
    (out_dir / f"{safe}_project.json").write_text(
        project.model_dump_json(indent=2), encoding="utf-8"
    )

    if not has_libass():
        logger.warning(
            "Rendered WITHOUT burned-in 字幕 (ffmpeg lacks libass). "
            "Sidecar %s / %s were still written. Install ffmpeg-full for burn-in.",
            ass_path.name, srt_path.name,
        )

    logger.info(
        "=== Done (no full render): project %s (%.1fs content, %d keep-ranges, "
        "%d topics) — topic previews on demand ===",
        out_dir / f"{safe}_project.json", edl.kept_duration,
        len(edl.keep_ranges),
        len(qa_tree.topics) if qa_tree else 0,
    )
    return ClipResult(
        source_video=member_src,
        member_wav=member_wav,
        clip_mp4=clip_mp4,
        ass_path=ass_path,
        srt_path=srt_path,
        edl_path=edl_path,
        edl=edl,
        member=member_info,
    )


def _remap_caption_time(
    t: float, orig_ranges: list[EditRange], enabled_ranges: list[EditRange]
) -> float | None:
    """元(全range)タイムライン上の時刻 t を、有効rangeのみのタイムラインに写す。

    t が属する元キャプション位置を、元 keep_ranges の累積で逆算 → そのソース時刻が
    どの有効 range に入るかで新時刻を求める。無効化された区間内なら None。
    """
    # t (元タイムライン) → ソース(member-WAV)時刻を求める
    acc = 0.0
    src_t: float | None = None
    for r in orig_ranges:
        dur = r.end - r.start
        if acc <= t <= acc + dur + 1e-6:
            src_t = r.start + (t - acc)
            break
        acc += dur
    if src_t is None:
        # 範囲外 → 末尾にクランプ
        src_t = orig_ranges[-1].end if orig_ranges else 0.0

    # ソース時刻 → 有効rangeのみのタイムライン
    new_acc = 0.0
    for r in enabled_ranges:
        if r.start <= src_t <= r.end + 1e-6:
            return new_acc + (src_t - r.start)
        if src_t < r.start:
            return new_acc  # 無効区間に落ちた → 直後の有効range先頭
        new_acc += r.end - r.start
    return new_acc  # 末尾


# ---------------------------------------------------------------------------
# Q&A 編集ツリー → フラット ranges の再計算
#
# 描画契約はあくまでフラットな EditRange (rerender_project / render_clip が消費)。
# ツリーは選択面で、リーフ(文)の enabled から「各 JetCut range をその中点が有効な
# 文 span に入るか」で enabled を再計算する。両者 member-WAV 時間なので直接比較。
# ---------------------------------------------------------------------------

_SPAN_EPS = 1e-6


def _enabled_leaf_spans(tree: QATree) -> list[tuple[float, float]]:
    """ツリー内の enabled な文リーフ span を昇順マージして返す。"""
    spans = sorted(
        (s.start, s.end)
        for topic in tree.topics
        for turn in topic.turns
        for s in turn.sentences
        if s.enabled and s.end > s.start
    )
    if not spans:
        return []
    merged: list[tuple[float, float]] = [spans[0]]
    for s, e in spans[1:]:
        ls, le = merged[-1]
        if s <= le + _SPAN_EPS:  # 重なり/隣接 → 結合
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _mid_in_spans(
    mid: float, starts: list[float], spans: list[tuple[float, float]]
) -> bool:
    """mid がいずれかの span [s,e] に入るか (starts は span 開始の昇順配列)。"""
    if not spans:
        return False
    i = bisect.bisect_right(starts, mid + _SPAN_EPS) - 1
    if i < 0:
        return False
    s, e = spans[i]
    return s - _SPAN_EPS <= mid <= e + _SPAN_EPS


def apply_tree_to_ranges(project: ClipProject) -> ClipProject:
    """qa_tree の leaf enabled から project.ranges[i].enabled を再計算する。

    各 range はその中点が有効な文 span に入れば enabled=True、否なら False。
    qa_tree=None の場合は手動フラット編集を尊重し ranges をそのまま返す。
    再計算後に有効 range が 0 個なら ValueError (空クリップ防止)。
    """
    if project.qa_tree is None:
        return project
    spans = _enabled_leaf_spans(project.qa_tree)
    starts = [s for s, _ in spans]
    for r in project.ranges:
        mid = (r.start + r.end) / 2.0
        r.enabled = _mid_in_spans(mid, starts, spans)
    if not any(r.enabled for r in project.ranges):
        raise ValueError("選択が空です: 少なくとも1つの発言を残してください")
    return project


def _member_to_post_cut(t: float, enabled_ranges: list[EditRange]) -> float:
    """member-WAV 時刻 t を、有効 range のみのカット後タイムラインへ写す。

    t より前にある有効 range の累積長 + 同 range 内オフセット。range 外なら
    最近接にクランプ (字幕色 span の端点に使うので厳密でなくてよい)。
    """
    acc = 0.0
    for r in enabled_ranges:
        if t < r.start:
            return acc
        if r.start <= t <= r.end:
            return acc + (t - r.start)
        acc += r.end - r.start
    return acc


def _role_spans_post_cut(
    qa_tree: QATree | None, enabled_ranges: list[EditRange]
) -> list[tuple[float, float, str]]:
    """qa_tree の各ターン (role) を、カット後タイムラインの (start, end, role) span に。

    字幕の話者色分け (質疑=ミント, 答弁=オレンジ) に使う。ターンは member-WAV 時間
    なので有効 range を通してカット後へ写す。空 (tree なし) は空リスト。
    """
    if qa_tree is None or not enabled_ranges:
        return []
    spans: list[tuple[float, float, str]] = []
    for topic in qa_tree.topics:
        for turn in topic.turns:
            if not turn.sentences:
                continue
            s = _member_to_post_cut(turn.start, enabled_ranges)
            e = _member_to_post_cut(turn.end, enabled_ranges)
            if e > s:
                spans.append((s, e, turn.role))
    spans.sort(key=lambda x: x[0])
    return spans


def _filter_edl_by_spans(edl: EDL, spans: list[tuple[float, float]]) -> EDL:
    """有効文 span に中点が入る keep_range/kept_word だけを残した EDL を作る。

    初回生成 (option b) で「低」を最初から落とすために使う。残った keep_ranges で
    kept_words のカット後時刻 (new_*) を再計算するので、字幕タイミングが新しい
    (低除外後の) タイムラインと一致する。
    """
    starts = [s for s, _ in spans]
    keep = [
        r for r in edl.keep_ranges
        if _mid_in_spans((r.start + r.end) / 2.0, starts, spans)
    ]
    if not keep:
        return edl  # 安全側: 全部落ちるなら元のまま (呼び出し側でガード済み想定)
    keep_lists = [[r.start, r.end] for r in keep]
    # 残す語: 旧時刻 (old_start/old_end) の中点が残存 range に入るもの
    kw_starts = [r[0] for r in keep_lists]
    kw_spans = [(r[0], r[1]) for r in keep_lists]
    surviving = [
        WhisperWord(word=w.word, start=w.old_start, end=w.old_end)
        for w in edl.kept_words
        if _mid_in_spans((w.old_start + w.old_end) / 2.0, kw_starts, kw_spans)
    ]
    remapped = _remap_words(surviving, keep_lists)
    return EDL(keep_ranges=keep, kept_words=remapped, params=dict(edl.params))


def _caption_in_ranges(
    c: EditCaption, orig_ranges: list[EditRange], window: list[EditRange]
) -> bool:
    """caption (全体カット後時間) の発話が window のいずれかの range に入るか。

    caption の中点を全 range タイムライン→member-WAV(ソース)時刻へ逆算し、window
    (= 描画する range 部分集合) に含まれるか判定する。トピック別クリップで、その
    トピック範囲外の caption を落とすのに使う。
    """
    mid = (c.start + c.end) / 2.0
    acc = 0.0
    src: float | None = None
    for r in orig_ranges:
        dur = r.end - r.start
        if acc <= mid <= acc + dur + 1e-6:
            src = r.start + (mid - acc)
            break
        acc += dur
    if src is None:
        return False
    return any(r.start - 1e-6 <= src <= r.end + 1e-6 for r in window)


@functools.lru_cache(maxsize=1)
def _budoux_parser():
    """budoux の日本語パーサ (改行に適した文節境界を推定。辞書不要・軽量)。"""
    import budoux

    return budoux.load_default_japanese_parser()


def _budoux_break_after(words: list[str]) -> set[int] | None:
    """語 (1 字) 列を budoux で文節分割し、各文節末の語 index を break_after に返す。

    words を連結 → budoux.parse で文節リスト → 各文節の文字数を積み上げ、文節の
    最後の文字に対応する語 index を境界にする。失敗時は字数フォールバック (~6字毎)。
    """
    n = len(words)
    if n == 0:
        return None
    # 各語の終端文字オフセット (語は通常 1 字だが複数字でも対応)。
    word_end: list[int] = []
    acc = 0
    for w in words:
        acc += len(w)
        word_end.append(acc)
    try:
        phrases = _budoux_parser().parse("".join(words))
        # 文節末の文字オフセット集合 → その位置以下で最大の word index を境界に。
        breaks: set[int] = set()
        char_pos = 0
        wi = 0
        for ph in phrases:
            char_pos += len(ph)
            # char_pos 以下に終端が収まる最後の語まで wi を進める
            while wi < n and word_end[wi] <= char_pos:
                wi += 1
            if wi > 0:
                breaks.add(wi - 1)  # 文節の最後の語
        breaks.add(n - 1)
        return breaks or None
    except Exception as e:  # noqa: BLE001 - budoux 失敗時は字数で粗く区切る
        logger.warning("budoux split failed (%s); fallback to char-count breaks", e)
        return {i for i in range(n) if (i + 1) % 6 == 0} | {n - 1}


def _karaoke_for_window(
    project: ClipProject, window: list[EditRange], out_path: Path
) -> tuple[list[KeptWord], set[int] | None]:
    """window 用のカラオケ語列 (window-local 時間) と文節境界 break_after を作る。

    _edl.json の kept_words (語+member-WAV/カット後時刻) を読み、各語の old (member-WAV)
    中点が window 内のものだけ残して **window-local 時間** へ new_start/new_end を振り直す。
    break_after は qa_tree の各文末 (member-WAV) を語列に対応づけて「文の最後の語 index」を
    集める (LLM 再実行なしの文節分割)。edl 無し等で取れなければ ([], None) を返す
    (呼び出し側はプレーン字幕にフォールバック)。
    """
    safe = _safe_name(project.member)
    edl_path = out_path.parent / f"{safe}_edl.json"
    if not edl_path.exists():
        return [], None
    try:
        full = EDL.model_validate_json(edl_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("edl.json load failed (%s); plain captions", e)
        return [], None

    # window に入る語 (old 中点が window 内) を window-local 時間へ remap。
    starts = [r.start for r in window]
    spans = [(r.start, r.end) for r in window]
    local: list[KeptWord] = []
    for w in full.kept_words:
        mid = (w.old_start + w.old_end) / 2.0
        if not _mid_in_spans(mid, starts, spans):
            continue
        ns = _window_local_time(w.old_start, window)
        ne = _window_local_time(w.old_end, window)
        if ns is None:
            ns = _window_local_time(mid, window)
        if ne is None or (ns is not None and ne < ns):
            ne = (ns or 0.0) + max(0.02, w.old_end - w.old_start)
        if ns is None:
            continue
        local.append(KeptWord(
            word=w.word, old_start=w.old_start, old_end=w.old_end,
            new_start=ns, new_end=ne,
        ))
    if not local:
        return [], None

    # カラオケのハイライト単位 = **budoux の文節**。Whisper の語は 1 字単位なので
    # 1 語ごとだと字単位でチラつく。budoux (改行に適した文節推定) で意味のまとまりに
    # 区切り、その文節末の語 index を break_after にする。build_ass_karaoke はこの単位で
    # 明色を進め、改行もこの境界で行う。
    break_after = _budoux_break_after([w.word for w in local])
    return local, (break_after or None)


def _render_span_clip(
    project: ClipProject,
    window: list[EditRange],
    out_path: Path,
    *,
    title: str,
    title_header: list[str] | None,
    fps: int = 30,
    preset: str = "veryfast",
    scale_factor: float = 1.0,
    input_seek: bool = False,
    seek_pad: float = 0.5,
) -> Path:
    """project の window (描画する range 部分集合) を 1 クリップに焼く共通処理。

    captions を window タイムラインへ remap (window 外は除外)、話者色 role_spans、
    タイトル/常時ヘッダーを付けて render_clip する。rerender_project (全体)、
    render_topic_clips (トピック別書き出し)、render_topic_preview が共有する。

    input_seek=True の場合、window の最初の range 開始 (ソース絶対秒) まで ffmpeg を
    入力シークしてその区間だけデコードする (トピックプレビュー高速化)。seek_pad ぶん
    手前から取り (キーフレーム対策)、trim で正確に刻む。
    """
    source = out_path.parent / project.source_video
    if not source.exists():
        raise FileNotFoundError(f"source video not found: {source}")
    if not window:
        raise ValueError("no enabled ranges; nothing to render")

    edl = EDL(keep_ranges=[KeepRange(start=r.start, end=r.end) for r in window])
    remapped: list[EditCaption] = []
    for c in project.captions:
        if not _caption_in_ranges(c, project.ranges, window):
            continue
        ns = _remap_caption_time(c.start, project.ranges, window)
        ne = _remap_caption_time(c.end, project.ranges, window)
        if ns is None or ne is None or ne <= ns:
            continue
        remapped.append(EditCaption(start=ns, end=ne, text=c.text))

    res_w, res_h = ASPECT_RESOLUTIONS[project.aspect]
    role_spans = _role_spans_post_cut(project.qa_tree, window)

    # カラオケ字幕を語単位で再構築する。kept_words は _edl.json から読み、window に
    # 属する語だけ window-local 時間へ再 remap。break_after は qa_tree の文末から導出。
    # 取れなければ (旧 edl 無し等) caption ベースのプレーン字幕にフォールバック。
    kw_local, break_after = _karaoke_for_window(project, window, out_path)
    if kw_local:
        ass = build_ass_karaoke(
            kw_local, res_x=res_w, res_y=res_h, break_after=break_after,
            title=title, title_header=title_header or None, role_spans=role_spans,
        )
    else:
        ass = build_ass_from_captions(
            remapped, res_x=res_w, res_y=res_h, title=title,
            title_header=title_header or None, role_spans=role_spans,
        )
    ass_path = out_path.with_suffix(".ass")
    ass_path.write_text(ass, encoding="utf-8")

    seek_arg: float | None = None
    end_arg: float | None = None
    if input_seek:
        # window はソース絶対 = range(member-WAV) + member_start。最小開始の少し手前から、
        # 最大終了の少し後ろまでをデコード対象にする。
        abs_start = min(r.start for r in window) + project.member_start
        abs_end = max(r.end for r in window) + project.member_start
        seek_arg = max(0.0, abs_start - seek_pad)
        end_arg = abs_end + seek_pad

    render_clip(
        source, edl, out_path,
        ass_path=ass_path, member_start=project.member_start, fps=fps,
        aspect=project.aspect, preset=preset, scale_factor=scale_factor,
        input_seek=seek_arg, input_end=end_arg,
    )
    return out_path


def rerender_project(
    project: ClipProject, out_dir: Path, *, fps: int = 30
) -> Path:
    """編集済み ClipProject から「全体クリップ」を焼き直す (Whisper/LLM 再実行なし)。

    有効 range のみで描画し、出力は <member>_clip.mp4 を上書き。トピック別は
    render_topic_clips。共通描画は _render_span_clip。
    """
    safe = _safe_name(project.member)
    enabled = [r for r in project.ranges if r.enabled]
    if not enabled:
        raise ValueError("no enabled ranges; nothing to render")

    clip_mp4 = out_dir / f"{safe}_clip.mp4"
    _render_span_clip(
        project, enabled, clip_mp4,
        title=project.title, title_header=project.title_header or None, fps=fps,
    )
    # project.json も更新保存
    (out_dir / f"{safe}_project.json").write_text(
        project.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("Re-rendered edited clip: %s (%d ranges)", clip_mp4, len(enabled))
    return clip_mp4


def _topic_window(
    project: ClipProject, topic, enabled: list[EditRange]
) -> tuple[list[EditRange], str, list[str]] | None:
    """トピックの (window, タイトル, ヘッダー) を返す。有効文/rangeが無ければ None。

    window = 有効 range のうち中点がトピックの有効文 span に入るもの。
    タイトルはトピック要旨、ヘッダーは定型 + 4行目にトピック要旨。
    """
    spans = sorted(
        (s.start, s.end)
        for turn in topic.turns for s in turn.sentences
        if s.enabled and s.end > s.start
    )
    if not spans:
        return None
    t_start, t_end = spans[0][0], spans[-1][1]
    window = [
        r for r in enabled
        if t_start - 1e-6 <= (r.start + r.end) / 2.0 <= t_end + 1e-6
    ]
    if not window:
        return None
    topic_title = topic.label or project.title
    topic_header = list(project.title_header or [])
    if topic.label:
        topic_header = topic_header + [topic.label]
    return window, topic_title, topic_header


def render_topic_preview(
    project: ClipProject, topic_index: int, out_dir: Path, *, fps: int = 15
) -> dict:
    """1 トピックだけを軽量・入力シークで焼く (編集画面のオンデマンドプレビュー用)。

    プレビューは**トピックの全文 (オフ含む)** で焼く。チェックを外した文は
    フロントが再生時にスキップする (= チェック切替で再レンダ不要)。入力シークで
    そのトピック区間だけデコードするので長尺ソースでも数秒。ultrafast + 360p。
    返り値の disabled_spans はオフ文のプレビュー(ローカル)時間区間 (フロントのスキップ用)。
    """
    if project.qa_tree is None:
        raise ValueError("qa_tree がありません")
    topics = project.qa_tree.topics
    if not (0 <= topic_index < len(topics)):
        raise ValueError(f"topic_index 範囲外: {topic_index}")
    topic = topics[topic_index]

    # トピックの全文 span (オフ含む) → そのトピックに属する全 range を window に。
    all_spans = sorted(
        (s.start, s.end)
        for turn in topic.turns for s in turn.sentences if s.end > s.start
    )
    if not all_spans:
        raise ValueError(f"トピック {topic_index} に発言がありません")
    t_start, t_end = all_spans[0][0], all_spans[-1][1]
    window = [
        r for r in project.ranges
        if t_start - 1e-6 <= (r.start + r.end) / 2.0 <= t_end + 1e-6
    ]
    if not window:
        raise ValueError(f"トピック {topic_index} に対応する区間がありません")

    title = topic.label or project.title
    header = list(project.title_header or [])
    if topic.label:
        header = header + [topic.label]

    safe = _safe_name(project.member)
    out_path = out_dir / f"{safe}_topic{topic_index}_preview.mp4"
    _render_span_clip(
        project, window, out_path, title=title, title_header=header or None,
        fps=fps, preset="ultrafast", scale_factor=1.0 / 3.0, input_seek=True,
    )

    # オフ文の「プレビュー(window連結)ローカル時間」区間を計算 (フロントのスキップ用)。
    disabled_spans = _disabled_spans_in_window(project.qa_tree, window)
    duration = sum(r.end - r.start for r in window)
    logger.info("Rendered topic %d PREVIEW: %s (%.1fs, %d off-spans)",
                topic_index, out_path.name, duration, len(disabled_spans))
    return {
        "clip_path": str(out_path.relative_to(out_dir.parent)),
        "topic_index": topic_index,
        "topic_label": topic.label,
        "duration": round(duration, 1),
        "disabled_spans": disabled_spans,
    }


def _window_local_time(t_member: float, window: list[EditRange]) -> float | None:
    """member-WAV 時刻 t を、window を連結したプレビューのローカル時間に写す。

    window 内 range の累積長 + range 内オフセット。window 外なら None。
    """
    acc = 0.0
    for r in window:
        if r.start <= t_member <= r.end:
            return acc + (t_member - r.start)
        acc += r.end - r.start
    return None


def _disabled_spans_in_window(
    qa_tree: QATree | None, window: list[EditRange]
) -> list[list[float]]:
    """オフ文 (enabled=False) の span を、window 連結プレビューのローカル時間に写す。

    再生時にこの区間をスキップすれば「オフ文を含めて焼いてスキップ」になる。
    端点を window にクランプしてローカル時間へ。連続区間はマージ。
    """
    if qa_tree is None or not window:
        return []
    w_start = window[0].start
    w_end = window[-1].end
    raw: list[tuple[float, float]] = []
    for topic in qa_tree.topics:
        for turn in topic.turns:
            for s in turn.sentences:
                if s.enabled or s.end <= s.start:
                    continue
                a = max(s.start, w_start)
                b = min(s.end, w_end)
                if b <= a:
                    continue
                la = _window_local_time(a, window)
                lb = _window_local_time(b, window)
                if la is None or lb is None or lb <= la:
                    continue
                raw.append((la, lb))
    raw.sort()
    merged: list[list[float]] = []
    for a, b in raw:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [[round(a, 3), round(b, 3)] for a, b in merged]


def render_topic_clips(
    project: ClipProject, out_dir: Path, *, fps: int = 30, only_index: int | None = None
) -> list[dict]:
    """qa_tree の各トピックを別クリップ (<member>_topicN_clip.mp4) に焼く (フル品質書き出し)。

    各トピックの有効文 span と project の有効 range の積をそのトピックの window と
    し、タイトル4行目にトピック要旨を足して描画する。返り値は
    [{clip_path, topic_index, topic_label, duration}] (出力できたトピックのみ)。

    only_index 指定時はその index のトピックだけを書き出す (トピック編集画面の書き出し)。
    """
    if project.qa_tree is None:
        raise ValueError("qa_tree がありません (トピック別書き出し不可)")
    safe = _safe_name(project.member)
    enabled = [r for r in project.ranges if r.enabled]
    if not enabled:
        raise ValueError("選択が空です: 少なくとも1つの発言を残してください")

    topics = project.qa_tree.topics
    if only_index is not None:
        topics = [t for t in topics if t.index == only_index]
        if not topics:
            raise ValueError(f"トピック {only_index} が見つかりません")

    results: list[dict] = []
    for topic in topics:
        tw = _topic_window(project, topic, enabled)
        if tw is None:
            continue
        window, topic_title, topic_header = tw
        out_path = out_dir / f"{safe}_topic{topic.index}_clip.mp4"
        try:
            _render_span_clip(
                project, window, out_path,
                title=topic_title, title_header=topic_header or None, fps=fps,
            )
        except Exception as e:  # noqa: BLE001 - 1 トピック失敗は飛ばして続行
            logger.warning("topic %d render failed: %s", topic.index, e)
            continue
        duration = sum(r.end - r.start for r in window)
        results.append({
            "clip_path": str(out_path.relative_to(out_dir.parent)),
            "topic_index": topic.index,
            "topic_label": topic.label,
            "duration": round(duration, 1),
        })
        logger.info("Rendered topic %d clip: %s (%.1fs)",
                    topic.index, out_path.name, duration)
    if not results:
        raise ValueError("書き出せるトピックがありません")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="衆議院TV の質疑から議員別 JetCut クリップ (字幕付き) を生成する"
    )
    parser.add_argument("--session-id", required=True, help="deli_id (例: 56325)")
    parser.add_argument("--member", required=True, help="議員名 (fuzzy 照合)")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="出力先 (既定: media/clips/<session_id>)",
    )
    parser.add_argument(
        "--min-height", type=int, default=360,
        help="動画 variant の最小縦解像度 (既定 360)",
    )
    parser.add_argument(
        "--aspect", choices=["9:16", "1:1", "16:9"], default="9:16",
        help="出力アスペクト比 (既定 9:16 縦長センタークロップ)",
    )
    parser.add_argument(
        "--preview-seconds", type=float, default=None,
        help="出力を先頭 N 秒に切り詰める (高速確認用)",
    )
    parser.add_argument(
        "--subtitle-style", choices=["karaoke", "rolling", "plain"], default="karaoke",
        help="字幕スタイル (既定 karaoke=3行全文+発話語ハイライト)",
    )
    parser.add_argument(
        "--no-title", action="store_true",
        help="質疑要旨タイトルの冒頭表示を無効化する",
    )
    parser.add_argument(
        "--full-source", action="store_true",
        help="議員区間だけの部分DLではなくフルセッションをダウンロードする",
    )
    parser.add_argument(
        "--no-correct", action="store_true",
        help="LLM 校正 (議員名/句読点修正) をスキップし raw Whisper のまま字幕にする",
    )
    parser.add_argument(
        "--no-phrase-split", action="store_true",
        help="LLM 文節分割をスキップし句読点ベースの改行にする",
    )
    parser.add_argument(
        "--no-disfluency", action="store_true",
        help="言い間違い・言い淀みの LLM 検出カットをスキップ (dead-air+フィラーのみ)",
    )
    parser.add_argument(
        "--dead-air-gap", type=float, default=AGGRESSIVE_DEAD_AIR_GAP,
        help=f"語間無音の除去しきい値 秒 (既定 {AGGRESSIVE_DEAD_AIR_GAP}=積極的。"
             "大きいほど保守的)",
    )
    parser.add_argument(
        "--no-reuse-source", action="store_true",
        help="--full-source 時、既存 source.mp4 を無視して再ダウンロードする",
    )
    parser.add_argument(
        "--preview-topic", type=int, default=None, metavar="N",
        help="生成後、トピック N の軽量プレビュー (入力シーク) を焼いて確認する",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_fd_limit()

    out_dir = args.out_dir or (Path("media/clips") / args.session_id)

    try:
        result = generate_clip(
            session_id=args.session_id,
            member=args.member,
            out_dir=out_dir,
            min_height=args.min_height,
            reuse_source=not args.no_reuse_source,
            full_source=args.full_source,
            correct=not args.no_correct,
            phrase_split=not args.no_phrase_split,
            aspect=args.aspect,
            preview_seconds=args.preview_seconds,
            subtitle_style=args.subtitle_style,
            make_title=not args.no_title,
            remove_disfluencies=not args.no_disfluency,
            dead_air_gap=args.dead_air_gap,
        )
    except Exception as e:
        logger.error("Clip generation failed: %s", e)
        sys.exit(1)

    safe = _safe_name(result.member.name)
    print(f"\n✅ Project: {out_dir / f'{safe}_project.json'}")
    print(f"   字幕: {result.ass_path.name} / {result.srt_path.name}")
    print(f"   EDL:  {result.edl_path.name}  ({result.edl.kept_duration:.1f}s, "
          f"{len(result.edl.keep_ranges)} ranges)")
    print("   (全体レンダなし — トピックプレビューは編集画面 or --preview-topic で)")

    if args.preview_topic is not None:
        proj = ClipProject.model_validate_json(
            (out_dir / f"{safe}_project.json").read_text(encoding="utf-8")
        )
        info = render_topic_preview(proj, args.preview_topic, out_dir)
        print(f"\n▶ Topic preview: {out_dir / Path(info['clip_path']).name} "
              f"({info['duration']}s, {info['topic_label']})")


if __name__ == "__main__":
    main()
