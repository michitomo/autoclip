"""言い間違い・言い淀みの検出 (autoclip 固有, LLM で映像カット範囲を出す)。

校正LLM (transcript_corrector) は **字幕テキストだけ** を直し、映像 (keep_ranges) は
カットしない。そのため「えー」「あのー」等のフィラーや、「予算、いや補正予算」のような
言い直しは字幕からは消えても**映像・音声にはそのまま残る**。JetCut で語間無音
(dead air) を切るだけでは、これらの短い disfluency は隣接語の gap が小さく除去されない。

このモジュールは **補正前 (raw) のテキスト** に対して LLM を 1 回 (長ければチャンク並列)
呼び、カットすべき箇所だけを区切り記号 ⟪ ⟫ で囲ませる。raw を使うのは、言い直しの
「言い直す前の誤った部分」が補正後テキストでは既に消えているため (校正がそれを
削除する)。raw の語タイムスタンプは member-WAV 時間で、補正後 (align 済み) の語と
同一タイムラインなので、ここで得た時刻スパンをそのまま build_edl(drop_spans=...) に
渡せば映像から実際に切除できる。

設計の要:
- マーカーは本文の文字を変えない (挿入のみ)。`insert_phrase_breaks` と同じ方式で
  「本文不変」を検証し、改変があったチャンクは捨てる (過剰カットを避ける)。
- char→時刻の写像は align._flatten_chars と同じく「語を 1 文字ずつに割って線形補間」。
- 過剰検出ガード: 1 スパンが長すぎる (文を丸ごと囲んだ) もの、合計が本文の大半を
  占めるものは破棄する。人手レビュー前提だが、LLM の暴走で本文が消えるのを防ぐ。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api_client import LLM_MODEL, get_client, with_retry
from src.models import WhisperWord

logger = logging.getLogger(__name__)

# 本文に出現しにくい囲み記号 (mathematical left/right double angle bracket)。
MARK_OPEN = "⟪"   # ⟪
MARK_CLOSE = "⟫"  # ⟫

# 1 回の LLM 呼び出しに送る最大文字数。長文は分割して並列に投げる。
_CHUNK_CHAR_LIMIT = 500
_MAX_WORKERS = 12

# 過剰検出ガード:
#   1 スパンの最大文字数 (これ超は「文を丸ごと囲んだ」誤検出として破棄)。
_MAX_SPAN_CHARS = 25
#   1 スパンの最大秒数 (これ超は本文を食った誤検出として破棄)。
_MAX_SPAN_SECONDS = 5.0
#   検出合計が本文に占める割合の上限 (超なら LLM 暴走として全破棄、保守側へ)。
_MAX_TOTAL_FRACTION = 0.35

_SYSTEM_PROMPT = """あなたは日本語の話し言葉から「不要な発話」を検出するエンジンです。
国会答弁の書き起こしテキストを受け取り、視聴者向けクリップから**カットすべき箇所**
だけを区切り記号 ⟪ ⟫ で囲んで返してください。

## カット対象（⟪ ⟫ で囲む）
1. フィラー・言い淀み: 「えー」「えーと」「あのー」「まあ」「んー」など意味を持たない繋ぎ。
   単独の言い淀みの「その」「あの」も対象。
2. 言い間違い・言い直し: 言い直したときの**言い直す前の誤った部分**だけを囲む。
   例:「⟪予算、いや⟫補正予算」「⟪三月、失礼⟫、四月」「⟪わた⟫、私は」
3. どもり・無意味な即時反復:「⟪こ、こ⟫、この」「⟪私は⟫私は」(重複した片方を囲む)

## 絶対に囲まないもの
- 意味のある本文・固有名詞・数値・日付・法令名・政党名。
- 強調や論理のための繰り返し（「重要です、本当に重要です」等）。
- 「その通り」「そのため」「その点」「あの方」のように後ろの語と一体の連体詞・指示語。

## 厳守事項
- **本文の文字は一切変更しない**。記号 ⟪ ⟫ を挿入する以外、削除・修正・並べ替え・
  要約・句読点の増減を一切しない。
- カット対象が無ければ、入力テキストをそのまま (記号なしで) 返す。
- 1 箇所の囲みは**短く**する。言い直しの誤り部分・フィラーだけを囲み、文や節を丸ごと
  囲まない。
- 出力は本文＋区切り記号 ⟪ ⟫ のみ。説明・前置きを付けない。"""


def _chars_with_times(
    words: list[WhisperWord],
) -> tuple[str, list[tuple[float, float]]]:
    """語列を (連結文字列, 各文字の(start,end)) に展開する。空白は除く。

    align._flatten_chars と同じ規約 (語内は時間を線形補間)。これにより
    「本文 i 文字目」→ member-WAV 時刻 が引ける。
    """
    chars: list[str] = []
    times: list[tuple[float, float]] = []
    for w in words:
        text = "".join(w.word.split())
        n = len(text)
        if n == 0:
            continue
        span = w.end - w.start
        for i, ch in enumerate(text):
            cs = w.start + span * (i / n)
            ce = w.start + span * ((i + 1) / n)
            chars.append(ch)
            times.append((cs, ce))
    return "".join(chars), times


def _normalize(s: str) -> str:
    """空白を除いて比較する (LLM が空白を増減しても本文一致とみなす)。"""
    return "".join(s.split())


def _split_chunks(raw: str, limit: int = _CHUNK_CHAR_LIMIT) -> list[tuple[int, str]]:
    """raw_str を limit 文字程度に分割し [(base_offset, chunk_body)] を返す。

    raw_str は空白を含まない (_chars_with_times の出力)。固定長で素朴に割る
    (disfluency がチャンク境界をまたぐのは稀で、またいでも片側だけ検出されるだけ)。
    """
    if not raw:
        return []
    out: list[tuple[int, str]] = []
    for base in range(0, len(raw), limit):
        out.append((base, raw[base : base + limit]))
    return out


def _spans_from_marked(
    marked: str, char_times: list[tuple[float, float]], base: int
) -> list[tuple[float, float]]:
    """⟪ ⟫ で囲まれた本文範囲を member-WAV 時刻スパン列に写す。

    marked は base からの 1 チャンク分の本文に ⟪ ⟫ を挿入したもの。本文文字
    (空白・マーカー以外) を 1 つずつ数え、その位置を char_times[base + pos] に対応づける。
    囲み範囲 [start_body, end_body) を時刻 [char_times[..start].start,
    char_times[..end-1].end] に変換する。

    ガード: 不対応のマーカーは無視。空囲み・長すぎる囲みは捨てる。
    """
    spans: list[tuple[float, float]] = []
    body_pos = 0          # このチャンク内の本文文字位置 (0 始まり)
    open_at: int | None = None
    n_times = len(char_times)
    for ch in marked:
        if ch == MARK_OPEN:
            if open_at is None:
                open_at = body_pos
            continue
        if ch == MARK_CLOSE:
            if open_at is not None and body_pos > open_at:
                start_body = open_at
                end_body = body_pos  # exclusive
                length = end_body - start_body
                gi = base + start_body
                gj = base + end_body - 1
                if length <= _MAX_SPAN_CHARS and 0 <= gi < n_times and gj < n_times:
                    s = char_times[gi][0]
                    e = char_times[gj][1]
                    if e - s <= _MAX_SPAN_SECONDS and e > s:
                        spans.append((s, e))
            open_at = None
            continue
        if ch.isspace():
            continue
        body_pos += 1
    return spans


def merge_spans(
    spans: list[tuple[float, float]], gap: float = 0.0
) -> list[tuple[float, float]]:
    """昇順ソートし、重なり/gap 以下で隣接するスパンを結合する。"""
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[list[float]] = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def _detect_one(chunk_body: str, model: str) -> str:
    """1 チャンクに ⟪ ⟫ を挿入させる。本文改変・失敗時は元チャンク (記号なし) を返す。"""
    client = get_client()

    def _do() -> object:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": chunk_body},
            ],
            temperature=0.0,
        )

    try:
        resp = with_retry(_do)
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 - 失敗時はカットなしで続行
        logger.warning("Disfluency chunk failed (%s); no cut for this chunk", e)
        return chunk_body

    stripped = out.replace(MARK_OPEN, "").replace(MARK_CLOSE, "")
    if _normalize(stripped) != _normalize(chunk_body):
        logger.warning("Disfluency chunk altered body; ignoring this chunk")
        return chunk_body
    return out


def detect_disfluency_spans(
    words: list[WhisperWord],
    *,
    model: str = LLM_MODEL,
    max_workers: int = _MAX_WORKERS,
) -> list[tuple[float, float]]:
    """raw 語列から、カットすべき言い間違い・言い淀みの member-WAV 時刻スパンを返す。

    LLM が ⟪ ⟫ で囲んだ範囲を時刻に写し、重なりを結合して返す。検出合計が本文の
    大半 (_MAX_TOTAL_FRACTION) を超えた場合は LLM 暴走とみなし全破棄 (保守側)。
    LLM 失敗・空入力では [] を返す (呼び出し側は dead-air カットのみで続行)。
    """
    raw_str, char_times = _chars_with_times(words)
    if not raw_str:
        return []

    chunks = _split_chunks(raw_str)
    marked_by_base: dict[int, str] = {}
    if len(chunks) == 1:
        base, body = chunks[0]
        marked_by_base[base] = _detect_one(body, model)
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as ex:
            futs = {ex.submit(_detect_one, body, model): base for base, body in chunks}
            for fut in as_completed(futs):
                marked_by_base[futs[fut]] = fut.result()

    spans: list[tuple[float, float]] = []
    for base, marked in marked_by_base.items():
        spans.extend(_spans_from_marked(marked, char_times, base))
    merged = merge_spans(spans)

    total = sum(e - s for s, e in merged)
    speech = char_times[-1][1] - char_times[0][0] if char_times else 0.0
    if speech > 0 and total / speech > _MAX_TOTAL_FRACTION:
        logger.warning(
            "Disfluency detection flagged %.0f%% of speech (%.1fs/%.1fs); "
            "discarding as likely over-detection",
            100 * total / speech, total, speech,
        )
        return []

    logger.info(
        "Disfluency: %d spans, %.1fs to cut (%.0f%% of %.1fs speech)",
        len(merged), total, (100 * total / speech) if speech else 0.0, speech,
    )
    return merged
