"""補正済みテキストを word タイムスタンプに再アラインする (autoclip 固有)。

問題: kokkaidb の `correct_transcript` は SegmentTranscript.text を校正する
(議員名/政党名の誤認識修正 + 句読点補完) が、whisper_segments.words は raw のまま
残す。字幕は word.word から作るため、補正がそのままでは字幕に届かない。

解決: raw word 列を連結した文字列と補正後テキストを文字レベルで差分アライン
(difflib.SequenceMatcher) し、補正後の各文字に最も近い raw word の時刻を割り当てて
**補正済み WhisperWord 列** を返す。これを JetCut/字幕に渡せば、タイムスタンプを
保ったまま正しい名前・句読点の字幕になる。

補正は主に局所置換 (高山智→高山聡史) と句読点挿入なので、文字 LCS アラインで十分。
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Sequence

from src.models import WhisperWord

logger = logging.getLogger(__name__)


def _flatten_chars(words: list[WhisperWord]) -> tuple[str, list[tuple[float, float]]]:
    """word 列を (連結文字列, 各文字の(start,end)) に展開する。空白は除く。"""
    chars: list[str] = []
    times: list[tuple[float, float]] = []
    for w in words:
        text = "".join(w.word.split())  # 内部空白除去
        n = len(text)
        if n == 0:
            continue
        # 文字ごとに word の時間を線形補間して割り当てる
        span = w.end - w.start
        for i, ch in enumerate(text):
            cs = w.start + span * (i / n)
            ce = w.start + span * ((i + 1) / n)
            chars.append(ch)
            times.append((cs, ce))
    return "".join(chars), times


# delete スパンがこれ以下なら無条件に捨てる (句読点ノイズ・短いフィラー・
# Whisper の言い直し除去など、校正が正しく消したとみなせる長さ)。
_MIN_GUARDED_DELETE = 12
# 削除テキストが校正後テキスト中に「ほぼ重複として」存在するかの一致しきい値。
# 重複 (Whisper ループ) なら校正が片方を残すので、削除部分は corr に再出現する。
# その場合は捨てて良い。再出現しない = 実発話の脱落なので raw を復元する。
_DUP_RATIO = 0.7


# この長さ以下の equal ブロックは、差分領域の中に紛れた coincidental 一致とみなして
# 前後の非 equal オペコードと結合する (削除領域の分断を防ぐ)。
_EQUAL_BRIDGE = 3


def _coalesce_opcodes(
    opcodes: Sequence[tuple[str, int, int, int, int]],
) -> list[tuple[str, int, int, int, int]]:
    """連続/近接する非 equal オペコードを 1 つの差分領域へまとめる。

    短い equal (<= _EQUAL_BRIDGE 文字) を挟んだ非 equal の並びは、実体としては
    1 つの編集 (主に削除) が coincidental 一致で割れたもの。これらを結合し、tag は
    raw/corr 両側に文字が残るなら "replace"、corr 側が空なら "delete" に正規化する。
    """
    out: list[tuple[str, int, int, int, int]] = []
    n = len(opcodes)
    idx = 0
    while idx < n:
        tag, i1, i2, j1, j2 = opcodes[idx]
        if tag == "equal":
            out.append(opcodes[idx])
            idx += 1
            continue
        # 非 equal の開始。短い equal を橋渡ししつつ末尾まで貪欲に伸ばす。
        end = idx
        k = idx + 1
        while k < n:
            t2 = opcodes[k][0]
            if t2 != "equal":
                end = k
                k += 1
                continue
            # equal: 短ければ橋渡し (後続に非 equal が続く場合のみ)、長ければ打ち切り
            eq_len = opcodes[k][2] - opcodes[k][1]
            has_more = k + 1 < n and opcodes[k + 1][0] != "equal"
            if eq_len <= _EQUAL_BRIDGE and has_more:
                k += 1
                continue
            break
        a1 = opcodes[idx][1]
        a2 = opcodes[end][2]
        b1 = opcodes[idx][3]
        b2 = opcodes[end][4]
        # raw 側が空 → insert、corr 側が空 → delete、両方あり → replace
        if a2 == a1:
            merged_tag = "insert"
        elif b2 == b1:
            merged_tag = "delete"
        else:
            merged_tag = "replace"
        out.append((merged_tag, a1, a2, b1, b2))
        idx = end + 1
    return out


def _looks_duplicated(deleted: str, corr_str: str) -> bool:
    """削除された raw 文字列が、校正後テキストのどこかにほぼ重複で存在するか。

    Whisper のループ/言い直しを校正が片側だけ残した場合、削除分は corr_str 内に
    同等の並びとして残っている → True (捨ててよい)。実発話の脱落なら corr に無い
    → False (復元すべき)。difflib で最長一致ブロック比率を見る簡易判定。
    """
    if not deleted:
        return True
    m = difflib.SequenceMatcher(a=deleted, b=corr_str, autojunk=False)
    match = m.find_longest_match(0, len(deleted), 0, len(corr_str))
    return match.size / len(deleted) >= _DUP_RATIO


def align_corrected_to_words(
    raw_words: list[WhisperWord],
    corrected_text: str,
) -> list[WhisperWord]:
    """補正後テキストの各文字に raw word の時刻を割り当てた WhisperWord 列を返す。

    1 文字 = 1 WhisperWord で返す (JetCut のフィラー照合は隣接トークンを連結する
    実装なので、文字単位でも正しく動く。字幕生成は word.word を連結するだけ)。

    校正が文の塊を丸ごと落とした場合 (LLM の誤作動・チャンク空応答) は字幕欠落に
    直結するため、delete スパンが長く かつ 校正後テキストに重複として残っていない
    ときは **その区間だけ raw の語を時刻つきで復元** する (案1: 捨てすぎガード)。

    Args:
        raw_words: raw Whisper の語列 (時刻付き)
        corrected_text: correct_transcript が返した補正後テキスト

    Returns:
        補正後テキスト由来の WhisperWord 列 (時刻は raw からアライン)。
        raw_words が空なら空リスト。
    """
    if not raw_words:
        return []

    raw_str, raw_times = _flatten_chars(raw_words)
    if not raw_str:
        return []

    # 補正後テキストも空白除去で比較対象を作る (時刻割当も除去後ベース)
    corr_chars = [c for c in corrected_text if not c.isspace()]
    corr_str = "".join(corr_chars)
    if not corr_str:
        return []

    matcher = difflib.SequenceMatcher(a=raw_str, b=corr_str, autojunk=False)
    # 連続する非 equal オペコードを (短い equal の橋渡しを挟んでも) 1 つの差分領域に
    # まとめる。校正の削除は coincidental な短一致で replace/delete に分断されやすく、
    # 領域単位で「重複か実発話か」を判定しないと一部の語を取りこぼすため。
    opcodes = _coalesce_opcodes(matcher.get_opcodes())

    out: list[WhisperWord] = []
    # 直近に割り当てた時刻 (insert で前の文字を引き継ぐ用)
    last_time = raw_times[0]
    restored_chars = 0  # ガードで復元した文字数 (ログ用)

    def _emit_raw(i1: int, i2: int) -> None:
        nonlocal last_time
        for k in range(i2 - i1):
            cs, ce = raw_times[i1 + k]
            out.append(WhisperWord(word=raw_str[i1 + k], start=cs, end=ce))
            last_time = (cs, ce)

    def _emit_replace(i1: int, i2: int, j1: int, j2: int) -> None:
        nonlocal last_time
        span_start = raw_times[i1][0]
        span_end = raw_times[i2 - 1][1] if i2 > i1 else raw_times[i1][1]
        total = span_end - span_start
        m = j2 - j1
        for k in range(m):
            cs = span_start + total * (k / m) if m else span_start
            ce = span_start + total * ((k + 1) / m) if m else span_end
            out.append(WhisperWord(word=corr_str[j1 + k], start=cs, end=ce))
            last_time = (cs, ce)

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(j2 - j1):
                cs, ce = raw_times[i1 + k]
                out.append(WhisperWord(word=corr_str[j1 + k], start=cs, end=ce))
                last_time = (cs, ce)
        elif tag == "insert":
            # corr に新規挿入 (主に句読点)。直前文字の終端時刻を 0 長で引き継ぐ。
            _, prev_end = last_time
            for k in range(j2 - j1):
                out.append(
                    WhisperWord(word=corr_str[j1 + k], start=prev_end, end=prev_end)
                )
        else:  # "replace" / "delete" (coalesce 後はまとめて差分領域として扱う)
            raw_seg = raw_str[i1:i2]
            corr_seg = corr_str[j1:j2]
            removed = len(raw_seg) - len(corr_seg)
            # raw が corr より十分長く (= 実質削除) かつ削除分が校正後に重複として
            # 残っていない場合は、その区間の raw を時刻つきで復元する。
            if (
                removed > _MIN_GUARDED_DELETE
                and not _looks_duplicated(raw_seg, corr_str)
            ):
                _emit_raw(i1, i2)
                restored_chars += removed
            elif tag == "replace":
                _emit_replace(i1, i2, j1, j2)
            else:  # delete: 校正が正しく消したとみなして時刻だけ進める
                last_time = raw_times[i2 - 1]

    # 時刻単調性を保証 (replace/insert で前後すると稀に乱れる)
    for idx in range(1, len(out)):
        if out[idx].start < out[idx - 1].start:
            out[idx] = out[idx].model_copy(
                update={
                    "start": out[idx - 1].start,
                    "end": max(out[idx - 1].end, out[idx].end),
                }
            )
    if restored_chars:
        logger.info(
            "Align guard: restored %d chars dropped by correction (likely real speech)",
            restored_chars,
        )
    logger.info(
        "Aligned corrected text to words: %d raw chars → %d corrected words",
        len(raw_str), len(out),
    )
    return out
