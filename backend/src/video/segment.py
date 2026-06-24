"""LLM による文節分割 (autoclip 固有, 字幕の改行を厳守させる)。

校正済みテキスト (correct_transcript の出力) を LLM に渡し、**文節境界に区切り
マーカー `|` を厳守で挿入**させる。返ってきたマーカー位置を、アライン済みの
KeptWord 列に「この語の後で改行」というフラグとして伝播する。group_captions は
そのフラグを最優先で改行に使う (句読点/文字数はフォールバック)。

shusantv は BudouX を使うが、ユーザー要望により LLM で厳守する。校正とは別の
軽量プロンプトで 1 回呼ぶ。マーカーは元テキストの文字を変えない (挿入のみ) ため、
align と同じ文字シーケンス対応で位置を特定できる。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api_client import LLM_MODEL, get_client, with_retry
from src.models import KeptWord

logger = logging.getLogger(__name__)

SEGMENT_MARKER = "｜"  # 全角縦棒 (本文に出現しにくい)

_TITLE_SYSTEM_PROMPT = """あなたは国会質疑の見出し編集者です。
与えられた質疑テキストから、**質疑の要旨を表す短いタイトル**を 1 つ作ってください。

## ルール
- 18 文字以内。体言止め・名詞句で簡潔に (例:「児童扶養手当の所得制限見直し」)。
- 質疑の主題 (何について問うているか) を表す。発言者名や挨拶は含めない。
- 「〜について」「〜の質疑」等の冗長な語尾は付けない。
- **改行してよい意味の切れ目に区切り記号「｜」を入れる**。各行が概ね同じ長さに
  なるよう 1〜2 個入れる。単語や熟語の途中では絶対に区切らない
  (例:「生物テロ｜への対策強化」「AI悪用による｜サイバー攻撃対策」)。
- 出力はタイトル本文＋区切り記号「｜」のみ。他の記号・引用符・説明は付けない。"""


def generate_title(
    text: str, *, member: str = "", committee: str = "", model: str = LLM_MODEL
) -> str:
    """質疑テキストから短い要旨タイトルを生成する (LLM)。失敗時は空文字。"""
    body = (text or "").strip()
    if not body:
        return ""
    # 長すぎる場合は先頭側だけ渡す (主題は冒頭付近に出やすい + コスト削減)
    snippet = body[:1500]
    ctx = f"委員会: {committee}\n発言者: {member}\n\n質疑テキスト:\n{snippet}"
    client = get_client()

    def _do() -> object:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _TITLE_SYSTEM_PROMPT},
                {"role": "user", "content": ctx},
            ],
            temperature=0.3,
        )

    try:
        resp = with_retry(_do)
        title = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("Title generation failed: %s", e)
        return ""
    # 1 行・余分な記号除去 (区切り記号 ｜ は残す)・長さ制限
    title = title.splitlines()[0].strip().strip("「」『』\"'。 ")
    # 端の ｜ は不要
    title = title.strip(SEGMENT_MARKER)
    # 長さ制限はマーカーを除いた可視文字数で判定
    if len(title.replace(SEGMENT_MARKER, "")) > 24:
        # マーカーを保ったまま 24 可視文字で切る
        kept, vis = [], 0
        for ch in title:
            kept.append(ch)
            if ch != SEGMENT_MARKER:
                vis += 1
            if vis >= 24:
                break
        title = "".join(kept).strip(SEGMENT_MARKER)
    logger.info("Generated title: %s", title)
    return title

_SEGMENT_SYSTEM_PROMPT = """あなたは日本語字幕の文節分割エンジンです。
入力テキストを**文節（意味のまとまり）単位**で区切り、各区切りの境界に区切り記号
「｜」を挿入して返してください。

## 厳守ルール
1. **本文の文字は一切変更しない**。誤字修正・語順変更・要約・追記をしない。
   句読点も増減しない。挿入してよいのは区切り記号「｜」だけ。
2. 文節は「自立語＋付属語」のまとまり。読点「、」の直後と句点「。」の直後には
   必ず「｜」を入れる。
3. 1 つの文節が長い場合は、助詞（は・が・を・に・へ・と・で・も・から・まで・
   など）の直後や接続助詞（て・が・けれど・ので・し）の直後で更に区切る。
4. 1 区切りが概ね 6〜20 文字に収まるようにする。短すぎる細切れにはしない。
5. 出力は区切り記号入りの本文のみ。説明や前置きを付けない。

## 例
入力: 本日は児童扶養手当の所得制限について質問します。
出力: 本日は｜児童扶養手当の｜所得制限について｜質問します。"""


# 1 回の LLM 呼び出しに送る最大文字数。長文を 1 度に送ると本文を勝手に
# 改変しがち (検証で落ちて全文フォールバックになる) ため、文単位で分割する。
_CHUNK_CHAR_LIMIT = 400


def _split_into_chunks(text: str, limit: int = _CHUNK_CHAR_LIMIT) -> list[str]:
    """文 (。！？ 区切り) を保ったまま limit 文字程度のチャンクに分割する。"""
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in "。！？!?":
            sentences.append(buf)
            buf = ""
    if buf.strip():
        sentences.append(buf)

    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + len(s) > limit:
            chunks.append(cur)
            cur = s
        else:
            cur += s
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _phrase_break_one(text: str, model: str) -> str:
    """1 チャンクに文節マーカーを挿入する。検証失敗時は元チャンクを返す。"""
    client = get_client()

    def _do() -> object:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SEGMENT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )

    try:
        resp = with_retry(_do)
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 - 失敗時はフォールバック
        logger.warning("Phrase-break chunk failed (%s); keeping chunk as-is", e)
        return text

    if _normalize(out.replace(SEGMENT_MARKER, "")) != _normalize(text):
        logger.warning("Phrase-break chunk altered body; keeping chunk as-is")
        return text
    return out


def insert_phrase_breaks(
    corrected_text: str, *, model: str = LLM_MODEL, max_workers: int = 16
) -> str:
    """校正済みテキストに文節区切り記号「｜」を挿入して返す (LLM, チャンク並列)。

    文単位のチャンクに分けて各チャンクを **並列に** 処理する。あるチャンクで本文改変が
    起きてもそのチャンクだけ素通しになり、他チャンクの文節境界は活きる。全失敗時は
    元テキスト相当 (マーカー無し) を返し、呼び出し側は句読点フォールバックになる。
    """
    text = corrected_text.strip()
    if not text:
        return corrected_text

    chunks = _split_into_chunks(text)
    results: list[str] = [""] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_phrase_break_one, c, model): i
            for i, c in enumerate(chunks)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    out = "".join(results)

    n_breaks = out.count(SEGMENT_MARKER)
    logger.info(
        "Inserted %d phrase breaks via LLM (%d chunks, parallel)",
        n_breaks, len(chunks),
    )
    return out


def _normalize(s: str) -> str:
    """空白を除いて比較する (LLM が空白を増減しても本文一致とみなす)。"""
    return "".join(s.split())


def break_after_indices_from_marked(
    marked_text: str, aligned_words: list[KeptWord]
) -> set[int]:
    """マーカー入りテキストと aligned_words を突き合わせ、
    「この語インデックスの後で改行」する index 集合を返す。

    aligned_words は marked_text からマーカーを除いた本文に文字単位で対応する
    (align_corrected_to_words が 1 文字 = 1 KeptWord で返すため)。マーカーを除いた
    本文を 1 文字ずつ走査し、各語に対応する本文位置の直後にマーカーがあれば
    その語 index を break 集合に入れる。
    """
    breaks: set[int] = set()
    if SEGMENT_MARKER not in marked_text or not aligned_words:
        return breaks

    # marked_text を走査して「本文 i 文字目の直後にマーカーがあるか」を作る
    body_pos = 0
    break_after_body_char: set[int] = set()
    chars = list(marked_text)
    j = 0
    while j < len(chars):
        ch = chars[j]
        if ch == SEGMENT_MARKER:
            if body_pos > 0:
                break_after_body_char.add(body_pos - 1)  # 直前の本文文字の後で改行
            j += 1
            continue
        if not ch.isspace():
            body_pos += 1
        j += 1

    # aligned_words は本文文字に 1:1 対応 (空白語は flatten 時に除かれている)。
    # 各語の末尾本文位置を数え、break_after_body_char に入っていれば break。
    word_body_pos = 0
    for idx, w in enumerate(aligned_words):
        wlen = len("".join(w.word.split()))
        if wlen == 0:
            continue
        word_end = word_body_pos + wlen - 1
        # この語が占める本文範囲内のいずれかが境界なら break (通常は語末)
        if any(p in break_after_body_char for p in range(word_body_pos, word_end + 1)):
            breaks.add(idx)
        word_body_pos += wlen
    return breaks


def break_after_times_from_marked(
    marked_text: str, aligned_words: list
) -> set[float]:
    """マーカー位置を「その境界語の終了時刻 (end)」の集合として返す。

    JetCut でフィラー語が落ちると語 index がズレるため、index ではなく時刻で
    境界を表現する。aligned_words は WhisperWord (align 出力) を想定し end を使う。
    後段で kept_words の old_end とこの集合を突き合わせて改行する。
    """
    idxs = break_after_indices_from_marked(marked_text, aligned_words)
    return {
        round(aligned_words[i].end, 3) for i in idxs if i < len(aligned_words)
    }


def break_after_indices_for_kept(
    kept_words: list[KeptWord], break_times: set[float], tol: float = 0.05
) -> set[int]:
    """break_times (境界語の end 時刻) を JetCut 後 kept_words の index に写す。

    kept_words.old_end は align 出力 WhisperWord.end と同一基準 (member-WAV 時間)
    なので old_end で突き合わせる。
    """
    out: set[int] = set()
    for idx, w in enumerate(kept_words):
        if any(abs(round(w.old_end, 3) - bt) <= tol for bt in break_times):
            out.add(idx)
    return out
