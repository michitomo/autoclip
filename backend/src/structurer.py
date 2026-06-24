"""LLM Q&Aペア生成・要約・トピック抽出 (DeepInfra DeepSeek V3.2)

セグメント単位で並列にQ&Aペアを生成し、抜け漏れを防止する。
LLMには utterance_indices と判断のみ返させ、full_text はコードが
utterance 全文を連結して組み立てる (docs/STRUCTURER_REWRITE.md §2.1 / Appendix A)。
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from src.api_client import get_client as _get_client
from src.api_client import with_retry
from src.models import (
    AnswerDetail,
    KeyCommitment,
    QAMetrics,
    QAPair,
    QAPairsOutput,
    QuestionDetail,
    RelatedLawTag,
    SegmentUtterances,
    SpeakerInfo,
    Topic,
    TopicsOutput,
    UtterancesOutput,
)
from src.prompts import (
    COMMITMENTS_SYSTEM_PROMPT,
    LAW_TAGGING_SYSTEM_PROMPT,
    QA_METRICS_V4_SYSTEM_PROMPT,
    QA_METRICS_V4_USER_TEMPLATE,
    QA_SEGMENT_SYSTEM_PROMPT,
    SESSION_SUMMARY_SYSTEM_PROMPT,
    TOPICS_FROM_UTTERANCES_SYSTEM_PROMPT,
    TOPICS_SYSTEM_PROMPT,
)
from src.speaker_lookup import find_by_name

# Step 6 LLM: OpenRouter 経由 Gemma 4 31B-it
STRUCTURER_MODEL = "google/gemma-4-31b-it"

# QA ペア生成のみ高精度モデルを使う (split_anchor 指定の精度が重要)
QA_MODEL = "google/gemini-3-flash-preview"

# 答弁本文がこの長さ未満かつ utterance_indices が空のペアは Q&A として成立していないため drop
MIN_ANSWER_LENGTH = 30

# 長文 utterance 判定 (sentence sub-number `(sN)` を併記する閾値)
# 代表質問・所信表明など共有 utterance 候補のみ対象。99% の utterance は素のテキストのみ
_LONG_UTTERANCE_CHARS = 800
_LONG_UTTERANCE_SENTENCES = 8

# プロンプトが極端に長くなったときの警告閾値 (block 分割不足の検知用)
_PROMPT_TEXT_WARN_THRESHOLD = 50000

logger = logging.getLogger(__name__)


_CTRL_ESCAPE = {
    '\n': '\\n', '\r': '\\r', '\t': '\\t',
    **{chr(c): f'\\u{c:04x}' for c in range(0x20) if chr(c) not in '\n\r\t'},
}


def _repair_json_string_controls(text: str) -> str:
    """JSON文字列リテラル内の未エスケープ制御文字を修復する。

    json_object モードを指定しても一部のモデル（gemini-3-flash-preview 等）が
    文字列値内に生の改行や制御文字を出力することがある。JSON仕様では文字列内の
    制御文字は \\n / \\uXXXX 形式にエスケープする必要があるため、ステートマシンで
    文字列リテラルの内側のみ対象に修復する。
    """
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_string:
            if c == '\\':
                out.append(c)
                i += 1
                if i < len(text):
                    out.append(text[i])
            elif c == '"':
                in_string = False
                out.append(c)
            elif ord(c) < 0x20:
                out.append(_CTRL_ESCAPE.get(c, f'\\u{ord(c):04x}'))
            else:
                out.append(c)
        else:
            if c == '"':
                in_string = True
            out.append(c)
        i += 1
    return ''.join(out)


def _split_sentences(text: str) -> list[str]:
    """テキストを文単位に分割する。句点・疑問符・感嘆符で分割。"""
    parts = re.split(r'(?<=[。？！])', text)
    result = [p.strip() for p in parts if p.strip()]
    return result if result else [text]


@dataclass
class _SegmentLayout:
    """セグメント内の utterance / sentence マッピング情報。

    LLM プロンプト構築 (`_build_utterance_map`) と full_text 組み立て
    (`_assemble_full_text_for_pair`) の両方で参照する。
    """

    per_utt_sentences: list[list[str]]  # [u_idx][local_sent_idx] -> sentence text
    utt_global_starts: list[int]  # [u_idx] -> 最初の sentence の global index
    total_sentences: int
    is_long_utt: list[bool]  # [u_idx] -> sentence sub-number を出すか


def _compute_segment_layout(seg: SegmentUtterances) -> _SegmentLayout:
    per_utt: list[list[str]] = [_split_sentences(u.text) for u in seg.utterances]
    starts: list[int] = []
    is_long: list[bool] = []
    cur = 0
    for sentences, u in zip(per_utt, seg.utterances, strict=True):
        starts.append(cur)
        cur += len(sentences)
        is_long.append(
            len(u.text) >= _LONG_UTTERANCE_CHARS
            or len(sentences) >= _LONG_UTTERANCE_SENTENCES
        )
    return _SegmentLayout(
        per_utt_sentences=per_utt,
        utt_global_starts=starts,
        total_sentences=cur,
        is_long_utt=is_long,
    )


def _build_utterance_map(seg: SegmentUtterances, layout: _SegmentLayout) -> str:
    """セグメントを `[U0]`, `[U1]`, ... の番号付きテキストに整形する。

    長文 utterance のみ sentence サブ番号 `(sN)` をグローバル通番で併記。
    LLM はこの番号を `utterance_indices` / `split_anchor_sentence_idx` に
    そのまま使う。
    """
    lines: list[str] = [
        f"セグメント発言者: {seg.segment_speaker}（{seg.segment_affiliation}）",
        "",
    ]
    for i, u in enumerate(seg.utterances):
        lines.append(f"[U{i}] [{u.role}] {u.speaker}:")
        if layout.is_long_utt[i]:
            start = layout.utt_global_starts[i]
            for j, s in enumerate(layout.per_utt_sentences[i]):
                lines.append(f"  (s{start + j}) {s}")
        else:
            lines.append(f"  {u.text}")
        lines.append("")
    return "\n".join(lines)


def _format_segments_for_prompt(segments: list[SegmentUtterances]) -> str:
    """全セグメントをLLMプロンプト用にテキスト化する。"""
    lines: list[str] = []
    for seg in segments:
        lines.append(f"\n--- セグメント {seg.segment_index}: {seg.segment_speaker}（{seg.segment_affiliation}）---")
        for u in seg.utterances:
            lines.append(f"[{u.role}] {u.speaker}: {u.text}")
    return "\n".join(lines)


def _assemble_full_text_for_pair(
    seg: SegmentUtterances,
    layout: _SegmentLayout,
    utterance_indices: list[int],
    split_anchor_sentence_idx: int | None,
    boundary_global_idx: int | None,
) -> str:
    """utterance_indices と anchor から full_text を機械的に組み立てる。

    - `split_anchor_sentence_idx` が None: 全 utterance の text を改行連結 (99% のケース)
    - anchor あり: 先頭 utterance を anchor で slice し、後続 utterance は丸ごと連結。
      `boundary_global_idx` は同じ utterance を共有する次のペアの anchor (排他的境界)。
    """
    valid_uidx = [i for i in utterance_indices if 0 <= i < len(seg.utterances)]
    if not valid_uidx:
        return ""

    # BUG FIX: LLM が anchor を文字列 "4" 等で返した場合 `"4" - int` で TypeError になる。
    # isinstance(int) チェックで非 int anchor はフォールバック扱いにする。
    if not isinstance(split_anchor_sentence_idx, int):
        return "\n".join(seg.utterances[i].text for i in valid_uidx)

    head_uidx = valid_uidx[0]
    sentences = layout.per_utt_sentences[head_uidx]
    g_start = layout.utt_global_starts[head_uidx]

    local_anchor = max(0, split_anchor_sentence_idx - g_start)
    if local_anchor >= len(sentences):
        # 範囲外 anchor は信頼できないので utterance 全文にフォールバック
        head_text = "".join(sentences)
    else:
        if boundary_global_idx is not None:
            local_end = max(local_anchor, boundary_global_idx - g_start)
            local_end = min(local_end, len(sentences))
        else:
            local_end = len(sentences)
        head_text = "".join(sentences[local_anchor:local_end])

    if len(valid_uidx) == 1:
        return head_text
    rest = "\n".join(seg.utterances[i].text for i in valid_uidx[1:])
    return f"{head_text}\n{rest}" if head_text else rest


def _compute_share_boundaries(
    parsed_pairs: list[dict[str, Any]],
    side: str,
    layout: _SegmentLayout,
) -> list[int | None]:
    """同一 head utterance を共有するペアの anchor 順序から境界を求める。

    side: "q" or "a"。各ペアの (uidx_key, anchor_key) を見て、anchor を持つペアが
    同じ head utterance に複数あれば anchor 昇順でソートし、各ペアの境界を
    「次のペアの anchor」とする。最後のペアは None (utterance 末尾まで)。

    PR28: 複数ペアが同一 head utterance を共有しているのに anchor が全/部分的に
    null の場合 (代表質問など、LLM が anchor 指示を省略した場合)、その utterance の
    sentence 数で均等分割した anchor を `parsed_pairs` に書き戻して duplicate
    full_text を防ぐ。
    """
    uidx_key = f"{side}_uidx"
    anchor_key = f"{side}_anchor"

    # 1. head utterance ごとにペアをグループ化 (入力順 = 時系列)
    head_groups: dict[int, list[int]] = {}
    for i, p in enumerate(parsed_pairs):
        uidx = p[uidx_key]
        if not uidx:
            continue
        head_groups.setdefault(uidx[0], []).append(i)

    # 2. 同一 head に N>=2 ペアあって anchor が欠けていれば均等補完 (PR28)
    for head, pair_indices in head_groups.items():
        if len(pair_indices) < 2:
            continue
        if head < 0 or head >= len(layout.per_utt_sentences):
            continue
        n_sent = len(layout.per_utt_sentences[head])
        if n_sent < 2:
            continue
        g_start = layout.utt_global_starts[head]
        n_pairs = len(pair_indices)

        explicit = [
            (j, parsed_pairs[i][anchor_key])
            for j, i in enumerate(pair_indices)
            if isinstance(parsed_pairs[i][anchor_key], int)
        ]

        if not explicit:
            # 全 null: 入力順に均等分割
            for j, pair_idx in enumerate(pair_indices):
                local_anchor = (j * n_sent) // n_pairs
                parsed_pairs[pair_idx][anchor_key] = g_start + local_anchor
            logger.info(
                "PR28 (%s): inferred even-split anchors for head U%d (n_sent=%d, n_pairs=%d)",
                side, head, n_sent, n_pairs,
            )
            continue

        # 部分的: null を前後 explicit anchor の中点で埋める
        for j, pair_idx in enumerate(pair_indices):
            if isinstance(parsed_pairs[pair_idx][anchor_key], int):
                continue
            prev_anchor = g_start
            next_anchor = g_start + n_sent
            for prev_j, prev_anc in explicit:
                if prev_j < j:
                    prev_anchor = max(prev_anchor, prev_anc)
            for next_j, next_anc in explicit:
                if next_j > j:
                    next_anchor = min(next_anchor, next_anc)
                    break
            inferred = (prev_anchor + next_anchor) // 2
            parsed_pairs[pair_idx][anchor_key] = max(
                g_start, min(inferred, g_start + n_sent - 1)
            )

    # 3. anchor ベースで境界を計算 (既存ロジック)
    shared: dict[int, list[tuple[int, int]]] = {}
    for i, p in enumerate(parsed_pairs):
        uidx = p[uidx_key]
        anchor = p[anchor_key]
        if not uidx or anchor is None or not isinstance(anchor, int):
            continue
        head = uidx[0]
        shared.setdefault(head, []).append((i, anchor))

    boundaries: list[int | None] = [None] * len(parsed_pairs)
    for items in shared.values():
        if len(items) <= 1:
            continue
        items.sort(key=lambda x: x[1])
        for j, (pair_idx, _) in enumerate(items):
            boundaries[pair_idx] = items[j + 1][1] if j + 1 < len(items) else None
    return boundaries


def _fuzzy_lookup(name: str, speakers_lookup: dict[str, SpeakerInfo]) -> SpeakerInfo | None:
    """完全一致 → 姓一致で speaker 情報を取得する（structurer 互換ラッパー）。"""
    return find_by_name(name, speakers_lookup, allow_single_char=True)


# PR23: 日本語の平均発話速度 (政治家の演説で 200-260 char/min ≒ 4 char/sec) で
# segment 内の utterance 開始秒を線形推定する。segment 起点固定よりは
# 大幅にマシなジャンプ精度になる (誤差 ±10-30s 程度)。
_AVG_CHARS_PER_SECOND = 4.0


def _estimate_pair_offset_seconds(
    seg: SegmentUtterances,
    layout: _SegmentLayout,
    utterance_indices: list[int],
    split_anchor_sentence_idx: int | None = None,
) -> float:
    """質問先頭の発言開始秒を、segment 内の文字位置から推定する (PR23 + PR23.1)。

    1. utterance 単位のオフセット: head utterance より前の utterances の総文字数 / 4
    2. PR23.1: anchor が指定されていれば、head utterance 内で anchor sentence
       より前の文字数も加算する。これにより同一 head_utt を共有する複数ペアの
       video_url が pair ごとに異なる時刻を指す (代表質問・所信表明での頭出し)。

    Returns: seg.start_seconds に加算するオフセット (秒)。先頭 utterance かつ
    anchor が先頭または None なら 0。
    """
    valid = [i for i in utterance_indices if 0 <= i < len(seg.utterances)]
    if not valid:
        return 0.0
    head = valid[0]
    chars_before_head = sum(len(seg.utterances[j].text) for j in range(head))

    chars_within_head = 0
    if (
        split_anchor_sentence_idx is not None
        and isinstance(split_anchor_sentence_idx, int)
        and 0 <= head < len(layout.per_utt_sentences)
    ):
        sentences = layout.per_utt_sentences[head]
        g_start = layout.utt_global_starts[head]
        local_anchor = max(0, split_anchor_sentence_idx - g_start)
        if 0 < local_anchor < len(sentences):
            chars_within_head = sum(len(sentences[k]) for k in range(local_anchor))

    total_chars = chars_before_head + chars_within_head
    if total_chars <= 0:
        return 0.0
    return total_chars / _AVG_CHARS_PER_SECOND


_VIDEO_TIME_PARAM_RE = re.compile(r"time=[\d.]+")
_VIDEO_HASH_TIME_RE = re.compile(r"#[\d.]+$")

# PR47: カタカナ名 (ラサール石井 等) に対応するため全 name-match pattern に ァ-ヿ を追加。
# 日本語名前に使える文字: 漢字 [一-鿿] + ひらがな [ぁ-ゟ] + カタカナ [ァ-ヿ]
_JP_NAME_CHARS = r"[一-鿿ぁ-ゟァ-ヿ]"

# PR43: full_text 末尾に混入した次発言者ラベルを除去する。
# Pattern A: 改行後のラベル — 「\n森本真治（立憲民主・無所属）」「\n藤川政人委員長」
# Pattern B: 改行なし+党名括弧 — 「三原じゅん子（自由民主党）。」
# Pattern C: 改行なし+敬称/役職 — 「小里君。」「山内君。」「藤川委員長。」
# PR47: カタカナ名前に対応（ラサール石井等）
_TRAILING_SPEAKER_LABEL_RE = re.compile(
    r"(?:"
    rf"\n+(?:[○◯])?{_JP_NAME_CHARS}{{2,10}}(?:（[^）]{{2,40}}）)?(?:委員長|議長|君|さん)?[。、]?"  # A
    rf"|(?:[○◯])?{_JP_NAME_CHARS}{{2,10}}（[^）]{{2,40}}）(?:委員長|議長|君|さん)?[。、]?"          # B
    rf"|(?:[○◯])?{_JP_NAME_CHARS}{{2,8}}(?:委員長|議長|君|さん)[。、]?"                             # C
    r")\s*$"
)

# PR43: answer.full_text 冒頭に混入した話者ラベルを除去する。
# PR47: カタカナ名前に対応
_LEADING_ANSWER_LABEL_KEYWORDS = (
    '内閣総理大臣','総理大臣','国務大臣','大臣政務官','副大臣','大臣',
    '副長官','長官','次長','局長','審議官','参事官','部長','参考人',
)
_LEADING_SPEAKER_LABEL_RE = re.compile(
    rf"(?:"
    # 漢字・カタカナのみ (ひらがな不可) で 1〜20字 — 「林大臣。」等の1字姓にも対応。
    # ひらがなを除外することで「今朝の大臣。」等の誤 strip を防ぐ。
    rf"^[一-鿿ァ-ヿ]{{1,20}}?(?:{'|'.join(re.escape(k) for k in _LEADING_ANSWER_LABEL_KEYWORDS)})[。、：]\s*"
    rf"|^{_JP_NAME_CHARS}{{2,8}}君[。、]\s*"  # PR47: 「赤澤亮正君。」等（ひらがな名前も対象）
    rf")"
)

# PR43 v3 / PR47: question.full_text 冒頭の質疑者識別ラベルを除去する。
# Pattern D (PR47 new): 全角コロンのみの bare name形式「西田英範：」「奥村祥大：」
# カタカナ名 (ラサール石井) にも対応。
_LEADING_QUESTIONER_LABEL_RE = re.compile(
    rf"(?:"
    rf"^{_JP_NAME_CHARS}{{2,10}}（[^）]{{2,40}}）(?:君|さん)?[。、：]\s*"  # name + (party)
    rf"|^{_JP_NAME_CHARS}{{2,8}}(?:君|さん)[。、：]\s*"                    # name + honorific
    rf"|^{_JP_NAME_CHARS}{{2,8}}：\s*"                                     # bare name + fullwidth colon
    r")"
)

# PR46: answer.full_text 内の「純粋な話者ラベル行」を除去する。
# 複数話者の発言が1answerに統合されたとき「\n砂原参考人。\n」等の行が挿入される。
# 行全体が話者ラベルのみの場合に限り除去（内容を含む行は保持）。
# PR47: カタカナ名前に対応
_PURE_LABEL_LINE_RE = re.compile(
    rf"^\s*(?:"
    rf"{_JP_NAME_CHARS}{{2,20}}?(?:内閣総理大臣|総理大臣|国務大臣|大臣政務官|副大臣|大臣|副長官|長官|次長|局長|審議官|参事官|部長|参考人)[。、：]"
    rf"|{_JP_NAME_CHARS}{{2,10}}（[^）]{{2,40}}）(?:君|さん)?[。、：]"
    rf"|{_JP_NAME_CHARS}{{2,8}}(?:委員長|議長|君|さん)[。、：]"
    r")\s*$"
)


def _strip_pure_label_lines(text: str) -> str:
    """answer.full_text 内の純粋な話者ラベル行を除去する (PR46)。

    行全体が「名前+役職/敬称+句点」のみで構成される行をスキップする。
    内容テキストを含む行は変更しない。
    """
    if '\n' not in text:
        return text
    lines = text.split('\n')
    result = [line for line in lines if not _PURE_LABEL_LINE_RE.match(line)]
    return '\n'.join(result).rstrip()


def _strip_trailing_speaker_label(text: str) -> str:
    """full_text 末尾の次発言者ラベルを除去する (PR43)。

    複数のラベルが積み重なっているケース（例: 「\n委員長\n次質疑者君。」）に対応するため
    最大 3 回まで繰り返し適用する。
    """
    for _ in range(3):
        stripped = _TRAILING_SPEAKER_LABEL_RE.sub("", text).rstrip()
        if stripped == text:
            break
        text = stripped
    return text


def _strip_leading_speaker_label(text: str) -> str:
    """answer.full_text 冒頭の話者ラベルを除去する (PR43)。

    「高市早苗内閣総理大臣。[答弁内容...]」→「[答弁内容...]」
    PR47: ダブルラベル echo (「林大臣。林芳正。」) に対応するため最大 3 回反復適用。
    """
    for _ in range(3):
        stripped = _LEADING_SPEAKER_LABEL_RE.sub("", text)
        if stripped == text:
            break
        text = stripped
    return text


def _strip_leading_questioner_label(text: str) -> str:
    """question.full_text 冒頭の質疑者識別ラベルを除去する (PR43 v3)。

    「森本真治（立憲民主・無所属）：[質問...]」→「[質問...]」
    """
    return _LEADING_QUESTIONER_LABEL_RE.sub("", text)


def _shift_video_url_time(video_url: str, new_start_seconds: float) -> str:
    """既存 video_url の時刻部分 (`time=` パラメータ or `#` ハッシュ) を差し替える。

    - shugiin: `?ex=VL&deli_id=XXX&time=1234.5` → `time=` を上書き
    - sangiin: `detail.php?sid=XXX#1234.5` → `#` 後を上書き

    パターンに合わない URL はそのまま返す。new_start_seconds は小数 1 桁に丸める
    (URL 短縮のため)。
    """
    if not video_url:
        return video_url
    rounded = f"{max(0.0, new_start_seconds):.1f}"
    if _VIDEO_TIME_PARAM_RE.search(video_url):
        return _VIDEO_TIME_PARAM_RE.sub(f"time={rounded}", video_url)
    if _VIDEO_HASH_TIME_RE.search(video_url):
        return _VIDEO_HASH_TIME_RE.sub(f"#{rounded}", video_url)
    return video_url


def _resolve_speaker_from_utterances(
    seg: SegmentUtterances,
    utterance_indices: list[int],
    speakers_lookup: dict[str, SpeakerInfo],
) -> tuple[str, str]:
    """utterance_indices から質疑者の speaker と party(=affiliation) を取得する。"""
    valid = [i for i in utterance_indices if 0 <= i < len(seg.utterances)]
    if not valid:
        return seg.segment_speaker, seg.segment_affiliation
    name = seg.utterances[valid[0]].speaker
    info = _fuzzy_lookup(name, speakers_lookup)
    if info:
        return info.name, info.affiliation
    return name, seg.segment_affiliation


# 政党名パターン: このいずれかが所属に含まれていれば議員（答弁者にはならない）
_PARTY_KEYWORDS = frozenset([
    "自由民主党", "立憲民主党", "日本維新の会", "公明党", "日本共産党",
    "国民民主党", "チームみらい", "参政党", "れいわ新選組", "日本保守党",
    "社会民主党", "中道改革連合", "無所属",
])


def _is_member_of_parliament(affiliation: str) -> bool:
    """所属から国会議員かどうかを判定する。委員長は議事進行なので議員扱いしない。"""
    if not affiliation:
        return False
    if affiliation.endswith("委員長") or affiliation.endswith("議長"):
        return False
    return any(p in affiliation for p in _PARTY_KEYWORDS)


def _answerer_role_from_info(info: SpeakerInfo | None, fallback_utt_role: str) -> str:
    """答弁者の role 文字列 (qa_pairs.answer.role) を info から取り出す。

    PR35: utterances/metadata の機能名 (答弁者/政府参考人/参考人) と統一するため、
    qa_pairs.answer.role も機能名カテゴリを返す。具体的な役職名 ("防衛大臣" 等) は
    metadata.speakers.affiliation に保持し、answer.role には用いない。
    優先順:
        1. info.role (機能名カテゴリ、例 "答弁者" / "政府参考人" / "参考人")
        2. fallback_utt_role (utterances 由来 role)
    """
    if info is None:
        return fallback_utt_role
    if info.role and info.role != "その他":
        return info.role
    return fallback_utt_role


def _resolve_answerer_from_utterances(
    seg: SegmentUtterances,
    utterance_indices: list[int],
    speakers_lookup: dict[str, SpeakerInfo],
) -> tuple[str, str]:
    """utterance_indices から答弁者の speaker と role(=affiliation/カテゴリ) を取得する。

    議員が答弁者として選ばれた場合は、同じ utterance 範囲内で「議員でない」発言者
    （大臣・政府参考人）を探す。見つからなければ議員のまま返す（ログで警告）。
    """
    valid = [i for i in utterance_indices if 0 <= i < len(seg.utterances)]
    if not valid:
        return "", ""

    head = valid[0]
    name = seg.utterances[head].speaker
    head_utt_role = seg.utterances[head].role
    info = _fuzzy_lookup(name, speakers_lookup)
    candidate_name = info.name if info else name
    candidate_aff = info.affiliation if info else ""
    candidate_role = _answerer_role_from_info(info, head_utt_role)

    if not _is_member_of_parliament(candidate_aff):
        return candidate_name, candidate_role

    for ui in valid:
        alt_name = seg.utterances[ui].speaker
        alt_info = _fuzzy_lookup(alt_name, speakers_lookup)
        alt_aff = alt_info.affiliation if alt_info else ""
        if alt_name != candidate_name and not _is_member_of_parliament(alt_aff):
            alt_role = _answerer_role_from_info(alt_info, seg.utterances[ui].role)
            logger.info(
                "Corrected answerer: %s (%s) → %s (%s)",
                candidate_name, candidate_role,
                alt_info.name if alt_info else alt_name, alt_role,
            )
            return (alt_info.name if alt_info else alt_name), alt_role

    logger.warning(
        "MP '%s' (%s) resolved as answerer but no non-MP alternative found in utterance range",
        candidate_name, candidate_role,
    )
    return candidate_name, candidate_role


# ---------------------------------------------------------------------------
# 案B: 委員長指名による質疑ブロック分割
# ---------------------------------------------------------------------------
# TVのタイムスタンプは質疑者単位だが、1セグメント内に複数の質疑者が
# 含まれることがある。委員長の指名発言（「次に〇〇君。」「〇〇君。」）を
# 境界として質疑ブロックに分割し、Q&A生成をブロック単位で行う。

_CHAIR_NOMINATION_RE = re.compile(
    r"^(?:次に)?(.+?)[君さ](?:ん)?[。.]?\s*$"
)


def _split_segment_into_blocks(seg: SegmentUtterances) -> list[SegmentUtterances]:
    """委員長の質疑者交代指名を境界としてセグメントを質疑ブロックに分割する。

    委員長が「次に〇〇君。」「〇〇君。」と発言している箇所を検出し、
    **その後の質疑者が直前ブロックの質疑者と異なる場合のみ**ブロックを分割する。
    同じ質疑者への続行指名（答弁後に「〇〇君。」と戻す）では分割しない。
    """
    # まず各 utterance の質疑者を特定するために、セグメント内の質疑者の変遷を追跡
    # 委員長指名の位置を検出
    candidate_splits: list[tuple[int, str]] = []  # (utterance_index_after_nomination, nominated_name)
    for i, u in enumerate(seg.utterances):
        if u.role != "委員長":
            continue
        text = u.text.strip()
        m = _CHAIR_NOMINATION_RE.match(text)
        if m and i + 1 < len(seg.utterances):
            nominated_name = m.group(1).strip()
            candidate_splits.append((i + 1, nominated_name))

    if not candidate_splits:
        return [seg]

    # 各候補分割点で、実際に質疑者が変わるかを確認
    def _find_questioner_before(idx: int) -> str:
        """idx より前の最後の質疑者名を返す。"""
        for j in range(idx - 1, -1, -1):
            if seg.utterances[j].role == "質疑者":
                return seg.utterances[j].speaker
        return ""

    def _find_questioner_after(idx: int) -> str:
        """idx 以降の最初の質疑者名を返す。"""
        for j in range(idx, len(seg.utterances)):
            if seg.utterances[j].role == "質疑者":
                return seg.utterances[j].speaker
        return ""

    split_points: list[int] = []
    for utt_idx, _nominated in candidate_splits:
        before = _find_questioner_before(utt_idx)
        after = _find_questioner_after(utt_idx)
        if before and after and before != after:
            split_points.append(utt_idx)

    if not split_points:
        return [seg]

    # 先頭が0でなければ追加（最初のブロック）
    if split_points[0] != 0:
        split_points = [0] + split_points

    split_points = sorted(set(split_points))

    if len(split_points) <= 1:
        return [seg]

    blocks: list[SegmentUtterances] = []
    for idx, start in enumerate(split_points):
        end = split_points[idx + 1] if idx + 1 < len(split_points) else len(seg.utterances)
        block_utterances = seg.utterances[start:end]
        if not block_utterances:
            continue

        questioners = [u.speaker for u in block_utterances if u.role == "質疑者"]
        block_speaker = questioners[0] if questioners else seg.segment_speaker

        blocks.append(SegmentUtterances(
            segment_index=seg.segment_index,
            segment_speaker=block_speaker,
            segment_affiliation=seg.segment_affiliation,
            start_seconds=seg.start_seconds,
            video_url=seg.video_url,
            utterances=block_utterances,
        ))

    if not blocks:
        return [seg]

    if len(blocks) > 1:
        logger.info(
            "Split segment %d (%s) into %d blocks: %s",
            seg.segment_index,
            seg.segment_speaker,
            len(blocks),
            ", ".join(b.segment_speaker for b in blocks),
        )

    return blocks


def _is_qa_segment(seg: SegmentUtterances) -> bool:
    """セグメントがQ&A抽出対象かどうかを判定する。

    質疑者の発言が含まれるセグメントのみ対象。
    議長の開会宣言や大臣の趣旨説明はスキップ。
    """
    has_questioner = any(u.role == "質疑者" for u in seg.utterances)
    if has_questioner:
        return True

    # 質疑者ロールがなくても、実質的な質疑が行われているセグメントを拾う
    # （role tagging が不完全な場合のフォールバック）
    roles = {u.role for u in seg.utterances}
    if "答弁者" in roles and roles - {"答弁者", "委員長"}:
        return True

    return False


# QA密度チェックの閾値
_MIN_QA_DENSITY = 0.5   # 1000文字あたり最低0.5ペア
_QA_DENSITY_RETRY_HINT = (
    "\n\n【注意】前回の抽出ではQ&Aペアが{prev_count}個しか得られませんでした（{total_chars}文字のセグメント）。"
    "roleラベルに頼らず発言内容から判断し、すべてのQ&Aペアを漏れなく抽出してください。"
)


def _extract_pairs_from_response(
    content: str | None,
    seg: SegmentUtterances,
    layout: _SegmentLayout,
    speakers_lookup: dict[str, SpeakerInfo],
) -> list[QAPair]:
    """LLM レスポンスをパースして QAPair リストを組み立てる。

    新スキーマ (utterance_indices + 任意 split_anchor_sentence_idx) を解釈し、
    full_text は `_assemble_full_text_for_pair` がコード側で組み立てる。
    """
    if not content:
        logger.warning("Empty response for segment %d", seg.segment_index)
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Some models (e.g. gemini-3-flash-preview via OpenRouter) emit raw
        # newlines / control chars inside JSON string values instead of escaping
        # them.  Repair by escaping bare control chars inside string literals.
        sanitized = _repair_json_string_controls(content)
        if sanitized != content:
            logger.warning(
                "Segment %d: repaired unescaped control chars in LLM JSON",
                seg.segment_index,
            )
        try:
            data = json.loads(sanitized)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON for segment %d: %s", seg.segment_index, e)
            return []
    raw_pairs = data.get("pairs", []) or []

    parsed_pairs: list[dict[str, Any]] = []
    for p in raw_pairs:
        q = p.get("question") or {}
        a = p.get("answer") or {}
        parsed_pairs.append({
            "topic": p.get("topic", "") or "",
            "q_uidx": [int(x) for x in (q.get("utterance_indices") or []) if isinstance(x, int)],
            "q_anchor": q.get("split_anchor_sentence_idx"),
            "q_summary": q.get("summary", "") or "",
            "q_intent": q.get("intent", "other") or "other",
            "a_uidx": [int(x) for x in (a.get("utterance_indices") or []) if isinstance(x, int)],
            "a_anchor": a.get("split_anchor_sentence_idx"),
            "a_summary": a.get("summary", "") or "",
        })

    q_boundaries = _compute_share_boundaries(parsed_pairs, "q", layout)
    a_boundaries = _compute_share_boundaries(parsed_pairs, "a", layout)

    n_utterances = len(seg.utterances)
    indices_total = 0
    indices_out_of_range = 0

    pairs: list[QAPair] = []
    dropped_short = 0
    dropped_short_q = 0
    dropped_empty_q = 0
    for i, p in enumerate(parsed_pairs):
        # 範囲外 utterance_indices の比率を計測 (LLM hallucination 検知)
        for side_uidx in (p["q_uidx"], p["a_uidx"]):
            for idx in side_uidx:
                indices_total += 1
                if idx < 0 or idx >= n_utterances:
                    indices_out_of_range += 1

        q_full = _assemble_full_text_for_pair(
            seg, layout, p["q_uidx"], p["q_anchor"], q_boundaries[i],
        )
        a_full = _assemble_full_text_for_pair(
            seg, layout, p["a_uidx"], p["a_anchor"], a_boundaries[i],
        )

        # PR43: answer / question 末尾に混入した次発言者ラベルを除去
        a_full = _strip_trailing_speaker_label(a_full)
        q_full = _strip_trailing_speaker_label(q_full)
        # PR43: answer 冒頭の話者ラベル、question 冒頭の質疑者ラベルを除去
        a_full = _strip_leading_speaker_label(a_full)
        q_full = _strip_leading_questioner_label(q_full)
        # PR46: answer 内の純粋な話者ラベル行を除去（「\n砂原参考人。\n」等）
        a_full = _strip_pure_label_lines(a_full)

        if len(a_full) < MIN_ANSWER_LENGTH and not p["a_uidx"]:
            dropped_short += 1
            continue
        # PR39: 質問 utterance の合計文字数が短すぎるペアを drop (挨拶のみ・数語ダミー対策)。
        # 共有 utterance では assembled q_full が短くなることがあるため、
        # utterance 自体の合計文字数で判定する (q_uidx が空の場合は次の空チェックに委ねる)
        if p["q_uidx"]:
            q_total_chars = sum(
                len(seg.utterances[i].text)
                for i in p["q_uidx"]
                if 0 <= i < n_utterances
            )
            if q_total_chars < MIN_ANSWER_LENGTH:
                dropped_short_q += 1
                continue
        # 質問本文が空かつ utterance_indices も空なら Q&A として成立しない (PR10, ISSUES2 §1-2)
        if not q_full and not p["q_uidx"]:
            dropped_empty_q += 1
            continue

        q_speaker, q_party = _resolve_speaker_from_utterances(seg, p["q_uidx"], speakers_lookup)
        a_speaker, a_role = _resolve_answerer_from_utterances(seg, p["a_uidx"], speakers_lookup)

        # PR23 + PR23.1: 質問先頭 utterance + anchor sentence の文字位置から
        # 開始秒を推定し video_url を補正。同一 head_utt を共有する複数ペアでも
        # 異なる時刻を指せるようにする。q_anchor は PR28 で推定された anchor も
        # 含む (parsed_pairs[i]["q_anchor"] は _compute_share_boundaries で更新済)。
        offset = _estimate_pair_offset_seconds(
            seg, layout, p["q_uidx"], p["q_anchor"]
        )
        if offset > 0.0:
            pair_video_url = _shift_video_url_time(seg.video_url, seg.start_seconds + offset)
        else:
            pair_video_url = seg.video_url

        pairs.append(
            QAPair(
                id="",  # 後でマージ時に付番
                segment_index=seg.segment_index,
                topic=p["topic"],
                question=QuestionDetail(
                    speaker=q_speaker,
                    party=q_party,
                    summary=p["q_summary"],
                    full_text=q_full,
                    intent=p["q_intent"],
                ),
                answer=AnswerDetail(
                    speaker=a_speaker,
                    role=a_role,
                    summary=p["a_summary"],
                    full_text=a_full,
                ),
                video_url=pair_video_url,
            )
        )

    # 範囲外 indices 比率が 50% 超なら LLM が utterance 番号を hallucinate している
    # 可能性が高い (PR10, §2.10)
    if indices_total > 0:
        oor_ratio = indices_out_of_range / indices_total
        if oor_ratio > 0.5:
            logger.warning(
                "Segment %d: %d/%d (%.0f%%) utterance_indices out of range — "
                "LLM may have hallucinated indices",
                seg.segment_index,
                indices_out_of_range,
                indices_total,
                oor_ratio * 100,
            )

    # 受理/drop 統計サマリ (PR10, §2.10)
    logger.info(
        "Segment %d: parsed %d raw → kept %d pairs "
        "(drop_short_a=%d, drop_short_q=%d, drop_empty_q=%d, oor_idx=%d/%d)",
        seg.segment_index,
        len(parsed_pairs),
        len(pairs),
        dropped_short,
        dropped_short_q,
        dropped_empty_q,
        indices_out_of_range,
        indices_total,
    )
    return pairs


def _generate_qa_for_segment(
    seg: SegmentUtterances,
    session_context: str,
    speakers_lookup: dict[str, SpeakerInfo],
) -> list[QAPair]:
    """1セグメントから Q&A ペアを生成する。

    LLM には utterance_indices と判断のみ返させ、full_text はコードで組み立てる。
    QA 密度が低い場合は 1 回リトライする。
    """
    client = _get_client()

    layout = _compute_segment_layout(seg)
    prompt_text = _build_utterance_map(seg, layout)
    total_chars = sum(len(u.text) for u in seg.utterances)

    if len(prompt_text) > _PROMPT_TEXT_WARN_THRESHOLD:
        logger.warning(
            "Segment %d prompt text very large (%d chars). "
            "Block-splitting may be insufficient — investigate.",
            seg.segment_index,
            len(prompt_text),
        )

    base_user_prompt = (
        f"以下は国会質疑の1つの発言セグメントです。"
        f"この発言者の持ち時間で行われた質疑応答を**すべて**Q&Aペアとして抽出してください。\n"
        f"utterance_indices には入力の [Un] の番号を使ってください。"
        f"split_anchor_sentence_idx は通常 null。1つの utterance を複数ペアで共有する場合のみ "
        f"(sN) のグローバル番号を入れてください。\n\n"
        f"セッション情報: {session_context}\n\n"
        f"{prompt_text}"
    )

    logger.info(
        "Generating Q&A pairs for segment %d: %s (%d utterances, %d chars)",
        seg.segment_index,
        seg.segment_speaker,
        len(seg.utterances),
        total_chars,
    )

    def _qa_call(prompt: str, tokens: int, temperature: float) -> Any:
        return with_retry(lambda: client.chat.completions.create(
            model=QA_MODEL,
            messages=[
                {"role": "system", "content": QA_SEGMENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=tokens,
            response_format={"type": "json_object"},
        ))

    response = _qa_call(base_user_prompt, _MAX_TOKENS_CEILING, 0.1)
    if response.choices[0].finish_reason == "length":
        logger.error(
            "Q&A output still truncated for segment %d at max_tokens=%d",
            seg.segment_index,
            _MAX_TOKENS_CEILING,
        )

    pairs = _extract_pairs_from_response(
        response.choices[0].message.content, seg, layout, speakers_lookup,
    )

    if total_chars >= 2000:
        density = len(pairs) / (total_chars / 1000)
        if density < _MIN_QA_DENSITY:
            logger.warning(
                "Low Q&A density for segment %d (%s): "
                "%d pairs / %d chars (density=%.2f < %.2f), retrying",
                seg.segment_index, seg.segment_speaker,
                len(pairs), total_chars, density, _MIN_QA_DENSITY,
            )
            retry_hint = _QA_DENSITY_RETRY_HINT.format(
                prev_count=len(pairs), total_chars=total_chars,
            )
            retry_prompt = base_user_prompt + retry_hint

            retry_response = _qa_call(retry_prompt, _MAX_TOKENS_CEILING, 0.3)
            if retry_response.choices[0].finish_reason == "length":
                logger.warning(
                    "Q&A density-retry output truncated for segment %d at max_tokens=%d",
                    seg.segment_index,
                    _MAX_TOKENS_CEILING,
                )

            retry_pairs = _extract_pairs_from_response(
                retry_response.choices[0].message.content, seg, layout, speakers_lookup,
            )

            if len(retry_pairs) > len(pairs):
                logger.info(
                    "Retry improved segment %d: %d → %d pairs",
                    seg.segment_index, len(pairs), len(retry_pairs),
                )
                pairs = retry_pairs
            else:
                logger.info(
                    "Retry did not improve segment %d: %d → %d pairs, keeping original",
                    seg.segment_index, len(pairs), len(retry_pairs),
                )

    logger.info(
        "Segment %d (%s): extracted %d Q&A pairs",
        seg.segment_index,
        seg.segment_speaker,
        len(pairs),
    )
    return pairs


def generate_qa_pairs(
    utterances: UtterancesOutput,
    speakers: list[SpeakerInfo] | None = None,
    max_workers: int = 16,
    skip_proposal_segments: bool = False,
) -> QAPairsOutput:
    """全セグメントからQ&Aペアを生成する（セグメント単位で並列処理）。

    Args:
        utterances: 話者タグ付き発言データ
        speakers: metadata.jsonのspeakers（名前→役職の解決に使用）
        max_workers: 並列数
        skip_proposal_segments: True なら、最初の質疑者 segment より前の答弁者 segment を
            Q&A 抽出から除外する（本会議代表質問の冒頭の趣旨説明スキップ用）。
    """

    # speakers_lookup: 名前 → SpeakerInfo
    speakers_lookup: dict[str, SpeakerInfo] = {}
    if speakers:
        for s in speakers:
            speakers_lookup[s.name] = s

    # セッションコンテキスト（各LLM呼び出しに共有）
    all_speakers = set()
    for seg in utterances.segments:
        for u in seg.utterances:
            all_speakers.add(u.speaker)
    session_context = f"発言者: {', '.join(sorted(all_speakers))}"

    target_segments = (
        _drop_leading_proposal_segments(utterances.segments)
        if skip_proposal_segments
        else utterances.segments
    )

    # Q&A対象セグメントを選別し、委員長指名で質疑ブロックに分割
    qa_blocks: list[SegmentUtterances] = []
    skipped = 0
    for seg in target_segments:
        if not _is_qa_segment(seg):
            skipped += 1
            continue
        blocks = _split_segment_into_blocks(seg)
        for block in blocks:
            if _is_qa_segment(block):
                qa_blocks.append(block)

    logger.info(
        "Processing %d Q&A blocks (from %d segments, skipped %d procedural)",
        len(qa_blocks),
        len(utterances.segments),
        skipped,
    )

    if not qa_blocks:
        return QAPairsOutput(pairs=[])

    # ブロック単位で並列にLLM呼び出し
    block_results: list[tuple[int, int, list[QAPair]]] = []  # (seg_index, block_order, pairs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_qa_for_segment, block, session_context, speakers_lookup): (i, block)
            for i, block in enumerate(qa_blocks)
        }
        for future in as_completed(futures):
            block_order, block = futures[future]
            try:
                pairs = future.result()
                block_results.append((block.segment_index, block_order, pairs))
            except Exception as e:
                logger.error(
                    "Failed to generate Q&A for block %d (%s): %s",
                    block.segment_index,
                    block.segment_speaker,
                    e,
                )
                block_results.append((block.segment_index, block_order, []))

    # セグメント順 → ブロック順にマージし、通し番号を付与
    block_results.sort(key=lambda x: (x[0], x[1]))
    all_pairs: list[QAPair] = []
    pair_counter = 0
    for _seg_idx, _block_order, pairs in block_results:
        for pair in pairs:
            pair_counter += 1
            pair.id = f"qa_{pair_counter:03d}"
            all_pairs.append(pair)

    # PR41: segment_index 境界誤帰属修正 — question.speaker が segment_speaker と不一致な
    # ペアを正しい segment に再帰属し、video_url を補正する
    _fix_boundary_mispairs(all_pairs, utterances.segments)

    # PR13: follow_up_ids — 同一 segment 内で同一質疑者の連続ペアを連鎖
    _assign_follow_up_ids(all_pairs)

    logger.info("Total Q&A pairs generated: %d (from %d blocks)", len(all_pairs), len(qa_blocks))
    return QAPairsOutput(pairs=all_pairs)


def _fix_boundary_mispairs(
    pairs: list[QAPair],
    segments: list[SegmentUtterances],
) -> None:
    """PR41: segment 境界誤帰属修正 (in-place)。

    LLM が前セグメントの末尾ブロックを次セグメントの先頭ペアとして返すとき、
    pair.question.speaker が seg.segment_speaker と一致しないケースが発生する。
    正しい segment を speaker 名で検索し、segment_index と video_url を補正する。
    """
    seg_by_idx = {s.segment_index: s for s in segments}
    # 同名が複数 segment にまたがる場合は最初のものを採用
    speaker_to_seg: dict[str, SegmentUtterances] = {}
    for s in segments:
        if s.segment_speaker not in speaker_to_seg:
            speaker_to_seg[s.segment_speaker] = s

    for pair in pairs:
        current_seg = seg_by_idx.get(pair.segment_index)
        if current_seg is None:
            continue
        q_speaker = pair.question.speaker.strip()
        if not q_speaker or q_speaker == current_seg.segment_speaker:
            continue
        correct_seg = speaker_to_seg.get(q_speaker)
        if correct_seg is None:
            continue
        logger.info(
            "PR41 boundary fix: pair segment %d (%s) → %d (%s) for speaker '%s'",
            pair.segment_index, current_seg.segment_speaker,
            correct_seg.segment_index, correct_seg.segment_speaker,
            q_speaker,
        )
        pair.segment_index = correct_seg.segment_index
        pair.video_url = correct_seg.video_url


def _assign_follow_up_ids(pairs: list[QAPair]) -> None:
    """同一 segment 内で同一質疑者の qa_pairs が時系列で 2 件以上ある場合、
    後続ペアの follow_up_ids に直前ペアの id を入れる (in-place)。

    判定基準 (docs/STRUCTURER_REWRITE.md §6 Q):
    - 同一 segment_index 内で時系列順 (= pairs リストの出現順) に走査
    - question.speaker が同一 (空文字でない) なら、直前同一 speaker ペアの id を follow_up_ids 先頭に追加
    - 別 segment や別 speaker は別質疑ブロック扱いで連鎖しない
    """
    last_id_by_key: dict[tuple[int, str], str] = {}
    for p in pairs:
        speaker = p.question.speaker.strip()
        if not speaker:
            continue
        key = (p.segment_index, speaker)
        prev_id = last_id_by_key.get(key)
        if prev_id and prev_id not in p.follow_up_ids:
            p.follow_up_ids = [prev_id, *p.follow_up_ids]
        last_id_by_key[key] = p.id


def _drop_leading_proposal_segments(
    segments: list[SegmentUtterances],
) -> list[SegmentUtterances]:
    """最初の質疑者 segment より前の答弁者 segment を除外する（代表質問の趣旨説明スキップ用）。"""
    first_questioner_idx = next(
        (
            i
            for i, seg in enumerate(segments)
            if any(u.role == "質疑者" for u in seg.utterances)
        ),
        None,
    )
    if first_questioner_idx is None:
        return segments
    if first_questioner_idx == 0:
        return segments
    leading = segments[:first_questioner_idx]
    is_only_proposal = all(
        all(u.role == "答弁者" for u in seg.utterances) for seg in leading if seg.utterances
    )
    if not is_only_proposal:
        return segments
    return segments[first_questioner_idx:]


def _format_qa_pairs_for_prompt(qa_pairs: QAPairsOutput) -> str:
    lines = []
    for p in qa_pairs.pairs:
        role_part = f"（{p.answer.role}）" if p.answer.role else ""
        lines.append(
            f"[{p.id}] トピック: {p.topic}\n"
            f"  質問者: {p.question.speaker}（{p.question.party}）\n"
            f"  質問要旨: {p.question.summary}\n"
            f"  回答者: {p.answer.speaker}{role_part}\n"
            f"  回答要旨: {p.answer.summary}"
        )
    return "\n".join(lines)


_MAX_TOKENS_CEILING = 16384


def _call_structurer(
    system_prompt: str, user_prompt: str, *, max_tokens: int, temperature: float = 0.1
) -> dict:
    client = _get_client()

    def _do_call(tokens: int) -> Any:
        return with_retry(
            lambda: client.chat.completions.create(
                model=STRUCTURER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=tokens,
                response_format={"type": "json_object"},
            )
        )

    response = _do_call(max_tokens)
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "length":
        retry_tokens = min(max_tokens * 2, _MAX_TOKENS_CEILING)
        logger.warning(
            "Output truncated (finish_reason=length, max_tokens=%d), retrying with max_tokens=%d",
            max_tokens,
            retry_tokens,
        )
        response = _do_call(retry_tokens)
        if response.choices[0].finish_reason == "length":
            raise ValueError(
                f"Output still truncated after retry with max_tokens={retry_tokens}"
            )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from LLM")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM JSON: {e}") from e
    return data


# PR12: session_summary 内で「<人名>大臣」「<人名>議員」等の honorific-attached 人名を抽出する正規表現
# qa_pairs に登場しない人名を要約が含むケース (summary_qa_divergence) を検出するために使う
_SUMMARY_PERSON_REF_RE = re.compile(
    r"([一-龥々ヶ]{1,8})"
    r"(?:大臣|副大臣|総理|長官|次官|議員|委員長|議長|参考人|政務官|氏|君|さん)"
)


def _collect_known_speaker_names(qa_pairs: QAPairsOutput) -> set[str]:
    """qa_pairs から既知 speaker 名 (question/answer 双方) を集める。"""
    known: set[str] = set()
    for p in qa_pairs.pairs:
        if p.question.speaker:
            known.add(p.question.speaker.strip())
        if p.answer.speaker:
            known.add(p.answer.speaker.strip())
    return {n for n in known if n}


def _validate_summary_person_refs(
    summary: str,
    qa_pairs: QAPairsOutput,
) -> list[str]:
    """summary 内の honorific-attached 人名で qa_pairs にないものを返す。

    既知 speaker 名と部分一致 (どちらかが substring) すれば known 扱い。
    qa_pairs が空の場合は検証しない (空リストを返す)。
    """
    if not qa_pairs.pairs:
        return []
    known = _collect_known_speaker_names(qa_pairs)
    if not known:
        return []
    refs = {m.group(1) for m in _SUMMARY_PERSON_REF_RE.finditer(summary)}
    unknown: list[str] = []
    for ref in sorted(refs):
        if any(ref in name or name in ref for name in known):
            continue
        unknown.append(ref)
    return unknown


_PLACEHOLDER_HEADER_PATTERNS = (
    "委員会名不明",
    "（委員会名不明）",
    "(委員会名不明)",
    "○○委員会",
    "〇〇委員会",
    "XX委員会",
)


def _has_placeholder_header(summary: str) -> bool:
    """summary 冒頭付近に「委員会名不明」等のプレースホルダ表現が含まれているか判定。"""
    head = summary[:120]
    return any(pat in head for pat in _PLACEHOLDER_HEADER_PATTERNS)


def _has_chamber_mismatch(summary: str, expected_chamber_ja: str) -> bool:
    """冒頭で期待した院と異なる院名が現れたら True。

    例: expected「参議院」だが summary 冒頭に「衆議院」が出てくるケースを検出。
    """
    if not expected_chamber_ja:
        return False
    head = summary[:80]
    other = "衆議院" if expected_chamber_ja == "参議院" else "参議院"
    if other in head and expected_chamber_ja not in head:
        return True
    return False


def generate_session_summary(
    qa_pairs: QAPairsOutput,
    utterances: UtterancesOutput | None = None,
    session_meta: dict | None = None,
) -> str:
    """セッション要約（3-5文）を生成する（Step 6b-1）。

    session_meta で院名・委員会名を渡すと冒頭の種別表記が正確になる。
    使用可能なキー: chamber ("shugiin"|"sangiin"), committee, session_kind, description
    """
    if qa_pairs.pairs:
        body = "## Q&Aペア一覧\n" + _format_qa_pairs_for_prompt(qa_pairs)
    elif utterances is not None and utterances.segments:
        body = "## 発言セグメント\n" + _format_segments_for_prompt(utterances.segments)
    elif session_meta and session_meta.get("description"):
        body = "## セッション概要\n" + session_meta["description"]
    else:
        return ""

    meta_prefix = ""
    expected_chamber_ja = ""
    expected_committee = ""
    if session_meta:
        chamber_raw = session_meta.get("chamber", "")
        expected_committee = session_meta.get("committee", "") or ""
        expected_chamber_ja = (
            "衆議院" if chamber_raw == "shugiin"
            else "参議院" if chamber_raw == "sangiin"
            else chamber_raw or ""
        )
        parts = []
        if expected_chamber_ja:
            parts.append(f"院: {expected_chamber_ja}")
        if expected_committee:
            parts.append(f"委員会: {expected_committee}")
        if parts:
            meta_prefix = "## セッション情報\n" + "\n".join(parts) + "\n\n"

    user_prompt = "以下の国会セッションの内容から、概要を作成してください。\n\n" + meta_prefix + body
    data = _call_structurer(SESSION_SUMMARY_SYSTEM_PROMPT, user_prompt, max_tokens=4096)
    summary = data.get("session_summary", "")
    if not isinstance(summary, str):
        return ""
    summary = summary.strip()

    # PR21: 冒頭にプレースホルダが残っている / 院が取り違えられている場合は 1 回リトライ
    header_issue: str | None = None
    if _has_placeholder_header(summary):
        header_issue = (
            "前回の要約の冒頭に「委員会名不明」等のプレースホルダ表現が残っていました。"
        )
    elif _has_chamber_mismatch(summary, expected_chamber_ja):
        header_issue = (
            f"前回の要約の冒頭が「{expected_chamber_ja}」ではない院を指していました。"
        )
    if header_issue and meta_prefix:
        logger.warning("generate_session_summary: header issue detected — retrying once")
        retry_prompt = (
            user_prompt
            + "\n\n## 注意（再生成）\n"
            + header_issue
            + f"冒頭の一文には「## セッション情報」の値（院: {expected_chamber_ja}"
            + (f"、委員会: {expected_committee}" if expected_committee else "")
            + "）をそのまま使い、プレースホルダや別の院名を出力しないでください。"
        )
        try:
            retry_data = _call_structurer(
                SESSION_SUMMARY_SYSTEM_PROMPT, retry_prompt, max_tokens=4096
            )
            retry_summary = retry_data.get("session_summary", "")
            if isinstance(retry_summary, str) and retry_summary.strip():
                retry_summary = retry_summary.strip()
                if (
                    not _has_placeholder_header(retry_summary)
                    and not _has_chamber_mismatch(retry_summary, expected_chamber_ja)
                ):
                    summary = retry_summary
                else:
                    logger.warning(
                        "generate_session_summary: header issue persisted after retry"
                    )
                    summary = retry_summary  # 元出力よりはマシなので採用
        except Exception as e:
            logger.warning("generate_session_summary header retry failed: %s", e)

    # PR12: post-validation — qa_pairs にない人名が含まれていれば 1 回だけリトライ
    unknown_refs = _validate_summary_person_refs(summary, qa_pairs)
    if unknown_refs:
        logger.warning(
            "generate_session_summary: detected unknown person refs not in qa_pairs: %s — retrying once",
            unknown_refs,
        )
        retry_prompt = (
            user_prompt
            + "\n\n## 注意（再生成）\n"
            + "前回の要約に Q&A ペアに登場しない人名"
            + f" ({', '.join(unknown_refs)}) が含まれていました。"
            + "qa_pairs の `質問者:` および `回答者:` に登場する人物のみを言及し、"
            + "推測による人名・法案名の補完を行わないでください。"
        )
        try:
            retry_data = _call_structurer(
                SESSION_SUMMARY_SYSTEM_PROMPT, retry_prompt, max_tokens=4096
            )
            retry_summary = retry_data.get("session_summary", "")
            if isinstance(retry_summary, str) and retry_summary.strip():
                retry_summary = retry_summary.strip()
                still_unknown = _validate_summary_person_refs(retry_summary, qa_pairs)
                if still_unknown:
                    logger.warning(
                        "generate_session_summary: retry still has unknown refs %s "
                        "— using retry output anyway",
                        still_unknown,
                    )
                summary = retry_summary
        except Exception as e:
            logger.warning("generate_session_summary retry failed: %s", e)

    return summary


def _parse_topics_data(
    data: dict,
    valid_qa_ids: set[str],
) -> tuple[list[Topic], list[str]]:
    """LLM 出力の topics データを Topic リストに変換する。"""
    topics_list: list[Topic] = []
    for t in data.get("topics", []):
        related_qa_ids = [q for q in (t.get("related_qa_ids") or []) if q in valid_qa_ids]
        related_speakers = [s for s in (t.get("related_speakers") or []) if isinstance(s, str)]
        topics_list.append(Topic(
            name=t.get("name") or "",
            description=t.get("description") or "",
            related_qa_ids=related_qa_ids,
            related_speakers=related_speakers,
        ))
    raw_key_topics = data.get("key_topics") or []
    key_topics = [n for n in raw_key_topics if isinstance(n, str) and n]
    return topics_list, key_topics


def generate_topics_and_key_topics(
    qa_pairs: QAPairsOutput,
) -> tuple[TopicsOutput, list[str]]:
    """topics + key_topics を生成する（Step 6b-2）。

    全 QA ペアを一括で LLM に渡す。
    key_topics は topics[].name のサブセットになるよう post-validate する。
    """
    if not qa_pairs.pairs:
        return TopicsOutput(topics=[]), []

    valid_qa_ids = {p.id for p in qa_pairs.pairs}
    user_prompt = "## Q&Aペア一覧\n" + _format_qa_pairs_for_prompt(qa_pairs)
    data = _call_structurer(TOPICS_SYSTEM_PROMPT, user_prompt, max_tokens=_MAX_TOKENS_CEILING)
    topics_list, raw_key_topics = _parse_topics_data(data, valid_qa_ids)

    valid_topic_names = {t.name for t in topics_list}
    key_topics: list[str] = []
    dropped: list[str] = []
    for name in raw_key_topics:
        if isinstance(name, str) and name in valid_topic_names:
            key_topics.append(name)
        else:
            dropped.append(str(name))
    if dropped:
        logger.warning(
            "generate_topics: dropped %d key_topics not found in topics[].name: %s",
            len(dropped),
            dropped,
        )

    return TopicsOutput(topics=topics_list), key_topics


def generate_topics_without_qa(
    utterances: UtterancesOutput,
) -> tuple[TopicsOutput, list[str]]:
    """utterances から直接 topics + key_topics を生成する (Step 6b-2 fallback)。

    Q&A が抽出されない floor_speech / procedural セッション (所信表明・施政方針演説等)
    でも topics を保持できるようにする。related_qa_ids は空 (QA が存在しないため)。

    Args:
        utterances: 話者タグ付き発言データ

    Returns:
        (topics, key_topics) のペア。utterances が空なら空タプル相当。
    """
    if not utterances.segments:
        return TopicsOutput(topics=[]), []

    user_prompt = (
        "## 発言セグメント (連続発言・所信表明・施政方針演説・趣旨説明等)\n"
        + _format_segments_for_prompt(utterances.segments)
    )
    data = _call_structurer(
        TOPICS_FROM_UTTERANCES_SYSTEM_PROMPT,
        user_prompt,
        max_tokens=_MAX_TOKENS_CEILING,
    )

    topics_list: list[Topic] = []
    for t in data.get("topics", []):
        related_speakers = [
            s for s in (t.get("related_speakers") or []) if isinstance(s, str)
        ]
        topics_list.append(
            Topic(
                name=t.get("name") or "",
                description=t.get("description") or "",
                related_qa_ids=[],
                related_speakers=related_speakers,
            )
        )

    valid_topic_names = {t.name for t in topics_list}
    raw_key_topics = data.get("key_topics") or []
    key_topics: list[str] = []
    dropped: list[str] = []
    for name in raw_key_topics:
        if isinstance(name, str) and name in valid_topic_names:
            key_topics.append(name)
        else:
            dropped.append(str(name))
    if dropped:
        logger.warning(
            "generate_topics_without_qa: dropped %d key_topics not in topics[].name: %s",
            len(dropped),
            dropped,
        )

    return TopicsOutput(topics=topics_list), key_topics


def _parse_commitments_payload(
    raw: dict,
    qa_pair_lookup: dict[str, QAPair],
) -> tuple[list[KeyCommitment], int, int]:
    """LLM 応答から key_commitments を構築 + (qa_id, speaker) 整合検証。

    Returns:
        (commitments, dropped_unknown_qa_id, dropped_speaker_mismatch)
    """
    commitments: list[KeyCommitment] = []
    dropped_qa_id = 0
    dropped_speaker = 0
    for c in raw.get("key_commitments", []) or []:
        qa_id = (c.get("qa_id") or "").strip()
        speaker = (c.get("speaker") or "").strip()
        if qa_id and qa_id not in qa_pair_lookup:
            dropped_qa_id += 1
            continue
        # PR12: qa_id が指定されていれば speaker と回答者の整合を検証
        if qa_id and speaker:
            expected = qa_pair_lookup[qa_id].answer.speaker.strip()
            if expected and not (speaker in expected or expected in speaker):
                dropped_speaker += 1
                continue
        commitments.append(
            KeyCommitment(
                speaker=speaker,
                role=c.get("role") or "",
                text=c.get("text") or "",
                topic=c.get("topic") or "",
                qa_id=qa_id or None,
            )
        )
    return commitments, dropped_qa_id, dropped_speaker


def generate_key_commitments(qa_pairs: QAPairsOutput) -> list[KeyCommitment]:
    """key_commitments を生成する（Step 6b-3）。

    PR12: 各 commitment の (qa_id, speaker) ペアが qa_pairs と整合するか検証し、
    不一致は drop。drop 率が高い (受理 0 件 + raw > 0) 場合は 1 回だけリトライする。
    """
    if not qa_pairs.pairs:
        return []

    qa_pair_lookup = {p.id: p for p in qa_pairs.pairs}

    user_prompt = "## Q&Aペア一覧\n" + _format_qa_pairs_for_prompt(qa_pairs)
    data = _call_structurer(COMMITMENTS_SYSTEM_PROMPT, user_prompt, max_tokens=8192)

    raw_count = len(data.get("key_commitments", []) or [])
    commitments, dropped_qa_id, dropped_speaker = _parse_commitments_payload(
        data, qa_pair_lookup
    )

    if dropped_qa_id:
        logger.warning(
            "generate_key_commitments: dropped %d commitments referencing unknown qa_id",
            dropped_qa_id,
        )
    if dropped_speaker:
        logger.warning(
            "generate_key_commitments: dropped %d commitments with speaker mismatched against qa_pair.answer.speaker",
            dropped_speaker,
        )

    # PR12: raw が 0 でないのに全て drop されたら 1 回リトライ
    if raw_count > 0 and not commitments:
        logger.info(
            "generate_key_commitments: all %d commitments dropped — retrying once with stronger guidance",
            raw_count,
        )
        retry_prompt = (
            user_prompt
            + "\n\n## 注意（再生成）\n"
            + "speakerは入力 Q&A 一覧の `回答者:` フィールドを正確に転記し、"
            + "qa_id とその Q&A の回答者が必ず一致するようにしてください。"
            + "回答者が不明確な発言からはコミットメントを抽出しないでください。"
        )
        try:
            retry_data = _call_structurer(
                COMMITMENTS_SYSTEM_PROMPT, retry_prompt, max_tokens=8192
            )
            commitments, dropped_qa_id_r, dropped_speaker_r = _parse_commitments_payload(
                retry_data, qa_pair_lookup
            )
            if dropped_qa_id_r or dropped_speaker_r:
                logger.warning(
                    "generate_key_commitments retry: still dropped %d unknown_qa_id + %d speaker_mismatch",
                    dropped_qa_id_r,
                    dropped_speaker_r,
                )
        except Exception as e:
            logger.warning("generate_key_commitments retry failed: %s", e)

    return commitments


def tag_related_laws(
    qa_pairs: QAPairsOutput,
    *,
    chamber: str,
    committee: str,
    date: str,
    laws_text: str,
    max_workers: int = 16,
) -> QAPairsOutput:
    """各 Q&A ペアに対し関連法案 ID を判定し、QAPair.related_law_ids に書き戻す（Step 6c）。"""
    if not qa_pairs.pairs or not laws_text:
        return qa_pairs

    chamber_ja = "衆議院" if chamber == "shugiin" else "参議院" if chamber == "sangiin" else chamber
    context = (
        f"## セッション情報\n"
        f"院: {chamber_ja}\n"
        f"委員会: {committee}\n"
        f"日付: {date}\n\n"
        f"## 法案一覧（このセッションで議論される可能性が高い順に提示）\n{laws_text}\n\n"
    )

    def tag_one(pair: QAPair) -> tuple[str, list[str]]:
        user_prompt = (
            context
            + "## 対象 Q&A ペア\n"
            + f"id: {pair.id}\n"
            + f"トピック: {pair.topic}\n"
            + f"質問: {pair.question.summary}\n"
            + f"答弁: {pair.answer.summary}"
        )
        try:
            data = _call_structurer(LAW_TAGGING_SYSTEM_PROMPT, user_prompt, max_tokens=512)
        except Exception as e:
            logger.warning("tag_related_laws failed for %s: %s", pair.id, e)
            return pair.id, []
        raw = data.get("law_ids") or []
        return pair.id, [law for law in raw if isinstance(law, str) and law]

    results: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(tag_one, p): p.id for p in qa_pairs.pairs}
        for future in as_completed(futures):
            pid, law_ids = future.result()
            results[pid] = law_ids

    for pair in qa_pairs.pairs:
        pair.related_law_ids = results.get(pair.id, [])

    tagged = sum(1 for p in qa_pairs.pairs if p.related_law_ids)
    logger.info(
        "tag_related_laws: tagged %d/%d pairs with related laws", tagged, len(qa_pairs.pairs)
    )
    return qa_pairs


def build_summary_related_laws(qa_pairs: QAPairsOutput) -> list[RelatedLawTag]:
    """qa_pairs.related_law_ids を集約して summary.related_laws を作る（幽霊タグは drop）。"""
    by_law: dict[str, list[str]] = {}
    for pair in qa_pairs.pairs:
        for law_id in pair.related_law_ids:
            by_law.setdefault(law_id, []).append(pair.id)
    return [
        RelatedLawTag(law_id=law_id, qa_ids=qa_ids)
        for law_id, qa_ids in by_law.items()
        if qa_ids
    ]


# V4 metrics LLM: OpenRouter 経由 Gemma 4 31B-it
_METRICS_MODEL = "google/gemma-4-31b-it"

# PR45: LLM が混同しやすい型の許容値セット (Literal から外れた値を coerce する)
_VALID_CONCRETE_ITEM_TYPES: frozenset[str] = frozenset(
    ("number", "proper_noun", "deadline", "evidence_citation")
)
_VALID_CITED_SOURCE_TYPES: frozenset[str] = frozenset(
    ("number", "organization", "law", "date", "past_answer", "field_case", "other")
)


def _sanitize_metrics_data(data: dict) -> None:
    """LLM 出力の metrics JSON を Pydantic Literal 制約に合わせて修正する (in-place)。

    ConcreteItem.type / CitedSource.type が想定外の値を返した場合に最近似値へ coerce。
    """
    # ConcreteItem.type の coerce
    as2 = (data.get("as2_information_density") or {})
    for item in (as2.get("concrete_items_in_answer") or []):
        if isinstance(item, dict) and item.get("type") not in _VALID_CONCRETE_ITEM_TYPES:
            # "date"/"organization"/"law" → "evidence_citation" (closest semantically)
            raw_type = str(item.get("type", ""))
            if raw_type in ("date", "deadline"):
                item["type"] = "deadline"
            elif raw_type in ("organization", "law", "past_answer", "field_case"):
                item["type"] = "evidence_citation"
            else:
                item["type"] = "proper_noun"

    # CitedSource.type の coerce
    qq2 = (data.get("qq2_groundedness") or {})
    for item in (qq2.get("cited_sources") or []):
        if isinstance(item, dict) and item.get("type") not in _VALID_CITED_SOURCE_TYPES:
            item["type"] = "other"


def _score_one_pair(pair: QAPair) -> QAMetrics | None:
    """1ペアをV4プロンプトで評価してQAMetricsを返す。失敗時はNoneを返す。"""
    if not pair.question.full_text or not pair.answer.full_text:
        logger.warning("score_qa_pairs_metrics: skipping %s (empty text)", pair.id)
        return None

    user_msg = QA_METRICS_V4_USER_TEMPLATE.format(
        intent=pair.question.intent or "other",
        question_text=pair.question.full_text,
        answer_text=pair.answer.full_text,
    )

    client = _get_client()

    def _call() -> dict:
        response = client.chat.completions.create(
            model=_METRICS_MODEL,
            messages=[
                {"role": "system", "content": QA_METRICS_V4_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return json.loads(raw)

    try:
        data = with_retry(_call)
        try:
            return QAMetrics.model_validate(data)
        except Exception:
            # PR45: Pydantic Literal 違反 (LLM が ConcreteItem.type に "date" 等を返す) を
            # sanitize して再試行する。sanitize 後も失敗したら None。
            _sanitize_metrics_data(data)
            return QAMetrics.model_validate(data)
    except Exception as e:
        logger.warning("score_qa_pairs_metrics: failed for %s: %s", pair.id, e)
        return None


def score_qa_pairs_metrics(
    qa_pairs: QAPairsOutput,
    *,
    max_workers: int = 8,
) -> QAPairsOutput:
    """全Q&AペアにV4評価指標を付与して QAPairsOutput を返す（Step 6d）。

    各ペアに QAMetrics を並列で付与する。失敗したペアは metrics=None のまま保持する。
    成功した場合は score_schema_version="2.0", prompt_version="V4" を設定する。
    """
    if not qa_pairs.pairs:
        return qa_pairs

    scored: dict[str, QAMetrics | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_score_one_pair, p): p.id for p in qa_pairs.pairs}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                scored[pid] = future.result()
            except Exception as e:
                logger.warning("score_qa_pairs_metrics: unexpected error for %s: %s", pid, e)
                scored[pid] = None

    success = 0
    for pair in qa_pairs.pairs:
        pair.metrics = scored.get(pair.id)
        if pair.metrics is not None:
            success += 1

    if success > 0:
        qa_pairs.score_schema_version = "2.0"
        qa_pairs.prompt_version = "V4"

    logger.info(
        "score_qa_pairs_metrics: scored %d/%d pairs with V4 metrics",
        success,
        len(qa_pairs.pairs),
    )
    return qa_pairs
