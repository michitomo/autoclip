"""JetCut: 語タイムスタンプ列から編集判断 (EDL) を組み立てる (autoclip 固有)。

入力は member-WAV 時間の WhisperWord 列。出力は EDL:
- keep_ranges: レンダラが連結する残存区間 (member-WAV 時間, 昇順, 重複なし)
- kept_words: 残存語の旧 (member-WAV) → 新 (カット後) タイムスタンプ。字幕の唯一の真実
- params: 使用パラメータ (再現性)

カット対象:
1. 無音 (dead air): 隣接語の gap が dead_air_gap 超なら、その隙間を pad を残して除去
2. フィラー: えー/あの/その/まあ … を **語境界で** 除去
   日本語は宿/泊/する のようにサブワード分割されるため、フィラーも え/ー/と のように
   複数トークンに割れる。隣接トークンを連結して照合する (sub-word reassembly)。
   「その/あの/まあ」等は連体詞・実副詞にもなるため、直後に間がある (単独の言い淀み)
   時だけ除去する曖昧フィラーガードを入れる (「そのため」「あの方」は残す)。
3. drop_spans: 言い間違い・言い淀みの時刻スパン (src/video/disfluency.py が LLM で
   検出)。語の有無に関わらず keep_ranges から区間減算して映像から確実に切る。

人手レビュー前提。意味を壊す過剰カットは避けるが、しきい値で積極カットも選べる。
"""

from __future__ import annotations

import logging

from src.models import EDL, KeepRange, KeptWord, WhisperWord

logger = logging.getLogger(__name__)

# --- 既定パラメータ (秒) ---
# 1.0 秒超の語間無音だけ除去する。0.6 だと日本語の文中の息継ぎ・句読点の自然な間
# まで切り、語/文の途中で繋がって不自然 (完成クリップを Whisper した検証で確認)。
DEAD_AIR_GAP = 1.0        # これ超の語間無音を除去対象にする
SILENCE_PAD = 0.20        # 除去する無音の前後に残す余白 (語尾の減衰を残す)
MERGE_GAP = 0.25          # この未満しか離れていない keep 区間は連結する
MIN_SEGMENT = 0.30        # これ未満の keep 区間は捨てる (チラつき防止)
EDGE_PAD = 0.10           # クリップ先頭/末尾に残す余白

# フィラー語 (サブワード連結後に完全一致で判定)。保守的に明確なものだけ。
DEFAULT_FILLERS: frozenset[str] = frozenset(
    {
        "えー", "えーと", "えっと", "ええと", "あの", "あのー", "あのう",
        "その", "そのー", "まあ", "まー", "ええ", "んー", "うー", "あー",
        "えーっと", "んーと",
    }
)

# 曖昧フィラー: 連体詞・指示語・実副詞としても使う語。直後に「間」がある時だけ
# (= 単独の言い淀みとして発された時だけ) 除去する。「そのため」「あの方」「まあいい」
# のように後続語へ glue している (直後 gap が小さい) 場合は実語として残す。
# (「えー」「あのー」等の長音フィラーは曖昧でないので常に除去対象)
AMBIGUOUS_FILLERS: frozenset[str] = frozenset(
    {"あの", "その", "まあ", "まー", "ええ", "あー"}
)
# 曖昧フィラーを「単独の言い淀み」と判定する直後 gap のしきい値 (秒)。これ未満なら
# 後続語と一体 (実語) とみなして残す。
FILLER_GLUE_GAP = 0.15

# サブワード連結でフィラーを探す際の最大トークン数 (えーっと ≈ 4 トークン程度)
_MAX_FILLER_TOKENS = 5


def _normalize(text: str) -> str:
    """照合用に空白を除去する。"""
    return "".join(text.split())


def _mark_filler_words(
    words: list[WhisperWord],
    fillers: frozenset[str],
    *,
    ambiguous: frozenset[str] = AMBIGUOUS_FILLERS,
    glue_gap: float = FILLER_GLUE_GAP,
) -> list[bool]:
    """各語がフィラーとして除去対象かを示す bool 列を返す (サブワード連結対応)。

    位置 i から最大 _MAX_FILLER_TOKENS 語を連結し、フィラー集合に完全一致する
    最長の並びを探す。一致すればその範囲を全て drop マークし、その先へ進む。
    「えーと」が え/ー/と に割れていても、また 1 トークンでも検出できる。

    曖昧フィラー (その/あの/まあ 等; 連体詞・実副詞にもなる語) の誤爆対策:
    一致した文字列が `ambiguous` に属する場合、**直後 gap が glue_gap 以上 (= 後ろに
    間がある単独の言い淀み)** の時だけ落とす。直後語へ glue している (gap が小さい)
    なら「そのため」「あの方」のような実語の一部とみなして残す。長音フィラー
    (「えー」「あのー」等) は曖昧でないので常に落とす。
    """
    n = len(words)
    drop = [False] * n
    i = 0
    while i < n:
        matched_len = 0
        matched_text = ""
        combined = ""
        # i から伸ばして完全一致する最長 run を探す
        for k in range(min(_MAX_FILLER_TOKENS, n - i)):
            combined += _normalize(words[i + k].word)
            if combined in fillers:
                matched_len = k + 1
                matched_text = combined
        if matched_len:
            # 曖昧フィラーは「直後に間がある」時だけ落とす (glue なら実語として残す)。
            keep_as_real = False
            if matched_text in ambiguous:
                nxt = i + matched_len
                if nxt < n:
                    gap_after = words[nxt].start - words[nxt - 1].end
                    if gap_after < glue_gap:
                        keep_as_real = True
            if not keep_as_real:
                for j in range(i, i + matched_len):
                    drop[j] = True
            i += matched_len
        else:
            i += 1
    return drop


def build_edl(
    words: list[WhisperWord],
    *,
    dead_air_gap: float = DEAD_AIR_GAP,
    silence_pad: float = SILENCE_PAD,
    merge_gap: float = MERGE_GAP,
    min_segment: float = MIN_SEGMENT,
    edge_pad: float = EDGE_PAD,
    fillers: frozenset[str] = DEFAULT_FILLERS,
    remove_fillers: bool = True,
    drop_spans: list[tuple[float, float]] | None = None,
) -> EDL:
    """語列から JetCut EDL を構築する。

    Args:
        words: member-WAV 時間の WhisperWord 列 (start 昇順想定)
        dead_air_gap..edge_pad: しきい値 (秒)
        fillers: フィラー語集合
        remove_fillers: False ならフィラー除去をスキップ (無音カットのみ)
        drop_spans: 言い間違い・言い淀み等の除去スパン (member-WAV 秒)。語の有無に
            関わらず keep_ranges から区間減算し、内部に入る語も落とす。

    Returns:
        EDL (keep_ranges, kept_words, params)
    """
    spans = _norm_spans(drop_spans)
    params: dict[str, object] = {
        "dead_air_gap": dead_air_gap,
        "silence_pad": silence_pad,
        "merge_gap": merge_gap,
        "min_segment": min_segment,
        "edge_pad": edge_pad,
        "remove_fillers": remove_fillers,
        "n_input_words": len(words),
        "n_drop_spans": len(spans),
        "drop_spans_seconds": round(sum(e - s for s, e in spans), 2),
    }
    if not words:
        return EDL(keep_ranges=[], kept_words=[], params=params)

    ws = sorted(words, key=lambda w: (w.start, w.end))

    drop_filler = (
        _mark_filler_words(ws, fillers) if remove_fillers else [False] * len(ws)
    )

    # 残す語を集める: フィラー、および drop_spans 内 (語の中点がスパン内) の語を除外。
    kept: list[WhisperWord] = [
        w for w, d in zip(ws, drop_filler)
        if not d and not _in_any_span((w.start + w.end) / 2.0, spans)
    ]
    n_removed = len(ws) - len(kept)
    if not kept:
        return EDL(
            keep_ranges=[], kept_words=[],
            params={**params, "n_words_removed": n_removed},
        )

    # 残した語から、語間 gap > dead_air_gap で分割しながら keep 区間を作る。
    # 各区間は [first.start - edge一部, last.end + edge一部]。区間内の無音は
    # 「語の連続」で表現されるため、語間 gap が小さいものは同一区間に含める。
    ranges: list[list[float]] = []  # [start, end]
    cur_start = kept[0].start
    cur_end = kept[0].end
    for prev, w in zip(kept, kept[1:]):
        gap = w.start - prev.end
        if gap > dead_air_gap:
            # 無音が大きい → 区間を閉じ、pad を足して保存
            ranges.append([cur_start, cur_end])
            cur_start = w.start
            cur_end = w.end
        else:
            # gap は無音だが残す (語が近接) → 区間を伸ばす
            cur_end = w.end
    ranges.append([cur_start, cur_end])

    # 無音 pad を区間境界に付与し、隣接区間が merge_gap 未満なら連結
    padded = _pad_and_merge(ranges, silence_pad, merge_gap, edge_pad)

    # min_segment 未満の区間を捨てる
    kept_ranges = [r for r in padded if (r[1] - r[0]) >= min_segment]

    # drop_spans を keep_ranges から区間減算する。フィラー語を落としても pad/merge で
    # 無音が残ることがあるため、語の有無に関わらず時刻ベースで確実に穴を空ける。
    if spans:
        kept_ranges = [
            r for r in _subtract_spans(kept_ranges, spans)
            if (r[1] - r[0]) >= min_segment
        ]

    if not kept_ranges:
        # 全部短すぎ / 減算で全滅した場合のフォールバック: 最長 1 区間だけ残す
        base = _subtract_spans(padded, spans) if spans else padded
        kept_ranges = [max(base or padded, key=lambda r: r[1] - r[0])]

    # 旧→新タイムラインのマッピングを作る (区間連結後の累積長で new 時間を計算)
    kept_words_out = _remap_words(kept, kept_ranges)

    keep_ranges = [KeepRange(start=r[0], end=r[1]) for r in kept_ranges]
    out_params = {
        **params,
        "n_words_removed": n_removed,
        "n_keep_ranges": len(keep_ranges),
        "n_kept_words": len(kept_words_out),
    }
    logger.info(
        "JetCut: %d words → %d kept (%d removed: filler+disfluency), "
        "%d drop-spans, %d keep-ranges, %.1fs kept",
        len(ws), len(kept), n_removed, len(spans), len(keep_ranges),
        sum(r.end - r.start for r in keep_ranges),
    )
    return EDL(keep_ranges=keep_ranges, kept_words=kept_words_out, params=out_params)


def _norm_spans(
    spans: list[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    """drop_spans を昇順・正値・重なりマージした [(s,e)] に正規化する。"""
    if not spans:
        return []
    clean = [(float(s), float(e)) for s, e in spans if e > s]
    if not clean:
        return []
    clean.sort()
    out: list[list[float]] = [list(clean[0])]
    for s, e in clean[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def _in_any_span(t: float, spans: list[tuple[float, float]]) -> bool:
    """t がいずれかの span [s,e] に入るか (spans は昇順想定で早期 return)。"""
    for s, e in spans:
        if t < s:
            return False
        if t <= e:
            return True
    return False


def _subtract_spans(
    ranges: list[list[float]], spans: list[tuple[float, float]]
) -> list[list[float]]:
    """ranges から spans を区間減算する (差集合)。ranges/spans とも昇順想定。"""
    if not spans:
        return [list(r) for r in ranges]
    out: list[list[float]] = []
    for rs, re in ranges:
        pieces = [[rs, re]]
        for ds, de in spans:
            nxt: list[list[float]] = []
            for ps, pe in pieces:
                if de <= ps or ds >= pe:  # 重なりなし → そのまま
                    nxt.append([ps, pe])
                    continue
                if ds > ps:               # 左側が残る
                    nxt.append([ps, ds])
                if de < pe:               # 右側が残る
                    nxt.append([de, pe])
            pieces = nxt
        out.extend(p for p in pieces if p[1] > p[0])
    return out


def _pad_and_merge(
    ranges: list[list[float]], silence_pad: float, merge_gap: float, edge_pad: float
) -> list[list[float]]:
    """各区間に pad を付け、近接区間を連結する。"""
    if not ranges:
        return []
    # pad: 各区間の端を silence_pad ぶん外側へ (区間境界の無音を少し残す)
    out: list[list[float]] = []
    for idx, (s, e) in enumerate(ranges):
        pad_lead = edge_pad if idx == 0 else silence_pad
        ns = max(0.0, s - pad_lead)
        ne = e + silence_pad
        if out and ns - out[-1][1] < merge_gap:
            out[-1][1] = max(out[-1][1], ne)
        else:
            out.append([ns, ne])
    return out


def _remap_words(
    kept_words: list[WhisperWord], keep_ranges: list[list[float]]
) -> list[KeptWord]:
    """残存語に、カット後タイムライン上の new_start/new_end を付ける。

    new 時間 = 「その語の旧時間より前にある keep 区間の累積長」+「同区間内オフセット」。
    keep_ranges に入らない語 (pad 調整で稀に外れる) は最近接区間にクランプする。
    """
    # 区間ごとの new 起点 (累積長)
    cum: list[float] = []
    acc = 0.0
    for s, e in keep_ranges:
        cum.append(acc)
        acc += e - s

    def map_time(t: float) -> float:
        for (s, e), base in zip(keep_ranges, cum):
            if s <= t <= e:
                return base + (t - s)
        # 区間外: 最近接にクランプ
        if t < keep_ranges[0][0]:
            return 0.0
        for (s, e), base in zip(keep_ranges, cum):
            if t < s:
                return base  # 直前区間と次区間の境目 → 次区間先頭
        last_s, last_e = keep_ranges[-1]
        return cum[-1] + (last_e - last_s)

    out: list[KeptWord] = []
    for w in kept_words:
        out.append(
            KeptWord(
                word=w.word,
                old_start=w.start,
                old_end=w.end,
                new_start=map_time(w.start),
                new_end=map_time(w.end),
            )
        )
    return out
