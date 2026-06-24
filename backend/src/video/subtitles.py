"""字幕生成 (autoclip 固有): kept_words → ASS / SRT。

タイミングは **カット後タイムライン** (KeptWord.new_start/new_end) のみを使う。
これが字幕とレンダラの唯一の真実 (旧時間は使わない)。

日本語はサブワード分割されるため、語を連結して読みやすいキャプション行にまとめる
(最大文字数 or 句読点で改行)。ASS は libass のスタイル付き (大きめ・縁取り・下寄せ) で
ソーシャル向け。SRT は副産物として併せて出力する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models import KeptWord

logger = logging.getLogger(__name__)

# キャプション 1 行の目安最大文字数 (日本語)。句読点が無いまま伸びた時の保険。
MAX_LINE_CHARS = 24
# この長さ以上の語間ポーズ (new 時間) があれば行を分ける
LINE_BREAK_GAP = 0.7
# 句点・終止系 (文の終わり → 必ず改行)
_SENTENCE_PUNCT = "。．！？!?"
# 読点 (文節の区切り → 積極的に改行)
_CLAUSE_PUNCT = "、，"
# 句読点で行を区切る (文節単位)
_BREAK_PUNCT = _SENTENCE_PUNCT + _CLAUSE_PUNCT
# これ以下の可視文字数の断片は単独キャプションにせず前の行に併合する
# (句読点のみ・1〜2文字の文節を防ぐ)
MIN_CAPTION_CHARS = 4


@dataclass
class Caption:
    """1 キャプション行 (カット後タイムライン)。"""

    start: float
    end: float
    text: str


def group_captions(
    kept_words: list[KeptWord],
    max_chars: int = MAX_LINE_CHARS,
    line_break_gap: float = LINE_BREAK_GAP,
    break_after: set[int] | None = None,
    max_lines: int = 3,
) -> list[Caption]:
    """kept_words を文節単位のキャプション行にまとめる (new タイムライン)。

    改行条件 (優先度順):
    - break_after (LLM 文節境界) にある語 index の直後で改行 (最優先・厳守)
    - 句読点 (。、！？ 等) の直後で改行 → 文節単位 (break_after が無い時)
    - 句読点が無いまま max_chars に達したら改行 (保険)
    - 語間 new ポーズが line_break_gap 超なら改行

    句読点のみ・極端に短い断片 (< MIN_CAPTION_CHARS) は単独行にせず、直前の行に
    併合する (「。」だけのキャプションを防ぐ)。

    Args:
        break_after: LLM が決めた文節境界。この index の語の後で必ず改行する。
            指定時は句読点ベースの自動改行より優先 (厳守)。
    """
    captions: list[Caption] = []
    if not kept_words:
        return captions

    use_llm_breaks = bool(break_after)
    cur_words: list[KeptWord] = []
    cur_text = ""

    def flush() -> None:
        nonlocal cur_words, cur_text
        if not cur_words:
            return
        text = cur_text.strip()
        start = cur_words[0].new_start
        end = cur_words[-1].new_end
        # 短すぎる断片 (句読点のみ等) は前の行に併合する
        if captions and _visible_len(text) < MIN_CAPTION_CHARS:
            prev = captions[-1]
            prev.text += text
            prev.end = max(prev.end, end)
        else:
            captions.append(Caption(start=start, end=end, text=text))
        cur_words = []
        cur_text = ""

    for i, kw in enumerate(kept_words):
        cur_words.append(kw)
        cur_text += kw.word
        if use_llm_breaks:
            # LLM 文節境界でのみ改行候補。ただし 1 キャプションに複数文節を載せ、
            # バブル容量 (max_chars × max_lines 文字) に近づくまで貯める
            # → スマホ向けに一度に出る文字数を増やす。語の途中では切らない。
            cap_chars = max_chars * max_lines
            at_boundary = i in break_after
            cur_len = _visible_len(cur_text)
            # 次の語を足すと容量超過しそう、かつ境界に来たら flush。
            # 容量を大きく超えた場合は境界でなくても保険で flush (極端な長文対策)。
            if at_boundary and cur_len >= cap_chars * 0.75:
                flush()
            elif cur_len >= cap_chars:
                flush()
            continue
        ends_with_punct = bool(kw.word) and kw.word[-1] in _BREAK_PUNCT
        next_gap = (
            kept_words[i + 1].new_start - kw.new_end if i + 1 < len(kept_words) else 0.0
        )
        if (
            ends_with_punct
            or _visible_len(cur_text) >= max_chars
            or next_gap > line_break_gap
        ):
            flush()
    flush()

    # ダブり防止: 連続キャプションが時間的に重なるなら、前者の end を後者 start に
    # 詰める (libass は重なると重ね描画するため)
    for a, b in zip(captions, captions[1:]):
        if a.end > b.start:
            a.end = b.start
    return captions


def _visible_len(text: str) -> int:
    """空白を除いた可視文字数を返す。"""
    return len("".join(text.split()))


# ---------------------------------------------------------------------------
# ASS
# ---------------------------------------------------------------------------

# ローリング字幕の色 (shusantv .old 相当)
_DIM_COLOUR = "&H00A2A9A9"  # グレー #a9a9a2 (古い語)

# shusantv 風グラスバブル字幕:
#  - 文字色 #ffeebe (クリーム) → ASS は &H00BBGGRR = &H00BEEEFF
#  - 背景バブル: BorderStyle=4 (不透明ボックス) を BackColour で塗る。
#    rgba(10,10,14,.58) 相当 → ASS &HAABBGGRR, alpha は反転 (.58不透明≒0x6B透明)
#    → &H6B0E0A0A
#  - 視認性のため細い縁取り (OutlineColour 黒)
#  - 下部中央やや上、最大3行 (折り返しは WrapStyle=0 で自動)
_TEXT_COLOUR = "&H00B7E05F"      # ミントグリーン #5FE0B7 (質疑者・既定の accent)
_ANSWER_COLOUR = "&H000F8CFF"    # オレンジ #FF8C0F (答弁者; 質疑と色分け)
_OUTLINE_COLOUR = "&H99000000"   # 半透明黒の縁
_BOX_COLOUR = "&H6B0E0A0A"       # rgba(10,10,14,.58) 相当のボックス塗り

# role → 明色 (ベース色)。発話強調の暗色 (_DIM_COLOUR) は role 共通。
# 答弁系はオレンジ、それ以外 (質疑者/委員長/不明) はミント。
_ANSWER_ROLES = frozenset({"答弁者", "政府参考人", "参考人"})


def _role_colour(role: str | None) -> str:
    """話者 role に対応する明色 (ベース色) を返す。"""
    if role in _ANSWER_ROLES:
        return _ANSWER_COLOUR
    return _TEXT_COLOUR


def _role_colour_at(
    t: float, role_spans: list[tuple[float, float, str]] | None
) -> str:
    """時刻 t (カット後) に発話している話者の色を返す。span 外は質疑者色。

    role_spans は (start, end, role) の昇順リスト (カット後タイムライン)。隣接 span の
    境界 (前 span の end = 次 span の start) は **後の span** に属させる (新しい話者が
    その時刻から話し始める、の方が直感的)。最後の span のみ end を含める。
    """
    if not role_spans:
        return _TEXT_COLOUR
    n = len(role_spans)
    for i, (s, e, role) in enumerate(role_spans):
        last = i == n - 1
        if s <= t < e or (last and t == e):
            return _role_colour(role)
    return _TEXT_COLOUR

# タイトル (本文と差別化): 白い背景パネル＋濃いめの緑の太い額縁＋黒文字 (縁取りなし)。
#  - パネル: \p ドローイングの矩形を白塗り、Plate スタイルの太い Outline を緑にして額縁に。
#  - 文字: 見出し (小2行) と要旨 (大) はいずれも黒・縁取りなし (Outline=0)。
_TITLE_PLATE_FILL = "&H00FFFFFF"      # パネルの白塗り (PrimaryColour)
_TITLE_PLATE_BORDER = "&H0082AE2B"    # 濃いめの緑の額縁 #2BAE82 (BGR)
_TITLE_PLATE_BORDER_PX = 18           # 額縁の太さ (1080 幅基準)
_TITLE_PLATE_MARGIN = 24              # パネル左右/上の余白 (PlayRes 基準)
_TITLE_TEXT_COLOUR = "&H00000000"     # 黒文字 (見出し・要旨共通)

# 常時ヘッダー (タイトル消滅後、画面上部に出し続ける帯): 半透明黒帯 + 白文字。
_BANNER_TEXT_COLOUR = "&H00FFFFFF"    # 白文字
_BANNER_BOX_COLOUR = "&H80000000"     # 半透明黒の帯 (AA=80 = 50%透過)

_ASS_HEADER_TMPL = """[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fontsize},{text_colour},{text_colour},{outline_colour},{box_colour},-1,0,0,0,100,100,0,0,4,{outline},0,2,{margin_h},{margin_h},{margin_v},1
Style: Plate,{font},1,{plate_fill},{plate_fill},{plate_border},{plate_fill},0,0,0,0,100,100,0,0,1,{plate_border_px},0,7,0,0,0,1
Style: Head,{font},{head_fontsize},{title_text_colour},{title_text_colour},{title_text_colour},{title_text_colour},-1,0,0,0,100,100,0,0,1,0,0,8,{margin_h},{margin_h},{head_margin_v},1
Style: Big,{font},{title_fontsize},{title_text_colour},{title_text_colour},{title_text_colour},{title_text_colour},-1,0,0,0,100,100,0,0,1,0,0,8,{margin_h},{margin_h},{big_margin_v},1
Style: Banner,{font},{banner_fontsize},{banner_text_colour},{banner_text_colour},{banner_box_colour},{banner_box_colour},-1,0,0,0,100,100,0,0,4,0,0,8,{margin_h},{margin_h},{banner_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(t: float) -> str:
    """秒 → ASS タイム H:MM:SS.cc。"""
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:  # 丸め桁上がり
        cs = 0
        s += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """ASS イベントテキストのエスケープ。"""
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def build_ass(
    kept_words: list[KeptWord],
    *,
    res_x: int = 1080,
    res_y: int = 1920,
    font: str = "Hiragino Sans",
    fontsize: int | None = None,
    outline: int = 2,
    margin_h: int | None = None,
    margin_v: int | None = None,
    max_chars: int | None = None,
    break_after: set[int] | None = None,
    max_lines: int = 5,
) -> str:
    """kept_words から shusantv 風グラスバブル ASS を生成する (new タイムライン)。

    既定は 9:16 (1080x1920)。フォント/マージン/折り返し文字数は解像度から自動算出
    (明示指定があれば優先)。break_after を渡すと LLM 文節境界で厳守改行する。
    max_lines は 1 キャプションの最大行数 (スマホ向けに既定 5 = 画面下 ~1/4)。
    """
    # 解像度に応じた既定値 (1080 幅基準でスケール)。スマホ視聴前提で大きめ。
    scale = res_x / 1080.0
    if fontsize is None:
        fontsize = round(66 * scale)  # 大きめ (旧52→66)
    if margin_h is None:
        margin_h = round(48 * scale)  # 左右を詰めて 1 行の文字数を増やす
    if margin_v is None:
        margin_v = round(res_y * 0.06)  # 下から ~6% (字幕帯を画面下 1/4 に寄せる)
    if max_chars is None:
        # バブル幅 (res_x - 2*margin_h) を fontsize で割った概算 (全角≒fontsize px)
        usable = res_x - 2 * margin_h
        max_chars = max(8, int(usable / fontsize))

    captions = group_captions(
        kept_words, max_chars=max_chars, break_after=break_after, max_lines=max_lines
    )
    header = _ass_header(res_x, res_y, font, fontsize, outline, margin_h, margin_v)
    lines = [header]
    for c in captions:
        if not c.text:
            continue
        # 3行を超える長文は稀だが、libass の自動折り返しに任せる (WrapStyle=0)。
        lines.append(
            f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Default,,0,0,0,,"
            f"{_ass_escape(c.text)}"
        )
    return "\n".join(lines) + "\n"


_TITLE_BREAK_MARKER = "｜"  # generate_title が入れる改行指定マーカー


def _fit_title_fontsize(
    title: str, res_x: int, margin_h: int, body_fontsize: int
) -> int:
    """要旨 (大きい方) の最長 ｜-セグメントが 1 行に収まるフォントサイズを返す。

    各 ｜-セグメントが途中で折り返されないことを優先する。承認デザイン (case K) の
    要旨は ~98px (1080 幅) なので、それを目安に上限 ~本文1.15倍に抑える (短い要旨が
    巨大化してパネルが間延びするのを防ぐ)。長い要旨は収まるまで縮める (下限 本文0.72倍、
    それ以下なら折り返す)。パネル内側余白 (額縁+パディング) を考慮した usable 幅で測る。
    """
    hi = round(body_fontsize * 1.15)   # 短い要旨でも巨大化させない (case K≈98 基準)
    lo = round(body_fontsize * 0.72)   # 収まり優先: ここまでは縮める
    if not title:
        return hi
    segs = [s.strip() for s in title.split(_TITLE_BREAK_MARKER) if s.strip()]
    longest = max((len(s) for s in segs), default=1)
    usable = _title_usable_width(res_x, margin_h)
    # usable / longest = その行を 1 行に収める最大フォント幅 (95% で少し余裕)
    fit = int(usable * 0.95 / max(1, longest))
    return max(lo, min(hi, fit))


def _title_usable_width(res_x: int, margin_h: int) -> int:
    """要旨/見出しテキストが使える横幅 (パネル内側、額縁の内側)。

    パネルは左右 _TITLE_PLATE_MARGIN の位置にあり、額縁 (Plate Outline) が内側に
    食い込む。テキストはさらに少し内側に置くので、額縁太さ + 余白を引く。
    (body の margin_h ではなくパネル基準。引数 margin_h は将来の調整用に残す。)
    """
    scale = res_x / 1080.0
    plate_margin = round(_TITLE_PLATE_MARGIN * scale)
    inset = round((_TITLE_PLATE_BORDER_PX + 24) * scale)  # 額縁 + 内パディング
    return max(100, res_x - 2 * plate_margin - 2 * inset)


def _wrap_title_line(seg: str, per_line: int) -> str:
    """1 セグメントを per_line で折り返し (禁則・英単語非分割) \\N 付きで返す。"""
    chars = list(seg)
    breaks = (
        _apply_kinsoku(chars, _wrap_break_positions(chars, per_line))
        if len(chars) > per_line else set()
    )
    line = ""
    for i, c in enumerate(chars):
        if i in breaks:
            line += "\\N"
        line += _ass_escape(c)
    return line


@dataclass
class _TitleLayout:
    """白パネルタイトルの幾何 (PlayRes 座標)。

    見出し (小・複数行) と要旨 (大・複数行) を上から積み、白い背景パネルと
    濃い緑の額縁で囲う。パネル高さは行数に応じて自動計算する。
    """

    head_lines: list[str]   # \N 連結済みの見出し各行 (折り返し後)
    big_lines: list[str]    # \N 連結済みの要旨各行 (折り返し後)
    head_fontsize: int
    big_fontsize: int
    plate_left: int
    plate_right: int
    plate_top: int
    plate_bottom: int
    head_top: int           # 見出しブロックの上端 y (top-anchor)
    big_top: int            # 要旨ブロックの上端 y (top-anchor)


def _build_title_layout(
    title: str,
    header_lines: list[str] | None,
    *,
    res_x: int,
    res_y: int,
    margin_h: int,
    title_fontsize: int,
    header_fontsize: int | None = None,
) -> _TitleLayout | None:
    """見出し/要旨を折り返してパネル幾何を計算する (case K 準拠・余白最小)。"""
    scale = res_x / 1080.0
    t = title.strip()
    usable = _title_usable_width(res_x, margin_h)

    # 見出し (小): 既定は要旨の 0.72 倍 (case K は 70/98 ≒ 0.71)。
    # 見出しフォント: 既定は要旨の 0.72 倍だが、見出し各行 (日付/委員会+議員) が
    # 1 行に収まるよう必要なら縮める (case K: 70px で「厚生労働委員会 古川あおい」が
    # 1 行)。これで「い」だけ折返される等の不格好な改行を防ぐ。
    hdr_clean = [h.strip() for h in (header_lines or []) if h.strip()]
    if header_fontsize:
        hfs = header_fontsize
    else:
        hfs = max(28, round(title_fontsize * 0.72))
        longest_h = max((len(h) for h in hdr_clean), default=1)
        # その行を 1 行に収める最大フォント幅 (98% 余裕)。上限は上の既定。
        fit_h = int(usable * 0.98 / max(1, longest_h))
        hfs = max(28, min(hfs, fit_h))
    head_per_line = max(6, int(usable / hfs))
    head_lines: list[str] = [_wrap_title_line(h, head_per_line) for h in hdr_clean]

    # 要旨 (大): ｜ 優先改行 + 収まらない行は文字数で補助折り返し。
    big_per_line = max(4, int(usable / title_fontsize))
    big_lines: list[str] = []
    if t:
        for s in t.split(_TITLE_BREAK_MARKER):
            s = s.strip()
            if not s:
                continue
            big_lines.append(_wrap_title_line(s, big_per_line))

    if not head_lines and not big_lines:
        return None

    # 行数 (折り返し後の実行数)。\N でカウント。
    n_head = sum(line.count("\\N") + 1 for line in head_lines)
    n_big = sum(line.count("\\N") + 1 for line in big_lines)

    # 行高 = フォント * 1.18 (日本語の実測行送り)。
    head_lh = round(hfs * 1.18)
    big_lh = round(title_fontsize * 1.18)

    # 余白 (case K: 上下とも詰める)。
    pad_top = round(28 * scale)        # パネル内・上パディング
    pad_bottom = round(34 * scale)     # パネル内・下パディング
    gap = round(24 * scale) if (n_head and n_big) else 0  # 見出し↔要旨の間

    plate_top = round(_TITLE_PLATE_MARGIN * scale) + round(24 * scale)  # 画面上の余白
    head_top = plate_top + pad_top
    big_top = head_top + n_head * head_lh + gap
    content_bottom = big_top + n_big * big_lh
    plate_bottom = content_bottom + pad_bottom

    plate_margin = round(_TITLE_PLATE_MARGIN * scale)
    plate_left = plate_margin
    plate_right = res_x - plate_margin

    return _TitleLayout(
        head_lines=head_lines,
        big_lines=big_lines,
        head_fontsize=hfs,
        big_fontsize=title_fontsize,
        plate_left=plate_left,
        plate_right=plate_right,
        plate_top=plate_top,
        plate_bottom=plate_bottom,
        head_top=head_top,
        big_top=big_top,
    )


def _ass_header(
    res_x: int, res_y: int, font: str, fontsize: int,
    outline: int, margin_h: int, margin_v: int, title_fontsize: int | None = None,
    title_layout: "_TitleLayout | None" = None,
) -> str:
    scale = res_x / 1080.0
    if title_fontsize is None:
        title_fontsize = round(fontsize * 2.0)
    # Head/Big の MarginV は \pos で上書きするので、style 値は妥当な既定で良い。
    head_fontsize = title_layout.head_fontsize if title_layout else max(
        28, round(title_fontsize * 0.72)
    )
    head_margin_v = title_layout.head_top if title_layout else round(res_y * 0.04)
    big_margin_v = title_layout.big_top if title_layout else round(res_y * 0.10)
    return _ASS_HEADER_TMPL.format(
        res_x=res_x, res_y=res_y, font=font, fontsize=fontsize,
        text_colour=_TEXT_COLOUR, outline_colour=_OUTLINE_COLOUR,
        box_colour=_BOX_COLOUR, outline=outline,
        margin_h=margin_h, margin_v=margin_v,
        # タイトル: 白パネル (Plate) ＋濃い緑の太い額縁、黒文字 (縁取りなし)。
        title_fontsize=title_fontsize,
        title_text_colour=_TITLE_TEXT_COLOUR,
        head_fontsize=head_fontsize,
        head_margin_v=head_margin_v,
        big_margin_v=big_margin_v,
        plate_fill=_TITLE_PLATE_FILL,
        plate_border=_TITLE_PLATE_BORDER,
        plate_border_px=round(_TITLE_PLATE_BORDER_PX * scale),
        # 常時ヘッダー: 字幕本文より少し小さい白文字 + 半透明黒帯、上端寄せ。
        banner_fontsize=round(fontsize * 0.62),
        banner_text_colour=_BANNER_TEXT_COLOUR,
        banner_box_colour=_BANNER_BOX_COLOUR,
        banner_margin_v=round(res_y * 0.02),
    )


def _title_event(
    title: str, res_x: int, title_seconds: float, *,
    margin_h: int, title_fontsize: int,
    res_y: int = 1920,
    header_lines: list[str] | None = None,
    header_fontsize: int | None = None,
    layout: "_TitleLayout | None" = None,
) -> str:
    """白パネルタイトルを画面上部に title_seconds 秒表示する Dialogue 群を返す。

    3 つの Dialogue を改行区切りで返す:
      1. Plate (白パネル＋緑額縁の \\p ドローイング矩形)
      2. Head  (日付/委員会+議員 等の定型見出し・小・黒)
      3. Big   (要旨・大・黒)
    パネル高さは見出し+要旨の行数に応じて自動計算 (余白最小, case K 準拠)。
    layout を渡せば再計算せずそれを使う (header と幾何を一致させるため)。
    """
    if layout is None:
        layout = _build_title_layout(
            title, header_lines,
            res_x=res_x, res_y=res_y, margin_h=margin_h,
            title_fontsize=title_fontsize, header_fontsize=header_fontsize,
        )
    if layout is None:
        return ""

    start, end = _ass_time(0.0), _ass_time(title_seconds)
    cx = (layout.plate_left + layout.plate_right) // 2  # 水平中心 (top-center anchor)
    lines: list[str] = []

    # 1) 白パネル + 緑額縁 (\p ドローイング)。Plate スタイルが塗り色/額縁を持つ。
    rect = (
        f"m {layout.plate_left} {layout.plate_top} "
        f"l {layout.plate_right} {layout.plate_top} "
        f"{layout.plate_right} {layout.plate_bottom} "
        f"{layout.plate_left} {layout.plate_bottom}"
    )
    lines.append(
        f"Dialogue: 0,{start},{end},Plate,,0,0,0,,{{\\p1\\pos(0,0)}}{rect}{{\\p0}}"
    )

    # 2) 見出し (小・黒)。top-center anchor で上端 y を \pos 指定。
    if layout.head_lines:
        head_text = "\\N".join(layout.head_lines)
        lines.append(
            f"Dialogue: 1,{start},{end},Head,,0,0,0,,"
            f"{{\\an8\\pos({cx},{layout.head_top})}}{head_text}"
        )

    # 3) 要旨 (大・黒)。
    if layout.big_lines:
        big_text = "\\N".join(layout.big_lines)
        lines.append(
            f"Dialogue: 1,{start},{end},Big,,0,0,0,,"
            f"{{\\an8\\pos({cx},{layout.big_top})}}{big_text}"
        )

    return "\n".join(lines)


def _banner_event(
    header_lines: list[str] | None, start_t: float, end_t: float
) -> str:
    """常時ヘッダー (半透明黒帯+白文字) の Dialogue を 1 行返す。

    タイトル消滅後 (start_t = title_seconds) から clip 終端 (end_t) まで、画面上部に
    ヘッダー3行 (日付/委員会/議員+トピック) を出し続ける。header_lines 無し or
    end<=start なら空文字。
    """
    clean = [h.strip() for h in (header_lines or []) if h.strip()]
    if not clean or end_t <= start_t:
        return ""
    text = "\\N".join(_ass_escape(h) for h in clean)
    return (
        f"Dialogue: 0,{_ass_time(start_t)},{_ass_time(end_t)},Banner,,0,0,0,,{text}"
    )


def _rolling_windows(
    kept_words: list[KeptWord],
    cap_chars: int,
    break_after: set[int] | None,
) -> list[tuple[float, float, int, int, int]]:
    """ローリング表示用ウィンドウを生成する。

    各語 i ごとに 1 ウィンドウ。ウィンドウは (start, end, win_lo, i, bright_lo):
      - start = words[i].new_start, end = words[i+1].new_start (最後は new_end)
      - [win_lo, i] が画面に出す語範囲 (累積文字 <= cap_chars に収まる下限 win_lo)
      - bright_lo = 直近文節(最後の break_after 以降) の開始 index。
        [bright_lo, i] を明色、[win_lo, bright_lo) をグレーにする。
    文末 (。!?) の語で一旦ウィンドウをリセット (次語は新規に積み直し)。
    """
    n = len(kept_words)
    if n == 0:
        return []
    lens = [len("".join(w.word.split())) for w in kept_words]
    breaks = break_after or set()

    # 各 i の「直近文節開始」= i 以下で最大の (break_after 該当 +1)、または直近文末+1
    out: list[tuple[float, float, int, int, int]] = []
    seg_start = 0  # 現在の文/積み上げの開始 index
    bright_lo = 0
    for i, w in enumerate(kept_words):
        # ウィンドウ下限: seg_start から、累積文字が cap_chars 以内に収まる範囲
        win_lo = seg_start
        total = sum(lens[win_lo : i + 1])
        while total > cap_chars and win_lo < i:
            total -= lens[win_lo]
            win_lo += 1
        start = w.new_start
        end = kept_words[i + 1].new_start if i + 1 < n else w.new_end
        if end <= start:
            end = start + 0.05
        out.append((start, end, win_lo, i, bright_lo))

        # 文節境界 / 文末で bright_lo / seg_start を更新
        if i in breaks:
            bright_lo = i + 1
        if w.word and w.word[-1] in _SENTENCE_PUNCT:
            seg_start = i + 1
            bright_lo = i + 1
    return out


def build_ass_rolling(
    kept_words: list[KeptWord],
    *,
    res_x: int = 1080,
    res_y: int = 1920,
    font: str = "Hiragino Sans",
    fontsize: int | None = None,
    outline: int = 2,
    margin_h: int | None = None,
    margin_v: int | None = None,
    max_chars: int | None = None,
    max_lines: int = 4,
    break_after: set[int] | None = None,
    role_spans: list[tuple[float, float, str]] | None = None,
    title_header: list[str] | None = None,
    title_seconds: float = 2.0,
) -> str:
    """ローリング/スクロール式の shusantv 風字幕 ASS を生成する。

    語の進行に合わせ、各語の出現時刻で「直近ウィンドウ」を 1 Dialogue として発行する。
    最新文節は明色 (#ffeebe)、それ以前のウィンドウ内語はグレー (#a9a9a2)。下揃えで
    テキスト増加とともに上へ伸び、画面下 ~1/4 を使う。大きいフォント前提。
    """
    scale = res_x / 1080.0
    if fontsize is None:
        fontsize = round(66 * scale)
    if margin_h is None:
        margin_h = round(48 * scale)
    if margin_v is None:
        margin_v = round(res_y * 0.06)
    if max_chars is None:
        usable = res_x - 2 * margin_h
        max_chars = max(8, int(usable / fontsize))

    cap_chars = max_chars * max_lines
    windows = _rolling_windows(kept_words, cap_chars, break_after)
    header = _ass_header(res_x, res_y, font, fontsize, outline, margin_h, margin_v)

    lines = [header]
    # 常時ヘッダー: タイトル消滅後〜clip終端まで上部に出し続ける。
    if title_header and kept_words:
        banner = _banner_event(title_header, title_seconds, kept_words[-1].new_end)
        if banner:
            lines.append(banner)
    for start, end, win_lo, i, bright_lo in windows:
        dim = "".join(kept_words[j].word for j in range(win_lo, max(win_lo, bright_lo)))
        bright = "".join(kept_words[j].word for j in range(max(win_lo, bright_lo), i + 1))
        # ASS インライン色タグは末尾 & が必要 (&H..BBGGRR&)
        bright_colour = _role_colour_at(start, role_spans)
        text = ""
        if dim:
            text += f"{{\\c{_DIM_COLOUR}&}}{_ass_escape(dim)}"
        text += f"{{\\c{bright_colour}&}}{_ass_escape(bright)}"
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def _paginate(
    kept_words: list[KeptWord],
    page_chars: int,
    break_after: set[int] | None,
) -> list[tuple[int, int]]:
    """kept_words を「3行ぶん (page_chars 文字以内)」のページ [lo, hi) に分割する。

    **文単位優先**: 原則として文末 (。!?) でのみページを区切る。ただし 1 文が
    page_chars に収まらない場合に限り、文節境界 (break_after) で途中分割する
    (どうしようもない時だけ文節で切る)。文節も無いまま超過したら強制的に閉じる。
    """
    n = len(kept_words)
    if n == 0:
        return []
    lens = [len("".join(w.word.split())) for w in kept_words]
    breaks = break_after or set()

    pages: list[tuple[int, int]] = []
    lo = 0
    cur = 0
    for i in range(n):
        cur += lens[i]
        w = kept_words[i]
        is_sentence_end = bool(w.word) and w.word[-1] in _SENTENCE_PUNCT
        is_clause = i in breaks
        over = cur > page_chars

        if is_sentence_end and not over:
            # 文末かつ容量内 → きれいに 1 ページ (文単位)。
            pages.append((lo, i + 1))
            lo = i + 1
            cur = 0
        elif over:
            # 容量超過 → どうしようもないので、ここまでで最も後ろの文節/文末境界で割る。
            # 直近に文節境界があればそこ、無ければこの位置で強制分割。
            split = i
            if not (is_clause or is_sentence_end):
                # i より手前で最後の境界を探す
                for j in range(i - 1, lo, -1):
                    if j in breaks or (
                        kept_words[j].word and kept_words[j].word[-1] in _SENTENCE_PUNCT
                    ):
                        split = j
                        break
            pages.append((lo, split + 1))
            lo = split + 1
            cur = sum(lens[lo : i + 1])
    if lo < n:
        pages.append((lo, n))
    return pages


def build_ass_karaoke(
    kept_words: list[KeptWord],
    *,
    res_x: int = 1080,
    res_y: int = 1920,
    font: str = "Hiragino Sans",
    fontsize: int | None = None,
    outline: int = 3,
    margin_h: int | None = None,
    margin_v: int | None = None,
    max_chars: int | None = None,
    max_lines: int = 3,
    break_after: set[int] | None = None,
    title: str = "",
    title_seconds: float = 2.0,
    title_header: list[str] | None = None,
    role_spans: list[tuple[float, float, str]] | None = None,
) -> str:
    """カラオケ式字幕 ASS を生成する (3行全文を表示し、発話中の語だけ明色)。

    全文を 3 行ぶんのページに分割し、各ページ内で語ごとに Dialogue を発行する。
    そのページの全テキストを表示しつつ、**今発話している語だけ明色 (#ffeebe)、
    前後はグレー (#a9a9a2)**。3 行を超える前にページを送る。大きいフォント (既定 96px)。
    title を渡すと冒頭 title_seconds 秒だけ画面上部に大きく表示する。
    title_header は要旨の上に小さめで積む定型見出し (日付/院/委員会/議員 等)。
    """
    scale = res_x / 1080.0
    if fontsize is None:
        fontsize = round(96 * scale)  # 倍近い大きさ
    if margin_h is None:
        margin_h = round(40 * scale)
    if margin_v is None:
        margin_v = round(res_y * 0.07)
    if max_chars is None:
        usable = res_x - 2 * margin_h
        max_chars = max(6, int(usable / fontsize))

    page_chars = max_chars * max_lines
    pages = _paginate(kept_words, page_chars, break_after)
    title_fontsize = _fit_title_fontsize(title, res_x, margin_h, fontsize)
    title_layout = _build_title_layout(
        title, title_header, res_x=res_x, res_y=res_y,
        margin_h=margin_h, title_fontsize=title_fontsize,
    )
    header = _ass_header(
        res_x, res_y, font, fontsize, outline, margin_h, margin_v,
        title_fontsize=title_fontsize, title_layout=title_layout,
    )

    breaks = break_after or set()
    lines = [header]
    if title_layout is not None:
        title_line = _title_event(
            title, res_x, title_seconds,
            margin_h=margin_h, title_fontsize=title_fontsize, res_y=res_y,
            header_lines=title_header, layout=title_layout,
        )
        if title_line:
            lines.append(title_line)
    # 常時ヘッダー: タイトル消滅後〜clip終端まで上部に出し続ける。
    if title_header and kept_words:
        banner = _banner_event(title_header, title_seconds, kept_words[-1].new_end)
        if banner:
            lines.append(banner)
    for lo, hi in pages:
        page = kept_words[lo:hi]
        if not page:
            continue
        # ページ内の語を「ハイライト単位 (文節)」にまとめる。break_after の境界で区切る。
        # 各単位は (unit_index, [page内word index...])。char にも unit_index を割り当てる。
        units = _highlight_units(lo, hi, breaks)
        word_unit: list[int] = [0] * len(page)
        for ui, idxs in enumerate(units):
            for j in idxs:
                word_unit[j - lo] = ui

        # ページ全文を文字単位に展開 (改行用)。各文字に「単位index」を付与。
        page_chars_list: list[str] = []
        char_unit: list[int] = []
        for wi, w in enumerate(page):
            page_chars_list.extend(list(w.word))
            char_unit.extend([word_unit[wi]] * len(w.word))
        # 改行は文節 (ハイライト単位) の境界で行い、文節を途中で割らない。
        line_break_at = _wrap_break_positions(
            page_chars_list, max_chars, char_unit=char_unit
        )

        # 単位ごとに 1 Dialogue。その単位の時間 [最初の語start, 最後の語end) を使う。
        for ui, idxs in enumerate(units):
            first, last = idxs[0], idxs[-1]
            start = kept_words[first].new_start
            end = (
                kept_words[last + 1].new_start
                if last + 1 < hi else kept_words[last].new_end
            )
            if end <= start:
                end = start + 0.05
            bright_colour = _role_colour_at(start, role_spans)
            text = _karaoke_line(
                page_chars_list, char_unit, line_break_at, bright_word=ui,
                bright_colour=bright_colour,
            )
            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}"
            )
    return "\n".join(lines) + "\n"


def _highlight_units(lo: int, hi: int, breaks: set[int]) -> list[list[int]]:
    """[lo, hi) の語インデックスを break_after 境界で「文節単位」にまとめる。

    返値は各単位の global word index リストのリスト。break_after が無ければ
    全体で 1 単位 (= ページ全体が 1 ハイライト) になるが、通常は LLM 文節境界で割れる。
    """
    units: list[list[int]] = []
    cur: list[int] = []
    for j in range(lo, hi):
        cur.append(j)
        if j in breaks:
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return units


# 禁則: 行頭に来てはいけない文字 (句読点・閉じ括弧・長音/小書き等)
_NO_LINE_START = set("。、，．！？!?）」』】〕》〉”’ゝ々ー…・")
# 禁則: 行末に来てはいけない文字 (開き括弧)
_NO_LINE_END = set("（「『【〔《〈“‘")


def _wrap_break_positions(
    chars: list[str], max_chars: int, char_unit: list[int] | None = None
) -> set[int]:
    """改行する文字インデックス集合を返す (その文字の前で改行)。

    char_unit が与えられた場合は **文節 (ハイライト単位) の境界でのみ改行** し、
    文節を途中で割らない。行に次の文節を足すと max_chars を超えるなら、その文節の
    手前で改行する。1 文節だけで max_chars を超える場合に限り単位内で強制改行する。

    char_unit が無い場合は従来通り max_chars ごとに改行 (禁則の微調整付き)。
    """
    breaks: set[int] = set()
    n = len(chars)
    if max_chars <= 0 or n == 0:
        return breaks

    if char_unit is not None:
        # 文節境界 = char_unit が変わる位置でのみ改行する。
        line_start = 0
        i = 0
        while i < n:
            # 現在の文節 [i, unit_end) を求める
            u = char_unit[i]
            unit_end = i
            while unit_end < n and char_unit[unit_end] == u:
                unit_end += 1
            unit_len = unit_end - i
            cur_line_len = i - line_start
            if cur_line_len > 0 and cur_line_len + unit_len > max_chars:
                # この文節を足すと溢れる → 文節の手前で改行
                breaks.add(i)
                line_start = i
            # 1 文節だけで 1 行に収まらない場合のみ、単位内を max_chars で強制分割
            if unit_end - line_start > max_chars:
                pos = line_start + max_chars
                while pos < unit_end:
                    breaks.add(pos)
                    pos += max_chars
                line_start = pos - max_chars
            i = unit_end
        return _apply_kinsoku(chars, breaks)

    # フォールバック: 文字数ベース + 禁則微調整 (char_unit 無し)
    col = 0
    for i in range(n):
        if col >= max_chars and i > 0:
            pos = i
            shifted = 0
            while pos < n and chars[pos] in _NO_LINE_START and shifted < 4:
                pos += 1
                shifted += 1
            while pos - 1 > 0 and chars[pos - 1] in _NO_LINE_END:
                pos -= 1
            if pos < n:
                breaks.add(pos)
                col = i - pos
            else:
                col = 0
        col += 1
    return breaks


def _is_latin(ch: str) -> bool:
    """ラテン文字・数字・関連記号 (英単語/数値を構成し、途中で割りたくない文字)。"""
    return ch.isascii() and (ch.isalnum() or ch in "._-/%")


def _apply_kinsoku(chars: list[str], breaks: set[int]) -> set[int]:
    """改行位置集合に禁則処理を適用して返す。

    - 行頭禁則: break 位置が句読点/閉じ括弧ならその連続を前行末に残す (後ろへずらす)。
    - 行末禁則: break 直前が開き括弧なら手前へずらす。
    - 英単語/数値の途中で割らない: break がラテン文字の連続の内側なら、その run の
      先頭まで手前にずらす (例: 「AI」が A|I にならない)。
    ずらした結果 0 や行外になる break は捨てる。
    """
    n = len(chars)
    out: set[int] = set()
    for b in sorted(breaks):
        pos = b
        # 行頭が禁則文字: 後ろへ (最大4文字)
        shifted = 0
        while pos < n and chars[pos] in _NO_LINE_START and shifted < 4:
            pos += 1
            shifted += 1
        # 行末が開き括弧: 手前へ
        while pos - 1 > 0 and chars[pos - 1] in _NO_LINE_END:
            pos -= 1
        # ラテン文字 run の内側なら run 先頭へ (英単語を割らない)
        if 0 < pos < n and _is_latin(chars[pos - 1]) and _is_latin(chars[pos]):
            while pos > 0 and _is_latin(chars[pos - 1]) and _is_latin(chars[pos]):
                pos -= 1
        if 0 < pos < n:
            out.add(pos)
    return out


def _karaoke_line(
    chars: list[str], char_word: list[int], break_at: set[int], bright_word: int,
    bright_colour: str = _TEXT_COLOUR,
) -> str:
    """文字列を色タグ + 改行付き ASS テキストに組み立てる。

    bright_word の語に属する文字だけ明色 (bright_colour)、他はグレー。break_at の
    位置で \\N。連続同色はまとめてタグ数を抑える。bright_colour は話者色 (質疑=ミント,
    答弁=オレンジ) を渡せる。
    """
    parts: list[str] = []
    cur_bright: bool | None = None
    buf = ""

    def flush_buf() -> None:
        nonlocal buf, cur_bright
        if buf:
            colour = bright_colour if cur_bright else _DIM_COLOUR
            parts.append(f"{{\\c{colour}&}}{_ass_escape(buf)}")
            buf = ""

    for i, ch in enumerate(chars):
        if i in break_at:
            flush_buf()
            parts.append("\\N")
            cur_bright = None
        is_bright = char_word[i] == bright_word
        if cur_bright is None or is_bright != cur_bright:
            flush_buf()
            cur_bright = is_bright
        buf += ch
    flush_buf()
    return "".join(parts)


def _wrap_text_plain(text: str, max_chars: int) -> str:
    """1 行テキストを max_chars で折り返し (禁則・英単語非分割) \\N を入れて返す。"""
    chars = list(text)
    breaks = _apply_kinsoku(chars, _wrap_break_positions(chars, max_chars))
    out = ""
    for i, c in enumerate(chars):
        if i in breaks:
            out += "\\N"
        out += _ass_escape(c)
    return out


def build_ass_from_captions(
    captions: list,
    *,
    res_x: int = 1080,
    res_y: int = 1920,
    font: str = "Hiragino Sans",
    fontsize: int | None = None,
    outline: int = 3,
    margin_h: int | None = None,
    margin_v: int | None = None,
    max_chars: int | None = None,
    title: str = "",
    title_seconds: float = 2.0,
    title_header: list[str] | None = None,
    role_spans: list[tuple[float, float, str]] | None = None,
) -> str:
    """編集済みキャプション (start/end/text を持つ) から ASS を生成する。

    手修正後の字幕用。各キャプションをその時間帯に 1 Dialogue として表示する
    (プレーン表示)。折り返し/禁則/タイトルは通常スタイルと共通。
    captions の要素は .start/.end/.text 属性を持つ (EditCaption 等)。
    role_spans (カット後時間, role) があれば、各キャプションの色を話者で分ける
    (質疑=ミント, 答弁=オレンジ)。
    """
    scale = res_x / 1080.0
    if fontsize is None:
        fontsize = round(96 * scale)
    if margin_h is None:
        margin_h = round(40 * scale)
    if margin_v is None:
        margin_v = round(res_y * 0.07)
    if max_chars is None:
        usable = res_x - 2 * margin_h
        max_chars = max(6, int(usable / fontsize))

    title_fontsize = _fit_title_fontsize(title, res_x, margin_h, fontsize)
    title_layout = _build_title_layout(
        title, title_header, res_x=res_x, res_y=res_y,
        margin_h=margin_h, title_fontsize=title_fontsize,
    )
    header = _ass_header(
        res_x, res_y, font, fontsize, outline, margin_h, margin_v,
        title_fontsize=title_fontsize, title_layout=title_layout,
    )
    lines = [header]
    if title_layout is not None:
        tl = _title_event(
            title, res_x, title_seconds,
            margin_h=margin_h, title_fontsize=title_fontsize, res_y=res_y,
            header_lines=title_header, layout=title_layout,
        )
        if tl:
            lines.append(tl)
    # 常時ヘッダー: タイトル消滅後〜最後のキャプション終端まで。
    if title_header and captions:
        clip_end = max((c.end for c in captions), default=0.0)
        banner = _banner_event(title_header, title_seconds, clip_end)
        if banner:
            lines.append(banner)

    for c in captions:
        text = (c.text or "").strip()
        if not text:
            continue
        body = _wrap_text_plain(text, max_chars)
        if role_spans is not None:
            colour = _role_colour_at((c.start + c.end) / 2.0, role_spans)
            body = f"{{\\c{colour}&}}" + body
        lines.append(
            f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Default,,0,0,0,,{body}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------


def _srt_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(
    kept_words: list[KeptWord],
    *,
    max_chars: int = MAX_LINE_CHARS,
    break_after: set[int] | None = None,
) -> str:
    """kept_words から SRT 文字列を生成する (new タイムライン)。"""
    captions = group_captions(kept_words, max_chars=max_chars, break_after=break_after)
    blocks: list[str] = []
    idx = 1
    for c in captions:
        if not c.text:
            continue
        blocks.append(
            f"{idx}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n{c.text}\n"
        )
        idx += 1
    return "\n".join(blocks)
