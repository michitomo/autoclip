"""Whisper 文字起こし (DeepInfra whisper-large-v3-turbo)

OpenAI 互換クライアントを使用して DeepInfra API を呼び出す。

[プロンプト戦略 V3 — suffix-only]
ablation 実験（56223 本会議、5セグメント、3モード比較）の結果:
  full        avg_logprob=-0.208  汚染あり（宮田大臣、民間民主党）
  suffix-only avg_logprob=-0.173  汚染なし・句読点あり・話者名正確
  none        avg_logprob=-0.142  句読点なし・先頭ハルシネーションあり

静的ブロック（閣僚名・政党名・議長名）が隣接音声の固有名詞に染み出し
誤認識を誘発することが判明。動的サフィックスのみに絞ることで
汚染を除去しつつ句読点・話者文脈を維持する。

_WHISPER_PROMPT_BASE は実験用スクリプト（scripts/compare_whisper_models.py）
が参照するため残しているが、本番パイプラインでは使用しない。

[オーバーラップウィンドウ — チャンク境界ロス対策]
DeepInfra Whisper API は内部で 30s 単位にチャンキングしており、その境界で
発言が途切れる現象が確認されている (docs/whisper-chunk-boundary-investigation.md)。
クライアント側で `WHISPER_WINDOW_SECONDS` (< 30s) のウィンドウ × `WHISPER_OVERLAP_SECONDS`
の重複で WAV を分割し並列に Whisper を呼び、midpoint ベースの trust window で
重複を取り除いてマージする。各ウィンドウは Whisper 内部チャンク 1 個に収まるため、
ウィンドウ内では境界ロスが発生せず、ウィンドウ間の境界はオーバーラップで保護される。
"""

from __future__ import annotations

import io
import logging
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import openai

from src.api_client import with_retry
from src.models import (
    RawTranscript,
    SegmentTranscript,
    SpeakerInfo,
    WhisperSegment,
    WhisperWord,
)

logger = logging.getLogger(__name__)

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# DeepInfra と Groq でモデル ID が異なる (DeepInfra は openai/ プレフィックス付き)。
# 両者とも OpenAI 互換で timestamp_granularities=["word","segment"] に対応。
_WHISPER_MODEL_BY_PROVIDER = {
    "deepinfra": "openai/whisper-large-v3-turbo",
    "groq": "whisper-large-v3-turbo",
}
# 既定 (DeepInfra) のモデル ID。後方互換のため公開名は WHISPER_MODEL のまま。
WHISPER_MODEL = _WHISPER_MODEL_BY_PROVIDER["deepinfra"]

# ---------------------------------------------------------------------------
# オーバーラップウィンドウ設定
# ---------------------------------------------------------------------------
# WHISPER_WINDOW_SECONDS: Whisper 内部チャンクサイズ (30s) より小さく取り、
#   1 ウィンドウが内部チャンク 1 個に収まることを保証する。
# WHISPER_OVERLAP_SECONDS: 隣接ウィンドウの重複秒数。境界ロスがあっても両側の
#   ウィンドウのどちらかで完全な発言が取れることを期待する。
#   注: 大きくしすぎる (例 15s) と隣接窓のセグメントが重なりすぎ、_dedupe_overlap_text
#   が長い本文セグメントを near-duplicate と誤判定して落とすため逆効果 (実測で欠落悪化)。
#   5s が安定。発話欠落の本質的な対策は _merge_window_results 側のカバレッジ補完。
# WHISPER_MIN_WINDOWED_DURATION: これ以下のセグメントはオーバーラップ処理を行わず
#   従来通り単一 API 呼び出しで処理する (調査コストの節約)。
# ---------------------------------------------------------------------------
WHISPER_WINDOW_SECONDS: float = 25.0
WHISPER_OVERLAP_SECONDS: float = 5.0
WHISPER_MIN_WINDOWED_DURATION: float = WHISPER_WINDOW_SECONDS
# 1 セグメント内のウィンドウ並列数。26分質疑 ≈ 79 窓を 8 並列だと 10 バッチで遅い。
# DeepInfra のレート制限が許す範囲で上げると文字起こしが大幅短縮 (429 は with_retry が
# 指数バックオフで吸収)。env WHISPER_WINDOW_CONCURRENCY で調整可。
WHISPER_WINDOW_CONCURRENCY: int = int(
    os.environ.get("WHISPER_WINDOW_CONCURRENCY", "24")
)

# ---------------------------------------------------------------------------
# 第221回国会（令和8年特別会）対応 Whisper プロンプト  [Prompt V2]
# ---------------------------------------------------------------------------
# Whisperのpromptは「指示」ではなく「直前の文脈」として機能する（スタイル模倣）。
# 224トークン制限内に収め、かつトークンループを抑制するための設計方針:
#
# [V1→V2 変更点と根拠]
# 1. 「石井啓一副議長」削除
#    → 実データで「石井啓一議長、石井啓一議長...」25秒ループが複数確認。
#      プロンプト末尾に固定配置されていたため、本会議セッションで特に危険。
#
# 2. 全法律名（健康保険法〜労働者災害補償保険法）削除
#    → 「社会福祉法」が「福祉法、福祉法...」24秒ループを誘発（9件確認）。
#      法律名はセッション固有かつトークン消費大（~40トークン）のため除去。
#      委員会固有の法律名は将来的に動的サフィックスで追加可能。
#
# 3. 動的サフィックスを「出席議員: 全員列挙」から「{委員会}。{発言者}：」へ変更
#    → 「出席議員」リストが最多ループ誘発源（42件）。任意の出席者名が
#      音響的に不明瞭な区間でループ起点になっていた。
#    → 新形式は議事録の自然な「直前テキスト」として機能し、トークン消費も
#      ~133トークン→~15トークンに削減。全プロンプトが224制限内に確実に収まる。
#
# [維持した要素]
# - 主要閣僚7名: 小泉進次郎・片山さつき・茂木敏充は100%正確認識を確認済み
# - 主要政党名: 国民民主党（50%誤認問題あり）・日本維新の会（69%誤認）等の改善に期待
# - 森英介議長（副議長は除外）
#
# [V2.1 拡張 (PR5, §2.6)]
# 目的: whisper_misrecognition (269件) のうち参議院議長・社会民主党関連の
#       Whisper 第一通過誤認を抑制する。
#
# 変更点:
#   - 「衆議院の」→「の」: 参議院セッションでも使うため (+suffix で chamber は伝わる)
#   - 政党に「社会民主党」を追加 (元々抜けていた)
#   - 議長を「森英介議長」→「森英介衆議院議長、関口昌一参議院議長」に拡張
#     (参議院セッションで議長 context が皆無だった問題を解消)
#   - 閣僚 7名 (V2 から維持) + 第2次高市内閣の他 9名は token 予算外のため
#     transcript_corrector のフルリスト (大臣 16名) に委任
#
# Token 予算注意:
#   Whisper prompt は 224-token 制限。overflow 時は **冒頭側を truncate**。
#   現状 base 209 + 動的 suffix 18-31 = 227-240 (約 -3〜-16 tokens overflow)。
#   truncate は冒頭の "第221回国会の質疑応答。" (~10 tokens) で起こり、
#   閣僚名・政党名・議長名・動的 suffix は preserve される。
#   → 年号は corrector でカバーされるため受容できる範囲。
#
# [既知の残存問題]
# - 高市早苗: 91%で「高市」に末尾省略。音声上での省略発言か Whisper 誤認かは未確定。
# - 国民民主党: 50%で「国民民主」に末尾省略。「党」の音が弱い可能性。
# - 安倍内閣ハルシネーション: Whisper学習バイアスで稀に出現（1件確認）。
#
# 議長: 森英介(衆議院議長)、石井啓一(衆議院副議長)※プロンプト外
#       関口昌一(参議院議長)、福山哲郎(参議院副議長)※プロンプト外
# ---------------------------------------------------------------------------

_WHISPER_PROMPT_BASE = (
    "第221回国会の質疑応答。"
    "高市早苗内閣総理大臣、木原稔内閣官房長官、茂木敏充外務大臣、"
    "片山さつき財務大臣、上野賢一郎厚生労働大臣、"
    "赤澤亮正経済産業大臣、小泉進次郎防衛大臣。"
    "自由民主党、立憲民主党、日本維新の会、公明党、日本共産党、"
    "国民民主党、チームみらい、参政党、れいわ新選組、日本保守党、社会民主党。"
    "森英介衆議院議長、関口昌一参議院議長。"
)


def strip_prompt_echo(
    words: list[WhisperWord],
    speaker: SpeakerInfo,
    committee: str,
    *,
    max_lead_chars: int = 60,
) -> list[WhisperWord]:
    """冒頭の議事録ヘッダ echo (「{委員会}。」「{氏名}（{所属}）：」「{氏名}委員長：」
    「{氏名}君。」等) を語列の先頭から繰り返し除去する。

    これらは Whisper のプロンプト echo / corrector の話者ラベルであって発話ではない。
    先頭 max_lead_chars 以内に出現する「：」「。」区切りのヘッダ断片を、本文が始まる
    まで剥がす。委員長・他議員名でも剥がす (誰のラベルでも非発話なので)。
    本物の発話を誤って削らないよう、ヘッダらしさ (役職語・（）・委員会名・氏名・君。)
    を伴う場合のみ剥がす。
    """
    if not words:
        return words

    # 先頭領域の連結テキストと各文字→語index
    lead = ""
    char_word_idx: list[int] = []
    for i, w in enumerate(words):
        t = "".join(w.word.split())
        for _ in t:
            char_word_idx.append(i)
        lead += t
        if len(lead) >= max_lead_chars:
            break

    # ヘッダ断片を示す手がかり語
    role_markers = ("委員長", "君", "大臣", "議長", "委員", "参考人", "副大臣", "政務官")
    # 「氏名（所属）」判定用の名前断片 (発言者名の先頭2文字)
    name_head = speaker.name[:2] if speaker.name else ""
    # 冒頭に紛れる Whisper ハルシネーション定型句 (発話ではない)。
    # lead_pad で前話者の無音/末尾を拾うと出やすい。
    halluc_phrases = (
        "ありがとうございました",
        "ご視聴ありがとう",
        "ご清聴ありがとう",
        "字幕",
        "チャンネル登録",
    )

    drop_chars = 0  # 先頭から剥がす文字数
    cursor = 0
    while cursor < len(lead):
        # 次の区切り (：/:/。/）) を探す
        seps = [
            lead.find(s, cursor)
            for s in ("：", ":", "。", "）")
        ]
        candidates = [p for p in seps if p >= 0]
        if not candidates:
            break
        sep = min(candidates)
        sep_char = lead[sep]
        head = lead[cursor : sep + 1]

        if sep_char == "）":
            # 「氏名（所属）」形のラベルだけ剥がす (本物の括弧発話を誤除去しない)。
            # head に「（」があり、かつ発言者名/委員会名/役職語を含む場合のみ。
            is_header = "（" in head and (
                (name_head and name_head in head)
                or (committee and committee in head)
                or any(r in head for r in role_markers)
            )
        else:
            # ：/:/。 区切り: 役職語・（）・委員会名・ハルシネーション句のいずれかを
            # 伴う場合のみ剥がす。単なる「…：」は本物の発話にもあるため不可 (誤爆防止)。
            has_aff = bool(speaker.affiliation) and speaker.affiliation[:3] in head
            # 発言者名を含むが所属を含まない冒頭断片 = 呼名/誤認識のノイズ
            # (本人の自己紹介「{所属}の{氏名}です」は所属を含むので除外)。
            name_noise = bool(name_head) and name_head in head and not has_aff
            is_header = (
                (committee and committee in head)
                or "（" in head
                or "）" in head
                or any(r in head for r in role_markers)
                or any(p in head for p in halluc_phrases)
                or name_noise
            )

        # 既に冒頭のヘッダ/ハルシネーションを剥がした後で、短い断片 (≤8字) が続く場合は
        # 境界ノイズ (誤認識された呼名の断片など) とみなして続けて剥がす。ただし発言者の
        # 所属を含む = 本人の自己紹介の可能性があるので止める。
        is_short_noise = (
            drop_chars > 0
            and sep - cursor <= 8
            and not (speaker.affiliation and speaker.affiliation[:3] in head)
        )

        if (is_header or is_short_noise) and sep - cursor <= 25:
            end = sep + 1
            # 「）」直後に「：」が続く議事録ラベル形なら、その「：」も一緒に落とす。
            if sep_char == "）" and end < len(lead) and lead[end] in "：:":
                end += 1
            drop_chars = end
            cursor = end
            continue
        break

    # 後処理: ヘッダ echo を剥がした直後に残る「孤立した開き括弧/空白」を落とす。
    # 議事録形式「氏名（所属）「発話…」では閉じ括弧の後に開き鉤括弧だけが残るが、
    # これは発話ではない (実音声には無い)。echo を1つ以上剥がした後のみ実施し、開き
    # 括弧/空白だけを除去する (本物の文頭括弧発話「（令和8年度）について」は echo を
    # 剥がしていない = drop_chars==0 なので触れない)。
    open_brackets = "「『（〔【《〈［｛“"
    if drop_chars > 0:
        while drop_chars < len(lead):
            ch = lead[drop_chars]
            if ch.isspace() or ch in open_brackets:
                drop_chars += 1
            else:
                break

    if drop_chars == 0:
        return words
    drop_word_idx = char_word_idx[drop_chars - 1]
    return words[drop_word_idx + 1 :]


def _build_whisper_prompt(
    speaker: SpeakerInfo,
    committee: str,
) -> str:
    """セグメント固有のWhisperプロンプトを構築する（suffix-only モード）。

    ablation 実験により、静的ブロック（_WHISPER_PROMPT_BASE）は logprob を
    悪化させ固有名詞汚染を誘発することが判明。動的サフィックスのみを使用する。

    「{委員会}。{発言者名}（{所属}）：」という議事録形式で Whisper に
    話者文脈を与え、句読点スタイルを維持する。
    """
    committee_label = committee or "委員会"
    return f"{committee_label}。{speaker.name}（{speaker.affiliation}）："


def _get_provider() -> str:
    """ASR プロバイダ名を返す ("deepinfra" 既定 | "groq")。"""
    provider = os.environ.get("ASR_PROVIDER", "deepinfra").lower()
    if provider not in _WHISPER_MODEL_BY_PROVIDER:
        raise ValueError(
            f"Unsupported ASR_PROVIDER={provider!r} "
            f"(expected one of {sorted(_WHISPER_MODEL_BY_PROVIDER)})"
        )
    return provider


def _get_model() -> str:
    """現在のプロバイダ向け Whisper モデル ID を返す。"""
    return _WHISPER_MODEL_BY_PROVIDER[_get_provider()]


def _get_client() -> openai.OpenAI:
    """ASR_PROVIDER に応じた OpenAI 互換クライアントを返す (DeepInfra 既定)。"""
    provider = _get_provider()
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise OSError("GROQ_API_KEY environment variable is not set")
        return openai.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise OSError("DEEPINFRA_API_KEY environment variable is not set")
    return openai.OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)


@dataclass
class _WavData:
    """WAV ファイルから読み出した PCM フレームとフォーマット情報。"""

    frames: bytes
    nchannels: int
    sampwidth: int
    framerate: int

    @property
    def duration(self) -> float:
        if not self.framerate:
            return 0.0
        return len(self.frames) / (self.nchannels * self.sampwidth * self.framerate)


def _read_wav(wav_path: Path) -> _WavData:
    """WAV ファイル全体を PCM フレームとして読み込む。"""
    with wave.open(str(wav_path), "rb") as w:
        nchannels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        frames = w.readframes(nframes)
    return _WavData(
        frames=frames,
        nchannels=nchannels,
        sampwidth=sampwidth,
        framerate=framerate,
    )


def _get_wav_duration(wav_path: Path) -> float:
    """WAV ファイルの長さ (秒) を返す。読み取れない場合は 0.0 を返す。

    `wave.open` 失敗時 (空ファイル・非WAV) は短パス選択のため 0.0 を返す。
    本番の WAV (ffmpeg 出力) では常に正常に読める想定。
    """
    try:
        with wave.open(str(wav_path), "rb") as w:
            framerate = w.getframerate()
            nframes = w.getnframes()
            if not framerate:
                return 0.0
            return nframes / framerate
    except (wave.Error, EOFError, FileNotFoundError) as e:
        logger.debug("Could not read WAV duration from %s: %s", wav_path, e)
        return 0.0


def _slice_wav_window(wav: _WavData, start_sec: float, end_sec: float) -> bytes:
    """WAV から [start_sec, end_sec) を切り出した WAV ファイルバイト列を返す。"""
    bytes_per_frame = wav.nchannels * wav.sampwidth
    start_frame = max(0, int(start_sec * wav.framerate))
    end_frame = min(len(wav.frames) // bytes_per_frame, int(end_sec * wav.framerate))
    sliced = wav.frames[start_frame * bytes_per_frame : end_frame * bytes_per_frame]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(wav.nchannels)
        w.setsampwidth(wav.sampwidth)
        w.setframerate(wav.framerate)
        w.writeframes(sliced)
    return buf.getvalue()


def _compute_windows(
    duration: float,
    window_seconds: float = WHISPER_WINDOW_SECONDS,
    overlap_seconds: float = WHISPER_OVERLAP_SECONDS,
) -> list[tuple[float, float]]:
    """[start, end) のウィンドウリストを生成する。

    最後のウィンドウは端を超えない (clip)。極端に短い末尾ウィンドウは
    一つ手前のウィンドウに統合せずそのまま採用する (内容を捨てないため)。
    """
    if duration <= 0:
        return []
    if duration <= window_seconds:
        return [(0.0, duration)]

    step = window_seconds - overlap_seconds
    if step <= 0:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be less than "
            f"window_seconds ({window_seconds})"
        )

    windows: list[tuple[float, float]] = []
    offset = 0.0
    while offset < duration:
        end = min(offset + window_seconds, duration)
        windows.append((offset, end))
        if end >= duration:
            break
        offset += step
    return windows


def _whisper_segment_to_dict(seg: object) -> dict:
    """Whisper API レスポンスのセグメント要素を dict に正規化する。"""
    if isinstance(seg, dict):
        d = dict(seg)
    else:
        d = {
            "id": getattr(seg, "id", 0),
            "seek": getattr(seg, "seek", 0),
            "start": getattr(seg, "start", 0.0),
            "end": getattr(seg, "end", 0.0),
            "text": getattr(seg, "text", ""),
            "tokens": list(getattr(seg, "tokens", [])),
            "temperature": getattr(seg, "temperature", 0.0),
            "avg_logprob": getattr(seg, "avg_logprob", 0.0),
            "compression_ratio": getattr(seg, "compression_ratio", 0.0),
            "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
        }
    d.setdefault("words", [])
    return d


def _whisper_word_to_dict(word: object) -> dict:
    """Whisper API レスポンスの語要素を {word,start,end} dict に正規化する。"""
    if isinstance(word, dict):
        return {
            "word": word.get("word", ""),
            "start": float(word.get("start", 0.0)),
            "end": float(word.get("end", 0.0)),
        }
    return {
        "word": getattr(word, "word", ""),
        "start": float(getattr(word, "start", 0.0)),
        "end": float(getattr(word, "end", 0.0)),
    }


def _bucket_words_into_segments(
    segments: list[dict], words: list[dict]
) -> None:
    """トップレベル語リストを、各語の midpoint を含むセグメントへ振り分ける (in-place)。

    Whisper verbose_json はセグメントとは別にトップレベル `words` 配列を返す。
    JetCut が扱いやすいよう、各セグメントの "words" にその区間内の語を格納する。
    両者 start 昇順前提の単一パス。どのセグメントにも入らない語 (境界の隙間) は
    最も近いセグメント末尾に寄せず捨てる (セグメント外の語は通常存在しない)。
    """
    if not segments or not words:
        return
    si = 0
    n = len(segments)
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        # mid がセグメント区間に入るまで進める
        while si < n and mid >= segments[si]["end"]:
            si += 1
        if si >= n:
            break
        seg = segments[si]
        if mid >= seg["start"]:
            seg.setdefault("words", []).append(w)
        # mid < seg["start"] (隙間) の語はスキップ


def _call_whisper(
    client: openai.OpenAI,
    wav_bytes: bytes,
    prompt: str,
) -> tuple[str, list[dict]]:
    """Whisper API を一度呼び出し、(text, segments) を返す。

    timestamp_granularities=["word","segment"] を要求し、トップレベル `words` を
    各セグメントの "words" に振り分けた上で返す (DeepInfra/Groq で確認済みの形式)。
    """
    def _do() -> object:
        return client.audio.transcriptions.create(
            model=_get_model(),
            file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
            language="ja",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            prompt=prompt,
        )

    result = with_retry(_do)
    text = getattr(result, "text", "") or ""
    raw_segments = getattr(result, "segments", None) or []
    segments = [_whisper_segment_to_dict(s) for s in raw_segments]
    raw_words = getattr(result, "words", None) or []
    words = [_whisper_word_to_dict(w) for w in raw_words]
    _bucket_words_into_segments(segments, words)
    return text, segments


def _suffix_prefix_overlap(prev_text: str, curr_text: str, max_len: int = 80) -> int:
    """prev_text の末尾と curr_text の冒頭で一致する最長文字数を返す。

    例: prev="ABCDEF", curr="DEFGHI" → 3 ("DEF")

    空白文字 (半角・全角スペース) を無視して比較する。戻り値は curr_text の
    オリジナル文字列における trim すべき接頭辞の長さ。
    """
    prev_norm = "".join(prev_text.split())

    curr_indices: list[int] = []
    curr_norm_chars: list[str] = []
    for i, c in enumerate(curr_text):
        if not c.isspace():
            curr_indices.append(i)
            curr_norm_chars.append(c)
    curr_norm = "".join(curr_norm_chars)

    n_max = min(len(prev_norm), len(curr_norm), max_len)
    for n in range(n_max, 0, -1):
        if prev_norm[-n:] == curr_norm[:n]:
            # n chars in curr_norm → curr_text のインデックス curr_indices[n-1] + 1 まで trim
            return curr_indices[n - 1] + 1
    return 0


def _dedupe_overlap_text(
    segments: list[WhisperSegment],
    time_overlap_threshold: float = 0.5,
    min_suffix_prefix: int = 5,
) -> list[WhisperSegment]:
    """連続 segment 間の時間重複でテキストが重複している場合に削除/トリムする。

    midpoint ベースの trust window フィルタ後でも、Whisper が両側のウィンドウで
    境界近傍に segment を生成すると near-duplicate が残る。隣接 segment が時間
    重複していて、テキストが包含/suffix-prefix 関係にある場合のみ修正する。
    内容が異なる場合は両方残す (情報損失を避けるため保守的に動く)。

    処理:
        - 完全一致 → 後者を削除
        - 前者が後者の substring → 前者を後者で置換 (より完全な方を残す)
        - 後者が前者の substring → 後者を削除
        - suffix-prefix overlap (>= min_suffix_prefix 文字) → 後者の prefix を trim
        - それ以外 → 両方残す
    """
    if not segments:
        return segments

    result: list[WhisperSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        time_overlap_amount = prev.end - seg.start
        if time_overlap_amount <= time_overlap_threshold:
            result.append(seg)
            continue

        prev_norm = "".join(prev.text.split())
        curr_norm = "".join(seg.text.split())
        if not prev_norm or not curr_norm:
            result.append(seg)
            continue

        if curr_norm == prev_norm or curr_norm in prev_norm:
            # 後者は前者に包含 → drop
            continue
        if prev_norm in curr_norm:
            # 前者は後者に包含 → 前者をより完全な後者で置換
            result[-1] = seg
            continue

        n = _suffix_prefix_overlap(prev.text, seg.text, max_len=80)
        if n >= min_suffix_prefix:
            new_text = seg.text[n:]
            if new_text.strip():
                # 先頭 n 文字を落としたぶん、先頭側の語も整合させる。
                # 語は char offset を持たないため、落とした語テキストの累積長が
                # trim 量 n に届くまで先頭から除去する近似で同期する。
                new_words = _drop_leading_words(seg.words, n)
                trimmed = seg.model_copy(update={"text": new_text, "words": new_words})
                result.append(trimmed)
            # 全部 trim されたら何も残さない (情報は prev に既にある)
            continue

        result.append(seg)
    return result


def _drop_leading_words(
    words: list[WhisperWord], trimmed_chars: int
) -> list[WhisperWord]:
    """先頭 trimmed_chars 文字分に相当する語を先頭から除去する。

    語は文字オフセットを持たないため、除去語テキスト (空白除去) の累積長が
    trimmed_chars 以上になるまで先頭から落とす近似。境界の near-duplicate を
    取り除く目的なので 1〜2 語の誤差は許容範囲。
    """
    if not words or trimmed_chars <= 0:
        return list(words)
    consumed = 0
    drop = 0
    for w in words:
        if consumed >= trimmed_chars:
            break
        consumed += len("".join(w.word.split()))
        drop += 1
    return words[drop:]


def _merge_window_results(
    windows: list[tuple[float, float]],
    window_segments: list[list[dict]],
    overlap_seconds: float = WHISPER_OVERLAP_SECONDS,
) -> list[WhisperSegment]:
    """重複ウィンドウのセグメントを絶対タイムスタンプにマージする。

    1. 各ウィンドウに trust window を設定し、whisper セグメントの midpoint が
       trust window に含まれる場合のみ採用する。
    2. start 昇順でソート。
    3. 時間重複しテキストが包含/suffix-prefix の near-duplicate を削除/トリム。

    Trust window 定義:
        最初のウィンドウ: [w_start, w_end - overlap/2]
        中間ウィンドウ: [w_start + overlap/2, w_end - overlap/2]
        最後のウィンドウ: [w_start + overlap/2, w_end]

    Args:
        windows: 絶対秒での (start, end) ウィンドウリスト
        window_segments: ウィンドウごとの whisper セグメント dict リスト
            (start/end はウィンドウ相対秒)
        overlap_seconds: ウィンドウ間のオーバーラップ秒数

    Returns:
        絶対タイムスタンプに変換・マージ・id 再採番済みの WhisperSegment リスト
    """
    n = len(windows)
    if n != len(window_segments):
        raise ValueError("windows and window_segments must have same length")

    half_overlap = overlap_seconds / 2.0
    merged: list[WhisperSegment] = []

    for i, ((w_start, w_end), segments) in enumerate(zip(windows, window_segments)):
        trust_start = w_start if i == 0 else w_start + half_overlap
        trust_end = w_end if i == n - 1 else w_end - half_overlap

        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            abs_start = seg_start + w_start
            abs_end = seg_end + w_start
            midpoint = (abs_start + abs_end) / 2.0
            if not (trust_start <= midpoint < trust_end):
                continue

            # 語も同じ +w_start オフセットで絶対時間に変換し、セグメントに同伴させる。
            seg_words = [
                WhisperWord(
                    word=w.get("word", ""),
                    start=float(w.get("start", 0.0)) + w_start,
                    end=float(w.get("end", 0.0)) + w_start,
                )
                for w in seg.get("words", [])
            ]

            merged.append(
                WhisperSegment(
                    id=0,  # 後で再採番
                    seek=int(round(w_start * 100)),  # Whisper seek 単位 (1/100s)
                    start=abs_start,
                    end=abs_end,
                    text=seg.get("text", ""),
                    tokens=list(seg.get("tokens", [])),
                    temperature=float(seg.get("temperature", 0.0)),
                    avg_logprob=float(seg.get("avg_logprob", 0.0)),
                    compression_ratio=float(seg.get("compression_ratio", 0.0)),
                    no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                    words=seg_words,
                )
            )

    merged.sort(key=lambda s: (s.start, s.end))
    merged = _dedupe_overlap_text(merged)
    for idx, seg in enumerate(merged):
        seg.id = idx
    return merged


def _transcribe_windowed(
    wav: _WavData,
    prompt: str,
    window_seconds: float = WHISPER_WINDOW_SECONDS,
    overlap_seconds: float = WHISPER_OVERLAP_SECONDS,
    max_workers: int = WHISPER_WINDOW_CONCURRENCY,
    client: openai.OpenAI | None = None,
) -> tuple[str, list[WhisperSegment]]:
    """WAV をオーバーラップウィンドウで分割し、Whisper を並列呼び出ししてマージする。

    重要: 各窓の Whisper 呼び出しには **prompt を渡さない**。プロンプト
    (「{委員会}。{氏名}（{所属}）：」) を渡すと Whisper Turbo が長い連続発話の
    窓で出力を非決定的に途中で打ち切り、発話が丸ごと欠落する (実測: prompt 無は
    同一WAVで長さ完全一致、prompt 有は窓により 169→83 等にランダム短縮)。
    氏名表記・句読点は後段の LLM 校正が担うため、プロンプトの利益は冗長。
    引数 prompt は API 互換のため残すが窓 ASR には使わない。
    """
    if client is None:
        client = _get_client()

    windows = _compute_windows(wav.duration, window_seconds, overlap_seconds)
    if not windows:
        return "", []

    logger.info(
        "Windowed transcribe: duration=%.1fs, %d windows (window=%.1fs, overlap=%.1fs)",
        wav.duration, len(windows), window_seconds, overlap_seconds,
    )

    def _process_one(idx_window: tuple[int, tuple[float, float]]) -> tuple[int, list[dict]]:
        idx, (w_start, w_end) = idx_window
        wav_bytes = _slice_wav_window(wav, w_start, w_end)
        # prompt は渡さない (上の docstring 参照: 打ち切り誘発を回避)
        _, segments = _call_whisper(client, wav_bytes, "")
        return idx, segments

    results: list[list[dict]] = [[] for _ in windows]
    if len(windows) == 1:
        _, segs = _process_one((0, windows[0]))
        results[0] = segs
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, segs in executor.map(_process_one, list(enumerate(windows))):
                results[idx] = segs

    merged = _merge_window_results(windows, results, overlap_seconds)
    full_text = "".join(s.text for s in merged)
    return full_text, merged


def transcribe_segment(
    wav_path: Path,
    segment_index: int,
    speaker: SpeakerInfo,
    committee: str,
) -> SegmentTranscript:
    """1セグメントの WAV ファイルを文字起こしする。

    WAV が `WHISPER_MIN_WINDOWED_DURATION` 以下なら従来通り単一 API 呼び出しで
    処理する。それより長い場合は `_transcribe_windowed` でオーバーラップウィンドウ
    分割→並列呼び出し→マージを行う (Whisper 内部 30s チャンク境界での発言ロス対策)。

    Args:
        wav_path: セグメント WAV ファイルパス
        segment_index: セグメントインデックス
        speaker: このセグメントの主発言者
        committee: 委員会名（動的サフィックスに使用）

    Returns:
        SegmentTranscript: 文字起こし結果

    Raises:
        openai.APIError: API 呼び出しが失敗した場合
    """
    client = _get_client()
    prompt = _build_whisper_prompt(speaker, committee)

    duration = _get_wav_duration(wav_path)

    logger.info(
        "Transcribing segment %d: %s (%s, %.1fs)",
        segment_index,
        speaker.name,
        wav_path.name,
        duration,
    )

    if duration <= WHISPER_MIN_WINDOWED_DURATION:
        # 短いセグメントはオーバーラップ不要、ファイルをそのまま送る。
        # duration が 0 になる test fixture (empty .touch() ファイル) もここに落ちる。
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        text, raw_segments = _call_whisper(client, wav_bytes, prompt)
        whisper_segments = [
            WhisperSegment(
                id=int(s.get("id", i)),
                seek=int(s.get("seek", 0)),
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=s.get("text", ""),
                tokens=list(s.get("tokens", [])),
                temperature=float(s.get("temperature", 0.0)),
                avg_logprob=float(s.get("avg_logprob", 0.0)),
                compression_ratio=float(s.get("compression_ratio", 0.0)),
                no_speech_prob=float(s.get("no_speech_prob", 0.0)),
                words=[
                    WhisperWord(
                        word=w.get("word", ""),
                        start=float(w.get("start", 0.0)),
                        end=float(w.get("end", 0.0)),
                    )
                    for w in s.get("words", [])
                ],
            )
            for i, s in enumerate(raw_segments)
        ]
    else:
        wav = _read_wav(wav_path)
        text, whisper_segments = _transcribe_windowed(wav, prompt, client=client)

    return SegmentTranscript(
        segment_index=segment_index,
        speaker_name=speaker.name,
        start_seconds=speaker.start_seconds,
        text=text,
        whisper_segments=whisper_segments,
    )


def transcribe_all_segments(
    segment_paths: list[Path],
    speakers: list[SpeakerInfo],
    session_id: str,
    committee: str = "",
    max_workers: int = 16,
) -> RawTranscript:
    """全セグメントを並列で文字起こしして RawTranscript を返す。

    Args:
        segment_paths: セグメント WAV ファイルのリスト
        speakers: 発言者リスト（segment_paths と同順）
        session_id: セッションID
        committee: 委員会名（Whisperプロンプトの動的サフィックスに使用）
        max_workers: 並列数（DeepInfra のレート制限に合わせて調整）

    Returns:
        RawTranscript: 全セグメントの文字起こし結果（segment_index 順にソート済み）
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _transcribe(args: tuple[int, Path, SpeakerInfo]) -> SegmentTranscript:
        i, wav_path, speaker = args
        return transcribe_segment(wav_path, i, speaker, committee)

    tasks = list(enumerate(zip(segment_paths, speakers)))
    work = [(i, wav, spk) for i, (wav, spk) in tasks]

    results: list[SegmentTranscript] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_transcribe, item): item[0] for item in work}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda s: s.segment_index)
    return RawTranscript(session_id=session_id, segments=results)
