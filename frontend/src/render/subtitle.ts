// 字幕の Canvas 描画 (backend src/video/subtitles.py の移植)
//
// ASS を生成する代わりに、post-cut 時刻 t のフレームに対して「今出すべき字幕」を
// Canvas 2D で直接描く。ASS の各要素との対応:
//  - 本文字幕 (カラオケ): build_ass_karaoke。語ごとに post-cut 時刻があり、発話中の語を
//    話者色、それ以外を白で描く。
//  - 話者色: _role_colour_at。role_spans (post-cut) から t の話者を引く。
//  - 常時バナー: _banner_event。title_seconds 以降ずっと上部に日付/委員会/議員。
//  - タイトルパネル: _title_event。冒頭 title_seconds だけ中央に白パネル。

import { loadDefaultJapaneseParser } from 'budoux'
import type { KeptWord } from './edl'

// 文節分割器 (backend の LLM 文節境界に相当する軽量版)。点灯/改行の単位に使う。
const budouxParser = loadDefaultJapaneseParser()

// Canvas / OffscreenCanvas 両方の 2D コンテキストを受ける
export type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D

// 色 (subtitles.py の定数。ASS は &H00BBGGRR なので RGB に直したもの)
const COLOR_Q = '#5FE0B7' // 質疑者 (ミント)
const COLOR_A = '#FF8C0F' // 答弁者 (オレンジ)
const OUTLINE = 'rgba(0,0,0,0.85)'

const ANSWER_ROLES = new Set(['答弁者', '政府参考人', '参考人'])

export type RoleSpan = { start: number; end: number; role: string } // post-cut 秒
export type TitleInfo = {
  title: string // 「｜」で改行
  header: string[] // [日付, 委員会, 議員+トピック]
}

export type SubtitleData = {
  keptWords: KeptWord[] // post-cut 時刻つき (new_start/new_end)
  roleSpans: RoleSpan[]
  title: TitleInfo
  titleSeconds: number // タイトルパネルを出す秒数 (既定 2.0)
  // ブラウザ内編集: ページ先頭時刻(キー) → 差し替えテキスト。指定があればその文字列で
  // 描画する (元の語の時刻範囲に文節を均等割りして点灯)。
  textOverrides?: Record<string, string>
  _pages?: Page[] // 内部キャッシュ (初回 draw 時に paginate)
}

/** ページ先頭時刻を編集オーバーライドのキーにする (0.01秒で丸めて安定化)。 */
export function pageKey(startSec: number): string {
  return startSec.toFixed(2)
}

function roleColorAt(t: number, spans: RoleSpan[]): string {
  for (const s of spans) {
    if (s.start <= t && t < s.end) return ANSWER_ROLES.has(s.role) ? COLOR_A : COLOR_Q
  }
  return COLOR_Q
}

// カラオケ字幕の核心 (backend build_ass_karaoke と同じ考え方):
//  - kept_words (1文字単位) を budoux で「文節」にまとめ、文節を点灯/改行の単位にする。
//  - 文節列を「ページ」(= 一度に画面に出すまとまり) に事前分割しておく。
//  - 時刻 t では、t を含むページを **丸ごと** 表示する (1文字ずつ増えない・安定表示)。
//  - そのページ内で「発話済みの文節」を話者色、未発話を白にする (= 進行で色だけ変わる)。
//  - 改行は **文節の境界** でだけ行う (文節の途中で切らない)。
const MAX_CHARS_PER_LINE = 13 // 1行の目安字数
const MAX_LINES = 2 // 1ページの最大行数
const SENTENCE_END = '。．！？!?'
const CLAUSE_END = '、，'

// 点灯/描画の単位。budoux 文節 1 つ (= 複数文字をまとめたもの)。
type Word = { text: string; start: number; end: number }
type Page = { words: Word[]; start: number; end: number }

/** kept_words (1文字×時刻) を budoux 文節にまとめる。各文節の start/end は
 * 構成文字の最初の new_start / 最後の new_end。 */
function groupIntoPhrases(kw: KeptWord[]): Word[] {
  // 連結テキストと、各文字 → 元 kept_word index の対応表を作る。
  const chars: { ch: string; start: number; end: number }[] = []
  for (const w of kw) {
    const t = w.word
    if (!t) continue
    // kept_word が複数文字でも 1 文字ずつに割る (時刻はその語のものを共有)。
    for (const ch of t) chars.push({ ch, start: w.new_start, end: w.new_end })
  }
  if (chars.length === 0) return []

  const fullText = chars.map((c) => c.ch).join('')
  const segments = budouxParser.parse(fullText) // 文節文字列の配列 (連結すると fullText)

  const phrases: Word[] = []
  let ci = 0
  for (const seg of segments) {
    const len = seg.length
    if (len === 0) continue
    const slice = chars.slice(ci, ci + len)
    ci += len
    phrases.push({
      text: seg,
      start: slice[0].start,
      end: slice[slice.length - 1].end,
    })
  }
  return phrases
}

/** kept_words をページへ事前分割する (文末/読点優先・行数上限でページ確定)。
 * 単位は budoux 文節。文節境界でしかページ・改行が起きない。
 *
 * 重要: ページ容量は wrapPage と同じ行折りで判定する。文節を足して MAX_LINES 行に
 * 収まらなくなったら、その文節は **次ページへ回す** (詰めすぎて wrapPage が末尾を
 * 捨てるのを防ぐ。括弧内などページ後半の文節が消える不具合の原因だった)。 */
function paginate(kw: KeptWord[]): Page[] {
  const phrases = groupIntoPhrases(kw)
  const pages: Page[] = []
  let cur: Word[] = []

  const flush = () => {
    if (cur.length === 0) return
    pages.push({
      words: cur,
      start: cur[0].start,
      end: cur[cur.length - 1].end,
    })
    cur = []
  }

  const curChars = () => cur.reduce((n, w) => n + w.text.length, 0)

  for (const ph of phrases) {
    // この文節を足すと MAX_LINES 行に収まらないなら、先に現ページを確定して次へ。
    if (cur.length > 0 && lineCount([...cur, ph]) > MAX_LINES) {
      flush()
    }
    cur.push(ph)
    const lastCh = ph.text[ph.text.length - 1]
    // 文末は必ず確定。読点は 1 行ぶん(=MAX_CHARS_PER_LINE)溜まってから確定する
    // (「、」だけ等の極短ページが乱立して一瞬しか出ないのを防ぐ)。
    if (SENTENCE_END.includes(lastCh)) {
      flush()
    } else if (CLAUSE_END.includes(lastCh) && curChars() >= MAX_CHARS_PER_LINE) {
      flush()
    }
  }
  flush()
  return pages
}

/** ページを作る (paginate + テキストオーバーライド適用)。描画とUIの両方で使う。 */
export function buildPages(data: SubtitleData): Page[] {
  const pages = paginate(data.keptWords)
  const ov = data.textOverrides
  if (!ov) return pages
  return pages.map((p) => {
    const repl = ov[pageKey(p.start)]
    if (repl == null || repl === pageText(p)) return p
    return applyOverride(p, repl)
  })
}

/** ページの表示テキスト (語を連結)。 */
export function pageText(p: Page): string {
  return p.words.map((w) => w.text).join('')
}

/** 差し替えテキストを budoux 分割し、元ページの [start,end] に均等割りした新ページを作る。
 * 語ごとの正確な時刻は失われるが、カラオケの進行は概ね合う (編集は誤字直しが主目的)。 */
function applyOverride(p: Page, text: string): Page {
  const segs = budouxParser.parse(text).filter((s) => s.length > 0)
  if (segs.length === 0) return { words: [{ text, start: p.start, end: p.end }], start: p.start, end: p.end }
  const total = segs.reduce((n, s) => n + s.length, 0)
  let acc = 0
  const span = p.end - p.start
  const words: Word[] = segs.map((s) => {
    const start = p.start + (span * acc) / total
    acc += s.length
    const end = p.start + (span * acc) / total
    return { text: s, start, end }
  })
  return { words, start: p.start, end: p.end }
}

/** 編集UI用: 現在の (オーバーライド適用後の) ページ一覧を返す。 */
export function pageList(data: SubtitleData): { key: string; start: number; end: number; text: string }[] {
  return buildPages(data).map((p) => ({
    key: pageKey(p.start), start: p.start, end: p.end, text: pageText(p),
  }))
}

/** 文節列を MAX_CHARS_PER_LINE で語境界折りしたときの行数 (容量判定用)。 */
function lineCount(words: Word[]): number {
  let lines = 1
  let lineChars = 0
  for (const w of words) {
    if (lineChars > 0 && lineChars + w.text.length > MAX_CHARS_PER_LINE) {
      lines++
      lineChars = 0
    }
    lineChars += w.text.length
  }
  return lines
}

/** ページを語境界で行に折る (語の途中で切らない)。返り値は行ごとの語配列。
 * paginate がページ容量を MAX_LINES に収めているので、ここで語を捨てることはない。 */
function wrapPage(page: Page): Word[][] {
  const lines: Word[][] = []
  let line: Word[] = []
  let lineChars = 0
  for (const w of page.words) {
    if (lineChars > 0 && lineChars + w.text.length > MAX_CHARS_PER_LINE) {
      lines.push(line)
      line = []
      lineChars = 0
    }
    line.push(w)
    lineChars += w.text.length
  }
  if (line.length > 0) lines.push(line)
  return lines
}

function roundRect(ctx: Ctx2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

/**
 * post-cut 時刻 t のフレームに字幕一式を描く。
 * ctx は既に背景(動画フレーム)が描かれた状態で渡される。canvas は W×H (9:16)。
 */
export function drawSubtitleFrame(
  ctx: Ctx2D,
  W: number,
  H: number,
  t: number,
  data: SubtitleData,
) {
  const scale = W / 1080 // backend PlayResX=1080 基準でサイズを合わせる

  // --- タイトルパネル (冒頭 titleSeconds だけ) ---
  if (t < data.titleSeconds) {
    drawTitlePanel(ctx, W, H, scale, data.title)
  } else {
    // --- 常時バナー (タイトル後ずっと) ---
    drawBanner(ctx, W, scale, data.title.header)
  }

  // --- 本文字幕 (カラオケ) ---
  // ページ事前分割 (初回のみ)。t を含むページを丸ごと表示し、語ごとに色を変える。
  if (!data._pages) data._pages = buildPages(data)
  const page = activePage(data._pages, t)
  if (!page) return
  const litColor = roleColorAt(t, data.roleSpans)
  const lines = wrapPage(page) // 語境界で折った行 (語の途中で切らない)
  if (lines.length === 0) return

  const fontPx = Math.round(58 * scale)
  ctx.save()
  ctx.font = `800 ${fontPx}px system-ui, "Hiragino Sans", "Noto Sans JP", sans-serif`
  ctx.textBaseline = 'alphabetic'
  ctx.lineJoin = 'round'
  ctx.lineWidth = Math.round(10 * scale)
  const lineH = Math.round(fontPx * 1.34)
  const totalH = lines.length * lineH
  // 下から ~15% の位置に最終行が来るよう配置 (1〜2行で安定)
  let y = H - Math.round(H * 0.15) - totalH + lineH
  for (const line of lines) {
    const text = line.map((w) => w.text).join('')
    const totalW = ctx.measureText(text).width
    let x = (W - totalW) / 2
    // 縁取りは行全体に一括 (語ごとだと継ぎ目が出る)
    ctx.strokeStyle = OUTLINE
    ctx.strokeText(text, x, y)
    // 語単位で色 (発話済み=話者色 / 未発話=白)。1文字ずつではなく語のまとまりで点灯。
    for (const w of line) {
      ctx.fillStyle = w.start <= t ? litColor : '#ffffff'
      ctx.fillText(w.text, x, y)
      x += ctx.measureText(w.text).width
    }
    y += lineH
  }
  ctx.restore()
}

/** ページ列から、時刻 t を含む (または直近の) ページを返す。 */
function activePage(pages: Page[], t: number): Page | null {
  let active: Page | null = null
  for (const p of pages) {
    if (p.start <= t && t < p.end) return p
    if (p.start > t) break
    active = p // t がページ間の隙間なら直近の (= 最後に始まった) ページを残す
  }
  // 直近ページの終端から大きく離れていたら消す (区間の終わりで字幕を残しすぎない)
  if (active && t > active.end + 0.6) return null
  return active
}

function drawBanner(ctx: Ctx2D, W: number, scale: number, header: string[]) {
  // header = [日付, 委員会, 議員質疑]。1行に詰めず 2行 (日付 / 委員会・議員) にする。
  const line1 = header[0] ?? ''
  const line2 = [header[1], header[2]].filter(Boolean).join('  ・  ')
  const lines = [line1, line2].filter(Boolean)
  ctx.save()
  const fontPx = Math.round(30 * scale)
  ctx.font = `600 ${fontPx}px system-ui, "Hiragino Sans", "Noto Sans JP", sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  const padX = Math.round(28 * scale)
  const padY = Math.round(14 * scale)
  const lineH = Math.round(fontPx * 1.32)
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + padX * 2
  const h = lines.length * lineH + padY * 2
  const top = Math.round(44 * scale)
  ctx.fillStyle = 'rgba(20,25,33,0.72)'
  roundRect(ctx, W / 2 - w / 2, top, w, h, Math.round(12 * scale)); ctx.fill()
  ctx.fillStyle = '#ffffff'
  let cy = top + padY + lineH / 2
  for (const l of lines) { ctx.fillText(l, W / 2, cy); cy += lineH }
  ctx.restore()
}

function drawTitlePanel(ctx: Ctx2D, W: number, H: number, scale: number, title: TitleInfo) {
  ctx.save()
  ctx.textAlign = 'center'
  const headFont = Math.round(40 * scale)
  const bigFont = Math.round(70 * scale)
  const headLines = title.header
  const bigLines = title.title.split('｜')
  // パネル寸法
  const padY = Math.round(48 * scale)
  const lineGap = Math.round(14 * scale)
  const headH = headLines.length * (headFont + lineGap)
  const bigH = bigLines.length * (bigFont + lineGap)
  const panelH = padY * 2 + headH + Math.round(20 * scale) + bigH
  const panelW = Math.round(W * 0.86)
  const px = (W - panelW) / 2, py = (H - panelH) / 2
  // 白パネル + 濃緑の額縁
  ctx.fillStyle = '#ffffff'
  roundRect(ctx, px, py, panelW, panelH, Math.round(16 * scale)); ctx.fill()
  ctx.lineWidth = Math.round(18 * scale)
  ctx.strokeStyle = '#2BAE82'
  roundRect(ctx, px + ctx.lineWidth / 2, py + ctx.lineWidth / 2, panelW - ctx.lineWidth, panelH - ctx.lineWidth, Math.round(12 * scale))
  ctx.stroke()
  // 見出し (黒・小)
  ctx.fillStyle = '#1a1d21'
  ctx.textBaseline = 'top'
  let y = py + padY
  ctx.font = `600 ${headFont}px system-ui, "Hiragino Sans", "Noto Sans JP", sans-serif`
  for (const l of headLines) { ctx.fillText(l, W / 2, y); y += headFont + lineGap }
  y += Math.round(20 * scale)
  // 要旨 (黒・大・太)
  ctx.font = `800 ${bigFont}px system-ui, "Hiragino Sans", "Noto Sans JP", sans-serif`
  for (const l of bigLines) { ctx.fillText(l, W / 2, y); y += bigFont + lineGap }
  ctx.restore()
}
