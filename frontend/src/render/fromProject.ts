// backend の project.json + _edl.json から、ブラウザレンダ用の入力を組み立てる。
// role_spans は qa_tree の発言ターン(member-WAV)を post-cut 時刻へ写像して作る
// (_role_spans_post_cut 相当)。

import type { Edl, KeepRange } from './edl'
import type { RoleSpan, SubtitleData, TitleInfo } from './subtitle'

type QASentence = { start: number; end: number; enabled: boolean; text: string }
type QATurn = { speaker: string; role: string; sentences: QASentence[] }
type QATopic = { index: number; turns: QATurn[] }
type QATree = { topics: QATopic[] }

export type Project = {
  source_video: string
  member: string
  member_start: number
  aspect: string
  title: string
  title_header: string[]
  qa_tree: QATree | null
}

// kept_words 連結 (正規化) のインデックスを1度だけ作るためのキャッシュ型。
type KwIndex = { hay: string; charToWi: number[]; kw: { word: string; old_start: number; old_end: number }[] }

function buildKwIndex(kw: KwIndex['kw']): KwIndex {
  const norm: string[] = []
  const charToWi: number[] = []
  for (let wi = 0; wi < kw.length; wi++) {
    for (const ch of normForMatch(kw[wi].word)) { norm.push(ch); charToWi.push(wi) }
  }
  return { hay: norm.join(''), charToWi, kw }
}

/** テキストを kept_words 連結内で照合し、対応する [wiStart, wiEnd] を返す (失敗で null)。 */
function matchSpan(idx: KwIndex, text: string): { wiStart: number; wiEnd: number } | null {
  const t = normForMatch(text)
  if (t.length < 8) return null
  const head = t.slice(0, Math.min(20, t.length))
  const tail = t.slice(-Math.min(20, t.length))
  const iHead = idx.hay.indexOf(head)
  const iTail = idx.hay.lastIndexOf(tail)
  if (iHead < 0 || iTail < 0 || iTail < iHead) return null
  return {
    wiStart: idx.charToWi[iHead],
    wiEnd: idx.charToWi[Math.min(iTail + tail.length - 1, idx.charToWi.length - 1)],
  }
}

/** roleSpans を kept_words の実時刻 (new_*) で作る。
 * qa_tree の turn 時刻はズレるため、各 turn のテキストを kept_words に照合して
 * その範囲の new_start/new_end を role span にする (字幕と完全に同じ時間軸 = 色が合う)。 */
function buildRoleSpans(qa: QATree | null, edl: Edl): RoleSpan[] {
  if (!qa) return []
  const kw = edl.kept_words as KwIndex['kw']
  const idx = buildKwIndex(kw)
  const spans: RoleSpan[] = []
  for (const topic of qa.topics) {
    for (const turn of topic.turns) {
      const sents = turn.sentences.filter((s) => s.enabled && s.end > s.start)
      if (sents.length === 0) continue
      const m = matchSpan(idx, sents.map((s) => s.text).join(''))
      if (!m) continue
      const start = (edl.kept_words[m.wiStart] as { new_start: number }).new_start
      const end = (edl.kept_words[m.wiEnd] as { new_end: number }).new_end
      if (end > start) spans.push({ start, end, role: turn.role })
    }
  }
  spans.sort((a, b) => a.start - b.start)
  return spans
}

export function buildSubtitleData(
  project: Project,
  edl: Edl,
  titleSeconds = 2.0,
): SubtitleData {
  const title: TitleInfo = {
    title: project.title || '',
    header: project.title_header || [],
  }
  return {
    keptWords: edl.kept_words,
    roleSpans: buildRoleSpans(project.qa_tree, edl),
    title,
    titleSeconds,
  }
}

/** 照合用にテキストを正規化 (記号・空白・話者ラベルのゆれを除去)。 */
function normForMatch(s: string): string {
  return s.replace(/[「」『』（）()【】、。，．・\s　]/g, '')
}

/**
 * トピックの member-WAV 窓 [lo,hi] を求める。
 *
 * 注意: qa_tree.sentences の start/end は backend の文字数ベース時刻付与
 * (_consume_span) によるもので、**校正でテキスト長が変わると kept_words の実時刻から
 * 累積ズレする** (後半トピックほど数十秒ズレ、映像が別話者になる不具合の原因)。
 * そこで qa_tree の時刻は使わず、トピックのテキストを kept_words の連結内で照合して、
 * **kept_words の old 時刻**で窓を引き当てる (kept_words が字幕=映像の真の時刻)。
 * 照合に失敗したら qa_tree の時刻にフォールバック。
 */
function topicWindow(
  topic: { turns: { sentences: { start: number; end: number; text: string }[] }[] },
  kw: KwIndex['kw'],
): { lo: number; hi: number } {
  const sents = topic.turns.flatMap((t) => t.sentences).filter((s) => s.end > s.start)
  const qaLo = Math.min(...sents.map((s) => s.start))
  const qaHi = Math.max(...sents.map((s) => s.end))
  const m = matchSpan(buildKwIndex(kw), sents.map((s) => s.text).join(''))
  if (!m) return { lo: qaLo, hi: qaHi }
  return { lo: kw[m.wiStart].old_start, hi: kw[m.wiEnd].old_end }
}

/**
 * EDL を1トピックぶんに絞り込む。トピックの member-WAV 窓 [lo,hi] と**重なる**
 * keep_range を採用し、post-cut(new_*) を 0 起点に振り直す。
 * 窓は kept_words のテキスト照合で引き当てる (qa_tree の時刻はズレるため。topicWindow 参照)。
 */
export function topicEdl(edl: Edl, project: Project, topicIndex: number): Edl {
  const qa = project.qa_tree
  const topic = qa?.topics.find((t) => t.index === topicIndex)
  if (!topic) return edl
  const sents = topic.turns.flatMap((t) => t.sentences).filter((s) => s.end > s.start)
  if (sents.length === 0) return edl
  const { lo, hi } = topicWindow(topic, edl.kept_words)

  // [lo,hi] と重なる keep_range を採用し、交差部分だけに切り詰める (member-WAV 時間)。
  const clipped: KeepRange[] = []
  for (const r of edl.keep_ranges) {
    const s = Math.max(r.start, lo)
    const e = Math.min(r.end, hi)
    if (e > s + 1e-6) clipped.push({ start: s, end: e })
  }
  if (clipped.length === 0) return edl

  // 採用 range ごとの post-cut オフセット (0 起点に前詰め)
  let acc = 0
  const offsetFor = new Map<KeepRange, number>()
  for (const r of clipped) { offsetFor.set(r, acc); acc += r.end - r.start }

  // 語の中点が採用 range のいずれかに入るものだけ残し、post-cut を振り直す。
  const kept_words = edl.kept_words
    .map((w) => {
      const mid = (w.old_start + w.old_end) / 2
      const r = clipped.find((rr) => rr.start - 1e-6 <= mid && mid <= rr.end + 1e-6)
      if (!r) return null
      const base = offsetFor.get(r)!
      return {
        ...w,
        new_start: base + (Math.max(w.old_start, r.start) - r.start),
        new_end: base + (Math.min(w.old_end, r.end) - r.start),
      }
    })
    .filter((w): w is NonNullable<typeof w> => w !== null)

  return { keep_ranges: clipped, kept_words, params: edl.params }
}

/** aspect 文字列 ("9:16" 等) から出力解像度を返す。 */
export function aspectToSize(aspect: string): { width: number; height: number } {
  switch (aspect) {
    case '1:1': return { width: 1080, height: 1080 }
    case '16:9': return { width: 1280, height: 720 }
    case '9:16':
    default: return { width: 720, height: 1280 }
  }
}
