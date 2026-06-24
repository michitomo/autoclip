// 静的カタログ (GitHub Pages 用) の読み込み。backend src/build_site.py が出力した
// data/catalog.json と data/<id>/{project,edl}.json を読む。

import type { Edl } from './edl'
import type { Project } from './fromProject'

export type CatalogTopic = {
  index: number
  label: string
  question_speaker: string
  answer_speakers: string[]
}
export type CatalogClip = {
  id: string
  session_id: string
  member: string
  affiliation: string
  committee: string
  date: string
  title: string
  n_topics: number
  topics: CatalogTopic[]
}
export type Catalog = { clips: CatalogClip[]; count: number }

// 母集合 (corpus): backend src/build_corpus.py が出力。日付→委員会→議員 の選択肢。
// 生成済みか否かは無関係 (生成済み判定は catalog の id と member.clip_id を突合)。
export type CorpusMember = {
  name: string
  affiliation: string
  role: string
  duration_minutes: number
  clip_id: string
}
export type CorpusSession = {
  session_id: string
  committee: string
  duration: string | null
  members: CorpusMember[]
}
export type Corpus = {
  generated_at: string
  days: Record<string, CorpusSession[]> // "YYYY-MM-DD" → 委員会[]
}

// data の置き場。Vite の base path 配下に置かれるので相対 'data/' で参照する
// (import.meta.env.BASE_URL を前置)。
function dataUrl(path: string): string {
  const base = import.meta.env.BASE_URL || '/'
  return `${base.replace(/\/$/, '')}/data/${path}`
}

export async function loadCatalog(): Promise<Catalog> {
  const r = await fetch(dataUrl('catalog.json'))
  if (!r.ok) throw new Error(`catalog.json 取得失敗: ${r.status}`)
  return r.json() as Promise<Catalog>
}

export async function loadCorpus(): Promise<Corpus> {
  const r = await fetch(dataUrl('corpus.json'))
  if (!r.ok) throw new Error(`corpus.json 取得失敗: ${r.status}`)
  return r.json() as Promise<Corpus>
}

export async function loadClip(id: string): Promise<{ project: Project & { hls_url: string }; edl: Edl }> {
  const [project, edl] = await Promise.all([
    fetch(dataUrl(`${id}/project.json`)).then((r) => {
      if (!r.ok) throw new Error(`project.json 取得失敗: ${r.status}`)
      return r.json() as Promise<Project & { hls_url: string }>
    }),
    fetch(dataUrl(`${id}/edl.json`)).then((r) => {
      if (!r.ok) throw new Error(`edl.json 取得失敗: ${r.status}`)
      return r.json() as Promise<Edl>
    }),
  ])
  return { project, edl }
}
