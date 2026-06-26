// autoclip API クライアント (型付き)

export type Session = {
  session_id: string
  committee: string
  date: string
  duration: string
  session_kind: string
  n_speakers: number
  n_members: number
}

// 日付ピッカー用の軽量プレビュー (カレンダーGET 1回。詳細取得なし)。
export type CalendarSession = {
  session_id: string
  committee: string
  duration: string | null
}

export type Member = {
  name: string
  affiliation: string
  role: string
  start_seconds: number
  duration_minutes: number
}

export type JobState = 'queued' | 'running' | 'done' | 'error'

export type Job = {
  id: string
  kind: string
  state: JobState
  step: string
  result: ClipResult | null
  error: string | null
  meta: Record<string, unknown>
}

export type ClipResult = {
  clip_path: string | null // 全体レンダ廃止のため生成時は null (プレビューはトピック単位)
  session_id: string
  member: string
  affiliation: string
  duration: number
  n_ranges: number
  n_topics?: number
}

export type ClipRequest = {
  session_id: string
  member: string
  aspect?: string
  subtitle_style?: string
  title?: string | null
  preview_seconds?: number | null
}

async function jget<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}

async function jpost<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}

export type EditRange = { start: number; end: number; enabled: boolean }
export type EditCaption = { start: number; end: number; text: string; edited: boolean }

// Q&A 編集ツリー (トピック>発言者>発言内容)。enabled はリーフ(文)のみが真実、
// 親(ターン/トピック)の状態は子から導出する。時間は member-WAV (元音声) 秒。
export type Importance = 'high' | 'mid' | 'low'
export type QASentence = {
  text: string
  start: number
  end: number
  summary: string
  importance: Importance
  enabled: boolean
}
export type QATurn = { speaker: string; role: string; sentences: QASentence[] }
export type QATopic = {
  index: number
  label: string
  question_speaker: string
  answer_speakers: string[]
  turns: QATurn[]
}
export type QATree = { topics: QATopic[] }

// トピック別/全体 書き出しの結果 (topic_index が null なら全体クリップ)
export type TopicClip = {
  clip_path: string
  topic_index: number | null
  topic_label: string
  duration: number
}
export type ExportResult = { clips: TopicClip[]; session_id: string; member: string }

// トピックのオンデマンドプレビュー結果。disabled_spans はオフ文の
// プレビュー(ローカル)時間区間 [start,end] (再生時スキップ用)。
export type TopicPreview = {
  clip_path: string
  topic_index: number
  topic_label: string
  duration: number
  disabled_spans: [number, number][]
}

export type ClipProject = {
  session_id: string
  member: string
  source_video: string
  member_start: number
  aspect: string
  subtitle_style: string
  title: string
  title_header?: string[]
  ranges: EditRange[]
  captions: EditCaption[]
  qa_tree?: QATree | null // null/未定義 = 旧プロジェクト → フラット UI
}

export const api = {
  // 軽量カレンダー: 指定日の委員会名一覧 (空なら審議なし)。日付ピッカーのプレビュー用。
  getCalendar: (date: string) =>
    jget<{ date: string; sessions: CalendarSession[] }>(
      `/api/calendar?date=${encodeURIComponent(date)}`,
    ),
  listSessions: (date: string) =>
    jget<{ date: string; sessions: Session[] }>(
      `/api/sessions?date=${encodeURIComponent(date)}`,
    ),
  listMembers: (sessionId: string) =>
    jget<{ session_id: string; committee: string; date: string; members: Member[] }>(
      `/api/sessions/${sessionId}/members`,
    ),
  createClip: (req: ClipRequest) =>
    jpost<{ job_id: string }>('/api/clips', req),
  rerender: (sessionId: string, req: ClipRequest) =>
    jpost<{ job_id: string }>(`/api/clips/${sessionId}/render`, req),
  getJob: (jobId: string) => jget<Job>(`/api/jobs/${jobId}`),
  getProject: (sessionId: string, member: string) =>
    jget<ClipProject>(
      `/api/clips/${sessionId}/${encodeURIComponent(member)}/project`,
    ),
  editClip: (sessionId: string, member: string, project: ClipProject) =>
    jpost<{ job_id: string }>(
      `/api/clips/${sessionId}/${encodeURIComponent(member)}/edit`,
      project,
    ),
  exportClips: (
    sessionId: string,
    member: string,
    mode: 'topics' | 'full' | 'both',
    project: ClipProject,
    topic?: number, // mode=topics のとき、その index のトピックだけ書き出す
  ) =>
    jpost<{ job_id: string }>(
      `/api/clips/${sessionId}/${encodeURIComponent(member)}/export?mode=${mode}` +
        (topic != null ? `&topic=${topic}` : ''),
      project,
    ),
  previewTopic: (
    sessionId: string,
    member: string,
    index: number,
    project: ClipProject,
  ) =>
    jpost<{ job_id: string }>(
      `/api/clips/${sessionId}/${encodeURIComponent(member)}/preview-topic?index=${index}`,
      project,
    ),
  clipFileUrl: (clipPath: string) => {
    // clip_path = "<session_id>/<file>"
    const [sid, ...rest] = clipPath.split('/')
    return `/api/clips/file/${sid}/${rest.join('/')}`
  },
}

// ジョブのステップ表示名
// 生成工程の表示名。内部用語 (JetCut 等) は使わず、何が起きているかを平易に。
export const STEP_LABELS: Record<string, string> = {
  queued: '待機中',
  starting: '開始',
  scraping: '映像を取得',
  downloading: '映像をダウンロード',
  transcribing: '文字起こし',
  correcting: '内容を整える（議員名・句読点）',
  titling: 'タイトルを作る',
  jetcut: '不要な間・言い淀み・言い間違いを除去',
  rendering: '動画を準備',
  done: '完了',
  error: 'エラー',
}

export const STEP_ORDER = [
  'scraping',
  'downloading',
  'transcribing',
  'correcting',
  'titling',
  'jetcut',
  'rendering',
  'done',
]
