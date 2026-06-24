import { useEffect, useState } from 'react'
import { api, type ClipProject, type QATopic } from './api'

function fmt(t: number): string {
  const m = Math.floor(t / 60)
  const s = Math.round(t % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

/** トピックの長さ (有効文の合計秒)。 */
function topicDuration(topic: QATopic): number {
  let sum = 0
  for (const turn of topic.turns)
    for (const s of turn.sentences)
      if (s.enabled && s.end > s.start) sum += s.end - s.start
  return sum
}

/** 答弁者名 (重複除去・最大3名)。 */
function answerers(topic: QATopic): string {
  const seen: string[] = []
  for (const turn of topic.turns)
    if (turn.role !== '質疑者' && !seen.includes(turn.speaker)) seen.push(turn.speaker)
  return seen.slice(0, 3).join('、')
}

/**
 * 生成完了後のトピック選択画面。質疑トピックを一覧表示し、選んだトピックの
 * 編集画面 (onSelect) へ進む。要旨・長さ・話者を出す。
 */
export function TopicPicker({
  sessionId,
  member,
  onSelect,
}: {
  sessionId: string
  member: string
  onSelect: (topicIndex: number) => void
}) {
  const [project, setProject] = useState<ClipProject | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api
      .getProject(sessionId, member)
      .then((p) => alive && setProject(p))
      .catch((e) => alive && setError(`プロジェクト取得失敗: ${e}`))
    return () => {
      alive = false
    }
  }, [sessionId, member])

  if (error) return <div className="error">⚠ {error}</div>
  if (!project) return <p className="muted loading-dots">読み込み中</p>
  const topics = project.qa_tree?.topics ?? []
  if (topics.length === 0)
    return (
      <div className="empty">
        <span className="empty-ic">🗂️</span>
        <span className="empty-text">トピックが検出されませんでした。</span>
      </div>
    )

  return (
    <div className="topic-picker">
      <p className="qa-hint muted">
        クリップにする質疑トピックを選んでください。選ぶと編集・書き出し画面に進みます。
      </p>
      <div className="topic-list">
        {topics.map((t) => (
          <button
            key={t.index}
            className="topic-card"
            onClick={() => onSelect(t.index)}
          >
            <div className="topic-card-head">
              <span className="topic-card-num">{t.index + 1}</span>
              <span className="topic-card-label">{t.label || `トピック ${t.index + 1}`}</span>
              <span className="topic-card-dur">{fmt(topicDuration(t))}</span>
            </div>
            <div className="topic-card-speakers">
              <span className="role-q">Q: {t.question_speaker || '—'}</span>
              {answerers(t) && <span className="role-a">A: {answerers(t)}</span>}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
