import { useEffect, useRef } from 'react'
import type { Importance, QASentence, QATree } from './api'

function fmt(t: number): string {
  const m = Math.floor(t / 60)
  const s = (t % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

type TriState = 'all' | 'none' | 'some'

function triOf(sentences: QASentence[]): TriState {
  const on = sentences.filter((s) => s.enabled).length
  if (on === 0) return 'none'
  if (on === sentences.length) return 'all'
  return 'some'
}

/** indeterminate を ref 経由で設定するチェックボックス (React は管理しないため)。 */
function TriCheckbox({
  state,
  onToggle,
  title,
}: {
  state: TriState
  onToggle: () => void
  title?: string
}) {
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = state === 'some'
  }, [state])
  return (
    <input
      ref={ref}
      type="checkbox"
      checked={state === 'all'}
      onChange={onToggle}
      title={title}
      onClick={(e) => e.stopPropagation()}
    />
  )
}

const IMP_LABEL: Record<Importance, string> = { high: '高', mid: '中', low: '低' }

/**
 * Q&A 階層エディタ: トピック > 発言者 > 発言内容(文)。
 * リーフ(文)の enabled だけが真実。親はチェック状態を子から導出 (tri-state)。
 * 時刻は member-WAV (元音声) 秒。チェックを外した文はクリップから除外される
 * (バックエンドが apply 時に ranges.enabled へ反映)。
 */
export function QATreeEditor({
  tree,
  onChange,
  onSeek,
  onPreviewTopic,
  previewIdx,
  onlyTopicIndex,
}: {
  tree: QATree
  onChange: (tree: QATree) => void
  onSeek?: (memberStart: number) => void
  onPreviewTopic?: (topicIndex: number) => void
  previewIdx?: number | null
  // 指定時はその topic.index のトピックだけ表示 (トピック編集画面)。
  // mutate は全 tree を対象に更新するので index 整合は崩れない。
  onlyTopicIndex?: number
}) {
  // すべての文に変換を適用した新ツリーを作って通知する (immutable 更新)。
  function mutate(fn: (s: QASentence, ti: number, ki: number, si: number) => QASentence) {
    onChange({
      topics: tree.topics.map((topic, ti) => ({
        ...topic,
        turns: topic.turns.map((turn, ki) => ({
          ...turn,
          sentences: turn.sentences.map((s, si) => fn(s, ti, ki, si)),
        })),
      })),
    })
  }

  function setSentence(ti: number, ki: number, si: number, enabled: boolean) {
    mutate((s, t, k, i) => (t === ti && k === ki && i === si ? { ...s, enabled } : s))
  }
  function setTurn(ti: number, ki: number, enabled: boolean) {
    mutate((s, t, k) => (t === ti && k === ki ? { ...s, enabled } : s))
  }
  function setTopic(ti: number, enabled: boolean) {
    mutate((s, t) => (t === ti ? { ...s, enabled } : s))
  }
  function bulkByImportance(imp: Importance, enabled: boolean) {
    mutate((s) => (s.importance === imp ? { ...s, enabled } : s))
  }
  function setAll(enabled: boolean) {
    mutate((s) => ({ ...s, enabled }))
  }

  // 表示・集計の対象トピック (onlyTopicIndex 指定時はそれ1件)。
  const shownTopics =
    onlyTopicIndex == null
      ? tree.topics
      : tree.topics.filter((t) => t.index === onlyTopicIndex)
  const allSentences = shownTopics.flatMap((t) => t.turns.flatMap((k) => k.sentences))
  const enabledCount = allSentences.filter((s) => s.enabled).length
  const total = allSentences.length
  const lowCount = allSentences.filter((s) => s.importance === 'low').length

  return (
    <div className="qatree">
      <div className="qa-bulk">
        <span className="muted">
          残す {enabledCount} / {total} 文
        </span>
        <span className="qa-bulk-btns">
          {lowCount > 0 && (
            <>
              <button onClick={() => bulkByImportance('low', false)}>低を外す</button>
              <button onClick={() => bulkByImportance('low', true)}>低を戻す</button>
            </>
          )}
          <button onClick={() => setAll(true)}>全部戻す</button>
        </span>
      </div>
      <p className="qa-hint muted">時刻は元音声の位置です（最終クリップの時刻ではありません）。</p>

      {tree.topics.map((topic, ti) => {
        // onlyTopicIndex 指定時はそのトピック以外を描画しない (ti は全 tree 基準のまま)。
        if (onlyTopicIndex != null && topic.index !== onlyTopicIndex) return null
        const topicSents = topic.turns.flatMap((k) => k.sentences)
        return (
          <details className="qa-topic" key={ti} open>
            <summary>
              <TriCheckbox
                state={triOf(topicSents)}
                onToggle={() => setTopic(ti, triOf(topicSents) !== 'all')}
                title="トピックごと残す/外す"
              />
              {onPreviewTopic && (
                <button
                  type="button"
                  className={`qa-preview-btn ${previewIdx === topic.index ? 'active' : ''}`}
                  title="このトピックをプレビュー"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    onPreviewTopic(topic.index)
                  }}
                >
                  ▶
                </button>
              )}
              <span className="qa-topic-label">
                {topic.label || `トピック ${topic.index + 1}`}
              </span>
              <span className="qa-span">
                {topicSents.length > 0
                  ? `${fmt(topicSents[0].start)}–${fmt(topicSents[topicSents.length - 1].end)}`
                  : ''}
              </span>
            </summary>

            {topic.turns.map((turn, ki) => (
              <div className="qa-turn" key={ki}>
                <div className="qa-turn-head">
                  <TriCheckbox
                    state={triOf(turn.sentences)}
                    onToggle={() => setTurn(ti, ki, triOf(turn.sentences) !== 'all')}
                    title="発言者ごと残す/外す"
                  />
                  <span className="qa-speaker">
                    {turn.speaker}
                    <small> ・{turn.role}</small>
                  </span>
                </div>
                {turn.sentences.map((s, si) => (
                  <div className={`qa-sent ${s.enabled ? 'on' : 'off'}`} key={si}>
                    <input
                      type="checkbox"
                      checked={s.enabled}
                      onChange={(e) => setSentence(ti, ki, si, e.target.checked)}
                      title="残す/外す"
                    />
                    <span className={`imp-badge imp-${s.importance}`}>
                      {IMP_LABEL[s.importance]}
                    </span>
                    <button
                      type="button"
                      className="qa-sent-text"
                      title={`クリックで再生: ${s.text}`}
                      onClick={() => onSeek?.(s.start)}
                    >
                      {s.summary || s.text}
                    </button>
                    <span className="qa-span">{fmt(s.start)}</span>
                  </div>
                ))}
              </div>
            ))}
          </details>
        )
      })}
    </div>
  )
}
