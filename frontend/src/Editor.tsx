import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  api,
  type ClipProject,
  type EditRange,
  type Job,
  type QATopic,
  type QATree,
  type TopicClip,
  type TopicPreview,
} from './api'
import { QATreeEditor } from './QATreeEditor'
import { CaptionModal } from './CaptionModal'
import { ToastStack, useToast } from './Toast'

function fmt(t: number): string {
  const m = Math.floor(t / 60)
  const s = (t % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

const EPS = 1e-6

/** トピックの全文 (オン/オフ問わず) の member-WAV span [start,end]。無ければ null。 */
function topicSpan(topic: QATopic): [number, number] | null {
  const sents = topic.turns.flatMap((t) => t.sentences).filter((s) => s.end > s.start)
  if (sents.length === 0) return null
  return [
    Math.min(...sents.map((s) => s.start)),
    Math.max(...sents.map((s) => s.end)),
  ]
}

/** トピックのプレビュー window = 中点がトピック span に入る全 range (オフ含む)。 */
function topicWindow(topic: QATopic, ranges: EditRange[]): EditRange[] {
  const span = topicSpan(topic)
  if (!span) return []
  const [s, e] = span
  return ranges.filter((r) => {
    const mid = (r.start + r.end) / 2
    return s - EPS <= mid && mid <= e + EPS
  })
}

/** member-WAV 時刻 t を window 連結プレビューのローカル時間へ (バックエンド _window_local_time と同一)。 */
function memberToLocal(t: number, window: EditRange[]): number | null {
  let acc = 0
  for (const r of window) {
    if (r.start <= t && t <= r.end) return acc + (t - r.start)
    acc += r.end - r.start
  }
  return null
}

/**
 * クリップ生成後の編集パネル (トピック単位オンデマンドプレビュー)。
 * - 生成時に全体クリップは焼かない。各トピックを ▶ で 2〜4 秒で焼いてプレビュー。
 * - プレビューはトピック全文 (オフ含む) を焼き、オフ文区間は再生時スキップ。
 * - 文クリックでその箇所へシーク。書き出しは「トピック別/全体」ボタン。
 */
export function Editor({
  sessionId,
  member,
  focusTopic,
}: {
  sessionId: string
  member: string
  // 指定時はそのトピック1件に絞った編集画面 (一覧→選択で開く)。未指定は全トピック。
  focusTopic?: number
  onRerendered?: (clipPath: string) => void
}) {
  const [project, setProject] = useState<ClipProject | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [clips, setClips] = useState<TopicClip[]>([]) // 書き出し済みクリップ
  const [activeTab, setActiveTab] = useState(0)
  const [cacheBust, setCacheBust] = useState(0)
  const [modalIndex, setModalIndex] = useState<number | null>(null)
  // 右ペインのタブ: 発言の取捨 / 字幕
  const [rightTab, setRightTab] = useState<'cuts' | 'captions'>('cuts')
  const { toasts, push: pushToast, dismiss: dismissToast } = useToast()

  // 現在プレビュー中のトピック
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [preview, setPreview] = useState<TopicPreview | null>(null)
  const [previewing, setPreviewing] = useState(false)

  const pollRef = useRef<number | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const loopRef = useRef<[number, number] | null>(null)

  useEffect(() => {
    let alive = true
    setLoading(true)
    api
      .getProject(sessionId, member)
      .then((p) => alive && setProject(p))
      .catch((e) => alive && setError(`プロジェクト取得失敗: ${e}`))
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
  }, [sessionId, member])

  useEffect(() => () => stopPoll(), [])
  function stopPoll() {
    if (pollRef.current) window.clearInterval(pollRef.current)
    pollRef.current = null
  }

  // トピック index をオンデマンドで焼いてプレビュー表示する。
  const requestPreview = useCallback(
    async (idx: number) => {
      if (!project) return
      setError(null)
      setPreviewIdx(idx)
      setPreviewing(true)
      loopRef.current = null
      try {
        const { job_id } = await api.previewTopic(sessionId, member, idx, project)
        stopPoll()
        pollRef.current = window.setInterval(async () => {
          const j = await api.getJob(job_id)
          if (j.state === 'done') {
            stopPoll()
            setPreviewing(false)
            setPreview((j.result as unknown as TopicPreview) ?? null)
            setCacheBust((n) => n + 1)
          } else if (j.state === 'error') {
            stopPoll()
            setPreviewing(false)
            setError(j.error || 'プレビュー生成失敗')
          }
        }, 800)
      } catch (e) {
        setPreviewing(false)
        setError(`プレビュー生成失敗: ${e}`)
      }
    },
    [project, sessionId, member],
  )

  // 編集画面を開いたら対象トピック (focusTopic、無ければ先頭) を自動プレビュー。
  // focusTopic が変わる (一覧から別トピックを選び直す) たびに焼き直す。
  const autoPreviewedRef = useRef<number | null>(null)
  useEffect(() => {
    const tree = project?.qa_tree
    if (!tree || tree.topics.length === 0) return
    const target = focusTopic ?? 0
    if (autoPreviewedRef.current === target) return
    if (!tree.topics.some((t) => t.index === target)) return
    autoPreviewedRef.current = target
    void requestPreview(target)
  }, [project, focusTopic, requestPreview])

  function setCaption(i: number, text: string) {
    if (!project) return
    const captions = project.captions.map((c, j) =>
      j === i ? { ...c, text, edited: true } : c,
    )
    setProject({ ...project, captions })
  }
  function setTitle(title: string) {
    if (!project) return
    setProject({ ...project, title })
  }
  function setQaTree(qa_tree: QATree) {
    if (!project) return
    setProject({ ...project, qa_tree })
  }

  // 現在プレビュー中トピックの window (member-WAV→ローカル変換用)。
  const activeWindow = useMemo(() => {
    if (!project?.qa_tree || previewIdx === null) return []
    const topic = project.qa_tree.topics[previewIdx]
    return topic ? topicWindow(topic, project.ranges) : []
  }, [project, previewIdx])

  // 現在プレビュー中トピックに属する caption (member-WAV span が window 内)。
  // caption は全体カット後時間なので member-WAV へ戻す代わりに、window の member-WAV
  // 範囲に start が入るかで判定する近似 (プレビューはトピック区間のみなので十分)。
  const topicCaptionIdx = useMemo(() => {
    if (!project || activeWindow.length === 0) return [] as number[]
    const lo = activeWindow[0].start
    const hi = activeWindow[activeWindow.length - 1].end
    // caption.start は全体カット後時間。member-WAV へ逆算して範囲判定。
    const idxs: number[] = []
    project.captions.forEach((c, i) => {
      const src = postCutToMember(c.start, project.ranges)
      if (src !== null && lo - 0.5 <= src && src <= hi + 0.5) idxs.push(i)
    })
    return idxs
  }, [project, activeWindow])

  // 「残す N / M 発言」のカウント (focusTopic 指定時はそのトピック、未指定は全体)。
  const sentCounts = useMemo(() => {
    const tree = project?.qa_tree
    if (!tree) return { kept: 0, total: 0 }
    const topics =
      focusTopic != null ? tree.topics.filter((t) => t.index === focusTopic) : tree.topics
    const sents = topics.flatMap((t) => t.turns.flatMap((k) => k.sentences))
    return { kept: sents.filter((s) => s.enabled).length, total: sents.length }
  }, [project, focusTopic])

  // 文テキストのクリック → その文の箇所へシーク (member-WAV→ローカル)。
  function seekToMember(memberStart: number) {
    const v = videoRef.current
    if (!v) return
    const local = memberToLocal(memberStart, activeWindow)
    if (local === null) return
    loopRef.current = null
    v.currentTime = local
    void v.play()
  }

  // モーダル行の ▶: その caption をループ再生 (caption を member-WAV→ローカルへ)。
  function loopPlayCaption(i: number) {
    const v = videoRef.current
    if (!v || !project) return
    const src = postCutToMember(project.captions[i].start, project.ranges)
    const srcEnd = postCutToMember(project.captions[i].end, project.ranges)
    if (src === null || srcEnd === null) return
    const a = memberToLocal(src, activeWindow)
    const b = memberToLocal(srcEnd, activeWindow)
    if (a === null || b === null || b <= a) return
    loopRef.current = [a, b]
    v.currentTime = a
    void v.play()
  }

  // 動画の字幕領域クリック → 今の位置に対応するトピック内 caption の編集モーダル。
  function openCaptionEditor() {
    const v = videoRef.current
    if (!v || topicCaptionIdx.length === 0) return
    // 現在のローカル時刻に最も近いトピック caption を選ぶ。
    const t = v.currentTime
    let best = topicCaptionIdx[0]
    let bestd = Infinity
    for (const i of topicCaptionIdx) {
      const src = postCutToMember(project!.captions[i].start, project!.ranges)
      const local = src === null ? null : memberToLocal(src, activeWindow)
      if (local === null) continue
      const d = Math.abs(local - t)
      if (d < bestd) {
        bestd = d
        best = i
      }
    }
    v.pause()
    loopRef.current = null
    setModalIndex(best)
  }

  async function exportClips(mode: 'topics' | 'full' | 'both') {
    if (!project) return
    setError(null)
    try {
      // フォーカス編集中はそのトピックだけ書き出す。
      const onlyTopic = mode === 'topics' ? focusTopic : undefined
      const { job_id } = await api.exportClips(sessionId, member, mode, project, onlyTopic)
      setJob({ id: job_id, kind: 'export', state: 'queued', step: 'rendering', result: null, error: null, meta: {} })
      stopPoll()
      pollRef.current = window.setInterval(async () => {
        const j = await api.getJob(job_id)
        setJob(j)
        if (j.state === 'done') {
          stopPoll()
          const out = (j.result as { clips?: TopicClip[] } | null)?.clips ?? []
          setClips(out)
          setActiveTab(0)
          setCacheBust((n) => n + 1)
          pushToast(
            'success',
            out.length === 1 ? '書き出しが完了しました' : `${out.length}本の書き出しが完了しました`,
          )
        } else if (j.state === 'error') {
          stopPoll()
          setError(j.error || '書き出し失敗')
          pushToast('error', '書き出しに失敗しました')
        }
      }, 1000)
    } catch (e) {
      setError(`書き出し失敗: ${e}`)
      pushToast('error', '書き出しを開始できませんでした')
    }
  }

  if (loading) return <p className="muted loading-dots">エディタを読み込み中</p>
  if (error && !project) return <div className="error">⚠ {error}</div>
  if (!project) return null

  const rendering = job != null && job.state !== 'done' && job.state !== 'error'
  const editedCount = project.captions.filter((c) => c.edited).length
  const tree = project.qa_tree
  const treeEnabledLeaves = tree
    ? tree.topics.flatMap((t) => t.turns.flatMap((k) => k.sentences)).filter((s) => s.enabled).length
    : 1
  const emptySelection = !!tree && treeEnabledLeaves === 0
  const skipSpans = preview?.disabled_spans ?? []
  const previewUrl = preview ? `${api.clipFileUrl(preview.clip_path)}?v=${cacheBust}` : null

  // 再生中: ループ区間優先、なければオフ区間 (disabled_spans, ローカル時間) をスキップ。
  function onTimeUpdate() {
    const v = videoRef.current
    if (!v) return
    const loop = loopRef.current
    if (loop) {
      if (v.currentTime >= loop[1] - 0.02 || v.currentTime < loop[0] - 0.3) v.currentTime = loop[0]
      return
    }
    const t = v.currentTime
    for (const [s, e] of skipSpans) {
      if (t >= s - 0.02 && t < e - 0.03) {
        v.currentTime = e
        break
      }
    }
  }

  return (
    <div className="editor2">
      {/* 左ペイン: 暗い動画ステージ */}
      <div className="stage">
        <div className="player-wrap">
          {previewUrl ? (
            <video
              key={previewUrl}
              ref={videoRef}
              src={previewUrl}
              controls
              autoPlay
              className="player"
              onTimeUpdate={onTimeUpdate}
            />
          ) : (
            <div className="preview-placeholder">
              <span className="loading-dots">プレビューを準備中</span>
            </div>
          )}
          {previewUrl && (
            <button className="caption-hit" onClick={openCaptionEditor} title="字幕をクリックして修正">
              <span className="caption-hit-hint">字幕をクリックで修正</span>
            </button>
          )}
        </div>
        <p className="stage-note">これがそのまま書き出される動画です{previewing && ' ・更新中…'}</p>
        <div className="stage-tools">
          <button
            className="ghost-btn"
            onClick={() => previewIdx !== null && requestPreview(previewIdx)}
            disabled={previewing || previewIdx === null}
            title="先頭から見直す"
          >
            ⟲ 頭出し
          </button>
          <button
            className="ghost-btn"
            onClick={openCaptionEditor}
            disabled={!previewUrl}
            title="字幕を直す"
          >
            ✎ 字幕を直す
          </button>
        </div>
        <label className="stage-title">
          タイトル（｜で改行位置）
          <input type="text" value={project.title} onChange={(e) => setTitle(e.target.value)} />
        </label>
      </div>

      {/* 右ペイン: タブ (発言の取捨 / 字幕) */}
      <div className="rightpane">
        {tree ? (
          <>
            <div className="rtabs">
              <button
                className={`rtab ${rightTab === 'cuts' ? 'active' : ''}`}
                onClick={() => setRightTab('cuts')}
              >
                発言の取捨
                <small> {sentCounts.kept}/{sentCounts.total}</small>
              </button>
              <button
                className={`rtab ${rightTab === 'captions' ? 'active' : ''}`}
                onClick={() => setRightTab('captions')}
              >
                字幕
                <small> {focusTopic != null ? topicCaptionIdx.length : project.captions.length}</small>
                {editedCount > 0 && <span className="rtab-dot" title={`${editedCount}件修正`} />}
              </button>
            </div>

            {rightTab === 'cuts' ? (
              <div className="rtab-body">
                <p className="qa-hint muted">
                  外した発言はプレビューで自動スキップ。発言をクリックでその箇所を再生。
                </p>
                <QATreeEditor
                  tree={tree}
                  onChange={setQaTree}
                  onSeek={seekToMember}
                  onPreviewTopic={requestPreview}
                  previewIdx={previewIdx}
                  onlyTopicIndex={focusTopic}
                />
              </div>
            ) : (
              <div className="rtab-body">
                <p className="qa-hint muted">
                  行クリック（or 動画の字幕部分クリック）で修正。修正は書き出し後に反映。
                </p>
                <div className="captions">
                  {project.captions.map((c, i) => {
                    if (focusTopic != null && !topicCaptionIdx.includes(i)) return null
                    return (
                      <button
                        key={i}
                        className={`cap-row ${c.edited ? 'edited' : ''} ${topicCaptionIdx.includes(i) ? 'in-topic' : ''}`}
                        onClick={() => {
                          videoRef.current?.pause()
                          loopRef.current = null
                          setModalIndex(i)
                        }}
                        title="クリックで修正"
                      >
                        <span className="cap-t">{fmt(c.start)}</span>
                        <span className="cap-row-text">{c.text}</span>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="muted">このプロジェクトには Q&A ツリーがありません (旧形式)。</p>
        )}
      </div>

      {/* 下部アクションバー: 残す件数 + 書き出し */}
      <div className="actionbar2">
        {tree && (
          <span className="actionbar2-sum muted">
            残す {sentCounts.kept} / {sentCounts.total} 発言
            {preview && ` ・ 約 ${fmt(preview.duration)}`}
          </span>
        )}
        <div className="actionbar2-btns">
          {tree ? (
            focusTopic != null ? (
              <button
                className="primary"
                onClick={() => exportClips('topics')}
                disabled={rendering || emptySelection}
              >
                {rendering ? '書き出し中…' : 'このトピックを書き出す'}
              </button>
            ) : (
              <>
                <button
                  className="primary"
                  onClick={() => exportClips('topics')}
                  disabled={rendering || emptySelection}
                >
                  {rendering ? '書き出し中…' : 'トピック別に書き出す'}
                </button>
                <button onClick={() => exportClips('full')} disabled={rendering || emptySelection}>
                  全体を書き出す
                </button>
              </>
            )
          ) : (
            <button className="primary" onClick={() => exportClips('full')} disabled={rendering}>
              {rendering ? '書き出し中…' : '書き出す'}
            </button>
          )}
        </div>
        {emptySelection && <span className="muted actionbar2-warn">少なくとも1つの発言を残してください</span>}
        {error && <div className="error">⚠ {error}</div>}

        {clips.length > 0 && (
          <div className="export-result">
            <h3>書き出し済み ({clips.length})</h3>
            <div className="tabs">
              {clips.map((c, i) => (
                <button
                  key={i}
                  className={`tab ${i === activeTab ? 'active' : ''}`}
                  onClick={() => setActiveTab(i)}
                >
                  {c.topic_index === null ? '全体' : `${c.topic_index + 1}. ${c.topic_label || 'トピック'}`}
                  <small> {fmt(c.duration)}</small>
                </button>
              ))}
            </div>
            {clips[activeTab] && (
              <div className="tab-panel">
                <video
                  key={`${activeTab}-${cacheBust}`}
                  src={`${api.clipFileUrl(clips[activeTab].clip_path)}?v=${cacheBust}`}
                  controls
                  className="player"
                />
                <a className="button" href={api.clipFileUrl(clips[activeTab].clip_path)} download>
                  この動画をダウンロード
                </a>
              </div>
            )}
          </div>
        )}
      </div>

      {modalIndex !== null && (
        <CaptionModal
          captions={project.captions}
          focusIndex={modalIndex}
          onEdit={setCaption}
          onLoopPlay={loopPlayCaption}
          onClose={() => {
            loopRef.current = null
            videoRef.current?.pause()
            setModalIndex(null)
          }}
        />
      )}

      <ToastStack toasts={toasts} onDismiss={dismissToast} />
    </div>
  )
}

/**
 * 全体カット後時刻 t を member-WAV (ソース) 時刻へ逆算する。
 * memberToClip の逆。t が属する range を全 range の累積で特定し、その range 内オフセットを足す。
 */
function postCutToMember(t: number, ranges: EditRange[]): number | null {
  let acc = 0
  for (const r of ranges) {
    const dur = r.end - r.start
    if (acc <= t && t <= acc + dur + EPS) return r.start + (t - acc)
    acc += dur
  }
  return null
}
