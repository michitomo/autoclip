// 統合UI (GitHub Pages 単体・カタログ駆動)。
// 日付→委員会→議員 のウィザード。情報源は静的JSON: corpus.json (母集合) と
// catalog.json (生成済みクリップ)。議員選択後:
//   - 生成済み → ブラウザ(WebCodecs)でトピックを選んでレンダ→DL
//   - 未生成   → GitHub Issue のプリフィルへ誘導 (生成リクエスト)
// backend (/api) には依存しない。

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { DatePicker } from './DatePicker'
import {
  loadCorpus, loadCatalog,
  type Corpus, type CorpusSession, type CorpusMember, type Catalog, type CatalogClip,
} from './render/catalog'
import { clipRequestIssueUrl } from './issue'
import { renderClip } from './render/compose'
import { loadClip } from './render/catalog'
import { buildSubtitleData, aspectToSize, topicEdl } from './render/fromProject'
import { pageList } from './render/subtitle'

const WD = ['日', '月', '火', '水', '木', '金', '土']
function fmtDateLabel(iso: string): string {
  const [y, m, d] = iso.split('-').map(Number)
  if (!y || !m || !d) return iso
  const wd = WD[new Date(y, m - 1, d).getDay()]
  return `${m}月${d}日(${wd})`
}

function todayISO(): string {
  const d = new Date()
  const off = d.getTimezoneOffset()
  return new Date(d.getTime() - off * 60000).toISOString().slice(0, 10)
}

type WizardStep = 'date' | 'committee' | 'member' | 'render'
const STEPS: { key: WizardStep; label: string }[] = [
  { key: 'date', label: '日付' },
  { key: 'committee', label: '委員会' },
  { key: 'member', label: '議員' },
  { key: 'render', label: '書き出し' },
]

export default function App() {
  return <WizardApp />
}

function WizardApp() {
  const [corpus, setCorpus] = useState<Corpus | null>(null)
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [loadErr, setLoadErr] = useState<string | null>(null)

  const [date, setDate] = useState(todayISO())
  const [session, setSession] = useState<CorpusSession | null>(null)
  const [member, setMember] = useState<CorpusMember | null>(null)
  const [step, setStep] = useState<WizardStep>('date')

  // 起動時に corpus / catalog を並行ロード
  useEffect(() => {
    Promise.all([loadCorpus(), loadCatalog()])
      .then(([c, cat]) => { setCorpus(c); setCatalog(cat) })
      .catch((e) => setLoadErr(`データ取得失敗: ${e}`))
  }, [])

  // ?clip=<id> ディープリンク: 生成済みクリップを直接 render ステップで開く
  useEffect(() => {
    if (!corpus || !catalog) return
    const cid = new URLSearchParams(window.location.search).get('clip')
    if (!cid) return
    const found = findMemberByClipId(corpus, cid)
    if (found) { setDate(found.date); setSession(found.session); setMember(found.member); setStep('render') }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [corpus, catalog])

  // corpus → DatePicker 用の "日付 → 委員会名[]" マップ
  const dayCommittees = useMemo(() => {
    const m: Record<string, string[]> = {}
    if (corpus) for (const [d, sessions] of Object.entries(corpus.days)) {
      m[d] = sessions.map((s) => s.committee)
    }
    return m
  }, [corpus])

  // 生成済み clip_id の集合 (議員が生成済みか判定)
  const generatedIds = useMemo(
    () => new Set((catalog?.clips ?? []).map((c) => c.id)),
    [catalog],
  )

  // 日付 → その日に生成済みクリップが1つでもあるか (日付チップの色分け用)
  const dayGenerated = useMemo(() => {
    const m: Record<string, boolean> = {}
    if (corpus) for (const [d, sessions] of Object.entries(corpus.days)) {
      m[d] = sessions.some((s) => s.members.some((mem) => generatedIds.has(mem.clip_id)))
    }
    return m
  }, [corpus, generatedIds])

  const sessionsForDate: CorpusSession[] = (corpus?.days[date]) ?? []

  function goDate(d: string) {
    setDate(d); setSession(null); setMember(null); setStep('committee')
  }
  function pickSession(s: CorpusSession) {
    setSession(s); setMember(null); setStep('member')
  }
  function pickMember(m: CorpusMember) {
    setMember(m); setStep('render')
  }

  const reachable: Record<WizardStep, boolean> = {
    date: true,
    committee: sessionsForDate.length > 0 || session != null,
    member: session != null,
    render: member != null,
  }
  const stepIndex = STEPS.findIndex((s) => s.key === step)
  function goToStep(t: WizardStep) { if (reachable[t]) setStep(t) }

  return (
    <div className="app">
      <header>
        <h1>autoclip</h1>
        <span className="sub">衆議院TV 議員クリップ — ブラウザで書き出し</span>
      </header>

      <StepBar
        current={step}
        currentIndex={stepIndex}
        reachable={reachable}
        onGo={goToStep}
        selections={{
          date: step !== 'date' ? date : null,
          committee: session?.committee ?? null,
          member: member?.name ?? null,
        }}
      />

      {loadErr && <div className="error">⚠ {loadErr}</div>}

      {step === 'date' && (
        <section>
          <h2>日付を選ぶ</h2>
          <DatePicker
            value={date}
            onChange={setDate}
            onSubmit={goDate}
            dayCommittees={dayCommittees}
            dayGenerated={dayGenerated}
            ready={corpus != null}
          />
        </section>
      )}

      {step === 'committee' && (
        <section>
          <h2>委員会を選ぶ</h2>
          <button className="date-back" onClick={() => setStep('date')}>
            <span className="date-back-ic">📅</span>
            {fmtDateLabel(date)} の審議
          </button>
          {sessionsForDate.length === 0 ? (
            <div className="empty">
              <span className="empty-ic">📭</span>
              <span className="empty-text">この日の審議は見つかりませんでした。<br />別の日を選んでください。</span>
            </div>
          ) : (
            <div className="chips">
              {sessionsForDate.map((s) => {
                const genCount = s.members.filter((m) => generatedIds.has(m.clip_id)).length
                const ready = genCount > 0
                return (
                  <button
                    key={s.session_id}
                    className={`chip ${session?.session_id === s.session_id ? 'active' : ''} ${ready ? 'chip-ready' : 'chip-pending'}`}
                    onClick={() => pickSession(s)}
                  >
                    <span className="chip-name">
                      {s.committee}
                      {ready && <span className="chip-badge rdy">✓ {genCount}本</span>}
                    </span>
                    <small>{s.members.length}名{s.duration ? ` / ${s.duration}` : ''}</small>
                  </button>
                )
              })}
            </div>
          )}
        </section>
      )}

      {step === 'member' && session && (
        <section>
          <h2>議員を選ぶ</h2>
          <p className="muted step-sub">{fmtDateLabel(date)} ・ {session.committee}</p>
          {session.members.length === 0 ? (
            <div className="empty">
              <span className="empty-ic">🔍</span>
              <span className="empty-text">議員一覧がまだ公開されていません。<br />しばらくしてからお試しください。</span>
            </div>
          ) : (
            <div className="chips">
              {session.members.map((m) => {
                const ready = generatedIds.has(m.clip_id)
                return (
                  <button
                    key={m.clip_id}
                    className={`chip ${member?.clip_id === m.clip_id ? 'active' : ''} ${ready ? 'chip-ready' : 'chip-pending'}`}
                    onClick={() => pickMember(m)}
                  >
                    <span className="chip-name">
                      {m.name}
                      <span className={`chip-badge ${ready ? 'rdy' : 'pend'}`}>
                        {ready ? '✓ 生成済み' : '未生成'}
                      </span>
                    </span>
                    <small>{m.affiliation} / {m.duration_minutes}分</small>
                  </button>
                )
              })}
            </div>
          )}
        </section>
      )}

      {step === 'render' && session && member && (
        generatedIds.has(member.clip_id) ? (
          <RenderView
            clip={catalog!.clips.find((c) => c.id === member.clip_id)!}
            onBack={() => setStep('member')}
          />
        ) : (
          <RequestView
            date={date}
            session={session}
            member={member}
            onBack={() => setStep('member')}
          />
        )
      )}

      <footer className="legal-note">
        映像の出典: <a href="https://www.shugiintv.go.jp/" target="_blank" rel="noreferrer">衆議院インターネット審議中継</a>。
        生成したクリップを利用・公開できるかどうかは、衆議院インターネット審議中継の利用規約等をご自身でご確認・ご判断ください。
        本サイトは利用の可否を判断・保証するものではありません。
      </footer>
    </div>
  )
}

// 未生成: GitHub Issue で生成リクエストへ誘導
function RequestView({
  date, session, member, onBack,
}: {
  date: string; session: CorpusSession; member: CorpusMember; onBack: () => void
}) {
  const url = clipRequestIssueUrl({
    sessionId: session.session_id, member: member.name,
    committee: session.committee, date,
  })
  return (
    <section>
      <div className="row editor-topbar">
        <button onClick={onBack}>← 議員選択へ戻る</button>
      </div>
      <h2>このクリップはまだありません</h2>
      <p className="muted step-sub">
        {member.name}（{member.affiliation}）・ {session.committee} ・ {fmtDateLabel(date)}
      </p>
      <div className="empty">
        <span className="empty-ic">🛠️</span>
        <span className="empty-text">
          このクリップはまだ生成されていません。<br />
          生成をリクエストすると自動で作成され、<strong>数分〜10分ほど</strong>でこのサイトに反映されて書き出せるようになります。
        </span>
      </div>
      <div className="row" style={{ marginTop: 14 }}>
        <a className="button primary" href={url} target="_blank" rel="noreferrer">
          生成をリクエスト（GitHub Issue）
        </a>
      </div>
      <p className="muted" style={{ marginTop: 10, fontSize: 13 }}>
        ※ GitHubのIssue作成画面が開きます。内容はそのままで「Submit new issue」を押すだけです。<br />
        リクエスト後は数分〜10分ほどで生成されます。完了したらこのページを再読み込みすると表示されます。
      </p>
    </section>
  )
}

// 生成済み: トピックを選ぶ → (字幕/タイトルを修正) → ブラウザでレンダ→DL。
function RenderView({ clip, onBack }: { clip: CatalogClip; onBack: () => void }) {
  const [topicIdx, setTopicIdx] = useState<number | null>(null)
  const dataRef = useRef<Awaited<ReturnType<typeof loadClip>> | null>(null)
  const [loadingTopic, setLoadingTopic] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function ensureData() {
    if (!dataRef.current) dataRef.current = await loadClip(clip.id)
    return dataRef.current
  }

  async function pickTopic(idx: number) {
    setError(null); setLoadingTopic(true)
    try {
      await ensureData()   // 先に project/edl を取っておく
      setTopicIdx(idx)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingTopic(false)
    }
  }

  // トピック選択中は編集+書き出しビュー
  if (topicIdx !== null && dataRef.current) {
    return (
      <TopicEditView
        clip={clip}
        topicIndex={topicIdx}
        data={dataRef.current}
        onBack={() => setTopicIdx(null)}
      />
    )
  }

  return (
    <section>
      <div className="row editor-topbar">
        <button onClick={onBack}>← 議員選択へ戻る</button>
      </div>
      <h2>{clip.member} のトピックを選ぶ</h2>
      <p className="muted step-sub">{clip.date} ・ {clip.committee}</p>

      <div className="topic-list">
        {clip.topics.map((t) => (
          <button
            key={t.index}
            className="topic-card"
            onClick={() => pickTopic(t.index)}
            disabled={loadingTopic}
          >
            <div className="topic-card-head">
              <span className="topic-card-num">{t.index + 1}</span>
              <span className="topic-card-label">{t.label || `トピック ${t.index + 1}`}</span>
            </div>
            <div className="topic-card-speakers">
              <span className="role-q">Q: {t.question_speaker || '—'}</span>
              {t.answer_speakers.length > 0 && (
                <span className="role-a">A: {t.answer_speakers.slice(0, 3).join('、')}</span>
              )}
            </div>
          </button>
        ))}
      </div>
      {loadingTopic && <p className="muted loading-dots" style={{ marginTop: 14 }}>読み込み中</p>}
      {error && <div className="error">⚠ {error}</div>}
    </section>
  )
}

// 1トピックの「字幕/タイトル修正 → 書き出し」ビュー。
function TopicEditView({
  clip, topicIndex, data, onBack,
}: {
  clip: CatalogClip
  topicIndex: number
  data: Awaited<ReturnType<typeof loadClip>>
  onBack: () => void
}) {
  const { project, edl } = data
  // このトピックに絞った EDL と、編集前の字幕ページ一覧。useMemo で安定化。
  const scoped = useMemo(() => topicEdl(edl, project, topicIndex), [edl, project, topicIndex])
  const basePages = useMemo(() => pageList(buildSubtitleData(project, scoped)), [project, scoped])

  // 編集状態 (ブラウザ内のみ・保存なし)
  const [title, setTitle] = useState(project.title || '')
  const [overrides, setOverrides] = useState<Record<string, string>>({})

  // レンダ状態。url = 現在表示中(書き出し済み)の動画。dirty = 最後の書き出し後に編集したか。
  const [busy, setBusy] = useState(false)
  // 進捗: ratio=フレーム書き出し率(0..1)、null=不定 (音声/仕上げ)。label=段階名。
  const [progress, setProgress] = useState<{ label: string; ratio: number | null; fps?: number }>(
    { label: '', ratio: null },
  )
  const [doneMsg, setDoneMsg] = useState('') // 完了/エラー後の一言
  const [error, setError] = useState<string | null>(null)
  const [url, setUrl] = useState<string | null>(null)
  const [dirty, setDirty] = useState(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const [activeKey, setActiveKey] = useState<string | null>(null)
  // 多重レンダ防止。busy(state) は非同期更新で間に合わないので ref で即時ガードする
  // (StrictMode の二重 effect / 連打で 2 本走ると進捗 % が行き来する)。
  const renderingRef = useRef(false)

  // 編集が入ったら dirty に (書き出しボタンを促す)
  function editPage(key: string, text: string) {
    setOverrides((o) => ({ ...o, [key]: text })); setDirty(true)
  }
  function resetPage(key: string) {
    setOverrides((o) => { const n = { ...o }; delete n[key]; return n }); setDirty(true)
  }
  function editTitle(v: string) { setTitle(v); setDirty(true) }

  const renderNow = useCallback(async () => {
    if (renderingRef.current) return // 既にレンダ中なら二重起動しない
    renderingRef.current = true
    setBusy(true); setError(null); setDoneMsg('')
    setProgress({ label: '動画を取得中…', ratio: null })
    try {
      const subtitle = buildSubtitleData(project, scoped)
      subtitle.title = { ...subtitle.title, title }
      if (Object.keys(overrides).length > 0) subtitle.textOverrides = overrides
      const { width, height } = aspectToSize(project.aspect)
      const res = await renderClip(project.hls_url, {
        edl: scoped, memberStart: project.member_start, subtitle,
        width, height, fps: 30,
        onProgress: (done, total, fps) =>
          setProgress({ label: '映像を書き出し中', ratio: total ? done / total : null, fps }),
        onStage: (stage) =>
          setProgress({ label: stage === 'audio' ? '音声を合成中…' : 'mp4 を仕上げ中…', ratio: null }),
      })
      setUrl((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(res.blob) })
      setDoneMsg(`更新しました (${res.elapsedSec.toFixed(1)}秒)`)
      setDirty(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      renderingRef.current = false
    }
  }, [project, scoped, title, overrides])

  // トピックを開いたら、まず既成のタイトル/字幕で1回自動レンダ (プレビュー生成)。
  useEffect(() => {
    void renderNow()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topicIndex])

  // 再生位置に追従して現在の字幕行をハイライト+自動スクロール。
  function onTimeUpdate() {
    const v = videoRef.current; if (!v) return
    const t = v.currentTime
    let cur: string | null = null
    for (const p of basePages) { if (p.start <= t + 0.05) cur = p.key; else break }
    if (cur !== activeKey) {
      setActiveKey(cur)
      if (cur && listRef.current) {
        const el = listRef.current.querySelector(`[data-key="${cur}"]`)
        el?.scrollIntoView({ block: 'center', behavior: 'smooth' })
      }
    }
  }

  function seekTo(start: number) {
    const v = videoRef.current; if (!v) return
    v.currentTime = Math.max(0, start + 0.01); void v.play()
  }

  const topic = clip.topics.find((t) => t.index === topicIndex)
  const editedCount = Object.keys(overrides).length

  return (
    <section>
      <div className="row editor-topbar">
        <button onClick={onBack} disabled={busy}>← トピック一覧へ戻る</button>
      </div>
      <h2>{topic?.label || `トピック ${topicIndex + 1}`}</h2>
      <p className="muted step-sub">{clip.member} ・ {clip.committee}</p>

      <div className="edit2">
        {/* 左: プレビュー動画。進捗バーは動画(またはプレースホルダ)の上にだけ重ねる。 */}
        <div className="edit2-video">
          <div className="edit2-video-frame">
            {url ? (
              <video
                ref={videoRef} src={url} controls className="player edit2-player"
                onTimeUpdate={onTimeUpdate}
              />
            ) : (
              <div className="edit2-video-ph">{!busy && 'プレビュー待ち'}</div>
            )}
            {busy && (
              <div className="edit2-video-overlay">
                <ProgressBar progress={progress} />
              </div>
            )}
          </div>
          {url && !busy && (
            <a className="button primary edit2-dl" href={url}
               download={`${clip.member}_topic${topicIndex + 1}.mp4`}>
              ⬇ ダウンロード
            </a>
          )}
        </div>

        {/* 右: タイトル + 字幕リスト */}
        <div className="edit2-side">
          <label className="wide">
            タイトル（「｜」で改行）
            <input type="text" value={title} onChange={(e) => editTitle(e.target.value)}
              placeholder="例: 国保システム｜発注管理のガバナンス" />
          </label>

          <div className="edit2-subhead">
            <span>字幕（クリックでその場面へ）</span>
            {editedCount > 0 && <span className="edit2-editmark">{editedCount}件 修正</span>}
          </div>
          <div className="sub-edit-list edit2-list" ref={listRef}>
            {basePages.map((p) => {
              const cur = overrides[p.key] ?? p.text
              const edited = overrides[p.key] != null && overrides[p.key] !== p.text
              const active = p.key === activeKey
              return (
                <div key={p.key} data-key={p.key}
                     className={`sub-edit-row ${edited ? 'edited' : ''} ${active ? 'active' : ''}`}>
                  <button className="sub-edit-time as-link" onClick={() => seekTo(p.start)}
                          title="この場面へ">{fmtClock(p.start)}</button>
                  <input className="sub-edit-text" value={cur}
                         onChange={(e) => editPage(p.key, e.target.value)} />
                  {edited && (
                    <button className="sub-edit-reset" title="元に戻す"
                            onClick={() => resetPage(p.key)}>↺</button>
                  )}
                </div>
              )
            })}
          </div>

          <div className="row edit2-actions">
            <button className="primary" onClick={renderNow} disabled={busy || !dirty}>
              {busy ? '書き出し中…' : dirty ? '更新して書き出す' : '最新です'}
            </button>
            {!busy && doneMsg && <span className="muted edit2-prog">{doneMsg}</span>}
          </div>
          {error && <div className="error">⚠ {error}</div>}
        </div>
      </div>
    </section>
  )
}

function fmtClock(t: number): string {
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

// レンダ進捗バー。ratio があれば割合バー、無ければ不定 (アニメ) バー。
function ProgressBar({ progress }: { progress: { label: string; ratio: number | null; fps?: number } }) {
  const { label, ratio, fps } = progress
  const pct = ratio != null ? Math.round(ratio * 100) : null
  return (
    <div className="prog">
      <div className="prog-row">
        <span className="prog-label">{label}</span>
        <span className="prog-meta">
          {pct != null ? `${pct}%` : ''}{fps ? ` ・ ${fps.toFixed(0)} fps` : ''}
        </span>
      </div>
      <div className={`prog-track ${ratio == null ? 'indet' : ''}`}>
        <div className="prog-fill" style={ratio != null ? { width: `${pct}%` } : undefined} />
      </div>
    </div>
  )
}

/** corpus 全体から clip_id で議員/セッション/日付を逆引き (?clip= ディープリンク用)。 */
function findMemberByClipId(
  corpus: Corpus, clipId: string,
): { date: string; session: CorpusSession; member: CorpusMember } | null {
  for (const [date, sessions] of Object.entries(corpus.days)) {
    for (const session of sessions) {
      const member = session.members.find((m) => m.clip_id === clipId)
      if (member) return { date, session, member }
    }
  }
  return null
}

function StepBar({
  current, currentIndex, reachable, onGo, selections,
}: {
  current: WizardStep
  currentIndex: number
  reachable: Record<WizardStep, boolean>
  onGo: (s: WizardStep) => void
  selections: { date: string | null; committee: string | null; member: string | null }
}) {
  return (
    <nav className="stepbar" aria-label="進行状況">
      {STEPS.map((s, i) => {
        const isCurrent = s.key === current
        const isDone = i < currentIndex && reachable[s.key]
        const clickable = reachable[s.key] && !isCurrent
        const sub =
          s.key === 'date' ? (selections.date ? fmtDateLabel(selections.date) : null)
          : s.key === 'committee' ? selections.committee
          : s.key === 'member' ? selections.member
          : null
        return (
          <div className="stepbar-item" key={s.key}>
            {i > 0 && <span className={`stepbar-bar ${i <= currentIndex ? 'done' : ''}`} />}
            <button
              type="button"
              className={
                'stepbar-node' +
                (isCurrent ? ' current' : '') +
                (isDone ? ' done' : '') +
                (clickable ? ' clickable' : '')
              }
              disabled={!clickable}
              onClick={() => onGo(s.key)}
              aria-current={isCurrent ? 'step' : undefined}
            >
              <span className="stepbar-dot">{isDone ? '✓' : i + 1}</span>
              <span className="stepbar-text">
                <span className="stepbar-label">{s.label}</span>
                {sub && <span className="stepbar-val">{sub}</span>}
              </span>
            </button>
          </div>
        )
      })}
    </nav>
  )
}
