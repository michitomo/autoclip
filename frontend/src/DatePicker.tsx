import { useMemo, useState } from 'react'

// ===== 日付ユーティリティ (ローカル時間。衆議院TVは JST 前提だが端末時計でよい) =====

function ymd(d: Date): string {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}
function addDays(d: Date, n: number): Date {
  const r = new Date(d)
  r.setDate(r.getDate() + n)
  return r
}
function isWeekend(d: Date): boolean {
  const g = d.getDay()
  return g === 0 || g === 6
}
function sameYmd(a: Date, b: Date): boolean {
  return ymd(a) === ymd(b)
}

const WD = ['日', '月', '火', '水', '木', '金', '土']

/** 今日から遡って直近 n 営業日 (平日) の Date を新しい順で返す。 */
function recentWeekdays(today: Date, n: number): Date[] {
  const out: Date[] = []
  let cur = new Date(today)
  while (out.length < n) {
    if (!isWeekend(cur)) out.push(new Date(cur))
    cur = addDays(cur, -1)
  }
  return out
}

/** チップの相対ラベル (今日/昨日/一昨日、それ以前は M/D(曜))。 */
function chipLabel(d: Date, today: Date): string {
  const diff = Math.round((today.getTime() - d.getTime()) / 86400000)
  if (diff === 0) return '今日'
  if (diff === 1) return '昨日'
  if (diff === 2) return '一昨日'
  return `${d.getMonth() + 1}/${d.getDate()}(${WD[d.getDay()]})`
}

// 委員会名を短縮 (「厚生労働委員会」→「厚生労働委」)。末尾の「委員会」だけ詰める。
// 過度な圧縮 (例: 経済産業委→経済委) は別委員会と紛らわしいので行わない。
function shortCommittee(name: string): string {
  return name.replace(/委員会$/, '委').replace(/審査会$/, '審')
}

// 親 (App) が corpus から作る「日付 → 委員会名[]」マップ。審議がある日のみキーを持つ。
// 値が空配列 = その日のキー自体が無い (審議なし)。
export type DayCommittees = Record<string, string[]>

// ===== 月カレンダー (平日のみ選択可・未来と週末はグレーアウト) =====

function MonthCalendar({
  month,
  today,
  selected,
  onPick,
  onPrev,
  onNext,
}: {
  month: Date // その月の1日
  today: Date
  selected: string
  onPick: (date: string) => void
  onPrev: () => void
  onNext: () => void
}) {
  const first = new Date(month.getFullYear(), month.getMonth(), 1)
  const startPad = first.getDay() // 日曜始まり
  const daysInMonth = new Date(month.getFullYear(), month.getMonth() + 1, 0).getDate()
  const cells: (Date | null)[] = []
  for (let i = 0; i < startPad; i++) cells.push(null)
  for (let d = 1; d <= daysInMonth; d++)
    cells.push(new Date(month.getFullYear(), month.getMonth(), d))
  while (cells.length % 7 !== 0) cells.push(null)

  const canNext = month.getFullYear() < today.getFullYear() || month.getMonth() < today.getMonth()

  return (
    <div className="cal">
      <div className="cal-head">
        <button type="button" className="cal-nav" onClick={onPrev} aria-label="前の月">
          ‹
        </button>
        <span className="cal-title">
          {month.getFullYear()}年 {month.getMonth() + 1}月
        </span>
        <button
          type="button"
          className="cal-nav"
          onClick={onNext}
          disabled={!canNext}
          aria-label="次の月"
        >
          ›
        </button>
      </div>
      <div className="cal-grid cal-dow">
        {WD.map((w, i) => (
          <span key={w} className={`cal-dow-cell ${i === 0 || i === 6 ? 'we' : ''}`}>
            {w}
          </span>
        ))}
      </div>
      <div className="cal-grid">
        {cells.map((d, i) => {
          if (!d) return <span key={i} className="cal-cell empty" />
          const future = d.getTime() > today.getTime()
          const weekend = isWeekend(d)
          const disabled = future || weekend
          return (
            <button
              key={i}
              type="button"
              className={
                'cal-cell' +
                (disabled ? ' disabled' : '') +
                (sameYmd(d, today) ? ' today' : '') +
                (ymd(d) === selected ? ' sel' : '')
              }
              disabled={disabled}
              onClick={() => onPick(ymd(d))}
            >
              {d.getDate()}
            </button>
          )
        })}
      </div>
      <p className="cal-foot muted">平日のみ選べます。</p>
    </div>
  )
}

// ===== 本体 =====

/**
 * 日付ピッカー: 直近の平日チップ + 自前カレンダー。
 * - チップ/選択日の委員会名は親が corpus から作った dayCommittees から引く (fetch 無し)。
 * - 日付クリックで onSubmit(date) → 親がその日の委員会一覧へ進む。
 */
export function DatePicker({
  value,
  onChange,
  onSubmit,
  loading,
  dayCommittees,
  dayGenerated,
  ready,
}: {
  value: string
  onChange: (date: string) => void
  onSubmit: (date: string) => void
  loading?: boolean
  dayCommittees: DayCommittees // 親が corpus から作る "YYYY-MM-DD" → 委員会名[]
  dayGenerated?: Record<string, boolean> // その日に生成済みクリップがあるか (色分け用)
  ready?: boolean // corpus ロード済みか (未ロード中は「確認中…」表示)
}) {
  const today = useMemo(() => {
    const t = new Date()
    return new Date(t.getFullYear(), t.getMonth(), t.getDate())
  }, [])
  const chips = useMemo(() => recentWeekdays(today, 5), [today])

  const [calOpen, setCalOpen] = useState(false)
  const [calMonth, setCalMonth] = useState(() => new Date(today.getFullYear(), today.getMonth(), 1))

  // 日付を選んだら即その日へ進む (選択状態も更新)。
  function pick(date: string) {
    onChange(date)
    onSubmit(date)
  }

  return (
    <div className="datepick">
      <p className="qa-hint muted">日付を選ぶと、その日の委員会一覧に進みます。</p>
      <div className="datepick-chips">
        {/* 古い → 新しい (右端が今日)。recentWeekdays は新しい順なので反転して表示。 */}
        {[...chips].reverse().map((d) => {
          const key = ymd(d)
          const active = key === value
          const names = (dayCommittees[key] ?? []).map(shortCommittee)
          const none = ready && names.length === 0
          // 審議はあるが生成済みクリップが無い日 = 未生成 (グレー寄せ)。
          const hasGen = !!dayGenerated?.[key]
          const pending = ready && names.length > 0 && !hasGen
          return (
            <button
              key={key}
              type="button"
              className={`daychip ${active ? 'active' : ''} ${none ? 'empty-day' : ''} ${pending ? 'pending-day' : ''} ${hasGen ? 'gen-day' : ''}`}
              onClick={() => pick(key)}
              disabled={loading}
              title={`${key} の委員会を見る`}
            >
              <span className="daychip-label">
                {chipLabel(d, today)}
                {hasGen && <span className="daychip-gen" title="生成済みクリップあり">✓</span>}
              </span>
              <span className="daychip-date">
                {d.getMonth() + 1}/{d.getDate()}({WD[d.getDay()]})
              </span>
              <span className="daychip-coms">
                {!ready ? (
                  <span className="daychip-dim">確認中…</span>
                ) : none ? (
                  <span className="daychip-dim">審議なし</span>
                ) : (
                  names.map((n, i) => (
                    <span className="daychip-com" key={i}>
                      {n}
                    </span>
                  ))
                )}
              </span>
            </button>
          )
        })}
        <button
          type="button"
          className={`daychip cal-toggle ${calOpen ? 'active' : ''}`}
          onClick={() => setCalOpen((o) => !o)}
        >
          <span className="daychip-label">📅 別の日</span>
          <span className="daychip-sub daychip-dim">カレンダー</span>
        </button>
      </div>

      {calOpen && (
        <MonthCalendar
          month={calMonth}
          today={today}
          selected={value}
          onPick={(d) => pick(d)}
          onPrev={() => setCalMonth((m) => new Date(m.getFullYear(), m.getMonth() - 1, 1))}
          onNext={() => setCalMonth((m) => new Date(m.getFullYear(), m.getMonth() + 1, 1))}
        />
      )}
    </div>
  )
}
