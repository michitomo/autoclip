import { useEffect, useRef } from 'react'
import type { EditCaption } from './api'

function fmt(t: number): string {
  const m = Math.floor(t / 60)
  const s = (t % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

/**
 * 字幕編集モーダル。動画の字幕領域クリックで開く。
 * 現在の字幕 (focusIndex) を中央に、前後数件を上下に並べて編集する。
 * 行の ▶ はその字幕区間をループ再生 (耳で確認しながら修正)。
 * 現在字幕の判定は時刻ベースなので、テキストを直してもフォーカスはずれない。
 */
export function CaptionModal({
  captions,
  focusIndex,
  context = 3,
  onEdit,
  onLoopPlay,
  onClose,
}: {
  captions: EditCaption[]
  focusIndex: number
  context?: number // 前後に何件表示するか
  onEdit: (index: number, text: string) => void
  onLoopPlay: (index: number) => void
  onClose: () => void
}) {
  const focusRef = useRef<HTMLTextAreaElement>(null)

  // 開いたら現在字幕の入力にフォーカス。Esc で閉じる。
  useEffect(() => {
    focusRef.current?.focus()
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const lo = Math.max(0, focusIndex - context)
  const hi = Math.min(captions.length - 1, focusIndex + context)
  const rows: number[] = []
  for (let i = lo; i <= hi; i++) rows.push(i)

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="字幕を修正"
      >
        <div className="modal-head">
          <h3>字幕を修正</h3>
          <button className="modal-close" onClick={onClose} title="閉じる (Esc)">
            ✕
          </button>
        </div>
        <p className="qa-hint muted">
          ▶ で区間をループ再生（耳で確認）。修正は書き出し後の動画に反映されます。
        </p>
        <div className="modal-rows">
          {rows.map((i) => {
            const c = captions[i]
            const isFocus = i === focusIndex
            return (
              <div
                key={i}
                className={`modal-row ${isFocus ? 'focus' : ''} ${c.edited ? 'edited' : ''}`}
              >
                <button
                  className="modal-play"
                  onClick={() => onLoopPlay(i)}
                  title="この区間をループ再生"
                >
                  ▶
                </button>
                <span className="modal-time">{fmt(c.start)}</span>
                <textarea
                  ref={isFocus ? focusRef : undefined}
                  className="modal-text"
                  value={c.text}
                  rows={2}
                  onChange={(e) => onEdit(i, e.target.value)}
                />
              </div>
            )
          })}
        </div>
        <div className="modal-actions">
          <button className="primary" onClick={onClose}>
            完了
          </button>
        </div>
      </div>
    </div>
  )
}
