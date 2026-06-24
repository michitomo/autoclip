import { useCallback, useEffect, useRef, useState } from 'react'

export type ToastKind = 'success' | 'error' | 'info'
export type ToastMsg = { id: number; kind: ToastKind; text: string }

/**
 * 軽量トースト。グローバル状態は持たず、使う側が useToast() で push し、
 * 返ってくる toasts を <ToastStack> に渡して右下に表示する。
 * 成功は数秒で自動消滅、エラーは長め。クリックで消せる。
 */
export function useToast() {
  const [toasts, setToasts] = useState<ToastMsg[]>([])
  const idRef = useRef(0)
  const timers = useRef<Record<number, number>>({})

  const dismiss = useCallback((id: number) => {
    setToasts((ts) => ts.filter((t) => t.id !== id))
    const tm = timers.current[id]
    if (tm) {
      window.clearTimeout(tm)
      delete timers.current[id]
    }
  }, [])

  const push = useCallback(
    (kind: ToastKind, text: string) => {
      const id = ++idRef.current
      setToasts((ts) => [...ts, { id, kind, text }])
      const ttl = kind === 'error' ? 6000 : 3500
      timers.current[id] = window.setTimeout(() => dismiss(id), ttl)
      return id
    },
    [dismiss],
  )

  // アンマウント時にタイマーを掃除する。
  useEffect(() => {
    const t = timers.current
    return () => {
      Object.values(t).forEach((tm) => window.clearTimeout(tm))
    }
  }, [])

  return { toasts, push, dismiss }
}

const ICON: Record<ToastKind, string> = { success: '✓', error: '⚠', info: 'ℹ' }

export function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: ToastMsg[]
  onDismiss: (id: number) => void
}) {
  if (toasts.length === 0) return null
  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {toasts.map((t) => (
        <button key={t.id} className={`toast toast-${t.kind}`} onClick={() => onDismiss(t.id)}>
          <span className="toast-ic">{ICON[t.kind]}</span>
          <span className="toast-text">{t.text}</span>
        </button>
      ))}
    </div>
  )
}
