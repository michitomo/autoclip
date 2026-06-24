// 未生成クリップの「生成リクエスト」を GitHub Issue のプリフィルURLで作る。
// Issue を clip-request ワークフローが拾って targets に追加→生成→catalog 反映する。
// (誰でもリクエスト可。生成済みなら ワークフロー側で skip。)

// リポは build 時の env で差し替え可能 (既定は本リポ)。
const REPO = (import.meta.env.VITE_REPO as string) || 'michitomo/autoclip'

export type IssueClipRequest = {
  sessionId: string
  member: string
  committee: string
  date: string
}

/**
 * Issue 新規作成のプリフィルURL。本文に機械可読な YAML ブロックを埋め、
 * clip-request ワークフローがそこから session_id / member を読む。
 */
export function clipRequestIssueUrl(req: IssueClipRequest): string {
  const body = [
    `衆議院TV クリップの生成リクエストです。`,
    ``,
    `- 日付: ${req.date}`,
    `- 委員会: ${req.committee}`,
    `- 議員: ${req.member}`,
    ``,
    '```yaml',
    `session_id: "${req.sessionId}"`,
    `member: ${req.member}`,
    '```',
    ``,
    `<!-- このYAMLブロックを clip-request ワークフローが読みます。編集不要です。 -->`,
  ].join('\n')
  const q = new URLSearchParams({
    title: `[clip] ${req.committee} / ${req.member}`,
    body,
    labels: 'clip-request',
  })
  return `https://github.com/${REPO}/issues/new?${q.toString()}`
}

export { REPO }
