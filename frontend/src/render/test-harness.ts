// 実データでブラウザレンダを試すテストハーネス (render-test.html から読み込む)。
import { renderClip } from './compose'
import { buildSubtitleData, aspectToSize, type Project } from './fromProject'
import type { Edl, KeepRange } from './edl'

const $ = (id: string) => document.getElementById(id)!
const log = (m: string) => { const el = $('log'); el.textContent += m + '\n'; el.scrollTop = 1e9 }

const fileUrl = (sid: string, name: string) =>
  `/api/clips/file/${sid}/${encodeURIComponent(name)}`

// EDL を先頭 limitSec 秒 (post-cut) までに切り詰める (テストを速くするため)。
function truncateEdl(edl: Edl, limitSec: number): Edl {
  const ranges: KeepRange[] = []
  let acc = 0
  for (const r of edl.keep_ranges) {
    const len = r.end - r.start
    if (acc >= limitSec) break
    if (acc + len <= limitSec) { ranges.push(r); acc += len }
    else { ranges.push({ start: r.start, end: r.start + (limitSec - acc) }); acc = limitSec; break }
  }
  // kept_words は post-cut(new_start) が limit 未満のものだけ
  const kept_words = edl.kept_words.filter((w) => w.new_start < limitSec)
  return { keep_ranges: ranges, kept_words, params: edl.params }
}

async function run() {
  $('log').textContent = ''
  $('result').textContent = '処理中…'
  try {
    const sid = ($('sid') as HTMLInputElement).value.trim()
    const member = ($('member') as HTMLInputElement).value.trim()
    const limit = Number(($('limit') as HTMLInputElement).value) || 20

    log('WebCodecs: ' + ('VideoEncoder' in window ? 'あり' : 'なし') +
      ' / crossOriginIsolated: ' + window.crossOriginIsolated)
    log('実データ取得中…')
    const [project, edlRaw] = await Promise.all([
      fetch(fileUrl(sid, `${member}_project.json`)).then((r) => r.json() as Promise<Project>),
      fetch(fileUrl(sid, `${member}_edl.json`)).then((r) => r.json() as Promise<Edl>),
    ])
    log(`project: ${project.member} / member_start=${project.member_start} / aspect=${project.aspect}`)
    log(`EDL: keep_ranges=${edlRaw.keep_ranges.length}, kept_words=${edlRaw.kept_words.length}`)

    const edl = truncateEdl(edlRaw, limit)
    log(`先頭 ${limit}s に切り詰め: keep_ranges=${edl.keep_ranges.length}, kept_words=${edl.kept_words.length}`)

    const subtitle = buildSubtitleData(project, edl)
    log(`role_spans=${subtitle.roleSpans.length}, title="${subtitle.title.title}"`)

    const qp = new URLSearchParams(location.search)
    const base = aspectToSize(project.aspect)
    const width = Number(qp.get('w')) || base.width
    const height = Number(qp.get('h')) || base.height
    const quality = (qp.get('q') as 'low' | 'medium' | 'high') || 'medium'
    const srcUrl = fileUrl(sid, project.source_video)
    log(`レンダ開始: ${width}x${height} q=${quality}, source=${project.source_video}`)

    const res = await renderClip(srcUrl, {
      edl,
      memberStart: project.member_start,
      subtitle,
      width, height, fps: 30, quality,
      onProgress: (done, total, fps) => {
        $('result').textContent = `${done}/${total} frame … ${fps.toFixed(0)} fps`
      },
    })

    const url = URL.createObjectURL(res.blob)
    const v = $('player') as HTMLVideoElement
    v.src = url; v.style.display = 'block'
    const dl = $('download') as HTMLAnchorElement
    dl.href = url; dl.download = `${member}_browser.mp4`; dl.style.display = 'inline-block'

    const speedup = (res.frames / 30) / res.elapsedSec
    $('result').textContent =
      `✓ ${res.frames} frame を ${res.elapsedSec.toFixed(2)}秒 / ` +
      `${(res.frames / res.elapsedSec).toFixed(0)} fps / ${speedup.toFixed(1)}× realtime / ` +
      `${(res.blob.size / 1024 / 1024).toFixed(1)} MB`
    log('✓ 完了')
    ;(window as unknown as { __renderResult: string }).__renderResult = $('result').textContent
    document.title = 'RENDER_DONE'
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    $('result').textContent = '失敗: ' + msg
    log('✗ ' + msg)
    console.error(e)
    ;(window as unknown as { __renderResult: string }).__renderResult = 'ERROR:' + msg
    document.title = 'RENDER_ERR'
  }
}

$('run').addEventListener('click', run)
if (new URLSearchParams(location.search).get('auto')) run()
log('準備完了。「ブラウザでレンダ」を押してください。')
