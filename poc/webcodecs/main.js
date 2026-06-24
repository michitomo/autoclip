// autoclip WebCodecs レンダ PoC
// 検証したいこと:
//  1. ブラウザだけで decode → Canvas(字幕焼き) → encode → mux(mp4) が通るか
//  2. それが「ネイティブ並み速度」か (= ffmpeg.wasm の 12〜25倍遅 ではないか)
//  3. autoclip 風の字幕 (ミント文字+縁取り+話者バナー) が Canvas で再現できるか
//
// 入力は2モード:
//  A) 合成モード   … 外部ファイル不要。Canvas で動く背景を生成 → encode (encode 速度の純粋計測)
//  B) ファイルモード … ユーザーが選んだ mp4 を mediabunny で decode → 字幕焼き → re-encode

import {
  Input, Output, BufferTarget, Mp4OutputFormat, CanvasSource,
  ALL_FORMATS, BlobSource, VideoSampleSink, QUALITY_HIGH,
} from 'mediabunny'

const $ = (id) => document.getElementById(id)
const log = (msg) => { $('log').textContent += msg + '\n'; $('log').scrollTop = 1e9 }
const setStat = (k, v) => { $(k).textContent = v }

// 9:16 縦動画 (autoclip の既定アスペクト)。720x1280 で実用解像度。
const W = 720, H = 1280

// ---- autoclip 風の字幕を Canvas に描く (renderer の libass 焼き込みの代替) ----
// 実際の autoclip は ASS のレイアウト(位置・色・改行・カラオケ)を計算済みなので、
// ここではその「描画先が Canvas に変わる」部分だけを再現する。
function drawSubtitle(ctx, text, speaker, role, t) {
  // 話者バナー (上部・タイトルパネル相当)
  ctx.save()
  ctx.font = '600 30px system-ui, "Hiragino Sans", sans-serif'
  ctx.textAlign = 'center'
  const bannerY = 70
  ctx.fillStyle = 'rgba(20,25,33,0.78)'
  const bw = ctx.measureText(speaker).width + 56
  roundRect(ctx, W / 2 - bw / 2, bannerY - 34, bw, 50, 12); ctx.fill()
  ctx.fillStyle = role === '質疑者' ? '#5FE0B7' : '#F5A35F'
  ctx.fillText(speaker, W / 2, bannerY)
  ctx.restore()

  // 本文字幕 (下部・カラオケ風に現在語をハイライト)
  ctx.save()
  ctx.textAlign = 'center'
  ctx.font = '800 46px system-ui, "Hiragino Sans", sans-serif'
  const lineY = H - 220
  // 縁取り (libass の Outline 相当)
  ctx.lineWidth = 8
  ctx.strokeStyle = 'rgba(0,0,0,0.85)'
  ctx.lineJoin = 'round'
  // カラオケ: 経過に応じて文字数ぶんを話者色、残りを白
  const lit = Math.min(text.length, Math.floor((t % 3) / 3 * text.length) + 1)
  const litText = text.slice(0, lit), restText = text.slice(lit)
  const full = text
  const totalW = ctx.measureText(full).width
  let x = W / 2 - totalW / 2
  ctx.textAlign = 'left'
  // outline 全体
  ctx.strokeText(full, x, lineY)
  // lit 部分 (ミント)
  ctx.fillStyle = '#5FE0B7'
  ctx.fillText(litText, x, lineY)
  // rest 部分 (白)
  ctx.fillStyle = '#ffffff'
  ctx.fillText(restText, x + ctx.measureText(litText).width, lineY)
  ctx.restore()
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

// 背景: 合成モードで使う「動いている」テスト映像 (encode 負荷をかけるため毎フレーム変化)
function drawSyntheticBg(ctx, frame, fps) {
  const t = frame / fps
  const g = ctx.createLinearGradient(0, 0, 0, H)
  g.addColorStop(0, `hsl(${(t * 40) % 360}, 45%, 22%)`)
  g.addColorStop(1, `hsl(${(t * 40 + 60) % 360}, 45%, 12%)`)
  ctx.fillStyle = g
  ctx.fillRect(0, 0, W, H)
  // 動く円 (フレーム差分を作ってエンコーダに仕事をさせる)
  for (let i = 0; i < 8; i++) {
    const x = W / 2 + Math.cos(t * 1.3 + i) * (200 + i * 18)
    const y = H / 2 + Math.sin(t * 1.1 + i * 0.7) * (380 + i * 10)
    ctx.beginPath()
    ctx.arc(x, y, 40 + i * 6, 0, Math.PI * 2)
    ctx.fillStyle = `hsla(${(t * 80 + i * 40) % 360},70%,60%,0.5)`
    ctx.fill()
  }
}

const SAMPLE_SUB = [
  { text: 'こういったシステムの発注プロセスに', sp: '古川あおい', role: '質疑者' },
  { text: 'どのように関与しているのでしょうか', sp: '古川あおい', role: '質疑者' },
  { text: '関係省庁との連携について', sp: '古川あおい', role: '質疑者' },
  { text: 'お答えいたします', sp: '狭間保健局長', role: '答弁者' },
  { text: '制度改正が行われた際には', sp: '狭間保健局長', role: '答弁者' },
]

let busy = false

// ===== 合成モード: encode 速度の純粋計測 =====
async function runSynthetic(seconds, fps) {
  const canvas = new OffscreenCanvas(W, H)
  const ctx = canvas.getContext('2d')

  const target = new BufferTarget()
  const output = new Output({ format: new Mp4OutputFormat(), target })
  const source = new CanvasSource(canvas, {
    codec: 'avc',          // H264。VP9 にしたいなら 'vp9'
    bitrate: QUALITY_HIGH,
    keyFrameInterval: 2,
  })
  output.addVideoTrack(source, { frameRate: fps })
  await output.start()

  const total = Math.round(seconds * fps)
  const t0 = performance.now()
  let lastUi = t0
  for (let f = 0; f < total; f++) {
    drawSyntheticBg(ctx, f, fps)
    const sub = SAMPLE_SUB[Math.floor(f / fps / 1.0) % SAMPLE_SUB.length]
    drawSubtitle(ctx, sub.text, sub.sp, sub.role, f / fps)
    await source.add(f / fps, 1 / fps)   // backpressure を尊重して await
    if (performance.now() - lastUi > 100) {
      const done = f + 1
      setStat('progress', `${done}/${total} frame`)
      setStat('fps', (done / ((performance.now() - t0) / 1000)).toFixed(0))
      lastUi = performance.now()
    }
  }
  await output.finalize()
  const elapsed = (performance.now() - t0) / 1000
  return { buffer: target.buffer, frames: total, elapsed, srcSeconds: seconds }
}

// ===== ファイルモード: 実 mp4 を decode → 字幕焼き → re-encode =====
async function runFromFile(file, fps) {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
  const track = await input.getPrimaryVideoTrack()
  if (!track) throw new Error('動画トラックが見つかりません')
  const srcDuration = await track.computeDuration()
  log(`入力: ${track.displayWidth}x${track.displayHeight}, ${srcDuration.toFixed(1)}s, codec=${track.codec}`)

  // 出力は 9:16 にクロップ/スケールして字幕焼き
  const canvas = new OffscreenCanvas(W, H)
  const ctx = canvas.getContext('2d')
  const target = new BufferTarget()
  const output = new Output({ format: new Mp4OutputFormat(), target })
  const source = new CanvasSource(canvas, { codec: 'avc', bitrate: QUALITY_HIGH, keyFrameInterval: 2 })
  output.addVideoTrack(source, { frameRate: fps })
  await output.start()

  const sink = new VideoSampleSink(track)
  // PoC は冒頭 最大8秒だけ処理 (速度計測には十分)
  const limit = Math.min(srcDuration, 8)
  const total = Math.round(limit * fps)
  const t0 = performance.now()
  let f = 0, lastUi = t0
  for (let i = 0; i < total; i++) {
    const t = i / fps
    const sample = await sink.getSample(t)   // その時刻のフレームを decode
    if (!sample) break
    // 9:16 に cover で描画 (crop/scale 相当)
    drawCover(ctx, sample, W, H)
    sample.close()
    const sub = SAMPLE_SUB[Math.floor(t) % SAMPLE_SUB.length]
    drawSubtitle(ctx, sub.text, sub.sp, sub.role, t)
    await source.add(t, 1 / fps)
    f++
    if (performance.now() - lastUi > 100) {
      setStat('progress', `${f}/${total} frame`)
      setStat('fps', (f / ((performance.now() - t0) / 1000)).toFixed(0))
      lastUi = performance.now()
    }
  }
  await output.finalize()
  const elapsed = (performance.now() - t0) / 1000
  return { buffer: target.buffer, frames: f, elapsed, srcSeconds: limit }
}

// VideoSample を 9:16 canvas に cover 描画
function drawCover(ctx, sample, w, h) {
  const sw = sample.displayWidth, sh = sample.displayHeight
  const scale = Math.max(w / sw, h / sh)
  const dw = sw * scale, dh = sh * scale
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h)
  sample.draw(ctx, (w - dw) / 2, (h - dh) / 2, dw, dh)
}

function finish(res, modeLabel) {
  const blob = new Blob([res.buffer], { type: 'video/mp4' })
  const url = URL.createObjectURL(blob)
  $('player').src = url
  $('player').style.display = 'block'
  const dl = $('download'); dl.href = url; dl.style.display = 'inline-block'
  dl.download = `autoclip-poc-${modeLabel}.mp4`

  const speedup = res.srcSeconds / res.elapsed   // 実時間の何倍速で焼けたか
  setStat('fps', (res.frames / res.elapsed).toFixed(0))
  setStat('progress', `${res.frames} frame 完了`)
  setStat('result',
    `${res.frames} フレームを ${res.elapsed.toFixed(2)}秒 で書き出し / ` +
    `平均 ${(res.frames / res.elapsed).toFixed(0)} fps / ` +
    `映像 ${res.srcSeconds.toFixed(1)}秒ぶんを ${speedup.toFixed(1)}× リアルタイムで処理 / ` +
    `出力 ${(blob.size / 1024 / 1024).toFixed(1)} MB`)
  log(`✓ 完了: ${(res.frames / res.elapsed).toFixed(0)} fps, ${speedup.toFixed(1)}× realtime, ${(blob.size/1024/1024).toFixed(1)}MB`)
}

async function guard(fn) {
  if (busy) return
  busy = true
  $('log').textContent = ''
  setStat('result', '処理中…')
  try {
    log('WebCodecs 対応: ' + ('VideoEncoder' in window ? 'はい' : 'いいえ'))
    log('crossOriginIsolated: ' + window.crossOriginIsolated + ' (false でも WebCodecs は動く)')
    const r = await fn()
    finish(r.res, r.label)
  } catch (e) {
    log('✗ エラー: ' + (e && e.message ? e.message : e))
    setStat('result', '失敗: ' + (e && e.message ? e.message : e))
    console.error(e)
  } finally {
    busy = false
  }
}

$('runSynthetic').onclick = () => guard(async () => {
  const secs = Number($('secs').value) || 10
  const fps = Number($('fps').value) || 30
  log(`合成モード: ${secs}秒 @ ${fps}fps, ${W}x${H}, H264 を encode…`)
  const res = await runSynthetic(secs, fps)
  return { res, label: 'synthetic' }
})

$('file').onchange = (e) => {
  const file = e.target.files[0]
  if (!file) return
  guard(async () => {
    const fps = Number($('fps').value) || 30
    log(`ファイルモード: ${file.name} を decode → 9:16 字幕焼き → re-encode…`)
    const res = await runFromFile(file, fps)
    return { res, label: 'file' }
  })
}

log('準備完了。「合成モードで計測」を押すか、mp4 をドロップしてください。')

// ヘッドレス計測用: ?auto=1 で合成モードを自動実行し、結果を window へ書き出す
if (new URLSearchParams(location.search).get('auto')) {
  window.__pocResult = null
  const secs = Number(new URLSearchParams(location.search).get('secs')) || 10
  $('secs').value = secs
  guard(async () => {
    const fps = 30
    log(`[auto] 合成モード ${secs}s @ ${fps}fps`)
    const res = await runSynthetic(secs, fps)
    return { res, label: 'synthetic' }
  }).then(() => {
    // finish() 後に result テキストを window へ
    window.__pocResult = $('result').textContent
    document.title = 'POC_DONE'
  }).catch((e) => { window.__pocResult = 'ERROR:' + e; document.title = 'POC_ERR' })
}
