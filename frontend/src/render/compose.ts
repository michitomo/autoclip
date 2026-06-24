// ブラウザレンダの本体 (backend renderer.py 相当)。
// EDL の keep_ranges に従ってソース動画から必要区間だけ decode し、9:16 にして
// 字幕を焼き、WebCodecs で encode → mediabunny で mp4 に mux する。

import {
  Input, Output, BufferTarget, Mp4OutputFormat, CanvasSource,
  ALL_FORMATS, BlobSource, UrlSource, VideoSampleSink,
  QUALITY_LOW, QUALITY_MEDIUM, QUALITY_HIGH,
  AudioSampleSink, AudioSampleSource, getFirstEncodableAudioCodec,
  type Quality,
} from 'mediabunny'
import { type Edl, sourceSegments, postCutDuration } from './edl'
import { drawSubtitleFrame, type SubtitleData, type Ctx2D } from './subtitle'

// 画質プリセット。ソースが 640x360 と低解像度なので medium で十分。低いほどエンコードが
// 速い (本パイプラインはエンコード律速 = 画質を下げると書き出しが速くなる)。
const QUALITY: Record<'low' | 'medium' | 'high', Quality> = {
  low: QUALITY_LOW,
  medium: QUALITY_MEDIUM,
  high: QUALITY_HIGH,
}

export type RenderOptions = {
  edl: Edl
  memberStart: number
  subtitle: SubtitleData
  width?: number // 既定 720
  height?: number // 既定 1280 (9:16)
  fps?: number // 既定 30
  quality?: 'low' | 'medium' | 'high' // 既定 medium (速度優先・360pソース相応)
  onProgress?: (done: number, total: number, fps: number) => void
  // フレーム書き出し後の段階 (音声・mp4仕上げ) を UI に伝える。無言の待ちを無くす用。
  onStage?: (stage: 'audio' | 'finalize') => void
}

export type RenderResult = {
  blob: Blob
  frames: number
  elapsedSec: number
}

/** ソースを Input にする: File(Blob) か URL(string)。 */
function makeInput(source: File | string): Input {
  const src = typeof source === 'string' ? new UrlSource(source) : new BlobSource(source)
  return new Input({ source: src, formats: ALL_FORMATS })
}

// VideoSample を 9:16 canvas に cover 描画 (crop/scale 相当)
function drawCover(
  ctx: Ctx2D,
  sample: { displayWidth: number; displayHeight: number; draw: (c: Ctx2D, x: number, y: number, w: number, h: number) => void },
  W: number,
  H: number,
) {
  const sw = sample.displayWidth, sh = sample.displayHeight
  const scale = Math.max(W / sw, H / sh)
  const dw = sw * scale, dh = sh * scale
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, W, H)
  sample.draw(ctx, (W - dw) / 2, (H - dh) / 2, dw, dh)
}

/**
 * ブラウザ内で 1 クリップをレンダする。
 * source: ソース動画 (バックエンドが配信する mp4 の URL、またはユーザー選択 File)。
 */
export async function renderClip(
  source: File | string,
  opts: RenderOptions,
): Promise<RenderResult> {
  const W = opts.width ?? 720
  const H = opts.height ?? 1280
  const fps = opts.fps ?? 30
  const vQuality = QUALITY[opts.quality ?? 'medium']
  const { edl, memberStart, subtitle } = opts

  const input = makeInput(source)
  const track = await input.getPrimaryVideoTrack()
  if (!track) throw new Error('動画トラックが見つかりません')
  const sink = new VideoSampleSink(track)

  // 音声トラック (あれば trim して mux する)
  const audioTrack = await input.getPrimaryAudioTrack()

  // 出力 (mp4 / H264 + AAC)
  const canvas = new OffscreenCanvas(W, H)
  const ctx = canvas.getContext('2d')!
  // 注: 映像ループ後の finalize に ~650ms(20s尺) かかるが、これは mp4 の moov を最後に
  // 構築する本質的コスト。StreamTarget / fastStart=false / 'fragmented' を試したが
  // いずれも縮まなかった (検証済み)。互換性の高い BufferTarget + 既定 fastStart を採る。
  const target = new BufferTarget()
  const output = new Output({ format: new Mp4OutputFormat(), target })
  const encSource = new CanvasSource(canvas, {
    codec: 'avc',
    bitrate: vQuality,
    keyFrameInterval: 2,
  })
  output.addVideoTrack(encSource, { frameRate: fps })

  // 音声出力ソース (エンコード可能なコーデックを選ぶ。通常 aac)
  let audioSource: AudioSampleSource | null = null
  if (audioTrack) {
    const codec = await getFirstEncodableAudioCodec(['aac', 'opus'])
    if (codec) {
      audioSource = new AudioSampleSource({ codec, bitrate: QUALITY_HIGH })
      output.addAudioTrack(audioSource)
    }
  }

  await output.start()

  const segs = sourceSegments(edl.keep_ranges, memberStart)
  const totalDur = postCutDuration(edl.keep_ranges)
  const totalFrames = Math.round(totalDur * fps)
  const t0 = performance.now()
  let frame = 0
  let lastUi = t0

  // ステージ別の所要時間 (ボトルネック計測用)
  const prof = { decode: 0, draw: 0, encode: 0 }

  // パイプライン化: encSource.add() は「現在の canvas を即座にフレーム化」し、返す
  // Promise は backpressure 用 (= 次を受け取れるまで)。これを即 await せず保持して、
  // 次フレームの decode/draw を encode と並行させる。canvas は add() 同期完了後に
  // 上書きしてよい (フレームは内部で確保済み)。backpressure は MAX_INFLIGHT で抑える。
  const MAX_INFLIGHT = 2
  const inflight: Promise<void>[] = []

  for (const seg of segs) {
    let s = performance.now()
    for await (const sample of sink.samples(seg.srcStart, seg.srcEnd)) {
      prof.decode += performance.now() - s
      // sink.samples は srcStart 直前のキーフレームから返すため、seg 開始より前の
      // サンプルが混じる。これを入れると outT が負→0クランプで先頭に潰れ、トピック
      // 途中開始のクリップで「冒頭ズレ/ゴミフレーム」になる。範囲外は捨てる。
      if (sample.timestamp < seg.srcStart - 1e-3) { sample.close(); s = performance.now(); continue }
      const outT = seg.outStart + (sample.timestamp - seg.srcStart)
      const dur = sample.duration || 1 / fps
      s = performance.now()
      drawCover(ctx, sample, W, H)
      sample.close()
      drawSubtitleFrame(ctx, W, H, outT, subtitle)
      prof.draw += performance.now() - s
      // add() は canvas を同期キャプチャ。Promise は保持して並行させる。
      s = performance.now()
      const p = encSource.add(Math.max(0, outT), dur)
      inflight.push(p)
      // backpressure: 一定数を超えたら最も古いものを待つ
      if (inflight.length >= MAX_INFLIGHT) {
        await inflight.shift()
      }
      prof.encode += performance.now() - s
      frame++
      const now = performance.now()
      if (opts.onProgress && now - lastUi > 100) {
        opts.onProgress(frame, totalFrames, frame / ((now - t0) / 1000))
        lastUi = now
      }
      s = performance.now()
    }
  }
  // 映像ループ後の各段階を計測 (「最後まで走った後の待ち」の正体を切り分ける)
  const tail = { videoFlush: 0, audio: 0, finalize: 0 }
  let ts = performance.now()
  // 残りの encode 完了を待つ
  await Promise.all(inflight)
  tail.videoFlush = performance.now() - ts
  // 計測結果を window に出す (デバッグ用)
  ;(globalThis as { __renderProf?: typeof prof }).__renderProf = prof

  // --- 音声: keep_range ごとに trim し、post-cut タイムラインへ前詰めして mux ---
  ts = performance.now()
  if (audioTrack && audioSource) {
    opts.onStage?.('audio')
    const aSink = new AudioSampleSink(audioTrack)
    for (const seg of segs) {
      // この区間の音声サンプルを順に取り、開始を outStart に揃える
      for await (const sample of aSink.samples(seg.srcStart, seg.srcEnd)) {
        // 映像と同様、seg 開始より前のサンプル (キーフレーム手前) は捨てる。
        // 残すと音声が前詰めされて A/V がズレる。
        if (sample.timestamp < seg.srcStart - 1e-3) { sample.close(); continue }
        // サンプルの src 時刻 → post-cut 時刻へシフト
        const shifted = seg.outStart + (sample.timestamp - seg.srcStart)
        sample.setTimestamp(Math.max(0, shifted))
        await audioSource.add(sample)
        sample.close()
      }
    }
  }
  tail.audio = performance.now() - ts

  ts = performance.now()
  opts.onStage?.('finalize')
  // finalize は同期的に走り UI 更新を挟めないので、1フレーム描画を待ってから実行する。
  await new Promise((r) => setTimeout(r, 0))
  await output.finalize()
  const blob = new Blob([target.buffer!], { type: 'video/mp4' })
  tail.finalize = performance.now() - ts
  ;(globalThis as { __renderTail?: typeof tail }).__renderTail = tail
  const elapsedSec = (performance.now() - t0) / 1000
  opts.onProgress?.(frame, totalFrames, frame / elapsedSec)
  return { blob, frames: frame, elapsedSec }
}
