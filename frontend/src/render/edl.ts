// EDL の時間計算 (backend src/video/jetcut.py + renderer.py の移植)
//
// 時間軸は2つ。混ぜないこと:
//  - member-WAV 時間: 議員区間先頭=0 のソース線形時間。keep_ranges / kept_words.old_* がこれ。
//  - post-cut 時間: keep_ranges を前詰め連結した後の時間。kept_words.new_* / captions がこれ。
// ソース動画(mp4)の絶対時間 = member-WAV 時間 + member_start。

export type KeepRange = { start: number; end: number } // member-WAV 秒
export type KeptWord = {
  word: string
  old_start: number // member-WAV
  old_end: number
  new_start: number // post-cut
  new_end: number
}

/** _edl.json の中身 (backend が書き出す)。 */
export type Edl = {
  keep_ranges: KeepRange[]
  kept_words: KeptWord[]
  params: Record<string, unknown>
}

/**
 * member-WAV 時刻 t を post-cut 時刻へ写像する (jetcut._remap_words の map_time と同一)。
 * t が属する keep_range を見つけ、それ以前の区間の累積長 + 区間内オフセット。
 * 区間外は最近接にクランプ。
 */
export function memberToPostCut(t: number, ranges: KeepRange[]): number {
  if (ranges.length === 0) return 0
  let acc = 0
  for (const r of ranges) {
    if (r.start <= t && t <= r.end) return acc + (t - r.start)
    acc += r.end - r.start
  }
  // 区間外クランプ
  if (t < ranges[0].start) return 0
  acc = 0
  for (const r of ranges) {
    if (t < r.start) return acc // 直前区間と次区間の境目 → 次区間先頭
    acc += r.end - r.start
  }
  return acc // 末尾より後 → 全長
}

/** post-cut 全長 (= 連結後クリップの尺, 秒)。 */
export function postCutDuration(ranges: KeepRange[]): number {
  let acc = 0
  for (const r of ranges) acc += r.end - r.start
  return acc
}

/**
 * post-cut 時刻 t が、ソース動画(mp4)の絶対秒のどこに当たるかを返す。
 * レンダ時に「今 post-cut の時刻 t を描く → ソースの何秒目をデコードすべきか」に使う。
 *  source_sec = (member-WAV 時刻) + member_start
 */
export function postCutToSource(
  t: number,
  ranges: KeepRange[],
  memberStart: number,
): number {
  let acc = 0
  for (const r of ranges) {
    const len = r.end - r.start
    if (t <= acc + len) return r.start + (t - acc) + memberStart
    acc += len
  }
  // 末尾クランプ
  const last = ranges[ranges.length - 1]
  return last.end + memberStart
}

/**
 * keep_ranges をソース動画の絶対秒の区間列に変換 (デコード範囲の決定用)。
 * 返り値の各区間 [srcStart, srcEnd] と、その区間が始まる post-cut 時刻 outStart。
 */
export function sourceSegments(
  ranges: KeepRange[],
  memberStart: number,
): { srcStart: number; srcEnd: number; outStart: number }[] {
  const out: { srcStart: number; srcEnd: number; outStart: number }[] = []
  let acc = 0
  for (const r of ranges) {
    out.push({
      srcStart: r.start + memberStart,
      srcEnd: r.end + memberStart,
      outStart: acc,
    })
    acc += r.end - r.start
  }
  return out
}
