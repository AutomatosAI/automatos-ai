/**
 * 3 Oct 2026 — each channel its own shape, as the server picks it (orchestrator
 * modules/socials/channel_sizes.py): of a template's sizes, the one closest in shape to the
 * channel's aspect (aspectOf in editor-model.ts), compared by the log of width over height.
 * The render makes one file per size so chosen, and the preview shows each channel its own.
 */

const SHAPE = /^\s*(\d+(?:\.\d+)?)\s*[:x]\s*(\d+(?:\.\d+)?)\s*$/

/** "4:5", "300:157" or "1080x1350" as width over height; null when it is no shape. */
export function shapeRatio(value: string | null | undefined): number | null {
  const match = SHAPE.exec(value ?? '')
  if (!match) return null
  const width = Number(match[1])
  const height = Number(match[2])
  return width && height ? width / height : null
}

/** Of `candidates`, the one closest in shape to `aspect` (the first of a tie); null when none compares. */
export function closestShape<T>(candidates: ReadonlyArray<T>, shapeOf: (candidate: T) => string, aspect: string | null): T | null {
  const wanted = shapeRatio(aspect)
  let best: T | null = null
  let bestGap = Infinity
  for (const candidate of candidates) {
    const ratio = shapeRatio(shapeOf(candidate))
    if (wanted === null || ratio === null) continue
    const gap = Math.abs(Math.log(ratio / wanted))
    if (gap < bestGap) {
      best = candidate
      bestGap = gap
    }
  }
  return best
}

/** "1080x1350" → "4:5": a size's aspect in lowest terms, as the renderer names its files. */
export function sizeAspect(size: string): string | null {
  const [width, height] = size.split('x').map(Number)
  if (!width || !height) return null
  const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))
  const divisor = gcd(width, height)
  return `${width / divisor}:${height / divisor}`
}
