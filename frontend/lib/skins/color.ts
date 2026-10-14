// PRD-259 §3 — a skin's colour values. Studio's tokens are HSL triplets ('30 12% 9%') read as
// hsl(var(--x)); a skin may also write '#rrggbb'. Both normalise to one triplet form here, and
// the backend's services/skins/color.py does the same sums in the same order.

export interface Hsl {
  readonly h: number
  readonly s: number
  readonly l: number
}

const HUE_MAX = 360
const PERCENT_MAX = 100
const CHANNEL_MAX = 255
const DECIMALS_SCALE = 100
const NUMBER = '([0-9]{1,3}(?:\\.[0-9]{1,2})?)'
const TRIPLET = new RegExp(`^${NUMBER} ${NUMBER}% ${NUMBER}%$`)
const HEX = /^#([0-9a-fA-F]{6})$/

/** Rounded to at most 2 decimals, half up (the backend rounds the same way). */
export function round2(value: number): number {
  return Math.round(value * DECIMALS_SCALE) / DECIMALS_SCALE
}

/** The one written form of a token value: 'H S% L%', at most 2 decimals each. */
export function formatTriplet(h: number, s: number, l: number): string {
  return `${String(round2(h))} ${String(round2(s))}% ${String(round2(l))}%`
}

function parseTriplet(value: string): Hsl | null {
  const match = TRIPLET.exec(value)
  if (!match) return null
  const [h, s, l] = [Number(match[1]), Number(match[2]), Number(match[3])]
  if (h > HUE_MAX || s > PERCENT_MAX || l > PERCENT_MAX) return null
  return { h, s, l }
}

function hueOf(r: number, g: number, b: number, max: number, delta: number): number {
  if (max === r) return ((g - b) / delta + (g < b ? 6 : 0)) * 60
  if (max === g) return ((b - r) / delta + 2) * 60
  return ((r - g) / delta + 4) * 60
}

function parseHex(value: string): Hsl | null {
  const digits = HEX.exec(value)?.[1]
  if (!digits) return null
  const channel = (at: number) => parseInt(digits.slice(at, at + 2), 16) / CHANNEL_MAX
  const [r, g, b] = [channel(0), channel(2), channel(4)]
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  if (max === min) return { h: 0, s: 0, l: round2(l * PERCENT_MAX) }
  const delta = max - min
  const s = l > 0.5 ? delta / (2 - max - min) : delta / (max + min)
  return { h: round2(hueOf(r, g, b, max, delta)), s: round2(s * PERCENT_MAX), l: round2(l * PERCENT_MAX) }
}

/** A token value as HSL, or null when it is neither form (BAD_VALUE). */
export function parseColor(value: unknown): Hsl | null {
  if (typeof value !== 'string') return null
  return parseTriplet(value) ?? parseHex(value)
}

/** A token value in its one written form, or null when it isn't a colour. */
export function normaliseColor(value: unknown): string | null {
  const hsl = parseColor(value)
  return hsl === null ? null : formatTriplet(hsl.h, hsl.s, hsl.l)
}

/** sRGB channels in 0..1 (CSS Color 4's hsl-to-rgb). */
export function hslToRgb({ h, s, l }: Hsl): readonly [number, number, number] {
  const sat = s / PERCENT_MAX
  const light = l / PERCENT_MAX
  const a = sat * Math.min(light, 1 - light)
  const f = (n: number) => {
    const k = (n + h / 30) % 12
    return light - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))
  }
  return [f(0), f(8), f(4)]
}
