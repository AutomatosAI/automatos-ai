import { parseHex, toHex, type Rgb } from './contrast'

/**
 * PRD-260 — #rrggbb to and from Studio's HSL triplet 'H S% L%' (no hsl()
 * wrapper: tailwind.config.ts reads every token as hsl(var(--x))). Every number
 * goes through formatNumber: at most two decimals, no trailing zeros, so a
 * triplet converts back to the same #rrggbb.
 */

export interface Hsl {
  /** Degrees, 0 up to (not including) 360. */
  readonly h: number
  /** Percent, 0-100. */
  readonly s: number
  /** Percent, 0-100. */
  readonly l: number
}

const CHANNEL_MAX = 255
const PERCENT = 100
const FULL_TURN = 360
const SEXTANT = 60
const DECIMALS_SCALE = 100
const TRIPLET = /^(\d{1,3}(?:\.\d{1,2})?) (\d{1,3}(?:\.\d{1,2})?)% (\d{1,3}(?:\.\d{1,2})?)%$/

/** At most two decimals and no trailing zeros: 20.089 → '20.09', 100 → '100'. */
export function formatNumber(value: number): string {
  return String(Math.round(value * DECIMALS_SCALE) / DECIMALS_SCALE)
}

function hueOf(red: number, green: number, blue: number, max: number, delta: number): number {
  if (delta === 0) return 0
  if (max === red) return SEXTANT * (((green - blue) / delta + 6) % 6)
  if (max === green) return SEXTANT * ((blue - red) / delta + 2)
  return SEXTANT * ((red - green) / delta + 4)
}

export function hexToHsl(hex: string): Hsl {
  const [red, green, blue] = parseHex(hex).map((value) => value / CHANNEL_MAX)
  const max = Math.max(red, green, blue)
  const min = Math.min(red, green, blue)
  const delta = max - min
  const lightness = (max + min) / 2
  const saturation = delta === 0 ? 0 : delta / (1 - Math.abs(2 * lightness - 1))
  return { h: hueOf(red, green, blue, max, delta), s: saturation * PERCENT, l: lightness * PERCENT }
}

/** 'H S% L%', the hue wrapped so rounding never prints 360. */
export function formatTriplet({ h, s, l }: Hsl): string {
  const hue = Number(formatNumber(h)) % FULL_TURN
  return `${formatNumber(hue)} ${formatNumber(s)}% ${formatNumber(l)}%`
}

export function hexToTriplet(hex: string): string {
  return formatTriplet(hexToHsl(hex))
}

export function parseTriplet(triplet: string): Hsl {
  const match = TRIPLET.exec(triplet)
  if (!match) throw new Error(`${triplet} isn't an 'H S% L%' triplet`)
  const hsl = { h: Number(match[1]), s: Number(match[2]), l: Number(match[3]) }
  if (hsl.h >= FULL_TURN || hsl.s > PERCENT || hsl.l > PERCENT) throw new Error(`${triplet} is out of range`)
  return hsl
}

export function hslToHex({ h, s, l }: Hsl): string {
  const saturation = s / PERCENT
  const lightness = l / PERCENT
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation
  const channel = (offset: number) => {
    const k = (offset + h / (FULL_TURN / 12)) % 12
    const value = lightness - chroma / 2 * Math.max(-1, Math.min(k - 3, 9 - k, 1))
    return Math.round(value * CHANNEL_MAX)
  }
  const rgb: Rgb = [channel(0), channel(8), channel(4)]
  return toHex(rgb)
}

/** A triplet back to #RRGGBB (the contrast tests measure Studio's tokens this way). */
export function tripletToHex(triplet: string): string {
  return hslToHex(parseTriplet(triplet))
}
