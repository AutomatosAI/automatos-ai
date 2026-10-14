/**
 * PRD-260 — WCAG 2.2 relative luminance and contrast ratio, plus readableOn():
 * the colour to use when an accent or the error colour is shown as text. Pure;
 * colours are #rrggbb.
 */

/** WCAG AA for body text. */
export const AA_TEXT = 4.5
/** WCAG AA for a focus ring or other non-text mark against its ground. */
export const AA_NON_TEXT = 3
/** readableOn() aims a little above AA so rounding never lands a colour just under it. */
export const READABLE_TARGET = 4.8
/** readableOn() mixes toward the ground's text in steps of this many percent. */
export const MIX_STEP_PERCENT = 2

const HEX = /^#([0-9a-f]{6})$/i
const CHANNEL_MAX = 255

export type Rgb = readonly [number, number, number]

export function parseHex(hex: string): Rgb {
  const digits = HEX.exec(hex)?.[1]
  if (!digits) throw new Error(`${hex} isn't a #rrggbb colour`)
  const channel = (at: number) => parseInt(digits.slice(at, at + 2), 16)
  return [channel(0), channel(2), channel(4)]
}

/** #RRGGBB for whole channels 0-255. */
export function toHex(rgb: Rgb): string {
  return `#${rgb.map((value) => value.toString(16).padStart(2, '0').toUpperCase()).join('')}`
}

function linear(value: number): number {
  const srgb = value / CHANNEL_MAX
  return srgb <= 0.04045 ? srgb / 12.92 : ((srgb + 0.055) / 1.055) ** 2.4
}

export function relativeLuminance(hex: string): number {
  const [red, green, blue] = parseHex(hex)
  return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)
}

export function contrastRatio(foreground: string, background: string): number {
  const a = relativeLuminance(foreground)
  const b = relativeLuminance(background)
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)
}

/** `amount` 0 is `from`, 1 is `to`; mixed per sRGB channel and rounded. */
export function mixHex(from: string, to: string, amount: number): string {
  const [r1, g1, b1] = parseHex(from)
  const [r2, g2, b2] = parseHex(to)
  const mix = (a: number, b: number) => Math.round(a + (b - a) * amount)
  return toHex([mix(r1, r2), mix(g1, g2), mix(b1, b2)])
}

/** True when `colour` reads at READABLE_TARGET on both `bg` and `surface`. */
export function readsOn(colour: string, bg: string, surface: string): boolean {
  return contrastRatio(colour, bg) >= READABLE_TARGET && contrastRatio(colour, surface) >= READABLE_TARGET
}

/**
 * `colour` itself when it reads at READABLE_TARGET on both bg and surface;
 * otherwise `colour` mixed toward the ground's `text` in MIX_STEP_PERCENT steps
 * until it does (at worst, `text`).
 */
export function readableOn(colour: string, bg: string, surface: string, text: string): string {
  for (let percent = 0; percent < 100; percent += MIX_STEP_PERCENT) {
    const candidate = mixHex(colour, text, percent / 100)
    if (readsOn(candidate, bg, surface)) return candidate
  }
  return mixHex(text, text, 0) // normalised #RRGGBB
}
