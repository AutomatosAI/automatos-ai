/**
 * PRD-260 — WCAG 2.2 relative luminance and contrast ratio, plus readableOn():
 * the colour to use when an accent or the error colour is shown as text, on a
 * ground's bg and surface or (readableOnAll, readableOnTinted) on any list of
 * fills. Pure; colours are #rrggbb.
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

/** True when `colour` reads at READABLE_TARGET on every one of `grounds`. */
export function readsOnAll(colour: string, grounds: readonly string[]): boolean {
  return grounds.every((ground) => contrastRatio(colour, ground) >= READABLE_TARGET)
}

/**
 * `colour` itself when `reads` accepts it; otherwise `colour` mixed toward the
 * ground's `text` in MIX_STEP_PERCENT steps until it is accepted (at worst, `text`).
 */
function readableWhere(colour: string, text: string, reads: (candidate: string) => boolean): string {
  for (let percent = 0; percent < 100; percent += MIX_STEP_PERCENT) {
    const candidate = mixHex(colour, text, percent / 100)
    if (reads(candidate)) return candidate
  }
  return mixHex(text, text, 0) // normalised #RRGGBB
}

/** `colour`, or the nearest mix toward `text`, that reads at READABLE_TARGET on every one of `grounds`. */
export function readableOnAll(colour: string, grounds: readonly string[], text: string): string {
  return readableWhere(colour, text, (candidate) => readsOnAll(candidate, grounds))
}

/**
 * readableOnAll() that also reads on each ground tinted by the colour itself at
 * `maxTint` alpha: text in a colour on a chip filled with the same colour (a
 * `bg-primary/20 text-primary` badge). Checked flat and at the strongest tint;
 * contrast.test.ts holds the lighter tints Studio uses as well.
 */
export function readableOnTinted(colour: string, grounds: readonly string[], text: string, maxTint: number): string {
  return readableWhere(colour, text, (candidate) =>
    readsOnAll(candidate, [...grounds, ...grounds.map((ground) => mixHex(ground, candidate, maxTint))]),
  )
}

/**
 * `colour` itself when it reads at READABLE_TARGET on both bg and surface;
 * otherwise `colour` mixed toward the ground's `text` in MIX_STEP_PERCENT steps
 * until it does (at worst, `text`).
 */
export function readableOn(colour: string, bg: string, surface: string, text: string): string {
  return readableOnAll(colour, [bg, surface], text)
}
