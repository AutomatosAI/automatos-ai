/**
 * PRD-265 W2 — Stripe's card fields in our theme.
 *
 * Stripe's fields are iframes, so our CSS never reaches them; Stripe takes an `appearance`
 * object instead. It is built here, when a payment form mounts, from the theme tokens the
 * active theme (Studio, Play, HUD, light or dark) sets on <html>: the same `--primary`,
 * `--background`, `--foreground`, `--border`, `--destructive` and `--radius` the rest of the
 * page paints with. The tokens are HSL triplets ("16 100% 50%"); Stripe wants colour strings,
 * so each becomes a hex colour. Without a document (a server render, a test with no theme) or
 * for a token the theme does not set, the defaults below apply.
 */
import type { Appearance } from '@stripe/stripe-js'

type AppearanceVariables = NonNullable<Appearance['variables']>
type ColorVariable =
  | 'colorPrimary'
  | 'colorBackground'
  | 'colorText'
  | 'colorTextSecondary'
  | 'colorDanger'
  | 'colorSuccess'
  | 'colorWarning'
  | 'inputColorBorder'
  | 'inputFocusColorBorder'
  | 'labelColorText'

/** Each Stripe colour, the theme token it reads, and the colour when the theme sets none. */
const COLOR_TOKENS: ReadonlyArray<readonly [ColorVariable, string, string]> = [
  ['colorPrimary', '--primary', '#ff5500'],
  ['colorBackground', '--background', '#fafafa'],
  ['colorText', '--foreground', '#1a1a1a'],
  ['colorTextSecondary', '--muted-foreground', '#595959'],
  ['colorDanger', '--destructive', '#ef4444'],
  ['colorSuccess', '--success', '#10b981'],
  ['colorWarning', '--warning', '#f59e0b'],
  ['inputColorBorder', '--border', '#bfbfbf'],
  ['inputFocusColorBorder', '--ring', '#ff5500'],
  ['labelColorText', '--foreground', '#1a1a1a'],
]
const DEFAULT_RADIUS = '8px'
const DEFAULT_FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
const PX_PER_REM = 16
const DARK_BELOW_LIGHTNESS = 50

const HSL_TRIPLET = /^(-?\d*\.?\d+)(?:deg)?[\s,]+(\d*\.?\d+)%[\s,]+(\d*\.?\d+)%/
const COLOR_LITERAL = /^(#[0-9a-f]{3,8}|rgba?\(|hsla?\()/i
const LENGTH = /^(\d*\.?\d+)(px|rem)$/

export type TokenReader = (token: string) => string

/** "#ff5500" for "16 100% 50%"; a colour literal passes through; null for anything else. */
export function tokenToColor(raw: string): string | null {
  const value = raw.trim()
  if (COLOR_LITERAL.test(value)) return value
  const match = HSL_TRIPLET.exec(value)
  if (!match) return null
  return hslToHex(Number(match[1]), Number(match[2]), Number(match[3]))
}

function hslToHex(hue: number, saturation: number, lightness: number): string {
  const h = ((hue % 360) + 360) % 360
  const s = Math.min(saturation, 100) / 100
  const l = Math.min(lightness, 100) / 100
  const a = s * Math.min(l, 1 - l)
  const channel = (n: number) => {
    const k = (n + h / 30) % 12
    const value = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))
    return Math.round(value * 255).toString(16).padStart(2, '0')
  }
  return `#${channel(0)}${channel(8)}${channel(4)}`
}

/** "8px" for "0.5rem" or "8px"; the default for anything else. */
export function radiusOf(raw: string): string {
  const match = LENGTH.exec(raw.trim())
  if (!match) return DEFAULT_RADIUS
  const size = Number(match[1])
  return match[2] === 'rem' ? `${size * PX_PER_REM}px` : `${size}px`
}

/** The theme is dark when its background token is below half lightness. */
function isDarkBackground(raw: string): boolean {
  const match = HSL_TRIPLET.exec(raw.trim())
  return match !== null && Number(match[3]) < DARK_BELOW_LIGHTNESS
}

function documentReader(): TokenReader | null {
  if (typeof window === 'undefined' || typeof document === 'undefined') return null
  const style = window.getComputedStyle(document.documentElement)
  return (token) => style.getPropertyValue(token)
}

function documentFont(): string {
  if (typeof document === 'undefined' || !document.body) return DEFAULT_FONT
  return window.getComputedStyle(document.body).fontFamily || DEFAULT_FONT
}

/** The appearance for `read` (a theme token's value, '' when unset). */
export function appearanceFrom(read: TokenReader, fontFamily: string = DEFAULT_FONT): Appearance {
  const colors = Object.fromEntries(
    COLOR_TOKENS.map(([variable, token, fallback]) => [variable, tokenToColor(read(token)) ?? fallback]),
  ) as Pick<AppearanceVariables, ColorVariable>
  return {
    theme: isDarkBackground(read('--background')) ? 'night' : 'stripe',
    labels: 'above',
    variables: { ...colors, fontFamily, borderRadius: radiusOf(read('--radius')) },
  }
}

/** The appearance for the theme on the page now (the defaults without a document). */
export function stripeAppearance(): Appearance {
  const read = documentReader()
  return read ? appearanceFrom(read, documentFont()) : appearanceFrom(() => '')
}
