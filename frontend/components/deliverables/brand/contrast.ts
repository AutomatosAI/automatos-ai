/**
 * PRD-255 US-007 — the WCAG contrast ratio, for the Brand kit page's badges. The server's
 * check on save (modules/documents/brand_system.py, core/brand_palette.py) is the one that
 * refuses a palette; this only shows the owner the same number while they choose.
 */
import type { BrandPalette, BrandPaletteRole } from '@/components/documents/blocks/types'

const HEX6 = /^#?([0-9a-f]{6})$/i
const HEX3 = /^#?([0-9a-f]{3})$/i
// WCAG 2.x relative luminance: the sRGB transfer function and the channel weights.
const SRGB_KNEE = 0.03928
const SRGB_LINEAR_DIVISOR = 12.92
const SRGB_OFFSET = 0.055
const SRGB_SCALE = 1.055
const SRGB_GAMMA = 2.4
const CHANNEL_WEIGHTS = [0.2126, 0.7152, 0.0722] as const
const LUMINANCE_FLARE = 0.05
const CHANNEL_MAX = 255
// WCAG AA, as the save check: text 4.5:1; large text, rules and fills 3:1.
export const TEXT_MIN_CONTRAST = 4.5
export const LARGE_TEXT_MIN_CONTRAST = 3
// A ratio is shown rounded down, so a near miss never reads as the target (the server's way).
const RATIO_SCALE = 10

/** ``hex`` (#rrggbb or #rgb) as [r, g, b] 0–255, or null when it is not a hex colour. */
export function parseHex(hex: string | undefined | null): [number, number, number] | null {
  const text = (hex ?? '').trim()
  const six = HEX6.exec(text)?.[1] ?? HEX3.exec(text)?.[1]?.replace(/./g, (c) => c + c)
  if (!six) return null
  return [0, 2, 4].map((i) => parseInt(six.slice(i, i + 2), 16)) as [number, number, number]
}

function linear(channel: number): number {
  const c = channel / CHANNEL_MAX
  return c <= SRGB_KNEE ? c / SRGB_LINEAR_DIVISOR : ((c + SRGB_OFFSET) / SRGB_SCALE) ** SRGB_GAMMA
}

/** The relative luminance of an [r, g, b] colour. */
export function luminance(rgb: [number, number, number]): number {
  return rgb.reduce((sum, channel, i) => sum + CHANNEL_WEIGHTS[i] * linear(channel), 0)
}

/** The contrast ratio of two hex colours (1 to 21), or null when either is not a hex colour. */
export function contrastRatio(foreground: string | undefined, background: string | undefined): number | null {
  const fg = parseHex(foreground)
  const bg = parseHex(background)
  if (!fg || !bg) return null
  const [light, dark] = [luminance(fg), luminance(bg)].sort((a, b) => b - a)
  return (light + LUMINANCE_FLARE) / (dark + LUMINANCE_FLARE)
}

/** ``ratio`` as the badge says it: "4.4:1", rounded down. */
export function formatRatio(ratio: number): string {
  return `${(Math.floor(ratio * RATIO_SCALE) / RATIO_SCALE).toFixed(1)}:1`
}

/** What a role's badge measures: which colour on which ground, and the least it needs (none: shown only). */
export interface RoleContrastSpec {
  foreground: BrandPaletteRole
  ground: BrandPaletteRole
  min?: number
}

// Text roles are read on the paper; a ground is read with the body text on it; a hairline is shown only.
export const ROLE_CONTRAST: Record<BrandPaletteRole, RoleContrastSpec> = {
  ink: { foreground: 'ink', ground: 'paper', min: TEXT_MIN_CONTRAST },
  heading: { foreground: 'heading', ground: 'paper', min: TEXT_MIN_CONTRAST },
  muted: { foreground: 'muted', ground: 'paper', min: TEXT_MIN_CONTRAST },
  accent: { foreground: 'accent', ground: 'paper', min: LARGE_TEXT_MIN_CONTRAST },
  accent_2: { foreground: 'accent_2', ground: 'paper', min: LARGE_TEXT_MIN_CONTRAST },
  paper: { foreground: 'ink', ground: 'paper', min: TEXT_MIN_CONTRAST },
  surface: { foreground: 'ink', ground: 'surface', min: TEXT_MIN_CONTRAST },
  surface_2: { foreground: 'ink', ground: 'surface_2', min: TEXT_MIN_CONTRAST },
  rule: { foreground: 'rule', ground: 'paper' },
}

export interface RoleContrast {
  ratio: number
  label: string
  /** True when it meets the least it needs; null when the role has no target. */
  passes: boolean | null
}

/** ``role``'s badge on ``palette``, or null when a colour it reads is missing. */
export function roleContrast(palette: BrandPalette, role: BrandPaletteRole): RoleContrast | null {
  const spec = ROLE_CONTRAST[role]
  const ratio = contrastRatio(palette[spec.foreground], palette[spec.ground])
  if (ratio === null) return null
  const on = spec.foreground === role ? `on ${spec.ground}` : `${spec.foreground} on it`
  return {
    ratio,
    label: `${formatRatio(ratio)} ${on}`,
    passes: spec.min === undefined ? null : ratio >= spec.min,
  }
}
