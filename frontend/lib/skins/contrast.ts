// PRD-259 §3 — WCAG 2.x relative luminance and contrast ratio for skin tokens, ported from
// Play's lib/looks/contrast.ts (read-only reference). Play measures #rrggbb; Studio's tokens are
// HSL triplets, so a value goes HSL → sRGB first. The backend's contrast.py mirrors this.

import { hslToRgb, parseColor } from './color'

const LUMINANCE_WEIGHTS = [0.2126, 0.7152, 0.0722] as const
const FLARE = 0.05

function linear(channel: number): number {
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
}

/** Relative luminance of a token value ('H S% L%' or '#rrggbb'). Throws on anything else. */
export function relativeLuminance(value: string): number {
  const hsl = parseColor(value)
  if (hsl === null) throw new Error('not a skin colour value')
  const [r, g, b] = hslToRgb(hsl)
  return LUMINANCE_WEIGHTS[0] * linear(r) + LUMINANCE_WEIGHTS[1] * linear(g) + LUMINANCE_WEIGHTS[2] * linear(b)
}

export function contrastRatio(foreground: string, background: string): number {
  const a = relativeLuminance(foreground)
  const b = relativeLuminance(background)
  return (Math.max(a, b) + FLARE) / (Math.min(a, b) + FLARE)
}
