/**
 * PRD-261 M-01, PRD-271 C02 — a colour a phone reads: what a `mobile.tokens` value and a layout's colour slot take.
 * `orchestrator/services/themes/mobile_tokens.py` `parse_colour` is its twin. mobile-manifest.ts (the skin's
 * tokens) and mobile-layout.ts (a layout's colour and tint slots) share it.
 */

/** look-kit hsl.ts TRIPLET and contrast.ts HEX: what hexToTriplet writes, and #rrggbb. */
const TRIPLET_PATTERN = /^([0-9]{1,3}(?:\.[0-9]{1,2})?) ([0-9]{1,3}(?:\.[0-9]{1,2})?)% ([0-9]{1,3}(?:\.[0-9]{1,2})?)%$/
const HEX_PATTERN = /^#[0-9a-f]{6}$/i
const FULL_TURN = 360
const PERCENT = 100

/**
 * True when `value` is a colour a phone can read: a triplet look-kit's parseTriplet accepts (hue below
 * 360, saturation and lightness at most 100) or a #rrggbb (mobile_tokens.py parse_colour is its twin).
 */
export function isMobileColour(value: unknown): value is string {
  if (typeof value !== 'string') return false
  if (HEX_PATTERN.test(value)) return true
  const match = TRIPLET_PATTERN.exec(value)
  if (match === null) return false
  return Number(match[1]) < FULL_TURN && Number(match[2]) <= PERCENT && Number(match[3]) <= PERCENT
}
