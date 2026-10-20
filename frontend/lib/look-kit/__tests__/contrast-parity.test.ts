// @vitest-environment node
/**
 * PRD-261 M-03 — the review's contrast maths is look-kit's. The server measures a theme's mobile colour
 * pairs with orchestrator/services/themes/mobile_contrast.py (a port of contrastRatio) over the hexes
 * mobile_tokens.py parse_colour reads (a port of tripletToHex); every case of the shared corpus's
 * `contrast` section (orchestrator/tests/fixtures/themes/cases.json, which
 * orchestrator/tests/test_prd261_mobile_contrast.py runs in Python) gives the same hexes and ratios here,
 * through look-kit itself. With M-01's mobile-tokens.test.ts, this is the core user that keeps
 * contrast.ts and hsl.ts in core (QB-19 = keep, PB-38).
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { AA_NON_TEXT, AA_TEXT, contrastRatio, parseHex, toHex } from '@/lib/look-kit/contrast'
import { tripletToHex } from '@/lib/look-kit/hsl'

const REPO_ROOT = path.resolve(__dirname, '../../../..')
const CORPUS_PATH = path.join(REPO_ROOT, 'orchestrator/tests/fixtures/themes/cases.json')
/** The corpus writes each ratio to two decimals. */
const RATIO_DIGITS = 2
const HEX = /^#[0-9a-f]{6}$/i

interface ContrastCase {
  readonly name: string
  /** A mobile token value: an 'H S% L%' triplet or #rrggbb. */
  readonly foreground: string
  readonly background: string
  readonly foregroundHex: string
  readonly backgroundHex: string
  readonly ratio: number
}

interface Corpus {
  readonly contrast: { readonly cases: readonly ContrastCase[] }
}

const CASES = (JSON.parse(readFileSync(CORPUS_PATH, 'utf8')) as Corpus).contrast.cases

/** A token value as #RRGGBB, read the way look-kit reads it: a hex through contrast.ts, a triplet through hsl.ts. */
function hexOf(value: string): string {
  return HEX.test(value) ? toHex(parseHex(value)) : tripletToHex(value)
}

describe("PRD-261 M-03 — the server's contrast ratios are look-kit's", () => {
  it('has cases on both sides of each AA minimum to measure', () => {
    const ratios = CASES.map((item) => item.ratio)
    expect(ratios.some((ratio) => ratio >= AA_TEXT)).toBe(true)
    expect(ratios.some((ratio) => ratio >= AA_NON_TEXT && ratio < AA_TEXT)).toBe(true)
    expect(ratios.some((ratio) => ratio < AA_NON_TEXT)).toBe(true)
  })

  it.each(CASES)('$name: tripletToHex or parseHex gives the hexes the case names', (item) => {
    expect(hexOf(item.foreground)).toBe(item.foregroundHex)
    expect(hexOf(item.background)).toBe(item.backgroundHex)
  })

  it.each(CASES)('$name: contrastRatio gives the case ratio to two decimals, either way round', (item) => {
    const foreground = hexOf(item.foreground)
    const background = hexOf(item.background)
    expect(contrastRatio(foreground, background)).toBeCloseTo(item.ratio, RATIO_DIGITS)
    expect(contrastRatio(background, foreground)).toBeCloseTo(item.ratio, RATIO_DIGITS)
  })
})
