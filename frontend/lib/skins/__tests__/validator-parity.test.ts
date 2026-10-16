/**
 * PRD-259 S-001 — the skin validator, frontend side, against the ONE shared corpus
 * (../orchestrator/tests/fixtures/skins/cases.json). The backend's
 * test_prd259_skin_validator.py runs the same cases: same verdict, same codes.
 */
import { describe, expect, it } from 'vitest'

import { formatTriplet, normaliseColor } from '@/lib/skins/color'
import { contrastRatio } from '@/lib/skins/contrast'
import {
  ALLOWED_TOKENS,
  CONTRAST_PAIRS,
  DERIVED_ALIASES,
  MASCOT_STATES,
  SKIN_ERROR_CODES,
  SKIN_ID_PATTERN,
  SKIN_SCHEMA_ID,
} from '@/lib/skins/schema'
import { validateSkin } from '@/lib/skins/validate'

interface CorpusCase {
  name: string
  skin: unknown
  expect: { ok: boolean; codes: string[] }
}

interface Corpus {
  allowedTokens: string[]
  contrastPairs: { foreground: string; background: string; min: number }[]
  cases: CorpusCase[]
}

async function loadCorpus(): Promise<Corpus> {
  const fs = await import('node:fs')
  const path = await import('node:path')
  const file = path.resolve(__dirname, '../../../../orchestrator/tests/fixtures/skins/cases.json')
  return JSON.parse(fs.readFileSync(file, 'utf8')) as Corpus
}

function codesOf(result: ReturnType<typeof validateSkin>): string[] {
  return result.ok ? [] : [...new Set(result.errors.map((error) => error.code))].sort()
}

describe('skin validator parity with the backend', () => {
  it('gives every corpus case its verdict and codes', async () => {
    const corpus = await loadCorpus()
    expect(corpus.cases.length).toBeGreaterThanOrEqual(25)
    const mismatches = corpus.cases.flatMap((testCase) => {
      const result = validateSkin(JSON.parse(JSON.stringify(testCase.skin)))
      const got = { ok: result.ok, codes: codesOf(result) }
      return got.ok === testCase.expect.ok && JSON.stringify(got.codes) === JSON.stringify(testCase.expect.codes)
        ? []
        : [{ name: testCase.name, expected: testCase.expect, got }]
    })
    expect(mismatches).toEqual([])
  })

  it('holds the same token list and contrast pairs as the corpus', async () => {
    const corpus = await loadCorpus()
    expect([...ALLOWED_TOKENS]).toEqual(corpus.allowedTokens)
    expect(ALLOWED_TOKENS).toHaveLength(47)
    expect(CONTRAST_PAIRS.map((pair) => ({ ...pair }))).toEqual(corpus.contrastPairs)
  })

  it('covers every error code in the corpus', async () => {
    const corpus = await loadCorpus()
    const covered = new Set(corpus.cases.flatMap((testCase) => testCase.expect.codes))
    expect([...covered].sort()).toEqual([...SKIN_ERROR_CODES].sort())
  })
})

describe('skin colour values', () => {
  it('normalises hex and triplets to one written form, as the backend does', () => {
    expect(normaliseColor('#1a1814')).toBe('40 13.04% 9.02%')
    expect(normaliseColor('#FFFFFF')).toBe('0 0% 100%')
    expect(normaliseColor('38 50.0% 92%')).toBe('38 50% 92%')
    expect(normaliseColor('hsl(38 50% 92%)')).toBeNull()
    expect(formatTriplet(12.344, 50, 99.999)).toBe('12.34 50% 100%')
  })

  it('measures contrast from HSL as the 14 Oct measurements did', () => {
    expect(contrastRatio('38 50% 96%', '15 76% 44%').toFixed(2)).toBe('4.57')
    expect(contrastRatio('0 0% 100%', '0 0% 0%').toFixed(2)).toBe('21.00')
  })
})

describe('a validated skin', () => {
  it('comes back fresh, with its values normalised and the input untouched', () => {
    const input = {
      schema: SKIN_SCHEMA_ID, slug: 'ink', name: 'Ink', version: '1.0.0',
      base: 'classic', tones: ['light'], tokens: { light: { 'chart-1': '#c44a1a' } },
    }
    const result = validateSkin(input)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.skin.tokens.light?.['chart-1']).toBe(normaliseColor('#c44a1a'))
    expect(result.skin).not.toBe(input)
    expect(input.tokens.light['chart-1']).toBe('#c44a1a')
  })

  it('names both tokens, the tone and the ratio in a contrast error', async () => {
    const corpus = await loadCorpus()
    const testCase = corpus.cases.find((c) => c.name.startsWith("Classic's own primary"))
    const result = validateSkin(testCase?.skin)
    const message = result.ok ? '' : result.errors.find((error) => error.code === 'CONTRAST')?.message ?? ''
    expect(message).toContain('"primary-foreground"')
    expect(message).toContain('"primary"')
    expect(message).toContain('light')
    expect(message).toContain('3.31:1')
  })

  it('keeps the derived aliases, the mascot states and the id pattern of the PRD', () => {
    expect([...DERIVED_ALIASES]).toEqual(['paper', 'bg-1', 'bg-2', 'fg', 'fg-2', 'fg-3'])
    expect(MASCOT_STATES).toHaveLength(12)
    expect(SKIN_ID_PATTERN.test('1b4e28ba-2fa1-11d2-883f-0016d3cca427')).toBe(true)
    expect(SKIN_ID_PATTERN.test('../etc')).toBe(false)
  })
})
