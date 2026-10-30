// @vitest-environment node
/**
 * PRD-268 US-C21a (K13, §6.12.2) — every status colour is a real token that reads on its ground.
 *
 * --running is a token in all four tone blocks of app/globals.css (Classic draws working in its own
 * --info until it gains running), and the Studio pip reads it. lookToStudioTokens sets the five
 * STUDIO_STATUS_TOKEN_NAMES for every ground, from Studio Light's or Studio Dark's values by the
 * ground's tone (isDarkGround), read here from globals.css itself. Measured on Studio light and dark
 * and on every ground of a made-up catalogue spanning both tones (no theme's own): each token drawn
 * as a status word (warn-ink, success, destructive, info, running) is 4.5:1 on the background and on
 * the card, and each drawn as a mark is 3:1. needs_you's mark is --warning on a dark ground only: on
 * a light one --warning (sand) is about 1.6:1, so the glyph and the word take --warn-ink there.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { cssRules } from '@/lib/__tests__/css-cascade'
import { AA_NON_TEXT, AA_TEXT, contrastRatio } from '@/lib/look-kit/contrast'
import { isDarkGround } from '@/lib/look-kit/css'
import { tripletToHex } from '@/lib/look-kit/hsl'
import { STUDIO_STATUS_TOKEN_NAMES, lookToStudioTokens } from '@/lib/look-kit/studio-tokens'
import type { LookCatalogue } from '@/lib/look-kit/types'

const FRONTEND = path.resolve(__dirname, '../../..')
const GLOBALS = readFileSync(path.join(FRONTEND, 'app/globals.css'), 'utf8')

/** The rules each Studio tone's <html> matches, lowest precedence first (as dark-tone-contrast.test.ts reads them). */
const STUDIO_LIGHT: readonly string[] = [':root', '.studio']
const STUDIO_DARK: readonly string[] = [':root', '.dark', '.studio', '.studio.dark']
const STUDIO_TONES = [
  { name: 'Studio light', selectors: STUDIO_LIGHT, dark: false },
  { name: 'Studio dark', selectors: STUDIO_DARK, dark: true },
]

/** The status tokens drawn as a word: 4.5:1 on the background and the card. */
const TEXT_TOKENS: readonly string[] = ['--warn-ink', '--success', '--destructive', '--info', '--running']
/** needs_you's mark on a dark ground. */
const DARK_MARK = '--warning'
const PIP_RUNNING = '.studio .studio-pip.running'
const RUNNING = '--running'
const INFO = '--info'

/** A made-up catalogue, two accents over light and dark grounds, cool and warm. */
const LAB: LookCatalogue = {
  prefix: 'lab',
  errorColour: '#D9442E',
  glow: { min: 0, max: 20 },
  grounds: {
    night: {
      bg: '#101418', surface: '#1A2028', text: '#F0F4F8', text2: '#A8B4C0',
      line: 'rgba(240,244,248,.12)', chip: 'rgba(240,244,248,.06)', panel: 'rgba(240,244,248,.03)',
    },
    dusk: {
      bg: '#1C2030', surface: '#262B3D', text: '#EEF0F8', text2: '#A6ADC2',
      line: 'rgba(238,240,248,.14)', chip: 'rgba(238,240,248,.07)', panel: 'rgba(238,240,248,.03)',
    },
    moss: {
      bg: '#141A14', surface: '#1E261E', text: '#EEF2EA', text2: '#A9B4A4',
      line: 'rgba(238,242,234,.13)', chip: 'rgba(238,242,234,.06)', panel: 'rgba(238,242,234,.03)',
    },
    day: {
      bg: '#FAFAF7', surface: '#FFFFFF', text: '#14181C', text2: '#5A6470',
      line: 'rgba(20,24,28,.14)', chip: 'rgba(20,24,28,.06)', panel: 'rgba(255,255,255,.5)',
    },
    cream: {
      bg: '#EFE8DA', surface: '#FBF8F2', text: '#1F1A14', text2: '#5E5548',
      line: 'rgba(31,26,20,.14)', chip: 'rgba(31,26,20,.06)', panel: 'rgba(255,255,255,.5)',
    },
  },
  accents: { teal: { hex: '#2EC4B6', ink: '#04201D' }, iris: { hex: '#6E56CF', ink: '#FFFFFF' } },
  typeStyles: {
    plain: {
      family: 'var(--font-geist-sans),sans-serif', style: 'normal', weight: 600, letterSpacing: '0',
      sizes: { hero: 56, brand: 18, status: 30, card: 17 },
    },
  },
  shells: { steel: { hex: '#8A96A3', filter: 'none' } },
  lights: { warm: { hex: '#FFB347' } },
}
const GROUNDS = Object.keys(LAB.grounds).map((ground) => ({ ground, dark: isDarkGround(LAB, ground) }))
const ACCENTS = Object.keys(LAB.accents)

type Tokens = Readonly<Record<string, string>>

/** A rule body's declarations as [property, value]. */
function pairsOf(body: string): [string, string][] {
  return body.split(';').flatMap((part): [string, string][] => {
    const colon = part.indexOf(':')
    return colon > 0 ? [[part.slice(0, colon).trim(), part.slice(colon + 1).trim()]] : []
  })
}

/** The custom properties the rules with exactly these selectors declare, a later rule winning. */
function declared(selectors: readonly string[]): Tokens {
  const rules = cssRules(GLOBALS)
  const bodies = selectors.flatMap((selector) => rules.filter((rule) => rule.selectors.includes(selector)).map((rule) => rule.body))
  return Object.fromEntries(bodies.flatMap(pairsOf).filter(([name]) => name.startsWith('--')))
}

/** The first rule with exactly this selector, its declarations property to value. */
function declarations(selector: string): Tokens {
  const rule = cssRules(GLOBALS).find((candidate) => candidate.selectors.includes(selector))
  return Object.fromEntries(pairsOf(rule?.body ?? ''))
}

/** A triplet token as #RRGGBB. */
function token(tokens: Tokens, name: string): string {
  const value = tokens[name]
  if (value === undefined) throw new Error(`${name} is not set`)
  return tripletToHex(value)
}

/** The status tokens drawn as a mark on this tone: the words' tokens, and --warning on a dark ground. */
function markTokens(dark: boolean): readonly string[] {
  return dark ? [...TEXT_TOKENS, DARK_MARK] : TEXT_TOKENS
}

/** Each token under its minimum on the background or the card, as 'name on fill: ratio'. */
function shortfalls(tokens: Tokens, dark: boolean): string[] {
  const fills = ['--background', '--card'].map((fill): [string, string] => [fill, token(tokens, fill)])
  const measure = (names: readonly string[], minimum: number) =>
    names.flatMap((name) => fills
      .map(([fill, hex]) => [fill, contrastRatio(token(tokens, name), hex)] as const)
      .filter(([, ratio]) => ratio < minimum)
      .map(([fill, ratio]) => `${name} on ${fill}: ${ratio.toFixed(2)}`))
  return [...measure(TEXT_TOKENS, AA_TEXT), ...measure(markTokens(dark), AA_NON_TEXT)]
}

/** Studio's own status values for a tone, from globals.css. */
function studioStatus(dark: boolean): Tokens {
  const tokens = declared(dark ? STUDIO_DARK : STUDIO_LIGHT)
  return Object.fromEntries(STUDIO_STATUS_TOKEN_NAMES.map((name) => [name, tokens[name]]))
}

describe('PRD-268 US-C21a — --running is a real token', () => {
  it('is set in Studio light and dark, and in Classic light and dark as its own --info', () => {
    expect(declared(['.studio'])[RUNNING]).toBe('40 20% 36%')
    expect(declared(['.studio.dark'])[RUNNING]).toBe('40 20% 65%')
    for (const block of [':root', '.dark']) {
      const tokens = declared([block])
      expect(tokens[RUNNING]).toBeDefined()
      expect(tokens[RUNNING]).toBe(tokens[INFO])
    }
  })

  it("is what the Studio pip draws working in, never a literal", () => {
    expect(declarations(PIP_RUNNING).background).toBe('hsl(var(--running))')
  })

  it('is one of the five status tokens every look sets', () => {
    expect([...STUDIO_STATUS_TOKEN_NAMES].sort()).toEqual(['--info', '--running', '--success', '--warn-ink', '--warning'])
  })
})

describe('PRD-268 US-C21a — the status tokens read on Studio light and dark', () => {
  it.each(STUDIO_TONES)('$name: each word is 4.5:1 and each mark 3:1 on the background and the card', ({ selectors, dark }) => {
    expect(shortfalls(declared(selectors), dark)).toEqual([])
  })
})

describe("PRD-268 US-C21a — lookToStudioTokens sets the status tokens by the ground's tone", () => {
  it('measures grounds of both tones', () => {
    expect(GROUNDS.some(({ dark }) => dark)).toBe(true)
    expect(GROUNDS.some(({ dark }) => !dark)).toBe(true)
  })

  it.each(GROUNDS)("the $ground ground takes Studio's status values for its tone, whatever the accent", ({ ground, dark }) => {
    const expected = studioStatus(dark)
    for (const accent of ACCENTS) {
      const tokens = lookToStudioTokens(LAB, { ground, accent })
      STUDIO_STATUS_TOKEN_NAMES.forEach((name) => expect(tokens[name]).toBe(expected[name]))
    }
  })

  it.each(GROUNDS)('the $ground ground: each word is 4.5:1 and each mark 3:1 on its background and card', ({ ground, dark }) => {
    for (const accent of ACCENTS) {
      expect(shortfalls(lookToStudioTokens(LAB, { ground, accent }), dark)).toEqual([])
    }
  })
})
