/**
 * PRD-259 S-004a — the skin's stylesheet: written from validated numbers only, under the exact
 * selectors that outrank Studio's and Classic's own, with Studio's pre-wrapped aliases derived
 * from the skin's core tokens. Studio's own tokens as a skin (the S-001 corpus case) must
 * reproduce the aliases app/globals.css writes by hand.
 */
import { describe, expect, it } from 'vitest'

import {
  FG3_ALPHA,
  SKIN_CSS_GRAMMAR,
  SkinCssError,
  cssIsFor,
  skinSelectors,
  skinToCss,
} from '@/lib/skins/css'
import type { Skin } from '@/lib/skins/schema'
import { validateSkin } from '@/lib/skins/validate'

const ID = '0f8c2b6e-4d1a-4c3e-9b7a-2e5d6f8a1c90'
const OTHER_ID = '11111111-2222-4333-8444-555555555555'
const STUDIO_CASE = "Studio's own light and dark tokens, as a skin"
const FONT_FILE_CASE = 'a font file from the package'

interface CorpusCase {
  name: string
  skin: unknown
}

async function readRepoFile(relative: string): Promise<string> {
  const fs = await import('node:fs')
  const path = await import('node:path')
  return fs.readFileSync(path.resolve(__dirname, relative), 'utf8')
}

async function corpusSkin(name: string): Promise<Skin> {
  const corpus = JSON.parse(await readRepoFile('../../../../orchestrator/tests/fixtures/skins/cases.json')) as { cases: CorpusCase[] }
  const found = corpus.cases.find((testCase) => testCase.name === name)
  const result = validateSkin(found?.skin)
  if (!result.ok) throw new Error(`corpus case "${name}" no longer validates`)
  return result.skin
}

/** selector → { name → value } for every rule of a stylesheet. */
function parseRules(css: string): Map<string, Map<string, string>> {
  return new Map(css.split('\n').map((line) => {
    const open = line.indexOf('{')
    const body = line.slice(open + 1, -1)
    const declarations = body.split(';').map((declaration) => {
      const colon = declaration.indexOf(':')
      return [declaration.slice(0, colon), declaration.slice(colon + 1)] as const
    })
    return [line.slice(0, open), new Map(declarations)] as const
  }))
}

/** One custom property of the first `.studio {` / `.studio.dark {` block of app/globals.css. */
function globalsValue(globals: string, block: string, name: string): string | undefined {
  const start = globals.indexOf(`\n  ${block} {`)
  const text = globals.slice(start, globals.indexOf('}', start))
  return new RegExp(`--${name}:\\s*([^;]+);`).exec(text)?.[1].trim()
}

const STUDIO_ALIASES = ['paper', 'bg-1', 'bg-2', 'fg', 'fg-2', 'border-2', 'accent-2', 'navy'] as const

describe('skinToCss: selectors and specificity', () => {
  it('uses exactly these selectors', () => {
    expect(skinSelectors(ID)).toEqual({
      root: `html[data-skin="${ID}"]`,
      light: `html:not(.dark)[data-skin="${ID}"]`,
      dark: `html.dark[data-skin="${ID}"]`,
      body: `html[data-skin="${ID}"] body`,
    })
  })

  it("writes Studio-as-skin under them, in order: shape and fonts, light, dark, body's font variables", async () => {
    const css = skinToCss(await corpusSkin(STUDIO_CASE), ID)
    const selectors = skinSelectors(ID)
    expect([...parseRules(css).keys()]).toEqual([selectors.root, selectors.light, selectors.dark, selectors.body])
    expect(SKIN_CSS_GRAMMAR.test(css)).toBe(true)
    expect(cssIsFor(css, ID)).toBe(true)
    expect(cssIsFor(css, OTHER_ID)).toBe(false)
  })
})

describe('skinToCss: the golden Studio-as-skin', () => {
  it('writes every token as its triplet, the wrapped aliases wrapped, olive/warn-ink/on-strong raw', async () => {
    const skin = await corpusSkin(STUDIO_CASE)
    const rules = parseRules(skinToCss(skin, ID))
    const light = rules.get(skinSelectors(ID).light)
    const dark = rules.get(skinSelectors(ID).dark)
    expect(light?.get('--background')).toBe('38 50% 92%')
    expect(light?.get('--olive')).toBe('82 50% 22%')
    expect(light?.get('--warn-ink')).toBe('38 78% 27%')
    expect(light?.get('--on-strong')).toBe('0 0% 100%')
    expect(light?.get('--border-2')).toBe('hsl(40 30% 70%)')
    expect(light?.get('--navy')).toBe('hsl(213 51% 23%)')
    expect(light?.get('--accent-2')).toBe('hsl(15 76% 44% / 0.10)')
    expect(dark?.get('--accent-2')).toBe('hsl(15 80% 56% / 0.14)')
    expect(dark?.get('--on-strong')).toBe('30 12% 9%')
  })

  it("reproduces the pre-wrapped aliases app/globals.css writes for Studio and Studio Dark", async () => {
    const globals = await readRepoFile('../../../app/globals.css')
    const rules = parseRules(skinToCss(await corpusSkin(STUDIO_CASE), ID))
    const blocks = [['.studio', skinSelectors(ID).light], ['.studio.dark', skinSelectors(ID).dark]] as const
    const mismatches = blocks.flatMap(([block, selector]) => STUDIO_ALIASES.flatMap((name) => {
      const expected = globalsValue(globals, block, name)
      const got = rules.get(selector)?.get(`--${name}`)
      return expected !== undefined && got === expected ? [] : [{ block, name, expected, got }]
    }))
    expect(mismatches).toEqual([])
  })

  it('derives --fg-3 from muted-foreground at FG3_ALPHA', async () => {
    const rules = parseRules(skinToCss(await corpusSkin(STUDIO_CASE), ID))
    expect(FG3_ALPHA).toBe(0.75)
    expect(rules.get(skinSelectors(ID).light)?.get('--fg-3')).toBe('hsl(36 12% 36% / 0.75)')
    expect(rules.get(skinSelectors(ID).dark)?.get('--fg-3')).toBe('hsl(38 18% 62% / 0.75)')
  })

  it('writes radius, glow and the shipped fonts through next/font, re-pointed on body', async () => {
    const rules = parseRules(skinToCss(await corpusSkin(STUDIO_CASE), ID))
    const root = rules.get(skinSelectors(ID).root)
    const body = rules.get(skinSelectors(ID).body)
    expect(root?.get('--radius')).toBe('8px')
    expect(root?.get('--skin-glow')).toBe('0.00')
    expect(root?.get('--skin-font-serif')).toBe('var(--font-newsreader)')
    expect(body?.get('--font-geist-sans')).toBe('var(--skin-font-sans)')
    expect(body?.get('font-family')).toBe('var(--skin-font-sans), ui-sans-serif, system-ui, sans-serif')
    expect(body?.get('--serif')).toBe('var(--skin-font-serif), Charter, Georgia, serif')
  })
})

describe('skinToCss: derived aliases follow the skin', () => {
  it('emits no alias whose source the skin does not set', () => {
    const skin = { tokens: { light: { foreground: '0 0% 10%', background: '0 0% 98%' } } }
    const light = parseRules(skinToCss(skin, ID)).get(skinSelectors(ID).light)
    expect(light?.get('--fg')).toBe('hsl(0 0% 10%)')
    expect(light?.get('--bg-1')).toBe('hsl(0 0% 98%)')
    expect(['--paper', '--bg-2', '--fg-2', '--fg-3'].filter((name) => light?.has(name))).toEqual([])
  })

  it('maps a swap of shipped fonts without a var() cycle, and names a woff2 by its Studio family', async () => {
    const swap = parseRules(skinToCss({ tokens: {}, fonts: { sans: 'newsreader', serif: 'geist' } }, ID))
    const root = swap.get(skinSelectors(ID).root)
    expect(root?.get('--skin-font-sans')).toBe('var(--font-newsreader)')
    expect(root?.get('--skin-font-serif')).toBe('var(--font-geist-sans)')
    expect([...(root?.keys() ?? [])].filter((name) => name.startsWith('--font-'))).toEqual([])
    const file = parseRules(skinToCss(await corpusSkin(FONT_FILE_CASE), ID)).get(skinSelectors(ID).root)
    expect(file?.get('--skin-font-sans')).toBe(`"skin-${ID}-sans", var(--font-geist-sans)`)
  })
})

const ATTACK_VALUES = ['0 0% 0%; } body{display:none', 'url(https://evil)', '</style><script>']
const FORBIDDEN_IN_OUTPUT = ['url(', '<', '@', '\\']

describe('skinToCss: a skin that bypassed the validator', () => {
  it('never interpolates a raw string', async () => {
    const attacks: { skin: unknown; id: string }[] = [
      ...ATTACK_VALUES.map((value) => ({ skin: { tokens: { light: { background: value } } }, id: ID })),
      { skin: { tokens: { light: { 'background;}html{x': '0 0% 0%' } } }, id: ID },
      { skin: { tokens: {}, shape: { radius: '8px;}' } }, id: ID },
      { skin: { tokens: {}, fonts: { sans: 'x"; } body{' } }, id: ID },
      { skin: { tokens: { light: { background: '0 0% 0%' } } }, id: 'x"]{}' },
    ]
    attacks.forEach(({ skin, id }) => expect(() => skinToCss(skin, id)).toThrow(SkinCssError))

    const outputs = [skinToCss(await corpusSkin(STUDIO_CASE), ID), skinToCss(await corpusSkin(FONT_FILE_CASE), ID)]
    outputs.forEach((css) => {
      FORBIDDEN_IN_OUTPUT.forEach((bad) => expect(css).not.toContain(bad))
      const values = [...parseRules(css).values()].flatMap((rule) => [...rule.values()])
      expect(values.filter((value) => /[{};]/.test(value))).toEqual([])
    })
  })
})

describe('SKIN_CSS_GRAMMAR', () => {
  it('matches the golden output and refuses url(, @, \\, <, nested braces and foreign selectors', async () => {
    const golden = skinToCss(await corpusSkin(STUDIO_CASE), ID)
    expect(SKIN_CSS_GRAMMAR.test(golden)).toBe(true)
    const rule = (body: string) => `html[data-skin="${ID}"]{${body}}`
    const refused = [
      rule('--background:url(https://evil)'),
      rule('--background:URL(x)'),
      rule('--background:0 0% 0%;--x:@import'),
      rule('--background:0 0% 0%\\'),
      rule('--background:</style><script>'),
      rule('--background:{0}'),
      `${rule('--radius:8px')}}body{display:none}`,
      `body{--radius:8px}`,
      `html[data-skin="x"]{--radius:8px}`,
      rule('--paper:image-set("//evil.example/a.png")'),
      rule(`--paper:image-set("skin-${ID}-sans" 1x)`),
      '',
      ...ATTACK_VALUES.map((value) => rule(`--background:${value}`)),
    ]
    expect(refused.filter((css) => SKIN_CSS_GRAMMAR.test(css))).toEqual([])
  })
})
