/**
 * PRD-260 D4, K3 — the pre-paint look: run as the browser runs it, in the
 * page, it puts the stored look's --pl-* variables AND Studio's tokens on
 * <html>, only under Play, and anything stored that isn't a catalogue look
 * gets the default look. Looks are kept per account (F5): it paints the
 * active look of the scope the pointer names, or with no pointer the unscoped
 * key a browser kept before.
 *
 * PRD-261 A-03: the look is painted by the one constant THEME_BOOT_SCRIPT from
 * builtin Play's boot data (lib/theme-host/builtin/play-boot.ts, the table from
 * lookPrepaintTable()); every case below runs against that pair with its
 * assertions unchanged. Under Studio and Classic the boot carries no table.
 */
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { GLOW_SCALE, rgba } from '@/lib/look-kit/css'
import { ACCENTS } from '@/lib/looks/catalogue'
import { lookDocumentVars } from '@/lib/looks/document'
import { GLOW_VAR, lookToCssVars, lookToStudioTokens, safeGlow } from '@/lib/looks/engine'
import { DAYLIGHT, DEFAULT_LOOK, EMBER, PRESETS } from '@/lib/looks/presets'
import { lookScope } from '@/lib/looks/scope'
import { ACTIVE_LOOK_KEY, LOOK_SCOPE_KEY, lookKeys } from '@/lib/theme-host/builtin/play-look-storage'
import { GLOW_MAX } from '@/lib/looks/types'
import { THEME_BOOT_SCRIPT, builtinThemeBoot } from '@/lib/theme-host/boot'
import { BUILTIN_PLAY_HTML_CLASS as PLAY_HTML_CLASS, BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { scriptSafeJson } from '@/lib/theme-host/script-safe-json'
import { STUDIO_HTML_CLASS } from '@/lib/ui-style'

import { cookieReader, runThemeBoot } from '@/lib/theme-host/__tests__/boot-harness'
import { clearRoot, lookNamesLeft, myLook, rootVars } from './helpers'

const LAYOUT = path.resolve(__dirname, '../../../app/layout.tsx')
const PLAY_BOOT = builtinThemeBoot(BUILTIN_PLAY_STYLE, cookieReader())

// Runs builtin Play's boot the way the browser does: its JSON element, then the constant
// script as a classic script in the page. Only in tests: the app itself never evaluates strings.
function runPrepaint(): void {
  runThemeBoot(PLAY_BOOT)
}

function store(value: unknown): void {
  window.localStorage.setItem(ACTIVE_LOOK_KEY, typeof value === 'string' ? value : JSON.stringify(value))
}

function underPlay(): void {
  document.documentElement.classList.add(STUDIO_HTML_CLASS, PLAY_HTML_CLASS)
}

beforeEach(() => {
  window.localStorage.clear()
  clearRoot()
  underPlay()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
  clearRoot()
})

describe('the pre-paint look script under Play', () => {
  it('sets exactly lookToCssVars() + lookToStudioTokens() for every preset and for saved looks', () => {
    const looks = [...PRESETS, myLook(), myLook({ glow: 0, font: 'bric', shell: 'silver', light: 'rose', ground: 'paper' })]
    for (const look of looks) {
      store(look)
      runPrepaint()
      const expected = { ...lookToCssVars(look), ...lookToStudioTokens(look) }
      expect({ id: look.id, vars: rootVars() }).toEqual({ id: look.id, vars: expected })
      clearRoot()
      underPlay()
    }
  })

  it("paints a stored glow past its ground and accent's safe glow at that cap, as lookToCssVars does (M50)", () => {
    for (const look of [myLook({ glow: GLOW_MAX }), { ...EMBER, glow: GLOW_MAX }, DAYLIGHT]) {
      const cap = safeGlow(look.ground, look.accent)
      store(look)
      runPrepaint()
      expect({ id: look.id, cap, under: cap < look.glow }).toEqual({ id: look.id, cap, under: true })
      expect(rootVars()[GLOW_VAR]).toBe(rgba(ACCENTS[look.accent].hex, cap / GLOW_SCALE))
      expect(rootVars()).toEqual(lookDocumentVars(look))
      clearRoot()
      underPlay()
    }
  })

  it("paints the active look of the account the pointer names, not another account's or the unscoped one (F5)", () => {
    const mine = lookScope('ws-1', 'user_a')
    window.localStorage.setItem(lookKeys(mine).active, JSON.stringify(DAYLIGHT))
    window.localStorage.setItem(lookKeys(lookScope('ws-1', 'user_b')).active, JSON.stringify(myLook()))
    store(EMBER)
    window.localStorage.setItem(LOOK_SCOPE_KEY, mine)
    runPrepaint()
    expect(rootVars()).toEqual(lookDocumentVars(DAYLIGHT))
  })

  it('applies the default look when the account the pointer names has no active look yet', () => {
    store(myLook())
    window.localStorage.setItem(LOOK_SCOPE_KEY, lookScope('ws-1', 'user_new'))
    runPrepaint()
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
  })

  it('applies the default look when nothing is stored', () => {
    runPrepaint()
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
  })

  it('applies the default look when the stored look is corrupt or names anything outside the catalogue', () => {
    const tampered = [
      '{not json',
      'null',
      '"daylight"',
      '[]',
      { ...EMBER, accent: '#FF0000' },
      { ...EMBER, accent: 'periwinkle', ground: '__proto__' },
      { ...EMBER, accent: 'mint', font: 'toString' },
      { ...EMBER, accent: 'gold', shell: 'bronze;background:url(x)' },
      { ...EMBER, accent: 'rose', glow: 41 },
      { ...EMBER, accent: 'rose', glow: -1 },
      { ...EMBER, accent: 'rose', glow: 2.5 },
      { ...EMBER, accent: 'rose', glow: '24' },
      { ...EMBER, accent: 'rose', schema: 'automatos-interface-look/v0' },
      { ...EMBER, accent: 'rose', interface: 'studio' },
      JSON.stringify({ ...EMBER, accent: 'rose' }).replace('"accent":"rose"', '"__proto__":{"accent":"rose"}'),
    ]
    for (const value of tampered) {
      store(value)
      runPrepaint()
      expect({ value, vars: rootVars() }).toEqual({ value, vars: lookDocumentVars(DEFAULT_LOOK) })
      clearRoot()
      underPlay()
    }
  })

  it('applies the default look, without throwing, when localStorage refuses to be read', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('The operation is insecure.', 'SecurityError')
    })
    store(myLook())
    expect(runPrepaint).not.toThrow()
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
  })
})

describe('the pre-paint look script outside Play', () => {
  it("sets nothing on <html> under Studio's and Classic's boot", () => {
    store(myLook())
    for (const style of ['studio', 'classic'] as const) {
      clearRoot()
      runThemeBoot(builtinThemeBoot(style, cookieReader()))
      expect({ style, left: lookNamesLeft() }).toEqual({ style, left: [] })
      expect(document.documentElement.style.length).toBe(0)
    }
  })
})

describe('the pre-paint look script as markup', () => {
  it("is safe inside a <script> element: the constant script, and Play's boot as its JSON element body", () => {
    for (const body of [THEME_BOOT_SCRIPT, scriptSafeJson(PLAY_BOOT)]) {
      expect(body).not.toMatch(/<\/?script/i)
      expect(body).not.toContain('<!--')
      expect(body).not.toContain('</')
      expect(body).not.toMatch(/[\u2028\u2029]/)
    }
  })

  it('escapes what would end a script element or a line inside it', () => {
    const json = scriptSafeJson({ text: '</script><!-- \u2028 \u2029' })
    expect(json).not.toMatch(/[<\u2028\u2029]/)
    expect(JSON.parse(json)).toEqual({ text: '</script><!-- \u2028 \u2029' })
  })

  it("is rendered in the <head> of the root layout, as the style's boot (Play's carries the look)", () => {
    const layout = readFileSync(LAYOUT, 'utf8')
    const tag = '<ThemeBootScripts boot={boot} />'
    expect(layout).toContain(tag)
    expect(layout).not.toContain('LOOK_PREPAINT_SCRIPT')
    expect(layout.indexOf('<head>')).toBeLessThan(layout.indexOf(tag))
    expect(layout.indexOf(tag)).toBeLessThan(layout.indexOf('</head>'))
    expect(layout.indexOf('</head>')).toBeLessThan(layout.indexOf('<body>'))
  })
})
