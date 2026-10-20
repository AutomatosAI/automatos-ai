/**
 * PRD-261 A-03 — first paint from data: the boot of every style, the gate it passes (bad classes
 * and bad prepaint values dropped, never written), its escape inside a <script> element, and
 * the one constant script that applies it.
 *
 * PRD-261 W-A6b (gap G2): an installed theme's boot names its own look keys (a '/' in the theme
 * id), which pass the gate, so the script paints the stored look; a 'from-look' installed boot
 * starts on the tone cookie, as builtin Play's does.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import {
  THEME_BOOT_ELEMENT_ID,
  THEME_BOOT_SCRIPT,
  baseThemeBoot,
  builtinThemeBoot,
  createThemeBootResolver,
  parseThemeTone,
  type ThemeBoot,
} from '@/lib/theme-host/boot'
import { CSS_VAR_NAME_PATTERN, isCssValue, sanitizeThemeBoot } from '@/lib/theme-host/boot-sanitize'
import { BUILTIN_PLAY_BOOT } from '@/lib/theme-host/builtin/play-boot'
import { BUILTIN_PLAY_HTML_CLASS, BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { bootLookStorage, installedLookKeys } from '@/lib/theme-host/builtin/play-look-storage'
import { HTML_CLASS_PATTERN } from '@/lib/theme-host/constants'
import { scriptSafeJson } from '@/lib/theme-host/script-safe-json'
import { STUDIO_HTML_CLASS, parseUiStyle, type UiStyle } from '@/lib/ui-style'

import { cookieReader, runThemeBoot, runThemeBootScriptAlone } from './boot-harness'
import { HELLO_PACKAGE_ENTRY, PLAY_PACKAGE_ENTRY } from './installed-entry-fixture'

const INSTALL_STYLE: UiStyle = 'theme:install:0b7e4c1a-9f3d-4c2e-8a51-3d2f6b7a9c10'
const NO_COOKIES = cookieReader()
const SAMPLE_VAR = '--sample-var'
const html = () => document.documentElement

afterEach(() => {
  vi.restoreAllMocks()
  html().removeAttribute('style')
  html().className = ''
})

/** A boot with a one-knob prepaint table, to probe the gate and the script. */
function sampleBoot(fields: Partial<ThemeBoot> = {}): ThemeBoot {
  return {
    ...baseThemeBoot('studio'),
    prepaint: { knobs: { tint: { warm: { [SAMPLE_VAR]: '#FF6A1F' } } }, pairs: [], numeric: [], defaults: { tint: 'warm' } },
    ...fields,
  }
}

describe("PRD-261 A-03 — each style's boot", () => {
  it('gives Classic no class, Studio its class, and removes every other style class', () => {
    expect(baseThemeBoot('classic')).toMatchObject({ addClasses: [], removeClasses: [STUDIO_HTML_CLASS, BUILTIN_PLAY_HTML_CLASS] })
    expect(baseThemeBoot('studio')).toMatchObject({ addClasses: [STUDIO_HTML_CLASS], removeClasses: [BUILTIN_PLAY_HTML_CLASS] })
    for (const style of ['classic', 'studio'] as const) {
      expect(builtinThemeBoot(style, NO_COOKIES)).toEqual({
        ...baseThemeBoot(style),
        tone: { policy: 'user', value: null },
        prepaint: null,
        lookStorage: null,
        lookGuard: null,
        integrity: null,
      })
    }
  })

  it("gives builtin Play 'studio play', its look table, storage keys and guard — the legacy cookie too", () => {
    for (const cookie of ['play', BUILTIN_PLAY_STYLE]) {
      const boot = builtinThemeBoot(parseUiStyle(cookie), NO_COOKIES)
      expect(boot.addClasses).toEqual([STUDIO_HTML_CLASS, BUILTIN_PLAY_HTML_CLASS])
      expect(boot.removeClasses).toEqual([])
      expect(boot.prepaint).toEqual(BUILTIN_PLAY_BOOT.prepaint)
      expect(boot.lookStorage).toEqual(BUILTIN_PLAY_BOOT.lookStorage)
      expect(boot.lookGuard).toEqual(BUILTIN_PLAY_BOOT.lookGuard)
      expect(boot.tone).toEqual({ policy: 'from-look', value: null })
    }
  })

  it("gives an installed theme its base's classes (Studio) and nothing to paint, until its resolver answers", async () => {
    const expected = { addClasses: [STUDIO_HTML_CLASS], removeClasses: [BUILTIN_PLAY_HTML_CLASS], prepaint: null, integrity: null }
    expect(builtinThemeBoot(INSTALL_STYLE, NO_COOKIES)).toMatchObject(expected)
    expect(await createThemeBootResolver(NO_COOKIES)(INSTALL_STYLE)).toMatchObject(expected)
  })

  it("reads the look's tone cookie only under Play, strictly (PRD-260 K4)", () => {
    const light = cookieReader({ [PLAY_TONE_COOKIE]: 'light' })
    expect(builtinThemeBoot(BUILTIN_PLAY_STYLE, light).tone).toEqual({ policy: 'from-look', value: 'light' })
    expect(builtinThemeBoot(BUILTIN_PLAY_STYLE, cookieReader({ [PLAY_TONE_COOKIE]: 'bogus' })).tone.value).toBeNull()
    expect(builtinThemeBoot('studio', light).tone.value).toBeNull()
    expect(builtinThemeBoot('classic', cookieReader({ [PLAY_TONE_COOKIE]: 'dark' })).tone.value).toBeNull()
    for (const value of ['Dark', 'system', '', ' dark', 'toString', null, undefined]) {
      expect({ value, tone: parseThemeTone(value) }).toEqual({ value, tone: null })
    }
  })

  it("asks the installed resolver for an installed style only, gates its boot, and falls back when it fails", async () => {
    const installed = vi.fn(async () => sampleBoot({ addClasses: ['studio', 'hello-theme', 'Not A Class'] }))
    const resolve = createThemeBootResolver(NO_COOKIES, installed)
    expect((await resolve(INSTALL_STYLE)).addClasses).toEqual(['studio', 'hello-theme'])
    expect(await resolve('studio')).toEqual(builtinThemeBoot('studio', NO_COOKIES))
    expect(installed).toHaveBeenCalledTimes(1)
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const failing = createThemeBootResolver(NO_COOKIES, async () => Promise.reject(new Error('timeout')))
    expect(await failing(INSTALL_STYLE)).toEqual(baseThemeBoot(INSTALL_STYLE))
    expect(console.error).toHaveBeenCalled()
  })
})

describe('PRD-261 A-03 — the gate before a boot is written', () => {
  it('drops a class that fails the theme-class pattern', () => {
    const bad = ['Play', 'pl', 'a'.repeat(33), '</script>', 'x y', '-dash', 'ok-class', 7]
    const boot = sanitizeThemeBoot(sampleBoot({ addClasses: bad as string[], removeClasses: ['ok-class', 'studio'] }))
    expect(boot.addClasses).toEqual(['ok-class'])
    expect(boot.removeClasses).toEqual(['studio'])
  })

  it('drops a prepaint value that fails the CSS-value grammar, and a bad property name', () => {
    const hostile = ['url(https://x.test/a.png)', '</script><!--', 'red;background:url(x)', '"quoted"', 'expression(alert(1))', 'a(b', '!important', '']
    for (const value of hostile) expect({ value, ok: isCssValue(value) }).toEqual({ value, ok: false })
    const fine = ['#FF6A1F', 'rgba(244,238,232,.12)', 'hsl(12 5% 7% / 0.14)', '12 5% 7%', 'var(--pl-font-fredoka),sans-serif', '-.03em', 'hue-rotate(18deg) saturate(1.2)']
    for (const value of fine) expect({ value, ok: isCssValue(value) }).toEqual({ value, ok: true })
    const vars = { [SAMPLE_VAR]: '#FF6A1F', '--x-bad': 'url(x)', '--Upper': '#000000', 'color': '#000000' }
    const boot = sanitizeThemeBoot(sampleBoot({ prepaint: { knobs: { tint: { warm: vars } }, pairs: [], numeric: [], defaults: { tint: 'warm' } } }))
    expect(boot.prepaint?.knobs.tint.warm).toEqual({ [SAMPLE_VAR]: '#FF6A1F' })
  })

  it('drops the whole table when its default look no longer fits it', () => {
    const boot = sanitizeThemeBoot(sampleBoot({ prepaint: { knobs: { tint: { warm: {} } }, pairs: [], numeric: [], defaults: { tint: 'cold' } } }))
    expect(boot.prepaint).toBeNull()
  })

  it("keeps builtin Play's whole table: every value passes the grammar", () => {
    expect(builtinThemeBoot(BUILTIN_PLAY_STYLE, NO_COOKIES).prepaint).toEqual(BUILTIN_PLAY_BOOT.prepaint)
  })

  it("escapes a '</script><!--' value so the element body cannot end early, and parses back", () => {
    const boot = { ...sampleBoot(), lookGuard: { schema: '</script><!--', iface: '\u2028\u2029' } }
    const body = scriptSafeJson(boot)
    expect(body).not.toMatch(/[<\u2028\u2029]/)
    expect(JSON.parse(body)).toEqual(boot)
  })
})

describe('PRD-261 A-03 — the one constant boot script', () => {
  it('is a literal holding the element id and the same patterns the gate uses', () => {
    expect(THEME_BOOT_SCRIPT).toContain(JSON.stringify(THEME_BOOT_ELEMENT_ID))
    expect(THEME_BOOT_SCRIPT).toContain(`C=/${HTML_CLASS_PATTERN.source}/`)
    expect(THEME_BOOT_SCRIPT).toContain(`V=/${CSS_VAR_NAME_PATTERN.source}/`)
  })

  it('is safe inside a <script> element', () => {
    expect(THEME_BOOT_SCRIPT).not.toMatch(/<\/?script/i)
    expect(THEME_BOOT_SCRIPT).not.toContain('<!--')
    expect(THEME_BOOT_SCRIPT).not.toContain('</')
    expect(THEME_BOOT_SCRIPT).not.toMatch(/[\u2028\u2029]/)
  })

  it("puts on exactly each style's classes and leaves the tone and font classes alone", () => {
    for (const style of ['classic', 'studio', BUILTIN_PLAY_STYLE, INSTALL_STYLE] as const) {
      html().className = 'font-a dark studio play'
      const boot = builtinThemeBoot(style, NO_COOKIES)
      runThemeBoot(boot)
      expect({ style, classes: [...html().classList].sort() }).toEqual({ style, classes: ['dark', 'font-a', ...boot.addClasses].sort() })
    }
  })

  it('does nothing, without throwing, with no element, a malformed one or a non-object', () => {
    html().className = 'dark'
    expect(runThemeBootScriptAlone).not.toThrow()
    for (const body of ['{not json', 'null', '"studio"', '42']) {
      expect(() => runThemeBoot(null, body)).not.toThrow()
    }
    expect(html().className).toBe('dark')
    expect(html().style.length).toBe(0)
  })

  it('re-checks classes and property names itself, so a boot that skipped the gate writes nothing bad', () => {
    const unchecked = sampleBoot({
      addClasses: ['ok-class', 'Bad Class', '"><img>'],
      prepaint: { knobs: { tint: { warm: { [SAMPLE_VAR]: '#FF6A1F', 'color': 'red', '--Up': 'red' } } }, pairs: [], numeric: [], defaults: { tint: 'warm' } },
    })
    runThemeBoot(unchecked)
    expect([...html().classList]).toEqual(['ok-class'])
    expect(html().style.getPropertyValue(SAMPLE_VAR)).toBe('#FF6A1F')
    expect(html().style.length).toBe(1)
  })

  it('paints the default look when no look storage is named', () => {
    runThemeBoot(sanitizeThemeBoot(sampleBoot()))
    expect(html().style.getPropertyValue(SAMPLE_VAR)).toBe('#FF6A1F')
  })
})

describe("PRD-261 W-A6b (gap G2) — an installed theme's stored look and tone before first paint", () => {
  const LOOK_STORAGE = bootLookStorage(installedLookKeys('automatos/play'))
  const GUARD = { schema: 'automatos-e2e-look/v1', iface: 'sample' }
  const SCOPE = 'ws-1:user_a'
  const COLD = '#1F6AFF'
  const FROM_LOOK: ThemeBoot['tone'] = { policy: 'from-look', value: null }

  /** An installed theme's boot: two tints, the warm one the default, its look keys and guard. */
  function installedBoot(fields: Partial<ThemeBoot> = {}): ThemeBoot {
    const knobs = { tint: { warm: { [SAMPLE_VAR]: '#FF6A1F' }, cold: { [SAMPLE_VAR]: COLD } } }
    return sampleBoot({ prepaint: { knobs, pairs: [], numeric: [], defaults: { tint: 'warm' } }, lookStorage: LOOK_STORAGE, lookGuard: GUARD, ...fields })
  }

  afterEach(() => window.localStorage.clear())

  it("keeps the theme's own look keys, a '/' in its id, through the gate, and still refuses a key that isn't one", () => {
    expect(sanitizeThemeBoot(installedBoot()).lookStorage).toEqual(LOOK_STORAGE)
    for (const pointerKey of ['</script>', 'a b', 'k"ey', 'x'.repeat(161), '']) {
      expect({ pointerKey, kept: sanitizeThemeBoot(installedBoot({ lookStorage: { ...LOOK_STORAGE, pointerKey } })).lookStorage }).toEqual({ pointerKey, kept: null })
    }
  })

  it("paints the look stored under the theme's own keys for the account the pointer names", () => {
    window.localStorage.setItem(LOOK_STORAGE.pointerKey, SCOPE)
    window.localStorage.setItem(`${LOOK_STORAGE.scopedPrefix}${SCOPE}`, JSON.stringify({ schema: GUARD.schema, interface: GUARD.iface, tint: 'cold' }))
    runThemeBoot(sanitizeThemeBoot(installedBoot()))
    expect(html().style.getPropertyValue(SAMPLE_VAR)).toBe(COLD)
  })

  const resolve = (cookies: Record<string, string>, boot: ThemeBoot) => createThemeBootResolver(cookieReader(cookies), async () => boot)(INSTALL_STYLE)

  it("starts a 'from-look' installed boot that names its look keys on the tone cookie, strictly", async () => {
    // Play's package: the theme the look runtime runs under, which keeps the tone cookie.
    const fromLook = installedBoot({ tone: FROM_LOOK, entry: PLAY_PACKAGE_ENTRY })
    expect((await resolve({ [PLAY_TONE_COOKIE]: 'light' }, fromLook)).tone).toEqual({ policy: 'from-look', value: 'light' })
    expect((await resolve({ [PLAY_TONE_COOKIE]: 'bogus' }, fromLook)).tone.value).toBeNull()
    expect((await resolve({}, fromLook)).tone.value).toBeNull()
    expect((await resolve({ [PLAY_TONE_COOKIE]: 'light' }, installedBoot({ tone: FROM_LOOK, entry: PLAY_PACKAGE_ENTRY, lookStorage: null }))).tone.value).toBeNull()
    expect((await resolve({ [PLAY_TONE_COOKIE]: 'light' }, installedBoot())).tone).toEqual({ policy: 'user', value: null })
  })

  it("leaves a 'from-look' installed boot whose class isn't Play's, or that carries no entry, off the tone cookie", async () => {
    const acme = { ...HELLO_PACKAGE_ENTRY, tone: 'from-look' as const }
    expect(acme.htmlClass).not.toBe(BUILTIN_PLAY_HTML_CLASS)
    expect((await resolve({ [PLAY_TONE_COOKIE]: 'dark' }, installedBoot({ tone: FROM_LOOK, entry: acme }))).tone).toEqual(FROM_LOOK)
    expect((await resolve({ [PLAY_TONE_COOKIE]: 'dark' }, installedBoot({ tone: FROM_LOOK, entry: null }))).tone).toEqual(FROM_LOOK)
  })
})
