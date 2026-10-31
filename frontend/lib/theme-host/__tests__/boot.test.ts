/**
 * PRD-261 A-03 — first paint from data: the boot of every style, the gate it passes (bad classes
 * and bad prepaint values dropped, never written), its escape inside a <script> element, and
 * the one constant script that applies it. No theme is compiled into the app (A-19f1): Classic
 * and Studio get their base boot (their classes, nothing to paint, the user's tone), and an
 * installed theme gets its base's until its resolver answers with its own boot.
 *
 * PRD-261 W-A6b (gap G2): an installed theme's boot names its own look keys (a '/' in the theme
 * id), which pass the gate, so the script paints the stored look; a 'from-look' installed boot
 * with looks and a tone rule starts on the generic look runtime's tone cookie (A-19d2:
 * `automatos-theme-tone`), only on its own theme's value, so a switch from installed theme A to
 * installed theme B never starts B on A's tone.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  THEME_BOOT_ELEMENT_ID,
  THEME_BOOT_SCRIPT,
  baseThemeBoot,
  createThemeBootResolver,
  parseThemeTone,
  type ThemeBoot,
  type ThemeTone,
} from '@/lib/theme-host/boot'
import { CSS_VAR_NAME_PATTERN, sanitizeThemeBoot } from '@/lib/theme-host/boot-sanitize'
import { HTML_CLASS_PATTERN } from '@/lib/theme-host/constants'
import { isBootValue } from '@/lib/theme-host/css-value'
import { bootLookStorage, themeLookKeys } from '@/lib/theme-host/looks/storage'
import { THEME_TONE_COOKIE, themeToneCookieValue } from '@/lib/theme-host/looks/tone-cookie'
import { scriptSafeJson } from '@/lib/theme-host/script-safe-json'
import { STUDIO_HTML_CLASS, type UiStyle } from '@/lib/ui-style'

import { cookieReader, runThemeBoot, runThemeBootScriptAlone } from './boot-harness'
import { HELLO_PACKAGE_ENTRY, PLAY_PACKAGE_ENTRY } from './installed-entry-fixture'

const INSTALL_STYLE: UiStyle = 'theme:install:0b7e4c1a-9f3d-4c2e-8a51-3d2f6b7a9c10'
const NO_COOKIES = cookieReader()
const SAMPLE_VAR = '--sample-var'
/** An installed theme's own <html> class, as its boot names it beside its base's. */
const SAMPLE_THEME_CLASS = 'sample-theme'
/** The user's own tone: what Classic, Studio and a theme whose look does not set the tone start on. */
const USER_TONE: ThemeBoot['tone'] = { policy: 'user', value: null }
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
  it("gives Classic no class and Studio its class, removes the other, and paints nothing in the user's tone", async () => {
    expect(baseThemeBoot('classic')).toMatchObject({ addClasses: [], removeClasses: [STUDIO_HTML_CLASS] })
    expect(baseThemeBoot('studio')).toMatchObject({ addClasses: [STUDIO_HTML_CLASS], removeClasses: [] })
    for (const style of ['classic', 'studio'] as const) {
      expect(await createThemeBootResolver(NO_COOKIES)(style)).toEqual({
        ...baseThemeBoot(style),
        tone: USER_TONE,
        prepaint: null,
        lookStorage: null,
        lookGuard: null,
        integrity: null,
      })
    }
  })

  it("gives an installed theme its base's classes (Studio) and nothing to paint, until its resolver answers", async () => {
    const expected = { addClasses: [STUDIO_HTML_CLASS], removeClasses: [], prepaint: null, integrity: null }
    expect(baseThemeBoot(INSTALL_STYLE)).toMatchObject(expected)
    expect(await createThemeBootResolver(NO_COOKIES)(INSTALL_STYLE)).toMatchObject(expected)
  })

  it('reads no tone cookie for Classic or Studio, and reads a tone strictly (PRD-260 K4)', async () => {
    const light = cookieReader({ [THEME_TONE_COOKIE]: themeToneCookieValue(PLAY_PACKAGE_ENTRY.themeId, 'light') })
    const installed = vi.fn(async () => sampleBoot())
    for (const style of ['classic', 'studio'] as const) {
      expect((await createThemeBootResolver(light)(style)).tone).toEqual(USER_TONE)
      expect((await createThemeBootResolver(light, installed)(style)).tone).toEqual(USER_TONE)
    }
    expect(installed).not.toHaveBeenCalled()
    for (const value of ['Dark', 'system', '', ' dark', 'toString', null, undefined]) {
      expect({ value, tone: parseThemeTone(value) }).toEqual({ value, tone: null })
    }
  })

  it("asks the installed resolver for an installed style only, gates its boot, and falls back when it fails", async () => {
    const installed = vi.fn(async () => sampleBoot({ addClasses: ['studio', 'hello-theme', 'Not A Class'] }))
    const resolve = createThemeBootResolver(NO_COOKIES, installed)
    expect((await resolve(INSTALL_STYLE)).addClasses).toEqual(['studio', 'hello-theme'])
    expect(await resolve('studio')).toEqual(baseThemeBoot('studio'))
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
    for (const value of hostile) expect({ value, ok: isBootValue(value) }).toEqual({ value, ok: false })
    const fine = ['#FF6A1F', 'rgba(244,238,232,.12)', 'hsl(12 5% 7% / 0.14)', '12 5% 7%', 'var(--theme-font-display),sans-serif', '-.03em', 'hue-rotate(18deg) saturate(1.2)']
    for (const value of fine) expect({ value, ok: isBootValue(value) }).toEqual({ value, ok: true })
    const vars = { [SAMPLE_VAR]: '#FF6A1F', '--x-bad': 'url(x)', '--Upper': '#000000', 'color': '#000000' }
    const boot = sanitizeThemeBoot(sampleBoot({ prepaint: { knobs: { tint: { warm: vars } }, pairs: [], numeric: [], defaults: { tint: 'warm' } } }))
    expect(boot.prepaint?.knobs.tint.warm).toEqual({ [SAMPLE_VAR]: '#FF6A1F' })
  })

  it('drops the whole table when its default look no longer fits it', () => {
    const boot = sanitizeThemeBoot(sampleBoot({ prepaint: { knobs: { tint: { warm: {} } }, pairs: [], numeric: [], defaults: { tint: 'cold' } } }))
    expect(boot.prepaint).toBeNull()
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
    for (const style of ['classic', 'studio', INSTALL_STYLE] as const) {
      html().className = `font-a dark ${STUDIO_HTML_CLASS}`
      const boot = baseThemeBoot(style)
      runThemeBoot(boot)
      expect({ style, classes: [...html().classList].sort() }).toEqual({ style, classes: ['dark', 'font-a', ...boot.addClasses].sort() })
    }
    html().className = 'font-a dark'
    runThemeBoot(sanitizeThemeBoot(sampleBoot({ addClasses: [STUDIO_HTML_CLASS, SAMPLE_THEME_CLASS] })))
    expect([...html().classList].sort()).toEqual(['dark', 'font-a', SAMPLE_THEME_CLASS, STUDIO_HTML_CLASS].sort())
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
  const LOOK_STORAGE = bootLookStorage(themeLookKeys(PLAY_PACKAGE_ENTRY.themeId))
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

  it("hands an installed theme's own classes, look table, look keys and guard through the resolver, whole", async () => {
    const boot = installedBoot({ addClasses: [STUDIO_HTML_CLASS, SAMPLE_THEME_CLASS] })
    const resolved = await resolve({}, boot)
    expect(resolved.addClasses).toEqual([STUDIO_HTML_CLASS, SAMPLE_THEME_CLASS])
    expect(resolved.removeClasses).toEqual([])
    expect(resolved.prepaint).toEqual(boot.prepaint)
    expect(resolved.lookStorage).toEqual(LOOK_STORAGE)
    expect(resolved.lookGuard).toEqual(GUARD)
    expect(resolved.tone).toEqual(USER_TONE)
  })

  /** A 'from-look' installed boot the generic look runtime runs: a tone rule on its tint, and one preset. */
  function lookingBoot(fields: Partial<ThemeBoot> = {}): ThemeBoot {
    const base = installedBoot()
    const prepaint = base.prepaint === null ? null : { ...base.prepaint, tone: { knob: 'tint', values: { warm: 'light' as const, cold: 'dark' as const } } }
    const preset = { schema: GUARD.schema, interface: GUARD.iface, id: 'warm', name: 'Warm', sub: 'The default', tint: 'warm' }
    return installedBoot({ tone: FROM_LOOK, entry: PLAY_PACKAGE_ENTRY, prepaint, looks: { defaultLook: preset.id, presets: [preset] }, ...fields })
  }

  /** The generic runtime's cookie as it keeps Play's package's look tone (lookingBoot's entry). */
  const playTone = (tone: ThemeTone) => ({ [THEME_TONE_COOKIE]: themeToneCookieValue(PLAY_PACKAGE_ENTRY.themeId, tone) })

  it("starts a 'from-look' installed boot with looks and a tone rule on the generic runtime's tone cookie, strictly (A-19d2)", async () => {
    expect((await resolve(playTone('light'), lookingBoot())).tone).toEqual({ policy: 'from-look', value: 'light' })
    expect((await resolve({ [THEME_TONE_COOKIE]: 'bogus' }, lookingBoot())).tone.value).toBeNull()
    expect((await resolve({}, lookingBoot())).tone.value).toBeNull()
    expect((await resolve(playTone('light'), lookingBoot({ lookStorage: null }))).tone.value).toBeNull()
    expect((await resolve(playTone('light'), installedBoot())).tone).toEqual(USER_TONE)
  })

  it("never starts installed theme B on installed theme A's tone (A -> B reloads without leaving A), nor on the old bare tone", async () => {
    const themeA = { [THEME_TONE_COOKIE]: themeToneCookieValue(HELLO_PACKAGE_ENTRY.themeId, 'dark') }
    expect((await resolve(themeA, lookingBoot())).tone).toEqual(FROM_LOOK)
    expect((await resolve({ [THEME_TONE_COOKIE]: 'dark' }, lookingBoot())).tone).toEqual(FROM_LOOK)
    expect((await resolve(playTone('dark'), lookingBoot())).tone).toEqual({ policy: 'from-look', value: 'dark' })
  })

  it("starts any installed theme with looks on it, whatever its class; none without an entry, a tone rule or looks", async () => {
    const acme = { ...HELLO_PACKAGE_ENTRY, tone: 'from-look' as const }
    expect(acme.htmlClass).not.toBe(PLAY_PACKAGE_ENTRY.htmlClass)
    const acmeTone = { [THEME_TONE_COOKIE]: themeToneCookieValue(acme.themeId, 'dark') }
    expect((await resolve(acmeTone, lookingBoot({ entry: acme }))).tone).toEqual({ policy: 'from-look', value: 'dark' })
    expect((await resolve(playTone('dark'), lookingBoot({ entry: null }))).tone).toEqual(FROM_LOOK)
    expect((await resolve(playTone('dark'), installedBoot({ tone: FROM_LOOK, entry: PLAY_PACKAGE_ENTRY }))).tone).toEqual(FROM_LOOK)
    expect((await resolve(playTone('dark'), lookingBoot({ looks: null }))).tone).toEqual(FROM_LOOK)
  })
})
