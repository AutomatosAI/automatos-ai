/**
 * PRD-261 A-19d2 — replaces lib/looks/__tests__/tone.test.ts (the core ledger's coreReplacement,
 * A-19f2): the tone an installed theme's look sets and its cookie, `automatos-theme-tone`, parsed
 * strictly, written and cleared by the generic look runtime's provider, and read by the boot for an
 * installed 'from-look' theme, so its first paint is already that tone. The tone each choice gives
 * is the theme's own tone rule (hello-a09b's here: paper is light, ink and dusk are dark); Play's
 * per-ground values are the kit's (K-19c, its compile-looks test).
 *
 * The value names its theme (`<tone>.<mark>`, the mark a hash of the theme's id): one cookie serves
 * the whole browser, and a switch from installed theme A to installed theme B reloads without
 * leaving A, so B's boot takes only its own value; A's, or the old bare tone, is no tone.
 *
 * Plain .ts (the ledger names this path), so the provider is rendered with createElement.
 */
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { lookScope } from '@/lib/look-kit/scope'
import { createThemeBootResolver, type ThemeBoot } from '@/lib/theme-host/boot'
import type { ThemeTone } from '@/lib/theme-host/boot-types'
import { ThemeLookProvider, type ThemeLookProviderProps } from '@/lib/theme-host/looks/provider'
import { lookKeys } from '@/lib/theme-host/looks/storage'
import {
  THEME_TONE_COOKIE,
  clearThemeToneCookie,
  parseThemeToneCookie,
  themeToneCookieValue,
  themeToneMark,
  toneOfLook,
  writeThemeToneCookie,
} from '@/lib/theme-host/looks/tone-cookie'
import type { ThemeTonePolicy, UiStyle } from '@/lib/theme-host/registry-types'
import { useLook, type HostLookState } from '@/lib/theme-sdk/looks'

import { cookieReader } from '@/lib/theme-host/__tests__/boot-harness'
import { DUSK, HELLO, HELLO_ENTRY, HELLO_RULES, HELLO_THEME_ID, helloThemeBoot, INK, memoryStorage, PAPER, putPageBoot } from './hello-looks'

const RULE = HELLO_RULES.table.tone
const SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_a')
const INSTALL_STYLE: UiStyle = 'theme:install:5d0f2a7e-1c4b-4e9a-b3d6-8f1a2c3e4b5d'
/** A tone cookie under another name: the boot never reads it, and the runtime never writes it. */
const OTHER_TONE_COOKIE = 'automatos-other-tone'
/** Another installed theme, the one a switch to hello-a09b reloads from. */
const OTHER_THEME_ID = 'acme/other'
/** The cookie's value for hello-a09b's look tone. */
const hello = (tone: ThemeTone): string => themeToneCookieValue(HELLO_THEME_ID, tone)

function toneCookie(name: string = THEME_TONE_COOKIE): string | undefined {
  return document.cookie.split('; ').find((entry) => entry.startsWith(`${name}=`))?.split('=')[1]
}

/** The name a `document.cookie = ...` write sets. */
function cookieName(written: unknown): string {
  return String(written).split('=')[0]
}

function dropCookies(): void {
  for (const name of [THEME_TONE_COOKIE, OTHER_TONE_COOKIE]) document.cookie = `${name}=; Max-Age=0; Path=/`
}

/** What the SDK's useLook() last handed a theme below the provider. */
const seen: { current: HostLookState | null } = { current: null }

function Probe() {
  seen.current = useLook()
  return null
}

interface Rendered {
  readonly rerender: (fields: Partial<ThemeLookProviderProps>) => void
}

function renderProvider(onToneChange: (tone: ThemeTone | undefined) => void, store: Storage, tonePolicy: ThemeTonePolicy = 'from-look'): Rendered {
  const base: ThemeLookProviderProps = {
    children: createElement(Probe),
    themeId: HELLO_THEME_ID,
    enabled: true,
    tonePolicy,
    scope: SCOPE,
    onToneChange,
    storage: () => store,
    server: null,
  }
  const view = render(createElement(ThemeLookProvider, base))
  return { rerender: (fields) => view.rerender(createElement(ThemeLookProvider, { ...base, ...fields })) }
}

let removeBoot: () => void = () => undefined

beforeEach(() => {
  dropCookies()
  seen.current = null
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  removeBoot()
  removeBoot = () => undefined
  vi.restoreAllMocks()
  dropCookies()
  document.documentElement.removeAttribute('style')
})

describe("the theme's tone rule", () => {
  it("gives each choice of its knob the theme's own tone", () => {
    expect([PAPER, DUSK, INK].map((look) => [look.id, toneOfLook(RULE, look)])).toEqual([
      ['paper', 'light'],
      ['dusk', 'dark'],
      ['ink', 'dark'],
    ])
  })

  it("gives no tone without a rule, or for a choice the rule doesn't map", () => {
    expect(toneOfLook(undefined, PAPER)).toBeNull()
    expect(toneOfLook(RULE, { ...PAPER, ground: 'graphite' })).toBeNull()
    expect(toneOfLook(RULE, { ...PAPER, ground: 'toString' })).toBeNull()
  })
})

describe('the tone cookie', () => {
  it("parses only the theme's own 'dark' or 'light'", () => {
    expect(parseThemeToneCookie(hello('dark'), HELLO_THEME_ID)).toBe('dark')
    expect(parseThemeToneCookie(hello('light'), HELLO_THEME_ID)).toBe('light')
    const mark = themeToneMark(HELLO_THEME_ID)
    // The old bare tone, another theme's value, and anything that isn't exactly the theme's own.
    const others = ['dark', 'light', themeToneCookieValue(OTHER_THEME_ID, 'dark'), `Dark.${mark}`, `system.${mark}`, `toString.${mark}`]
    const malformed = [`dark.${mark}.x`, ` ${hello('dark')}`, `${hello('dark')}; Path=/`, 'toString', '', '.', null, undefined]
    for (const value of [...others, ...malformed]) {
      expect({ value, tone: parseThemeToneCookie(value, HELLO_THEME_ID) }).toEqual({ value, tone: null })
    }
  })

  it("names its theme by a bounded, URL-safe mark of the id (FNV-1a), never the id itself", () => {
    expect(['', 'a', 'foobar'].map(themeToneMark)).toEqual(['811c9dc5', 'e40c292c', 'bf9cf968'])
    expect(themeToneMark(HELLO_THEME_ID)).not.toBe(themeToneMark(OTHER_THEME_ID))
    for (const themeId of [HELLO_THEME_ID, OTHER_THEME_ID, 'x'.repeat(200)]) {
      expect(themeToneCookieValue(themeId, 'light')).toMatch(/^light\.[0-9a-f]{8}$/)
    }
  })

  it('is written and cleared on the whole site, and reads back as the tone for its own theme only', () => {
    writeThemeToneCookie(HELLO_THEME_ID, 'light')
    expect(toneCookie()).toBe(hello('light'))
    expect(parseThemeToneCookie(toneCookie(), HELLO_THEME_ID)).toBe('light')
    expect(parseThemeToneCookie(toneCookie(), OTHER_THEME_ID)).toBeNull()
    writeThemeToneCookie(HELLO_THEME_ID, 'dark')
    expect(parseThemeToneCookie(toneCookie(), HELLO_THEME_ID)).toBe('dark')
    clearThemeToneCookie()
    expect(toneCookie()).toBeUndefined()
  })

  it('clearing writes nothing when there is no tone cookie (a Classic or Studio load)', () => {
    const write = vi.spyOn(Document.prototype, 'cookie', 'set')
    clearThemeToneCookie()
    expect(write).not.toHaveBeenCalled()
  })
})

describe("the generic look runtime's provider", () => {
  it("writes the look's tone to the cookie and hands it to forcedTheme, follows the look, and clears it when switched off", async () => {
    removeBoot = putPageBoot(helloThemeBoot())
    const writes = vi.spyOn(Document.prototype, 'cookie', 'set')
    const onToneChange = vi.fn()
    const { rerender } = renderProvider(onToneChange, memoryStorage({ [lookKeys(SCOPE, HELLO.keys).active]: JSON.stringify(INK) }))
    await waitFor(() => expect(toneCookie()).toBe(hello('dark')))
    expect(onToneChange).toHaveBeenLastCalledWith('dark')
    expect(seen.current?.tone).toBe('dark')

    act(() => void seen.current?.setLook(PAPER))
    await waitFor(() => expect(toneCookie()).toBe(hello('light')))
    expect(onToneChange).toHaveBeenLastCalledWith('light')

    rerender({ enabled: false })
    await waitFor(() => expect(toneCookie()).toBeUndefined())
    // Switched off it hands up no tone: the host gives the tone back when no runtime runs (ThemeLookRuntime).
    expect(onToneChange).not.toHaveBeenCalledWith(undefined)
    // The only cookie it wrote, or deleted, is its own: no other tone cookie name.
    expect(writes).toHaveBeenCalled()
    expect(Array.from(new Set(writes.mock.calls.map(([value]) => cookieName(value))))).toEqual([THEME_TONE_COOKIE])
  })

  it("forces no tone and keeps no cookie under a theme whose tone doesn't follow its look", async () => {
    removeBoot = putPageBoot(helloThemeBoot())
    writeThemeToneCookie(HELLO_THEME_ID, 'dark')
    const onToneChange = vi.fn()
    renderProvider(onToneChange, memoryStorage(), 'user')
    await waitFor(() => expect(seen.current?.available).toBe(true))
    await waitFor(() => expect(toneCookie()).toBeUndefined())
    expect(seen.current?.tone).toBeNull()
    expect(onToneChange).not.toHaveBeenCalled()
  })
})

describe('the boot reads it for an installed from-look theme', () => {
  const resolve = (cookies: Record<string, string>, boot: ThemeBoot) => createThemeBootResolver(cookieReader(cookies), async () => boot)(INSTALL_STYLE)

  it("starts the theme on the cookie's tone, strictly", async () => {
    expect((await resolve({ [THEME_TONE_COOKIE]: hello('dark') }, helloThemeBoot())).tone).toEqual({ policy: 'from-look', value: 'dark' })
    expect((await resolve({ [THEME_TONE_COOKIE]: hello('light') }, helloThemeBoot())).tone.value).toBe('light')
    expect((await resolve({ [THEME_TONE_COOKIE]: 'bogus' }, helloThemeBoot())).tone.value).toBeNull()
    expect((await resolve({}, helloThemeBoot())).tone.value).toBeNull()
  })

  it("never starts it on another theme's tone (a switch from that installed theme) or on the old bare tone", async () => {
    expect((await resolve({ [THEME_TONE_COOKIE]: themeToneCookieValue(OTHER_THEME_ID, 'dark') }, helloThemeBoot())).tone).toEqual({ policy: 'from-look', value: null })
    expect((await resolve({ [THEME_TONE_COOKIE]: 'dark' }, helloThemeBoot())).tone).toEqual({ policy: 'from-look', value: null })
  })

  it("what the runtime wrote under a theme starts that theme again, never the installed theme a switch from it reloads into", async () => {
    removeBoot = putPageBoot(helloThemeBoot())
    renderProvider(vi.fn(), memoryStorage({ [lookKeys(SCOPE, HELLO.keys).active]: JSON.stringify(INK) }))
    await waitFor(() => expect(toneCookie()).toBe(hello('dark')))
    const written = { [THEME_TONE_COOKIE]: toneCookie() ?? '' }
    const other = helloThemeBoot({ entry: { ...HELLO_ENTRY, themeId: OTHER_THEME_ID } })
    expect((await resolve(written, helloThemeBoot())).tone).toEqual({ policy: 'from-look', value: 'dark' })
    expect((await resolve(written, other)).tone).toEqual({ policy: 'from-look', value: null })
  })

  it('reads no cookie but THEME_TONE_COOKIE: a tone under any other name is no tone', async () => {
    const read: string[] = []
    const cookies = cookieReader({ [OTHER_TONE_COOKIE]: hello('dark') })
    const recording = (name: string) => {
      read.push(name)
      return cookies(name)
    }
    const boot = await createThemeBootResolver(recording, async () => helloThemeBoot())(INSTALL_STYLE)
    expect(boot.tone).toEqual({ policy: 'from-look', value: null })
    expect(read).toEqual([THEME_TONE_COOKIE])
  })

  it('leaves it unread when nothing would keep it: no tone rule, no looks, no entry, no look keys, or not from-look', async () => {
    const { knobs, pairs, numeric, defaults } = HELLO_RULES.table
    const boots: ThemeBoot[] = [
      helloThemeBoot({ prepaint: { knobs, pairs, numeric, defaults } }),
      helloThemeBoot({ looks: null }),
      helloThemeBoot({ entry: null }),
      helloThemeBoot({ lookStorage: null }),
      helloThemeBoot({ tone: { policy: 'user', value: null } }),
    ]
    for (const boot of boots) {
      expect((await resolve({ [THEME_TONE_COOKIE]: hello('dark') }, boot)).tone.value).toBeNull()
    }
  })
})
