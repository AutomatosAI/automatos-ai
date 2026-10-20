/**
 * PRD-261 A-19d2 — replaces lib/looks/__tests__/tone.test.ts (the core ledger's coreReplacement,
 * A-19f2): the tone an installed theme's look sets and its cookie, `automatos-theme-tone`, parsed
 * strictly, written and cleared by the generic look runtime's provider, and read by the boot for an
 * installed 'from-look' theme, so its first paint is already that tone. The tone each choice gives
 * is the theme's own tone rule (hello-a09b's here: paper is light, ink and dusk are dark); Play's
 * per-ground values are the kit's (K-19c, its compile-looks test).
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
  toneOfLook,
  writeThemeToneCookie,
} from '@/lib/theme-host/looks/tone-cookie'
import type { ThemeTonePolicy, UiStyle } from '@/lib/theme-host/registry-types'
import { useLook, type HostLookState } from '@/lib/theme-sdk/looks'

import { cookieReader } from '@/lib/theme-host/__tests__/boot-harness'
import { DUSK, HELLO, HELLO_RULES, HELLO_THEME_ID, helloThemeBoot, INK, memoryStorage, PAPER, putPageBoot } from './hello-looks'

const RULE = HELLO_RULES.table.tone
const SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_a')
const INSTALL_STYLE: UiStyle = 'theme:install:5d0f2a7e-1c4b-4e9a-b3d6-8f1a2c3e4b5d'
/** Builtin Play's own tone cookie (lib/theme-host/builtin/play-boot.ts names it), never read for an installed theme. */
const BUILTIN_TONE_COOKIE = 'automatos-play-tone'

function toneCookie(name: string = THEME_TONE_COOKIE): string | undefined {
  return document.cookie.split('; ').find((entry) => entry.startsWith(`${name}=`))?.split('=')[1]
}

function dropCookies(): void {
  for (const name of [THEME_TONE_COOKIE, BUILTIN_TONE_COOKIE]) document.cookie = `${name}=; Max-Age=0; Path=/`
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
  it("parses only 'dark' and 'light'", () => {
    expect(parseThemeToneCookie('dark')).toBe('dark')
    expect(parseThemeToneCookie('light')).toBe('light')
    for (const value of ['Dark', 'system', '', ' dark', 'dark; Path=/', 'toString', null, undefined]) {
      expect({ value, tone: parseThemeToneCookie(value) }).toEqual({ value, tone: null })
    }
  })

  it('is written and cleared on the whole site', () => {
    writeThemeToneCookie('light')
    expect(toneCookie()).toBe('light')
    writeThemeToneCookie('dark')
    expect(toneCookie()).toBe('dark')
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
    const onToneChange = vi.fn()
    const { rerender } = renderProvider(onToneChange, memoryStorage({ [lookKeys(SCOPE, HELLO.keys).active]: JSON.stringify(INK) }))
    await waitFor(() => expect(toneCookie()).toBe('dark'))
    expect(onToneChange).toHaveBeenLastCalledWith('dark')
    expect(seen.current?.tone).toBe('dark')

    act(() => void seen.current?.setLook(PAPER))
    await waitFor(() => expect(toneCookie()).toBe('light'))
    expect(onToneChange).toHaveBeenLastCalledWith('light')

    rerender({ enabled: false })
    await waitFor(() => expect(toneCookie()).toBeUndefined())
    // Switched off it hands up no tone: the host gives the tone back when no runtime runs (ThemeLookRuntime).
    expect(onToneChange).not.toHaveBeenCalledWith(undefined)
    expect(toneCookie(BUILTIN_TONE_COOKIE)).toBeUndefined()
  })

  it("forces no tone and keeps no cookie under a theme whose tone doesn't follow its look", async () => {
    removeBoot = putPageBoot(helloThemeBoot())
    writeThemeToneCookie('dark')
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
    expect((await resolve({ [THEME_TONE_COOKIE]: 'dark' }, helloThemeBoot())).tone).toEqual({ policy: 'from-look', value: 'dark' })
    expect((await resolve({ [THEME_TONE_COOKIE]: 'light' }, helloThemeBoot())).tone.value).toBe('light')
    expect((await resolve({ [THEME_TONE_COOKIE]: 'bogus' }, helloThemeBoot())).tone.value).toBeNull()
    expect((await resolve({}, helloThemeBoot())).tone.value).toBeNull()
  })

  it("never reads builtin Play's tone cookie for an installed theme", async () => {
    expect((await resolve({ [BUILTIN_TONE_COOKIE]: 'dark' }, helloThemeBoot())).tone).toEqual({ policy: 'from-look', value: null })
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
      expect((await resolve({ [THEME_TONE_COOKIE]: 'dark' }, boot)).tone.value).toBeNull()
    }
  })
})
