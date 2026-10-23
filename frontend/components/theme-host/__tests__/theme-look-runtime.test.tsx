/**
 * PRD-261 A-13c, A-19f2 — the host mounts the active theme's look runtime (ThemeLookRuntime), at every
 * width, whenever an installed theme that declares looks runs, as components/providers.tsx wires it
 * (the tone it forces goes to ThemeProvider's forcedTheme). The theme here is hello-a09b, its boot on
 * the page (lib/theme-host/looks/__tests__/hello-looks.ts). Classic and Studio mount none: no scope,
 * no storage read, nothing on <html>. Below the theme's minWidth, where its frame gives way to the
 * core's, the runtime still paints its look and forces the look's tone. Leaving the theme switches
 * the runtime off and the host gives the tone back; the page below never remounts while the runtime
 * turns on and off (M69). The SDK's looks area reads the mounted runtime, every look it hands in
 * checked first.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { useEffect, type ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const pageMounts = vi.hoisted(() => ({ count: 0 }))

vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: 'user_a' } }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-1' }, isLoading: false, error: null, refreshWorkspace: async () => undefined }),
}))

import { useLookScope } from '@/components/theme-host/look-scope'
import { declaresLooks } from '@/components/theme-host/theme-look-runtime'
import { useUiStyle } from '@/contexts/ui-style-context'
import { useThemeLayout } from '@/hooks/use-studio-theme'
import { lookScope } from '@/lib/look-kit/scope'
import { DUSK, HELLO, HELLO_RULES, HELLO_THEME_ID, helloThemeBoot, myLook, putPageBoot } from '@/lib/theme-host/looks/__tests__/hello-looks'
import { clearRoot, cookieValue, dropCookies, htmlTone, lookNamesLeft, motionMarks, rootVars } from '@/lib/theme-host/looks/__tests__/look-test-helpers'
import { paint } from '@/lib/theme-host/looks/paint'
import { INSTALLED_LOOK_KEY_PREFIX, lookKeys } from '@/lib/theme-host/looks/storage'
import { LOOKS_NOT_AVAILABLE } from '@/lib/theme-host/looks/store'
import { THEME_TONE_COOKIE, themeToneCookieValue } from '@/lib/theme-host/looks/tone-cookie'
import { lookFile } from '@/lib/theme-host/looks/transfer'
import type { ThemeHostStatus } from '@/lib/theme-host/types'
import { exportLook, importLook, listLooks, saveLook, useLook, type HostLook, type HostLookResult } from '@/lib/theme-sdk/looks'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

import { HELLO_LOOKS_STYLE, HELLO_LOOKS_THEME, LookAppearance, TONE_STORAGE_KEY, USER_DEFAULT_TONE } from './look-runtime-harness'

const PROVIDERS = path.resolve(__dirname, '../../providers.tsx')
const LOOK_RUNTIME = path.resolve(__dirname, '../theme-look-runtime.tsx')
const TABLE = HELLO_RULES.table
const SCOPE = lookScope('ws-1', 'user_a')
const KEYS = lookKeys(SCOPE, HELLO.keys)
/** The account's active look: a light one (hello-a09b's tone rule: paper is light), unlike the user's own tone. */
const STORED = myLook({ id: 'my-light-look', name: 'My light look', ground: 'paper' })
/** Narrower than the 1024 px an installed theme's frame needs (INSTALLED_FRAME_MIN_WIDTH_PX): its frame gives way here. */
const NARROW_PX = 800
const LIGHT = 'light'
const DARK = 'dark'

/** A theme's view of its looks, through the SDK only; it counts its own mounts, as a page's state would see them. */
function Probe() {
  const { available, look, saved, tone, setLook } = useLook()
  const { setStyle } = useUiStyle()
  useEffect(() => {
    pageMounts.count += 1
  }, [])
  return (
    <div>
      <p>Layout: {useThemeLayout()}</p>
      <p>Scope: {useLookScope() ?? 'none'}</p>
      <p>Looks: {available ? 'available' : 'none'}</p>
      <p>Look: {look?.id ?? 'none'}</p>
      <p>Saved: {saved.map((entry) => entry.name).join(', ') || 'none'}</p>
      <p>Tone: {tone ?? 'not forced'}</p>
      <button type="button" onClick={() => setLook(DUSK)}>Use Dusk</button>
      <button type="button" onClick={() => setStyle('studio')}>Go Studio</button>
    </div>
  )
}

/** The app as the root layout renders it for `style`: hello-a09b's entry under its install, none otherwise. */
function app(style: UiStyle, status: ThemeHostStatus) {
  const entry = style === HELLO_LOOKS_STYLE ? HELLO_LOOKS_THEME : null
  return <LookAppearance style={style} status={status} entry={entry}><Probe /></LookAppearance>
}

function renderApp(style: UiStyle = HELLO_LOOKS_STYLE, status: ThemeHostStatus = 'active') {
  return render(app(style, status))
}

function click(name: string): void {
  act(() => screen.getByRole('button', { name }).click())
}

/** The user leaves the theme for Studio, in place; the host's status follows (it stops running the theme). */
function leaveForStudio(rerender: (ui: ReactElement) => void): void {
  click('Go Studio')
  rerender(app('studio', 'off'))
}

/** A window `width` px wide, for every max-width / min-width query; no reduced motion, no dark preference. */
function stubWidth(width: number): void {
  const noop = () => undefined
  vi.stubGlobal('matchMedia', (query: string) => {
    const max = /max-width:\s*(\d+)px/.exec(query)
    const min = /min-width:\s*(\d+)px/.exec(query)
    const matches = max !== null ? width <= Number(max[1]) : min !== null && width >= Number(min[1])
    return { matches, media: query, onchange: null, addEventListener: noop, removeEventListener: noop, addListener: noop, removeListener: noop, dispatchEvent: () => false }
  })
}

/** `order`'s tags all appear in `src`, in that order. */
function expectInOrder(src: string, order: readonly string[]): void {
  const at = order.map((tag) => src.indexOf(tag))
  expect(at.every((index) => index >= 0), order.join(' ')).toBe(true)
  expect([...at].sort((a, b) => a - b)).toEqual(at)
}

let removeBoot: () => void = () => undefined

beforeEach(() => {
  pageMounts.count = 0
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  window.localStorage.setItem(TONE_STORAGE_KEY, USER_DEFAULT_TONE)
  window.localStorage.setItem(KEYS.active, JSON.stringify(STORED))
  dropCookies([THEME_TONE_COOKIE, UI_STYLE_COOKIE])
  clearRoot()
  removeBoot = putPageBoot(helloThemeBoot())
  stubWidth(NARROW_PX)
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  removeBoot()
  removeBoot = () => undefined
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  window.localStorage.clear()
  dropCookies([THEME_TONE_COOKIE, UI_STYLE_COOKIE])
  clearRoot()
})

describe('PRD-261 A-13c — which themes get a look runtime', () => {
  it('is an installed theme that fills the look editor slot, never Classic or Studio, nor a theme without one', () => {
    expect(declaresLooks(HELLO_LOOKS_THEME)).toBe(true)
    expect(declaresLooks(null)).toBe(false)
    expect(declaresLooks({ ...HELLO_LOOKS_THEME, slots: ['frame@1', 'notFound@1'] })).toBe(false)
  })

  it.each<UiStyle>(['classic', 'studio'])('%s: no look runtime, no scope, nothing read or painted', async (style) => {
    const reads = vi.spyOn(Storage.prototype, 'getItem')
    renderApp(style, 'off')
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(screen.getByText('Scope: none')).toBeInTheDocument()
    expect(reads.mock.calls.map(([key]) => key).filter((key) => key.startsWith(INSTALLED_LOOK_KEY_PREFIX))).toEqual([])
    expect(screen.getByText('Look: none')).toBeInTheDocument()
    expect(lookNamesLeft()).toEqual([])
    expect(motionMarks()).toEqual({ attribute: null, variable: '' })
    expect(window.localStorage.getItem(HELLO.keys.pointer)).toBeNull()
    expect(htmlTone()).toBe(USER_DEFAULT_TONE)
    expect(cookieValue(THEME_TONE_COOKIE)).toBeUndefined()
  })
})

describe("PRD-261 A-13c — an installed theme with looks below its minWidth", () => {
  it("still forces the look's tone and paints its variables at 800 px, where its frame gives way", async () => {
    renderApp()
    expect(screen.getByText('Layout: fallback')).toBeInTheDocument()
    await waitFor(() => expect(rootVars()).toEqual(paint(TABLE, STORED)))
    await waitFor(() => expect(htmlTone()).toBe(LIGHT))
    expect(cookieValue(THEME_TONE_COOKIE)).toBe(themeToneCookieValue(HELLO_THEME_ID, LIGHT))
    expect(screen.getByText(`Scope: ${SCOPE}`)).toBeInTheDocument()
    expect(screen.getByText(`Look: ${STORED.id}`)).toBeInTheDocument()
    expect(await screen.findByText('Tone: light')).toBeInTheDocument()
    expect(window.localStorage.getItem(TONE_STORAGE_KEY)).toBe(USER_DEFAULT_TONE)
  })
})

describe("PRD-261 A-13c — the look's tone, forced over the user's own", () => {
  it('follows the look, never overwrites the stored tone, and the stored tone returns on Studio', async () => {
    window.localStorage.setItem(TONE_STORAGE_KEY, LIGHT)
    const { rerender } = renderApp()
    expect(await screen.findByText('Tone: light')).toBeInTheDocument()
    click('Use Dusk')
    await waitFor(() => expect(htmlTone()).toBe(DARK))
    expect(cookieValue(THEME_TONE_COOKIE)).toBe(themeToneCookieValue(HELLO_THEME_ID, DARK))
    expect(await screen.findByText('Tone: dark')).toBeInTheDocument()
    expect(window.localStorage.getItem(TONE_STORAGE_KEY)).toBe(LIGHT)
    leaveForStudio(rerender)
    await waitFor(() => expect(htmlTone()).toBe(LIGHT))
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
    expect(window.localStorage.getItem(TONE_STORAGE_KEY)).toBe(LIGHT)
  })

  it('leaving the theme switches the runtime off: the look comes off <html>, the user tone returns, the cookie goes', async () => {
    const { rerender } = renderApp()
    await waitFor(() => expect(htmlTone()).toBe(LIGHT))
    leaveForStudio(rerender)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    await waitFor(() => expect(htmlTone()).toBe(USER_DEFAULT_TONE))
    await waitFor(() => expect(lookNamesLeft()).toEqual([]))
    expect(motionMarks()).toEqual({ attribute: null, variable: '' })
    expect(cookieValue(THEME_TONE_COOKIE)).toBeUndefined()
    expect(listLooks()).toEqual([])
    expect(saveLook(myLook())).toEqual(LOOKS_NOT_AVAILABLE)
  })
})

describe('PRD-261 A-13c, M69 — the runtime turning on and off never remounts the page', () => {
  it('keeps the page mounted while the host starts the theme and the user then leaves it for Studio', async () => {
    // The host starts off and loads the theme after mount (its first state is 'off'): no first paint of
    // the runtime's ran, so it paints the stored look and forces its tone the moment it turns on.
    const { rerender } = renderApp(HELLO_LOOKS_STYLE, 'off')
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(lookNamesLeft()).toEqual([])
    rerender(app(HELLO_LOOKS_STYLE, 'loading'))
    expect(await screen.findByText(`Look: ${STORED.id}`)).toBeInTheDocument()
    await waitFor(() => expect(rootVars()).toEqual(paint(TABLE, STORED)))
    await waitFor(() => expect(htmlTone()).toBe(LIGHT))
    rerender(app(HELLO_LOOKS_STYLE, 'active'))
    leaveForStudio(rerender)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    await waitFor(() => expect(htmlTone()).toBe(USER_DEFAULT_TONE))
    expect(lookNamesLeft()).toEqual([])
    expect(pageMounts.count).toBe(1)
  })
})

describe("PRD-261 A-13c — the SDK's looks area on the mounted runtime", () => {
  it('saves a checked look, lists it, and hands out frozen copies', async () => {
    renderApp()
    await screen.findByText(`Look: ${STORED.id}`)
    const answer: { result?: HostLookResult } = {}
    act(() => {
      answer.result = saveLook(myLook())
    })
    expect(answer.result).toMatchObject({ ok: true, look: { id: 'my-look-1', name: 'My look 1' } })
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
    await waitFor(() => expect(listLooks().map((look) => look.id)).toEqual(['my-look-1']))
    expect(Object.isFrozen(listLooks()[0])).toBe(true)
    expect(JSON.parse(window.localStorage.getItem(KEYS.saved) ?? 'null')).toEqual([myLook()])
  })

  it("refuses a look the theme's checks refuse, and never keeps it", async () => {
    renderApp()
    await screen.findByText(`Look: ${STORED.id}`)
    const forged: HostLook = { ...myLook(), accent: 'url(https://evil.example/x.png)' } as HostLook
    const answer: { result?: HostLookResult } = {}
    act(() => {
      answer.result = saveLook(forged)
    })
    expect(answer.result?.ok).toBe(false)
    expect(screen.getByText('Saved: none')).toBeInTheDocument()
    expect(window.localStorage.getItem(KEYS.saved)).toBeNull()
    expect(exportLook(forged)).toBeNull()
  })

  it('sets a look through useLook, exports it as a file and imports one under a fresh id', async () => {
    window.localStorage.removeItem(KEYS.active)
    renderApp()
    await screen.findByText('Looks: available')
    click('Use Dusk')
    await waitFor(() => expect(rootVars()).toEqual(paint(TABLE, DUSK)))
    expect(JSON.parse(window.localStorage.getItem(KEYS.active) ?? 'null')).toEqual(DUSK)
    expect(exportLook(DUSK)).toEqual(lookFile(HELLO_RULES, DUSK))
    expect(exportLook(DUSK)?.fileName).toBe('Dusk.look.json')
    const imported = importLook(JSON.stringify(myLook()))
    expect(imported).toMatchObject({ ok: true, look: { name: 'My look 1' } })
    expect(imported.ok && imported.look.id).not.toBe('my-look-1')
    expect(importLook('not json').ok).toBe(false)
  })
})

describe('PRD-260 K4, F5 — how components/providers.tsx wires the runtime', () => {
  it("forces the tone the runtime hands up, from the server's tone cookie on, inside the WorkspaceProvider", () => {
    const providers = readFileSync(PROVIDERS, 'utf8')
    // At every width, as this file's harness renders it: the look paints below the frame's width too (K4).
    expect(providers).toContain('useState<ThemeTone | undefined>(initialTone)')
    expect(providers).toContain('forcedTheme={lookTone}')
    expectInOrder(providers, ['<WorkspaceProvider>', '<ThemeLookRuntime onToneChange={setLookTone}>', '</ThemeLookRuntime>', '</WorkspaceProvider>'])
  })

  it("hands the provider the tone setter and the account's scope, below AccountLookScope", () => {
    const runtime = readFileSync(LOOK_RUNTIME, 'utf8')
    expectInOrder(runtime, ['<AccountLookScope enabled={mounted}>', '<InstalledLooks', '</InstalledLooks>', '</AccountLookScope>'])
    expectInOrder(runtime, ['<ThemeLookProvider', 'scope={scope}', 'onToneChange={onToneChange}', '</ThemeLookProvider>'])
  })
})
