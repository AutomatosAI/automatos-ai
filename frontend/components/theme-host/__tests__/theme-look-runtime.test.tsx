/**
 * PRD-261 A-13c — the host mounts the active theme's look runtime (ThemeLookRuntime), at every width,
 * whenever the theme declares looks, as components/providers.tsx wires it (the tone it forces goes to
 * ThemeProvider's forcedTheme). Classic and Studio mount none: no scope, no storage read, nothing on
 * <html>. Builtin Play below its 1024 px minWidth, where its frame gives way to the core's, still paints
 * its look and forces the look's tone. Leaving Play switches the runtime off and the host gives the tone
 * back; the page below never remounts on that switch (M69). The SDK's looks area reads the mounted
 * runtime, every look it hands in checked first.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { useEffect, useState, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const pageMounts = vi.hoisted(() => ({ count: 0 }))

vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: 'user_a' } }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-1' }, isLoading: false, error: null, refreshWorkspace: async () => undefined }),
}))

import { useLookScope } from '@/components/looks/look-scope'
import { declaresLooks, ThemeLookRuntime } from '@/components/theme-host/theme-look-runtime'
import { ThemeProvider } from '@/components/theme-provider'
import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { useThemeLayout } from '@/hooks/use-studio-theme'
import { clearRoot, lookNamesLeft, myLook, rootVars } from '@/lib/looks/__tests__/helpers'
import { lookDocumentVars } from '@/lib/looks/document'
import { DAYLIGHT } from '@/lib/looks/presets'
import { lookScope } from '@/lib/looks/scope'
import { LOOK_SCOPE_KEY, lookKeys } from '@/lib/looks/storage'
import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import { lookJson } from '@/lib/looks/transfer'
import type { ThemeTone } from '@/lib/theme-host/boot'
import { BUILTIN_PLAY_ENTRY, BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { BUILTIN_PLAY_LOOK_KEYS } from '@/lib/theme-host/builtin/play-look-keys'
import { exportLook, importLook, listLooks, saveLook, useLook, type HostLook, type HostLookResult } from '@/lib/theme-sdk/looks'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, type UiStyle } from '@/lib/ui-style'

/** next-themes' storage key in components/providers.tsx: the user's own tone. */
const TONE_STORAGE_KEY = 'automatos-theme'
const KEYS = lookKeys(lookScope('ws-1', 'user_a'))
/** Narrower than builtin Play's 1024 px minWidth: its frame gives way to the core's here. */
const NARROW_PX = 800
const NOT_AVAILABLE = "Looks aren't available here."

let queryClient = new QueryClient()

function quietQueryClient(): QueryClient {
  const quiet = () => undefined
  return new QueryClient({ defaultOptions: { queries: { retry: false } }, logger: { log: quiet, warn: quiet, error: quiet } })
}

/** The appearance providers as components/providers.tsx nests them, the look runtime inside the query client. */
function Appearance({ style, children }: { readonly style: UiStyle; readonly children: ReactNode }) {
  const [playTone, setPlayTone] = useState<ThemeTone | undefined>(undefined)
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem
      themes={['light', 'dark']}
      storageKey={TONE_STORAGE_KEY}
      disableTransitionOnChange
      forcedTheme={playTone}
    >
      <UiStyleProvider initialStyle={style}>
        <QueryClientProvider client={queryClient}>
          <ThemeLookRuntime onToneChange={setPlayTone}>{children}</ThemeLookRuntime>
        </QueryClientProvider>
      </UiStyleProvider>
    </ThemeProvider>
  )
}

/** Every key builtin Play keeps its looks under starts with this one (lib/theme-host/builtin/play-look-keys.ts). */
const LOOK_KEY_PREFIX = BUILTIN_PLAY_LOOK_KEYS.active

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
      <button type="button" onClick={() => setLook(DAYLIGHT)}>Use Daylight</button>
      <button type="button" onClick={() => setStyle('studio')}>Go Studio</button>
      <button type="button" onClick={() => setStyle(BUILTIN_PLAY_STYLE)}>Go Play</button>
    </div>
  )
}

function renderApp(style: UiStyle) {
  return render(<Appearance style={style}><Probe /></Appearance>)
}

function click(name: string): void {
  act(() => screen.getByRole('button', { name }).click())
}

function htmlTone(): string {
  const classes = document.documentElement.classList
  return ['light', 'dark'].filter((tone) => classes.contains(tone)).join(' ')
}

function toneCookie(): string | undefined {
  return document.cookie.split('; ').find((entry) => entry.startsWith(`${PLAY_TONE_COOKIE}=`))?.split('=')[1]
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

beforeEach(() => {
  queryClient = quietQueryClient()
  pageMounts.count = 0
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  window.localStorage.setItem(TONE_STORAGE_KEY, 'dark')
  window.localStorage.setItem(KEYS.active, JSON.stringify(DAYLIGHT))
  document.cookie = `${PLAY_TONE_COOKIE}=; Max-Age=0; Path=/`
  clearRoot()
  stubWidth(NARROW_PX)
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  window.localStorage.clear()
  clearRoot()
})

describe('PRD-261 A-13c — which themes get a look runtime', () => {
  it('is a theme that fills the look editor slot: builtin Play, never Classic or Studio, nor a theme without one', () => {
    expect(declaresLooks(BUILTIN_PLAY_ENTRY)).toBe(true)
    expect(declaresLooks(null)).toBe(false)
    expect(declaresLooks({ ...BUILTIN_PLAY_ENTRY, slots: ['frame@1', 'notFound@1'] })).toBe(false)
  })

  it.each<UiStyle>(['classic', 'studio'])('%s: no look runtime, no scope, nothing read or painted', async (style) => {
    const reads = vi.spyOn(Storage.prototype, 'getItem')
    renderApp(style)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(screen.getByText('Scope: none')).toBeInTheDocument()
    expect(reads.mock.calls.map(([key]) => key).filter((key) => key.startsWith(LOOK_KEY_PREFIX))).toEqual([])
    expect(screen.getByText('Look: none')).toBeInTheDocument()
    expect(lookNamesLeft()).toEqual([])
    expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBeNull()
    expect(htmlTone()).toBe('dark')
    expect(toneCookie()).toBeUndefined()
  })
})

describe('PRD-261 A-13c — builtin Play below its minWidth', () => {
  it("still forces the look's tone and paints its variables at 800 px, where its frame gives way", async () => {
    renderApp(BUILTIN_PLAY_STYLE)
    expect(screen.getByText('Layout: fallback')).toBeInTheDocument()
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DAYLIGHT)))
    await waitFor(() => expect(htmlTone()).toBe('light'))
    expect(toneCookie()).toBe('light')
    expect(screen.getByText(`Scope: ${lookScope('ws-1', 'user_a')}`)).toBeInTheDocument()
    expect(screen.getByText('Look: daylight')).toBeInTheDocument()
    expect(await screen.findByText('Tone: light')).toBeInTheDocument()
    expect(window.localStorage.getItem(TONE_STORAGE_KEY)).toBe('dark')
  })

  it('leaving Play switches the runtime off: the look comes off <html>, the user tone returns, the cookie goes', async () => {
    renderApp(BUILTIN_PLAY_STYLE)
    await waitFor(() => expect(htmlTone()).toBe('light'))
    click('Go Studio')
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    await waitFor(() => expect(htmlTone()).toBe('dark'))
    expect(lookNamesLeft()).toEqual([])
    expect(toneCookie()).toBeUndefined()
    expect(listLooks()).toEqual([])
    expect(saveLook(myLook())).toEqual({ ok: false, errors: [NOT_AVAILABLE] })
  })
})

describe('PRD-261 A-13c, M69 — a style switched in place never remounts the page', () => {
  it('keeps the page mounted across Studio -> Play -> Studio, while the runtime turns on and off', async () => {
    renderApp('studio')
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    click('Go Play')
    expect(await screen.findByText('Look: daylight')).toBeInTheDocument()
    await waitFor(() => expect(htmlTone()).toBe('light'))
    click('Go Studio')
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    await waitFor(() => expect(htmlTone()).toBe('dark'))
    expect(lookNamesLeft()).toEqual([])
    expect(pageMounts.count).toBe(1)
  })
})

describe("PRD-261 A-13c — the SDK's looks area on the mounted runtime", () => {
  it('saves a checked look, lists it, and hands out frozen copies', async () => {
    renderApp(BUILTIN_PLAY_STYLE)
    await screen.findByText('Look: daylight')
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
    renderApp(BUILTIN_PLAY_STYLE)
    await screen.findByText('Look: daylight')
    const forged: HostLook = { ...myLook(), accent: 'url(https://evil.example/x.png)' } as HostLook
    const answer: { result?: HostLookResult } = {}
    act(() => {
      answer.result = saveLook(forged)
    })
    expect(answer.result?.ok).toBe(false)
    expect(screen.getByText('Saved: none')).toBeInTheDocument()
    expect(exportLook(forged)).toBeNull()
  })

  it('sets a look through useLook, exports it as a file and imports one under a fresh id', async () => {
    window.localStorage.removeItem(KEYS.active)
    renderApp(BUILTIN_PLAY_STYLE)
    click('Use Daylight')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DAYLIGHT)))
    expect(exportLook(DAYLIGHT)).toEqual({ fileName: 'Daylight.look.json', text: lookJson(DAYLIGHT), type: 'application/json' })
    const imported = importLook(lookJson(myLook()))
    expect(imported).toMatchObject({ ok: true, look: { name: 'My look 1' } })
    expect(imported.ok && imported.look.id).not.toBe('my-look-1')
    expect(importLook('not json').ok).toBe(false)
  })
})
