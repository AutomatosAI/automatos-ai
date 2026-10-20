/**
 * PRD-261 W-A6 — an installed theme keeps its first paint after hydration. With no registry entry
 * (its boot carried none) no look runtime mounts, so the look providers are switched off; one used
 * to clear every look variable on mount, wiping what the root layout's boot painted from the
 * installed theme's prepaint table (Play's package paints Play's own names, Studio's tokens among
 * them), and nothing painted it again. While the host loads or runs the installed theme the
 * providers now leave <html> alone; a failed load, safe mode, Studio and Classic clear as before.
 *
 * PRD-261 A-19d2 (was W-A6b's gap G1 on builtin Play's runtime) — an installed theme with looks runs
 * on the generic look runtime (lib/theme-host/looks), from its own boot on the page: the SDK's looks
 * area is available, the look paints from the theme's table, <html> gets data-theme-motion and
 * --theme-motion and never data-pl-motion, and the look's tone follows the table's tone rule into
 * `automatos-theme-tone` and forcedTheme. Play's package is such a theme (its class no longer picks
 * builtin Play's runtime), and so is a theme whose class isn't Play's. Builtin Play keeps its own
 * runtime until W-A8.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: 'user_a' } }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-1' }, isLoading: false, error: null, refreshWorkspace: async () => undefined }),
}))

import { ThemeHostContext, type ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { ThemeLookRuntime } from '@/components/theme-host/theme-look-runtime'
import { ThemeProvider } from '@/components/theme-provider'
import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { clearRoot, lookNamesLeft, rootVars } from '@/lib/looks/__tests__/helpers'
import { GROUND_KEYS } from '@/lib/looks/catalogue'
import { applyLook, LOOK_DOCUMENT_NAMES, lookDocumentVars, MOTION_ATTRIBUTE } from '@/lib/looks/document'
import { DEFAULT_LOOK, PRESETS } from '@/lib/looks/presets'
import { PLAY_TONE_COOKIE, toneOfGround } from '@/lib/looks/tone'
import { LOOK_INTERFACE, LOOK_SCHEMA } from '@/lib/looks/types'
import { runThemeBoot } from '@/lib/theme-host/__tests__/boot-harness'
import type { ThemeBoot, ThemeTone } from '@/lib/theme-host/boot'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { lookPrepaintTable } from '@/lib/theme-host/builtin/play-prepaint'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import { HELLO_ENTRY, HELLO_RULES, helloThemeBoot, PAPER, putPageBoot } from '@/lib/theme-host/looks/__tests__/hello-looks'
import { THEME_MOTION_ATTRIBUTE, THEME_MOTION_VAR } from '@/lib/theme-host/looks/motion'
import { paint, paintNames } from '@/lib/theme-host/looks/paint'
import { bootLookStorage, themeLookKeys } from '@/lib/theme-host/looks/storage'
import { THEME_TONE_COOKIE, themeToneCookieValue } from '@/lib/theme-host/looks/tone-cookie'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'
import type { ThemeHostStatus } from '@/lib/theme-host/types'
import { useLook } from '@/lib/theme-sdk/looks'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'
import {
  HELLO_PACKAGE_ENTRY,
  INSTALLED_HELLO_STYLE,
  INSTALLED_PLAY_STYLE,
  PLAY_PACKAGE_ENTRY,
} from '@/lib/theme-host/__tests__/installed-entry-fixture'

/** next-themes' storage key in components/providers.tsx: the user's own tone. */
const TONE_STORAGE_KEY = 'automatos-theme'
const INSTALLED_STYLE: UiStyle = 'theme:install:0b9a3c3e-6a1d-4f9e-9a43-0c1f2d3e4f50'
/** The install of hello-a09b, a small theme with looks whose class isn't Play's. */
const HELLO_LOOKS_STYLE: UiStyle = 'theme:install:6e1f3b8f-2d5c-4fab-a4e7-9a2b3d4f5c6e'

function hostValue(status: ThemeHostStatus): ThemeHostContextValue {
  return { status, reason: status === 'failed' ? 'load' : null, frame: null, slots: null, reportRenderFailure: () => undefined }
}

interface AppearanceProps {
  readonly style: UiStyle
  readonly status: ThemeHostStatus
  /** The installed theme's entry, as the root layout builds it from the boot; none by default. */
  readonly entry?: ThemeEntry | null
  readonly children: ReactNode
}

/** The providers as components/providers.tsx nests them, the theme host's status given. */
function Appearance({ style, status, entry = null, children }: AppearanceProps) {
  const [playTone, setPlayTone] = useState<ThemeTone | undefined>(undefined)
  const [queryClient] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }))
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem themes={['light', 'dark']} storageKey={TONE_STORAGE_KEY} disableTransitionOnChange forcedTheme={playTone}>
      <UiStyleProvider initialStyle={style} installedTheme={entry}>
        <QueryClientProvider client={queryClient}>
          <ThemeHostContext.Provider value={hostValue(status)}>
            <ThemeLookRuntime onToneChange={setPlayTone}>{children}</ThemeLookRuntime>
          </ThemeHostContext.Provider>
        </QueryClientProvider>
      </UiStyleProvider>
    </ThemeProvider>
  )
}

function Probe() {
  const { available, tone } = useLook()
  return (
    <div>
      <p>Looks: {available ? 'available' : 'none'}</p>
      <p>Tone: {tone ?? 'not forced'}</p>
    </div>
  )
}

/** The style menu's switch, made in place for a style that isn't an installed theme. */
function StyleSwitch({ to }: { readonly to: UiStyle }) {
  const { setStyle } = useUiStyle()
  return <button type="button" onClick={() => setStyle(to)}>Switch to {to}</button>
}

/** hello-a09b's table variables currently inline on <html> (none of them builtin Play's names). */
function helloVars(): Readonly<Record<string, string>> {
  const { style } = document.documentElement
  const entries = paintNames(HELLO_RULES.table).map((name): [string, string] => [name, style.getPropertyValue(name).trim()])
  return Object.fromEntries(entries.filter(([, value]) => value !== ''))
}

function renderApp(style: UiStyle, status: ThemeHostStatus, entry: ThemeEntry | null = null) {
  return render(<Appearance style={style} status={status} entry={entry}><Probe /></Appearance>)
}

/** What the root layout's boot does before any page code runs: the default look's variables on <html>. */
function paintFirstFrame(): void {
  applyLook(document.documentElement, DEFAULT_LOOK)
}

function cookie(name: string): string | undefined {
  return document.cookie.split('; ').find((entry) => entry.startsWith(`${name}=`))?.split('=')[1]
}

function dropCookies(): void {
  for (const name of [PLAY_TONE_COOKIE, THEME_TONE_COOKIE, UI_STYLE_COOKIE]) document.cookie = `${name}=; Max-Age=0; Path=/`
}

function motionMarks(): { readonly theme: string | null; readonly variable: string; readonly builtin: string | null } {
  const root = document.documentElement
  return {
    theme: root.getAttribute(THEME_MOTION_ATTRIBUTE),
    variable: root.style.getPropertyValue(THEME_MOTION_VAR),
    builtin: root.getAttribute(MOTION_ATTRIBUTE),
  }
}

/** Play's package boot as the kit compiles it (K-19c): Play's table with its tone rule, its guard and presets. */
function playPackageBoot(): ThemeBoot {
  const values = Object.fromEntries(GROUND_KEYS.map((key) => [key, toneOfGround(key)]))
  return helloThemeBoot({
    addClasses: ['studio', PLAY_PACKAGE_ENTRY.htmlClass],
    prepaint: { ...lookPrepaintTable(), tone: { knob: 'ground', values } },
    lookStorage: bootLookStorage(themeLookKeys(PLAY_PACKAGE_ENTRY.themeId)),
    lookGuard: { schema: LOOK_SCHEMA, iface: LOOK_INTERFACE },
    looks: { defaultLook: DEFAULT_LOOK.id, presets: PRESETS.map((look) => ({ ...look })) },
    entry: PLAY_PACKAGE_ENTRY,
  })
}

let removeBoot: () => void = () => undefined

beforeEach(() => {
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  dropCookies()
  clearRoot()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  removeBoot()
  removeBoot = () => undefined
  vi.restoreAllMocks()
  window.localStorage.clear()
  dropCookies()
  clearRoot()
  document.documentElement.removeAttribute(THEME_MOTION_ATTRIBUTE)
  document.documentElement.classList.remove(HELLO_ENTRY.htmlClass)
})

describe('PRD-261 W-A6 — an installed theme keeps the first paint the boot put on <html>', () => {
  it.each<ThemeHostStatus>(['loading', 'active'])('host %s: every look variable the boot painted stays after mount', async (status) => {
    paintFirstFrame()
    renderApp(INSTALLED_STYLE, status)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
    expect(cookie(PLAY_TONE_COOKIE)).toBeUndefined()
    expect(cookie(THEME_TONE_COOKIE)).toBeUndefined()
  })

  it('a load that fails takes the first paint off, as Studio shows instead', async () => {
    paintFirstFrame()
    const { rerender } = renderApp(INSTALLED_STYLE, 'active')
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
    rerender(<Appearance style={INSTALLED_STYLE} status="failed"><Probe /></Appearance>)
    expect(lookNamesLeft()).toEqual([])
  })

  it.each<ThemeHostStatus>(['failed', 'safe-mode'])('host %s from the start: the look variables are cleared on mount', async (status) => {
    paintFirstFrame()
    renderApp(INSTALLED_STYLE, status)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(lookNamesLeft()).toEqual([])
  })

  it.each<UiStyle>(['studio', 'classic'])('%s (host off): stray look variables are cleared on mount, as before', async (style) => {
    paintFirstFrame()
    renderApp(style, 'off')
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(lookNamesLeft()).toEqual([])
  })
})

describe("PRD-261 A-19d2 — Play's package runs on the generic look runtime", () => {
  const play = installedThemeEntry(INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY)

  it.each<ThemeHostStatus>(['loading', 'active'])('host %s: the looks are available, the look paints and forces its tone', async (status) => {
    const boot = playPackageBoot()
    runThemeBoot(boot)
    removeBoot = putPageBoot(boot)
    renderApp(INSTALLED_PLAY_STYLE, status, play)
    expect(await screen.findByText('Looks: available')).toBeInTheDocument()
    expect(await screen.findByText('Tone: dark')).toBeInTheDocument()
    await waitFor(() => expect(cookie(THEME_TONE_COOKIE)).toBe(themeToneCookieValue(PLAY_PACKAGE_ENTRY.themeId, 'dark')))
    await waitFor(() => expect(document.documentElement.classList.contains('dark')).toBe(true))
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
  })

  it("marks motion with data-theme-motion and --theme-motion, never data-pl-motion, and keeps no builtin tone cookie", async () => {
    document.cookie = `${PLAY_TONE_COOKIE}=light; Path=/`
    removeBoot = putPageBoot(playPackageBoot())
    renderApp(INSTALLED_PLAY_STYLE, 'active', play)
    await waitFor(() => expect(motionMarks()).toEqual({ theme: 'full', variable: '1', builtin: null }))
    await waitFor(() => expect(cookie(PLAY_TONE_COOKIE)).toBeUndefined())
  })

  it.each<ThemeHostStatus>(['safe-mode', 'failed'])(
    'host %s before it released the entry: no look runtime, nothing repainted, no tone forced',
    async (status) => {
      paintFirstFrame()
      removeBoot = putPageBoot(playPackageBoot())
      // The host's release reaches the style context a commit later; here the entry still stands.
      renderApp(INSTALLED_PLAY_STYLE, status, play)
      expect(await screen.findByText('Looks: none')).toBeInTheDocument()
      expect(lookNamesLeft()).toEqual([])
      expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
      expect(cookie(THEME_TONE_COOKIE)).toBeUndefined()
      expect(motionMarks()).toEqual({ theme: null, variable: '', builtin: null })
    },
  )
})

describe('PRD-261 A-19d2 — any installed theme with looks runs on it, from its own boot', () => {
  const hello = installedThemeEntry(HELLO_LOOKS_STYLE, HELLO_ENTRY)

  it("a theme whose class isn't Play's: its look paints from its table and its tone rule forces the tone", async () => {
    const boot = helloThemeBoot()
    runThemeBoot(boot)
    removeBoot = putPageBoot(boot)
    renderApp(HELLO_LOOKS_STYLE, 'active', hello)
    expect(await screen.findByText('Looks: available')).toBeInTheDocument()
    expect(await screen.findByText('Tone: light')).toBeInTheDocument()
    await waitFor(() => expect(cookie(THEME_TONE_COOKIE)).toBe(themeToneCookieValue(HELLO_ENTRY.themeId, 'light')))
    expect(helloVars()).toEqual(paint(HELLO_RULES.table, PAPER))
    expect(motionMarks()).toEqual({ theme: 'full', variable: '1', builtin: null })
  })

  it("left in place for Studio: its table's variables and the motion marks come off once the host stops, also names builtin Play's runtime never clears", async () => {
    const table = HELLO_RULES.table
    expect(paintNames(table).filter((name) => LOOK_DOCUMENT_NAMES.includes(name))).toEqual([])
    const boot = helloThemeBoot()
    runThemeBoot(boot)
    removeBoot = putPageBoot(boot)
    const { rerender } = render(<Appearance style={HELLO_LOOKS_STYLE} status="active" entry={hello}><Probe /><StyleSwitch to="studio" /></Appearance>)
    await waitFor(() => expect(helloVars()).toEqual(paint(table, PAPER)))
    await waitFor(() => expect(motionMarks().theme).toBe('full'))

    // A switch away from an installed theme is made in place (only one TO an installed theme reloads),
    // and the host's status trails the style by a commit: it still runs the theme here.
    fireEvent.click(screen.getByRole('button', { name: 'Switch to studio' }))
    // UiStyleProvider reads its props once, at mount, as from the root layout: only the host's status changes here.
    rerender(<Appearance style="studio" status="off" entry={null}><Probe /><StyleSwitch to="studio" /></Appearance>)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    await waitFor(() => expect(helloVars()).toEqual({}))
    expect(motionMarks()).toEqual({ theme: null, variable: '', builtin: null })
    expect(cookie(THEME_TONE_COOKIE)).toBeUndefined()
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
  })

  it('a theme with looks whose boot on the page carries none: no looks, the first paint kept, no tone forced', async () => {
    paintFirstFrame()
    const entry = installedThemeEntry(INSTALLED_HELLO_STYLE, HELLO_PACKAGE_ENTRY)
    render(<Appearance style={INSTALLED_HELLO_STYLE} status="active" entry={entry}><Probe /></Appearance>)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
    expect(cookie(THEME_TONE_COOKIE)).toBeUndefined()
  })
})

describe('PRD-261 A-19d2 — builtin Play keeps its own runtime until W-A8', () => {
  it('marks motion with data-pl-motion only, and keeps its tone in its own cookie', async () => {
    renderApp(BUILTIN_PLAY_STYLE, 'off')
    expect(await screen.findByText('Looks: available')).toBeInTheDocument()
    await waitFor(() => expect(motionMarks()).toEqual({ theme: null, variable: '', builtin: 'full' }))
    await waitFor(() => expect(cookie(PLAY_TONE_COOKIE)).toBe('dark'))
    expect(cookie(THEME_TONE_COOKIE)).toBeUndefined()
  })
})
