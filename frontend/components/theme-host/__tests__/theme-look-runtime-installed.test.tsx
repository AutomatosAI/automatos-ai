/**
 * PRD-261 W-A6 — an installed theme keeps its first paint after hydration. With no registry entry
 * (its boot carried none) no look runtime mounts, so the LookProvider is switched off; it used to
 * clear every look variable on mount, wiping what the root layout's boot painted from the installed
 * theme's prepaint table (Play's package paints Play's own names, Studio's tokens among them), and
 * nothing painted it again. While the host loads or runs the installed theme the provider now leaves
 * <html> alone; a failed load, safe mode, Studio and Classic clear as before.
 *
 * PRD-261 W-A6b (gap G1) — with the entry the root layout built from its boot, Play's package gets
 * the look runtime as builtin Play does: the SDK's looks area is available, the look paints and
 * forces its tone. An installed theme with looks whose class isn't Play's gets no look from the one
 * runtime the host has (Play's), and keeps its first paint.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
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
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { clearRoot, lookNamesLeft, rootVars } from '@/lib/looks/__tests__/helpers'
import { applyLook, lookDocumentVars } from '@/lib/looks/document'
import { DEFAULT_LOOK } from '@/lib/looks/presets'
import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import type { ThemeTone } from '@/lib/theme-host/boot'
import type { ThemeHostStatus } from '@/lib/theme-host/types'
import { useLook } from '@/lib/theme-sdk/looks'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, type UiStyle } from '@/lib/ui-style'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'
import {
  HELLO_PACKAGE_ENTRY,
  INSTALLED_HELLO_STYLE,
  INSTALLED_PLAY_STYLE,
  PLAY_PACKAGE_ENTRY,
} from '@/lib/theme-host/__tests__/installed-entry-fixture'

/** next-themes' storage key in components/providers.tsx: the user's own tone. */
const TONE_STORAGE_KEY = 'automatos-theme'
const INSTALLED_STYLE: UiStyle = 'theme:install:0b9a3c3e-6a1d-4f9e-9a43-0c1f2d3e4f50'

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

function renderApp(style: UiStyle, status: ThemeHostStatus) {
  return render(<Appearance style={style} status={status}><Probe /></Appearance>)
}

/** What the root layout's boot does before any page code runs: the default look's variables on <html>. */
function paintFirstFrame(): void {
  applyLook(document.documentElement, DEFAULT_LOOK)
}

function toneCookie(): string | undefined {
  return document.cookie.split('; ').find((entry) => entry.startsWith(`${PLAY_TONE_COOKIE}=`))?.split('=')[1]
}

beforeEach(() => {
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  document.cookie = `${PLAY_TONE_COOKIE}=; Max-Age=0; Path=/`
  clearRoot()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.localStorage.clear()
  clearRoot()
})

describe('PRD-261 W-A6 — an installed theme keeps the first paint the boot put on <html>', () => {
  it.each<ThemeHostStatus>(['loading', 'active'])('host %s: every look variable the boot painted stays after mount', async (status) => {
    paintFirstFrame()
    renderApp(INSTALLED_STYLE, status)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
    expect(toneCookie()).toBeUndefined()
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

describe('PRD-261 W-A6b — an installed theme with its registry entry (gap G1)', () => {
  const play = installedThemeEntry(INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY)
  const hello = installedThemeEntry(INSTALLED_HELLO_STYLE, HELLO_PACKAGE_ENTRY)

  it.each<ThemeHostStatus>(['loading', 'active'])("Play's package, host %s: the looks are available, the look paints and forces its tone", async (status) => {
    paintFirstFrame()
    render(<Appearance style={INSTALLED_PLAY_STYLE} status={status} entry={play}><Probe /></Appearance>)
    expect(await screen.findByText('Looks: available')).toBeInTheDocument()
    expect(await screen.findByText('Tone: dark')).toBeInTheDocument()
    await waitFor(() => expect(toneCookie()).toBe('dark'))
    await waitFor(() => expect(document.documentElement.classList.contains('dark')).toBe(true))
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
  })

  it("a theme with looks whose class isn't Play's: no looks from Play's runtime, the first paint kept, no tone forced", async () => {
    paintFirstFrame()
    render(<Appearance style={INSTALLED_HELLO_STYLE} status="active" entry={hello}><Probe /></Appearance>)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK))
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
    expect(toneCookie()).toBeUndefined()
  })
})
