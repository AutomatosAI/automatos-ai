/**
 * PRD-260 K4, K9, D2, D7, M85 — the theme menu (components/ui/theme-toggle.tsx) under a theme whose
 * look sets the tone: an installed theme with looks (hello-a09b, tone 'from-look'), whose registry
 * entry the root layout builds from its boot and hands the style provider as `installedTheme`.
 *
 * - The Tone items rest, described by 'Set by your look': aria-disabled, so they stay in the
 *   keyboard's reach (M85), and choosing one changes nothing. The checked tone is the one the look
 *   forces, not the stored one it keeps for later. Under Studio, Classic and a theme whose tone is
 *   the user's, the Tone items are live with no note.
 * - 'Customise look…' (/looks) shows only while a theme is the style, and only where the theme's
 *   layout renders (its entry's minWidth), which is where /looks opens (D7).
 * - The look paints <html> at every width, so its tone is forced at every width (P260-SEAM): below
 *   the theme's width the generic look runtime still forces the look's tone through
 *   components/providers.tsx's forcedTheme, and only 'Customise look…' goes.
 * - useIsStudio stays true under a theme drawn on Studio; useActiveTheme names its entry only there.
 *
 * Replaces, in the core test ledger, components/__tests__/prd260-theme-menu.test.tsx's host cases,
 * which ran on builtin Play.
 */
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { useState, type ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest'

const tone = vi.hoisted(() => ({
  theme: 'dark',
  setTheme: vi.fn(),
  forcedTheme: undefined as string | undefined,
  /** True: useTheme reads the real ThemeProvider, for the tests that mount the providers as the app does. */
  real: false,
}))
vi.mock('next-themes', async (importOriginal) => {
  const actual = await importOriginal<typeof import('next-themes')>()
  return {
    ...actual,
    useTheme: () => {
      const live = actual.useTheme()
      return tone.real ? live : tone
    },
  }
})
// A plain anchor that forwards its ref, as Radix's asChild item needs.
vi.mock('next/link', async () => {
  const { forwardRef } = await vi.importActual<typeof import('react')>('react')
  return {
    default: forwardRef<HTMLAnchorElement, { href: string; children: ReactNode }>(function MockLink({ href, children, ...rest }, ref) {
      return <a ref={ref} href={href} {...rest}>{children}</a>
    }),
  }
})
// The look runtime keeps the signed-in user's looks in this workspace (not a uuid: nothing is sent to a server).
vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: 'user_a' } }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-1' }, isLoading: false, error: null, refreshWorkspace: async () => undefined }),
}))

import { ThemeHostContext, type ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { ThemeLookRuntime } from '@/components/theme-host/theme-look-runtime'
import { ThemeProvider } from '@/components/theme-provider'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { PHONE_WIDTH_PX, stubWindowWidth } from '@/hooks/__tests__/window-width'
import { useActiveTheme, useIsStudio } from '@/hooks/use-studio-theme'
import { HELLO_PACKAGE_ENTRY } from '@/lib/theme-host/__tests__/installed-entry-fixture'
import type { InstalledThemeEntryData, ThemeTone } from '@/lib/theme-host/boot'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import { HELLO_ENTRY, helloThemeBoot, putPageBoot } from '@/lib/theme-host/looks/__tests__/hello-looks'
import { THEME_MOTION_ATTRIBUTE } from '@/lib/theme-host/looks/motion'
import { THEME_TONE_COOKIE } from '@/lib/theme-host/looks/tone-cookie'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'
import { LOOKS_HREF } from '@/lib/ticket-links'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

/** An install of hello-a09b: a theme with looks whose look sets the tone. */
const LOOKS_STYLE: UiStyle = 'theme:install:3c8d2f1a-7b4e-4c6d-9e2f-1a19f2b0c0de'
/** An install of a theme whose tone is the user's (the hello package). */
const USER_TONE_STYLE: UiStyle = 'theme:install:9a1b2c3d-4e5f-4a6b-8c7d-0e1f2a3b4c5d'
/** next-themes' storage key in components/providers.tsx: the user's own tone. */
const TONE_STORAGE_KEY = 'automatos-theme'
const TONE_NOTE = 'Set by your look'
const TONES = ['Light', 'Dark', 'System']
const CUSTOMISE_LOOK = 'Customise look…'
const PROVIDERS = path.resolve(__dirname, '../providers.tsx')
/** More than the menu's items: enough to reach any of them from the first. */
const MAX_ARROW_PRESSES = 10

function entryOf(style: UiStyle, data: InstalledThemeEntryData): ThemeEntry {
  const entry = installedThemeEntry(style, data)
  if (entry === null) throw new Error(`no installed entry for ${style}`)
  return entry
}

/** hello-a09b's entry as the root layout builds it from the boot: tone 'from-look', drawn from the desktop width (1024) up. */
const LOOKS_THEME = entryOf(LOOKS_STYLE, { ...HELLO_ENTRY, minWidth: 1024 })
const USER_TONE_THEME = entryOf(USER_TONE_STYLE, HELLO_PACKAGE_ENTRY)

const html = () => document.documentElement.classList

function renderMenu(style: UiStyle, entry: ThemeEntry | null = null) {
  return render(<UiStyleProvider initialStyle={style} installedTheme={entry}><ThemeToggle /></UiStyleProvider>)
}

/** Opens the Appearance menu the way a keyboard user does (Radix opens on Enter). */
async function openMenu() {
  const trigger = await screen.findByRole('button', { name: 'Appearance' })
  await waitFor(() => expect(trigger).toBeEnabled())
  fireEvent.keyDown(trigger, { key: 'Enter' })
  await screen.findByRole('menu')
}

function choose(name: string) {
  act(() => { fireEvent.click(screen.getByRole('menuitemradio', { name })) })
}

/** Lets effects, and Radix's focus moves (a tick after the key), finish. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

/** Presses ArrowDown wherever focus is, as a keyboard user does, until `target` has focus or the presses run out. */
async function arrowDownTo(target: HTMLElement) {
  for (let press = 0; press < MAX_ARROW_PRESSES && document.activeElement !== target; press += 1) {
    fireEvent.keyDown(document.activeElement ?? document.body, { key: 'ArrowDown' })
    await settle()
  }
}

function hostValue(): ThemeHostContextValue {
  return { status: 'active', reason: null, frame: null, slots: null, reportRenderFailure: () => undefined }
}

interface AppearanceProps {
  readonly style: UiStyle
  readonly entry: ThemeEntry | null
  /** The tone the root layout hands Providers from the boot (the look's last tone, its cookie). */
  readonly serverTone?: ThemeTone
  readonly children: ReactNode
}

/** The appearance providers as components/providers.tsx wires them: the look's tone is forced at every width. */
function Appearance({ style, entry, serverTone, children }: AppearanceProps) {
  const [lookTone, setLookTone] = useState<ThemeTone | undefined>(serverTone)
  const [queryClient] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }))
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem themes={['light', 'dark']} storageKey={TONE_STORAGE_KEY} disableTransitionOnChange forcedTheme={lookTone}>
      <UiStyleProvider initialStyle={style} installedTheme={entry}>
        <QueryClientProvider client={queryClient}>
          <ThemeHostContext.Provider value={hostValue()}>
            <ThemeLookRuntime onToneChange={setLookTone}>{children}</ThemeLookRuntime>
          </ThemeHostContext.Provider>
        </QueryClientProvider>
      </UiStyleProvider>
    </ThemeProvider>
  )
}

function htmlTone(): string {
  return ['light', 'dark'].filter((name) => html().contains(name)).join(' ')
}

function resetRoot(): void {
  const root = document.documentElement
  root.className = ''
  root.removeAttribute('style')
  root.removeAttribute(THEME_MOTION_ATTRIBUTE)
}

let removeBoot: () => void = () => undefined

beforeAll(() => {
  // jsdom has no layout; Radix's popper may observe the menu's size.
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
})

beforeEach(() => {
  resetRoot()
  for (const name of [UI_STYLE_COOKIE, THEME_TONE_COOKIE]) document.cookie = `${name}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  tone.setTheme.mockClear()
  tone.real = false
})
afterEach(() => {
  cleanup()
  removeBoot()
  removeBoot = () => undefined
  vi.restoreAllMocks()
  resetRoot()
})

describe("PRD-260 K9, PRD-261 PA-14 — 'Customise look…'", () => {
  it('is offered (to /looks) only while a theme is the style', async () => {
    renderMenu(LOOKS_STYLE, LOOKS_THEME)
    await openMenu()
    expect(screen.getByRole('menuitem', { name: CUSTOMISE_LOOK })).toHaveAttribute('href', LOOKS_HREF)
    cleanup()

    for (const style of ['studio', 'classic'] as const) {
      renderMenu(style)
      await openMenu()
      expect(screen.queryByRole('menuitem', { name: CUSTOMISE_LOOK }), style).toBeNull()
      cleanup()
    }
  })
})

describe('PRD-260 D2 — useIsStudio and useActiveTheme under each style', () => {
  const wrap = (style: UiStyle, entry: ThemeEntry | null) =>
    function Wrapper({ children }: { children: ReactNode }) {
      return <UiStyleProvider initialStyle={style} installedTheme={entry}>{children}</UiStyleProvider>
    }
  const both = () => ({ studio: useIsStudio(), theme: useActiveTheme()?.ref === LOOKS_THEME.ref })
  const cases: Array<[UiStyle, ThemeEntry | null, { studio: boolean; theme: boolean }]> = [
    ['classic', null, { studio: false, theme: false }],
    ['studio', null, { studio: true, theme: false }],
    [LOOKS_STYLE, LOOKS_THEME, { studio: true, theme: true }],
  ]

  it.each(cases)('%s → %o', (style, entry, expected) => {
    const { result } = renderHook(both, { wrapper: wrap(style, entry) })
    expect(result.current).toEqual(expected)
  })

  it('both are false outside the provider', () => {
    const { result } = renderHook(both)
    expect(result.current).toEqual({ studio: false, theme: false })
  })
})

describe('PRD-260 K4 — under a theme whose look sets the tone', () => {
  it('shows the Tone items disabled with "Set by your look", and choosing one changes nothing', async () => {
    renderMenu(LOOKS_STYLE, LOOKS_THEME)
    await openMenu()
    expect(screen.getByText(TONE_NOTE)).toBeInTheDocument()
    for (const name of TONES) {
      expect(screen.getByRole('menuitemradio', { name })).toHaveAttribute('aria-disabled', 'true')
    }
    choose('Light')
    expect(tone.setTheme).not.toHaveBeenCalled()
  })

  it('checks the tone the look forces, not the stored one it keeps for later', async () => {
    tone.forcedTheme = 'light'
    try {
      renderMenu(LOOKS_STYLE, LOOKS_THEME)
      await openMenu()
      expect(screen.getByRole('menuitemradio', { name: 'Light' })).toHaveAttribute('aria-checked', 'true')
      expect(screen.getByRole('menuitemradio', { name: 'Dark' })).toHaveAttribute('aria-checked', 'false')
    } finally {
      tone.forcedTheme = undefined
    }
  })

  it("keeps the Tone items live, with no note, under Studio, Classic and a theme whose tone is the user's", async () => {
    const styles: Array<[UiStyle, ThemeEntry | null]> = [['studio', null], ['classic', null], [USER_TONE_STYLE, USER_TONE_THEME]]
    for (const [style, entry] of styles) {
      renderMenu(style, entry)
      await openMenu()
      expect(screen.queryByText(TONE_NOTE), style).toBeNull()
      for (const name of TONES) {
        expect(screen.getByRole('menuitemradio', { name }), `${style} ${name}`).not.toHaveAttribute('aria-disabled')
      }
      cleanup()
    }
  })
})

describe('PRD-260 M85 — at the theme\'s width the resting Tone items stay in reach', () => {
  it('ArrowDown reaches Dark, described by the note; Enter keeps the stored tone and the menu open', async () => {
    renderMenu(LOOKS_STYLE, LOOKS_THEME)
    await openMenu()
    const menu = screen.getByRole('menu')
    expect(menu).toContainElement(document.activeElement as HTMLElement)
    const dark = screen.getByRole('menuitemradio', { name: 'Dark' })
    await arrowDownTo(dark)
    expect(dark).toHaveFocus()
    expect(dark).toHaveAttribute('aria-disabled', 'true')
    expect(dark).toHaveAccessibleDescription(TONE_NOTE)
    act(() => { fireEvent.keyDown(dark, { key: 'Enter' }) })
    expect(tone.setTheme).not.toHaveBeenCalled()
    expect(screen.getByRole('menu')).toBeInTheDocument()
  })
})

describe("PRD-260 K4, D7 — below the theme's width the look still sets the tone (P260-SEAM)", () => {
  it("keeps the look's Light over a stored Dark: the Tone items rest under the note, and 'Customise look…' waits for the width", async () => {
    // The theme's layout reads the window against its entry's minWidth.
    const phone = stubWindowWidth(PHONE_WIDTH_PX)
    onTestFinished(() => phone.restore())
    // hello-a09b's default look, paper, sets Light; the user's own tone, kept for when they leave the theme, is Dark.
    removeBoot = putPageBoot(helloThemeBoot())
    window.localStorage.setItem(TONE_STORAGE_KEY, 'dark')
    tone.real = true
    render(<Appearance style={LOOKS_STYLE} entry={LOOKS_THEME} serverTone="light"><ThemeToggle /></Appearance>)
    await openMenu()
    await settle()
    expect(htmlTone()).toBe('light')
    expect(screen.getByText(TONE_NOTE)).toBeInTheDocument()
    expect(screen.queryByRole('menuitem', { name: CUSTOMISE_LOOK })).toBeNull()
    for (const name of TONES) {
      expect(screen.getByRole('menuitemradio', { name }), name).toHaveAttribute('aria-disabled', 'true')
    }
    choose('Dark')
    await settle()
    expect(htmlTone()).toBe('light')
    choose('Light')
    await settle()
    expect(window.localStorage.getItem(TONE_STORAGE_KEY)).toBe('dark')
  })

  it('is wired that way in components/providers.tsx: the tone is forced whatever the width', () => {
    const src = readFileSync(PROVIDERS, 'utf8')
    expect(src).toContain('forcedTheme={lookTone}')
    expect(src).toContain('<ThemeLookRuntime onToneChange={setLookTone}>')
    expect(src).not.toMatch(/forcedTheme=\{[^}]*narrow/)
  })
})
