/**
 * PRD-260 D1/D2/K9 — the theme menu reads Classic · Studio · Play. Choosing
 * Play writes the cookie and puts both classes on <html>; leaving it takes
 * 'play' off (and 'studio' too for Classic). "Customise look…" shows only
 * while Play is the style. useIsStudio stays true under Play; useActiveTheme is
 * Play's entry only there.
 *
 * PRD-261 A-02b: the menu lists Classic, Studio, then the theme registry's entries;
 * Play's item is its entry and sets `theme:builtin:play`. The tone rests under a
 * theme whose entry's tone policy is the look's, and 'Customise look…' follows the
 * theme's layout (its entry's minWidth), so the narrow case stubs the window width.
 *
 * P260-FX-06: Play survives a browser whose localStorage is unreadable (M87);
 * under the look's tone the Tone items stay in the keyboard's reach, described
 * by 'Set by your look' (M85).
 *
 * P260-SEAM (K4 over FX-06 and FX-17): the look paints <html> at every width,
 * so its tone is forced at every width too; below 1024 px only 'Customise
 * look…' goes, since /looks opens from 1024 px up (D7).
 */
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach, onTestFinished } from 'vitest'
import { render, renderHook, screen, cleanup, waitFor, fireEvent, act } from '@testing-library/react'
import { useState, type ReactNode } from 'react'

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
const width = vi.hoisted(() => ({ narrow: false }))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => width.narrow, useIsMobile: () => width.narrow }))
// A plain anchor that forwards its ref, as Radix's asChild item needs.
vi.mock('next/link', async () => {
  const { forwardRef } = await vi.importActual<typeof import('react')>('react')
  return {
    default: forwardRef<HTMLAnchorElement, { href: string; children: ReactNode }>(function MockLink(
      { href, children, ...rest },
      ref,
    ) {
      return <a ref={ref} href={href} {...rest}>{children}</a>
    }),
  }
})

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { LookProvider } from '@/lib/theme-host/builtin/play-look-provider'
import { ThemeProvider } from '@/components/theme-provider'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { useActiveTheme, useIsStudio } from '@/hooks/use-studio-theme'
import { PHONE_WIDTH_PX, stubWindowWidth } from '@/hooks/__tests__/window-width'
import { clearRoot } from '@/lib/looks/__tests__/helpers'
import { DAYLIGHT } from '@/lib/looks/presets'
import { DEFAULT_LOOK_SCOPE } from '@/lib/looks/scope'
import { lookKeys } from '@/lib/theme-host/builtin/play-look-storage'
import { PLAY_TONE_COOKIE, type PlayTone } from '@/lib/looks/tone'
import {
  BUILTIN_PLAY_HTML_CLASS as PLAY_HTML_CLASS,
  BUILTIN_PLAY_REF,
  BUILTIN_PLAY_STYLE as PLAY,
} from '@/lib/theme-host/builtin/play-entry'
import { BUILTIN_THEMES } from '@/lib/theme-host/registry'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  STUDIO_HTML_CLASS,
  UI_STYLE_COOKIE,
  type UiStyle,
} from '@/lib/ui-style'

const html = () => document.documentElement.classList

/** next-themes' storage key in components/providers.tsx: the user's own tone. */
const TONE_STORAGE_KEY = 'automatos-theme'
const TONE_NOTE = 'Set by your look'
const PROVIDERS = path.resolve(__dirname, '../providers.tsx')
/** More than the menu's seven items: enough to reach any of them from the first. */
const MAX_ARROW_PRESSES = 10

function renderMenu(style: UiStyle) {
  return render(<UiStyleProvider initialStyle={style}><ThemeToggle /></UiStyleProvider>)
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

/** The appearance providers as components/providers.tsx wires them: the look's tone is forced at every width. */
function Appearance({ style, playTone: serverTone, children }: { style: UiStyle; playTone?: PlayTone; children: ReactNode }) {
  const [playTone, setPlayTone] = useState<PlayTone | undefined>(serverTone)
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
        <LookProvider onToneChange={setPlayTone}>{children}</LookProvider>
      </UiStyleProvider>
    </ThemeProvider>
  )
}

function StyleProbe() {
  const { style } = useUiStyle()
  return <span data-testid="style">{style}</span>
}

/** The menu inside the real next-themes, style and look providers. */
function renderApp(style: UiStyle, playTone?: PlayTone) {
  tone.real = true
  return render(<Appearance style={style} playTone={playTone}><ThemeToggle /><StyleProbe /></Appearance>)
}

function htmlTone(): string {
  return ['light', 'dark'].filter((name) => html().contains(name)).join(' ')
}

function toneCookie(): string | undefined {
  return document.cookie.split('; ').find((entry) => entry.startsWith(`${PLAY_TONE_COOKIE}=`))?.split('=')[1]
}

/** Storage that refuses every read and write, like blocked site data or a sandboxed frame. */
function blockStorage() {
  const refuse = () => {
    throw new DOMException('The operation is insecure.', 'SecurityError')
  }
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(refuse)
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(refuse)
}

beforeAll(() => {
  // jsdom has no layout; Radix's popper may observe the menu's size.
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
})

beforeEach(() => {
  // No style or tone class, no look on <html>.
  clearRoot()
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  document.cookie = `${PLAY_TONE_COOKIE}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  tone.setTheme.mockClear()
  tone.real = false
  width.narrow = false
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('PRD-260 — the theme menu', () => {
  it('lists Classic, Studio, Play in the Style group, then the tones', async () => {
    renderMenu('studio')
    await openMenu()
    const items = screen.getAllByRole('menuitemradio').map((el) => el.textContent?.trim())
    expect(items).toEqual(['Classic', 'Studio', 'Play', 'Light', 'Dark', 'System'])
  })

  it("lists the registry's themes after Classic and Studio, by their entries' names", async () => {
    renderMenu('studio')
    await openMenu()
    const items = screen.getAllByRole('menuitemradio').map((el) => el.textContent?.trim())
    expect(items.slice(0, 2 + BUILTIN_THEMES.length)).toEqual(['Classic', 'Studio', ...BUILTIN_THEMES.map((entry) => entry.name)])
  })

  it("choosing Play sets the style to builtin Play's (theme:builtin:play) and checks it", async () => {
    render(<UiStyleProvider initialStyle="studio"><ThemeToggle /><StyleProbe /></UiStyleProvider>)
    await openMenu()
    choose('Play')
    expect(screen.getByTestId('style')).toHaveTextContent(PLAY)
    await openMenu()
    expect(screen.getByRole('menuitemradio', { name: 'Play' })).toHaveAttribute('aria-checked', 'true')
  })

  it('choosing Play writes the cookie and puts studio and play on <html>, without touching the tone', async () => {
    renderMenu('studio')
    await openMenu()
    choose('Play')
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${PLAY}`)
    expect(html().contains(STUDIO_HTML_CLASS)).toBe(true)
    expect(html().contains(PLAY_HTML_CLASS)).toBe(true)
    expect(tone.setTheme).not.toHaveBeenCalled()
  })

  it('leaving Play for Studio removes play and keeps studio; Classic removes both', async () => {
    renderMenu('studio')
    await openMenu()
    choose('Play')
    await openMenu()
    choose('Studio')
    expect(html().contains(PLAY_HTML_CLASS)).toBe(false)
    expect(html().contains(STUDIO_HTML_CLASS)).toBe(true)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=studio`)
    await openMenu()
    choose('Play')
    await openMenu()
    choose('Classic')
    expect(html().contains(PLAY_HTML_CLASS)).toBe(false)
    expect(html().contains(STUDIO_HTML_CLASS)).toBe(false)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=classic`)
  })

  it('offers "Customise look…" (to /looks) only while Play is the style', async () => {
    renderMenu(PLAY)
    await openMenu()
    const link = screen.getByRole('menuitem', { name: 'Customise look…' })
    expect(link).toHaveAttribute('href', '/looks')
    cleanup()

    for (const style of ['studio', 'classic'] as const) {
      renderMenu(style)
      await openMenu()
      expect(screen.queryByRole('menuitem', { name: 'Customise look…' }), style).toBeNull()
      cleanup()
    }
  })

  it('shows "Customise look…" as soon as Play is chosen', async () => {
    renderMenu('classic')
    await openMenu()
    expect(screen.queryByRole('menuitem', { name: 'Customise look…' })).toBeNull()
    choose('Play')
    await openMenu()
    expect(screen.getByRole('menuitem', { name: 'Customise look…' })).toBeInTheDocument()
  })
})

describe('PRD-260 D2 — useIsStudio and useActiveTheme under each style', () => {
  const wrap = (style: UiStyle) => ({ children }: { children: ReactNode }) => (
    <UiStyleProvider initialStyle={style}>{children}</UiStyleProvider>
  )
  const both = () => ({ studio: useIsStudio(), play: useActiveTheme()?.ref === BUILTIN_PLAY_REF })
  const cases: Array<[UiStyle, { studio: boolean; play: boolean }]> = [
    ['classic', { studio: false, play: false }],
    ['studio', { studio: true, play: false }],
    [PLAY, { studio: true, play: true }],
  ]

  it.each(cases)('%s → %o', (style, expected) => {
    const { result } = renderHook(both, { wrapper: wrap(style) })
    expect(result.current).toEqual(expected)
  })

  it('both are false outside the provider', () => {
    const { result } = renderHook(both)
    expect(result.current).toEqual({ studio: false, play: false })
  })
})

describe('PRD-260 K4 — under Play the look sets the tone', () => {
  it('shows the Tone items disabled with "Set by your look", and choosing one changes nothing', async () => {
    renderMenu(PLAY)
    await openMenu()
    expect(screen.getByText('Set by your look')).toBeInTheDocument()
    for (const name of ['Light', 'Dark', 'System']) {
      expect(screen.getByRole('menuitemradio', { name })).toHaveAttribute('aria-disabled', 'true')
    }
    choose('Light')
    expect(tone.setTheme).not.toHaveBeenCalled()
  })

  it('checks the tone the look forces, not the stored one it keeps for later', async () => {
    tone.forcedTheme = 'light'
    try {
      renderMenu(PLAY)
      await openMenu()
      expect(screen.getByRole('menuitemradio', { name: 'Light' })).toHaveAttribute('aria-checked', 'true')
      expect(screen.getByRole('menuitemradio', { name: 'Dark' })).toHaveAttribute('aria-checked', 'false')
    } finally {
      tone.forcedTheme = undefined
    }
  })

  it('keeps the Tone items live, with no note, under Studio and Classic', async () => {
    for (const style of ['studio', 'classic'] as const) {
      renderMenu(style)
      await openMenu()
      expect(screen.queryByText('Set by your look'), style).toBeNull()
      for (const name of ['Light', 'Dark', 'System']) {
        expect(screen.getByRole('menuitemradio', { name }), `${style} ${name}`).not.toHaveAttribute('aria-disabled')
      }
      cleanup()
    }
  })
})

describe('PRD-260 M85 — at desktop width the resting Tone items stay in reach', () => {
  it('ArrowDown reaches Dark, described by the note; Enter keeps the stored tone and the menu open', async () => {
    renderMenu(PLAY)
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

describe('PRD-260 K4, D7 — below 1024 px the look still sets the tone (P260-SEAM)', () => {
  it("keeps Daylight's Light over a stored Dark: the Tone items rest under the note, and 'Customise look…' waits for 1024 px", async () => {
    width.narrow = true
    // The theme's layout reads the window against Play's entry's minWidth (PRD-261 A-02b).
    const phone = stubWindowWidth(PHONE_WIDTH_PX)
    onTestFinished(() => phone.restore())
    // Daylight's Paper ground forces Light; the user's own tone, kept for when they leave Play, is Dark.
    window.localStorage.setItem(lookKeys(DEFAULT_LOOK_SCOPE).active, JSON.stringify(DAYLIGHT))
    window.localStorage.setItem(TONE_STORAGE_KEY, 'dark')
    renderApp(PLAY, 'light')
    await openMenu()
    await settle()
    expect(htmlTone()).toBe('light')
    expect(screen.getByText(TONE_NOTE)).toBeInTheDocument()
    expect(screen.queryByRole('menuitem', { name: 'Customise look…' })).toBeNull()
    for (const name of ['Light', 'Dark', 'System']) {
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
    expect(src).toContain('forcedTheme={playTone}')
    expect(src).not.toMatch(/forcedTheme=\{[^}]*narrow/)
  })
})

describe('PRD-260 M87 — Play in a browser whose localStorage is unreadable', () => {
  it('choosing Play keeps Play: it runs in Ember and the defaults never run again', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    blockStorage()
    renderApp('studio')
    await openMenu()
    choose('Play')
    await settle()
    expect(screen.getByTestId('style')).toHaveTextContent(PLAY)
    expect(html().contains(PLAY_HTML_CLASS)).toBe(true)
    // Ember's Graphite ground forced Dark: the change that hands out a new setTheme happened.
    expect(toneCookie()).toBe('dark')
  })
})
