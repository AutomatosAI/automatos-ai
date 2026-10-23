/**
 * PRD-244 W3 — the Style axis: html class + cookie + storage on switch; the legacy "studio" tone migrates once.
 * PRD-260 M87 — once per mount: a new setTheme from next-themes never re-applies the defaults.
 * PRD-261 A-02a — the context names the active theme (its registry entry) and has no isPlay.
 * PRD-261 W-A6b (gap G1) — under an installed theme's style it names the entry the root layout built from the
 * boot, from the first render, until the theme host releases it; Classic and Studio name none (A-19f1: no theme is
 * compiled in).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'
import { renderToString } from 'react-dom/server'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, STUDIO_HTML_CLASS, UI_STYLE_COOKIE, UI_STYLE_STORAGE_KEY } from '@/lib/ui-style'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import {
  HELLO_PACKAGE_ENTRY,
  INSTALLED_HELLO_STYLE,
  INSTALLED_PLAY_STYLE,
  PLAY_PACKAGE_ENTRY,
} from '@/lib/theme-host/__tests__/installed-entry-fixture'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'

/** A browser that has already been moved to the current defaults. */
const seenDefaults = () => window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)

function Probe() {
  const { style, setStyle, isStudio, activeTheme, releaseInstalledTheme } = useUiStyle()
  return (
    <div>
      <span data-testid="style">{style}</span>
      <span data-testid="studio">{String(isStudio)}</span>
      <span data-testid="theme">{activeTheme?.name ?? 'none'}</span>
      <span data-testid="ref">{activeTheme?.ref ?? 'none'}</span>
      <button onClick={() => setStyle('studio')}>go studio</button>
      <button onClick={() => setStyle('classic')}>go classic</button>
      <button onClick={() => setStyle(INSTALLED_PLAY_STYLE)}>go installed</button>
      <button onClick={() => releaseInstalledTheme(style)}>release</button>
      <button onClick={() => releaseInstalledTheme(null)}>runs again</button>
    </div>
  )
}

const INSTALLED_PLAY = installedThemeEntry(INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY)

beforeEach(() => {
  document.documentElement.classList.remove(STUDIO_HTML_CLASS)
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  tone.theme = 'dark'
  tone.setTheme.mockClear()
})
afterEach(cleanup)

describe('UiStyleProvider', () => {
  it('starts from the server-read style and reports it', () => {
    seenDefaults()
    render(<UiStyleProvider initialStyle="studio"><Probe /></UiStyleProvider>)
    expect(screen.getByTestId('style').textContent).toBe('studio')
    expect(screen.getByTestId('studio').textContent).toBe('true')
    expect(screen.getByTestId('theme').textContent).toBe('none')
  })

  it('switching writes the html class, the cookie and storage — and never the tone', () => {
    seenDefaults()
    render(<UiStyleProvider initialStyle="classic"><Probe /></UiStyleProvider>)
    act(() => { screen.getByText('go studio').click() })
    expect(document.documentElement.classList.contains(STUDIO_HTML_CLASS)).toBe(true)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=studio`)
    expect(window.localStorage.getItem(UI_STYLE_STORAGE_KEY)).toBe('studio')
    expect(tone.setTheme).not.toHaveBeenCalled()
    act(() => { screen.getByText('go classic').click() })
    expect(document.documentElement.classList.contains(STUDIO_HTML_CLASS)).toBe(false)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=classic`)
  })

  it('moves a browser that has not seen the current defaults to Studio + Dark, once', () => {
    render(<UiStyleProvider initialStyle="classic"><Probe /></UiStyleProvider>)
    expect(screen.getByTestId('style').textContent).toBe('studio')
    expect(document.documentElement.classList.contains(STUDIO_HTML_CLASS)).toBe(true)
    expect(tone.setTheme).toHaveBeenCalledWith('dark')
    expect(window.localStorage.getItem(APPEARANCE_DEFAULTS_KEY)).toBe(APPEARANCE_DEFAULTS_VERSION)
  })

  it('leaves a browser that has seen the defaults on its own choice', () => {
    seenDefaults()
    render(<UiStyleProvider initialStyle="classic"><Probe /></UiStyleProvider>)
    expect(screen.getByTestId('style').textContent).toBe('classic')
    expect(tone.setTheme).not.toHaveBeenCalled()
  })

  it('applies the defaults once per mount, even when next-themes hands it a new setTheme (PRD-260 M87)', () => {
    // A browser whose storage can't be read never records the defaults version.
    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('The operation is insecure.', 'SecurityError')
    })
    const mountSetTheme = tone.setTheme
    try {
      const { rerender } = render(<UiStyleProvider initialStyle="studio"><Probe /></UiStyleProvider>)
      expect(mountSetTheme).toHaveBeenCalledWith('dark')
      act(() => { screen.getByText('go classic').click() })
      // next-themes 0.2.1 rebuilds setTheme whenever forcedTheme changes, as a theme's look forcing its tone does.
      tone.setTheme = vi.fn()
      rerender(<UiStyleProvider initialStyle="studio"><Probe /></UiStyleProvider>)
      expect(screen.getByTestId('style').textContent).toBe('classic')
      expect(document.documentElement.classList.contains(STUDIO_HTML_CLASS)).toBe(false)
      expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=classic`)
      expect(tone.setTheme).not.toHaveBeenCalled()
    } finally {
      tone.setTheme = mountSetTheme
      getItem.mockRestore()
    }
  })
})

describe('PRD-261 W-A6b — an installed theme as the active theme (gap G1)', () => {
  it("names the entry the root layout built from the install's boot, on the server and the first client render", () => {
    seenDefaults()
    const tree = <UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE} installedTheme={INSTALLED_PLAY}><Probe /></UiStyleProvider>
    expect(renderToString(tree)).toContain(`>${INSTALLED_PLAY?.ref}<`)
    render(tree)
    expect(screen.getByTestId('theme').textContent).toBe('Play')
    expect(screen.getByTestId('ref').textContent).toBe(INSTALLED_PLAY?.ref)
    expect(screen.getByTestId('studio').textContent).toBe('true')
  })

  it('names none when the boot carried no entry (the theme may not run, or the boot failed)', () => {
    seenDefaults()
    render(<UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE}><Probe /></UiStyleProvider>)
    expect(screen.getByTestId('theme').textContent).toBe('none')
  })

  it("never names another install's entry", () => {
    seenDefaults()
    const other = 'theme:install:00000000-0000-4000-8000-000000000000' as const
    render(<UiStyleProvider initialStyle={other} installedTheme={INSTALLED_PLAY}><Probe /></UiStyleProvider>)
    expect(screen.getByTestId('theme').textContent).toBe('none')
  })

  it('names none once the theme host releases it, and the entry again once the host runs it', () => {
    seenDefaults()
    render(<UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE} installedTheme={INSTALLED_PLAY}><Probe /></UiStyleProvider>)
    act(() => { screen.getByText('release').click() })
    expect(screen.getByTestId('theme').textContent).toBe('none')
    act(() => { screen.getByText('runs again').click() })
    expect(screen.getByTestId('theme').textContent).toBe('Play')
  })

  it('keeps the entry the page loaded with when a server re-render (router.refresh) hands in another entry or none', () => {
    seenDefaults()
    const seen: Array<ThemeEntry | null> = []
    function Seen() {
      seen.push(useUiStyle().activeTheme)
      return null
    }
    const page = (entry: ThemeEntry | null) => (
      <UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE} installedTheme={entry}><Probe /><Seen /></UiStyleProvider>
    )
    const { rerender } = render(page(INSTALLED_PLAY))
    rerender(page(null))
    expect(screen.getByTestId('ref').textContent).toBe(INSTALLED_PLAY?.ref)
    rerender(page(installedThemeEntry(INSTALLED_HELLO_STYLE, HELLO_PACKAGE_ENTRY)))
    expect(screen.getByTestId('ref').textContent).toBe(INSTALLED_PLAY?.ref)
    rerender(page(installedThemeEntry(INSTALLED_PLAY_STYLE, PLAY_PACKAGE_ENTRY)))
    expect(screen.getByTestId('theme').textContent).toBe('Play')
    // The same object every render: the context value never changes identity for a refresh alone.
    expect(seen.filter((entry) => entry !== INSTALLED_PLAY)).toEqual([])
  })

  it('names none under Classic or Studio, whatever installed entry came with the page; a switch to the install reloads', () => {
    seenDefaults()
    const reloadPage = vi.fn()
    render(
      <UiStyleProvider initialStyle="studio" installedTheme={INSTALLED_PLAY} reloadPage={reloadPage}><Probe /></UiStyleProvider>,
    )
    expect(screen.getByTestId('theme').textContent).toBe('none')
    act(() => { screen.getByText('go classic').click() })
    expect(screen.getByTestId('theme').textContent).toBe('none')
    act(() => { screen.getByText('go installed').click() })
    expect(reloadPage).toHaveBeenCalledTimes(1)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${INSTALLED_PLAY_STYLE}`)
    expect(screen.getByTestId('style').textContent).toBe('classic')
    expect(screen.getByTestId('ref').textContent).toBe('none')
  })
})
