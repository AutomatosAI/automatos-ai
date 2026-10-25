/**
 * PRD-244 W3 — the Style axis: html class + cookie + storage on switch; the legacy "studio" tone migrates once.
 * PRD-260 M87 — once per mount: a new setTheme from next-themes never re-applies the defaults.
 * PRD-261 W-A8 — a browser whose storage cannot keep the defaults version is never moved to them: its stored
 * choice (the style cookie the server painted from) stays, Classic or an installed theme alike.
 * PRD-261 A-02a — the context names the active theme (its registry entry) and has no isPlay.
 * PRD-261 W-A6b (gap G1) — under an installed theme's style it names the entry the root layout built from the
 * boot, from the first render, until the theme host releases it; Classic and Studio name none (A-19f1: no theme is
 * compiled in).
 * PRD-261 A-08c — a page whose boot carried no entry (the server's boot fetch missed its budget or was refused) names
 * the entry of the boot the browser fetched once it answers: the theme, with its slots.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act, waitFor } from '@testing-library/react'
import { renderToString } from 'react-dom/server'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { useActiveTheme } from '@/hooks/use-studio-theme'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, STUDIO_HTML_CLASS, UI_STYLE_COOKIE, UI_STYLE_STORAGE_KEY } from '@/lib/ui-style'
import { baseThemeBoot, type ThemeBoot } from '@/lib/theme-host/boot'
import type { InstalledBootSource } from '@/lib/theme-host/boot-client'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import { installIdOfStyle } from '@/lib/theme-host/registry'
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

/** What the browser's boot fetch answers: the install's boot with the theme's entry (or the base boot, with none). */
function bootSourceAnswering(boot: ThemeBoot | null) {
  return vi.fn<InstalledBootSource>(async () => boot)
}

/** useActiveTheme's view, as the theme's frame, pages and look runtime read it. */
function ActiveThemeProbe() {
  const theme = useActiveTheme()
  return <span data-testid="slots">{theme === null ? 'none' : theme.slots.join(' ')}</span>
}

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
    const mountSetTheme = tone.setTheme
    try {
      const { rerender } = render(<UiStyleProvider initialStyle="classic"><Probe /></UiStyleProvider>)
      expect(mountSetTheme).toHaveBeenCalledWith('dark')
      expect(screen.getByTestId('style').textContent).toBe('studio')
      act(() => { screen.getByText('go classic').click() })
      // next-themes 0.2.1 rebuilds setTheme whenever forcedTheme changes, as a theme's look forcing its tone does.
      tone.setTheme = vi.fn()
      rerender(<UiStyleProvider initialStyle="classic"><Probe /></UiStyleProvider>)
      expect(screen.getByTestId('style').textContent).toBe('classic')
      expect(document.documentElement.classList.contains(STUDIO_HTML_CLASS)).toBe(false)
      expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=classic`)
      expect(tone.setTheme).not.toHaveBeenCalled()
    } finally {
      tone.setTheme = mountSetTheme
    }
  })
})

/** Storage that refuses: reading throws (blocked storage), or writing throws (Safari's old private mode). */
type StorageMethod = 'getItem' | 'setItem'
const REFUSING_STORAGE: ReadonlyArray<readonly [string, StorageMethod]> = [['reading', 'getItem'], ['writing', 'setItem']]

function refuseStorage(method: StorageMethod): { mockRestore: () => void } {
  const refuse = (): never => {
    throw new DOMException('The operation is insecure.', 'SecurityError')
  }
  return method === 'getItem'
    ? vi.spyOn(Storage.prototype, 'getItem').mockImplementation(refuse)
    : vi.spyOn(Storage.prototype, 'setItem').mockImplementation(refuse)
}

describe('PRD-261 W-A8 — a browser whose storage cannot keep the defaults version keeps its stored choice', () => {
  it.each(REFUSING_STORAGE)('%s storage throws: the installed theme the server painted from the cookie stays, cookie and all', (_, method) => {
    document.cookie = `${UI_STYLE_COOKIE}=${INSTALLED_PLAY_STYLE}; Path=/`
    const refused = refuseStorage(method)
    try {
      render(<UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE} installedTheme={INSTALLED_PLAY}><Probe /></UiStyleProvider>)
      expect(screen.getByTestId('style').textContent).toBe(INSTALLED_PLAY_STYLE)
      expect(screen.getByTestId('theme').textContent).toBe('Play')
      expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${INSTALLED_PLAY_STYLE}`)
      expect(tone.setTheme).not.toHaveBeenCalled()
    } finally {
      refused.mockRestore()
    }
  })

  it.each(REFUSING_STORAGE)('%s storage throws: Classic stays Classic, and a switch still works', (_, method) => {
    document.cookie = `${UI_STYLE_COOKIE}=classic; Path=/`
    const refused = refuseStorage(method)
    try {
      render(<UiStyleProvider initialStyle="classic"><Probe /></UiStyleProvider>)
      expect(screen.getByTestId('style').textContent).toBe('classic')
      expect(document.documentElement.classList.contains(STUDIO_HTML_CLASS)).toBe(false)
      expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=classic`)
      act(() => { screen.getByText('go studio').click() })
      expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=studio`)
      expect(tone.setTheme).not.toHaveBeenCalled()
    } finally {
      refused.mockRestore()
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

  it("names none when neither the page's boot nor the browser's fetch of it carries an entry (the theme may not run)", async () => {
    seenDefaults()
    const bootSource = bootSourceAnswering(baseThemeBoot(INSTALLED_PLAY_STYLE))
    render(<UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE} bootSource={bootSource}><Probe /></UiStyleProvider>)
    await waitFor(() => expect(bootSource).toHaveBeenCalledTimes(1))
    await act(async () => undefined)
    expect(screen.getByTestId('theme').textContent).toBe('none')
  })

  it("a page whose boot carried no entry names the entry of the boot the browser fetched, with its slots", async () => {
    seenDefaults()
    const bootSource = bootSourceAnswering({ ...baseThemeBoot(INSTALLED_PLAY_STYLE), entry: PLAY_PACKAGE_ENTRY })
    const tree = <UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE} bootSource={bootSource}><Probe /><ActiveThemeProbe /></UiStyleProvider>
    // The server render (and so the first client render) names none: the page carried the base boot.
    expect(renderToString(tree)).toContain('>none<')
    render(tree)
    await waitFor(() => expect(screen.getByTestId('theme').textContent).toBe('Play'))
    expect(screen.getByTestId('ref').textContent).toBe(INSTALLED_PLAY?.ref)
    expect(screen.getByTestId('slots').textContent).toBe(PLAY_PACKAGE_ENTRY.slots.join(' '))
    expect(bootSource.mock.calls).toEqual([[installIdOfStyle(INSTALLED_PLAY_STYLE)]])
  })

  it('in safe mode a page whose boot carried no entry asks for none (safe mode loads no theme code)', async () => {
    seenDefaults()
    window.history.replaceState(null, '', '/chat?safe_mode=themes')
    try {
      const bootSource = bootSourceAnswering({ ...baseThemeBoot(INSTALLED_PLAY_STYLE), entry: PLAY_PACKAGE_ENTRY })
      render(<UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE} bootSource={bootSource}><Probe /></UiStyleProvider>)
      await act(async () => undefined)
      expect(bootSource).not.toHaveBeenCalled()
      expect(screen.getByTestId('theme').textContent).toBe('none')
    } finally {
      window.history.replaceState(null, '', '/')
      window.sessionStorage.clear()
    }
  })

  it('a page whose boot carried the entry never asks for it again', async () => {
    seenDefaults()
    const bootSource = bootSourceAnswering(null)
    render(<UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE} installedTheme={INSTALLED_PLAY} bootSource={bootSource}><Probe /></UiStyleProvider>)
    await act(async () => undefined)
    expect(screen.getByTestId('theme').textContent).toBe('Play')
    expect(bootSource).not.toHaveBeenCalled()
  })

  it('the host still releases a theme whose entry the browser fetched', async () => {
    seenDefaults()
    const bootSource = bootSourceAnswering({ ...baseThemeBoot(INSTALLED_PLAY_STYLE), entry: PLAY_PACKAGE_ENTRY })
    render(<UiStyleProvider initialStyle={INSTALLED_PLAY_STYLE} bootSource={bootSource}><Probe /></UiStyleProvider>)
    await waitFor(() => expect(screen.getByTestId('theme').textContent).toBe('Play'))
    act(() => { screen.getByText('release').click() })
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
