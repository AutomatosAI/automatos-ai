/**
 * PRD-244 W3 — the Style axis: html class + cookie + storage on switch; the legacy "studio" tone migrates once.
 * PRD-260 M87 — once per mount: a new setTheme from next-themes never re-applies the defaults.
 * PRD-261 A-02a — the context names the active theme (its registry entry) and has no isPlay; Play is builtin Play.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { BUILTIN_PLAY_HTML_CLASS as PLAY_HTML_CLASS, BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, STUDIO_HTML_CLASS, UI_STYLE_COOKIE, UI_STYLE_STORAGE_KEY } from '@/lib/ui-style'

/** A browser that has already been moved to the current defaults. */
const seenDefaults = () => window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)

function Probe() {
  const { style, setStyle, isStudio, activeTheme } = useUiStyle()
  return (
    <div>
      <span data-testid="style">{style}</span>
      <span data-testid="studio">{String(isStudio)}</span>
      <span data-testid="theme">{activeTheme?.name ?? 'none'}</span>
      <button onClick={() => setStyle('studio')}>go studio</button>
      <button onClick={() => setStyle('classic')}>go classic</button>
      <button onClick={() => setStyle(BUILTIN_PLAY_STYLE)}>go play</button>
    </div>
  )
}

beforeEach(() => {
  document.documentElement.classList.remove(STUDIO_HTML_CLASS, PLAY_HTML_CLASS)
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

  it('names builtin Play as the active theme under it, Studio still rendering beneath (PRD-260 D2)', () => {
    seenDefaults()
    render(<UiStyleProvider initialStyle={BUILTIN_PLAY_STYLE}><Probe /></UiStyleProvider>)
    expect(screen.getByTestId('style').textContent).toBe(BUILTIN_PLAY_STYLE)
    expect(screen.getByTestId('studio').textContent).toBe('true')
    expect(screen.getByTestId('theme').textContent).toBe('Play')
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
      act(() => { screen.getByText('go play').click() })
      // next-themes 0.2.1 rebuilds setTheme whenever forcedTheme changes, as Play's look forcing its tone does.
      tone.setTheme = vi.fn()
      rerender(<UiStyleProvider initialStyle="studio"><Probe /></UiStyleProvider>)
      expect(screen.getByTestId('style').textContent).toBe(BUILTIN_PLAY_STYLE)
      expect(document.documentElement.classList.contains(PLAY_HTML_CLASS)).toBe(true)
      expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${BUILTIN_PLAY_STYLE}`)
      expect(tone.setTheme).not.toHaveBeenCalled()
    } finally {
      tone.setTheme = mountSetTheme
      getItem.mockRestore()
    }
  })
})
