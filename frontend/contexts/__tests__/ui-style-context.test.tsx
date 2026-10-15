/**
 * PRD-244 W3 — the Style axis: html class + cookie + storage on switch; the legacy "studio" tone migrates once.
 * PRD-260 M87 — once per mount: a new setTheme from next-themes never re-applies the defaults.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, PLAY_HTML_CLASS, STUDIO_HTML_CLASS, UI_STYLE_COOKIE, UI_STYLE_STORAGE_KEY } from '@/lib/ui-style'

/** A browser that has already been moved to the current defaults. */
const seenDefaults = () => window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)

function Probe() {
  const { style, setStyle, isStudio } = useUiStyle()
  return (
    <div>
      <span data-testid="style">{style}</span>
      <span data-testid="studio">{String(isStudio)}</span>
      <button onClick={() => setStyle('studio')}>go studio</button>
      <button onClick={() => setStyle('classic')}>go classic</button>
      <button onClick={() => setStyle('play')}>go play</button>
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
      expect(screen.getByTestId('style').textContent).toBe('play')
      expect(document.documentElement.classList.contains(PLAY_HTML_CLASS)).toBe(true)
      expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=play`)
      expect(tone.setTheme).not.toHaveBeenCalled()
    } finally {
      tone.setTheme = mountSetTheme
      getItem.mockRestore()
    }
  })
})
