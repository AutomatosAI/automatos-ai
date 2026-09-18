/** PRD-244 W3 — the Style axis: html class + cookie + storage on switch; the legacy "studio" tone migrates once. */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, STUDIO_HTML_CLASS, UI_STYLE_COOKIE, UI_STYLE_STORAGE_KEY } from '@/lib/ui-style'

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
    </div>
  )
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
})
