/** PRD-244 W3 — the Style axis: html class + cookie + storage on switch; the legacy "studio" tone migrates once. */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { STUDIO_HTML_CLASS, TONE_STORAGE_KEY, UI_STYLE_COOKIE, UI_STYLE_STORAGE_KEY } from '@/lib/ui-style'

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
    render(<UiStyleProvider initialStyle="studio"><Probe /></UiStyleProvider>)
    expect(screen.getByTestId('style').textContent).toBe('studio')
    expect(screen.getByTestId('studio').textContent).toBe('true')
  })

  it('switching writes the html class, the cookie and storage — and never the tone', () => {
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

  it('migrates a browser whose tone key still says "studio": Studio style + System tone, once', () => {
    window.localStorage.setItem(TONE_STORAGE_KEY, 'studio')
    render(<UiStyleProvider initialStyle="classic"><Probe /></UiStyleProvider>)
    expect(screen.getByTestId('style').textContent).toBe('studio')
    expect(document.documentElement.classList.contains(STUDIO_HTML_CLASS)).toBe(true)
    expect(tone.setTheme).toHaveBeenCalledWith('system')
  })
})
