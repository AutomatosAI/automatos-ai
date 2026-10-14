/**
 * PRD-260 D1/D2/K9 — the theme menu reads Classic · Studio · Play. Choosing
 * Play writes the cookie and puts both classes on <html>; leaving it takes
 * 'play' off (and 'studio' too for Classic). "Customise look…" shows only
 * while Play is the style. useIsStudio stays true under Play; useIsPlay is
 * true only there.
 */
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from 'vitest'
import { render, renderHook, screen, cleanup, waitFor, fireEvent, act } from '@testing-library/react'
import type { ReactNode } from 'react'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))
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

import { UiStyleProvider } from '@/contexts/ui-style-context'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { useIsPlay, useIsStudio } from '@/hooks/use-studio-theme'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  PLAY_HTML_CLASS,
  STUDIO_HTML_CLASS,
  UI_STYLE_COOKIE,
  type UiStyle,
} from '@/lib/ui-style'

const html = () => document.documentElement.classList

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

beforeAll(() => {
  // jsdom has no layout; Radix's popper may observe the menu's size.
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
})

beforeEach(() => {
  html().remove(STUDIO_HTML_CLASS, PLAY_HTML_CLASS)
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  tone.setTheme.mockClear()
})
afterEach(cleanup)

describe('PRD-260 — the theme menu', () => {
  it('lists Classic, Studio, Play in the Style group, then the tones', async () => {
    renderMenu('studio')
    await openMenu()
    const items = screen.getAllByRole('menuitemradio').map((el) => el.textContent?.trim())
    expect(items).toEqual(['Classic', 'Studio', 'Play', 'Light', 'Dark', 'System'])
  })

  it('choosing Play writes the cookie and puts studio and play on <html>, without touching the tone', async () => {
    renderMenu('studio')
    await openMenu()
    choose('Play')
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=play`)
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
    renderMenu('play')
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

describe('PRD-260 D2 — useIsStudio and useIsPlay under each style', () => {
  const wrap = (style: UiStyle) => ({ children }: { children: ReactNode }) => (
    <UiStyleProvider initialStyle={style}>{children}</UiStyleProvider>
  )
  const both = () => ({ studio: useIsStudio(), play: useIsPlay() })
  const cases: Array<[UiStyle, { studio: boolean; play: boolean }]> = [
    ['classic', { studio: false, play: false }],
    ['studio', { studio: true, play: false }],
    ['play', { studio: true, play: true }],
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
