/**
 * PRD-244 W3 — the Studio detector reads the Style axis (Classic | Studio),
 * never the tone. It is stable across SSR (the provider is seeded from the
 * cookie) and Classic outside the provider.
 *
 * PRD-260 P260-FX-25 (M69) — the one width source for a theme:
 * 'pending' on the server, where no width is known, and on a browser mount's
 * very first render already the theme's ('theme', >= 1024 px) or 'fallback'
 * (below it, or no theme), never a guess the next render takes back; it follows
 * the window across 1024 px and reads the band useIsTabletOrBelow reads.
 *
 * PRD-261 A-02a — that source is useThemeLayout, driven by the active theme's
 * minWidth. The theme here is an installed one (the neutral fixture theme,
 * lib/theme-host/__tests__/fixture-theme-entry.ts), named by the entry the root
 * layout built from its boot: the host draws an installed theme from 1024 px
 * (INSTALLED_FRAME_MIN_WIDTH_PX).
 */
import { afterEach, beforeEach, describe, it, expect, vi } from 'vitest'
import { act, cleanup, render, renderHook, screen } from '@testing-library/react'
import { renderToString } from 'react-dom/server'
import type { ReactNode } from 'react'

vi.mock('next-themes', () => ({ useTheme: () => ({ theme: 'dark', resolvedTheme: 'dark', setTheme: vi.fn() }) }))

import { UiStyleProvider } from '@/contexts/ui-style-context'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import * as hook from '@/hooks/use-studio-theme'
import { FIXTURE_THEME_ENTRY, FIXTURE_THEME_REF, FIXTURE_THEME_STYLE as THEME } from '@/lib/theme-host/__tests__/fixture-theme-entry'
import { INSTALLED_FRAME_MIN_WIDTH_PX as THEME_MIN_WIDTH_PX } from '@/lib/theme-host/installed-entry'
import type { ThemeEntry, ThemeLayout } from '@/lib/theme-host/registry-types'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, type UiStyle } from '@/lib/ui-style'

import { LAPTOP_WIDTH_PX, PHONE_WIDTH_PX, stubWindowWidth, type WindowWidth } from './window-width'

/** The entry the root layout builds from the boot: the fixture's under its style, none under Classic and Studio. */
const entryOf = (style: UiStyle): ThemeEntry | null => (style === THEME ? FIXTURE_THEME_ENTRY : null)

const wrap = (style: UiStyle) => ({ children }: { children: ReactNode }) => (
  <UiStyleProvider initialStyle={style} installedTheme={entryOf(style)}>{children}</UiStyleProvider>
)

describe('useIsStudio', () => {
  it('is true when the style is Studio, whatever the tone', () => {
    const { result } = renderHook(() => hook.useIsStudio(), { wrapper: wrap('studio') })
    expect(result.current).toBe(true)
  })

  it('is false for the Classic style', () => {
    const { result } = renderHook(() => hook.useIsStudio(), { wrapper: wrap('classic') })
    expect(result.current).toBe(false)
  })

  it('is Classic outside the provider', () => {
    const { result } = renderHook(() => hook.useIsStudio())
    expect(result.current).toBe(false)
  })

  it('no longer exports the URL flag hook', () => {
    expect(Object.keys(hook)).not.toContain('useStudioThemeFlag')
  })

  it("no longer exports Play's own hooks: the axis names no theme (PRD-261 A-02a)", () => {
    expect(Object.keys(hook)).not.toContain('useIsPlay')
    expect(Object.keys(hook)).not.toContain('usePlayLayout')
    expect(Object.keys(hook)).toEqual(expect.arrayContaining(['useIsStudio', 'useActiveTheme', 'useThemeLayout']))
  })

  it('is true under an installed theme drawn on Studio, which useActiveTheme names (PRD-260 D2)', () => {
    // A browser already moved to the current defaults, so the provider keeps the theme's style.
    window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
    try {
      const { result } = renderHook(() => ({ studio: hook.useIsStudio(), theme: hook.useActiveTheme()?.ref }), { wrapper: wrap(THEME) })
      expect(result.current).toEqual({ studio: true, theme: FIXTURE_THEME_REF })
    } finally {
      window.localStorage.clear()
    }
  })
})

/** Records every layout useThemeLayout gives, render by render. */
function LayoutProbe({ seen }: { readonly seen: ThemeLayout[] }) {
  const layout = hook.useThemeLayout()
  seen.push(layout)
  return <p data-testid="layout">{layout}</p>
}

/** The theme's layout beside useIsTabletOrBelow's answer, under the same window. */
function BandProbe() {
  const layout = hook.useThemeLayout()
  const tabletOrBelow = useIsTabletOrBelow()
  return <p data-testid="band">{`${layout} · ${tabletOrBelow ? 'tablet or below' : 'wider'}`}</p>
}

/** The distinct media queries asked while mounting, at the theme's narrowest fallback width. */
function queriesAsked(mount: () => unknown): ReadonlySet<string> {
  const width = stubWindowWidth(THEME_MIN_WIDTH_PX - 1)
  try {
    mount()
    return new Set(width.asked)
  } finally {
    cleanup()
    width.restore()
  }
}

function layoutTree(style: UiStyle, seen: ThemeLayout[]) {
  return (
    <UiStyleProvider initialStyle={style} installedTheme={entryOf(style)}>
      <LayoutProbe seen={seen} />
    </UiStyleProvider>
  )
}

describe('useThemeLayout under an installed theme (P260-FX-25, M69)', () => {
  let window_: WindowWidth | null = null

  beforeEach(() => {
    // A browser already moved to the current defaults, so the provider keeps the given style.
    window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  })

  afterEach(() => {
    window_?.restore()
    window_ = null
    window.localStorage.clear()
  })

  it("is 'pending' on the server under the theme: no width is known there", () => {
    const seen: ThemeLayout[] = []
    expect(renderToString(layoutTree(THEME, seen))).toContain('pending')
    expect(seen).toEqual(['pending'])
  })

  it("is 'fallback' on a phone from the very first render, never 'theme' for a render", () => {
    window_ = stubWindowWidth(PHONE_WIDTH_PX)
    const seen: ThemeLayout[] = []
    render(layoutTree(THEME, seen))
    expect(screen.getByTestId('layout')).toHaveTextContent('fallback')
    expect(seen.length).toBeGreaterThan(0)
    expect(new Set(seen)).toEqual(new Set(['fallback']))
  })

  it("is 'theme' on a laptop from the very first render", () => {
    window_ = stubWindowWidth(LAPTOP_WIDTH_PX)
    const seen: ThemeLayout[] = []
    render(layoutTree(THEME, seen))
    expect(screen.getByTestId('layout')).toHaveTextContent('theme')
    expect(new Set(seen)).toEqual(new Set(['theme']))
  })

  it.each<UiStyle>(['classic', 'studio'])("is 'fallback' under %s at any width, on the server too", (style) => {
    expect(renderToString(layoutTree(style, []))).toContain('fallback')
    window_ = stubWindowWidth(LAPTOP_WIDTH_PX)
    const seen: ThemeLayout[] = []
    render(layoutTree(style, seen))
    expect(new Set(seen)).toEqual(new Set(['fallback']))
  })

  it('follows the window across 1024 px', () => {
    window_ = stubWindowWidth(LAPTOP_WIDTH_PX)
    render(layoutTree(THEME, []))
    expect(screen.getByTestId('layout')).toHaveTextContent('theme')
    act(() => window_?.resize(PHONE_WIDTH_PX))
    expect(screen.getByTestId('layout')).toHaveTextContent('fallback')
    act(() => window_?.resize(LAPTOP_WIDTH_PX))
    expect(screen.getByTestId('layout')).toHaveTextContent('theme')
  })

  it('asks the query useIsTabletOrBelow asks, and they agree on each side of 1024 px', () => {
    const tabletQueries = queriesAsked(() => renderHook(() => useIsTabletOrBelow()))
    const themeQueries = queriesAsked(() => render(layoutTree(THEME, [])))
    expect(themeQueries).toEqual(tabletQueries)

    window_ = stubWindowWidth(THEME_MIN_WIDTH_PX - 1)
    render(
      <UiStyleProvider initialStyle={THEME} installedTheme={FIXTURE_THEME_ENTRY}>
        <BandProbe />
      </UiStyleProvider>,
    )
    expect(screen.getByTestId('band')).toHaveTextContent('fallback · tablet or below')
    act(() => window_?.resize(THEME_MIN_WIDTH_PX))
    expect(screen.getByTestId('band')).toHaveTextContent('theme · wider')
  })

  it('treats a browser without matchMedia as a wide one', () => {
    window_ = stubWindowWidth(LAPTOP_WIDTH_PX)
    Reflect.deleteProperty(window, 'matchMedia')
    render(layoutTree(THEME, []))
    expect(screen.getByTestId('layout')).toHaveTextContent('theme')
  })
})
