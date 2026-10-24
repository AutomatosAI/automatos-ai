/**
 * PRD-261 A-02a — the Style axis takes theme refs: Classic | Studio | `theme:<ref>`. No theme is
 * compiled into the app (A-19f1), so the axis lists Classic and Studio only and the registry finds
 * only the installed entries it is given; an installed theme's style is accepted only in its exact
 * shape; Studio's layout renders under a theme (drawn on Studio, the default base); the theme
 * layout follows the active entry's minWidth; and a switch to an installed theme writes the cookie
 * and reloads instead of switching in place (PA-13), while Classic and Studio switch in place.
 */
import { createElement, type ReactNode } from 'react'
import { act, cleanup, render, renderHook, screen } from '@testing-library/react'
import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('next-themes', () => ({ useTheme: () => ({ theme: 'dark', resolvedTheme: 'dark', setTheme: vi.fn() }) }))

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { useActiveTheme, useThemeLayout } from '@/hooks/use-studio-theme'
import { stubWindowWidth, type WindowWidth } from '@/hooks/__tests__/window-width'
import { findThemeEntry, isInstalledThemeStyle, themeRefOf, themeStyleOf } from '@/lib/theme-host/registry'
import type { ThemeEntry, ThemeLayout, ThemeRef } from '@/lib/theme-host/registry-types'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  DEFAULT_UI_STYLE,
  STUDIO_HTML_CLASS,
  STYLE_HTML_CLASS_NAMES,
  UI_STYLE_COOKIE,
  UI_STYLE_STORAGE_KEY,
  UI_STYLES,
  isUiStyle,
  parseUiStyle,
  rendersStudio,
  styleHtmlClasses,
  type UiStyle,
} from '@/lib/ui-style'

import { FIXTURE_THEME_ENTRY, FIXTURE_THEME_HTML_CLASS, FIXTURE_THEME_STYLE } from './fixture-theme-entry'

const INSTALL_ID = '0f8c2b6e-4d1a-4c3b-9e7f-1a2b3c4d5e6f'
const INSTALL_REF: ThemeRef = `install:${INSTALL_ID}`
const INSTALL_STYLE: UiStyle = `theme:install:${INSTALL_ID}`
const OTHER_INSTALL_STYLE: UiStyle = 'theme:install:5d0f2a7e-1c4b-4e9a-b3d6-8f1a2c3e4b5d'
/** A theme style in no installed theme's exact shape: it names no theme this build knows. */
const UNKNOWN_THEME_STYLE: UiStyle = 'theme:install:not-a-uuid'
/** The widths the plan names: a laptop above an installed theme's 1024 px, a small window below it. */
const WIDE_PX = 1440
const NARROW_PX = 800
/** An entry's own minWidth, other than the installed frame's 1024 px, between NARROW_PX and WIDE_PX. */
const OWN_MIN_WIDTH_PX = 1200

/** An installed theme as A-02c will hand its entry to the registry. */
const INSTALLED_CLASSIC: ThemeEntry = {
  ...FIXTURE_THEME_ENTRY,
  ref: INSTALL_REF,
  name: 'Installed',
  base: 'classic',
  htmlClass: 'installed-theme',
}

describe('PRD-261 A-02a — the registry holds no compiled-in theme', () => {
  it('names an installed theme on the axis as theme:install:<id>, and maps a ref to its style and back', () => {
    expect(themeStyleOf(INSTALL_REF)).toBe(INSTALL_STYLE)
    expect(themeRefOf(INSTALL_STYLE)).toBe(INSTALL_REF)
    expect(themeRefOf('studio')).toBeNull()
    expect(themeRefOf('classic')).toBeNull()
  })

  it('lists Classic and Studio only: every theme is installed', () => {
    expect(UI_STYLES).toEqual(['classic', 'studio'])
  })

  it("puts only Studio's class on the axis's class list (PA-8): an installed theme's own comes with its boot", () => {
    expect(STYLE_HTML_CLASS_NAMES).toEqual([STUDIO_HTML_CLASS])
    expect(STYLE_HTML_CLASS_NAMES).not.toContain(FIXTURE_THEME_HTML_CLASS)
  })

  it('finds an installed theme it is given, and none otherwise', () => {
    expect(findThemeEntry('studio')).toBeNull()
    expect(findThemeEntry('classic')).toBeNull()
    expect(findThemeEntry(UNKNOWN_THEME_STYLE)).toBeNull()
    expect(findThemeEntry(INSTALL_STYLE)).toBeNull()
    expect(findThemeEntry(INSTALL_STYLE, [INSTALLED_CLASSIC])).toBe(INSTALLED_CLASSIC)
    expect(findThemeEntry(OTHER_INSTALL_STYLE, [INSTALLED_CLASSIC])).toBeNull()
    expect(findThemeEntry('studio', [INSTALLED_CLASSIC])).toBeNull()
    expect(findThemeEntry(FIXTURE_THEME_STYLE, [INSTALLED_CLASSIC, FIXTURE_THEME_ENTRY])).toBe(FIXTURE_THEME_ENTRY)
  })
})

describe('PRD-261 A-02a — parseUiStyle', () => {
  it('leaves classic and studio as they were', () => {
    expect(parseUiStyle('classic')).toBe('classic')
    expect(parseUiStyle('studio')).toBe('studio')
  })

  it("accepts an installed theme's style only in its exact shape", () => {
    expect(isInstalledThemeStyle(INSTALL_STYLE)).toBe(true)
    expect(parseUiStyle(INSTALL_STYLE)).toBe(INSTALL_STYLE)
    expect(isUiStyle(INSTALL_STYLE)).toBe(true)
  })

  it('falls back to Studio for a tampered, unknown or missing value', () => {
    const tampered = [
      'theme:install:x',
      `theme:install:${INSTALL_ID.toUpperCase()}`,
      `theme:install:${INSTALL_ID}x`,
      ` ${INSTALL_STYLE}`,
      'theme:install:../../../../api/admin/x',
      UNKNOWN_THEME_STYLE,
      'theme:other',
      'STUDIO',
      'toString',
      '__proto__',
      'constructor',
      'hasOwnProperty',
      '',
    ]
    for (const raw of tampered) expect(parseUiStyle(raw), JSON.stringify(raw)).toBe(DEFAULT_UI_STYLE)
    expect(parseUiStyle(null)).toBe(DEFAULT_UI_STYLE)
    expect(parseUiStyle(undefined)).toBe(DEFAULT_UI_STYLE)
    expect(DEFAULT_UI_STYLE).toBe('studio')
  })
})

describe('PRD-261 A-02a — rendersStudio and the <html> classes follow the base', () => {
  it('renders Studio for Studio and a theme drawn on Studio, never for Classic', () => {
    expect(rendersStudio('studio')).toBe(true)
    expect(rendersStudio(FIXTURE_THEME_STYLE)).toBe(true)
    expect(rendersStudio('classic')).toBe(false)
  })

  it('draws an installed theme the registry does not know yet on Studio, the default', () => {
    expect(rendersStudio(INSTALL_STYLE)).toBe(true)
    expect(styleHtmlClasses(INSTALL_STYLE)).toEqual([STUDIO_HTML_CLASS])
  })

  it('gives Classic no class and Studio its own', () => {
    expect(styleHtmlClasses('classic')).toEqual([])
    expect(styleHtmlClasses('studio')).toEqual([STUDIO_HTML_CLASS])
  })
})

const wrapper = (style: UiStyle, reloadPage?: () => void, installedTheme: ThemeEntry | null = null) =>
  function Wrapper({ children }: { readonly children: ReactNode }) {
    return createElement(UiStyleProvider, { initialStyle: style, installedTheme, reloadPage, children })
  }

function LayoutProbe({ seen }: { readonly seen: ThemeLayout[] }) {
  const layout = useThemeLayout()
  seen.push(layout)
  return createElement('p', { 'data-testid': 'layout' }, layout)
}

/** The layout probe under `style`; an installed style runs `entry`, the entry its boot carried. */
function layoutTree(style: UiStyle, seen: ThemeLayout[], entry: ThemeEntry | null = null) {
  return createElement(UiStyleProvider, { initialStyle: style, installedTheme: entry, children: createElement(LayoutProbe, { seen }) })
}

describe('PRD-261 A-02a — useActiveTheme and useThemeLayout', () => {
  let window_: WindowWidth | null = null

  beforeEach(() => {
    // A browser already moved to the current defaults, so the provider keeps the given style.
    window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  })

  afterEach(() => {
    cleanup()
    window_?.restore()
    window_ = null
    window.localStorage.clear()
  })

  it("is the installed theme's entry under its style, and null under Classic, Studio and outside the provider", () => {
    const active = (style: UiStyle, entry: ThemeEntry | null = null) =>
      renderHook(() => useActiveTheme(), { wrapper: wrapper(style, undefined, entry) }).result.current
    expect(active(FIXTURE_THEME_STYLE, FIXTURE_THEME_ENTRY)).toBe(FIXTURE_THEME_ENTRY)
    expect(active(FIXTURE_THEME_STYLE)).toBeNull()
    expect(active('studio', FIXTURE_THEME_ENTRY)).toBeNull()
    expect(active('studio')).toBeNull()
    expect(active('classic')).toBeNull()
    expect(renderHook(() => useActiveTheme()).result.current).toBeNull()
  })

  it("is 'pending' on the server under a theme: no width is known there", () => {
    const seen: ThemeLayout[] = []
    expect(renderToString(layoutTree(FIXTURE_THEME_STYLE, seen, FIXTURE_THEME_ENTRY))).toContain('pending')
    expect(seen).toEqual(['pending'])
  })

  it(`is 'theme' at ${WIDE_PX} px and 'fallback' at ${NARROW_PX} px, from the first render`, () => {
    const wideWindow = stubWindowWidth(WIDE_PX)
    const wide: ThemeLayout[] = []
    try {
      render(layoutTree(FIXTURE_THEME_STYLE, wide, FIXTURE_THEME_ENTRY))
    } finally {
      cleanup()
      wideWindow.restore()
    }
    expect(new Set(wide)).toEqual(new Set(['theme']))

    window_ = stubWindowWidth(NARROW_PX)
    const narrow: ThemeLayout[] = []
    render(layoutTree(FIXTURE_THEME_STYLE, narrow, FIXTURE_THEME_ENTRY))
    expect(new Set(narrow)).toEqual(new Set(['fallback']))
  })

  it(`reads the entry's minWidth: an entry of ${OWN_MIN_WIDTH_PX} px switches at ${OWN_MIN_WIDTH_PX} px`, () => {
    const entry: ThemeEntry = { ...FIXTURE_THEME_ENTRY, minWidth: OWN_MIN_WIDTH_PX }
    window_ = stubWindowWidth(OWN_MIN_WIDTH_PX)
    render(layoutTree(FIXTURE_THEME_STYLE, [], entry))
    expect(screen.getByTestId('layout')).toHaveTextContent('theme')
    act(() => window_?.resize(OWN_MIN_WIDTH_PX - 1))
    expect(screen.getByTestId('layout')).toHaveTextContent('fallback')
  })

  it.each<UiStyle>(['classic', 'studio', INSTALL_STYLE])("is 'fallback' under %s, on the server too", (style) => {
    expect(renderToString(layoutTree(style, []))).toContain('fallback')
    window_ = stubWindowWidth(WIDE_PX)
    const seen: ThemeLayout[] = []
    render(layoutTree(style, seen))
    expect(new Set(seen)).toEqual(new Set(['fallback']))
  })
})

interface MountedStyle {
  readonly setStyle: (next: UiStyle) => void
  readonly current: () => UiStyle
}

/** The provider's setStyle, held for the test to call, and the style it reports now. */
function mountStyle(style: UiStyle, reloadPage: () => void): MountedStyle {
  const { result } = renderHook(() => useUiStyle(), { wrapper: wrapper(style, reloadPage) })
  return {
    setStyle: (next) => act(() => result.current.setStyle(next)),
    current: () => result.current.style,
  }
}

describe('PRD-261 A-02a — a switch to an installed theme reloads (PA-13)', () => {
  const studioOnHtml = () => document.documentElement.classList.contains(STUDIO_HTML_CLASS)

  beforeEach(() => {
    window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
    document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
    document.documentElement.classList.remove(...STYLE_HTML_CLASS_NAMES)
  })

  afterEach(() => {
    cleanup()
    window.localStorage.clear()
    document.documentElement.classList.remove(...STYLE_HTML_CLASS_NAMES)
  })

  it('writes the cookie and calls the reload seam, and switches nothing in place', () => {
    const reloadPage = vi.fn()
    const mounted = mountStyle('studio', reloadPage)
    mounted.setStyle(INSTALL_STYLE)
    expect(reloadPage).toHaveBeenCalledTimes(1)
    expect(mounted.current()).toBe('studio')
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${INSTALL_STYLE}`)
    expect(window.localStorage.getItem(UI_STYLE_STORAGE_KEY)).toBe(INSTALL_STYLE)
    // In place, the installed style (drawn on Studio) would have put Studio's class on <html>.
    expect(studioOnHtml()).toBe(false)
  })

  it('switches to Classic and to Studio in place, with no reload', () => {
    const reloadPage = vi.fn()
    const mounted = mountStyle('studio', reloadPage)
    mounted.setStyle('classic')
    expect(mounted.current()).toBe('classic')
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=classic`)
    expect(studioOnHtml()).toBe(false)
    mounted.setStyle('studio')
    expect(mounted.current()).toBe('studio')
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=studio`)
    expect(window.localStorage.getItem(UI_STYLE_STORAGE_KEY)).toBe('studio')
    expect(studioOnHtml()).toBe(true)
    expect(reloadPage).not.toHaveBeenCalled()
  })
})
