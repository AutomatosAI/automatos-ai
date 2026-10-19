/**
 * PRD-261 A-02a — the Style axis takes theme refs: Classic | Studio | `theme:<ref>`, with builtin
 * Play as one registry entry. The legacy `play` cookie still means builtin Play; an installed
 * theme's style is accepted only in its exact shape; Studio's layout renders wherever the entry's
 * base is Studio; the theme layout follows the entry's minWidth; and a switch to an installed
 * theme writes the cookie and reloads instead of switching in place (PA-13).
 */
import { createElement, type ReactNode } from 'react'
import { act, cleanup, render, renderHook, screen } from '@testing-library/react'
import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('next-themes', () => ({ useTheme: () => ({ theme: 'dark', resolvedTheme: 'dark', setTheme: vi.fn() }) }))

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { useActiveTheme, useThemeLayout } from '@/hooks/use-studio-theme'
import { stubWindowWidth, type WindowWidth } from '@/hooks/__tests__/window-width'
import {
  BUILTIN_PLAY_ENTRY,
  BUILTIN_PLAY_HTML_CLASS,
  BUILTIN_PLAY_MIN_WIDTH_PX,
  BUILTIN_PLAY_STYLE,
  LEGACY_STYLE_ALIASES,
} from '@/lib/theme-host/builtin/play-entry'
import {
  BUILTIN_THEMES,
  findThemeEntry,
  isInstalledThemeStyle,
  resolveStyleAlias,
  themeRefOf,
  themeStyleOf,
} from '@/lib/theme-host/registry'
import type { ThemeEntry, ThemeLayout } from '@/lib/theme-host/registry-types'
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

const INSTALL_ID = '0f8c2b6e-4d1a-4c3b-9e7f-1a2b3c4d5e6f'
const INSTALL_STYLE: UiStyle = `theme:install:${INSTALL_ID}`
/** The widths the plan names: a laptop above Play's 1024 px, a small window below it. */
const WIDE_PX = 1440
const NARROW_PX = 800

/** An installed theme as A-02c will hand its entry to the registry. */
const INSTALLED_CLASSIC: ThemeEntry = {
  ...BUILTIN_PLAY_ENTRY,
  ref: `install:${INSTALL_ID}`,
  name: 'Installed',
  base: 'classic',
  htmlClass: 'installed-theme',
  kind: 'installed',
  reviewed: false,
}

describe('PRD-261 A-02a — builtin Play is one registry entry', () => {
  it('lists builtin Play with the entry the plan fixes', () => {
    expect(BUILTIN_THEMES).toEqual([BUILTIN_PLAY_ENTRY])
    expect(BUILTIN_PLAY_ENTRY).toMatchObject({
      ref: 'builtin:play',
      base: 'studio',
      htmlClass: 'play',
      minWidth: 1024,
      tone: 'from-look',
      kind: 'builtin',
      ownsPages: ['/chat', '/deliverables'],
      routes: ['conversations'],
    })
    expect(BUILTIN_PLAY_ENTRY.nav.map((item) => item.path)).toEqual(['/x/conversations', '/looks', '/looks/style'])
    expect(BUILTIN_PLAY_ENTRY.slots).toEqual(expect.arrayContaining(['frame@1', 'lookEditor@1', 'styleSheet@1']))
  })

  it("names Play on the axis as theme:builtin:play, and maps a ref to its style and back", () => {
    expect(BUILTIN_PLAY_STYLE).toBe('theme:builtin:play')
    expect(themeStyleOf(BUILTIN_PLAY_ENTRY.ref)).toBe(BUILTIN_PLAY_STYLE)
    expect(themeRefOf(BUILTIN_PLAY_STYLE)).toBe('builtin:play')
    expect(themeRefOf('studio')).toBeNull()
    expect(themeRefOf('classic')).toBeNull()
  })

  it('lists Classic, Studio, then each builtin theme', () => {
    expect(UI_STYLES).toEqual(['classic', 'studio', BUILTIN_PLAY_STYLE])
  })

  it("puts Studio's class and each builtin theme's own class on the axis's class list (PA-8)", () => {
    expect(BUILTIN_PLAY_HTML_CLASS).toBe('play')
    expect(STYLE_HTML_CLASS_NAMES).toEqual([STUDIO_HTML_CLASS, BUILTIN_PLAY_HTML_CLASS])
  })

  it('finds an entry for builtin Play, for an installed theme it is given, and none otherwise', () => {
    expect(findThemeEntry(BUILTIN_PLAY_STYLE)).toBe(BUILTIN_PLAY_ENTRY)
    expect(findThemeEntry('studio')).toBeNull()
    expect(findThemeEntry('classic')).toBeNull()
    expect(findThemeEntry('theme:builtin:unknown')).toBeNull()
    expect(findThemeEntry(INSTALL_STYLE)).toBeNull()
    expect(findThemeEntry(INSTALL_STYLE, [INSTALLED_CLASSIC])).toBe(INSTALLED_CLASSIC)
  })
})

describe('PRD-261 A-02a — parseUiStyle', () => {
  it('reads the legacy play cookie and theme:builtin:play both as builtin Play', () => {
    expect(LEGACY_STYLE_ALIASES).toEqual({ play: BUILTIN_PLAY_STYLE })
    expect(parseUiStyle('play')).toBe(BUILTIN_PLAY_STYLE)
    expect(parseUiStyle('theme:builtin:play')).toBe(BUILTIN_PLAY_STYLE)
  })

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
      'theme:builtin:unknown',
      'theme:play',
      'builtin:play',
      'PLAY',
      'toString',
      '__proto__',
      'constructor',
      '',
    ]
    for (const raw of tampered) expect(parseUiStyle(raw), JSON.stringify(raw)).toBe(DEFAULT_UI_STYLE)
    expect(parseUiStyle(null)).toBe(DEFAULT_UI_STYLE)
    expect(parseUiStyle(undefined)).toBe(DEFAULT_UI_STYLE)
    expect(DEFAULT_UI_STYLE).toBe('studio')
  })

  it('never takes an inherited property for an alias', () => {
    for (const raw of ['toString', '__proto__', 'hasOwnProperty']) expect(resolveStyleAlias(raw)).toBe(raw)
    expect(resolveStyleAlias('play')).toBe(BUILTIN_PLAY_STYLE)
  })

  it('does not count the legacy value itself as a style: it is only read from old cookies', () => {
    expect(isUiStyle('play')).toBe(false)
  })
})

describe('PRD-261 A-02a — rendersStudio and the <html> classes follow the entry', () => {
  it("renders Studio for Studio and builtin Play (base studio), never for Classic", () => {
    expect(rendersStudio('studio')).toBe(true)
    expect(rendersStudio(BUILTIN_PLAY_STYLE)).toBe(true)
    expect(rendersStudio('classic')).toBe(false)
  })

  it('draws an installed theme the registry does not know yet on Studio, the default', () => {
    expect(rendersStudio(INSTALL_STYLE)).toBe(true)
    expect(styleHtmlClasses(INSTALL_STYLE)).toEqual([STUDIO_HTML_CLASS])
  })

  it("gives builtin Play the same <html> classes as before: studio, then play", () => {
    expect(styleHtmlClasses('classic')).toEqual([])
    expect(styleHtmlClasses('studio')).toEqual([STUDIO_HTML_CLASS])
    expect(styleHtmlClasses(BUILTIN_PLAY_STYLE)).toEqual([STUDIO_HTML_CLASS, BUILTIN_PLAY_HTML_CLASS])
  })
})

const wrapper = (style: UiStyle, reloadPage?: () => void) =>
  function Wrapper({ children }: { readonly children: ReactNode }) {
    return createElement(UiStyleProvider, { initialStyle: style, reloadPage, children })
  }

function LayoutProbe({ seen }: { readonly seen: ThemeLayout[] }) {
  const layout = useThemeLayout()
  seen.push(layout)
  return createElement('p', { 'data-testid': 'layout' }, layout)
}

function layoutTree(style: UiStyle, seen: ThemeLayout[]) {
  return createElement(UiStyleProvider, { initialStyle: style, children: createElement(LayoutProbe, { seen }) })
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

  it("is builtin Play's entry under builtin Play, and null under Classic, Studio and outside the provider", () => {
    expect(renderHook(() => useActiveTheme(), { wrapper: wrapper(BUILTIN_PLAY_STYLE) }).result.current).toBe(BUILTIN_PLAY_ENTRY)
    expect(renderHook(() => useActiveTheme(), { wrapper: wrapper('studio') }).result.current).toBeNull()
    expect(renderHook(() => useActiveTheme(), { wrapper: wrapper('classic') }).result.current).toBeNull()
    expect(renderHook(() => useActiveTheme()).result.current).toBeNull()
  })

  it("is 'pending' on the server under a theme: no width is known there", () => {
    const seen: ThemeLayout[] = []
    expect(renderToString(layoutTree(BUILTIN_PLAY_STYLE, seen))).toContain('pending')
    expect(seen).toEqual(['pending'])
  })

  it(`is 'theme' at ${WIDE_PX} px and 'fallback' at ${NARROW_PX} px, from the first render`, () => {
    const wideWindow = stubWindowWidth(WIDE_PX)
    const wide: ThemeLayout[] = []
    try {
      render(layoutTree(BUILTIN_PLAY_STYLE, wide))
    } finally {
      cleanup()
      wideWindow.restore()
    }
    expect(new Set(wide)).toEqual(new Set(['theme']))

    window_ = stubWindowWidth(NARROW_PX)
    const narrow: ThemeLayout[] = []
    render(layoutTree(BUILTIN_PLAY_STYLE, narrow))
    expect(new Set(narrow)).toEqual(new Set(['fallback']))
  })

  it("reads the entry's minWidth: the switch is at builtin Play's 1024 px", () => {
    window_ = stubWindowWidth(BUILTIN_PLAY_MIN_WIDTH_PX)
    render(layoutTree(BUILTIN_PLAY_STYLE, []))
    expect(screen.getByTestId('layout')).toHaveTextContent('theme')
    act(() => window_?.resize(BUILTIN_PLAY_MIN_WIDTH_PX - 1))
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
    expect(document.documentElement.classList.contains(BUILTIN_PLAY_HTML_CLASS)).toBe(false)
  })

  it('switches to builtin Play in place, with no reload', () => {
    const reloadPage = vi.fn()
    const mounted = mountStyle('studio', reloadPage)
    mounted.setStyle(BUILTIN_PLAY_STYLE)
    expect(reloadPage).not.toHaveBeenCalled()
    expect(mounted.current()).toBe(BUILTIN_PLAY_STYLE)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${BUILTIN_PLAY_STYLE}`)
    expect(document.documentElement.classList.contains(STUDIO_HTML_CLASS)).toBe(true)
    expect(document.documentElement.classList.contains(BUILTIN_PLAY_HTML_CLASS)).toBe(true)
  })
})
