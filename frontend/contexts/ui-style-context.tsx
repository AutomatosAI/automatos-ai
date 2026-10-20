'use client'

/**
 * PRD-244 D1 — the Style axis (Classic | Studio | a theme, PRD-261 A-02a) as React context.
 *
 * The server reads the style cookie and puts the style's classes on <html> before first paint
 * with the theme boot (lib/theme-host/boot.ts), plus `initialStyle`, so the first paint is
 * already the right design system. Switching writes the cookie (for the next request), the html classes (for this
 * document) and localStorage (a mirror), and never touches the tone. A switch to an installed
 * theme writes the cookie and reloads instead (PA-13): its first paint comes from the server.
 * What the switch must carry across that reload runs in its `beforeReload`, just before it (G3:
 * the note's 'Switch to Play' keeps focus for the screen's heading, lib/theme-host/switch-focus.ts).
 *
 * PRD-261 A-02c (PA-12): the server holds the choice. `setStyle` is the local half (cookie, html,
 * reload); lib/theme-host/choice-sync.ts stores a pick on the server and checks the server's
 * choice once per workspace and tab. It also fills `installedThemes`, which the theme menu and
 * ⌘K list; this provider sits above the workspace and query providers, so it cannot read them.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useTheme } from 'next-themes'
import { writeCookie } from '@/lib/cookies'
import type { InstalledTheme } from '@/lib/api-client'
import { findThemeEntry, isInstalledThemeStyle } from '@/lib/theme-host/registry'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  DEFAULT_TONE,
  DEFAULT_UI_STYLE,
  STYLE_HTML_CLASS_NAMES,
  UI_STYLE_COOKIE,
  UI_STYLE_COOKIE_MAX_AGE_S,
  UI_STYLE_STORAGE_KEY,
  rendersStudio,
  styleHtmlClasses,
  type UiStyle,
} from '@/lib/ui-style'

/** What a style switch carries across the reload into an installed theme (PA-13). */
export interface StyleSwitchOptions {
  /** Runs just before the page reloads into an installed theme; never for a switch made in place. */
  readonly beforeReload?: () => void
}

export interface UiStyleContextValue {
  style: UiStyle
  setStyle: (next: UiStyle, options?: StyleSwitchOptions) => void
  /** True for Studio AND every theme drawn on Studio (PRD-260 D2): the theme renders Studio's layout where it has no design. */
  isStudio: boolean
  /** The active theme's registry entry: null for Classic and Studio, and for a theme the registry doesn't know. */
  activeTheme: ThemeEntry | null
  /** PRD-261 A-02c: the workspace's installed themes, for the menu and ⌘K; [] until ThemeChoiceSync reads them. */
  installedThemes: readonly InstalledTheme[]
  setInstalledThemes: (rows: readonly InstalledTheme[]) => void
}

const NO_INSTALLED_THEMES: readonly InstalledTheme[] = []

const UiStyleContext = createContext<UiStyleContextValue | null>(null)

/**
 * The cookie the next request reads, and the localStorage mirror. Nothing else: no html class, no
 * reload (choice-sync.ts writes an installed server choice with it when the tab may not reload).
 */
export function persistUiStyle(style: UiStyle): void {
  writeCookie(UI_STYLE_COOKIE, style, UI_STYLE_COOKIE_MAX_AGE_S)
  try {
    window.localStorage.setItem(UI_STYLE_STORAGE_KEY, style)
  } catch {
    // Storage blocked (private mode): the cookie alone carries the choice.
  }
}

/** Where the next request (cookie) and this document (html class) see the style. */
export function applyUiStyleToDocument(style: UiStyle): void {
  if (typeof document === 'undefined') return
  const wanted = styleHtmlClasses(style)
  for (const name of STYLE_HTML_CLASS_NAMES) {
    document.documentElement.classList.toggle(name, wanted.includes(name))
  }
  persistUiStyle(style)
}

/** The stored defaults version, or null when storage is unavailable or unset. */
function readAppliedDefaultsVersion(): string | null {
  try {
    return window.localStorage.getItem(APPEARANCE_DEFAULTS_KEY)
  } catch {
    return null
  }
}

function reloadDocument(): void {
  window.location.reload()
}

interface UiStyleProviderProps {
  initialStyle?: UiStyle
  /** How a switch to an installed theme reloads the page (PA-13); tests pass their own. */
  reloadPage?: () => void
  children: ReactNode
}

export function UiStyleProvider({ initialStyle = DEFAULT_UI_STYLE, reloadPage = reloadDocument, children }: UiStyleProviderProps) {
  const [style, setStyleState] = useState<UiStyle>(initialStyle)
  const [installedThemes, setInstalledThemes] = useState<readonly InstalledTheme[]>(NO_INSTALLED_THEMES)
  const { setTheme } = useTheme()

  // PRD-261 PA-13: an installed theme's boot data and import map come with the page, so a
  // switch to one is never made in place: the cookie is written and the page reloads.
  const setStyle = useCallback((next: UiStyle, options?: StyleSwitchOptions) => {
    if (isInstalledThemeStyle(next)) {
      persistUiStyle(next)
      options?.beforeReload?.()
      reloadPage()
      return
    }
    setStyleState(next)
    applyUiStyleToDocument(next)
  }, [reloadPage])

  // PRD-244 D1 (Gerard, 2026-09-17): Studio + Dark are the defaults for
  // everyone, once — a browser that has not seen this defaults version is moved
  // to them on its next load and then picks freely. This also retires the
  // pre-W3 "studio" tone value, since the tone is rewritten here.
  // PRD-260 M87: once per mount, not once per setTheme. next-themes 0.2.1 hands out a new
  // setTheme whenever forcedTheme changes (a theme's look forcing its tone), and a browser whose
  // storage can't be read never records the defaults version, so a re-run moved it straight
  // back to Studio the moment it chose a theme.
  // PRD-261 A-02c: this step runs once, at mount, before any workspace is known; the server's
  // choice is checked after it (choice-sync.ts), so the defaults never override a server choice.
  // It is local only: it never stores a choice on the server.
  const defaultsChecked = useRef(false)
  useEffect(() => {
    if (defaultsChecked.current) return
    defaultsChecked.current = true
    if (readAppliedDefaultsVersion() === APPEARANCE_DEFAULTS_VERSION) return
    setStyle(DEFAULT_UI_STYLE)
    setTheme(DEFAULT_TONE)
    try {
      window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
    } catch {
      // Storage blocked: the defaults apply again next load, which is harmless.
    }
  }, [setTheme, setStyle])

  const value = useMemo<UiStyleContextValue>(
    () => ({
      style,
      setStyle,
      isStudio: rendersStudio(style),
      activeTheme: findThemeEntry(style),
      installedThemes,
      setInstalledThemes,
    }),
    [style, setStyle, installedThemes],
  )
  return <UiStyleContext.Provider value={value}>{children}</UiStyleContext.Provider>
}

export function useUiStyle(): UiStyleContextValue {
  const ctx = useContext(UiStyleContext)
  if (!ctx) throw new Error('useUiStyle must be used within UiStyleProvider')
  return ctx
}

/** Null outside the provider (isolated mounts, tests) — callers treat null as Classic. */
export function useUiStyleOptional(): UiStyleContextValue | null {
  return useContext(UiStyleContext)
}
