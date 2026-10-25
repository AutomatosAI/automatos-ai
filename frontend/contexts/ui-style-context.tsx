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
 * a switch can keep focus for the theme screen's heading, lib/theme-host/switch-focus.ts).
 *
 * PRD-261 A-02c (PA-12): the server holds the choice. `setStyle` is the local half (cookie, html,
 * reload); lib/theme-host/choice-sync.ts stores a pick on the server and checks the server's
 * choice once per workspace and tab. It also fills `installedThemes`, which the theme menu and
 * ⌘K list; this provider sits above the workspace and query providers, so it cannot read them.
 *
 * PRD-261 W-A6b (gap G1): an installed theme's registry entry comes with the page: the root
 * layout builds it from the boot the server resolved for the style (lib/theme-host/installed-entry.ts)
 * and passes it in as `installedTheme`, so the server render and the first client render name the
 * same activeTheme. The theme host (components/theme-host/theme-host-provider.tsx), which sits
 * below, releases it when it cannot run the theme (a failed load, safe mode, no install to run):
 * activeTheme is then null, as for a theme the registry doesn't know.
 *
 * When the page's boot carried no entry under an installed theme's style (the server's boot fetch
 * timed out, was refused or failed, so the page has the base boot), the browser asks the boot route
 * itself, once per install (lib/theme-host/boot-client.ts), and the entry it answers becomes the
 * active theme: later than the first render, but the theme's frame, pages and look runtime mount.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useTheme } from 'next-themes'
import { writeCookie } from '@/lib/cookies'
import type { InstalledTheme } from '@/lib/api-client'
import { pageInstalledBoot, type InstalledBootSource } from '@/lib/theme-host/boot-client'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import { findThemeEntry, installIdOfStyle, isInstalledThemeStyle } from '@/lib/theme-host/registry'
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
  /**
   * The active theme's registry entry: null for Classic and Studio, for a theme the registry doesn't
   * know, for an installed theme the theme host released, and for one whose boot carried no entry
   * until the browser's own fetch of it answers one (lib/theme-host/boot-client.ts).
   */
  activeTheme: ThemeEntry | null
  /**
   * PRD-261 W-A6b: the theme host names the installed theme's style it cannot run (failed, safe mode,
   * no install to run), or null while it loads or runs one.
   */
  releaseInstalledTheme: (style: UiStyle | null) => void
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

/**
 * Where this browser stands with the current appearance defaults (PRD-244 D1): `applied` (it holds
 * the current version), `due` (it did not, and now does: the marker is stored before the defaults
 * apply), or `unavailable` (its storage cannot keep the marker: blocked, or private mode).
 */
type AppearanceDefaults = 'applied' | 'due' | 'unavailable'

/** Claims the current defaults version for this browser; see AppearanceDefaults. */
function claimAppearanceDefaults(): AppearanceDefaults {
  try {
    const storage = window.localStorage
    if (storage.getItem(APPEARANCE_DEFAULTS_KEY) === APPEARANCE_DEFAULTS_VERSION) return 'applied'
    storage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
    return storage.getItem(APPEARANCE_DEFAULTS_KEY) === APPEARANCE_DEFAULTS_VERSION ? 'due' : 'unavailable'
  } catch {
    return 'unavailable'
  }
}

function reloadDocument(): void {
  window.location.reload()
}

/** An installed theme from its boot's entry, until the host releases it; Classic and Studio have none. */
function activeThemeOf(style: UiStyle, installed: ThemeEntry | null, released: UiStyle | null): ThemeEntry | null {
  if (!isInstalledThemeStyle(style) || installed === null || released === style) return null
  return findThemeEntry(style, [installed])
}

/**
 * The installed theme's entry from the boot the browser fetched (lib/theme-host/boot-client.ts),
 * when the page's boot carried none for `style`; null until it answers, and when it answers none.
 */
function useFetchedEntry(style: UiStyle, installedAtLoad: ThemeEntry | null, bootSource: InstalledBootSource): ThemeEntry | null {
  const [fetched, setFetched] = useState<ThemeEntry | null>(null)
  const installId = installedAtLoad === null ? installIdOfStyle(style) : null
  useEffect(() => {
    if (installId === null) return undefined
    let live = true
    void bootSource(installId).then((boot) => {
      const entry = installedThemeEntry(style, boot?.entry ?? null)
      if (live && entry !== null) setFetched(entry)
    })
    return () => {
      live = false
    }
  }, [style, installId, bootSource])
  return fetched
}

/**
 * The active theme's entry, and the host's way to release an installed one it cannot run. The
 * installed entry is the one the page loaded with, held for the provider's life as `style` is: a
 * server re-render (router.refresh) builds the boot again and may hand in another entry or none
 * (a boot that timed out, another tab's style cookie), but this document still runs the theme it
 * loaded, and a switch to another installed theme always reloads (PA-13). A page that loaded with
 * none takes the one the browser fetched (useFetchedEntry).
 */
function useActiveThemeEntry(style: UiStyle, installed: ThemeEntry | null, bootSource: InstalledBootSource) {
  const [installedAtLoad] = useState(installed)
  const fetched = useFetchedEntry(style, installedAtLoad, bootSource)
  const [released, releaseInstalledTheme] = useState<UiStyle | null>(null)
  const entry = installedAtLoad ?? fetched
  const activeTheme = useMemo(() => activeThemeOf(style, entry, released), [style, entry, released])
  return { activeTheme, releaseInstalledTheme }
}

interface UiStyleProviderProps {
  initialStyle?: UiStyle
  /**
   * PRD-261 W-A6b: the installed theme's entry the root layout built from its boot; null for every
   * other style. Read once, at mount: a later server re-render never changes the active theme.
   */
  installedTheme?: ThemeEntry | null
  /** How a switch to an installed theme reloads the page (PA-13); tests pass their own. */
  reloadPage?: () => void
  /**
   * Where the browser gets the installed theme's boot when the page's carried no entry; tests pass
   * their own. Default: GET /api/themes/boot/{installId}, once per install (boot-client.ts).
   */
  bootSource?: InstalledBootSource
  children: ReactNode
}

export function UiStyleProvider({
  initialStyle = DEFAULT_UI_STYLE,
  installedTheme = null,
  reloadPage = reloadDocument,
  bootSource = pageInstalledBoot,
  children,
}: UiStyleProviderProps) {
  const [style, setStyleState] = useState<UiStyle>(initialStyle)
  const [installedThemes, setInstalledThemes] = useState<readonly InstalledTheme[]>(NO_INSTALLED_THEMES)
  const { activeTheme, releaseInstalledTheme } = useActiveThemeEntry(style, installedTheme, bootSource)
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
  // setTheme whenever forcedTheme changes (a theme's look forcing its tone).
  // PRD-261 W-A8: a browser whose storage cannot keep the defaults version can never be moved
  // "once", so the step leaves it alone: applying it on every load moved such a browser back to
  // Studio after each reload, rewriting the style cookie, so no Classic or installed theme it
  // chose could stick. Its stored choice wins instead: the style cookie the server painted the
  // first paint from, then the server's choice (choice-sync.ts). Such a browser with no cookie
  // already starts on the defaults: the server paints DEFAULT_UI_STYLE and next-themes Dark.
  // PRD-261 A-02c: this step runs once, at mount, before any workspace is known; the server's
  // choice is checked after it (choice-sync.ts), so the defaults never override a server choice.
  // It is local only: it never stores a choice on the server.
  const defaultsChecked = useRef(false)
  useEffect(() => {
    if (defaultsChecked.current) return
    defaultsChecked.current = true
    if (claimAppearanceDefaults() !== 'due') return
    setStyle(DEFAULT_UI_STYLE)
    setTheme(DEFAULT_TONE)
  }, [setTheme, setStyle])

  const value = useMemo<UiStyleContextValue>(
    () => ({
      style,
      setStyle,
      isStudio: rendersStudio(style),
      activeTheme,
      releaseInstalledTheme,
      installedThemes,
      setInstalledThemes,
    }),
    [style, setStyle, activeTheme, releaseInstalledTheme, installedThemes],
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
