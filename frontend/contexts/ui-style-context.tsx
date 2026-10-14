'use client'

/**
 * PRD-244 D1 — the Style axis (Classic | Studio) as React context; PRD-259 §4 adds a
 * workspace's installed skins (`skin:<id>`).
 *
 * The server reads the style cookie (and, for a skin, the base cookie) and renders
 * <html class="studio" data-skin="<id>"> plus `initialStyle` / `initialSkinBase`, so the
 * first paint is already the right design system; the pre-paint script applies the cached
 * skin stylesheet. Switching writes the cookies (for the next request), the html class and
 * the skin (for this document) and localStorage (a mirror), and never touches the tone: a
 * single-tone skin LOCKS the tone through `onToneLock` (ThemeProvider's forcedTheme), which
 * leaves the user's own stored tone alone; the server starts that lock from the tone cookie
 * (`initialLockedTone`), so it holds from the first paint. `isStudio` follows the BASE, so every page that
 * branches on it keeps its layout under a skin.
 */
import { createContext, useContext, useEffect, useMemo, type ReactNode } from 'react'
import { useTheme } from 'next-themes'
import type { InstalledSkins } from '@/hooks/use-installed-skins'
import type { ResolvedSkin } from '@/lib/skins/installed'
import type { SkinTone } from '@/lib/skins/schema'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  DEFAULT_TONE,
  DEFAULT_UI_STYLE,
  baseOf,
  type BaseStyle,
  type UiStyle,
} from '@/lib/ui-style'
import { useSkinFonts, useSkinStyle, useToneLock } from './skin-style'

export interface UiStyleContextValue {
  style: UiStyle
  setStyle: (next: UiStyle) => void
  /** True when the base is Studio: Studio itself, or a skin built on it. */
  isStudio: boolean
  base: BaseStyle
  /** The active skin once it resolved against the installed list (null for Classic and Studio). */
  activeSkin: ResolvedSkin | null
  installedSkins: InstalledSkins
}

const UiStyleContext = createContext<UiStyleContextValue | null>(null)

/** The stored defaults version, or null when storage is unavailable or unset. */
function readAppliedDefaultsVersion(): string | null {
  try {
    return window.localStorage.getItem(APPEARANCE_DEFAULTS_KEY)
  } catch {
    return null
  }
}

interface UiStyleProviderProps {
  initialStyle?: UiStyle
  /** The active skin's base, from its cookie (decision K1); ignored unless the style is a skin. */
  initialSkinBase?: BaseStyle
  /** The tone the server locked from the tone cookie; holds until the installed list resolves the skin. */
  initialLockedTone?: SkinTone
  /** Called with the tone a single-tone skin locks, or undefined when nothing is locked. */
  onToneLock?: (tone: SkinTone | undefined) => void
  children: ReactNode
}

export function UiStyleProvider({
  initialStyle = DEFAULT_UI_STYLE, initialSkinBase, initialLockedTone, onToneLock, children,
}: UiStyleProviderProps) {
  const { style, base, active, setStyle, installed } = useSkinStyle(initialStyle, baseOf(initialStyle, initialSkinBase))
  const { setTheme } = useTheme()
  useSkinFonts(active)
  useToneLock({ style, active }, initialLockedTone, onToneLock)

  // PRD-244 D1 (Gerard, 2026-09-17): Studio + Dark are the defaults for
  // everyone, once — a browser that has not seen this defaults version is moved
  // to them on its next load and then picks freely. This also retires the
  // pre-W3 "studio" tone value, since the tone is rewritten here.
  useEffect(() => {
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
    () => ({ style, setStyle, isStudio: base === 'studio', base, activeSkin: active, installedSkins: installed }),
    [style, setStyle, base, active, installed],
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
