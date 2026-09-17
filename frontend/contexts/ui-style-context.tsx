'use client'

/**
 * PRD-244 D1 — the Style axis (Classic | Studio) as React context.
 *
 * The server reads the style cookie and renders <html class="studio"> plus
 * `initialStyle`, so the first paint is already the right design system.
 * Switching writes the cookie (for the next request), the html class (for
 * this document) and localStorage (a mirror), and never touches the tone.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useTheme } from 'next-themes'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  DEFAULT_TONE,
  DEFAULT_UI_STYLE,
  STUDIO_HTML_CLASS,
  UI_STYLE_COOKIE,
  UI_STYLE_COOKIE_MAX_AGE_S,
  UI_STYLE_STORAGE_KEY,
  type UiStyle,
} from '@/lib/ui-style'

export interface UiStyleContextValue {
  style: UiStyle
  setStyle: (next: UiStyle) => void
  isStudio: boolean
}

const UiStyleContext = createContext<UiStyleContextValue | null>(null)

/** Where the next request (cookie) and this document (html class) see the style. */
export function applyUiStyleToDocument(style: UiStyle): void {
  if (typeof document === 'undefined') return
  document.documentElement.classList.toggle(STUDIO_HTML_CLASS, style === 'studio')
  const secure = window.location.protocol === 'https:' ? '; Secure' : ''
  document.cookie = `${UI_STYLE_COOKIE}=${style}; Path=/; Max-Age=${UI_STYLE_COOKIE_MAX_AGE_S}; SameSite=Lax${secure}`
  try {
    window.localStorage.setItem(UI_STYLE_STORAGE_KEY, style)
  } catch {
    // Storage blocked (private mode): the cookie alone carries the choice.
  }
}

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
  children: ReactNode
}

export function UiStyleProvider({ initialStyle = DEFAULT_UI_STYLE, children }: UiStyleProviderProps) {
  const [style, setStyleState] = useState<UiStyle>(initialStyle)
  const { setTheme } = useTheme()

  const setStyle = useCallback((next: UiStyle) => {
    setStyleState(next)
    applyUiStyleToDocument(next)
  }, [])

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
    () => ({ style, setStyle, isStudio: style === 'studio' }),
    [style, setStyle],
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
