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
  DEFAULT_UI_STYLE,
  LEGACY_STUDIO_TONE_VALUE,
  STUDIO_HTML_CLASS,
  TONE_STORAGE_KEY,
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

function readLegacyStudioTone(): boolean {
  try {
    return window.localStorage.getItem(TONE_STORAGE_KEY) === LEGACY_STUDIO_TONE_VALUE
  } catch {
    return false
  }
}

interface UiStyleProviderProps {
  initialStyle?: UiStyle
  children: ReactNode
}

export function UiStyleProvider({ initialStyle = DEFAULT_UI_STYLE, children }: UiStyleProviderProps) {
  const [style, setStyleState] = useState<UiStyle>(initialStyle)
  const { theme, setTheme } = useTheme()

  const setStyle = useCallback((next: UiStyle) => {
    setStyleState(next)
    applyUiStyleToDocument(next)
  }, [])

  // One-time migration: before W3 "Studio" was a value of the tone key. Such a
  // browser becomes Studio style + System tone, once, on its first load.
  useEffect(() => {
    if (theme === LEGACY_STUDIO_TONE_VALUE || readLegacyStudioTone()) {
      setStyle('studio')
      setTheme('system')
    }
  }, [theme, setTheme, setStyle])

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
