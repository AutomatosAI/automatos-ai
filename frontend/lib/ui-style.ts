/**
 * PRD-244 D1 — the Style axis: Classic | Studio.
 *
 * Two styles, two tones (Gerard, 2026-09-17). Style is *which design system*
 * (Classic = the original design; Studio = the paper/ink redesign). Tone is
 * Light | Dark | System and stays with next-themes (`.dark` on <html>, the
 * class every Tailwind rule depends on). The style is its own persisted
 * setting: a cookie the server reads (app/layout.tsx) so the Studio chrome
 * renders on first paint with no classic flash, mirrored to localStorage.
 */
export type UiStyle = 'classic' | 'studio'

export const UI_STYLES: readonly UiStyle[] = ['classic', 'studio'] as const
export const DEFAULT_UI_STYLE: UiStyle = 'classic'

/** Cookie + localStorage key for the style. */
export const UI_STYLE_COOKIE = 'automatos-style'
export const UI_STYLE_STORAGE_KEY = 'automatos-style'
export const UI_STYLE_COOKIE_MAX_AGE_S = 60 * 60 * 24 * 365

/** The class the Studio stylesheet hangs off (`.studio {}` in globals.css). */
export const STUDIO_HTML_CLASS = 'studio'

/**
 * next-themes' storage key (components/providers.tsx). Before W3 "Studio" was
 * a value of this key; UiStyleProvider migrates such a browser once.
 */
export const TONE_STORAGE_KEY = 'automatos-theme'
export const LEGACY_STUDIO_TONE_VALUE = 'studio'

export function isUiStyle(value: unknown): value is UiStyle {
  return value === 'classic' || value === 'studio'
}

/** Unknown, missing or tampered → the default style. */
export function parseUiStyle(value: string | null | undefined): UiStyle {
  return isUiStyle(value) ? value : DEFAULT_UI_STYLE
}
