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

/**
 * The defaults (Gerard, 2026-09-17): Studio in the Dark tone. A browser with
 * no stored choice gets them on the server (the style) and from next-themes
 * (the tone); every browser is moved to them ONCE — see
 * APPEARANCE_DEFAULTS_VERSION — and picks freely afterwards.
 */
export const DEFAULT_UI_STYLE: UiStyle = 'studio'
export const DEFAULT_TONE = 'dark'

/**
 * Bump to move every browser to the current defaults one more time. The
 * provider compares the stored value on mount and applies the defaults when
 * it differs, then stores the version.
 */
export const APPEARANCE_DEFAULTS_KEY = 'automatos-appearance-defaults'
export const APPEARANCE_DEFAULTS_VERSION = '2026-09-17-studio-dark'

/** Cookie + localStorage key for the style. */
export const UI_STYLE_COOKIE = 'automatos-style'
export const UI_STYLE_STORAGE_KEY = 'automatos-style'
export const UI_STYLE_COOKIE_MAX_AGE_S = 60 * 60 * 24 * 365

/** The class the Studio stylesheet hangs off (`.studio {}` in globals.css). */
export const STUDIO_HTML_CLASS = 'studio'

export function isUiStyle(value: unknown): value is UiStyle {
  return value === 'classic' || value === 'studio'
}

/** Unknown, missing or tampered → the default style. */
export function parseUiStyle(value: string | null | undefined): UiStyle {
  return isUiStyle(value) ? value : DEFAULT_UI_STYLE
}
