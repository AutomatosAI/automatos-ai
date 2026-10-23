/**
 * PRD-244 D1 — the Style axis; PRD-261 A-02a — Classic | Studio | a theme.
 *
 * Style is *which design system* (Classic = the original design; Studio = the
 * paper/ink redesign; a theme = a design drawn on one of them, named on the
 * axis as `theme:<ref>` and described by its entry in the theme registry,
 * lib/theme-host/registry.ts). Tone is Light | Dark | System and stays with
 * next-themes (`.dark` on <html>, the class every Tailwind rule depends on).
 * The style is its own persisted setting: a cookie the server reads
 * (app/layout.tsx) so the chosen chrome renders on first paint with no classic
 * flash, mirrored to localStorage.
 *
 * PRD-260 D1/D2: a theme drawn on Studio is layered on it — <html> carries
 * `studio` and the theme's own class — so a page the theme has no design for
 * renders Studio's version inside the theme's frame.
 */
import { DEFAULT_THEME_BASE, isInstalledThemeStyle } from '@/lib/theme-host/registry'
import type { UiStyle } from '@/lib/theme-host/registry-types'

export type { UiStyle }

/** The styles the app knows without asking the server: Classic and Studio (no theme is compiled in). */
export const UI_STYLES: readonly UiStyle[] = ['classic', 'studio']

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

/**
 * Every class the Style axis may put on <html>; switching style toggles each, and a style's
 * theme boot (lib/theme-host/boot.ts) removes the ones its style does not carry before first
 * paint (PRD-260 M07, PRD-261 A-03). An installed theme's own class comes from its boot and goes
 * with the host's cleanup (lib/theme-host/look.ts), not from this list.
 */
export const STYLE_HTML_CLASS_NAMES: readonly string[] = [STUDIO_HTML_CLASS]

/**
 * PRD-260 D2: whether the Studio layout renders: Studio, and every theme drawn on Studio, where
 * the theme has no design of its own. The axis knows no theme's entry (an installed one comes
 * with its boot), so a theme is drawn on DEFAULT_THEME_BASE.
 */
export function rendersStudio(style: UiStyle): boolean {
  if (style === 'classic') return false
  return style === 'studio' || DEFAULT_THEME_BASE === 'studio'
}

/** The <html> classes for a style: [] Classic, ['studio'] Studio and a theme drawn on it (its own class comes with its boot). */
export function styleHtmlClasses(style: UiStyle): string[] {
  return rendersStudio(style) ? [STUDIO_HTML_CLASS] : []
}

/** Classic, Studio, or an installed theme's style (`theme:install:<uuid>`). */
export function isUiStyle(value: unknown): value is UiStyle {
  if (typeof value !== 'string') return false
  return (UI_STYLES as readonly string[]).includes(value) || isInstalledThemeStyle(value)
}

/**
 * The stored style as the axis reads it; unknown, missing or tampered → the default style. The
 * values builtin Play's choice left in the cookie (its bare name, written before the axis took
 * theme refs, and its compiled-in theme ref) name no style any more, so they read as the
 * default (A-19f1; lib/__tests__/ui-style.test.ts holds both).
 */
export function parseUiStyle(value: string | null | undefined): UiStyle {
  return isUiStyle(value) ? value : DEFAULT_UI_STYLE
}
