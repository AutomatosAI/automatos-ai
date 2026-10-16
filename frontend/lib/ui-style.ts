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
import {
  BUILTIN_THEME_HTML_CLASSES,
  BUILTIN_THEME_STYLES,
  DEFAULT_THEME_BASE,
  findThemeEntry,
  isInstalledThemeStyle,
  resolveStyleAlias,
} from '@/lib/theme-host/registry'
import type { UiStyle } from '@/lib/theme-host/registry-types'

export type { UiStyle }

/** The styles the app knows without asking the server: Classic, Studio, then every builtin theme. */
export const UI_STYLES: readonly UiStyle[] = ['classic', 'studio', ...BUILTIN_THEME_STYLES]

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

/** Every class the Style axis may put on <html>; switching style toggles each. */
export const STYLE_HTML_CLASS_NAMES: readonly string[] = [STUDIO_HTML_CLASS, ...BUILTIN_THEME_HTML_CLASSES]

/**
 * PRD-260 M07 — the inline <head> script, per style, that puts exactly that
 * style's classes on <html> before first paint (app/layout.tsx), ahead of the
 * look's pre-paint script, which needs the theme's class. It adds the style's
 * classes, removes the rest of STYLE_HTML_CLASS_NAMES and leaves every other
 * class (next-themes' tone, the font variables) alone. React never owns these
 * classes: <html className> carries only the font variables, the same on every
 * render, so a server re-render (router.refresh) never rewrites the attribute
 * and drops the tone. Each body is a literal: nothing is built into script
 * text, and the parsed style only picks the entry, so no cookie text ever goes
 * into it. ui-style-play.test.ts runs each one against styleHtmlClasses, so the
 * literals cannot drift from it. PRD-261 A-03 replaces it with the boot script.
 */
export const STYLE_CLASS_SCRIPT: Readonly<Record<UiStyle, string>> = {
  classic: "(function(){var c=document.documentElement.classList;c.remove('studio');c.remove('play');})();",
  studio: "(function(){var c=document.documentElement.classList;c.remove('play');c.add('studio');})();",
  'theme:builtin:play': "(function(){var c=document.documentElement.classList;c.add('studio');c.add('play');})();",
}

/**
 * PRD-260 D2: whether the Studio layout renders: Studio, and every theme drawn on Studio, where
 * the theme has no design of its own. A theme the registry doesn't know yet is drawn on
 * DEFAULT_THEME_BASE.
 */
export function rendersStudio(style: UiStyle): boolean {
  if (style === 'classic') return false
  if (style === 'studio') return true
  return (findThemeEntry(style)?.base ?? DEFAULT_THEME_BASE) === 'studio'
}

/** The <html> classes for a style: [] Classic, ['studio'] Studio, the base's plus the theme's own for a theme. */
export function styleHtmlClasses(style: UiStyle): string[] {
  const base = rendersStudio(style) ? [STUDIO_HTML_CLASS] : []
  const entry = findThemeEntry(style)
  return entry === null ? base : [...base, entry.htmlClass]
}

/** Classic, Studio, a builtin theme's style, or an installed theme's (`theme:install:<uuid>`). */
export function isUiStyle(value: unknown): value is UiStyle {
  if (typeof value !== 'string') return false
  return (UI_STYLES as readonly string[]).includes(value) || isInstalledThemeStyle(value)
}

/**
 * The stored style as the axis reads it: a legacy value becomes the style it
 * stood for (LEGACY_STYLE_ALIASES); unknown, missing or tampered → the default style.
 */
export function parseUiStyle(value: string | null | undefined): UiStyle {
  if (typeof value !== 'string') return DEFAULT_UI_STYLE
  const resolved = resolveStyleAlias(value)
  return isUiStyle(resolved) ? resolved : DEFAULT_UI_STYLE
}
