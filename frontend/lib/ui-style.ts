/**
 * PRD-244 D1 — the Style axis: Classic | Studio | Play.
 *
 * Style is *which design system* (Classic = the original design; Studio = the
 * paper/ink redesign; Play = the mascot theme, PRD-260). Tone is
 * Light | Dark | System and stays with next-themes (`.dark` on <html>, the
 * class every Tailwind rule depends on). The style is its own persisted
 * setting: a cookie the server reads (app/layout.tsx) so the chosen chrome
 * renders on first paint with no classic flash, mirrored to localStorage.
 *
 * PRD-260 D1/D2: Play is layered on Studio — <html class="studio play"> — so
 * a page with no Play design renders Studio's version inside Play's frame.
 */
export type UiStyle = 'classic' | 'studio' | 'play'

export const UI_STYLES: readonly UiStyle[] = ['classic', 'studio', 'play'] as const

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

/** The class Play's frame and the look's variables hang off (`html.play`). */
export const PLAY_HTML_CLASS = 'play'

/** Every class the Style axis may put on <html>; switching style toggles each. */
export const STYLE_HTML_CLASS_NAMES: readonly string[] = [STUDIO_HTML_CLASS, PLAY_HTML_CLASS]

const HTML_CLASSES_BY_STYLE: Readonly<Record<UiStyle, readonly string[]>> = {
  classic: [],
  studio: [STUDIO_HTML_CLASS],
  play: [STUDIO_HTML_CLASS, PLAY_HTML_CLASS],
}

/** The <html> classes for a style: [] Classic, ['studio'] Studio, ['studio', 'play'] Play. */
export function styleHtmlClasses(style: UiStyle): string[] {
  return [...HTML_CLASSES_BY_STYLE[style]]
}

function styleClassScript(style: UiStyle): string {
  const wanted = styleHtmlClasses(style)
  const unwanted = STYLE_HTML_CLASS_NAMES.filter((name) => !wanted.includes(name))
  return `(function(){var c=document.documentElement.classList;${JSON.stringify(unwanted)}.forEach(function(n){c.remove(n);});${JSON.stringify(wanted)}.forEach(function(n){c.add(n);});})();`
}

/**
 * PRD-260 M07 — the inline <head> script, per style, that puts exactly that
 * style's classes on <html> before first paint (app/layout.tsx), ahead of the
 * look's pre-paint script, which needs 'play'. It adds the style's classes,
 * removes the rest of STYLE_HTML_CLASS_NAMES and leaves every other class
 * (next-themes' tone, the font variables) alone. React never owns these
 * classes: <html className> carries only the font variables, the same on every
 * render, so a server re-render (router.refresh) never rewrites the attribute
 * and drops the tone. Each body is a constant built here at module load from
 * the class table above; the parsed style only picks the entry, and no cookie
 * text ever goes into it.
 */
export const STYLE_CLASS_SCRIPT: Readonly<Record<UiStyle, string>> = {
  classic: styleClassScript('classic'),
  studio: styleClassScript('studio'),
  play: styleClassScript('play'),
}

/** PRD-260 D2: Studio and Play both render Studio's layout where Play has no design of its own. */
export function rendersStudio(style: UiStyle): boolean {
  return style === 'studio' || style === 'play'
}

export function isUiStyle(value: unknown): value is UiStyle {
  return typeof value === 'string' && (UI_STYLES as readonly string[]).includes(value)
}

/** Unknown, missing or tampered → the default style. */
export function parseUiStyle(value: string | null | undefined): UiStyle {
  return isUiStyle(value) ? value : DEFAULT_UI_STYLE
}
