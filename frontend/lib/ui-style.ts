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
import { SKIN_ID_PATTERN } from '@/lib/skins/schema'

/** The two design systems. A skin sits on one of them (its `base`), never a third layout. */
export type BaseStyle = 'classic' | 'studio'

/**
 * PRD-259 §4: a workspace's installed skin is a third kind of style, `skin:<id>`. It
 * restyles its base (colours, fonts, shape) and never changes a layout.
 */
export type SkinStyle = `skin:${string}`
export type UiStyle = BaseStyle | SkinStyle

export const UI_STYLES: readonly BaseStyle[] = ['classic', 'studio'] as const
export const SKIN_STYLE_PREFIX = 'skin:'

/**
 * The defaults (Gerard, 2026-09-17): Studio in the Dark tone. A browser with
 * no stored choice gets them on the server (the style) and from next-themes
 * (the tone); every browser is moved to them ONCE — see
 * APPEARANCE_DEFAULTS_VERSION — and picks freely afterwards.
 */
export const DEFAULT_UI_STYLE: BaseStyle = 'studio'
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

/**
 * PRD-259 decision K1: the server sets the base class without a database call, so the
 * active skin's base rides in its own cookie beside `automatos-style=skin:<id>`.
 */
export const UI_SKIN_BASE_COOKIE = 'automatos-skin-base'
export const DEFAULT_SKIN_BASE: BaseStyle = 'studio'

/** The class the Studio stylesheet hangs off (`.studio {}` in globals.css). */
export const STUDIO_HTML_CLASS = 'studio'

export function isBaseStyle(value: unknown): value is BaseStyle {
  return value === 'classic' || value === 'studio'
}

export function isSkinStyle(value: unknown): value is SkinStyle {
  return typeof value === 'string'
    && value.startsWith(SKIN_STYLE_PREFIX)
    && SKIN_ID_PATTERN.test(value.slice(SKIN_STYLE_PREFIX.length))
}

export function isUiStyle(value: unknown): value is UiStyle {
  return isBaseStyle(value) || isSkinStyle(value)
}

/** Unknown, missing or tampered → the default style. A skin id must be a UUID. */
export function parseUiStyle(value: string | null | undefined): UiStyle {
  return isUiStyle(value) ? value : DEFAULT_UI_STYLE
}

/** The base cookie, strictly: 'studio' or 'classic', anything else → Studio. */
export function parseSkinBase(value: string | null | undefined): BaseStyle {
  return isBaseStyle(value) ? value : DEFAULT_SKIN_BASE
}

export function skinStyle(skinId: string): SkinStyle {
  return `${SKIN_STYLE_PREFIX}${skinId}`
}

/** The skin id a style names, or null for Classic and Studio. */
export function skinIdOf(style: UiStyle): string | null {
  return isSkinStyle(style) ? style.slice(SKIN_STYLE_PREFIX.length) : null
}

/** The design system a style draws on: itself, or a skin's base (from its cookie). */
export function baseOf(style: UiStyle, baseCookie: string | null | undefined): BaseStyle {
  return isSkinStyle(style) ? parseSkinBase(baseCookie) : style
}

export interface HtmlStyleAttrs {
  readonly style: UiStyle
  readonly base: BaseStyle
  readonly studioClass: boolean
  readonly skinId: string | null
}

/**
 * What app/layout.tsx puts on <html> from the two cookies: the studio class when the base
 * is Studio, and data-skin when a skin is chosen. Tampered values fall back to the default.
 */
export function htmlAttrsFor(styleCookie: string | null | undefined, baseCookie: string | null | undefined): HtmlStyleAttrs {
  const style = parseUiStyle(styleCookie)
  const base = baseOf(style, baseCookie)
  return { style, base, studioClass: base === 'studio', skinId: skinIdOf(style) }
}
