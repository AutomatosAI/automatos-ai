/**
 * PRD-261 A-02a — the theme registry: every theme the Style axis can name, one entry each.
 *
 * No theme is compiled into the app (A-19f1: builtin Play left core; Play ships as a package).
 * An installed theme's entry comes with the page (W-A6b, lib/theme-host/installed-entry.ts) and is
 * passed to findThemeEntry; nothing here is changed at runtime. lib/ui-style.ts builds the axis
 * (parsing, <html> classes) from this file, so a new theme needs no edit there.
 */
import type { ThemeBase, ThemeEntry, ThemeRef, UiStyle } from './registry-types'

const THEME_STYLE_PREFIX = 'theme:'
/** What an installed theme's style holds before its install id. */
const INSTALL_STYLE_PREFIX = `${THEME_STYLE_PREFIX}install:`

/** An installed theme's style: `theme:install:` and a lowercase uuid, nothing else. */
export const INSTALLED_THEME_STYLE_PATTERN = /^theme:install:[0-9a-f-]{36}$/

/**
 * The base an installed theme is drawn on while the registry has no entry for it yet: Studio,
 * the default style, so an unknown theme looks like the default until its entry arrives.
 */
export const DEFAULT_THEME_BASE: ThemeBase = 'studio'

/** The style that names a theme. */
export function themeStyleOf(ref: ThemeRef): UiStyle {
  return `${THEME_STYLE_PREFIX}${ref}`
}

/** The theme a style names, or null for Classic and Studio. */
export function themeRefOf(style: UiStyle): ThemeRef | null {
  return style.startsWith(THEME_STYLE_PREFIX) ? (style.slice(THEME_STYLE_PREFIX.length) as ThemeRef) : null
}

export function isInstalledThemeStyle(value: string): value is `theme:install:${string}` {
  return INSTALLED_THEME_STYLE_PATTERN.test(value)
}

/** The install a style names (`theme:install:<uuid>`), or null for Classic, Studio and anything else. */
export function installIdOfStyle(style: UiStyle): string | null {
  return isInstalledThemeStyle(style) ? style.slice(INSTALL_STYLE_PREFIX.length) : null
}

/**
 * The entry of the theme a style names, one of `installed` (the workspace's, A-02c). Null for
 * Classic, Studio and a theme the registry doesn't know.
 */
export function findThemeEntry(style: UiStyle, installed: readonly ThemeEntry[] = []): ThemeEntry | null {
  const ref = themeRefOf(style)
  if (ref === null) return null
  return installed.find((entry) => entry.ref === ref) ?? null
}
