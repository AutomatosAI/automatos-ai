/**
 * PRD-261 A-02a — the theme registry: every theme the Style axis can name, one entry each.
 *
 * The builtin themes are compiled in and listed here. Installed themes join later (A-02c) by
 * passing their entries to findThemeEntry; nothing here is changed at runtime. lib/ui-style.ts
 * builds the axis (parsing, <html> classes) from this file, so a new theme needs no edit there.
 */
import { BUILTIN_PLAY_ENTRY, LEGACY_STYLE_ALIASES } from './builtin/play-entry'
import type { ThemeBase, ThemeEntry, ThemeRef, UiStyle } from './registry-types'

const THEME_STYLE_PREFIX = 'theme:'

/** An installed theme's style: `theme:install:` and a lowercase uuid, nothing else. */
export const INSTALLED_THEME_STYLE_PATTERN = /^theme:install:[0-9a-f-]{36}$/

/** The themes compiled into the app, in the order the theme menu lists them. */
export const BUILTIN_THEMES: readonly ThemeEntry[] = [BUILTIN_PLAY_ENTRY]

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

/** The styles of the builtin themes. */
export const BUILTIN_THEME_STYLES: readonly UiStyle[] = BUILTIN_THEMES.map((entry) => themeStyleOf(entry.ref))

export function isBuiltinThemeStyle(value: string): boolean {
  return BUILTIN_THEME_STYLES.includes(value as UiStyle)
}

export function isInstalledThemeStyle(value: string): value is `theme:install:${string}` {
  return INSTALLED_THEME_STYLE_PATTERN.test(value)
}

/**
 * The entry of the theme a style names: a builtin one, or one of `installed` (A-02c passes the
 * workspace's). Null for Classic, Studio and a theme the registry doesn't know.
 */
export function findThemeEntry(style: UiStyle, installed: readonly ThemeEntry[] = []): ThemeEntry | null {
  const ref = themeRefOf(style)
  if (ref === null) return null
  return BUILTIN_THEMES.find((entry) => entry.ref === ref) ?? installed.find((entry) => entry.ref === ref) ?? null
}

/**
 * A stored style as the axis reads it today: a legacy value (`LEGACY_STYLE_ALIASES`) becomes the
 * style it stood for, anything else is returned as it came. Own properties only, so a value such
 * as `toString` or `__proto__` is never taken for an alias.
 */
export function resolveStyleAlias(value: string): string {
  return Object.prototype.hasOwnProperty.call(LEGACY_STYLE_ALIASES, value) ? LEGACY_STYLE_ALIASES[value] : value
}

/** Every class a builtin theme puts on <html>, once each. */
export const BUILTIN_THEME_HTML_CLASSES: readonly string[] = BUILTIN_THEMES.map((entry) => entry.htmlClass).filter(
  (name, index, names) => names.indexOf(name) === index,
)
