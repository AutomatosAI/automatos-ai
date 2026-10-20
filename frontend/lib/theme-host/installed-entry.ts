/**
 * PRD-261 W-A6b (gap G1) — an installed theme's registry entry.
 *
 * The builtin themes' entries are compiled in (registry.ts). An installed theme's comes from what
 * the server already resolves for it: the backend's boot route answers a theme that runs with its
 * entry's data, read from the theme's checked theme.json (orchestrator serve.py theme_entry), and
 * the root layout's boot carries it (boot-server.ts). The layout makes it the entry here
 * (installedThemeEntry) and hands it to Providers, so the server render and the client's first
 * render name the same active theme: no hydration mismatch, no flip after mount.
 *
 * The data passes sanitizeInstalledEntry with the rest of the boot (boot-sanitize.ts): a scalar
 * that fails its theme.json rule drops the whole entry (no entry: the theme is drawn as before,
 * with no registry entry), and a list item that fails is dropped alone.
 *
 * PRD-261 Phase A: the host draws an installed theme's frame and pages from the desktop width up
 * (INSTALLED_FRAME_MIN_WIDTH_PX, where MainLayout draws a frame), whatever minWidth its theme.json
 * names, as it did before installed themes had entries: a smaller one would draw the theme's pages
 * on a phone with no frame around them, a larger one its frame around the core's pages.
 */
import type { InstalledThemeEntryData } from './boot-types'
import { HTML_CLASS_PATTERN, THEME_SLOT_IDS } from './constants'
import { isRecord, isWholeNumber, plainTextProblem } from './css-value'
import {
  LABEL_MAX,
  MAX_NAV,
  MAX_ROUTES,
  MIN_WIDTH_MAX,
  NAME_MAX,
  NAV_ID_PATTERN,
  NAV_PATH_PATTERN,
  PAGE_SCOPE,
  ROUTE_PATTERN,
  THEME_BASES,
  THEME_TONES,
} from './package-manifest'
import { isInstalledThemeStyle, themeRefOf } from './registry'
import type { SlotId, ThemeBase, ThemeEntry, ThemeNavItem, ThemeTonePolicy, UiStyle } from './registry-types'

export type { InstalledThemeEntryData } from './boot-types'

/**
 * Where MainLayout draws an installed theme's frame: its desktop shell, from useIsTabletOrBelow's
 * 1024 px (hooks/use-mobile.ts). The host loads only Studio-based installs for now (Phase A): the
 * loader refuses any other base (lib/theme-host/manifest.ts, STUDIO_BASE_ONLY).
 */
export const INSTALLED_FRAME_MIN_WIDTH_PX = 1024

type Scalars = Omit<InstalledThemeEntryData, 'ownsPages' | 'routes' | 'nav' | 'slots'>

function textOf(value: unknown, maxChars: number): string | null {
  return typeof value === 'string' && plainTextProblem(value, 'text', maxChars) === null ? value : null
}

function oneOf<T extends string>(list: readonly string[], value: unknown): T | null {
  return typeof value === 'string' && list.includes(value) ? (value as T) : null
}

function widthOf(value: unknown): number | null {
  return isWholeNumber(value) && value >= 0 && value <= MIN_WIDTH_MAX ? value : null
}

/** The entry's single fields, or null when any fails its theme.json rule. */
function scalarsOf(raw: Readonly<Record<string, unknown>>): Scalars | null {
  const name = textOf(raw.name, NAME_MAX)
  const base = oneOf<ThemeBase>(THEME_BASES, raw.base)
  const htmlClass = typeof raw.htmlClass === 'string' && HTML_CLASS_PATTERN.test(raw.htmlClass) ? raw.htmlClass : null
  const minWidth = widthOf(raw.minWidth)
  const tone = oneOf<ThemeTonePolicy>(THEME_TONES, raw.tone)
  const reviewed = typeof raw.reviewed === 'boolean' ? raw.reviewed : null
  if (name === null || base === null || htmlClass === null || minWidth === null || tone === null || reviewed === null) return null
  return { name, base, htmlClass, minWidth, tone, reviewed }
}

/** The items of `value` that pass `keep`, once each, at most `max` of them. */
function listOf<T>(value: unknown, max: number, keep: (item: unknown) => item is T): T[] {
  if (!Array.isArray(value)) return []
  return value.filter(keep).filter((item, index, kept) => kept.indexOf(item) === index).slice(0, max)
}

const isSlot = (item: unknown): item is SlotId => typeof item === 'string' && (THEME_SLOT_IDS as readonly string[]).includes(item)
const isOwnedPage = (item: unknown): item is string => typeof item === 'string' && PAGE_SCOPE.includes(item)
const isRoute = (item: unknown): item is string => typeof item === 'string' && ROUTE_PATTERN.test(item)

function navItemOf(item: unknown): ThemeNavItem | null {
  if (!isRecord(item)) return null
  const { id, path } = item
  const label = textOf(item.label, LABEL_MAX)
  if (typeof id !== 'string' || !NAV_ID_PATTERN.test(id) || typeof path !== 'string' || !NAV_PATH_PATTERN.test(path)) return null
  return label === null ? null : { id, label, path }
}

function navOf(value: unknown): ThemeNavItem[] {
  const items = Array.isArray(value) ? value.map(navItemOf) : []
  return items.filter((item): item is ThemeNavItem => item !== null).slice(0, MAX_NAV)
}

/** The boot route's entry as the page may use it; null when it is missing or a single field fails. */
export function sanitizeInstalledEntry(value: unknown): InstalledThemeEntryData | null {
  if (!isRecord(value)) return null
  const scalars = scalarsOf(value)
  if (scalars === null) return null
  return {
    ...scalars,
    ownsPages: listOf(value.ownsPages, PAGE_SCOPE.length, isOwnedPage),
    routes: listOf(value.routes, MAX_ROUTES, isRoute),
    nav: navOf(value.nav),
    slots: listOf(value.slots, THEME_SLOT_IDS.length, isSlot),
  }
}

/**
 * The registry entry of the installed theme `style` names, from its boot's entry data; null for
 * any other style and when the boot carries none (the theme may not run, or the boot failed).
 */
export function installedThemeEntry(style: UiStyle, data: InstalledThemeEntryData | null): ThemeEntry | null {
  const ref = isInstalledThemeStyle(style) ? themeRefOf(style) : null
  if (ref === null || data === null) return null
  return {
    ...data,
    ref,
    kind: 'installed',
    minWidth: INSTALLED_FRAME_MIN_WIDTH_PX,
  }
}
