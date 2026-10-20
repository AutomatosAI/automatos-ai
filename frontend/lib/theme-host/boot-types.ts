/**
 * PRD-261 A-03 — the shape of a theme's first-paint data (plan §2.2), in a file of its own so
 * the modules that build a theme's prepaint table (lib/theme-host/builtin/play-prepaint.ts) can name the types
 * without loading the boot resolver. lib/theme-host/boot.ts re-exports every one of them.
 */
import type { SlotId, ThemeBase, ThemeNavItem, ThemeTonePolicy, UiStyle } from './registry-types'

/** The tone next-themes is forced to from the first paint, when the theme sets one. */
export type ThemeTone = 'light' | 'dark'

/** Custom property name → value, applied with style.setProperty only. */
export type PrepaintVars = Readonly<Record<string, string>>

/** One knob's options (a look's field value → the variables it paints). */
export type PrepaintKnobTable = Readonly<Record<string, PrepaintVars>>

/** Two knobs' values → a value, read as table[look[a]][look[b]]. */
export type PrepaintPairTable<V> = Readonly<Record<string, Readonly<Record<string, V>>>>

/** Variables that depend on two knobs at once (Play: the accent's text on each ground). */
export interface PrepaintPair {
  readonly a: string
  readonly b: string
  readonly table: PrepaintPairTable<PrepaintVars>
}

/**
 * A numeric knob painted as data, not script. With the 'rgba-alpha' template the variable is
 * `rgba(<color.rgb[look[color.knob]]>,<alpha>)`, where alpha = look[knob] / scale, capped at
 * caps[look[capTable[0]]][look[capTable[1]]] when capTable is set (Play's glow: never past the
 * ground and accent's safe glow, M50).
 */
export interface PrepaintNumericRule {
  readonly var: string
  readonly knob: string
  readonly min: number
  readonly max: number
  readonly integer: boolean
  readonly scale: number
  readonly capTable: readonly [string, string] | null
  readonly caps: PrepaintPairTable<number> | null
  readonly color: { readonly knob: string; readonly rgb: Readonly<Record<string, string>> }
  readonly template: 'rgba-alpha'
}

/**
 * PRD-261 A-19c (host 1.1.0) — the tone each choice of one choice knob gives a look
 * (boot/prepaint.json `tone`), so a look runtime sets the tone from the look alone (A-19d2).
 * The server refuses a rule whose knob is not a choice knob of the table or that leaves a
 * choice without 'light' or 'dark' (css_value.py tone_rule_errors; css-value.ts toneRuleErrors).
 */
export interface PrepaintToneRule {
  readonly knob: string
  readonly values: Readonly<Record<string, ThemeTone>>
}

/**
 * Everything the boot script may paint for a look, keyed the way it reads a stored look: each
 * knob's table, the pair tables, the numeric rules, and the default look painted when nothing
 * valid is stored. `tone` (A-19c) is absent when the table sets no tone; the boot script never
 * reads it.
 */
export interface PrepaintTable {
  readonly knobs: Readonly<Record<string, PrepaintKnobTable>>
  readonly pairs: readonly PrepaintPair[]
  readonly numeric: readonly PrepaintNumericRule[]
  readonly defaults: Readonly<Record<string, string | number>>
  readonly tone?: PrepaintToneRule
}

/** Where the boot script finds the cached active look (Play: lib/theme-host/builtin/play-look-storage.ts). */
export interface ThemeLookStorage {
  /** Holds the current scope; the scoped key is scopedPrefix + that scope. */
  readonly pointerKey: string
  readonly scopedPrefix: string
  /** The unscoped key, read when no pointer is set. */
  readonly fallbackKey: string
}

/** A stored look paints only when its schema and interface are these. */
export interface ThemeLookGuard {
  readonly schema: string
  readonly iface: string
}

/**
 * PRD-261 A-19c — one preset as the boot carries it: a look document of the theme's format
 * (`{schema, interface?, id, name, sub, <every knob>}`), checked by the server against the
 * theme's looks/schema.json.
 */
export type ThemeBootPreset = Readonly<Record<string, string | number>>

/**
 * PRD-261 A-19c (host 1.1.0, plan §2.2.3) — a theme's presets, from its looks/presets/*.look.json:
 * the default look first (`defaultLook` is its id), the rest in file-name order, at most
 * BOOT_PRESETS_MAX of at most BOOT_PRESET_MAX_BYTES each (boot-sanitize.ts).
 */
export interface ThemeBootLooks {
  readonly defaultLook: string
  readonly presets: readonly ThemeBootPreset[]
}

/**
 * PRD-261 W-A6b (gap G1) — what the theme registry needs of an installed theme that runs, as the
 * backend's boot route answers it from the theme's checked theme.json (serve.py theme_entry).
 * lib/theme-host/installed-entry.ts checks it (sanitizeInstalledEntry) and makes it the theme's
 * registry entry (installedThemeEntry).
 */
export interface InstalledThemeEntryData {
  /** The theme's id (theme.json `id`), which its looks are kept under, in this browser and on the server (W-A6b, G2). */
  readonly themeId: string
  readonly name: string
  readonly base: ThemeBase
  readonly htmlClass: string
  /** The theme's own; in Phase A the host draws an installed theme from INSTALLED_FRAME_MIN_WIDTH_PX up. */
  readonly minWidth: number
  readonly tone: ThemeTonePolicy
  readonly reviewed: boolean
  /** The core pages the theme draws (its theme.json `pages`). */
  readonly ownsPages: readonly string[]
  readonly routes: readonly string[]
  readonly nav: readonly ThemeNavItem[]
  readonly slots: readonly SlotId[]
}

export interface ThemeBoot {
  readonly addClasses: readonly string[]
  readonly removeClasses: readonly string[]
  readonly tone: { readonly policy: ThemeTonePolicy; readonly value: ThemeTone | null }
  readonly prepaint: PrepaintTable | null
  readonly lookStorage: ThemeLookStorage | null
  readonly lookGuard: ThemeLookGuard | null
  /** Module URL → 'sha256-<base64>' for every module of the active theme (PA-13; filled by A-08c). */
  readonly integrity: Readonly<Record<string, string>> | null
  /** An installed theme that runs: its registry entry's data (W-A6b). Null for every other boot. */
  readonly entry: InstalledThemeEntryData | null
  /**
   * An installed theme that runs, with looks: its presets (A-19c). sanitizeThemeBoot always sets
   * it, null for every other boot; a boot built in the app may leave it out.
   */
  readonly looks?: ThemeBootLooks | null
}

/** A theme compiled into the app: its first-paint data, server-safe (no CSS, no client code). */
export interface BuiltinThemeBootData {
  readonly prepaint: PrepaintTable | null
  readonly lookStorage: ThemeLookStorage | null
  readonly lookGuard: ThemeLookGuard | null
  /** The cookie the client writes the look's tone to, read on the server (PRD-260 K4); null: none. */
  readonly toneCookie: string | null
}

/** Reads one request cookie by name. */
export type CookieReader = (name: string) => string | undefined

/** The boot for a style. A-08c plugs installed themes in through createThemeBootResolver. */
export interface ThemeBootResolver {
  (style: UiStyle): Promise<ThemeBoot>
}
