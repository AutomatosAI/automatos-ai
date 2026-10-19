/**
 * PRD-261 A-01, A-04a — an installed theme's description and look preset, checked before anything
 * is applied.
 *
 * Each parser returns a result and never throws on bad input. The active-theme answer must name
 * the install the style asks for, with a well-formed package digest; that digest fixes the
 * theme's files folder (`/api/themes/files/<digest>/` on the page's own origin).
 *
 * Two manifest schemas are read:
 * - `automatos-theme/v1`, a real package (A-06b): its files sit at the package's fixed layout
 *   (orchestrator/services/themes/package.py FILE_KINDS): the entry at `dist/entry.js`, the
 *   stylesheet at `dist/theme.css`, and the first look at `looks/presets/<id>.look.json`, where
 *   `<id>` is the manifest's `look`, else the answer's `defaultLook` (the server's first preset).
 * - `automatos-theme-spike/v1`, the browser lane's fixtures: `entry`, `styleSheet` and `preset`
 *   name the files, as plain relative paths.
 * Either way every file resolves under the digest's folder, so a manifest cannot point the host
 * at another site, another package or another folder.
 *
 * A spike preset's `vars` value is a colour, a length or a number, nothing that could carry
 * `url(` or close a rule, and it is applied only through style.setProperty. A package's look
 * (`{schema, id, name, sub, <knobs>}`, checked against its schema when the package was imported)
 * sets no host variables: the theme's own code and CSS read it.
 */
import {
  FRAME_SLOT,
  HTML_CLASS_PATTERN,
  THEME_DIGEST_PATTERN,
  THEME_FILES_PATH,
  THEME_SLOT_IDS,
} from './constants'
import { LOOK_ID_PATTERN } from '@/lib/looks/types'

import { HTML_CLASS_OWNERS, MANIFEST_SCHEMA as PACKAGE_SCHEMA, RESERVED_HTML_CLASSES } from './package-manifest'
import { DEFAULT_THEME_BASE } from './registry'
import type { LookPreset, SpikeManifest } from './types'

interface ParseFailure {
  readonly ok: false
  readonly detail: string
}

export type ManifestParse = { readonly ok: true; readonly manifest: SpikeManifest } | ParseFailure
export type PresetParse = { readonly ok: true; readonly preset: LookPreset } | ParseFailure
/**
 * The active install's package digest, its files folder, its manifest as the server sent it
 * (unchecked), and the server's default look id (null when it sent none).
 */
export type ActiveInstallParse =
  | {
      readonly ok: true
      readonly digest: string
      readonly filesRoot: string
      readonly manifest: unknown
      readonly defaultLook: string | null
    }
  | ParseFailure

const SPIKE_SCHEMA = 'automatos-theme-spike/v1'
const MANIFEST_SCHEMAS: readonly string[] = [PACKAGE_SCHEMA, SPIKE_SCHEMA]
/** services/themes/package.py FILE_KINDS: where a package's entry, stylesheet and presets sit. */
export const PACKAGE_ENTRY_PATH = 'dist/entry.js'
export const PACKAGE_STYLE_PATH = 'dist/theme.css'
const PACKAGE_PRESETS_DIR = 'looks/presets/'
const PACKAGE_PRESET_SUFFIX = '.look.json'
const KNOWN_SLOTS: readonly string[] = THEME_SLOT_IDS
const IDENTITY_FIELDS = ['id', 'name', 'version'] as const
const RELATIVE_PATH = /^[a-z0-9][a-z0-9._/-]*$/
const CSS_VARIABLE_NAME = /^--[a-z][a-z0-9-]{1,63}$/
const CSS_VALUE_PATTERNS: readonly RegExp[] = [
  /^#[0-9a-fA-F]{3,8}$/,
  /^-?(\d+(\.\d+)?|\.\d+)(px|rem|em|%)?$/,
  /^(rgb|rgba|hsl|hsla)\([\d.%,/ ]+\)$/,
]

function failure(detail: string): ParseFailure {
  return { ok: false, detail }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0
}

/** The files folder of a package digest, or null when the digest is not sha256 hex. */
export function themeFilesRoot(digest: unknown): string | null {
  return typeof digest === 'string' && THEME_DIGEST_PATTERN.test(digest) ? `${THEME_FILES_PATH}${digest}/` : null
}

/**
 * The absolute URL of a theme file, or null when the value is not a plain relative path under
 * `root` (a folder path such as themeFilesRoot gives): not a string, a leading '/', '//', '\',
 * ':', a '..' segment or any character outside [a-z0-9._/-] is refused, and so is a result off
 * this origin or folder.
 */
export function resolveUnderRoot(relativePath: unknown, root: string): string | null {
  if (typeof relativePath !== 'string' || !RELATIVE_PATH.test(relativePath)) return null
  if (relativePath.includes('//') || relativePath.split('/').includes('..')) return null
  return urlUnderRoot(relativePath, root)
}

/** The absolute URL of `relativePath` under `root`, or null when it lands off this origin or folder. */
function urlUnderRoot(relativePath: string, root: string): string | null {
  const origin = window.location.origin
  const url = new URL(relativePath, origin + root)
  return url.origin === origin && url.pathname.startsWith(root) ? url.href : null
}

/**
 * PRD-261 Phase A limit: the host draws an installed theme on Studio only (its first paint, its
 * frame width and its slots assume Studio's chrome). The server's manifest check accepts both
 * THEME_BASES (services/themes/contract.py), so a `base: "classic"` package installs and can be
 * chosen; the loader then refuses it with this reason, which the provider shows.
 */
const STUDIO_BASE_ONLY =
  `base must be "${DEFAULT_THEME_BASE}": this host runs only Studio-based themes for now ` +
  '(a "classic" base installs but cannot load yet)'

function identityProblem(raw: Record<string, unknown>): string | null {
  if (typeof raw.schema !== 'string' || !MANIFEST_SCHEMAS.includes(raw.schema)) {
    return `schema must be "${PACKAGE_SCHEMA}" (or the browser lane's "${SPIKE_SCHEMA}")`
  }
  const missing = IDENTITY_FIELDS.find((field) => !isNonEmptyString(raw[field]))
  if (missing !== undefined) return `${missing} must be a non-empty string`
  if (!isRecord(raw.engines) || !isNonEmptyString(raw.engines.sdk)) return 'engines.sdk must be a non-empty string'
  if (raw.base !== DEFAULT_THEME_BASE) return STUDIO_BASE_ONLY
  return null
}

/** A reserved class is refused unless HTML_CLASS_OWNERS gives it to this theme's id (PA-8). */
function htmlClassProblem(value: unknown, id: unknown): string | null {
  if (typeof value !== 'string' || !HTML_CLASS_PATTERN.test(value)) {
    return 'htmlClass must be 3 to 32 lower-case letters, digits or hyphens, starting with a letter'
  }
  if (!RESERVED_HTML_CLASSES.includes(value)) return null
  const owner = HTML_CLASS_OWNERS.get(value)
  return owner !== undefined && owner === id ? null : `htmlClass "${value}" is reserved`
}

function slotsProblem(value: unknown): string | null {
  if (!Array.isArray(value) || !value.every((slot) => typeof slot === 'string')) return 'slots must be a list of slot names'
  const unknownSlot = (value as string[]).find((slot) => !KNOWN_SLOTS.includes(slot))
  if (unknownSlot !== undefined) return `slot "${unknownSlot}" is not one this host serves`
  return value.includes(FRAME_SLOT) ? null : `slots must include "${FRAME_SLOT}"`
}

function manifestFrom(raw: Record<string, unknown>, entry: string, styleSheet: string, preset: string): SpikeManifest {
  return Object.freeze({
    id: raw.id as string,
    name: raw.name as string,
    version: raw.version as string,
    sdkRange: (raw.engines as { sdk: string }).sdk,
    htmlClass: raw.htmlClass as string,
    slots: Object.freeze([...(raw.slots as string[])]),
    entry,
    styleSheet,
    preset,
  })
}

interface ThemeFiles {
  readonly entry: string
  readonly styleSheet: string
  readonly preset: string
}
type FilesResult = { readonly ok: true; readonly files: ThemeFiles } | ParseFailure

/** A spike manifest's three named files, each a plain relative path under `root`. */
function spikeFiles(raw: Record<string, unknown>, root: string): FilesResult {
  const entry = resolveUnderRoot(raw.entry, root)
  const styleSheet = resolveUnderRoot(raw.styleSheet, root)
  const preset = resolveUnderRoot(raw.preset, root)
  if (entry === null || styleSheet === null || preset === null) {
    return failure(`entry, styleSheet and preset must be files inside ${root}`)
  }
  return { ok: true, files: { entry, styleSheet, preset } }
}

/** A package's files by its fixed layout; the preset is the manifest's `look`, else `defaultLook`. */
function packageFiles(raw: Record<string, unknown>, root: string, defaultLook: unknown): FilesResult {
  const look = raw.look !== undefined ? raw.look : defaultLook
  if (typeof look !== 'string' || !LOOK_ID_PATTERN.test(look)) {
    return failure('the theme names no default look: theme.json has no valid "look" and the server sent no defaultLook')
  }
  // LOOK_ID_PATTERN allows no slash, dot or backslash, so the id cannot leave the presets folder.
  const entry = urlUnderRoot(PACKAGE_ENTRY_PATH, root)
  const styleSheet = urlUnderRoot(PACKAGE_STYLE_PATH, root)
  const preset = urlUnderRoot(`${PACKAGE_PRESETS_DIR}${look}${PACKAGE_PRESET_SUFFIX}`, root)
  if (entry === null || styleSheet === null || preset === null) return failure(`the theme's files must be inside ${root}`)
  return { ok: true, files: { entry, styleSheet, preset } }
}

/**
 * A checked theme.json whose files resolve under `root`, or why it was refused. `defaultLook` is
 * the active answer's (parseActiveInstall), used when a package manifest names no `look`.
 */
export function parseThemeManifest(raw: unknown, root: string, defaultLook: unknown = null): ManifestParse {
  if (!isRecord(raw)) return failure('theme.json must be an object')
  const problem = identityProblem(raw) ?? htmlClassProblem(raw.htmlClass, raw.id) ?? slotsProblem(raw.slots)
  if (problem !== null) return failure(problem)
  const files = raw.schema === PACKAGE_SCHEMA ? packageFiles(raw, root, defaultLook) : spikeFiles(raw, root)
  if (!files.ok) return files
  return { ok: true, manifest: manifestFrom(raw, files.files.entry, files.files.styleSheet, files.files.preset) }
}

/** The spike-era name of parseThemeManifest, kept for the browser lane's fixtures and their tests. */
export const parseSpikeManifest = parseThemeManifest

/**
 * The answer of GET /api/themes/active (A-07b), checked for the install the style names: a theme
 * whose installId is exactly `installId`, with a sha256 digest. The manifest is checked next, by
 * parseThemeManifest under the digest's folder, with the answer's `defaultLook`.
 */
export function parseActiveInstall(raw: unknown, installId: string): ActiveInstallParse {
  if (!isRecord(raw)) return failure('the active-theme answer must be an object')
  const theme = raw.theme
  if (!isRecord(theme)) return failure('the workspace has no active installed theme')
  if (theme.installId !== installId) return failure(`the active theme is not the install ${installId}`)
  const filesRoot = themeFilesRoot(theme.digest)
  if (filesRoot === null) return failure('the active theme has no valid package digest')
  const defaultLook = typeof theme.defaultLook === 'string' ? theme.defaultLook : null
  return { ok: true, digest: theme.digest as string, filesRoot, manifest: theme.manifest, defaultLook }
}

function isSafeCssValue(value: unknown): value is string {
  return typeof value === 'string' && CSS_VALUE_PATTERNS.some((pattern) => pattern.test(value))
}

/** A package look: its schema id and its own id; its knobs are the theme's to read. */
function isPackageLook(raw: Record<string, unknown>): boolean {
  return isNonEmptyString(raw.schema) && typeof raw.id === 'string' && LOOK_ID_PATTERN.test(raw.id)
}

const NO_VARS: LookPreset = Object.freeze({ vars: Object.freeze({}) })

/**
 * A checked look preset, or why it was refused: a spike preset's `vars`, or a package look,
 * which sets no host variables (the module header).
 */
export function parseLookPreset(raw: unknown): PresetParse {
  if (isRecord(raw) && !isRecord(raw.vars) && isPackageLook(raw)) return { ok: true, preset: NO_VARS }
  if (!isRecord(raw) || !isRecord(raw.vars)) return failure('the preset must be an object with a vars object, or a look with its schema and id')
  const entries = Object.entries(raw.vars)
  const refused = entries.find(([name, value]) => !CSS_VARIABLE_NAME.test(name) || !isSafeCssValue(value))
  if (refused !== undefined) return failure(`preset variable "${refused[0]}" has a name or value this host does not accept`)
  const vars = Object.freeze(Object.fromEntries(entries)) as Readonly<Record<string, string>>
  return { ok: true, preset: Object.freeze({ vars }) }
}
