/**
 * PRD-261 A-01 — the spike's theme.json and look preset, checked before anything is applied.
 *
 * Each parser returns a result and never throws on bad input. The theme's files (entry,
 * stylesheet, preset) must be plain relative paths that resolve under the theme's fixed folder on
 * the page's own origin, so a manifest cannot point the host at another site or another folder.
 * A preset value is a colour, a length or a number, nothing that could carry `url(` or close a
 * rule, and it is applied only through style.setProperty.
 */
import { FRAME_SLOT, HTML_CLASS_PATTERN, RESERVED_HTML_CLASSES, SPIKE_BASE_STYLE, THEME_SPIKE_ROOT } from './constants'
import type { LookPreset, SpikeManifest } from './types'

interface ParseFailure {
  readonly ok: false
  readonly detail: string
}

export type ManifestParse = { readonly ok: true; readonly manifest: SpikeManifest } | ParseFailure
export type PresetParse = { readonly ok: true; readonly preset: LookPreset } | ParseFailure

const MANIFEST_SCHEMA = 'automatos-theme-spike/v1'
const KNOWN_SLOTS: readonly string[] = [FRAME_SLOT]
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

/**
 * The absolute URL of a theme file, or null when the value is not a plain relative path under
 * the theme's folder: not a string, a leading '/', '//', '\', ':', a '..' segment or any
 * character outside [a-z0-9._/-] is refused, and so is a result off this origin or folder.
 */
export function resolveUnderRoot(relativePath: unknown): string | null {
  if (typeof relativePath !== 'string' || !RELATIVE_PATH.test(relativePath)) return null
  if (relativePath.includes('//') || relativePath.split('/').includes('..')) return null
  const origin = window.location.origin
  const url = new URL(relativePath, origin + THEME_SPIKE_ROOT)
  return url.origin === origin && url.pathname.startsWith(THEME_SPIKE_ROOT) ? url.href : null
}

function identityProblem(raw: Record<string, unknown>): string | null {
  if (raw.schema !== MANIFEST_SCHEMA) return `schema must be "${MANIFEST_SCHEMA}"`
  const missing = IDENTITY_FIELDS.find((field) => !isNonEmptyString(raw[field]))
  if (missing !== undefined) return `${missing} must be a non-empty string`
  if (!isRecord(raw.engines) || !isNonEmptyString(raw.engines.sdk)) return 'engines.sdk must be a non-empty string'
  if (raw.base !== SPIKE_BASE_STYLE) return `base must be "${SPIKE_BASE_STYLE}"`
  return null
}

function htmlClassProblem(value: unknown): string | null {
  if (typeof value !== 'string' || !HTML_CLASS_PATTERN.test(value)) {
    return 'htmlClass must be 3 to 32 lower-case letters, digits or hyphens, starting with a letter'
  }
  return RESERVED_HTML_CLASSES.includes(value) ? `htmlClass "${value}" is reserved` : null
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

/** A checked theme.json, or why it was refused. */
export function parseSpikeManifest(raw: unknown): ManifestParse {
  if (!isRecord(raw)) return failure('theme.json must be an object')
  const problem = identityProblem(raw) ?? htmlClassProblem(raw.htmlClass) ?? slotsProblem(raw.slots)
  if (problem !== null) return failure(problem)
  const entry = resolveUnderRoot(raw.entry)
  const styleSheet = resolveUnderRoot(raw.styleSheet)
  const preset = resolveUnderRoot(raw.preset)
  if (entry === null || styleSheet === null || preset === null) {
    return failure(`entry, styleSheet and preset must be files inside ${THEME_SPIKE_ROOT}`)
  }
  return { ok: true, manifest: manifestFrom(raw, entry, styleSheet, preset) }
}

function isSafeCssValue(value: unknown): value is string {
  return typeof value === 'string' && CSS_VALUE_PATTERNS.some((pattern) => pattern.test(value))
}

/** A checked look preset, or why it was refused. */
export function parseLookPreset(raw: unknown): PresetParse {
  if (!isRecord(raw) || !isRecord(raw.vars)) return failure('the preset must be an object with a vars object')
  const entries = Object.entries(raw.vars)
  const refused = entries.find(([name, value]) => !CSS_VARIABLE_NAME.test(name) || !isSafeCssValue(value))
  if (refused !== undefined) return failure(`preset variable "${refused[0]}" has a name or value this host does not accept`)
  const vars = Object.freeze(Object.fromEntries(entries)) as Readonly<Record<string, string>>
  return { ok: true, preset: Object.freeze({ vars }) }
}
