/**
 * PRD-261 A-06b — a theme package's `theme.json` (schema `automatos-theme/v1`), checked field by
 * field, and what the host promises a theme: its version, the names no theme may take, the slots
 * and pages a theme may fill.
 *
 * The TypeScript twin of `orchestrator/services/themes/manifest.py` and `contract.py`.
 * `__tests__/validator-parity.test.ts` holds both to the same corpus
 * (`orchestrator/tests/fixtures/themes/cases.json`, `manifests`) and reads contract.py as text, so
 * the lists below must equal the server's. The host version is `hostVersion` in
 * shared-modules.json (PA-2), next to the SDK's; a theme's `engines` is checked against both.
 * M-01 (host 1.1.0): the optional `mobile` block is checked in `mobile-manifest.ts`, with its own lists.
 * The storefront (30 Oct 2026): `phonePreviews` lists phone screenshots (`assets/previews/phone/*.png` or
 * `*.jpg`), and it and `previews` hold up to MAX_PREVIEWS each.
 * Nothing here throws on bad input: every refusal is a ThemeError.
 */
import { HTML_CLASS_PATTERN, THEME_HOST_VERSION } from './constants'
import {
  echo, errorIf, hasOwn, isOneOf, isRecord, isWholeNumber, own, plainTextProblem, themeError, unknownFieldErrors,
  type JsonRecord, type ThemeError,
} from './css-value'
import { satisfiesSdkRange } from './engines'
import { LOOK_ID_PATTERN } from '@/lib/look-kit/validate'
import { mobileErrors, type ThemeMobileBlock } from './mobile-manifest'
import { SDK_VERSION } from './sdk'

/** PA-2: the host contract's version (slots, pages, routes, boot data). */
export { THEME_HOST_VERSION }
export const MANIFEST_SCHEMA = 'automatos-theme/v1'

/**
 * Classes the core owns on <html>: the Style axis, Classic and the tones (and next-themes' "system").
 * No theme's id keeps one (A-20, QB-11): any other class that fits the pattern is any theme's.
 */
export const RESERVED_HTML_CLASSES: readonly string[] = ['studio', 'classic', 'dark', 'light', 'system']
export const KNOWN_SLOTS: readonly string[] = [
  'frame@1', 'lookEditor@1', 'styleSheet@1', 'notFound@1', 'asleep@1', 'page-header.mascot@1', 'page-header.art@1',
  'empty-state.mascot@1',
]
/** PA-14: every theme ships a look editor and a style sheet. */
export const REQUIRED_SLOTS: readonly string[] = ['lookEditor@1', 'styleSheet@1']
export const THEME_BASES: readonly string[] = ['studio', 'classic']
export const THEME_TONES: readonly string[] = ['from-look', 'user', 'light', 'dark']
const PAGE_MODES: readonly string[] = ['own', 'clone']
/** Gerard's Q3 list (ii), as the app routes it: knowledge is /documents, missions /assignments and /missions/[id]. */
export const PAGE_SCOPE: readonly string[] = [
  '/chat', '/deliverables', '/assignments', '/missions/[id]', '/agents', '/documents', '/playbooks', '/command-center', '/',
]
/** The core files a clone of each page copies, relative to frontend/ (contract.py PAGE_SOURCES). */
export const PAGE_SOURCES: Readonly<Record<string, readonly string[]>> = Object.freeze({
  '/chat': ['app/chat/page.tsx', 'components/chatbot/studio-chat-shell.tsx'],
  '/deliverables': ['app/deliverables/page.tsx', 'components/deliverables/studio/deliverables-studio.tsx'],
  '/assignments': ['app/assignments/page.tsx'],
  '/missions/[id]': ['app/missions/[id]/page.tsx'],
  '/agents': ['app/agents/page.tsx', 'components/agents/studio/agent-management-studio.tsx'],
  '/documents': ['app/documents/page.tsx', 'components/documents/document-management.tsx'],
  '/playbooks': ['app/playbooks/page.tsx', 'components/playbooks/PlaybooksPanel.tsx'],
  '/command-center': ['app/command-center/page.tsx', 'components/command-center/command-center-shell.tsx'],
  '/': ['app/page.tsx'],
})
/** PRD §4.4: pages that are always the host's. PRD-264 EUAI-10: /verify, the public AI-content check. */
export const HOST_OWNED_ROUTES: readonly string[] = [
  '/sign-in', '/sign-up', '/auth', '/sso-callback', '/reset-password', '/accept-invitation', '/settings', '/team',
  '/admin', '/marketplace', '/looks', '/safe-mode', '/x', '/api', '/dev', '/verify',
]

export const MANIFEST_ERROR_CODES: readonly string[] = [
  'NOT_AN_OBJECT', 'UNKNOWN_FIELD', 'MISSING_FIELD', 'BAD_SCHEMA', 'BAD_ID', 'BAD_TEXT', 'BAD_VERSION',
  'BAD_ENGINES', 'ENGINES_MISMATCH', 'BAD_HTML_CLASS', 'RESERVED_HTML_CLASS', 'BAD_BASE', 'BAD_MIN_WIDTH',
  'BAD_TONE', 'BAD_SLOTS', 'UNKNOWN_SLOT', 'MISSING_SLOT', 'BAD_PAGE', 'BAD_PAGE_MODE', 'BAD_KEEP_CORE_PAGE',
  'BAD_CLONED_FROM', 'CLONED_FROM_OUT_OF_SCOPE', 'PAGE_OUT_OF_SCOPE', 'HOST_OWNED_PAGE', 'BAD_ROUTES', 'BAD_NAV',
  'BAD_USES', 'BAD_ASSET_PATH', 'BAD_BRAND', 'BAD_LOOK_ID', 'BAD_MASCOT', 'BAD_MOBILE',
]
const REQUIRED_FIELDS: readonly string[] = [
  'schema', 'id', 'name', 'version', 'engines', 'base', 'htmlClass', 'minWidth', 'tone', 'slots',
]
const OPTIONAL_FIELDS: readonly string[] = [
  'description', 'pages', 'routes', 'nav', 'uses', 'pricing', 'previews', 'phonePreviews', 'fonts', 'brand', 'look',
  'mascot', 'mobile',
]
const ALL_FIELDS: readonly string[] = [...REQUIRED_FIELDS, ...OPTIONAL_FIELDS]
const ENGINE_FIELDS = ['automatos', 'sdk'] as const
const PAGE_FIELDS: readonly string[] = ['mode', 'keepCorePage', 'clonedFrom']
const CLONED_FROM_FIELDS: readonly string[] = ['path', 'sha']
const NAV_FIELDS: readonly string[] = ['id', 'label', 'path']
const BRAND_FIELDS: readonly string[] = ['name', 'wordmark']
const MASCOT_FIELDS: readonly string[] = ['states']
const MANIFEST_OWNER = 'theme.json'

/** A theme's id, `<publisher>/<name>`: the one its looks are kept under (installed-entry.ts reads it too). */
export const ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,38}\/[a-z0-9][a-z0-9-]{0,62}$/
const VERSION_PATTERN = /^(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$/
const RANGE_PATTERN = /^\^?[0-9]{1,6}\.[0-9]{1,6}\.[0-9]{1,6}$/
export const ROUTE_PATTERN = /^[a-z0-9][a-z0-9-]{0,31}(?:\/[a-z0-9][a-z0-9-]{0,31}){0,3}$/
export const NAV_ID_PATTERN = /^[a-z][a-z0-9-]{0,31}$/
export const NAV_PATH_PATTERN = /^\/(?:[a-z0-9][a-z0-9-]{0,63}(?:\/[a-z0-9][a-z0-9-]{0,63}){0,7})?$/
const PARAM_PATTERN = /^[a-z][A-Za-z0-9_-]{0,31}$/
const USES_PATTERN = /^[a-z][A-Za-z0-9.-]{0,63}$/
const SHA256_PATTERN = /^[0-9a-f]{64}$/
const STATE_PATTERN = /^[a-z][a-z0-9-]{0,31}$/
const PREVIEW_PATTERN = /^assets\/previews\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.png$/
const PHONE_PREVIEW_PATTERN = /^assets\/previews\/phone\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.(?:png|jpg)$/
const FONT_PATTERN = /^assets\/fonts\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.woff2$/
const WORDMARK_PATTERN = /^assets\/brand\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.(?:png|webp)$/
const X_ROUTE_PREFIX = '/x/'

export const NAME_MAX = 60
const DESCRIPTION_MAX = 280
export const LABEL_MAX = 40
export const MIN_WIDTH_MAX = 3840
export const MAX_ROUTES = 16
export const MAX_NAV = 16
const MAX_KEEP = 16
const MAX_USES = 64
const MAX_FILES_LISTED = 16
/** The Marketplace storefront shows screenshots per look: web and phone lists hold this many each. */
const MAX_PREVIEWS = 24
const MAX_STATES = 32

export interface ThemeNavEntry { readonly id: string; readonly label: string; readonly path: string }
export interface ThemePageEntry {
  readonly mode: 'own' | 'clone'
  readonly keepCorePage?: readonly string[]
  readonly clonedFrom?: { readonly path: string; readonly sha: string }
}

/** A checked theme.json. */
export interface ThemePackageManifest {
  readonly schema: typeof MANIFEST_SCHEMA
  readonly id: string
  readonly name: string
  readonly version: string
  readonly description?: string
  readonly engines: { readonly automatos: string; readonly sdk: string }
  readonly base: 'studio' | 'classic'
  readonly htmlClass: string
  readonly minWidth: number
  readonly tone: 'from-look' | 'user' | 'light' | 'dark'
  readonly slots: readonly string[]
  readonly pages?: Readonly<Record<string, ThemePageEntry>>
  readonly routes?: readonly string[]
  readonly nav?: readonly ThemeNavEntry[]
  readonly uses?: readonly string[]
  readonly pricing?: string
  readonly previews?: readonly string[]
  readonly phonePreviews?: readonly string[]
  readonly fonts?: readonly string[]
  readonly brand?: { readonly name?: string; readonly wordmark?: string }
  readonly look?: string
  readonly mascot?: { readonly states: readonly string[] } | null
  readonly mobile?: ThemeMobileBlock
}

/** The checked manifest (a frozen copy of the input) when `errors` is empty, else null. */
export interface ManifestResult {
  readonly manifest: ThemePackageManifest | null
  readonly errors: readonly ThemeError[]
}

function single(code: string, path: string, message: string): ThemeError[] {
  return [themeError(code, path, message)]
}

function unknownFields(record: JsonRecord, known: readonly string[], where: string): ThemeError[] {
  return unknownFieldErrors(record, known, 'UNKNOWN_FIELD', where, MANIFEST_OWNER)
}

/** A list of at most `maxItems` different strings, each matching `pattern`. */
function isStringList(value: unknown, pattern: RegExp, maxItems: number): value is string[] {
  if (!Array.isArray(value) || value.length > maxItems) return false
  const strings = value.every((item) => typeof item === 'string' && pattern.test(item))
  return strings && new Set(value).size === value.length
}

/** `value` is text matching `pattern`. */
function matches(value: unknown, pattern: RegExp): value is string {
  return typeof value === 'string' && pattern.test(value)
}

function textErrors(value: unknown, field: string, maxChars: number): ThemeError[] {
  const problem = plainTextProblem(value, field, maxChars)
  return problem === null ? [] : single('BAD_TEXT', field, problem)
}

function engineErrors(field: (typeof ENGINE_FIELDS)[number], value: unknown): ThemeError[] {
  const where = `engines.${field}`
  const host = field === 'automatos' ? THEME_HOST_VERSION : SDK_VERSION
  if (!matches(value, RANGE_PATTERN)) {
    return single('BAD_ENGINES', where, `"${where}" must be an exact version or a caret range such as ^1.0.0; "*" is refused.`)
  }
  const met = satisfiesSdkRange(value, host)
  return errorIf(!met, 'ENGINES_MISMATCH', where, `"${where}" asks for ${echo(value)}, and this host is ${host}.`)
}

function enginesErrors(engines: unknown): ThemeError[] {
  if (!isRecord(engines)) return single('BAD_ENGINES', 'engines', '"engines" must be {"automatos": "^1.0.0", "sdk": "^1.0.0"}.')
  return [
    ...unknownFields(engines, ENGINE_FIELDS, 'engines'),
    ...ENGINE_FIELDS.flatMap((field) => engineErrors(field, own(engines, field))),
  ]
}

function identityErrors(document: JsonRecord): ThemeError[] {
  return [
    ...errorIf(own(document, 'schema') !== MANIFEST_SCHEMA, 'BAD_SCHEMA', 'schema', `"schema" must be "${MANIFEST_SCHEMA}".`),
    ...errorIf(!matches(own(document, 'id'), ID_PATTERN), 'BAD_ID', 'id', (
      '"id" must be publisher/slug in lower-case letters, digits and dashes.'
    )),
    ...errorIf(!matches(own(document, 'version'), VERSION_PATTERN), 'BAD_VERSION', 'version', (
      '"version" must be MAJOR.MINOR.PATCH, such as 1.0.0.'
    )),
    ...textErrors(own(document, 'name'), 'name', NAME_MAX),
    ...(hasOwn(document, 'description') ? textErrors(document.description, 'description', DESCRIPTION_MAX) : []),
  ]
}

function htmlClassErrors(document: JsonRecord): ThemeError[] {
  const htmlClass = own(document, 'htmlClass')
  if (!matches(htmlClass, HTML_CLASS_PATTERN)) {
    return single('BAD_HTML_CLASS', 'htmlClass', (
      '"htmlClass" must be 3 to 32 lower-case letters, digits or dashes, starting with a letter.'
    ))
  }
  const reserved = RESERVED_HTML_CLASSES.includes(htmlClass)
  return errorIf(reserved, 'RESERVED_HTML_CLASS', 'htmlClass', `"htmlClass" "${htmlClass}" is reserved by the host.`)
}

function layoutErrors(document: JsonRecord): ThemeError[] {
  const width = own(document, 'minWidth')
  const widthOk = isWholeNumber(width) && width >= 0 && width <= MIN_WIDTH_MAX
  return [
    ...errorIf(!isOneOf(THEME_BASES, own(document, 'base')), 'BAD_BASE', 'base', `"base" must be one of ${THEME_BASES.join(', ')}.`),
    ...errorIf(!widthOk, 'BAD_MIN_WIDTH', 'minWidth', `"minWidth" must be a whole number of pixels from 0 to ${MIN_WIDTH_MAX}.`),
    ...errorIf(!isOneOf(THEME_TONES, own(document, 'tone')), 'BAD_TONE', 'tone', `"tone" must be one of ${THEME_TONES.join(', ')}.`),
  ]
}

function slotErrors(slots: unknown): ThemeError[] {
  const names = Array.isArray(slots) && slots.every((slot) => typeof slot === 'string') ? (slots as string[]) : null
  if (names === null || new Set(names).size !== names.length) {
    return single('BAD_SLOTS', 'slots', '"slots" must be a list of different slot names, such as "frame@1".')
  }
  return [
    ...names.filter((slot) => !KNOWN_SLOTS.includes(slot))
      .map((slot) => themeError('UNKNOWN_SLOT', 'slots', `"${echo(slot)}" is not a slot this host serves.`)),
    ...REQUIRED_SLOTS.filter((slot) => !names.includes(slot))
      .map((slot) => themeError('MISSING_SLOT', 'slots', `Every theme ships "${slot}": add it to "slots" and register it.`)),
  ]
}

function isHostOwned(route: string): boolean {
  return HOST_OWNED_ROUTES.some((owned) => route === owned || route.startsWith(`${owned}/`))
}

/** `{path, sha}` exactly. */
function isClonedFromShape(value: unknown): value is JsonRecord {
  if (!isRecord(value)) return false
  const keys = Object.keys(value)
  return keys.length === CLONED_FROM_FIELDS.length && CLONED_FROM_FIELDS.every((field) => keys.includes(field))
}

function clonedFromErrors(route: string, entry: JsonRecord, where: string): ThemeError[] {
  const cloned = own(entry, 'clonedFrom')
  if (own(entry, 'mode') !== 'clone') {
    const named = cloned !== undefined && cloned !== null
    return errorIf(named, 'BAD_CLONED_FROM', where, `"${where}" names "clonedFrom" only for mode "clone".`)
  }
  if (!isClonedFromShape(cloned)) {
    return single('BAD_CLONED_FROM', `${where}.clonedFrom`, `"${where}.clonedFrom" must be {"path", "sha"}.`)
  }
  if (!matches(cloned.sha, SHA256_PATTERN)) {
    return single('BAD_CLONED_FROM', `${where}.clonedFrom.sha`, `"${where}.clonedFrom.sha" must be the file's sha256 in hex.`)
  }
  const sources = PAGE_SOURCES[route] ?? []
  return errorIf(!isOneOf(sources, cloned.path), 'CLONED_FROM_OUT_OF_SCOPE', `${where}.clonedFrom.path`, (
    `"${where}.clonedFrom.path" must be one of ${sources.join(', ')}.`
  ))
}

function pageEntryErrors(route: string, entry: unknown): ThemeError[] {
  const where = `pages.${route}`
  if (!isRecord(entry)) return single('BAD_PAGE', where, `"${where}" must be an object with a "mode".`)
  const keepOk = !hasOwn(entry, 'keepCorePage') || isStringList(entry.keepCorePage, PARAM_PATTERN, MAX_KEEP)
  return [
    ...unknownFields(entry, PAGE_FIELDS, where),
    ...errorIf(!isOneOf(PAGE_MODES, own(entry, 'mode')), 'BAD_PAGE_MODE', `${where}.mode`, `"${where}.mode" must be "own" or "clone".`),
    ...errorIf(!keepOk, 'BAD_KEEP_CORE_PAGE', `${where}.keepCorePage`, (
      `"${where}.keepCorePage" must list up to ${MAX_KEEP} different link parameter names.`
    )),
    ...clonedFromErrors(route, entry, where),
  ]
}

function pageErrors(pages: unknown): ThemeError[] {
  if (!isRecord(pages)) return single('BAD_PAGE', 'pages', '"pages" must be an object keyed by route.')
  return Object.keys(pages).flatMap((route) => {
    const where = `pages.${echo(route)}`
    if (isHostOwned(route)) return single('HOST_OWNED_PAGE', where, `"${echo(route)}" is a host page: a theme can't replace it.`)
    if (!PAGE_SCOPE.includes(route)) {
      return single('PAGE_OUT_OF_SCOPE', where, `"${echo(route)}" is not a page a theme may design: ${PAGE_SCOPE.join(', ')}.`)
    }
    return pageEntryErrors(route, pages[route])
  })
}

function navPathErrors(path: unknown, where: string, routes: readonly string[]): ThemeError[] {
  if (!matches(path, NAV_PATH_PATTERN)) {
    return single('BAD_NAV', `${where}.path`, `"${where}.path" must be a path in this app, such as /chat.`)
  }
  const undeclared = path.startsWith(X_ROUTE_PREFIX) && !routes.includes(path.slice(X_ROUTE_PREFIX.length))
  return errorIf(undeclared, 'BAD_NAV', `${where}.path`, `"${where}.path" goes to a /x/ route the theme doesn't declare.`)
}

function navItemErrors(item: unknown, index: number, routes: readonly string[]): ThemeError[] {
  const where = `nav[${index}]`
  if (!isRecord(item)) return single('BAD_NAV', where, `"${where}" must be {"id", "label", "path"}.`)
  return [
    ...unknownFields(item, NAV_FIELDS, where),
    ...errorIf(!matches(own(item, 'id'), NAV_ID_PATTERN), 'BAD_NAV', `${where}.id`, (
      `"${where}.id" must be 1 to 32 lower-case letters, digits or dashes.`
    )),
    ...textErrors(own(item, 'label'), `${where}.label`, LABEL_MAX),
    ...navPathErrors(own(item, 'path'), where, routes),
  ]
}

function routeAndNavErrors(document: JsonRecord): ThemeError[] {
  const declared = hasOwn(document, 'routes') ? document.routes : []
  const routes = isStringList(declared, ROUTE_PATTERN, MAX_ROUTES) ? declared : null
  const routeErrors = errorIf(routes === null, 'BAD_ROUTES', 'routes', (
    `"routes" must list up to ${MAX_ROUTES} different paths such as "conversations".`
  ))
  const nav = hasOwn(document, 'nav') ? document.nav : []
  if (!Array.isArray(nav) || nav.length > MAX_NAV) {
    return [...routeErrors, ...single('BAD_NAV', 'nav', `"nav" must be a list of up to ${MAX_NAV} items.`)]
  }
  const ids = nav.filter(isRecord).map((item) => String(own(item, 'id')))
  return [
    ...routeErrors,
    ...errorIf(new Set(ids).size !== ids.length, 'BAD_NAV', 'nav', 'Each "nav" item needs its own "id".'),
    ...nav.flatMap((item: unknown, index: number) => navItemErrors(item, index, routes ?? [])),
  ]
}

function brandErrors(brand: unknown): ThemeError[] {
  if (!isRecord(brand)) return single('BAD_BRAND', 'brand', '"brand" must be {"name", "wordmark"}.')
  const wordmarkOk = !hasOwn(brand, 'wordmark') || matches(brand.wordmark, WORDMARK_PATTERN)
  return [
    ...unknownFields(brand, BRAND_FIELDS, 'brand'),
    ...(hasOwn(brand, 'name') ? textErrors(brand.name, 'brand.name', NAME_MAX) : []),
    ...errorIf(!wordmarkOk, 'BAD_ASSET_PATH', 'brand.wordmark', '"brand.wordmark" must be a .png or .webp in assets/brand/.'),
  ]
}

function mascotErrors(mascot: unknown): ThemeError[] {
  if (mascot === undefined || mascot === null) return []
  if (!isRecord(mascot)) return single('BAD_MASCOT', 'mascot', '"mascot" must be null or {"states": [...]}.')
  return [
    ...unknownFields(mascot, MASCOT_FIELDS, 'mascot'),
    ...errorIf(!isStringList(own(mascot, 'states'), STATE_PATTERN, MAX_STATES), 'BAD_MASCOT', 'mascot.states', (
      `"mascot.states" must list up to ${MAX_STATES} different state names.`
    )),
  ]
}

function listedFileErrors(document: JsonRecord, field: string, pattern: RegExp, shape: string, most: number): ThemeError[] {
  const listOk = !hasOwn(document, field) || isStringList(document[field], pattern, most)
  return errorIf(!listOk, 'BAD_ASSET_PATH', field, `"${field}" must list up to ${most} different ${shape} files.`)
}

function extraErrors(document: JsonRecord): ThemeError[] {
  const usesOk = !hasOwn(document, 'uses') || isStringList(document.uses, USES_PATTERN, MAX_USES)
  const lookOk = !hasOwn(document, 'look') || matches(document.look, LOOK_ID_PATTERN)
  return [
    ...listedFileErrors(document, 'previews', PREVIEW_PATTERN, 'assets/previews/*.png', MAX_PREVIEWS),
    ...listedFileErrors(document, 'phonePreviews', PHONE_PREVIEW_PATTERN, 'assets/previews/phone/*.png or *.jpg', MAX_PREVIEWS),
    ...listedFileErrors(document, 'fonts', FONT_PATTERN, 'assets/fonts/*.woff2', MAX_FILES_LISTED),
    ...errorIf(!usesOk, 'BAD_USES', 'uses', `"uses" must list up to ${MAX_USES} different SDK area names.`),
    ...(hasOwn(document, 'pricing') ? textErrors(document.pricing, 'pricing', LABEL_MAX) : []),
    ...errorIf(!lookOk, 'BAD_LOOK_ID', 'look', '"look" must be the id of one of the theme\'s presets.'),
    ...(hasOwn(document, 'brand') ? brandErrors(document.brand) : []),
    ...(hasOwn(document, 'mobile') ? mobileErrors(document) : []),
    ...mascotErrors(own(document, 'mascot')),
  ]
}

/** Every reason `document` is not a valid theme.json; the manifest itself when there is none. */
export function validateManifest(document: unknown): ManifestResult {
  if (!isRecord(document)) return { manifest: null, errors: single('NOT_AN_OBJECT', '', 'theme.json must be a JSON object.') }
  const missing = REQUIRED_FIELDS.filter((field) => !hasOwn(document, field))
  const checks = [
    ...identityErrors(document), ...enginesErrors(own(document, 'engines')), ...htmlClassErrors(document),
    ...layoutErrors(document), ...slotErrors(own(document, 'slots')),
    ...(hasOwn(document, 'pages') ? pageErrors(document.pages) : []),
    ...routeAndNavErrors(document), ...extraErrors(document),
  ]
  // A missing field is said once, as MISSING_FIELD, not again by its own check.
  const errors = [
    ...unknownFields(document, ALL_FIELDS, ''),
    ...missing.map((field) => themeError('MISSING_FIELD', field, `theme.json needs "${field}".`)),
    ...checks.filter((error) => !missing.includes(error.path.split('.')[0])),
  ]
  if (errors.length > 0) return { manifest: null, errors }
  return { manifest: Object.freeze({ ...document }) as unknown as ThemePackageManifest, errors }
}
