/**
 * PRD-261 M-01 (host 1.1.0, plan §2.2.11) — theme.json's optional `mobile` block, checked: the data a
 * phone draws a theme from. A phone never loads `dist/entry.js` or `dist/theme.css`.
 *
 * The TypeScript twin of `orchestrator/services/themes/mobile_manifest.py` and of contract.py's MOBILE_*
 * lists. package-manifest.ts calls `mobileErrors` for a theme.json with the field;
 * `__tests__/validator-parity.test.ts` holds both languages to the shared corpus
 * (`orchestrator/tests/fixtures/themes/cases.json`, `manifests`) and reads the lists from contract.py as
 * text, and `__tests__/mobile-tokens.test.ts` holds the token list to look-kit's `lookToStudioTokens`.
 *
 * - exactly one of `tokens` and `fromLooks`, and `fromLooks` is only ever `true`;
 * - `tokens.light` is required and `tokens.dark` optional: Studio token names only, including
 *   MOBILE_REQUIRED_TOKENS, each value an `H S% L%` triplet as hexToTriplet writes it, or `#rrggbb`;
 * - `fonts` gives MOBILE_FONT_ROLES only, each a different `assets/fonts/mobile/<NAME>.ttf` or `.otf`;
 * - `mascot` is one of MOBILE_MASCOT_MODES; `clips` and `stills` need the top-level `mascot` with its states;
 * - `engines.automatos` admits no host below MOBILE_MIN_HOST.
 *
 * Each refusal is BAD_MOBILE naming the field, or UNKNOWN_FIELD for a key outside the block.
 */
import { echo, errorIf, hasOwn, isRecord, own, themeError, unknownFieldErrors, type JsonRecord, type ThemeError } from './css-value'

/** The 24 Studio tokens lookToStudioTokens writes as 'H S% L%' triplets, less its status tokens (PRD-268 K13; US-T01) and never its eight pre-wrapped aliases. */
export const MOBILE_TOKEN_NAMES: readonly string[] = [
  '--background', '--foreground', '--card', '--card-foreground', '--popover', '--popover-foreground',
  '--primary', '--primary-foreground', '--secondary', '--secondary-foreground', '--muted', '--muted-foreground',
  '--accent', '--accent-foreground', '--destructive', '--destructive-foreground', '--border', '--input', '--ring',
  '--background-gradient-from', '--background-gradient-to', '--glass-border-hover', '--cc-hover', '--cc-hover-border',
]
export const MOBILE_REQUIRED_TOKENS: readonly string[] = ['--background', '--foreground']
export const MOBILE_FONT_ROLES: readonly string[] = ['sans', 'serif', 'mono']
export const MOBILE_MASCOT_MODES: readonly string[] = ['clips', 'stills', 'none']

/** A text pair is measured against MOBILE_AA_TEXT, a UI pair (a mark on its ground) against MOBILE_AA_UI. */
export type MobileContrastKind = 'text' | 'ui'
export type MobileContrastPair = readonly [foreground: string, background: string, kind: MobileContrastKind]
/** What the review measures (WCAG 2.x), per scheme of `mobile.tokens` or per preset under `fromLooks`. */
export const MOBILE_CONTRAST_PAIRS: readonly MobileContrastPair[] = [
  ['--foreground', '--background', 'text'], ['--card-foreground', '--card', 'text'],
  ['--popover-foreground', '--popover', 'text'], ['--secondary-foreground', '--secondary', 'text'],
  ['--muted-foreground', '--muted', 'text'], ['--primary-foreground', '--primary', 'text'],
  ['--accent-foreground', '--accent', 'text'], ['--destructive-foreground', '--destructive', 'text'],
  ['--primary', '--background', 'ui'], ['--ring', '--background', 'ui'],
]
export const MOBILE_AA_TEXT = 4.5
export const MOBILE_AA_UI = 3
/** A theme with a `mobile` block must not admit a host below this one: an older host refuses the field. */
export const MOBILE_MIN_HOST = '1.1.0'

export const MOBILE_ERROR_CODE = 'BAD_MOBILE'
const MOBILE_FIELD = 'mobile'
const MOBILE_FIELDS: readonly string[] = ['tokens', 'fromLooks', 'fonts', 'mascot']
const MOBILE_SCHEMES: readonly string[] = ['light', 'dark']
const REQUIRED_SCHEME = 'light'
const MASCOT_NONE = 'none'
const MANIFEST_OWNER = 'theme.json'
const MOBILE_FONT_PATTERN = /^assets\/fonts\/mobile\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.(?:ttf|otf)$/
/** An engines range (package-manifest.ts RANGE_PATTERN): the lowest host it admits is its own version. */
const HOST_RANGE_PATTERN = /^\^?([0-9]{1,6})\.([0-9]{1,6})\.([0-9]{1,6})$/
/** look-kit hsl.ts TRIPLET and contrast.ts HEX: what hexToTriplet writes, and #rrggbb. */
const TRIPLET_PATTERN = /^([0-9]{1,3}(?:\.[0-9]{1,2})?) ([0-9]{1,3}(?:\.[0-9]{1,2})?)% ([0-9]{1,3}(?:\.[0-9]{1,2})?)%$/
const HEX_PATTERN = /^#[0-9a-f]{6}$/i
const FULL_TURN = 360
const PERCENT = 100

export type MobileScheme = Readonly<Record<string, string>>
/** A checked `mobile` block. */
export interface ThemeMobileBlock {
  readonly tokens?: { readonly light: MobileScheme; readonly dark?: MobileScheme }
  readonly fromLooks?: true
  readonly fonts?: Readonly<Partial<Record<'sans' | 'serif' | 'mono', string>>>
  readonly mascot?: 'clips' | 'stills' | 'none'
}

/**
 * True when `value` is a colour a phone can read: a triplet look-kit's parseTriplet accepts (hue below
 * 360, saturation and lightness at most 100) or a #rrggbb (mobile_tokens.py parse_colour is its twin).
 */
export function isMobileColour(value: unknown): value is string {
  if (typeof value !== 'string') return false
  if (HEX_PATTERN.test(value)) return true
  const match = TRIPLET_PATTERN.exec(value)
  if (match === null) return false
  return Number(match[1]) < FULL_TURN && Number(match[2]) <= PERCENT && Number(match[3]) <= PERCENT
}

function bad(path: string, message: string): ThemeError[] {
  return [themeError(MOBILE_ERROR_CODE, path, message)]
}

function schemeErrors(scheme: string, values: unknown): ThemeError[] {
  const where = `mobile.tokens.${scheme}`
  if (!isRecord(values)) return bad(where, `"${where}" must be an object of Studio token names and their colours.`)
  const nameErrors = Object.keys(values).flatMap((name) => {
    if (!MOBILE_TOKEN_NAMES.includes(name)) {
      return bad(`${where}.${echo(name)}`, `"${echo(name)}" is not one of the ${MOBILE_TOKEN_NAMES.length} Studio tokens a phone draws.`)
    }
    return errorIf(!isMobileColour(values[name]), MOBILE_ERROR_CODE, `${where}.${name}`, (
      `"${where}.${name}" must be "H S% L%" (such as "0 0% 100%") or "#rrggbb".`
    ))
  })
  const missing = MOBILE_REQUIRED_TOKENS.filter((name) => !hasOwn(values, name))
  return [...nameErrors, ...missing.map((name) => themeError(MOBILE_ERROR_CODE, where, `"${where}" needs "${name}".`))]
}

function tokensErrors(tokens: unknown): ThemeError[] {
  if (!isRecord(tokens)) return bad('mobile.tokens', '"mobile.tokens" must be {"light": {...}, "dark": {...}}, with "dark" optional.')
  return [
    ...Object.keys(tokens).filter((key) => !MOBILE_SCHEMES.includes(key)).map((key) => themeError(
      MOBILE_ERROR_CODE, `mobile.tokens.${echo(key)}`, `"${echo(key)}" is not a scheme: "mobile.tokens" holds "light" and "dark".`,
    )),
    ...errorIf(!hasOwn(tokens, REQUIRED_SCHEME), MOBILE_ERROR_CODE, `mobile.tokens.${REQUIRED_SCHEME}`, (
      `"mobile.tokens" needs "${REQUIRED_SCHEME}"; "dark" is optional.`
    )),
    ...MOBILE_SCHEMES.filter((scheme) => hasOwn(tokens, scheme)).flatMap((scheme) => schemeErrors(scheme, tokens[scheme])),
  ]
}

/** Exactly one of `tokens` and `fromLooks`, and that one well formed. */
function sourceErrors(block: JsonRecord): ThemeError[] {
  const hasTokens = hasOwn(block, 'tokens')
  const hasLooks = hasOwn(block, 'fromLooks')
  if (hasTokens === hasLooks) return bad(MOBILE_FIELD, '"mobile" needs exactly one of "tokens" and "fromLooks".')
  if (hasLooks) {
    return errorIf(block.fromLooks !== true, MOBILE_ERROR_CODE, 'mobile.fromLooks', (
      '"mobile.fromLooks" is only ever true: leave it out and give "tokens" instead.'
    ))
  }
  return tokensErrors(block.tokens)
}

function fontErrors(block: JsonRecord): ThemeError[] {
  if (!hasOwn(block, 'fonts')) return []
  const fonts = block.fonts
  if (!isRecord(fonts)) {
    return bad('mobile.fonts', '"mobile.fonts" must be {"sans", "serif", "mono"}, each a .ttf or .otf in assets/fonts/mobile/.')
  }
  const roleErrors = Object.keys(fonts).flatMap((role) => {
    const where = `mobile.fonts.${echo(role)}`
    if (!MOBILE_FONT_ROLES.includes(role)) return bad(where, `"${echo(role)}" is not a font role: "sans", "serif" or "mono".`)
    const path = fonts[role]
    const pathOk = typeof path === 'string' && MOBILE_FONT_PATTERN.test(path)
    return errorIf(!pathOk, MOBILE_ERROR_CODE, where, `"${where}" must be a .ttf or .otf file in assets/fonts/mobile/.`)
  })
  const paths = Object.values(fonts).filter((path): path is string => typeof path === 'string')
  return [
    ...roleErrors,
    ...errorIf(new Set(paths).size !== paths.length, MOBILE_ERROR_CODE, 'mobile.fonts', 'Each role in "mobile.fonts" needs its own file.'),
  ]
}

/** The mode is known; clips and stills need the top-level `mascot` (a malformed one is BAD_MASCOT's). */
function mascotErrors(block: JsonRecord, mascot: unknown): ThemeError[] {
  if (!hasOwn(block, 'mascot')) return []
  const mode = block.mascot
  if (typeof mode !== 'string' || !MOBILE_MASCOT_MODES.includes(mode)) {
    return bad('mobile.mascot', '"mobile.mascot" must be "clips", "stills" or "none".')
  }
  const states = isRecord(mascot) ? own(mascot, 'states') : undefined
  const noStates = mascot === undefined || mascot === null || (Array.isArray(states) && states.length === 0)
  return errorIf(mode !== MASCOT_NONE && noStates, MOBILE_ERROR_CODE, 'mobile.mascot', (
    `"mobile.mascot" "${mode}" needs the top-level "mascot" with its "states".`
  ))
}

/** True when `version` (X.Y.Z parts) is below MOBILE_MIN_HOST. */
function belowMinHost(version: readonly number[]): boolean {
  const minimum = MOBILE_MIN_HOST.split('.').map(Number)
  const at = version.findIndex((part, index) => part !== minimum[index])
  return at !== -1 && version[at] < minimum[at]
}

/** The block is host 1.1.0's: a range that admits an older host is refused (BAD_ENGINES covers a malformed one). */
function enginesErrors(engines: unknown): ThemeError[] {
  const host = isRecord(engines) ? own(engines, 'automatos') : undefined
  const match = typeof host === 'string' ? HOST_RANGE_PATTERN.exec(host) : null
  if (match === null || !belowMinHost(match.slice(1).map(Number))) return []
  return bad('engines.automatos', (
    `A theme with "mobile" needs host ${MOBILE_MIN_HOST} or later: "engines.automatos" must not admit an earlier one `
    + `(use ^${MOBILE_MIN_HOST}).`
  ))
}

/** Every reason theme.json's `mobile` block is refused; empty when it is valid. */
export function mobileErrors(document: JsonRecord): ThemeError[] {
  const block = own(document, MOBILE_FIELD)
  if (!isRecord(block)) return bad(MOBILE_FIELD, '"mobile" must be an object with "tokens" or "fromLooks".')
  return [
    ...unknownFieldErrors(block, MOBILE_FIELDS, 'UNKNOWN_FIELD', MOBILE_FIELD, MANIFEST_OWNER),
    ...sourceErrors(block),
    ...fontErrors(block),
    ...mascotErrors(block, own(document, 'mascot')),
    ...enginesErrors(own(document, 'engines')),
  ]
}
