// PRD-259 §4 — a skin as the one stylesheet Studio writes. Pure. Every value in the output is
// re-parsed to numbers here and written by formatTriplet / formatPx; a name is one of the
// schema's own lists. A string the skin carried never reaches the CSS, so a skin that slipped
// past validateSkin (a tampered cache, a bug) makes skinToCss throw, never emit.
//
// Specificity, not cascade layers, decides (Tailwind 3 compiles @layer base to plain CSS):
//   html[data-skin="<id>"]            (0,1,1) shape and fonts, both tones: beats .studio, :root
//   html:not(.dark)[data-skin="<id>"] (0,2,1) tokens.light: beats .studio and never leaks into
//                                     dark, where Classic's .dark (0,1,0) would otherwise lose to it
//   html.dark[data-skin="<id>"]       (0,2,1) tokens.dark: beats .studio.dark and .dark (0,2,0/0,1,0)
//   html[data-skin="<id>"] body       (0,1,2) the font variables, one element below next/font's
//
// Fonts: next/font sets --font-geist-sans, --font-newsreader and --font-geist-mono on <html>.
// Overriding them there would make a swap (sans ← newsreader, serif ← geist) a var() cycle, so
// <html> gets --skin-font-<slot> (read from next/font's originals) and <body> re-points Studio's
// variables at them; body's font-family re-reads the sans slot, because <html> computed its own
// before the override. No @font-face and no url(): a skin's woff2 is registered by fonts.ts
// through the FontFace API under the family skinFontFamily() names (decision K2).

import { formatTriplet, parseColor } from './color'
import { hasOwn, isRecord, oneOf, type Fields } from './fields'
import {
  ALLOWED_TOKENS,
  FONT_FILE_PATTERN,
  FONT_SLOTS,
  GLOW_MAX,
  GLOW_MIN,
  RADIUS_MAX,
  RADIUS_MIN,
  SHIPPED_FONTS,
  SKIN_ID_PATTERN,
  SKIN_TONES,
  type FontSlot,
  type SkinToken,
  type SkinTone,
} from './schema'

export const SKIN_DATA_ATTRIBUTE = 'data-skin'

/** --fg-3 is the skin's muted-foreground at this alpha (Studio's faded meta ink). */
export const FG3_ALPHA = 0.75
/** --accent-2 is the skin's accent-2 at these alphas, as globals.css writes Studio's. */
export const ACCENT_2_ALPHA: Readonly<Record<SkinTone, number>> = { light: 0.1, dark: 0.14 }
/** --skin-glow is shape.glow / GLOW_SCALE: an alpha, read as hsl(var(--accent) / var(--skin-glow)). */
export const GLOW_SCALE = 100
const ALPHA_DECIMALS = 2

/** Allowed tokens that globals.css holds pre-wrapped in hsl(): written wrapped here too. */
const WRAPPED_TOKENS: readonly SkinToken[] = ['border-2', 'navy']
const ALPHA_TOKEN: SkinToken = 'accent-2'

/** Studio's pre-wrapped aliases, derived from the skin's own core tokens (never set by a skin). */
const DERIVED_FROM: readonly (readonly [string, SkinToken])[] = [
  ['paper', 'card'],
  ['bg-1', 'background'],
  ['bg-2', 'secondary'],
  ['fg', 'foreground'],
  ['fg-2', 'muted-foreground'],
]
const FG3_ALIAS = 'fg-3'
const FG3_SOURCE: SkinToken = 'muted-foreground'

type ShippedFont = (typeof SHIPPED_FONTS)[number]

/** The next/font variable (app/layout.tsx) each slot is read through in Studio. */
export const SLOT_VARIABLES: Readonly<Record<FontSlot, string>> = {
  sans: '--font-geist-sans',
  serif: '--font-newsreader',
  mono: '--font-geist-mono',
}
const SHIPPED_FONT_VARIABLES: Readonly<Record<ShippedFont, string>> = {
  geist: '--font-geist-sans',
  'geist-mono': '--font-geist-mono',
  newsreader: '--font-newsreader',
}
/** The tail of body's font-family once the sans slot is set (the .studio stack's last resort). */
const BODY_SANS_FALLBACK = 'ui-sans-serif, system-ui, sans-serif'
/**
 * globals.css's first .studio block also defines --serif from --font-newsreader on <html>, where
 * var() was already substituted, so body re-declares it from the serif slot (its stack's tail).
 */
const SERIF_ALIAS = '--serif'
const SERIF_FALLBACK = 'Charter, Georgia, serif'

type Declaration = readonly [name: string, value: string]

export class SkinCssError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SkinCssError'
  }
}

/** The FontFace family Studio registers a skin's woff2 under: made from a checked id only. */
export function skinFontFamily(skinId: string, slot: FontSlot): string {
  return `skin-${checkedId(skinId)}-${slot}`
}

function checkedId(skinId: unknown): string {
  if (typeof skinId !== 'string' || !SKIN_ID_PATTERN.test(skinId)) throw new SkinCssError('A skin id must be a UUID.')
  return skinId
}

export function formatPx(value: number): string {
  return `${String(value)}px`
}

function formatAlpha(value: number): string {
  return value.toFixed(ALPHA_DECIMALS)
}

/** A token value re-parsed to its numbers and written back as a triplet, or a throw. */
function triplet(value: unknown, where: string): string {
  const hsl = typeof value === 'string' ? parseColor(value) : null
  if (hsl === null) throw new SkinCssError(`${where} isn't a colour.`)
  return formatTriplet(hsl.h, hsl.s, hsl.l)
}

function wholeNumber(value: unknown, min: number, max: number, where: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
    throw new SkinCssError(`${where} must be a whole number from ${min} to ${max}.`)
  }
  return value
}

/** The tone's token values, every one re-parsed; an unknown name throws. */
function toneValues(tone: SkinTone, raw: unknown): Partial<Record<SkinToken, string>> {
  if (!isRecord(raw)) throw new SkinCssError(`tokens.${tone} must be an object.`)
  const unknown = Object.keys(raw).filter((key) => !oneOf(ALLOWED_TOKENS, key))
  if (unknown.length > 0) throw new SkinCssError(`tokens.${tone} holds a name that isn't a token.`)
  const entries = ALLOWED_TOKENS.filter((token) => hasOwn(raw, token))
    .map((token) => [token, triplet(raw[token], `tokens.${tone}.${token}`)] as const)
  return Object.fromEntries(entries)
}

function tokenDeclaration(tone: SkinTone, token: SkinToken, value: string): Declaration {
  if (token === ALPHA_TOKEN) return [`--${token}`, `hsl(${value} / ${formatAlpha(ACCENT_2_ALPHA[tone])})`]
  if (WRAPPED_TOKENS.includes(token)) return [`--${token}`, `hsl(${value})`]
  return [`--${token}`, value]
}

function derivedDeclarations(values: Partial<Record<SkinToken, string>>): Declaration[] {
  const plain = DERIVED_FROM.flatMap(([alias, source]): Declaration[] => {
    const value = values[source]
    return value === undefined ? [] : [[`--${alias}`, `hsl(${value})`]]
  })
  const muted = values[FG3_SOURCE]
  return muted === undefined ? plain : [...plain, [`--${FG3_ALIAS}`, `hsl(${muted} / ${formatAlpha(FG3_ALPHA)})`]]
}

function toneDeclarations(tone: SkinTone, raw: unknown): Declaration[] {
  const values = toneValues(tone, raw)
  const tokens = ALLOWED_TOKENS.flatMap((token): Declaration[] => {
    const value = values[token]
    return value === undefined ? [] : [tokenDeclaration(tone, token, value)]
  })
  return [...tokens, ...derivedDeclarations(values)]
}

function shapeDeclarations(raw: unknown): Declaration[] {
  if (raw === undefined) return []
  if (!isRecord(raw)) throw new SkinCssError('shape must be an object.')
  const radius: Declaration[] = hasOwn(raw, 'radius')
    ? [['--radius', formatPx(wholeNumber(raw.radius, RADIUS_MIN, RADIUS_MAX, 'shape.radius'))]]
    : []
  const glow: Declaration[] = hasOwn(raw, 'glow')
    ? [['--skin-glow', formatAlpha(wholeNumber(raw.glow, GLOW_MIN, GLOW_MAX, 'shape.glow') / GLOW_SCALE)]]
    : []
  return [...radius, ...glow]
}

/** Where a slot's face comes from: a shipped face's next/font variable, or the skin's own woff2. */
function fontSource(skinId: string, slot: FontSlot, value: unknown): string {
  if (oneOf(SHIPPED_FONTS, value)) return `var(${SHIPPED_FONT_VARIABLES[value]})`
  if (typeof value === 'string' && FONT_FILE_PATTERN.test(value)) {
    return `"${skinFontFamily(skinId, slot)}", var(${SLOT_VARIABLES[slot]})`
  }
  throw new SkinCssError(`fonts.${slot} isn't a shipped font or a woff2 in assets/fonts/.`)
}

interface FontDeclarations {
  readonly root: readonly Declaration[]
  readonly body: readonly Declaration[]
}

function fontDeclarations(skinId: string, raw: unknown): FontDeclarations {
  if (raw === undefined) return { root: [], body: [] }
  if (!isRecord(raw)) throw new SkinCssError('fonts must be an object.')
  if (Object.keys(raw).some((key) => !oneOf(FONT_SLOTS, key))) throw new SkinCssError('fonts holds a name that isn\'t a slot.')
  const slots = FONT_SLOTS.filter((slot) => hasOwn(raw, slot))
  const root = slots.map((slot): Declaration => [`--skin-font-${slot}`, fontSource(skinId, slot, raw[slot])])
  const repoint = slots.map((slot): Declaration => [SLOT_VARIABLES[slot], `var(--skin-font-${slot})`])
  const serif: Declaration[] = slots.includes('serif') ? [[SERIF_ALIAS, `var(--skin-font-serif), ${SERIF_FALLBACK}`]] : []
  const family: Declaration[] = slots.includes('sans') ? [['font-family', `var(--skin-font-sans), ${BODY_SANS_FALLBACK}`]] : []
  return { root, body: [...repoint, ...serif, ...family] }
}

/** The four selectors a skin's stylesheet uses, made from a checked id only. */
export function skinSelectors(skinId: string) {
  const attribute = `[${SKIN_DATA_ATTRIBUTE}="${checkedId(skinId)}"]`
  return {
    root: `html${attribute}`,
    light: `html:not(.dark)${attribute}`,
    dark: `html.dark${attribute}`,
    body: `html${attribute} body`,
  } as const
}

function rule(selector: string, declarations: readonly Declaration[]): string[] {
  if (declarations.length === 0) return []
  return [`${selector}{${declarations.map(([name, value]) => `${name}:${value}`).join(';')}}`]
}

function toneRules(skin: Fields, selectors: ReturnType<typeof skinSelectors>): string[] {
  const tokens = skin.tokens
  if (!isRecord(tokens)) throw new SkinCssError('tokens must be an object.')
  return SKIN_TONES.filter((tone) => hasOwn(tokens, tone))
    .flatMap((tone) => rule(selectors[tone], toneDeclarations(tone, tokens[tone])))
}

/**
 * The skin's stylesheet: one rule per selector that has something to set, joined by newlines.
 * `skin` is a validated Skin; anything that isn't one throws (SkinCssError), and the caller
 * falls back to the base style.
 */
export function skinToCss(skin: unknown, skinId: string): string {
  const selectors = skinSelectors(skinId)
  if (!isRecord(skin)) throw new SkinCssError('A skin must be an object.')
  const fonts = fontDeclarations(skinId, skin.fonts)
  const css = [
    ...rule(selectors.root, [...shapeDeclarations(skin.shape), ...fonts.root]),
    ...toneRules(skin, selectors),
    ...rule(selectors.body, fonts.body),
  ].join('\n')
  if (!SKIN_CSS_GRAMMAR.test(css)) throw new SkinCssError('The generated stylesheet failed its own grammar.')
  return css
}

// The grammar every stylesheet above matches. The pre-paint script tests a cached stylesheet
// against it before it touches the page, so it refuses what skinToCss could never write:
// url( in any case, //, @, \, <, a brace inside a value, a selector that isn't a skin's, and a
// quoted string anywhere but a --skin-font-<slot> value's leading Studio family name.
const UUID = '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
const SELECTOR = `html(?:\\.dark|:not\\(\\.dark\\))?\\[data-skin="${UUID}"\\](?: body)?`
const NAME = '(?:--[a-z0-9-]+|font-family)'
const VALUE = '[A-Za-z0-9%., /()-]+'
const SLOT = '(?:sans|serif|mono)'
const FONT_DECLARATION = `--skin-font-${SLOT}:(?:"skin-${UUID}-${SLOT}", )?${VALUE}`
const DECLARATION = `(?:${FONT_DECLARATION}|${NAME}:${VALUE})`
const RULE = `${SELECTOR}\\{${DECLARATION}(?:;${DECLARATION})*\\}`

export const SKIN_CSS_GRAMMAR = new RegExp(`^(?![\\s\\S]*[uU][rR][lL]\\()(?![\\s\\S]*//)${RULE}(?:\\n${RULE})*$`)

/** True when every selector in `css` names this skin id (the cache must not apply another skin's). */
export function cssIsFor(css: string, skinId: string): boolean {
  const marker = `${SKIN_DATA_ATTRIBUTE}="`
  const parts = css.split(marker)
  return parts.length > 1 && parts.slice(1).every((part) => part.startsWith(`${skinId}"`))
}
