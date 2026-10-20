import { AA_TEXT, contrastRatio, mixHex, readableOn, readableOnAll, relativeLuminance } from './contrast'
import {
  entryOf,
  keysOf,
  type LookAccent,
  type LookCatalogue,
  type LookGround,
  type LookPicks,
} from './types'

/**
 * PRD-260 — a look as a theme's CSS custom properties, set on <html> only while
 * the theme is active (K3). Pure. Every name carries the catalogue's prefix
 * (K2), so a theme's --accent and --ring never clobber Studio's HSL tokens of
 * the same names; Studio's tokens come from studio-tokens.ts. Each
 * section has its own builder so a pre-paint table is made from exactly the
 * same values.
 *
 * PRD-261 A-13a — moved from lib/looks/css.ts into the core look runtime (PA-9):
 * the catalogue, the prefix and the glow range are the theme's, passed in; the
 * runtime names no theme.
 */

export type CssVars = Readonly<Record<string, string>>

/** <prefix>-accent-line: the accent at 45% alpha (selected borders, focus halos). */
export const ACCENT_LINE_ALPHA = 0.45
/** <prefix>-accent-fill: the accent at 12% alpha (tinted grounds: a running tool line). */
export const ACCENT_FILL_ALPHA = 0.12
/** The focus halo and the top of a Deliverable card's tint. */
export const ACCENT_HALO_ALPHA = 0.18
/** The far end of a Deliverable card's tint. */
export const ACCENT_FAINT_ALPHA = 0.04
/** The accent tints a theme sets the accent as text on (M50): <prefix>-accent-text reads on each, over the bg. */
export const ACCENT_TEXT_TINTS: readonly number[] = [ACCENT_HALO_ALPHA, ACCENT_FILL_ALPHA, ACCENT_FAINT_ALPHA]
/** <prefix>-glow is the accent at glow / GLOW_SCALE alpha. */
export const GLOW_SCALE = 100
/** The snag glow's strongest error colour, as a whole glow: 15%, lower on a ground whose text needs it. */
export const SNAG_GLOW = 15
/** How far dusk sits from the ground's secondary text toward its bg (dusk = 45% text2, 55% bg). */
export const DUSK_TOWARD_BG = 0.55

/** The page glow's name after the prefix (the pre-paint script builds its value from the stored glow). */
const GLOW_NAME = 'glow'
/** The snag glow's name after the prefix: the error colour at the ground's snag alpha (snagGlowAlpha). */
const GLOW_SNAG_NAME = 'glow-snag'

const HEX = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i

/** '--<prefix>-<name>'. */
export function cssVarName(catalogue: Pick<LookCatalogue, 'prefix'>, name: string): string {
  return `--${catalogue.prefix}-${name}`
}

/** The page glow's variable. */
export function glowVar(catalogue: Pick<LookCatalogue, 'prefix'>): string {
  return cssVarName(catalogue, GLOW_NAME)
}

/** The snag glow's variable. */
export function glowSnagVar(catalogue: Pick<LookCatalogue, 'prefix'>): string {
  return cssVarName(catalogue, GLOW_SNAG_NAME)
}

/** `vars` with every name prefixed, in the same order. */
function prefixed(catalogue: LookCatalogue, vars: Readonly<Record<string, string>>): CssVars {
  return Object.fromEntries(Object.entries(vars).map(([name, value]) => [cssVarName(catalogue, name), value]))
}

function groundOf(catalogue: LookCatalogue, key: string): LookGround {
  return entryOf(catalogue.grounds, key, 'ground')
}

function accentOf(catalogue: LookCatalogue, key: string): LookAccent {
  return entryOf(catalogue.accents, key, 'accent')
}

/** 'r,g,b' for a #rrggbb colour, to build rgba() from. */
export function rgbList(hex: string): string {
  const match = HEX.exec(hex)
  if (!match) throw new Error(`${hex} isn't a #rrggbb colour`)
  return match.slice(1, 4).map((pair) => parseInt(pair, 16)).join(',')
}

export function rgba(hex: string, alpha: number): string {
  return `rgba(${rgbList(hex)},${alpha})`
}

export interface Sheet {
  readonly bg: string
  readonly text: string
}

/** True for a ground whose bg is darker than its text: a dark look. */
export function isDarkGround(catalogue: LookCatalogue, key: string): boolean {
  const { bg, text } = groundOf(catalogue, key)
  return relativeLuminance(bg) < relativeLuminance(text)
}

/**
 * A Deliverable's page is a printed sheet on the desk (Ember mock 05): the
 * ground's text colour with its bg as ink on a dark ground (the cream sheet),
 * the ground's surface on a light one. Either way the pair is one the catalogue
 * tests already hold to AA.
 */
export function sheetOf(catalogue: LookCatalogue, key: string): Sheet {
  const { bg, surface, text } = groundOf(catalogue, key)
  return isDarkGround(catalogue, key) ? { bg: text, text: bg } : { bg: surface, text }
}

/**
 * The mascot's ring while he sleeps or is lost: a dim colour made from the
 * ground, so it suits every look and never borrows the light or the accent.
 */
export function duskOf(catalogue: LookCatalogue, key: string): string {
  const { bg, text2 } = groundOf(catalogue, key)
  return mixHex(text2, bg, DUSK_TOWARD_BG)
}

/** The error as text on this ground (<prefix>-error-text): the error colour, darkened where it doesn't read. */
export function errorTextOf(catalogue: LookCatalogue, key: string): string {
  const { bg, surface, text } = groundOf(catalogue, key)
  return readableOn(catalogue.errorColour, bg, surface, text)
}

/**
 * The accent as text on this ground (<prefix>-accent-text): the raw accent where
 * it reads, else darkened toward the ground's text until it reads on the bg, the
 * surface and each of ACCENT_TEXT_TINTS (the accent tinting the bg).
 */
export function accentTextOf(catalogue: LookCatalogue, ground: string, accent: string): string {
  const { bg, surface, text } = groundOf(catalogue, ground)
  const { hex } = accentOf(catalogue, accent)
  const tints = ACCENT_TEXT_TINTS.map((alpha) => mixHex(bg, hex, alpha))
  return readableOnAll(hex, [bg, surface, ...tints], text)
}

/** What the theme sets straight on the page, so over its glow: text, text2, the error text and these accents as text. */
function pageTexts(catalogue: LookCatalogue, ground: string, accents: readonly string[]): readonly string[] {
  const { text, text2 } = groundOf(catalogue, ground)
  const accentTexts = accents.map((accent) => accentTextOf(catalogue, ground, accent))
  return [text, text2, errorTextOf(catalogue, ground), ...accentTexts]
}

function allReadOn(texts: readonly string[], fill: string): boolean {
  return texts.every((text) => contrastRatio(text, fill) >= AA_TEXT)
}

/**
 * The largest whole glow, the catalogue's glow minimum to `limit`, at which
 * every one of `texts` reads at AA on `bg` tinted by `colour` at
 * glow / GLOW_SCALE, and at every glow below it.
 */
function largestReadableGlow(min: number, bg: string, colour: string, texts: readonly string[], limit: number): number {
  let glow = min
  while (glow < limit && allReadOn(texts, mixHex(bg, colour, (glow + 1) / GLOW_SCALE))) glow += 1
  return glow
}

/**
 * The largest whole glow (the catalogue's glow maximum at most) this ground and
 * accent can take: at its brightest the page glow tints the bg under text, and
 * text, text2, the accent text and the error text must still read at AA there
 * (M50). A glow slider stops here; glowVars() never paints past it.
 */
export function safeGlow(catalogue: LookCatalogue, ground: string, accent: string): number {
  const { bg } = groundOf(catalogue, ground)
  const { hex } = accentOf(catalogue, accent)
  return largestReadableGlow(catalogue.glow.min, bg, hex, pageTexts(catalogue, ground, [accent]), catalogue.glow.max)
}

/** safeGlow() as the glow's alpha. Pure. */
export function safeGlowAlpha(catalogue: LookCatalogue, ground: string, accent: string): number {
  return safeGlow(catalogue, ground, accent) / GLOW_SCALE
}

/**
 * The snag glow's alpha on this ground: SNAG_GLOW at most, lower where the
 * error colour would take text, text2, the error text or any accent's text
 * under AA. Per ground, because the snag glow is the same colour in every look.
 */
export function snagGlowAlpha(catalogue: LookCatalogue, ground: string): number {
  const { bg } = groundOf(catalogue, ground)
  const texts = pageTexts(catalogue, ground, keysOf(catalogue.accents))
  return largestReadableGlow(catalogue.glow.min, bg, catalogue.errorColour, texts, SNAG_GLOW) / GLOW_SCALE
}

/** The ground's colours, the Deliverable sheet, the mascot's dusk, the fixed error with its readable text, and the snag glow. */
export function groundVars(catalogue: LookCatalogue, key: string): CssVars {
  const ground = groundOf(catalogue, key)
  const sheet = sheetOf(catalogue, key)
  return prefixed(catalogue, {
    'ground-bg': ground.bg,
    'ground-surface': ground.surface,
    'ground-text': ground.text,
    'ground-text-2': ground.text2,
    'ground-line': ground.line,
    'ground-chip': ground.chip,
    'ground-panel': ground.panel,
    'sheet-bg': sheet.bg,
    'sheet-text': sheet.text,
    'auto-dusk': duskOf(catalogue, key),
    error: catalogue.errorColour,
    'error-text': errorTextOf(catalogue, key),
    [GLOW_SNAG_NAME]: rgba(catalogue.errorColour, snagGlowAlpha(catalogue, key)),
  })
}

export function accentVars(catalogue: LookCatalogue, key: string): CssVars {
  const accent = accentOf(catalogue, key)
  return prefixed(catalogue, {
    accent: accent.hex,
    'accent-ink': accent.ink,
    'accent-line': rgba(accent.hex, ACCENT_LINE_ALPHA),
    'accent-fill': rgba(accent.hex, ACCENT_FILL_ALPHA),
  })
}

/**
 * The accent as text (accentTextOf: on the ground and on the accent's own
 * tints), and the same for text in the accent on the Deliverable sheet.
 */
export function accentTextVars(catalogue: LookCatalogue, ground: string, accent: string): CssVars {
  const sheet = sheetOf(catalogue, ground)
  return prefixed(catalogue, {
    'accent-text': accentTextOf(catalogue, ground, accent),
    'sheet-accent': readableOn(accentOf(catalogue, accent).hex, sheet.bg, sheet.bg, sheet.text),
  })
}

/** The display type style (headings, the mascot's caption, card titles). Body text stays the core's. */
export function fontVars(catalogue: LookCatalogue, key: string): CssVars {
  const font = entryOf(catalogue.typeStyles, key, 'type style')
  return prefixed(catalogue, {
    'display-family': font.family,
    'display-style': font.style,
    'display-weight': String(font.weight),
    'display-tracking': font.letterSpacing,
    'display-hero': `${font.sizes.hero}px`,
    'display-brand': `${font.sizes.brand}px`,
    'display-status': `${font.sizes.status}px`,
    'display-card': `${font.sizes.card}px`,
  })
}

export function shellVars(catalogue: LookCatalogue, key: string): CssVars {
  const shell = entryOf(catalogue.shells, key, 'shell')
  return prefixed(catalogue, { 'auto-shell': shell.hex, 'auto-shell-filter': shell.filter })
}

export function lightVars(catalogue: LookCatalogue, key: string): CssVars {
  return prefixed(catalogue, { 'auto-light': entryOf(catalogue.lights, key, 'light').hex })
}

/** The page glow: the accent at glow / GLOW_SCALE, never past the pair's safeGlowAlpha(). */
export function glowVars(catalogue: LookCatalogue, ground: string, accent: string, glow: number): CssVars {
  const alpha = Math.min(glow / GLOW_SCALE, safeGlowAlpha(catalogue, ground, accent))
  return prefixed(catalogue, { [GLOW_NAME]: rgba(accentOf(catalogue, accent).hex, alpha) })
}

export function lookToCssVars(catalogue: LookCatalogue, look: LookPicks): CssVars {
  return {
    ...groundVars(catalogue, look.ground),
    ...accentVars(catalogue, look.accent),
    ...accentTextVars(catalogue, look.ground, look.accent),
    ...fontVars(catalogue, look.font),
    ...shellVars(catalogue, look.shell),
    ...lightVars(catalogue, look.light),
    ...glowVars(catalogue, look.ground, look.accent, look.glow),
  }
}

/** Sets every property of `vars` on `element` (normally document.documentElement). */
export function applyCssVars(element: HTMLElement, vars: CssVars): void {
  Object.entries(vars).forEach(([name, value]) => element.style.setProperty(name, value))
}
