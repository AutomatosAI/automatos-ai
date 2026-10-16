import {
  ACCENTS,
  ACCENT_KEYS,
  ERROR_COLOUR,
  GROUNDS,
  LIGHTS,
  SHELLS,
  TYPE_STYLES,
  type AccentKey,
  type GroundKey,
  type LightKey,
  type ShellKey,
  type TypeStyleKey,
} from './catalogue'
import { AA_TEXT, contrastRatio, mixHex, readableOn, readableOnAll, relativeLuminance } from './contrast'
import { DEFAULT_LOOK } from './presets'
import { GLOW_MAX, GLOW_MIN, type Look } from './types'

/**
 * PRD-260 — a look as Play's CSS custom properties, set on <html> only under
 * Play (K3). Pure. Every name is --pl-* (K2): Play's own --accent and --ring
 * would clobber Studio's HSL tokens of the same names; Studio's tokens come
 * from studio-tokens.ts. Each section has its own builder so the pre-paint
 * script's table is made from exactly the same values.
 */

export type CssVars = Readonly<Record<string, string>>

/** --pl-accent-line: the accent at 45% alpha (selected borders, focus halos). */
export const ACCENT_LINE_ALPHA = 0.45
/** --pl-accent-fill: the accent at 12% alpha (tinted grounds: a running tool line). */
export const ACCENT_FILL_ALPHA = 0.12
/** app/play-theme.css --pl-tint-accent-halo: the focus halo and the top of a Deliverable card's tint. */
export const ACCENT_HALO_ALPHA = 0.18
/** app/play-theme.css --pl-tint-accent-faint: the far end of a Deliverable card's tint. */
export const ACCENT_FAINT_ALPHA = 0.04
/** The accent tints Play sets the accent as text on (M50): --pl-accent-text reads on each, over the bg. */
export const ACCENT_TEXT_TINTS: readonly number[] = [ACCENT_HALO_ALPHA, ACCENT_FILL_ALPHA, ACCENT_FAINT_ALPHA]
/** --pl-glow is the accent at glow / GLOW_SCALE alpha. */
export const GLOW_SCALE = 100
/** The page glow's variable (the pre-paint script builds its value from the stored glow). */
export const GLOW_VAR = '--pl-glow'
/** The snag glow's variable: the error coral at the ground's snag alpha (snagGlowAlpha). */
export const GLOW_SNAG_VAR = '--pl-glow-snag'
/** The snag glow's strongest coral, as a whole glow: Play's 15%, lower on a ground whose text needs it. */
export const SNAG_GLOW = 15

const HEX = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i

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

/** True for a ground whose bg is darker than its text (every ground but Paper). */
export function isDarkGround(key: GroundKey): boolean {
  const { bg, text } = GROUNDS[key]
  return relativeLuminance(bg) < relativeLuminance(text)
}

/**
 * A Deliverable's page is a printed sheet on the desk (Ember mock 05): the
 * ground's text colour with its bg as ink on a dark ground (the cream sheet),
 * the ground's surface on a light one. Either way the pair is one the catalogue
 * tests already hold to AA.
 */
export function sheetOf(key: GroundKey): Sheet {
  const { bg, surface, text } = GROUNDS[key]
  return isDarkGround(key) ? { bg: text, text: bg } : { bg: surface, text }
}

/** How far dusk sits from the ground's secondary text toward its bg (dusk = 45% text2, 55% bg). */
export const DUSK_TOWARD_BG = 0.55

/**
 * Auto's ring while he sleeps or is lost: a dim colour made from the ground, so
 * it suits every look and never borrows the light or the accent.
 */
export function duskOf(key: GroundKey): string {
  const { bg, text2 } = GROUNDS[key]
  return mixHex(text2, bg, DUSK_TOWARD_BG)
}

/** The error as text on this ground (--pl-error-text): the coral, darkened where it doesn't read. */
export function errorTextOf(key: GroundKey): string {
  const { bg, surface, text } = GROUNDS[key]
  return readableOn(ERROR_COLOUR, bg, surface, text)
}

/**
 * The accent as text on this ground (--pl-accent-text): the raw accent where it
 * reads, else darkened toward the ground's text until it reads on the bg, the
 * surface and each of ACCENT_TEXT_TINTS (the accent tinting the bg).
 */
export function accentTextOf(ground: GroundKey, accent: AccentKey): string {
  const { bg, surface, text } = GROUNDS[ground]
  const { hex } = ACCENTS[accent]
  const tints = ACCENT_TEXT_TINTS.map((alpha) => mixHex(bg, hex, alpha))
  return readableOnAll(hex, [bg, surface, ...tints], text)
}

/** What Play sets straight on the page, so over its glow: text, text2, the error text and these accents as text. */
function pageTexts(ground: GroundKey, accents: readonly AccentKey[]): readonly string[] {
  const { text, text2 } = GROUNDS[ground]
  return [text, text2, errorTextOf(ground), ...accents.map((accent) => accentTextOf(ground, accent))]
}

function allReadOn(texts: readonly string[], fill: string): boolean {
  return texts.every((text) => contrastRatio(text, fill) >= AA_TEXT)
}

/**
 * The largest whole glow, GLOW_MIN to `limit`, at which every one of `texts`
 * reads at AA on `bg` tinted by `colour` at glow / GLOW_SCALE, and at every
 * glow below it.
 */
function largestReadableGlow(bg: string, colour: string, texts: readonly string[], limit: number): number {
  let glow = GLOW_MIN
  while (glow < limit && allReadOn(texts, mixHex(bg, colour, (glow + 1) / GLOW_SCALE))) glow += 1
  return glow
}

/**
 * The largest whole glow (GLOW_MAX at most) this ground and accent can take:
 * at its brightest (data-glow="bright", opacity 1) the page glow tints the bg
 * under text, and text, text2, the accent text and the error text must still
 * read at AA there (M50). The Glow slider stops here; glowVars() never paints past it.
 */
export function safeGlow(ground: GroundKey, accent: AccentKey): number {
  return largestReadableGlow(GROUNDS[ground].bg, ACCENTS[accent].hex, pageTexts(ground, [accent]), GLOW_MAX)
}

/** safeGlow() as the glow's alpha. Pure. */
export function safeGlowAlpha(ground: GroundKey, accent: AccentKey): number {
  return safeGlow(ground, accent) / GLOW_SCALE
}

/**
 * The snag glow's alpha on this ground: SNAG_GLOW at most, lower where the
 * coral would take text, text2, the error text or any accent's text under AA.
 * Per ground, because the snag glow is the same coral in every look.
 */
export function snagGlowAlpha(ground: GroundKey): number {
  return largestReadableGlow(GROUNDS[ground].bg, ERROR_COLOUR, pageTexts(ground, ACCENT_KEYS), SNAG_GLOW) / GLOW_SCALE
}

/** The ground's colours, the Deliverable sheet, Auto's dusk, the fixed error with its readable text, and the snag glow. */
export function groundVars(key: GroundKey): CssVars {
  const ground = GROUNDS[key]
  const sheet = sheetOf(key)
  return {
    '--pl-ground-bg': ground.bg,
    '--pl-ground-surface': ground.surface,
    '--pl-ground-text': ground.text,
    '--pl-ground-text-2': ground.text2,
    '--pl-ground-line': ground.line,
    '--pl-ground-chip': ground.chip,
    '--pl-ground-panel': ground.panel,
    '--pl-sheet-bg': sheet.bg,
    '--pl-sheet-text': sheet.text,
    '--pl-auto-dusk': duskOf(key),
    '--pl-error': ERROR_COLOUR,
    '--pl-error-text': errorTextOf(key),
    [GLOW_SNAG_VAR]: rgba(ERROR_COLOUR, snagGlowAlpha(key)),
  }
}

export function accentVars(key: AccentKey): CssVars {
  const accent = ACCENTS[key]
  return {
    '--pl-accent': accent.hex,
    '--pl-accent-ink': accent.ink,
    '--pl-accent-line': rgba(accent.hex, ACCENT_LINE_ALPHA),
    '--pl-accent-fill': rgba(accent.hex, ACCENT_FILL_ALPHA),
  }
}

/**
 * The accent as text (accentTextOf: on the ground and on the accent's own
 * tints), and the same for text in the accent on the Deliverable sheet.
 */
export function accentTextVars(ground: GroundKey, accent: AccentKey): CssVars {
  const sheet = sheetOf(ground)
  return {
    '--pl-accent-text': accentTextOf(ground, accent),
    '--pl-sheet-accent': readableOn(ACCENTS[accent].hex, sheet.bg, sheet.bg, sheet.text),
  }
}

/** The display type style (headings, Auto's caption, card titles). Body text stays Geist. */
export function fontVars(key: TypeStyleKey): CssVars {
  const font = TYPE_STYLES[key]
  return {
    '--pl-display-family': font.family,
    '--pl-display-style': font.style,
    '--pl-display-weight': String(font.weight),
    '--pl-display-tracking': font.letterSpacing,
    '--pl-display-hero': `${font.sizes.hero}px`,
    '--pl-display-brand': `${font.sizes.brand}px`,
    '--pl-display-status': `${font.sizes.status}px`,
    '--pl-display-card': `${font.sizes.card}px`,
  }
}

export function shellVars(key: ShellKey): CssVars {
  const shell = SHELLS[key]
  return { '--pl-auto-shell': shell.hex, '--pl-auto-shell-filter': shell.filter }
}

export function lightVars(key: LightKey): CssVars {
  return { '--pl-auto-light': LIGHTS[key].hex }
}

/** The page glow: the accent at glow / GLOW_SCALE, never past the pair's safeGlowAlpha(). */
export function glowVars(ground: GroundKey, accent: AccentKey, glow: number): CssVars {
  return { [GLOW_VAR]: rgba(ACCENTS[accent].hex, Math.min(glow / GLOW_SCALE, safeGlowAlpha(ground, accent))) }
}

export function lookToCssVars(look: Look): CssVars {
  return {
    ...groundVars(look.ground),
    ...accentVars(look.accent),
    ...accentTextVars(look.ground, look.accent),
    ...fontVars(look.font),
    ...shellVars(look.shell),
    ...lightVars(look.light),
    ...glowVars(look.ground, look.accent, look.glow),
  }
}

/** Every property a look sets: the same set for every look. */
export const LOOK_VAR_NAMES: readonly string[] = Object.keys(lookToCssVars(DEFAULT_LOOK))

/** Sets every property of `vars` on `element` (normally document.documentElement). */
export function applyCssVars(element: HTMLElement, vars: CssVars): void {
  Object.entries(vars).forEach(([name, value]) => element.style.setProperty(name, value))
}
