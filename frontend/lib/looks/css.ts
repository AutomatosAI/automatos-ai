import {
  ACCENTS,
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
import { mixHex, readableOn, relativeLuminance } from './contrast'
import { DEFAULT_LOOK } from './presets'
import type { Look } from './types'

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
/** --pl-accent-fill: the accent at 12% alpha (tinted grounds). */
export const ACCENT_FILL_ALPHA = 0.12
/** --pl-glow is the accent at glow / GLOW_SCALE alpha. */
export const GLOW_SCALE = 100

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

/** The ground's colours, the Deliverable sheet, Auto's dusk, and the fixed error with its readable text on this ground. */
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
    '--pl-error-text': readableOn(ERROR_COLOUR, ground.bg, ground.surface, ground.text),
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
 * The accent as text: darkened toward the ground's text where the raw accent
 * doesn't read, and the same for text in the accent on the Deliverable sheet.
 */
export function accentTextVars(ground: GroundKey, accent: AccentKey): CssVars {
  const { bg, surface, text } = GROUNDS[ground]
  const sheet = sheetOf(ground)
  return {
    '--pl-accent-text': readableOn(ACCENTS[accent].hex, bg, surface, text),
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

export function glowVars(accent: AccentKey, glow: number): CssVars {
  return { '--pl-glow': rgba(ACCENTS[accent].hex, glow / GLOW_SCALE) }
}

export function lookToCssVars(look: Look): CssVars {
  return {
    ...groundVars(look.ground),
    ...accentVars(look.accent),
    ...accentTextVars(look.ground, look.accent),
    ...fontVars(look.font),
    ...shellVars(look.shell),
    ...lightVars(look.light),
    ...glowVars(look.accent, look.glow),
  }
}

/** Every property a look sets: the same set for every look. */
export const LOOK_VAR_NAMES: readonly string[] = Object.keys(lookToCssVars(DEFAULT_LOOK))

/** Sets every property of `vars` on `element` (normally document.documentElement). */
export function applyCssVars(element: HTMLElement, vars: CssVars): void {
  Object.entries(vars).forEach(([name, value]) => element.style.setProperty(name, value))
}
