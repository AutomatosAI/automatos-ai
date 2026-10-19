import { ACCENTS, ERROR_COLOUR, GROUNDS, type AccentKey, type GroundKey } from './catalogue'
import {
  AA_TEXT,
  MIX_STEP_PERCENT,
  contrastRatio,
  mixHex,
  readableOn,
  readableOnAll,
  readableOnTinted,
  readsOnAll,
  toHex,
} from './contrast'
import { isDarkGround, type CssVars } from './css'
import { hexToTriplet } from './hsl'
import { DEFAULT_LOOK } from './presets'
import type { Look } from './types'

/**
 * PRD-260 D3 — a look as Studio's tokens, so every Studio page under Play takes
 * the look's colours with no change to it. Values are HSL triplets 'H S% L%'
 * (tailwind.config.ts reads hsl(var(--x))), except the aliases globals.css
 * writes pre-wrapped (--paper, --bg-1, --bg-2, --fg, --fg-2, --fg-3,
 * --border-2, --accent-2), which are written the same way here.
 *
 * Studio's consequence tokens (--accent, --destructive, their foregrounds and
 * the --accent-2 tint) are Play's coral made readable on the ground, never the
 * look's accent (D-REVIEW F9): Studio draws errors and destructive actions in
 * them. The other semantic tokens (success, warning, info, agent, chart-*,
 * stage-*) are NOT set: the tone follows the ground under Play (K4), so
 * Studio's own Light or Dark values apply. Pure; split into the ground's
 * tokens and the ground × accent tokens so the pre-paint table stays small.
 */

/** --accent-2, the consequence colour's tint: Studio Dark's alpha on a dark ground. */
export const ACCENT_TINT_ALPHA_DARK = 0.14
/** --accent-2 on Paper: Studio Light's alpha. */
export const ACCENT_TINT_ALPHA_LIGHT = 0.1
const ALPHA_DECIMALS = 2

/** --border-2 (hover borders): the border this far toward the text (Studio's own step: 80% → 70%, 22% → 32%). */
export const BORDER_STRONG_TOWARD_TEXT = 0.15
/** --fg-3 (faded meta text): text2 at most this many percent toward the bg, never below READABLE_TARGET. */
export const FADED_TEXT_MAX_PERCENT = 20
/** The strongest tint of --primary that Studio puts text-primary on (its bg-primary/20 chips). */
export const PRIMARY_TINT_MAX = 0.2

const PERCENT = 100
const RGBA = /^rgba\((\d{1,3}),(\d{1,3}),(\d{1,3}),(\d*\.?\d+)\)$/

/** A catalogue rgba() line or chip composited over an opaque #rrggbb ground. */
export function overGround(colour: string, ground: string): string {
  const match = RGBA.exec(colour)
  if (!match) throw new Error(`${colour} isn't an rgba() colour`)
  const [, red, green, blue, alpha] = match
  return mixHex(ground, toHex([Number(red), Number(green), Number(blue)]), Number(alpha))
}

/** The fills Studio sets text on under a look: bg, surface, and the chip over each (its --secondary and --muted). */
export function textFills(key: GroundKey): readonly string[] {
  const { bg, surface, chip } = GROUNDS[key]
  return [bg, surface, overGround(chip, bg), overGround(chip, surface)]
}

/**
 * text2 faded toward the bg as far as FADED_TEXT_MAX_PERCENT while it still
 * reads on every text fill; when no fade does, text2 made readable on them.
 */
export function fadedText(key: GroundKey): string {
  const { bg, text, text2 } = GROUNDS[key]
  const fills = textFills(key)
  for (let percent = FADED_TEXT_MAX_PERCENT; percent > 0; percent -= MIX_STEP_PERCENT) {
    const candidate = mixHex(text2, bg, percent / PERCENT)
    if (readsOnAll(candidate, fills)) return candidate
  }
  return readableOnAll(text2, fills, text)
}

/** The ground's opaque colours, rgba lines and chips composited first. */
export interface GroundPalette {
  readonly bg: string
  readonly card: string
  readonly text: string
  readonly text2: string
  readonly faded: string
  readonly border: string
  readonly borderStrong: string
  /** Chip over the bg: a step off the page, as Studio's secondary is. */
  readonly secondary: string
  /** Chip over the card: hover and muted fills, where text2 still reads. */
  readonly muted: string
}

export function groundPalette(key: GroundKey): GroundPalette {
  const ground = GROUNDS[key]
  const border = overGround(ground.line, ground.bg)
  return {
    bg: ground.bg,
    card: ground.surface,
    text: ground.text,
    text2: ground.text2,
    faded: fadedText(key),
    border,
    borderStrong: mixHex(border, ground.text, BORDER_STRONG_TOWARD_TEXT),
    secondary: overGround(ground.chip, ground.bg),
    muted: overGround(ground.chip, ground.surface),
  }
}

function wrapped(hex: string): string {
  return `hsl(${hexToTriplet(hex)})`
}

/** Studio's fills and lines: page, card, chips, borders. */
function fillTokens(palette: GroundPalette): CssVars {
  const triplet = hexToTriplet
  return {
    '--background': triplet(palette.bg),
    '--background-gradient-from': triplet(palette.bg),
    '--background-gradient-to': triplet(palette.bg),
    '--card': triplet(palette.card),
    '--popover': triplet(palette.card),
    '--secondary': triplet(palette.secondary),
    '--muted': triplet(palette.muted),
    '--cc-hover': triplet(palette.muted),
    '--border': triplet(palette.border),
    '--input': triplet(palette.border),
    '--glass-border-hover': triplet(palette.borderStrong),
    '--cc-hover-border': triplet(palette.borderStrong),
  }
}

/** Studio's text on those fills. */
function textTokens(palette: GroundPalette): CssVars {
  const triplet = hexToTriplet
  return {
    '--foreground': triplet(palette.text),
    '--card-foreground': triplet(palette.text),
    '--popover-foreground': triplet(palette.text),
    '--secondary-foreground': triplet(palette.text),
    '--muted-foreground': triplet(palette.text2),
  }
}

/** The aliases globals.css writes pre-wrapped. */
function aliasTokens(palette: GroundPalette): CssVars {
  return {
    '--paper': wrapped(palette.card),
    '--bg-1': wrapped(palette.bg),
    '--bg-2': wrapped(palette.secondary),
    '--fg': wrapped(palette.text),
    '--fg-2': wrapped(palette.text2),
    '--fg-3': wrapped(palette.faded),
    '--border-2': wrapped(palette.borderStrong),
  }
}

/** A colour as a fill with the text that reads on it. */
export interface StudioAccent {
  /** The fill, also used as text (and, for the accent, as the focus ring). */
  readonly fill: string
  /** Text on the fill. */
  readonly ink: string
}

/**
 * Studio's consequence colour under a look: Play's coral, readable on the bg
 * and the surface (the same colour as --pl-error-text), with the ground's bg as
 * its ink when that reads at AA, else the surface.
 */
export function studioConsequence(key: GroundKey): StudioAccent {
  const { bg, surface, text } = GROUNDS[key]
  const fill = readableOn(ERROR_COLOUR, bg, surface, text)
  return { fill, ink: contrastRatio(bg, fill) >= AA_TEXT ? bg : surface }
}

/** --accent-2's alpha: Studio Dark's on a dark ground, Studio Light's on Paper. */
function tintAlpha(key: GroundKey): string {
  return (isDarkGround(key) ? ACCENT_TINT_ALPHA_DARK : ACCENT_TINT_ALPHA_LIGHT).toFixed(ALPHA_DECIMALS)
}

/** --accent and --destructive (one colour in Studio, burnt orange there), their inks and --accent-2. */
function consequenceTokens(key: GroundKey): CssVars {
  const { fill, ink } = studioConsequence(key)
  return {
    '--accent': hexToTriplet(fill),
    '--accent-foreground': hexToTriplet(ink),
    '--destructive': hexToTriplet(fill),
    '--destructive-foreground': hexToTriplet(ink),
    '--accent-2': `hsl(${hexToTriplet(fill)} / ${tintAlpha(key)})`,
  }
}

/** Studio's tokens that depend only on the ground. */
export function studioGroundTokens(key: GroundKey): CssVars {
  const palette = groundPalette(key)
  return { ...fillTokens(palette), ...textTokens(palette), ...aliasTokens(palette), ...consequenceTokens(key) }
}

/**
 * The look's accent as Studio's --primary, with its ink. The fill reads on all
 * four text fills, flat and under a PRIMARY_TINT_MAX tint of itself
 * (readableOnTinted); the ink is the accent's designed ink when the raw accent
 * passes (the catalogue holds that pair to AA), else the ground's bg, which
 * reads on the fill by readableOnTinted's own guarantee (the bg is a fill).
 */
export function studioAccent(ground: GroundKey, accent: AccentKey): StudioAccent {
  const { bg, text } = GROUNDS[ground]
  const { hex, ink } = ACCENTS[accent]
  const fill = readableOnTinted(hex, textFills(ground), text, PRIMARY_TINT_MAX)
  return { fill, ink: fill.toUpperCase() === hex.toUpperCase() ? ink : bg }
}

/** Studio's tokens that depend on the ground and the accent: --primary and the focus ring. */
export function studioAccentTokens(ground: GroundKey, accent: AccentKey): CssVars {
  const { fill, ink } = studioAccent(ground, accent)
  return {
    '--primary': hexToTriplet(fill),
    '--primary-foreground': hexToTriplet(ink),
    '--ring': hexToTriplet(fill),
  }
}

export function lookToStudioTokens(look: Look): CssVars {
  return { ...studioGroundTokens(look.ground), ...studioAccentTokens(look.ground, look.accent) }
}

/** Every Studio token a look sets: the same set for every look. */
export const STUDIO_TOKEN_NAMES: readonly string[] = Object.keys(lookToStudioTokens(DEFAULT_LOOK))
