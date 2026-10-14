import { ACCENTS, GROUNDS, type AccentKey, type GroundKey } from './catalogue'
import { MIX_STEP_PERCENT, mixHex, readableOn, readsOn, toHex } from './contrast'
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
 * Semantic tokens (destructive, success, warning, info, agent, chart-*,
 * stage-*) are NOT set: the tone follows the ground under Play (K4), so
 * Studio's own Light or Dark values apply. Pure; split into the ground's
 * tokens and the ground × accent tokens so the pre-paint table stays small.
 */

/** --accent-2, the accent tint: Studio Dark's alpha on a dark ground. */
export const ACCENT_TINT_ALPHA_DARK = 0.14
/** --accent-2 on Paper: Studio Light's alpha. */
export const ACCENT_TINT_ALPHA_LIGHT = 0.1
const ALPHA_DECIMALS = 2

/** --border-2 (hover borders): the border this far toward the text (Studio's own step: 80% → 70%, 22% → 32%). */
export const BORDER_STRONG_TOWARD_TEXT = 0.15
/** --fg-3 (faded meta text): text2 at most this many percent toward the bg, never below READABLE_TARGET. */
export const FADED_TEXT_MAX_PERCENT = 20

const PERCENT = 100
const RGBA = /^rgba\((\d{1,3}),(\d{1,3}),(\d{1,3}),(\d*\.?\d+)\)$/

/** A catalogue rgba() line or chip composited over an opaque #rrggbb ground. */
export function overGround(colour: string, ground: string): string {
  const match = RGBA.exec(colour)
  if (!match) throw new Error(`${colour} isn't an rgba() colour`)
  const [, red, green, blue, alpha] = match
  return mixHex(ground, toHex([Number(red), Number(green), Number(blue)]), Number(alpha))
}

/** text2 faded toward the bg as far as FADED_TEXT_MAX_PERCENT while it still reads on bg and surface. */
export function fadedText(key: GroundKey): string {
  const { bg, surface, text2 } = GROUNDS[key]
  for (let percent = FADED_TEXT_MAX_PERCENT; percent > 0; percent -= MIX_STEP_PERCENT) {
    const candidate = mixHex(text2, bg, percent / PERCENT)
    if (readsOn(candidate, bg, surface)) return candidate
  }
  return mixHex(text2, text2, 0) // normalised #RRGGBB
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

/** Studio's tokens that depend only on the ground. */
export function studioGroundTokens(key: GroundKey): CssVars {
  const palette = groundPalette(key)
  const triplet = hexToTriplet
  return {
    '--background': triplet(palette.bg),
    '--foreground': triplet(palette.text),
    '--card': triplet(palette.card),
    '--card-foreground': triplet(palette.text),
    '--popover': triplet(palette.card),
    '--popover-foreground': triplet(palette.text),
    '--secondary': triplet(palette.secondary),
    '--secondary-foreground': triplet(palette.text),
    '--muted': triplet(palette.muted),
    '--muted-foreground': triplet(palette.text2),
    '--border': triplet(palette.border),
    '--input': triplet(palette.border),
    '--background-gradient-from': triplet(palette.bg),
    '--background-gradient-to': triplet(palette.bg),
    '--glass-border-hover': triplet(palette.borderStrong),
    '--cc-hover': triplet(palette.muted),
    '--cc-hover-border': triplet(palette.borderStrong),
    '--paper': wrapped(palette.card),
    '--bg-1': wrapped(palette.bg),
    '--bg-2': wrapped(palette.secondary),
    '--fg': wrapped(palette.text),
    '--fg-2': wrapped(palette.text2),
    '--fg-3': wrapped(palette.faded),
    '--border-2': wrapped(palette.borderStrong),
  }
}

export interface StudioAccent {
  /** The accent as a fill, as text and as the focus ring: readable on the bg and the card. */
  readonly fill: string
  /** Text on the fill. */
  readonly ink: string
}

/**
 * The readable accent (readableOn) with its ink: the accent's designed ink when
 * the raw accent reads (the catalogue holds that pair to AA), else the ground's
 * bg, which reads on a readable accent by readableOn's own guarantee.
 */
export function studioAccent(ground: GroundKey, accent: AccentKey): StudioAccent {
  const { bg, surface, text } = GROUNDS[ground]
  const { hex, ink } = ACCENTS[accent]
  const fill = readableOn(hex, bg, surface, text)
  return { fill, ink: fill.toUpperCase() === hex.toUpperCase() ? ink : bg }
}

/** Studio's tokens that depend on the ground and the accent. */
export function studioAccentTokens(ground: GroundKey, accent: AccentKey): CssVars {
  const { fill, ink } = studioAccent(ground, accent)
  const tint = isDarkGround(ground) ? ACCENT_TINT_ALPHA_DARK : ACCENT_TINT_ALPHA_LIGHT
  return {
    '--primary': hexToTriplet(fill),
    '--primary-foreground': hexToTriplet(ink),
    '--accent': hexToTriplet(fill),
    '--accent-foreground': hexToTriplet(ink),
    '--ring': hexToTriplet(fill),
    '--accent-2': `hsl(${hexToTriplet(fill)} / ${tint.toFixed(ALPHA_DECIMALS)})`,
  }
}

export function lookToStudioTokens(look: Look): CssVars {
  return { ...studioGroundTokens(look.ground), ...studioAccentTokens(look.ground, look.accent) }
}

/** Every Studio token a look sets: the same set for every look. */
export const STUDIO_TOKEN_NAMES: readonly string[] = Object.keys(lookToStudioTokens(DEFAULT_LOOK))
