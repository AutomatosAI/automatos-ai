import * as kitCss from '@/lib/look-kit/css'
import * as kitStudio from '@/lib/look-kit/studio-tokens'
import type { LookCatalogue } from '@/lib/look-kit/types'
import { validateLook as validateWithSpec, type LookKnobSpec, type LookSpec } from '@/lib/look-kit/validate'

import {
  ACCENTS,
  ACCENT_KEYS,
  ERROR_COLOUR,
  GROUNDS,
  GROUND_KEYS,
  LIGHTS,
  LIGHT_KEYS,
  SHELLS,
  SHELL_KEYS,
  TYPE_STYLES,
  TYPE_STYLE_KEYS,
  type AccentKey,
  type GroundKey,
  type LightKey,
  type ShellKey,
  type TypeStyleKey,
} from './catalogue'
import { DEFAULT_LOOK } from './presets'
import { GLOW_MAX, GLOW_MIN, LOOK_INTERFACE, LOOK_SCHEMA, type Look, type ValidationResult } from './types'

/**
 * PRD-261 A-13a — Play's look engine: the core look runtime (lib/look-kit, PA-9) with Play's
 * catalogue, look format and --pl- prefix bound in, under the names Play's code has always
 * called. The runtime knows no theme; this file is where Play tells it which one. It moves to
 * the Play package with the rest of lib/looks (A-16b).
 */

/** Play's variables are --pl-* (K2): Play's own --accent and --ring would clobber Studio's tokens. */
const PLAY_VAR_PREFIX = 'pl'

export const PLAY_LOOK_CATALOGUE: LookCatalogue = {
  prefix: PLAY_VAR_PREFIX,
  errorColour: ERROR_COLOUR,
  glow: { min: GLOW_MIN, max: GLOW_MAX },
  grounds: GROUNDS,
  accents: ACCENTS,
  typeStyles: TYPE_STYLES,
  shells: SHELLS,
  lights: LIGHTS,
}

/** Play's look format: types.ts's header and fields, the catalogue's keys as each knob's choices. */
export const PLAY_LOOK_SPEC: LookSpec = {
  lookSchema: LOOK_SCHEMA,
  interface: LOOK_INTERFACE,
  knobs: new Map<string, LookKnobSpec>([
    ['ground', { kind: 'choice', choices: GROUND_KEYS, listName: "the catalogue's backgrounds" }],
    ['accent', { kind: 'choice', choices: ACCENT_KEYS, listName: "the catalogue's accents" }],
    ['font', { kind: 'choice', choices: TYPE_STYLE_KEYS, listName: "the catalogue's type styles" }],
    ['shell', { kind: 'choice', choices: SHELL_KEYS, listName: "the catalogue's shells" }],
    ['light', { kind: 'choice', choices: LIGHT_KEYS, listName: "the catalogue's lights" }],
    ['glow', { kind: 'number', min: GLOW_MIN, max: GLOW_MAX, integer: true }],
  ]),
}

const CATALOGUE = PLAY_LOOK_CATALOGUE

type CssVars = kitCss.CssVars

/** The page glow's variable (the pre-paint script builds its value from the stored glow). */
export const GLOW_VAR = kitCss.glowVar(CATALOGUE)
/** The snag glow's variable: the error coral at the ground's snag alpha (snagGlowAlpha). */
export const GLOW_SNAG_VAR = kitCss.glowSnagVar(CATALOGUE)

export function isDarkGround(key: GroundKey): boolean {
  return kitCss.isDarkGround(CATALOGUE, key)
}

export function sheetOf(key: GroundKey): kitCss.Sheet {
  return kitCss.sheetOf(CATALOGUE, key)
}

export function duskOf(key: GroundKey): string {
  return kitCss.duskOf(CATALOGUE, key)
}

export function safeGlow(ground: GroundKey, accent: AccentKey): number {
  return kitCss.safeGlow(CATALOGUE, ground, accent)
}

export function safeGlowAlpha(ground: GroundKey, accent: AccentKey): number {
  return kitCss.safeGlowAlpha(CATALOGUE, ground, accent)
}

export function snagGlowAlpha(ground: GroundKey): number {
  return kitCss.snagGlowAlpha(CATALOGUE, ground)
}

export function groundVars(key: GroundKey): CssVars {
  return kitCss.groundVars(CATALOGUE, key)
}

export function accentVars(key: AccentKey): CssVars {
  return kitCss.accentVars(CATALOGUE, key)
}

export function accentTextVars(ground: GroundKey, accent: AccentKey): CssVars {
  return kitCss.accentTextVars(CATALOGUE, ground, accent)
}

export function fontVars(key: TypeStyleKey): CssVars {
  return kitCss.fontVars(CATALOGUE, key)
}

export function shellVars(key: ShellKey): CssVars {
  return kitCss.shellVars(CATALOGUE, key)
}

export function lightVars(key: LightKey): CssVars {
  return kitCss.lightVars(CATALOGUE, key)
}

/** A look as Play's --pl-* variables (K2, K3). */
export function lookToCssVars(look: Look): CssVars {
  return kitCss.lookToCssVars(CATALOGUE, look)
}

/** Every property a look sets: the same set for every look. */
export const LOOK_VAR_NAMES: readonly string[] = Object.keys(lookToCssVars(DEFAULT_LOOK))

export function textFills(key: GroundKey): readonly string[] {
  return kitStudio.textFills(CATALOGUE, key)
}

export function studioAccent(ground: GroundKey, accent: AccentKey): kitStudio.StudioAccent {
  return kitStudio.studioAccent(CATALOGUE, ground, accent)
}

export function studioGroundTokens(key: GroundKey): CssVars {
  return kitStudio.studioGroundTokens(CATALOGUE, key)
}

export function studioAccentTokens(ground: GroundKey, accent: AccentKey): CssVars {
  return kitStudio.studioAccentTokens(CATALOGUE, ground, accent)
}

/** A look as Studio's tokens (PRD-260 D3), so every Studio page under Play takes the look's colours. */
export function lookToStudioTokens(look: Look): CssVars {
  return kitStudio.lookToStudioTokens(CATALOGUE, look)
}

/** Every Studio token a look sets: the same set for every look. */
export const STUDIO_TOKEN_NAMES: readonly string[] = Object.keys(lookToStudioTokens(DEFAULT_LOOK))

/**
 * The one gate for a Play look from outside the code (storage, an imported file, pasted JSON):
 * the runtime's validateLook against PLAY_LOOK_SPEC. The spec lists exactly Look's fields, its
 * header and the catalogue's keys, so a look that passes is a Look.
 */
export function validateLook(input: unknown): ValidationResult {
  const result = validateWithSpec(PLAY_LOOK_SPEC, input)
  return result.ok ? { ok: true, look: result.look as unknown as Look } : result
}
