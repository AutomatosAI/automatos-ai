import type { CSSProperties } from 'react'

/**
 * PRD-260 §4 — the style sheet draws values that are not the page's own tokens
 * (a swatch's colour, a spacing step, a radius) through custom properties set
 * inline, read by the sheet's CSS modules. The values come from the active look
 * (lib/looks/css.ts) or the theme's token names (lib/looks/style.ts) only:
 * never user text, never a literal colour. Ported from Automatos Play's
 * components/looks/swatchStyle.ts; the names are --pl-* (K2).
 */

export type SpecimenVars = Readonly<Record<`--pl-${string}`, string>>

/** Custom properties as a style prop: React sets "--name" keys with setProperty. */
export function specimenStyle(vars: SpecimenVars): CSSProperties {
  // CSSProperties has no "--*" keys; a cast is the documented way to pass custom properties.
  return vars as CSSProperties
}

/** A swatch's colour, from the look's variables. */
export function swatchStyle(colour: string): CSSProperties {
  return specimenStyle({ '--pl-specimen-swatch': colour })
}

/** A spacing step drawn at a theme token's own value, e.g. var(--pl-space-4). */
export function stepStyle(token: string): CSSProperties {
  return specimenStyle({ '--pl-specimen-step': `var(${token})` })
}

/** A radius drawn at a theme token's own value, e.g. var(--pl-radius-card). */
export function radiusStyle(token: string): CSSProperties {
  return specimenStyle({ '--pl-specimen-radius': `var(${token})` })
}
