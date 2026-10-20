import type { CSSProperties } from 'react'

import { ACCENTS, GROUNDS, LIGHTS, TYPE_STYLES, type TypeStyleKey } from '../../lib/looks/catalogue'
import type { Look } from '../../lib/looks/types'

/**
 * PRD-260 — the Looks page shows options that aren't the active look, so each
 * swatch, type sample and mini card draws its own catalogue entry. The values
 * reach the CSS modules as --pl-look-* custom properties set inline (K2),
 * taken from lib/looks/catalogue.ts only: never user text, never a literal
 * colour. Ported from Automatos Play's components/looks/swatchStyle.ts.
 */

/** A style prop that may also carry Play's custom properties (React sets those with setProperty). */
export type SwatchStyle = CSSProperties & { readonly [name: `--pl-${string}`]: string }

/** One colour for a swatch, a dot or a chip. */
export function colourStyle(colour: string): SwatchStyle {
  return { '--pl-look-swatch': colour }
}

/** A type option set in its own display style, at its card size. */
export function typeStyle(key: TypeStyleKey): SwatchStyle {
  const font = TYPE_STYLES[key]
  return {
    '--pl-look-option-family': font.family,
    '--pl-look-option-style': font.style,
    '--pl-look-option-weight': String(font.weight),
    '--pl-look-option-tracking': font.letterSpacing,
    '--pl-look-option-size': `${font.sizes.card}px`,
  }
}

/** A look's mini card in the Looks row: its ground, type, light and accent. */
export function cardStyle(look: Look): SwatchStyle {
  const ground = GROUNDS[look.ground]
  const font = TYPE_STYLES[look.font]
  return {
    '--pl-look-card-bg': ground.bg,
    '--pl-look-card-text': ground.text,
    '--pl-look-card-accent': ACCENTS[look.accent].hex,
    '--pl-look-card-light': LIGHTS[look.light].hex,
    '--pl-look-card-family': font.family,
    '--pl-look-card-style': font.style,
    '--pl-look-card-weight': String(font.weight),
  }
}
