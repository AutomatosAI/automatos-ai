/**
 * PRD-261 A-13a — the look runtime's model of a theme's catalogue (PA-9).
 *
 * The runtime holds no catalogue, no variable prefix and no look format of its own: a theme hands
 * it a LookCatalogue (its grounds, accents, type styles, shells and lights, its error colour, its
 * glow range and the prefix its CSS reads), and a look's picks, one key per section plus a glow.
 * lib/look-kit/css.ts turns those into the theme's custom properties, studio-tokens.ts into
 * Studio's tokens. Colours are #rrggbb, except a ground's line, chip and panel: rgba() over the bg.
 */

export interface LookGround {
  readonly bg: string
  readonly surface: string
  readonly text: string
  readonly text2: string
  readonly line: string
  readonly chip: string
  readonly panel: string
}

export interface LookAccent {
  readonly hex: string
  /** Button text on the accent fill, chosen per accent for AA. */
  readonly ink: string
}

export interface LookTypeSizes {
  readonly hero: number
  readonly brand: number
  readonly status: number
  readonly card: number
}

export interface LookTypeStyle {
  /** The CSS font-family. */
  readonly family: string
  readonly style: string
  readonly weight: number
  readonly letterSpacing: string
  /** Display sizes in px. */
  readonly sizes: LookTypeSizes
}

export interface LookShell {
  readonly hex: string
  /** CSS filter applied to the mascot's video and still. */
  readonly filter: string
}

export interface LookLight {
  readonly hex: string
}

export interface LookGlowRange {
  /** Whole numbers: the page glow is the accent at glow / GLOW_SCALE alpha. */
  readonly min: number
  readonly max: number
}

export interface LookCatalogue {
  /** The custom-property prefix the theme's CSS reads: 'acme' names --acme-ground-bg. */
  readonly prefix: string
  /** The error colour, the same in every look. */
  readonly errorColour: string
  readonly glow: LookGlowRange
  readonly grounds: Readonly<Record<string, LookGround>>
  readonly accents: Readonly<Record<string, LookAccent>>
  readonly typeStyles: Readonly<Record<string, LookTypeStyle>>
  readonly shells: Readonly<Record<string, LookShell>>
  readonly lights: Readonly<Record<string, LookLight>>
}

/** One look's picks: a key of each catalogue section, and the glow. */
export interface LookPicks {
  readonly ground: string
  readonly accent: string
  readonly font: string
  readonly shell: string
  readonly light: string
  readonly glow: number
}

/** The section's own entry for `key`; a key the section lacks (or inherits, '__proto__') throws. */
export function entryOf<T>(section: Readonly<Record<string, T>>, key: string, what: string): T {
  if (!Object.prototype.hasOwnProperty.call(section, key)) throw new Error(`"${key}" isn't a ${what} of this catalogue`)
  return section[key]
}

/** The section's own keys, in its order. */
export function keysOf(section: Readonly<Record<string, unknown>>): readonly string[] {
  return Object.keys(section).filter((key) => Object.prototype.hasOwnProperty.call(section, key))
}
