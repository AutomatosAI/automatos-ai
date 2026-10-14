/**
 * PRD-260 — the Ember catalogue: the only colours, type styles and Auto finishes
 * a look can name. Values are Play's (docs/design/ember, "Automatos Play Looks":
 * G, A, F, S, L), copied exactly. This file is the one place in the frontend
 * allowed to hold literal colours (the PRD-260 guard test).
 *
 * K6: a type style's `family` reaches its face through the next/font variable
 * (app/play-fonts.ts; Geist through the core's --font-geist-sans), because
 * next/font renames the family and a literal 'Instrument Serif' would not
 * resolve. `face` keeps the design's name for the style sheet.
 */

/** Errors stay coral in every look. Not themeable. */
export const ERROR_COLOUR = '#FF6B6B'

export interface Ground {
  readonly name: string
  readonly bg: string
  readonly surface: string
  readonly text: string
  readonly text2: string
  readonly line: string
  readonly chip: string
  readonly panel: string
}

export interface Accent {
  readonly name: string
  readonly hex: string
  /** Button text on the accent fill, chosen per accent for AA. */
  readonly ink: string
}

export interface TypeSizes {
  readonly hero: number
  readonly brand: number
  readonly status: number
  readonly card: number
}

export interface TypeStyle {
  readonly name: string
  /** The design's name for the face, as the style sheet shows it. */
  readonly face: string
  /** The CSS font-family: the face's next/font variable, then the generic family. */
  readonly family: string
  readonly style: 'italic' | 'normal'
  readonly weight: number
  readonly letterSpacing: string
  /** Display sizes in px. */
  readonly sizes: TypeSizes
}

export interface Shell {
  readonly name: string
  readonly hex: string
  /** CSS filter applied to the mascot video and PNG. */
  readonly filter: string
}

export interface Light {
  readonly name: string
  readonly hex: string
}

export const GROUNDS = {
  graphite: { name: 'Graphite', bg: '#131110', surface: '#1D1917', text: '#F4EEE8', text2: '#B7ADA4', line: 'rgba(244,238,232,.12)', chip: 'rgba(244,238,232,.06)', panel: 'rgba(244,238,232,.025)' },
  obsidian: { name: 'Obsidian', bg: '#0B0B0C', surface: '#151517', text: '#EDEDED', text2: '#A1A1A6', line: 'rgba(255,255,255,.1)', chip: 'rgba(255,255,255,.06)', panel: 'rgba(255,255,255,.02)' },
  indigo: { name: 'Night', bg: '#120F2A', surface: '#1B1740', text: '#F3F0FF', text2: '#B9B3DA', line: 'rgba(180,170,255,.16)', chip: 'rgba(180,170,255,.08)', panel: 'rgba(180,170,255,.03)' },
  paper: { name: 'Paper', bg: '#F4EEE8', surface: '#FFFFFF', text: '#1F1A17', text2: '#6B635C', line: 'rgba(31,26,23,.14)', chip: 'rgba(31,26,23,.06)', panel: 'rgba(255,255,255,.5)' },
} as const satisfies Record<string, Ground>

export const ACCENTS = {
  molten: { name: 'Molten', hex: '#FF6A1F', ink: '#1A0A02' },
  cream: { name: 'Cream', hex: '#EDE6DE', ink: '#131110' },
  mint: { name: 'Mint', hex: '#4BF0B7', ink: '#04140E' },
  gold: { name: 'Gold', hex: '#F2C14E', ink: '#1E1600' },
  periwinkle: { name: 'Periwinkle', hex: '#8C9BFF', ink: '#0A0D2A' },
  rose: { name: 'Rose', hex: '#FF5C8A', ink: '#2A0512' },
} as const satisfies Record<string, Accent>

export const TYPE_STYLES = {
  serif: { name: 'Editorial', face: 'Instrument Serif', family: 'var(--pl-font-instrument-serif),serif', style: 'italic', weight: 400, letterSpacing: '0', sizes: { hero: 72, brand: 28, status: 40, card: 22 } },
  clean: { name: 'Clean', face: 'Geist', family: 'var(--font-geist-sans),sans-serif', style: 'normal', weight: 600, letterSpacing: '-.03em', sizes: { hero: 58, brand: 18, status: 30, card: 17 } },
  bric: { name: 'Characterful', face: 'Bricolage Grotesque', family: 'var(--pl-font-bricolage),sans-serif', style: 'normal', weight: 700, letterSpacing: '-.02em', sizes: { hero: 58, brand: 18, status: 30, card: 17 } },
  round: { name: 'Playful', face: 'Fredoka', family: 'var(--pl-font-fredoka),sans-serif', style: 'normal', weight: 600, letterSpacing: '0', sizes: { hero: 58, brand: 20, status: 30, card: 17 } },
} as const satisfies Record<string, TypeStyle>

export const SHELLS = {
  bronze: { name: 'Bronze', hex: '#C8692A', filter: 'none' },
  gold: { name: 'Gold', hex: '#D9A441', filter: 'hue-rotate(18deg) saturate(1.2) brightness(1.1)' },
  silver: { name: 'Silver', hex: '#B9BDC4', filter: 'grayscale(1) brightness(1.25)' },
  cobalt: { name: 'Cobalt', hex: '#3D6FD6', filter: 'hue-rotate(195deg) saturate(1.1)' },
} as const satisfies Record<string, Shell>

export const LIGHTS = {
  mint: { name: 'Mint', hex: '#4BF0B7' },
  amber: { name: 'Amber', hex: '#FFC24B' },
  ice: { name: 'Ice', hex: '#9FD8FF' },
  rose: { name: 'Rose', hex: '#FF8FB1' },
} as const satisfies Record<string, Light>

export type GroundKey = keyof typeof GROUNDS
export type AccentKey = keyof typeof ACCENTS
export type TypeStyleKey = keyof typeof TYPE_STYLES
export type ShellKey = keyof typeof SHELLS
export type LightKey = keyof typeof LIGHTS

/** Own keys only: 'toString' or '__proto__' are never catalogue entries. */
export function isKeyOf<T extends object>(section: T, key: unknown): key is keyof T & string {
  return typeof key === 'string' && Object.prototype.hasOwnProperty.call(section, key)
}

function keysOf<T extends object>(section: T): readonly (keyof T & string)[] {
  return Object.keys(section).filter((key): key is keyof T & string => isKeyOf(section, key))
}

export const GROUND_KEYS = keysOf(GROUNDS)
export const ACCENT_KEYS = keysOf(ACCENTS)
export const TYPE_STYLE_KEYS = keysOf(TYPE_STYLES)
export const SHELL_KEYS = keysOf(SHELLS)
export const LIGHT_KEYS = keysOf(LIGHTS)
