import type { AccentKey, GroundKey, LightKey, ShellKey, TypeStyleKey } from './catalogue'

/**
 * PRD-260 — a look is data, not code: catalogue keys plus a glow number. Never
 * raw colours, CSS or URLs.
 */

export const LOOK_SCHEMA = 'automatos-interface-look/v1'
export const LOOK_INTERFACE = 'play'

export const GLOW_MIN = 0
export const GLOW_MAX = 40
/** For `name` and `sub`, counted in characters (code points). */
export const LOOK_TEXT_MAX = 40
/** Letters, digits and dashes, so an id is safe as a storage key or a React key. */
export const LOOK_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/

export interface Look {
  readonly schema: typeof LOOK_SCHEMA
  readonly id: string
  readonly name: string
  readonly sub: string
  readonly interface: typeof LOOK_INTERFACE
  readonly ground: GroundKey
  readonly accent: AccentKey
  readonly font: TypeStyleKey
  readonly shell: ShellKey
  readonly light: LightKey
  /** Whole number, GLOW_MIN to GLOW_MAX: the page glow's alpha (glow / 100) of the accent. */
  readonly glow: number
}

export const LOOK_FIELDS = [
  'schema',
  'id',
  'name',
  'sub',
  'interface',
  'ground',
  'accent',
  'font',
  'shell',
  'light',
  'glow',
] as const satisfies readonly (keyof Look)[]

export type ValidationResult =
  | { readonly ok: true; readonly look: Look }
  | { readonly ok: false; readonly errors: readonly string[] }
