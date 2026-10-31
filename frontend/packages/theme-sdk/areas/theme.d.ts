/**
 * PRD-261 A-10b — SDK area `theme` (frontend/lib/theme-sdk/theme.ts): the SDK's version and defineTheme, and (SDK
 * 1.3.0) DISPLAY_STATE_LABELS, each row state's glyph and word.
 */
import type { SlotId } from '../internal/host'
import type { DisplayState } from './proactive'

/**
 * What register(host) receives: the two versions and the three ways to hand the host a component.
 */
export interface ThemeRegistrationHost {
  readonly sdkVersion: string
  readonly hostVersion: string
  registerSlot(slot: SlotId, component: unknown): void
  /** Only the routes of the host's theme page list (the Q3 pages). */
  registerPage(route: string, component: unknown): void
  /** Served at /x/<path>. */
  registerRoute(path: string, component: unknown): void
}

/** A theme's default export, as defineTheme returns it. */
export interface ThemeDefinition {
  register(host: ThemeRegistrationHost): void
  unregister?(): void
}

/** The colour roles of the row and readiness states. Colour is never the only signal: the glyph and the word are. */
export type StatusRole = 'needs_you' | 'working' | 'ready' | 'done' | 'failed' | 'needs_setup' | 'not_checked'

/** The Studio token (a CSS custom property holding an HSL triplet) that paints a role, on a light ground and a dark one. */
export interface StatusToken {
  readonly light: string
  readonly dark: string
}

/** A state's glyph (● in motion, ○ waiting, ◆ a person must act, ✓ finished or fine, ✕ failed), its word and its role. */
export interface DisplayStateLabel {
  readonly glyph: string
  /** The word in Studio, Classic and Play. */
  readonly word: string
  /** The word in capitals, as HUD writes it. */
  readonly hudWord: string
  readonly role: StatusRole
  readonly token: StatusToken
}

/** The version of the SDK the host serves; this package's version. */
export declare const SDK_VERSION: string

/** A theme's definition, checked and frozen: it must carry a register(host) function, or this throws a TypeError. */
export declare function defineTheme(definition: ThemeDefinition): Readonly<ThemeDefinition>

/** Each row state's glyph, word and role, as every host surface draws them (a row's `display_state`). */
export declare const DISPLAY_STATE_LABELS: Readonly<Record<DisplayState, DisplayStateLabel>>
