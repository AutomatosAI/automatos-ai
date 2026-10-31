/**
 * PRD-268 §6.11.1, §6.12.2 (US-C21b) — what a client draws for a state: a glyph, a word and a colour.
 *
 * The server computes the state (`display_state` on a row, `state` on a readiness item); a client
 * only maps it. Each glyph means one thing everywhere: ● in motion, ○ waiting or not known yet,
 * ◆ a person must act, ✓ finished or fine, ✕ failed. §6.11.1 corrects the handoff's: done is ✓
 * (never —, which means stopped) and readiness ready is ✓ (never ●, which means in motion).
 *
 * Colour is a role, never the only signal: the glyph and the word always carry the meaning. Each
 * role names the Studio token that paints it on a light ground and on a dark one (each theme's token
 * file decides the colour). A status never goes through `--accent`, which is the error colour under
 * every Play look. HUD paints every status in its own accent (the kit's rule), so it takes only the
 * capitalised word from here.
 *
 * Unknown draws nothing: a state that isn't one of these gives null, never a placeholder.
 */
import type { DisplayState, ReadinessState } from './types'

/** The colour roles of §6.12.2. */
export type StatusRole = 'needs_you' | 'working' | 'ready' | 'done' | 'failed' | 'needs_setup' | 'not_checked'

/** A Studio token (a CSS custom property holding an HSL triplet), by the ground's tone. */
export interface StatusToken {
  readonly light: string
  readonly dark: string
}

export interface StateLabel {
  readonly glyph: string
  /** The word in Studio, Classic and Play. A Brand Kit may relabel it; the glyph and role stay. */
  readonly word: string
  /** HUD's word, in capitals. */
  readonly hudWord: string
  readonly role: StatusRole
  readonly token: StatusToken
}

/** The glyph before a running step's line (`▸ next_step`). */
export const NEXT_STEP_GLYPH = '▸'

const same = (token: string): StatusToken => Object.freeze({ light: token, dark: token })

/**
 * §6.12.2: `warning` is 1.58:1 on Studio light, so needs-you is drawn in `warn-ink` on light grounds
 * and in `warning` on dark ones. `running` is the token US-C21a made real.
 */
export const STATUS_ROLE_TOKENS: Readonly<Record<StatusRole, StatusToken>> = Object.freeze({
  needs_you: Object.freeze({ light: '--warn-ink', dark: '--warning' }),
  working: same('--running'),
  ready: same('--success'),
  done: same('--success'),
  failed: same('--destructive'),
  needs_setup: same('--info'),
  not_checked: same('--muted-foreground'),
})

function label(glyph: string, word: string, hudWord: string, role: StatusRole): StateLabel {
  return Object.freeze({ glyph, word, hudWord, role, token: STATUS_ROLE_TOKENS[role] })
}

/** §6.11.1's table: each row state's glyph, its word, its HUD word and its colour role. */
export const DISPLAY_STATE_LABELS: Readonly<Record<DisplayState, StateLabel>> = Object.freeze({
  needs_you: label('◆', 'Needs you', 'NEEDS YOU', 'needs_you'),
  working: label('●', 'Working', 'WORKING', 'working'),
  ready: label('✓', 'Ready to look at', 'READY', 'ready'),
  done: label('✓', 'Done', 'DONE', 'done'),
  failed: label('✕', "Couldn't finish", 'FAILED', 'failed'),
})

/** §6.11.3: a readiness item's glyph and word. "Not checked" never stands in for a known failure. */
export const READINESS_STATE_LABELS: Readonly<Record<ReadinessState, StateLabel>> = Object.freeze({
  ready: label('✓', 'Ready', 'READY', 'ready'),
  not_checked: label('○', 'Not checked', 'NOT CHECKED', 'not_checked'),
  needs_setup: label('◆', 'Needs setup', 'NEEDS SETUP', 'needs_setup'),
})

function own<T>(table: Readonly<Record<string, T>>, key: unknown): T | null {
  return typeof key === 'string' && Object.prototype.hasOwnProperty.call(table, key) ? table[key] : null
}

/** The label of a row's `display_state`, or null for anything else (unknown draws nothing). */
export function displayStateLabel(state: unknown): StateLabel | null {
  return own<StateLabel>(DISPLAY_STATE_LABELS, state)
}

/** The label of a readiness item's `state`, or null for anything else. */
export function readinessStateLabel(state: unknown): StateLabel | null {
  return own<StateLabel>(READINESS_STATE_LABELS, state)
}

/** True for one of §6.11.1's states. */
export function isDisplayState(state: unknown): state is DisplayState {
  return displayStateLabel(state) !== null
}
