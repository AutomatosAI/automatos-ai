/**
 * PRD-268 §6.9, FR-13 — Proactive Auto's switches and caps, as Settings → Orchestrator reads and edits them.
 *
 * - `GET`/`PUT /api/proactive/settings` (owners and admins, `workspace:manage`): `{enabled, caps, platform_caps}`
 *   (services/proactive/settings.py `WorkspaceSettings.to_dict`). `PUT` takes `{enabled?, caps?}` (api/proactive.py
 *   `SettingsRequest`, `CapsChange`): each cap a whole number, or null for its default. A workspace can only tighten
 *   the platform's caps (`platform_caps`): every cap is tightened by lowering it, except `min_interval_minutes`, which
 *   is tightened by raising it (services/proactive/limits.py `RAISED_TO_TIGHTEN`); and the shortest interval may not
 *   pass the longest (modules/proactive/caps.py). The server answers 422 `caps_looser` or `caps_wrong` otherwise.
 * - `GET`/`PUT /api/proactive/me` (any member): `{paused, quiet_start, quiet_end, timezone}`, the person's own.
 * - Anyone else reads the workspace switch from the workspace they already load (GET /api/workspaces/current
 *   answers its `settings`), where `settings.proactive.enabled` is on only when it is exactly true
 *   (services/proactive/switches.py `enabled_in`).
 *
 * Every answer is checked here before anything draws it; nothing here mutates what it reads.
 */
import type { Workspace } from '@/components/workspace-provider'
import { recordOf, textOf } from './wire'

/** The caps a workspace may set, in the order Settings shows them (services/proactive/limits.py `EDITABLE_CAPS`). */
export const CAP_NAMES = [
  'max_model_runs_per_day',
  'max_suggestions_per_run',
  'max_new_per_person_per_day',
  'max_open_per_person',
  'max_open_per_agent',
  'min_interval_minutes',
  'max_interval_minutes',
] as const
export type CapName = (typeof CAP_NAMES)[number]
export type ProactiveCaps = Readonly<Record<CapName, number>>

/** The one cap a workspace tightens by raising it. */
const RAISED_TO_TIGHTEN: CapName = 'min_interval_minutes'
/** A count cap may be 0 (it allows nothing); an interval is at least a minute (`CapsChange`). */
const LOWEST_COUNT = 0
const LOWEST_INTERVAL = 1

export interface ProactiveSettings {
  readonly enabled: boolean
  readonly caps: ProactiveCaps
  readonly platform_caps: ProactiveCaps
}

export interface ProactiveMe {
  readonly paused: boolean
  readonly quiet_start: string | null
  readonly quiet_end: string | null
  readonly timezone: string | null
}

/** `PUT /api/proactive/settings`: what is left out stays; a cap sent as null is back at its default. */
export interface ProactiveSettingsChange {
  readonly enabled?: boolean
  readonly caps?: Readonly<Partial<Record<CapName, number | null>>>
}

/** The refusal of a settings read whose stored caps don't fit (api/proactive.py `CAPS_STORED_WRONG`). */
export const CAPS_STORED_WRONG = 'caps_stored_wrong'

/** Every cap back at its default: what puts right caps stored wrong (409 `caps_stored_wrong`). */
export const DEFAULT_CAPS_CHANGE: ProactiveSettingsChange = Object.freeze({
  caps: Object.freeze(Object.fromEntries(CAP_NAMES.map((name) => [name, null])) as Record<CapName, null>),
})

/** `PUT /api/proactive/me`: the pause only, from Settings. */
export interface ProactiveMeChange {
  readonly paused: boolean
}

export interface CapBounds {
  readonly min: number
  readonly max: number
}

/** The cap values as typed, by name; a name left out shows its saved value. */
export type CapDraft = Readonly<Partial<Record<CapName, string>>>

const MANAGE_ROLES: ReadonlySet<Workspace['role']> = new Set<Workspace['role']>(['owner', 'admin'])

/** Who may change the switch and the caps: the `workspace:manage` roles the settings route requires. */
export function canManageProactive(role: Workspace['role'] | undefined): boolean {
  return !!role && MANAGE_ROLES.has(role)
}

function capsOf(value: unknown): ProactiveCaps | null {
  const fields = recordOf(value)
  if (fields === null) return null
  const entries = CAP_NAMES.map((name) => [name, fields[name]] as const)
  if (!entries.every(([, cap]) => Number.isInteger(cap))) return null
  return Object.freeze(Object.fromEntries(entries) as Record<CapName, number>)
}

/** GET and PUT /api/proactive/settings' answer; anything else is an error, so nothing half-read is drawn. */
export function proactiveSettingsOf(body: unknown): ProactiveSettings {
  const fields = recordOf(body)
  const enabled = fields?.enabled
  const caps = capsOf(fields?.caps)
  const platform = capsOf(fields?.platform_caps)
  if (typeof enabled !== 'boolean' || caps === null || platform === null) {
    throw new Error('The Proactive Auto settings came back in a shape Settings cannot read.')
  }
  return { enabled, caps, platform_caps: platform }
}

/** GET and PUT /api/proactive/me's answer. */
export function proactiveMeOf(body: unknown): ProactiveMe {
  const fields = recordOf(body)
  const paused = fields?.paused
  if (typeof paused !== 'boolean') {
    throw new Error('Your Proactive Auto settings came back in a shape Settings cannot read.')
  }
  return {
    paused,
    quiet_start: textOf(fields?.quiet_start) ?? null,
    quiet_end: textOf(fields?.quiet_end) ?? null,
    timezone: textOf(fields?.timezone) ?? null,
  }
}

/** The workspace switch as the workspace's own `settings` hold it: on only when exactly true. */
export function workspaceSwitchOf(settings: unknown): boolean {
  return recordOf(recordOf(settings)?.proactive)?.enabled === true
}

/** The settings with a change applied (a cap sent as null at the platform's): a new object, the old one untouched. */
export function withChange(settings: ProactiveSettings, change: ProactiveSettingsChange): ProactiveSettings {
  const asked = change.caps ?? {}
  const caps = Object.fromEntries(
    CAP_NAMES.map((name) => {
      const value = asked[name]
      return [name, value === undefined ? settings.caps[name] : value ?? settings.platform_caps[name]]
    }),
  ) as Record<CapName, number>
  return { ...settings, enabled: change.enabled ?? settings.enabled, caps: Object.freeze(caps) }
}

/** A typed value as a whole number, or null. */
export function wholeNumber(text: string): number | null {
  const trimmed = text.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const value = Number(trimmed)
  return Number.isSafeInteger(value) ? value : null
}

/** Each cap's value as shown: what was typed, else the saved value. */
export function capValues(caps: ProactiveCaps, draft: CapDraft): Readonly<Record<CapName, string>> {
  return Object.fromEntries(CAP_NAMES.map((name) => [name, draft[name] ?? String(caps[name])])) as Record<CapName, string>
}

function typedOrSaved(name: CapName, values: Readonly<Record<CapName, string>>, caps: ProactiveCaps): number {
  return wholeNumber(values[name]) ?? caps[name]
}

/** What the server accepts for one cap, given the other values typed: no looser than the platform's. */
export function capBounds(name: CapName, values: Readonly<Record<CapName, string>>, settings: ProactiveSettings): CapBounds {
  const platform = settings.platform_caps
  if (name === RAISED_TO_TIGHTEN) {
    return { min: Math.max(LOWEST_INTERVAL, platform[name]), max: typedOrSaved('max_interval_minutes', values, settings.caps) }
  }
  if (name === 'max_interval_minutes') {
    return { min: Math.max(LOWEST_INTERVAL, typedOrSaved(RAISED_TO_TIGHTEN, values, settings.caps)), max: platform[name] }
  }
  return { min: LOWEST_COUNT, max: platform[name] }
}

/** True when the typed value is a whole number within its bounds. */
export function capFits(name: CapName, values: Readonly<Record<CapName, string>>, settings: ProactiveSettings): boolean {
  const value = wholeNumber(values[name])
  const { min, max } = capBounds(name, values, settings)
  return value !== null && value >= min && value <= max
}

/** The caps whose typed value differs from the saved one, as numbers: what Save sends. */
export function changedCaps(values: Readonly<Record<CapName, string>>, caps: ProactiveCaps): Partial<Record<CapName, number>> {
  return Object.fromEntries(
    CAP_NAMES.flatMap((name) => {
      const value = wholeNumber(values[name])
      return value !== null && value !== caps[name] ? [[name, value]] : []
    }),
  )
}
