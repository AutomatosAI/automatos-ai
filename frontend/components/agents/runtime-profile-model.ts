/**
 * PRD-274 — the pure half of an agent's runtime profile: what `configuration.runtime_profile`
 * holds and what the agent's settings show for it. Nothing here fetches or renders.
 *
 * `full` is the default, and an absent value reads as full. A `lean` agent answers inside a
 * product's app through its server key. The backend refuses any other value, and it never runs a
 * system agent lean, whatever its configuration says, so the settings show a system agent as Full
 * and do not let it change.
 */

export type RuntimeProfile = 'full' | 'lean'

export const RUNTIME_PROFILE_FULL: RuntimeProfile = 'full'
export const RUNTIME_PROFILE_LEAN: RuntimeProfile = 'lean'

export interface RuntimeProfileOption {
  id: RuntimeProfile
  label: string
}

/** The choices, in the order the settings list them. */
export const RUNTIME_PROFILE_OPTIONS: readonly RuntimeProfileOption[] = [
  { id: RUNTIME_PROFILE_FULL, label: 'Full' },
  { id: RUNTIME_PROFILE_LEAN, label: 'Lean' },
]

/** "Runtime" already names the API model / CLI session choice on the same tab. */
export const RUNTIME_PROFILE_LABEL = 'Runtime profile'

export const RUNTIME_PROFILE_HELP =
  "Lean agents answer inside a product's app through its server key: only the tools you grant, no Auto personality."

export const SYSTEM_AGENT_PROFILE_REASON = 'A system agent always runs Full.'

export const PROFILE_READ_FAILED_REASON = "Couldn't load this agent's runtime profile, so it can't be changed right now."

export function isRuntimeProfile(value: unknown): value is RuntimeProfile {
  return RUNTIME_PROFILE_OPTIONS.some((option) => option.id === value)
}

export interface SavedRuntimeProfile {
  profile: RuntimeProfile
  /** the backend never runs a system agent lean, so its choice is fixed at Full */
  systemAgent: boolean
}

/**
 * What the saved agent (`GET /api/agents/{id}`) says: its stored profile, Full when absent or
 * unreadable, and Full for a system agent whatever its configuration holds. Pure.
 */
export function savedRuntimeProfile(detail: unknown): SavedRuntimeProfile {
  const src = (detail ?? {}) as Record<string, unknown>
  const systemAgent = src.is_system_agent === true
  const configuration = (src.configuration ?? {}) as Record<string, unknown>
  const lean = !systemAgent && configuration.runtime_profile === RUNTIME_PROFILE_LEAN
  return { profile: lean ? RUNTIME_PROFILE_LEAN : RUNTIME_PROFILE_FULL, systemAgent }
}
