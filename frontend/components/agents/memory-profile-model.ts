/**
 * PRD-274 Wave 4 — the pure half of an agent's memory profile choice: the profiles a workspace
 * can pick (`GET /api/v1/memory/profiles`), the agent's own (`configuration.memory_profile`, a
 * profile's slug) and what the picker shows. Nothing here fetches or renders.
 *
 * An agent with no `memory_profile` follows its workspace's own setting, else `general`.
 * `general` is reserved: today's distillation, offered as "General" when the list does not hold
 * it. While the install's `MEMORY_PROFILES_ENABLED` is off the route answers `enabled: false` and
 * every agent keeps facts as `general`, whatever it chose.
 */

export const MEMORY_PROFILES_PATH = '/api/v1/memory/profiles'

export const GENERAL_SLUG = 'general'
const GENERAL_NAME = 'General'

/** The picker's value for "no profile of its own"; `_` is never in a slug, so it names none. */
export const WORKSPACE_DEFAULT_VALUE = '_workspace_default'
export const WORKSPACE_DEFAULT_NAME = 'Workspace default'

/** "Memory Limit (MB)" already names a memory setting of the agent, on the Resources tab. */
export const MEMORY_PROFILE_LABEL = 'Memory profile'

export const WORKSPACE_DEFAULT_HELP =
  "This agent uses the workspace's memory profile, or General when the workspace has none."

export const PROFILES_OFF_NOTE =
  'Memory profiles are off for this install: the choice is saved, and it applies once they are switched on.'

export const PROFILES_LOAD_FAILED_TITLE = "Memory profiles couldn't be loaded"
export const PROFILES_LOAD_FAILED_TEXT = 'The rest of this tab still works. Try again, or reopen the agent.'
export const AGENT_LOAD_FAILED_TEXT = "Couldn't load this agent's memory profile, so it can't be changed right now."

export interface MemoryProfileOption {
  slug: string
  name: string
  /** one line on what the profile keeps; '' when it has none */
  description: string
}

export interface MemoryProfileList {
  profiles: MemoryProfileOption[]
  /** false while the install's memory profiles are off */
  enabled: boolean
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function toOption(row: unknown): MemoryProfileOption[] {
  const src = (row ?? {}) as Record<string, unknown>
  const slug = text(src.slug)
  return slug ? [{ slug, name: text(src.name) || slug, description: text(src.description) }] : []
}

/**
 * The route's answer, read defensively: rows without a slug dropped, a nameless row shown by its
 * slug, and General first when the list does not hold it. Pure.
 */
export function normalizeMemoryProfiles(body: unknown): MemoryProfileList {
  const src = (body ?? {}) as Record<string, unknown>
  const listed = (Array.isArray(src.profiles) ? src.profiles : []).flatMap(toOption)
  const hasGeneral = listed.some((option) => option.slug === GENERAL_SLUG)
  const general: MemoryProfileOption = { slug: GENERAL_SLUG, name: GENERAL_NAME, description: '' }
  return { profiles: hasGeneral ? listed : [general, ...listed], enabled: src.enabled === true }
}

/** The agent's own profile from its saved detail, or null when it follows the workspace. Pure. */
export function savedMemoryProfile(detail: unknown): string | null {
  const src = (detail ?? {}) as Record<string, unknown>
  const configuration = (src.configuration ?? {}) as Record<string, unknown>
  return text(configuration.memory_profile) || null
}

/**
 * The picker's options: "Workspace default" first while the agent has no profile of its own, then
 * the listed profiles, and the saved slug kept by its own name when the list no longer holds it,
 * so the picker never shows blank. Pure.
 */
export function memoryProfileOptions(list: MemoryProfileList, saved: string | null): MemoryProfileOption[] {
  const fallback: MemoryProfileOption[] =
    saved && !list.profiles.some((option) => option.slug === saved) ? [{ slug: saved, name: saved, description: '' }] : []
  const workspaceDefault: MemoryProfileOption[] =
    saved === null ? [{ slug: WORKSPACE_DEFAULT_VALUE, name: WORKSPACE_DEFAULT_NAME, description: WORKSPACE_DEFAULT_HELP }] : []
  return [...workspaceDefault, ...list.profiles, ...fallback]
}
