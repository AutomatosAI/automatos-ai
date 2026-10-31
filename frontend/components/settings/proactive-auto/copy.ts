/**
 * PRD-268 §6.9 — the words of Settings → Orchestrator's "Proactive Auto" section, in one place, as the readiness
 * section beside it keeps its own (components/proactive/readiness-list.tsx). A refusal's sentence is the server's,
 * in the person's language (`refusedLine`), never one of these.
 */
import type { CapBounds, CapName, EffectiveMode } from '@/lib/proactive/settings'

export const PROACTIVE_TITLE = 'Proactive Auto'
export const PROACTIVE_INTRO =
  "When it's on, Auto checks in on the workspace's work in the background and suggests what needs doing. " +
  'A suggestion never runs by itself: someone presses Start.'

export const WORKSPACE_SWITCH_LABEL = 'Proactive Auto for this workspace'
export const WORKSPACE_ON = 'On for this workspace.'
export const WORKSPACE_OFF = 'Off for this workspace.'
export const ASK_AN_OWNER_OR_ADMIN = 'A workspace owner or admin turns Proactive Auto on or off and sets its limits.'

/** What Auto is doing here, from the workspace's `effective_mode` (both settings reads answer it). */
export const MODE_LINES = Object.freeze({
  live: 'Auto is checking in and suggesting.',
  shadow: 'Auto is checking in quietly: suggestions are recorded, but nothing is sent yet.',
  offButSwitchedOn: "Switched on here, but Proactive Auto isn't running on this server yet.",
})

/** The one line for the mode, or none: off with the switch off says nothing extra, and an unknown mode draws nothing. */
export function modeLine(mode: EffectiveMode | null, switchedOn: boolean): string | null {
  if (mode === 'live') return MODE_LINES.live
  if (mode === 'shadow') return MODE_LINES.shadow
  return mode === 'off' && switchedOn ? MODE_LINES.offButSwitchedOn : null
}

export const LOADING = 'Loading…'

export const CAPS_TITLE = 'Limits'
export const CAPS_INTRO = 'Each limit can be made stricter than its default, never looser.'
export const SAVE_CAPS = 'Save limits'
export const SAVING = 'Saving…'
/** Caps stored wrong refuse every check-in until saved again (409 `caps_stored_wrong`): this saves the defaults. */
export const RESET_CAPS = 'Put the limits back to their defaults'

export const PAUSE_LABEL = 'Pause suggestions'
export const PAUSE_LINE =
  'While paused, you get no suggestions and no suggestion notifications. ' +
  "Your daily brief still comes, unless you've switched the digest off."

interface CapCopy {
  readonly label: string
  readonly note?: string
}

export const CAP_COPY: Readonly<Record<CapName, CapCopy>> = {
  max_model_runs_per_day: {
    label: 'Check-ins that reach the model, per day',
    note: "A check-in that finds nothing new doesn't count.",
  },
  max_suggestions_per_run: { label: 'Suggestions per check-in' },
  max_new_per_person_per_day: { label: 'New suggestions per person, per day' },
  max_open_per_person: {
    label: 'Open suggestions per person',
    note: 'Past this, their oldest open suggestions expire.',
  },
  max_open_per_agent: { label: 'Open suggestions per agent' },
  min_interval_minutes: { label: "Shortest gap between an agent's check-ins, in minutes" },
  max_interval_minutes: {
    label: "Longest gap between an agent's check-ins, in minutes",
    note: "A failing agent's check-ins slow down to this, at most.",
  },
}

/** "From 0 to 48." */
export function rangeLine({ min, max }: CapBounds): string {
  return `From ${min} to ${max}.`
}

/** "Enter a whole number from 0 to 48." */
export function outOfRangeLine({ min, max }: CapBounds): string {
  return `Enter a whole number from ${min} to ${max}.`
}
