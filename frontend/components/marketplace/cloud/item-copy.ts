/**
 * PRD-270 W2 (§5.6, §8-Q3a, K19, K30) — the words the Marketplace shows an Automatos Cloud agent,
 * Playbook, package or plugin with. "From Automatos Cloud" is W1's (listing-format.ts `CLOUD_BADGE`), so
 * a Cloud item reads the same as a Cloud theme. A Cloud item is Reviewed on Automatos Cloud, never
 * Official: nothing here says Official.
 */
import { formatRelative } from '@/lib/format-relative'

export { ADD_TO_WORKSPACE, ADDING, CLOUD_BADGE } from '@/components/marketplace/themes/listing-format'

export const REVIEWED_BADGE = 'Reviewed on Automatos Cloud'

/** The disclosure (disclosure-dialog.tsx, US-270-14c): what a Cloud item will use, read before it is added. */
export const CLOSE = 'Close'
export const CLOUD_ITEM_TITLE = 'Automatos Cloud item'
export const READING_DISCLOSURE = 'Reading what this item will use…'
export const DISCLOSURE_UNREADABLE = "Couldn't read what this item will use, so it can't be added from here. Close and try again."
export const NO_CUSTOM_PROMPT = 'No custom prompt: it uses the default for its type'
export const NO_MODEL = "No model named: it uses this install's default model"
export const NEEDS_CONNECTION = 'needs a connection'
export const NOT_ON_THIS_INSTALL = 'not on this install'
export const ON_THIS_INSTALL = 'on this install'
export const NEEDS_A_HELPER = 'needs a helper'
export const NO_AGENT_STEP = 'No agent runs this step'
export const UNNAMED = 'Unnamed'
export const OPTIONAL = 'optional'
export const NONE = 'None'
export const UPDATES_IN_PLACE = 'Updates apply to every workspace that enabled it'

/** "by Cloud Publisher · Version 1.2.0 · Synced 3d ago": what is known of the item's source. */
export function publishedLine(publisher: string | null, version: string, synced: string | null): string {
  return [publisher && `by ${publisher}`, `Version ${version}`, synced].filter(Boolean).join(' · ')
}

/** "Synced 3d ago" (a date beyond a month), or null when the Sync recorded no time. */
export function syncedOn(syncedAt: string | null, now?: number): string | null {
  return syncedAt ? `Synced ${formatRelative(syncedAt, now)}` : null
}

/** One line under the header: what the item is, and that it is read before it is added. */
export const DISCLOSURE_LEDE: Readonly<Record<'agent' | 'playbook' | 'package' | 'plugin', string>> = {
  agent: 'An agent from Automatos Cloud. Read what it will use before you add it.',
  playbook: 'A Playbook from Automatos Cloud. Read what it will use before you add it.',
  package: 'A package from Automatos Cloud. Read what it will use before you add it.',
  plugin: 'A plugin from Automatos Cloud. Read what it will use before you enable it.',
}

/** What didn't resolve on this install, by the kind the Sync recorded (resolve.py's SKILL … HELPER). */
const NEEDS_KIND: Readonly<Record<string, string>> = {
  skill: 'the skill',
  persona: 'the persona',
  plugin: 'the plugin',
  model: 'the model',
  helper: 'the helper agent',
}

/** "Needs the skill Ledger Lookup"; a kind this page doesn't know names only the item. */
export function needsLine(kind: string, name: string): string {
  const noun = Object.prototype.hasOwnProperty.call(NEEDS_KIND, kind) ? NEEDS_KIND[kind] : null
  return noun ? `Needs ${noun} ${name}` : `Needs ${name}`
}

/** "Prompt from the persona Bookkeeper", or the persona's own prompt when it has no name here. */
export function personaPromptLine(personaName: string | null): string {
  return personaName ? `Prompt from the persona ${personaName}` : 'Prompt from its persona'
}

export function stepTitle(order: number | null, index: number): string {
  return `Step ${order ?? index + 1}`
}

export function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`
}

/** A plugin's reports (disclosure-plugin.tsx). */
export const NO_REPORT = 'No report kept'

/** "Verdict: safe", or null when there is no value. */
export function statusLine(label: string, value: string | null): string | null {
  return value ? `${label}: ${value}` : null
}

export function riskLine(score: number | null): string | null {
  return typeof score === 'number' ? `Risk score: ${score} of 100` : null
}

/** "high: Reads a file outside its folder (scripts/run.sh:12)". */
export function findingLine(
  severity: string | null, description: string | null, file: string | null, line: number | null,
): string {
  const where = file ? (typeof line === 'number' ? `${file}:${line}` : file) : null
  const head = severity ? `${severity}: ` : ''
  return `${head}${description ?? 'A finding'}${where ? ` (${where})` : ''}`
}

/** "3 skills · 1 command · 0 agents · 2 hooks". */
export function contentsLine(contents: { skills: number; commands: number; agents: number; hooks: number }): string {
  return [
    plural(contents.skills, 'skill', 'skills'),
    plural(contents.commands, 'command', 'commands'),
    plural(contents.agents, 'agent', 'agents'),
    plural(contents.hooks, 'hook', 'hooks'),
  ].join(' · ')
}
