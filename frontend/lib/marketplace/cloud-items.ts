/**
 * PRD-270 W2 US-270-14b (§5.6, K18, K19) — the local reads the web shows Automatos Cloud items with
 * (api/marketplace_cloud_items.py, US-270-14a), all under `/api/marketplace/cloud-sync`:
 *
 * - `GET /items`: the Cloud agents, Playbooks, packages and plugins this install holds, each by the key
 *   its install route takes (an agent's and a Playbook's integer id, a package's slug, a plugin's uuid);
 * - `GET /items/{kind}/{key}/disclosure`: everything one Cloud item will use, per kind;
 * - `GET /updates`: the caller's workspace's copies with a newer Cloud version, and the Cloud plugins it
 *   enabled that have moved since.
 *
 * Local edition only: every route answers 404 on the hosted edition, so the callers ask only when
 * `isLocal`. The calls go through apiClient's generic `get`, so api-client.ts doesn't grow (K3). A failed
 * read throws the client's error, and an items answer that isn't the shape above throws too
 * (cloud-item-rows.ts `cloudItemsOf`, which the install gate reads with too, US-270-14d): the badges and
 * the gate never treat an answer they can't read as "no Cloud items". The
 * disclosure dialog reads a disclosure through `cloudDisclosureOf` (US-270-14c), so it never offers Add on
 * an answer it can't show whole, and the update notices read the updates through `cloudUpdatesOf`
 * (US-270-14e). Every text in an answer is the server's, rendered as text (K30).
 */
import { apiClient } from '@/lib/api-client'

import { CLOUD_ITEMS_BASE, CLOUD_ITEMS_ENDPOINT, cloudItemsOf, isJson } from './cloud-item-rows'
import type { CloudItemKey, CloudItemKind, CloudItems } from './cloud-item-rows'

export const CLOUD_UPDATES_ENDPOINT = `${CLOUD_ITEMS_BASE}/updates`
export const CLOUD_DISCLOSURE_UNREADABLE = "Couldn't read what this item will use."
export const CLOUD_UPDATES_UNREADABLE = "Couldn't read the updates from Automatos Cloud."

/** The prompt an agent runs with: its own ('custom'), else its persona's, else none. */
export interface CloudPrompt {
  source: 'custom' | 'persona' | null
  text: string | null
  personaName: string | null
}

export interface CloudUnresolved { kind: string; name: string }

interface DisclosureHeader {
  name: string
  /** Automatos Cloud's version. */
  version: string
  syncedAt: string | null
  /** Every Cloud item is Reviewed on Automatos Cloud, never Official (§8-Q3a). */
  trust: 'reviewed'
  publisher: string | null
  /** What didn't resolve on this install when the Sync wrote it. */
  unresolved: CloudUnresolved[]
}

export interface AgentDisclosure extends DisclosureHeader {
  kind: 'agent'
  prompt: CloudPrompt
  /** The model it names and whether this install has it, or null when it names none. */
  model: { modelId: string; provider: string | null; here: boolean } | null
  apps: Array<{ name: string; needsConnection: boolean }>
  skills: Array<{ name: string; here: boolean }>
  plugins: Array<{ slug: string; here: boolean }>
}

export interface PlaybookStep {
  order: number | null
  /** The helper agent's name: one found here, or the one it needs. */
  helper: string | null
  needsHelper: boolean
  prompt: string | null
}

export interface PlaybookDisclosure extends DisclosureHeader {
  kind: 'playbook'
  steps: PlaybookStep[]
  /** The agents its install adds, by name. */
  agents: string[]
  apps: string[]
}

export interface PackageDisclosure extends DisclosureHeader {
  kind: 'package'
  members: Array<{ type: string | null; name: string | null; here: boolean }>
  connects: Array<{ app: string | null; optional: boolean; note: string | null }>
  agents: Array<{ name: string; prompt: CloudPrompt }>
  playbooks: Array<{ name: string; steps: number }>
}

/** Automatos Cloud's review of a plugin, as the Sync kept it. */
export interface PluginCloudReport {
  verdict: string | null
  status: string | null
  riskScore: number | null
  summary: string | null
  findings: Array<{ severity: string | null; description: string | null }>
}

/** This install's static re-run over the plugin's files. */
export interface PluginLocalCheck {
  status: string | null
  findings: Array<{
    type: string | null; severity: string | null; file: string | null; line: number | null; description: string | null
  }>
}

export interface PluginDisclosure extends DisclosureHeader {
  kind: 'plugin'
  scan: PluginCloudReport | null
  localCheck: PluginLocalCheck | null
  contents: { skills: number; commands: number; agents: number; hooks: number }
  /** The global skills it made on this install, by name. */
  skills: string[]
  /** An update moves the row every workspace that enabled it runs (K15). */
  updatesInPlace: true
}

/** `GET /items/{kind}/{key}/disclosure` (disclosure.disclosure). */
export type CloudDisclosure = AgentDisclosure | PlaybookDisclosure | PackageDisclosure | PluginDisclosure

/** One update notice (updates.updates): a copy with a newer Cloud version, or a Cloud plugin that moved. */
export interface CloudUpdate {
  kind: 'agent' | 'playbook' | 'plugin'
  /** The Cloud row's key: an integer id, a plugin's uuid. */
  cloudId: number | string
  name: string
  /** The workspace's copies (none for a plugin: it is enabled by reference). */
  installed: Array<{ id: number; name: string; version: string | null }>
  available: { version: string; revision: string; syncedAt: string | null }
}

/** `GET /updates`, for the caller's workspace. */
export interface CloudUpdates { updates: CloudUpdate[] }

type Json = Record<string, unknown>

export async function getCloudItems(): Promise<CloudItems> {
  return cloudItemsOf(await apiClient.get<unknown>(CLOUD_ITEMS_ENDPOINT))
}

export function cloudDisclosurePath(kind: CloudItemKind, key: CloudItemKey): string {
  return `${CLOUD_ITEMS_ENDPOINT}/${encodeURIComponent(kind)}/${encodeURIComponent(String(key))}/disclosure`
}

/** Each kind's lists the disclosure dialog shows: lists of rows, and lists of names. */
const DISCLOSURE_LISTS: Readonly<Record<CloudItemKind, readonly [rows: readonly string[], names: readonly string[]]>> = {
  agent: [['apps', 'skills', 'plugins'], []],
  playbook: [['steps'], ['agents', 'apps']],
  package: [['members', 'connects', 'agents', 'playbooks'], []],
  plugin: [[], ['skills']],
}

function listsOf(answer: Json, fields: readonly string[], isEntry: (entry: unknown) => boolean): boolean {
  return fields.every((field) => {
    const list = answer[field]
    return Array.isArray(list) && list.every(isEntry)
  })
}

/** A plugin's report as the Sync kept it: none, or one whose findings are a list of rows. */
function reportOf(report: unknown): boolean {
  return report === null || (isJson(report) && Array.isArray(report.findings) && report.findings.every(isJson))
}

function kindParts(answer: Json, kind: CloudItemKind): boolean {
  if (kind === 'agent') return isJson(answer.prompt) && (answer.model === null || isJson(answer.model))
  // `agents` is a list of rows (checked first): each carries its prompt.
  if (kind === 'package') return (answer.agents as unknown[]).every((agent) => isJson(agent) && isJson(agent.prompt))
  if (kind === 'plugin') return isJson(answer.contents) && reportOf(answer.scan) && reportOf(answer.localCheck)
  return true
}

/**
 * A disclosure answer as the dialog reads it, for the `kind` it asked about; throws when it is another kind's
 * or lacks what the dialog shows (its name, version, what didn't resolve, its kind's lists), so the dialog
 * says it couldn't read it and offers no Add rather than show part of an item.
 */
export function cloudDisclosureOf(answer: unknown, kind: CloudItemKind): CloudDisclosure {
  if (!isJson(answer) || answer.kind !== kind) throw new Error(CLOUD_DISCLOSURE_UNREADABLE)
  const [rows, names] = DISCLOSURE_LISTS[kind]
  const readable = typeof answer.name === 'string' && typeof answer.version === 'string'
    && listsOf(answer, ['unresolved', ...rows], isJson)
    && listsOf(answer, names, (entry) => typeof entry === 'string')
    && kindParts(answer, kind)
  if (!readable) throw new Error(CLOUD_DISCLOSURE_UNREADABLE)
  return answer as unknown as CloudDisclosure
}

export function getCloudDisclosure(kind: CloudItemKind, key: CloudItemKey): Promise<CloudDisclosure> {
  return apiClient.get<CloudDisclosure>(cloudDisclosurePath(kind, key))
}

export function getCloudUpdates(): Promise<CloudUpdates> {
  return apiClient.get<CloudUpdates>(CLOUD_UPDATES_ENDPOINT)
}

/** Each notice's kind and the type of its Cloud row's key: an agent's or a Playbook's id, a plugin's uuid. */
const UPDATE_KEY_TYPES: Readonly<Record<CloudUpdate['kind'], 'number' | 'string'>> = {
  agent: 'number',
  playbook: 'number',
  plugin: 'string',
}

function isUpdate(entry: unknown): boolean {
  if (!isJson(entry) || typeof entry.kind !== 'string') return false
  if (!Object.prototype.hasOwnProperty.call(UPDATE_KEY_TYPES, entry.kind)) return false
  const available = entry.available
  return typeof entry.cloudId === UPDATE_KEY_TYPES[entry.kind as CloudUpdate['kind']]
    && typeof entry.name === 'string' && Array.isArray(entry.installed)
    && isJson(available) && typeof available.version === 'string' && typeof available.revision === 'string'
}

/**
 * An updates answer as the notices read it (US-270-14e); throws when it isn't a list of notices each carrying
 * its kind, its Cloud row's key of that kind's type, its name and the version and revision it moved to, so the
 * notices show nothing rather than a line they can't install or dismiss by.
 */
export function cloudUpdatesOf(answer: unknown): CloudUpdates {
  const readable = isJson(answer) && Array.isArray(answer.updates) && answer.updates.every(isUpdate)
  if (!readable) throw new Error(CLOUD_UPDATES_UNREADABLE)
  return answer as unknown as CloudUpdates
}
