/**
 * PRD-270 §5.5 — the words of Sync with Automatos Cloud: the button, its status line and the toast
 * after a Sync, and Settings › Automatos Cloud. The badge on a synced listing is the Themes tab's
 * (listing-format.ts `CLOUD_BADGE`).
 */
import type { CloudItemKind } from '@/lib/marketplace/cloud-item-rows'
import type { CloudSyncCounts, CloudSyncStatus } from '@/lib/marketplace/cloud-sync'
import { formatRelative } from '@/lib/format-relative'

export const SYNC_WITH_CLOUD = 'Sync with Automatos Cloud'
export const SYNCING = 'Syncing with Automatos Cloud…'
export const NOTHING_NEW = 'Nothing new from Automatos Cloud'
export const NOT_SYNCED_YET = 'Not synced yet'

export const CLOUD_TITLE = 'Automatos Cloud'
export const CLOUD_LEDE =
  'Where this install syncs Marketplace themes, agents, Playbooks, packages and plugins from. A Sync runs only when someone presses it. It lists only theme versions an Automatos key signed; agents, Playbooks, packages and plugins are Reviewed on Automatos Cloud, and each shows what it will use before it is added.'
export const SYNC_NOW = 'Sync now'
export const READING_STATUS = 'Reading Sync with Automatos Cloud…'
export const SYNC_OFF = 'Sync with Automatos Cloud is off on this install: AUTOMATOS_CLOUD_URL is empty.'
export const TURN_OFF = 'To turn it off, set AUTOMATOS_CLOUD_URL= (empty) in envs/api.local and restart.'

/** The last successful Sync's counts, for Settings: "2 new, 1 updated, 5 unchanged, 0 refused". */
export function countsLine(counts: CloudSyncCounts): string {
  return `${counts.new} new, ${counts.updated} updated, ${counts.unchanged} unchanged, ${counts.refused} refused`
}

function themes(count: number): string {
  return count === 1 ? '1 theme' : `${count} themes`
}

/** The toast after a Sync: "2 new, 1 updated from Automatos Cloud", or nothing new. */
export function syncedMessage(counts: Pick<CloudSyncCounts, 'new' | 'updated'>): string {
  if (counts.new === 0 && counts.updated === 0) return NOTHING_NEW
  return `${counts.new} new, ${counts.updated} updated from Automatos Cloud`
}

/** After a Sync, the themes whose newest Cloud version could not be applied to their installs. */
export function updatesFailedMessage(failed: number): string {
  return `${themes(failed)} could not be updated from Automatos Cloud. Their installs stay as they were; the next Sync tries again.`
}

/** Each item type the status line names, in its order, as one and as many. */
const ITEM_NOUNS: ReadonlyArray<readonly [kind: CloudItemKind, one: string, many: string]> = [
  ['agent', 'agent', 'agents'],
  ['playbook', 'Playbook', 'Playbooks'],
  ['package', 'package', 'packages'],
  ['plugin', 'plugin', 'plugins'],
]

/** "2 agents", "1 Playbook" … for each item type held here; a type with none is left out. */
function itemsHeld(items: CloudSyncStatus['items']): string[] {
  return ITEM_NOUNS.filter(([kind]) => items[kind] > 0).map(([kind, one, many]) =>
    items[kind] === 1 ? `1 ${one}` : `${items[kind]} ${many}`,
  )
}

/** Settings' items held here: "2 agents, 1 Playbook", or "none". */
export function itemsHeldLine(items: CloudSyncStatus['items']): string {
  const held = itemsHeld(items)
  return held.length > 0 ? held.join(', ') : 'none'
}

/**
 * The status line beside the button: "Last synced 2m ago · 3 themes · 2 agents · 1 Playbook", or "Not synced
 * yet". The themes are always named; an item type this install holds none of is left out.
 */
export function syncedLine(status: Pick<CloudSyncStatus, 'lastSuccessAt' | 'themes' | 'items'>, now?: number): string {
  if (!status.lastSuccessAt) return NOT_SYNCED_YET
  const held = [themes(status.themes), ...itemsHeld(status.items)]
  return `Last synced ${formatRelative(status.lastSuccessAt, now)} · ${held.join(' · ')}`
}
