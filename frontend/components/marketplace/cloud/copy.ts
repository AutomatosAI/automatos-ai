/**
 * PRD-270 §5.5 — the words of Sync with Automatos Cloud: the button, its status line and the toast
 * after a Sync. The badge on a synced listing is the Themes tab's (listing-format.ts `CLOUD_BADGE`).
 */
import type { CloudSyncCounts, CloudSyncStatus } from '@/lib/marketplace/cloud-sync'
import { formatRelative } from '@/lib/format-relative'

export const SYNC_WITH_CLOUD = 'Sync with Automatos Cloud'
export const SYNCING = 'Syncing with Automatos Cloud…'
export const NOTHING_NEW = 'Nothing new from Automatos Cloud'
export const NOT_SYNCED_YET = 'Not synced yet'

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

/** The status line beside the button: "Last synced 2m ago · 3 themes", or "Not synced yet". */
export function syncedLine(status: Pick<CloudSyncStatus, 'lastSuccessAt' | 'themes'>, now?: number): string {
  if (!status.lastSuccessAt) return NOT_SYNCED_YET
  return `Last synced ${formatRelative(status.lastSuccessAt, now)} · ${themes(status.themes)}`
}
