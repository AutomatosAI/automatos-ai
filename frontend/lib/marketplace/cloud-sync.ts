/**
 * PRD-270 US-270-05a (§5.5, K17) — Sync with Automatos Cloud's two routes (api/marketplace_cloud_sync.py):
 * its status, `GET /api/marketplace/cloud-sync/status`, and one Sync now, `POST /api/marketplace/cloud-sync`.
 *
 * Local edition only: both answer 404 on the hosted edition, 403 to anyone but a workspace owner or
 * admin, and the POST 409 while `AUTOMATOS_CLOUD_URL` is empty or another Sync runs, 502 with the plain
 * error the Sync recorded. The calls go through apiClient's generic `get`/`post`, so api-client.ts
 * doesn't grow (K3).
 */
import { apiClient } from '@/lib/api-client'

import type { CloudItemKind } from './cloud-item-rows'

export const CLOUD_SYNC_ENDPOINT = '/api/marketplace/cloud-sync'
export const CLOUD_SYNC_STATUS_ENDPOINT = `${CLOUD_SYNC_ENDPOINT}/status`

/** One Sync's listings (pull.SyncCounts.view): new themes or versions, updated, unchanged, refused. */
export interface CloudSyncCounts {
  new: number; updated: number; unchanged: number; refused: number
}

/** The Sync's status (pull.sync_status). */
export interface CloudSyncStatus {
  /** The local edition with `AUTOMATOS_CLOUD_URL` set to an origin a Sync may use. */
  enabled: boolean
  /** The Automatos Cloud origin in use, or null while the setting is empty or not an origin. */
  cloudUrl: string | null
  lastAttemptAt: string | null; lastSuccessAt: string | null
  /** The last failed Sync's plain error, or null. */
  lastError: string | null
  /** The `generatedAt` of the last catalogue applied. */
  generatedAt: string | null
  /** The last successful Sync's counts. */
  counts: CloudSyncCounts
  /** How many themes this install holds from Automatos Cloud. */
  themes: number
  /** How many agents, Playbooks, packages and plugins this install holds from Automatos Cloud (held_items.items_held). */
  items: Readonly<Record<CloudItemKind, number>>
  /** No Sync succeeded in the last 7 days, or none ever did. */
  stale: boolean
}

/** One Sync's answer: its counts, the installs moved to a newer Cloud version, the themes whose update failed. */
export interface CloudSyncResult {
  counts: CloudSyncCounts
  updatesApplied: number
  updatesFailed: number
  status: CloudSyncStatus
}

export function getCloudSyncStatus(): Promise<CloudSyncStatus> {
  return apiClient.get<CloudSyncStatus>(CLOUD_SYNC_STATUS_ENDPOINT)
}

export function runCloudSync(): Promise<CloudSyncResult> {
  return apiClient.post<CloudSyncResult>(CLOUD_SYNC_ENDPOINT)
}
