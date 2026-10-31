/**
 * PRD-270 W2 US-270-14b (K19) — which Marketplace rows came from Automatos Cloud, for the badges
 * (components/marketplace/cloud/cloud-item-badges.tsx).
 *
 * React Query over lib/marketplace/cloud-items.ts. Local edition only: on the hosted edition the query is
 * never enabled, so nothing is asked (the route answers 404 there). The list is the install's, not a
 * workspace's, so one cache serves every card; a Sync with Automatos Cloud reads it again
 * (hooks/use-cloud-sync.ts invalidates `cloudItemsKeys.all`).
 *
 * `fetchCloudItems` is the same list read outside a component, for the install gate (CloudItemsLayer,
 * US-270-14d): read afresh for every install, since a Sync in another tab can add a Cloud item the cached
 * list doesn't know yet; installs at the same moment share one read, and the badges get the new list.
 *
 * `useCloudDisclosure` reads what one Cloud item will use, for the disclosure dialog (US-270-14c), under the
 * same root, so a Sync reads it again too. Nothing of it is kept once its dialog closes, so each opening
 * reads it afresh (what is here, a model, a skill, an app's connection, can change between two) and never
 * shows an earlier answer while it does; it isn't read again under a reader's eyes, and a failed read, or
 * an answer the dialog can't show whole (`cloudDisclosureOf`), is shown at once, not retried.
 *
 * `useCloudUpdates` reads the workspace's update notices (US-270-14e), keyed by the workspace (a switch never
 * shows another workspace's notices) under the same root, so a Sync reads them again; an answer the notices
 * can't read whole (`cloudUpdatesOf`) is an error, and they show nothing. `refreshCloudUpdates` reads them
 * again after an install (the agent and Playbook install hooks call it): a new copy may clear a notice.
 */
import { useQuery } from '@tanstack/react-query'
import type { QueryClient } from '@tanstack/react-query'

import { isLocal } from '@/lib/auth-edition'
import type { CloudItemKey, CloudItemKind, CloudItems } from '@/lib/marketplace/cloud-item-rows'
import {
  cloudDisclosureOf, cloudUpdatesOf, getCloudDisclosure, getCloudItems, getCloudUpdates,
} from '@/lib/marketplace/cloud-items'
import type { CloudDisclosure, CloudUpdates } from '@/lib/marketplace/cloud-items'

export const CLOUD_ITEMS_STALE_MS = 60_000

const CLOUD_ITEMS_ROOT = ['marketplace', 'cloud-items'] as const

export const cloudItemsKeys = {
  all: CLOUD_ITEMS_ROOT,
  items: () => [...CLOUD_ITEMS_ROOT, 'items'] as const,
  disclosure: (kind: CloudItemKind, key: CloudItemKey) => [...CLOUD_ITEMS_ROOT, 'disclosure', kind, String(key)] as const,
  updatesRoot: () => [...CLOUD_ITEMS_ROOT, 'updates'] as const,
  updates: (workspaceId: string | null) => [...CLOUD_ITEMS_ROOT, 'updates', workspaceId] as const,
}

/** The Cloud agents, Playbooks, packages and plugins this install holds; never asked on the hosted edition. */
export function useCloudItems() {
  return useQuery<CloudItems>({
    queryKey: cloudItemsKeys.items(),
    enabled: isLocal,
    queryFn: getCloudItems,
    staleTime: CLOUD_ITEMS_STALE_MS,
  })
}

/** The same Cloud items, read afresh through `queryClient` (a read already under way is shared). */
export function fetchCloudItems(queryClient: QueryClient): Promise<CloudItems> {
  return queryClient.fetchQuery<CloudItems>({ queryKey: cloudItemsKeys.items(), queryFn: getCloudItems, staleTime: 0 })
}

/** What one Cloud item will use (`GET /items/{kind}/{key}/disclosure`); never asked on the hosted edition. */
export function useCloudDisclosure(kind: CloudItemKind, key: CloudItemKey) {
  return useQuery<CloudDisclosure>({
    queryKey: cloudItemsKeys.disclosure(kind, key),
    enabled: isLocal,
    queryFn: async () => cloudDisclosureOf(await getCloudDisclosure(kind, key), kind),
    cacheTime: 0,
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
}

/** The update notices of `workspaceId` (`GET /updates`); never asked on the hosted edition or with no workspace. */
export function useCloudUpdates(workspaceId: string | null) {
  return useQuery<CloudUpdates>({
    queryKey: cloudItemsKeys.updates(workspaceId),
    enabled: isLocal && !!workspaceId,
    queryFn: async () => cloudUpdatesOf(await getCloudUpdates()),
  })
}

/** Reads the update notices again: after an install, a copy at a Cloud item's current version clears its notice. */
export function refreshCloudUpdates(queryClient: QueryClient): Promise<void> {
  return queryClient.invalidateQueries({ queryKey: cloudItemsKeys.updatesRoot() })
}
