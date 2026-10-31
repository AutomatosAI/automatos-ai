/**
 * PRD-270 W2 US-270-14b (K19) — which Marketplace rows came from Automatos Cloud, for the badges
 * (components/marketplace/cloud/cloud-item-badges.tsx).
 *
 * React Query over lib/marketplace/cloud-items.ts. Local edition only: on the hosted edition the query is
 * never enabled, so nothing is asked (the route answers 404 there). The list is the install's, not a
 * workspace's, so one cache serves every card; a Sync with Automatos Cloud reads it again
 * (hooks/use-cloud-sync.ts invalidates `cloudItemsKeys.all`).
 *
 * `useCloudDisclosure` reads what one Cloud item will use, for the disclosure dialog (US-270-14c), under the
 * same root, so a Sync reads it again too. Nothing of it is kept once its dialog closes, so each opening
 * reads it afresh (what is here, a model, a skill, an app's connection, can change between two) and never
 * shows an earlier answer while it does; it isn't read again under a reader's eyes, and a failed read, or
 * an answer the dialog can't show whole (`cloudDisclosureOf`), is shown at once, not retried.
 */
import { useQuery } from '@tanstack/react-query'

import { isLocal } from '@/lib/auth-edition'
import { cloudDisclosureOf, getCloudDisclosure, getCloudItems } from '@/lib/marketplace/cloud-items'
import type { CloudDisclosure, CloudItemKey, CloudItemKind, CloudItems } from '@/lib/marketplace/cloud-items'

export const CLOUD_ITEMS_STALE_MS = 60_000

const CLOUD_ITEMS_ROOT = ['marketplace', 'cloud-items'] as const

export const cloudItemsKeys = {
  all: CLOUD_ITEMS_ROOT,
  items: () => [...CLOUD_ITEMS_ROOT, 'items'] as const,
  disclosure: (kind: CloudItemKind, key: CloudItemKey) => [...CLOUD_ITEMS_ROOT, 'disclosure', kind, String(key)] as const,
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
