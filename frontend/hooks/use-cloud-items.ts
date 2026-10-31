/**
 * PRD-270 W2 US-270-14b (K19) — which Marketplace rows came from Automatos Cloud, for the badges
 * (components/marketplace/cloud/cloud-item-badges.tsx).
 *
 * React Query over lib/marketplace/cloud-items.ts. Local edition only: on the hosted edition the query is
 * never enabled, so nothing is asked (the route answers 404 there). The list is the install's, not a
 * workspace's, so one cache serves every card; a Sync with Automatos Cloud reads it again
 * (hooks/use-cloud-sync.ts invalidates `cloudItemsKeys.all`).
 */
import { useQuery } from '@tanstack/react-query'

import { isLocal } from '@/lib/auth-edition'
import { getCloudItems } from '@/lib/marketplace/cloud-items'
import type { CloudItems } from '@/lib/marketplace/cloud-items'

export const CLOUD_ITEMS_STALE_MS = 60_000

const CLOUD_ITEMS_ROOT = ['marketplace', 'cloud-items'] as const

export const cloudItemsKeys = {
  all: CLOUD_ITEMS_ROOT,
  items: () => [...CLOUD_ITEMS_ROOT, 'items'] as const,
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
