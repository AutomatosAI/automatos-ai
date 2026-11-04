/**
 * The Marketplace seam's public side (source.ts), for a signed-out visitor on SaaS: the hosted
 * edition's public reads (lib/marketplace/source/public-reads.ts), mapped to the member shapes
 * (public-shapes.ts), through React Query so a read is asked for once per five minutes (the routes'
 * own `max-age=300`) however many tabs and dialogs draw it.
 *
 * The item list filters by type, category and the search on the client, as the visitor's search
 * always has; an item the public read doesn't list is an error the item dialog shows.
 */
import type { QueryClient } from '@tanstack/react-query'

import type { MarketplaceItem } from '@/hooks/use-marketplace-api'
import { filterPublicItems } from '@/lib/marketplace/public-items'
import type { PublicMarketplaceType } from '@/lib/marketplace/public-items'
import { readPublicCounts, readPublicEntries, readPublicFeatured } from '@/lib/marketplace/source/public-reads'
import type { PublicEntry } from '@/lib/marketplace/source/public-reads'

import {
  AGENT_TYPE,
  PLAYBOOK_TYPE,
  PUBLIC_TYPE_OF,
  featuredItemsOf,
  marketplaceItemsOf,
  packageOf,
} from './public-shapes'
import type { ItemType } from './public-shapes'
import type { MarketplaceItemFilters, MarketplaceSource } from './source'

/** The public routes cache for five minutes (`Cache-Control: public, max-age=300`). */
export const PUBLIC_STALE_MS = 300_000
export const NOT_LISTED = 'This item isn’t listed in the Marketplace.'
const PUBLIC_KEY = 'public-marketplace'
const ITEM_TYPES: readonly ItemType[] = [AGENT_TYPE, PLAYBOOK_TYPE]

/** The public reads as React Query queries, shared by the source and the visitor's own tabs. */
export const publicQueries = {
  entries: (type: PublicMarketplaceType) => ({
    queryKey: [PUBLIC_KEY, 'entries', type] as const,
    queryFn: () => readPublicEntries(type),
    staleTime: PUBLIC_STALE_MS,
  }),
  featured: () => ({ queryKey: [PUBLIC_KEY, 'featured'] as const, queryFn: readPublicFeatured, staleTime: PUBLIC_STALE_MS }),
  counts: () => ({ queryKey: [PUBLIC_KEY, 'counts'] as const, queryFn: readPublicCounts, staleTime: PUBLIC_STALE_MS }),
}

function isItemType(type: string | undefined): type is ItemType {
  return ITEM_TYPES.some((known) => known === type)
}

function matching(entries: readonly PublicEntry[], filters: MarketplaceItemFilters): PublicEntry[] {
  const inCategory = filters.category ? entries.filter((entry) => entry.item.category === filters.category) : entries
  const shown = new Set(filterPublicItems(inCategory.map((entry) => entry.item), filters.search ?? '').map((item) => item.key))
  return inCategory.filter((entry) => shown.has(entry.item.key))
}

export function createPublicSource(client: QueryClient): MarketplaceSource {
  const entries = (type: PublicMarketplaceType) => client.fetchQuery(publicQueries.entries(type))
  const featuredRead = () => client.fetchQuery(publicQueries.featured())

  async function items(filters: MarketplaceItemFilters = {}): Promise<MarketplaceItem[]> {
    const types = filters.type === undefined ? ITEM_TYPES : ITEM_TYPES.filter((type) => type === filters.type)
    const lists = await Promise.all(
      types.map(async (type) => marketplaceItemsOf(matching(await entries(PUBLIC_TYPE_OF[type]), filters), type)),
    )
    const all = lists.flat()
    return filters.limit === undefined ? all : all.slice(filters.offset ?? 0, (filters.offset ?? 0) + filters.limit)
  }

  async function item(id: number, type: string): Promise<MarketplaceItem> {
    const found = isItemType(type) ? (await items({ type })).find((entry) => entry.id === id) : undefined
    if (!found) throw new Error(NOT_LISTED)
    return found
  }

  return {
    kind: 'public',
    items,
    item,
    featured: async (limit) => featuredItemsOf((await featuredRead()).featured).slice(0, limit),
    popular: async () => featuredItemsOf((await featuredRead()).popular),
    counts: async () => {
      const counted = await client.fetchQuery(publicQueries.counts())
      return { totalItems: counted.items, totalInstalls: counted.installs }
    },
    packages: async () => (await entries('packages')).map(packageOf),
  }
}
