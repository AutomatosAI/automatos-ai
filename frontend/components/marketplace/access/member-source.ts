/**
 * The Marketplace seam's member side (source.ts): today's routes, read exactly as the homepage and
 * its tabs read them before the seam (the header's counts over the first 100 items, the
 * "Recommended for You" strip from the 20 most installed less the featured ones).
 */
import type { MarketplaceItem } from '@/hooks/use-marketplace-api'
import { apiClient } from '@/lib/api-client'

import { marketplaceItemKey } from '../marketplace-item-ref'
import type { MarketplacePackage } from '../package-types'
import type { MarketplaceCounts, MarketplaceSource } from './source'

const MARKETPLACE_ITEMS = '/api/marketplace/items'
const COUNTED_ITEMS = 100
const RECOMMENDATION_POOL = 20
const RECOMMENDATIONS_SHOWN = 6

async function counts(): Promise<MarketplaceCounts> {
  const items: MarketplaceItem[] = await apiClient.get(`${MARKETPLACE_ITEMS}?limit=${COUNTED_ITEMS}`)
  return {
    totalItems: items.length,
    totalInstalls: items.reduce((sum, item) => sum + item.install_count, 0),
  }
}

async function recommended(featured: readonly MarketplaceItem[]): Promise<MarketplaceItem[]> {
  const items: MarketplaceItem[] = await apiClient.get(`${MARKETPLACE_ITEMS}?limit=${RECOMMENDATION_POOL}`)
  const featuredKeys = new Set(featured.map(marketplaceItemKey))
  return items
    .filter((item) => !featuredKeys.has(marketplaceItemKey(item)))
    .sort((a, b) => b.install_count - a.install_count)
    .slice(0, RECOMMENDATIONS_SHOWN)
}

export const MEMBER_SOURCE: MarketplaceSource = {
  kind: 'member',
  items: (filters) => apiClient.getMarketplaceItems(filters) as Promise<MarketplaceItem[]>,
  item: (id, type) => apiClient.get(`${MARKETPLACE_ITEMS}/${id}?type=${encodeURIComponent(type)}`),
  featured: (limit) => apiClient.getFeaturedMarketplaceItems(limit) as Promise<MarketplaceItem[]>,
  counts,
  popular: recommended,
  packages: () => apiClient.get<MarketplacePackage[]>('/api/marketplace/packages'),
}
