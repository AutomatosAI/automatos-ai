/**
 * PRD-270 US-270-08b (§5.7, K20) — the signed-out visitor's Marketplace data: the hosted catalogue's
 * themes and one public read per tab (lib/marketplace/public-catalog.ts).
 *
 * React Query, keyed outside every workspace (a visitor has none, and nothing here depends on who
 * asks). The answers are cached for five minutes on the server (`max-age=300`), so they are kept
 * as fresh for as long here. A theme's detail is the same catalogue query, selected by listing:
 * the catalogue carries every version's previews, so opening a theme asks for nothing more. A tab
 * reads a page at a time, and Show more asks for the next one.
 */
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'

import type { ThemeListingDetail } from '@/lib/api-client'
import { getPublicPage, getPublicThemes } from '@/lib/marketplace/public-catalog'
import type { PublicMarketplaceType, PublicPage } from '@/lib/marketplace/public-items'

const PUBLIC_STALE_MS = 300_000

export const publicMarketplaceKeys = {
  themes: ['public-marketplace', 'themes'] as const,
  items: (type: PublicMarketplaceType) => ['public-marketplace', 'items', type] as const,
}

/** Every theme the hosted catalogue lists, one listing each. */
export function usePublicThemes() {
  return useQuery<ThemeListingDetail[]>({
    queryKey: publicMarketplaceKeys.themes,
    queryFn: getPublicThemes,
    staleTime: PUBLIC_STALE_MS,
  })
}

/** One theme's detail from the catalogue; undefined while it loads, or when it lists no such theme. */
export function usePublicThemeDetail(listingId: string | null) {
  return useQuery<ThemeListingDetail[], unknown, ThemeListingDetail | undefined>({
    queryKey: publicMarketplaceKeys.themes,
    queryFn: getPublicThemes,
    staleTime: PUBLIC_STALE_MS,
    select: (listings) => listings.find((listing) => listing.listingId === listingId),
  })
}

/** A tab's public items, a page at a time from the first. */
export function usePublicItems(type: PublicMarketplaceType) {
  return useInfiniteQuery<PublicPage>({
    queryKey: publicMarketplaceKeys.items(type),
    queryFn: ({ pageParam }) => getPublicPage(type, typeof pageParam === 'number' ? pageParam : 0),
    getNextPageParam: (last) => last.next ?? undefined,
    staleTime: PUBLIC_STALE_MS,
  })
}
