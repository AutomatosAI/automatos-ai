/**
 * PRD-270 US-270-08b (§5.7, K20) — the signed-out visitor's Themes tab data: the hosted catalogue's
 * themes (lib/marketplace/public-catalog.ts). The other tabs' public reads go through the Marketplace
 * seam (components/marketplace/access/public-source.ts).
 *
 * React Query, keyed outside every workspace (a visitor has none, and nothing here depends on who
 * asks). The answers are cached for five minutes on the server (`max-age=300`), so they are kept
 * as fresh for as long here. A theme's detail is the same catalogue query, selected by listing:
 * the catalogue carries every version's previews, so opening a theme asks for nothing more.
 */
import { useQuery } from '@tanstack/react-query'

import type { ThemeListingDetail } from '@/lib/api-client'
import { getPublicThemes } from '@/lib/marketplace/public-catalog'

const PUBLIC_STALE_MS = 300_000

export const publicMarketplaceKeys = {
  themes: ['public-marketplace', 'themes'] as const,
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
