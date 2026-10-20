'use client'

/**
 * PRD-261 C-04b — the Themes tab's listing grid: every approved theme (C-04a), filtered by the
 * Marketplace's search box. A refusal (the Marketplace switch off, signed out) shows the server's
 * sentence; nothing is listed that the server did not list.
 */
import { themeErrorMessage } from '@/lib/theme-host/api'

import { ListingCard } from './listing-card'
import { LOADING_LISTINGS, NO_LISTINGS, NO_MATCHES, filterListings } from './listing-format'
import { useThemeListings } from './listing-hooks'

interface ListingGridProps {
  readonly searchQuery: string
  readonly onOpen: (listingId: string) => void
}

export function ListingGrid({ searchQuery, onOpen }: ListingGridProps) {
  const listings = useThemeListings()

  if (listings.isLoading) return <p className="text-sm text-muted-foreground">{LOADING_LISTINGS}</p>
  if (listings.isError) {
    return (
      <p className="text-sm text-destructive" role="alert">
        {themeErrorMessage(listings.error)}
      </p>
    )
  }
  const all = listings.data ?? []
  const shown = filterListings(all, searchQuery)
  if (shown.length === 0) {
    return <p className="text-sm text-muted-foreground">{all.length === 0 ? NO_LISTINGS : NO_MATCHES}</p>
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3" data-testid="theme-listing-grid">
      {shown.map((listing) => (
        <ListingCard key={listing.listingId} listing={listing} onOpen={onOpen} />
      ))}
    </div>
  )
}
