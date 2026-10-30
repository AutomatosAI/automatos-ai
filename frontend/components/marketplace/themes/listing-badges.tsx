'use client'

/**
 * PRD-261 C-04b — a listing's badges, wherever the Themes tab shows it (the storefront's hero and
 * strip cards, the detail): Official, in this workspace, update available, needs a newer Automatos.
 * No install count, rating or price (plan §2.6). Official comes only from `publisher.official`
 * (QB-21), never from the signature alone.
 */
import { Badge } from '@/components/ui/badge'
import type { ThemeListing } from '@/lib/api-client'

import { INSTALLED_BADGE, NOT_COMPATIBLE, NOT_COMPATIBLE_HINT, OFFICIAL_BADGE, UPDATE_BADGE } from './listing-format'

export function ListingBadges({ listing }: { listing: ThemeListing }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {listing.publisher.official && <Badge>{OFFICIAL_BADGE}</Badge>}
      {listing.installed && <Badge variant="secondary">{INSTALLED_BADGE}</Badge>}
      {listing.installed?.updatePackageId && <Badge variant="outline">{UPDATE_BADGE}</Badge>}
      {!listing.compatible && (
        <Badge variant="destructive" title={NOT_COMPATIBLE_HINT}>
          {NOT_COMPATIBLE}
        </Badge>
      )}
    </div>
  )
}
