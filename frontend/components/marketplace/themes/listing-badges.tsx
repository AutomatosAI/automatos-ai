'use client'

/**
 * PRD-261 C-04b — one card of the Themes tab: cover, name, publisher, version, `uses` and this
 * workspace's state (in this workspace, update available, needs a newer Automatos, a warning
 * revocation). Opening it shows the detail. No install count, rating or price (plan §2.6).
 */
import { Palette } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import type { ThemeListing } from '@/lib/api-client'

import {
  INSTALLED_BADGE,
  NOT_COMPATIBLE,
  NOT_COMPATIBLE_HINT,
  OFFICIAL_BADGE,
  UPDATE_BADGE,
  listingName,
  publisherLabel,
  themeFileUrl,
} from './listing-format'

const MAX_USES_SHOWN = 3

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

function Cover({ listing }: { listing: ThemeListing }) {
  const src = themeFileUrl(listing.cover)
  if (!src) {
    return (
      <div className="flex aspect-video items-center justify-center rounded-lg bg-secondary/40" aria-hidden>
        <Palette className="h-8 w-8 text-muted-foreground" />
      </div>
    )
  }
  // A theme file on the app's own origin (A-08b proxy); next/image would re-encode it.
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt="" className="aspect-video w-full rounded-lg object-cover" loading="lazy" />
}

interface ListingCardProps {
  readonly listing: ThemeListing
  readonly onOpen: (listingId: string) => void
}

export function ListingCard({ listing, onOpen }: ListingCardProps) {
  const name = listingName(listing)
  return (
    <Card
      role="button"
      tabIndex={0}
      aria-label={`${name} theme`}
      data-testid={`theme-listing-${listing.listingId}`}
      className="cursor-pointer border-border/40 bg-card/50 transition-colors hover:border-primary/20"
      onClick={() => onOpen(listing.listingId)}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onOpen(listing.listingId)
        }
      }}
    >
      <CardContent className="space-y-3 p-4">
        <Cover listing={listing} />
        <div className="min-w-0 space-y-1">
          <p className="truncate font-medium text-foreground">
            <span>{name}</span> <span className="text-xs text-muted-foreground">{listing.version}</span>
          </p>
          <p className="truncate text-xs text-muted-foreground">{publisherLabel(listing)}</p>
          {listing.description && <p className="line-clamp-2 text-sm text-muted-foreground">{listing.description}</p>}
        </div>
        <ListingBadges listing={listing} />
        {listing.revoked && <p className="text-xs text-warning">{listing.revoked.reason}</p>}
        {listing.uses.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {listing.uses.slice(0, MAX_USES_SHOWN).map((use) => (
              <Badge key={use} variant="outline" className="px-1.5 py-0 text-[10px]">
                {use}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
