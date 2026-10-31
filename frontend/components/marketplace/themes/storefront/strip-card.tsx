'use client'

/**
 * PRD-261 — one card of the storefront's strip: the listing's cover with its Automatos Go screen
 * peeking in front (when it has one), name, badges (ListingBadges), a warning revocation's
 * reason, two lines of description, and its id and version. A toggle button: choosing it puts
 * the listing in the hero; the one in the hero is pressed. Its images are lazy: the strip is
 * below the hero. No install count, rating or price (plan §2.6).
 */
import { Palette } from 'lucide-react'

import type { ThemeListing } from '@/lib/api-client'
import { cn } from '@/lib/utils'

import { ListingBadges } from '../listing-badges'
import { listingName, themeFileUrl } from '../listing-format'

interface StripCardProps {
  readonly listing: ThemeListing
  readonly selected: boolean
  readonly onSelect: (listingId: string) => void
}

function CardCover({ listing }: { listing: ThemeListing }) {
  const cover = themeFileUrl(listing.cover)
  const phone = themeFileUrl(listing.phoneCover)
  return (
    <div className="relative aspect-[16/10] w-full overflow-hidden bg-muted">
      {cover ? (
        // A theme file on the app's own origin (A-08b proxy); next/image would re-encode it.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={cover}
          alt=""
          loading="lazy"
          className="block h-full w-full object-cover motion-safe:transition-transform motion-safe:duration-500 motion-safe:group-hover:scale-[1.03]"
        />
      ) : (
        <div className="flex h-full items-center justify-center" aria-hidden>
          <Palette className="h-8 w-8 text-muted-foreground" />
        </div>
      )}
      {phone && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={phone}
          alt=""
          loading="lazy"
          className="absolute -bottom-10 right-3 h-[138px] w-16 rounded-xl border-[3px] border-background object-cover object-top"
        />
      )}
    </div>
  )
}

export function StripCard({ listing, selected, onSelect }: StripCardProps) {
  const name = listingName(listing)
  return (
    <button
      type="button"
      aria-pressed={selected}
      aria-label={`${name} theme`}
      data-testid={`theme-listing-${listing.listingId}`}
      onClick={() => onSelect(listing.listingId)}
      className={cn(
        'group flex w-[300px] shrink-0 snap-start flex-col overflow-hidden rounded-2xl border-2 bg-card text-left text-foreground sm:w-[360px]',
        'ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
        selected ? 'border-primary' : 'border-border hover:border-primary/40',
      )}
    >
      <CardCover listing={listing} />
      <span className="flex flex-col gap-2 px-4 pb-4 pt-4">
        <span className="text-lg font-semibold tracking-tight">{name}</span>
        <ListingBadges listing={listing} />
        {listing.revoked && <span className="text-xs text-warning">{listing.revoked.reason}</span>}
        {listing.description && (
          <span className="line-clamp-2 text-sm leading-snug text-muted-foreground">{listing.description}</span>
        )}
        <span className="font-mono text-xs text-muted-foreground">
          {listing.themeId} · {listing.version}
        </span>
      </span>
    </button>
  )
}
