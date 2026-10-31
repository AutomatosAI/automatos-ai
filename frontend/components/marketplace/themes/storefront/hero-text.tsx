'use client'

/**
 * PRD-261 — the hero's words and actions for the listing it shows: its place among the shown
 * listings, name, badges (ListingBadges: Official, in this workspace, update, needs a newer
 * Automatos), id and version, a warning revocation's reason, description, the looks chips, and
 * the install controls.
 *
 * Install is C-04b's `AddToWorkspace`, reused as it is: owners and admins add (behind "who
 * installs" and step-up), every member then gets "Use this theme", and a member who can't add
 * reads MANAGERS_ADD. It is keyed by listing, so the result of adding one listing never carries
 * over to the next. "Details" opens C-04b's detail dialog (facts, versions, documents). The Add
 * control comes from the listing source (../listing-source.tsx, PRD-270 K20): `AddToWorkspace` by
 * default, sign-in on the signed-out visitor's page.
 */
import { Button } from '@/components/ui/button'
import type { ThemeListing } from '@/lib/api-client'

import { ListingBadges } from '../listing-badges'
import { listingName } from '../listing-format'
import { useListingSource } from '../listing-source'
import { DETAILS, UPDATES_NOTE, featuredPosition } from './copy'
import { LooksChips } from './looks-chips'
import type { StorefrontView } from './storefront-views'

interface HeroTextProps {
  readonly listing: ThemeListing
  /** 1-based place among the shown listings; 0 when the search leaves it out. */
  readonly position: number
  readonly count: number
  readonly views: readonly StorefrontView[]
  readonly look: number
  readonly onLook: (look: number) => void
  readonly canManage: boolean
  readonly onDetails: (listingId: string) => void
}

function HeroActions({ listing, canManage, onDetails }: Pick<HeroTextProps, 'listing' | 'canManage' | 'onDetails'>) {
  const { AddControl } = useListingSource()
  return (
    <div className="space-y-3 pt-1">
      <div className="flex flex-wrap items-start gap-3">
        <AddControl key={listing.listingId} listing={listing} canManage={canManage} />
        <Button variant="outline" onClick={() => onDetails(listing.listingId)}>
          {DETAILS}
        </Button>
      </div>
      <p className="text-sm text-muted-foreground">{UPDATES_NOTE}</p>
    </div>
  )
}

export function HeroText({ listing, position, count, views, look, onLook, canManage, onDetails }: HeroTextProps) {
  return (
    <div className="flex min-w-0 flex-[1_1_340px] flex-col gap-5">
      <div className="space-y-3" aria-live="polite">
        {position > 0 && (
          <p className="font-mono text-xs uppercase tracking-[0.12em] text-muted-foreground">
            {featuredPosition(position, count)}
          </p>
        )}
        <h3
          className="break-words text-5xl font-bold leading-[0.95] tracking-tight text-foreground sm:text-7xl"
          data-testid="storefront-hero-name"
        >
          {listingName(listing)}
        </h3>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <ListingBadges listing={listing} />
        <span className="rounded-full border border-border px-2.5 py-0.5 font-mono text-xs text-muted-foreground">
          {listing.themeId} · {listing.version}
        </span>
      </div>
      {listing.revoked && <p className="text-sm text-warning">{listing.revoked.reason}</p>}
      {listing.description && (
        <p className="max-w-md text-lg leading-relaxed text-foreground/90">{listing.description}</p>
      )}
      <LooksChips views={views} look={look} onLook={onLook} />
      <HeroActions listing={listing} canManage={canManage} onDetails={onDetails} />
    </div>
  )
}
