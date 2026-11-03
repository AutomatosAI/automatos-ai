'use client'

/**
 * PRD-270 US-270-08b (§5.7, K20) — the signed-out visitor's Themes tab: the member storefront
 * (hero, looks, web and phone screenshots, the strip) and its detail dialog, unchanged, over the
 * hosted catalogue (hooks/use-public-marketplace.ts).
 *
 * The visitor's listing source answers the hero's and the dialog's detail from the catalogue item,
 * its Add goes to sign-in (visitor-add.tsx), and "Make your own" offers the Design Kit but no
 * submit. Nothing workspace-scoped is drawn: no installed list, no "In this workspace" or update
 * badge (a catalogue listing has no install), no Publish form, no theme settings read. A shared
 * link's `item` is shown in the hero once the catalogue lists it. The one Marketplace page renders it
 * in the Themes tab's place for a signed-out visitor (marketplace-tab-set.ts).
 */
import { useEffect, useMemo, useRef, useState } from 'react'

import { usePublicThemeDetail, usePublicThemes } from '@/hooks/use-public-marketplace'
import type { ThemeListing } from '@/lib/api-client'

import type { MarketplaceLink } from '../deep-link/deep-link-params'
import { DetailDialog } from '../themes/detail-dialog'
import { filterListings } from '../themes/listing-format'
import { ListingSourceProvider } from '../themes/listing-source'
import type { ListingSource } from '../themes/listing-source'
import { Storefront } from '../themes/storefront/storefront'
import { useStorefrontSelection } from '../themes/storefront/use-storefront-selection'
import { VisitorThemeAdd } from './visitor-add'

const VISITOR_LISTING_SOURCE: ListingSource = {
  useDetail: usePublicThemeDetail,
  AddControl: VisitorThemeAdd,
  canPublish: false,
}

const NONE_LISTED: readonly ThemeListing[] = []

/** Show the linked listing in the hero once, when the catalogue first lists it. */
function useLinkedListing(all: readonly ThemeListing[], item: string | null, select: (listingId: string) => void) {
  const shown = useRef(false)
  useEffect(() => {
    if (shown.current || !item || !all.some((listing) => listing.listingId === item)) return
    shown.current = true
    select(item)
  }, [all, item, select])
}

interface VisitorThemesTabProps {
  readonly searchQuery?: string
  /** The address's Themes link (`?tab=themes&item=`), if it carries one: shown in the hero. */
  readonly link?: MarketplaceLink | null
}

/** The Themes tab for a signed-out visitor, in the member tab's place (same props). */
export function VisitorThemesTab({ searchQuery = '', link = null }: VisitorThemesTabProps) {
  const item = link?.item ?? null
  const listings = usePublicThemes()
  const all = listings.data ?? NONE_LISTED
  const shown = useMemo(() => filterListings(all, searchQuery), [all, searchQuery])
  const selection = useStorefrontSelection(all, shown, searchQuery)
  const [openListing, setOpenListing] = useState<string | null>(null)
  const stage = useRef<HTMLElement>(null)
  useLinkedListing(all, item, selection.select)

  return (
    <ListingSourceProvider source={VISITOR_LISTING_SOURCE}>
      <div className="space-y-10" data-testid="visitor-themes-tab">
        <Storefront
          listings={listings}
          shown={shown}
          selection={selection}
          canManage={false}
          onDetails={setOpenListing}
          stageRef={stage}
        />
        <DetailDialog listingId={openListing} canManage={false} onClose={() => setOpenListing(null)} />
      </div>
    </ListingSourceProvider>
  )
}
