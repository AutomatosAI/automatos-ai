/**
 * PRD-261 — which listing the storefront's hero shows, and which of its looks.
 *
 * The hero opens on the first listing the search shows. A card, a dot, Previous / Next or an
 * installed row's View picks one; picking a listing starts it on its first look. A pick holds while
 * the search it was made under holds (so View can show a listing the search leaves out); once the
 * search changes, the pick holds only if the search still shows it, else the hero shows the first
 * one it does. Previous and Next wrap around the shown listings.
 */
import { useCallback, useState } from 'react'

import type { ThemeListing } from '@/lib/api-client'

export interface StorefrontSelection {
  /** The listing in the hero, or null when nothing is shown. */
  readonly current: ThemeListing | null
  /** Its place among the shown listings; -1 when the search leaves it out. */
  readonly index: number
  /** The hero's look: an index into its views (storefront-views.ts). */
  readonly look: number
  readonly select: (listingId: string) => void
  /** Move by `delta` among the shown listings, wrapping around. */
  readonly step: (delta: number) => void
  readonly pickLook: (look: number) => void
}

interface Picked {
  readonly listingId: string
  /** The search the pick was made under. */
  readonly query: string
}

function currentListing(
  all: readonly ThemeListing[],
  shown: readonly ThemeListing[],
  pick: Picked | null,
  query: string,
): ThemeListing | null {
  const pool = pick?.query === query ? all : shown
  return pool.find((listing) => listing.listingId === pick?.listingId) ?? shown[0] ?? null
}

function stepFrom(index: number, delta: number, count: number): number {
  const from = index < 0 && delta > 0 ? -1 : Math.max(index, 0)
  return (((from + delta) % count) + count) % count
}

export function useStorefrontSelection(
  all: readonly ThemeListing[],
  shown: readonly ThemeListing[],
  query: string,
): StorefrontSelection {
  const [pick, setPick] = useState<Picked | null>(null)
  const [look, setLook] = useState(0)
  const current = currentListing(all, shown, pick, query)
  const index = current ? shown.indexOf(current) : -1

  const select = useCallback(
    (listingId: string) => {
      setPick({ listingId, query })
      setLook(0)
    },
    [query],
  )
  const step = (delta: number) => {
    if (shown.length > 0) select(shown[stepFrom(index, delta, shown.length)].listingId)
  }

  return { current, index, look, select, step, pickLook: setLook }
}
