'use client'

/**
 * PRD-270 US-270-08b (K20) — where the Themes storefront reads a listing's detail and its Add
 * control from, so the same storefront draws for a member and for a signed-out visitor.
 *
 * The DEFAULT is the member's: `useThemeListing` (C-04a's detail route, this workspace's install
 * state) and C-04b's `AddToWorkspace`, and "Submit a theme package for review" shows; with no
 * provider, the member page renders exactly as before. The visitor's provider
 * (components/marketplace/public/visitor-themes.tsx) answers the detail from the public catalogue,
 * its Add goes to sign-in, and it offers no submit.
 *
 * `useDetail` is a hook: a source is a module-level constant, never built per render, so the hooks
 * a component calls never change between its renders.
 */
import { createContext, useContext } from 'react'
import type { ComponentType, ReactNode } from 'react'

import type { ThemeListing, ThemeListingDetail } from '@/lib/api-client'

import { AddToWorkspace } from './install-listing'
import { useThemeListing } from './listing-hooks'

/** What the hero and the detail dialog read of a listing's detail (a react-query result fits). */
export interface ListingDetailRead {
  readonly data?: ThemeListingDetail
  readonly isLoading: boolean
  readonly isError: boolean
  readonly error: unknown
}

export interface AddControlProps {
  readonly listing: ThemeListing
  readonly canManage: boolean
  /** Take focus once shown: the add dialog a `?item=&add=1` link opens (US-270-09b). */
  readonly autoFocus?: boolean
}

export interface ListingSource {
  /** One listing's detail, for the hero's listing or the open detail; null asks for none. */
  readonly useDetail: (listingId: string | null) => ListingDetailRead
  /** "Add to my workspace" for a listing. */
  readonly AddControl: ComponentType<AddControlProps>
  /** "Make your own" offers "Submit a theme package for review" (C-01b's Publish theme form). */
  readonly canPublish: boolean
}

/** Today's storefront: the member routes and controls. */
export const MEMBER_LISTING_SOURCE: ListingSource = {
  useDetail: useThemeListing,
  AddControl: AddToWorkspace,
  canPublish: true,
}

const ListingSourceContext = createContext<ListingSource>(MEMBER_LISTING_SOURCE)

export function ListingSourceProvider({ source, children }: { source: ListingSource; children: ReactNode }) {
  return <ListingSourceContext.Provider value={source}>{children}</ListingSourceContext.Provider>
}

export function useListingSource(): ListingSource {
  return useContext(ListingSourceContext)
}
