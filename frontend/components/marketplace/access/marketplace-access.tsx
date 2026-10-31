'use client'

/**
 * One Marketplace page for everyone (PRD-270 §5.7) — who is looking, where the catalogue is read
 * from, and the one guard every add goes through.
 *
 * The DEFAULT is the member's: signed in, today's routes (member-source.ts), and `requireSignIn`
 * lets every add through. With no provider (the local edition, a signed-in member on SaaS, any page
 * that renders a Marketplace piece elsewhere) everything works exactly as before. The /marketplace
 * page provides the visitor's once, signed out on SaaS (visitor-access.tsx): the public reads, and an
 * add that goes to sign-in carrying the item's link instead of adding anything.
 *
 * Each add button needs one line: `if (requireSignIn(tab, item)) return`.
 */
import { createContext, useContext } from 'react'
import type { ReactNode } from 'react'

import { MEMBER_SOURCE } from './member-source'
import type { MarketplaceSource } from './source'

export interface MarketplaceAccess {
  /** False only for a signed-out visitor on SaaS. */
  readonly signedIn: boolean
  /**
   * Signed out: keeps the item's Marketplace link and goes to sign-in with it, then answers true, so
   * the caller adds nothing. Signed in: answers false. `tab` is the item's `?tab=` value, `item` its key.
   */
  readonly requireSignIn: (tab: string, item: string | number) => boolean
  readonly source: MarketplaceSource
}

export const MEMBER_ACCESS: MarketplaceAccess = {
  signedIn: true,
  requireSignIn: () => false,
  source: MEMBER_SOURCE,
}

const MarketplaceAccessContext = createContext<MarketplaceAccess>(MEMBER_ACCESS)

export function MarketplaceAccessProvider({ access, children }: { access: MarketplaceAccess; children: ReactNode }) {
  return <MarketplaceAccessContext.Provider value={access}>{children}</MarketplaceAccessContext.Provider>
}

export function useMarketplaceAccess(): MarketplaceAccess {
  return useContext(MarketplaceAccessContext)
}
