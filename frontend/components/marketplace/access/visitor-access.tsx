'use client'

/**
 * The signed-out visitor's Marketplace access (marketplace-access.tsx), provided once by the
 * /marketplace page on SaaS: the public reads (public-source.ts), and an add that adds nothing.
 *
 * Every Install, Add to my workspace, Add and Enable goes through `requireSignIn`: it remembers the
 * item's Marketplace link for the way back (lib/marketplace/return-after-sign-in.ts) and goes to
 * sign-in carrying the same link as `redirect_url`, `/marketplace?tab=<tab>&item=<key>&add=1`, so
 * the person lands on that item with its add dialog ready (US-270-09b). The install itself stays the
 * member's normal, permission-checked route, after a click.
 */
import { useMemo, useRef } from 'react'
import type { ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useRouter } from 'next/navigation'

import { marketplaceItemPath, remember, signInUrl } from '@/lib/marketplace/return-after-sign-in'

import { MarketplaceAccessProvider } from './marketplace-access'
import type { MarketplaceAccess } from './marketplace-access'
import { createPublicSource } from './public-source'

export function VisitorAccessProvider({ children }: { children: ReactNode }) {
  const client = useQueryClient()
  // The access must stay one object for the page's life: the homepage re-reads when its source changes.
  const router = useRef(useRouter())
  const access = useMemo<MarketplaceAccess>(
    () => ({
      signedIn: false,
      source: createPublicSource(client),
      requireSignIn: (tab, item) => {
        const returnPath = marketplaceItemPath(tab, String(item))
        remember(returnPath)
        router.current.push(signInUrl(returnPath))
        return true
      },
    }),
    [client],
  )
  return <MarketplaceAccessProvider access={access}>{children}</MarketplaceAccessProvider>
}
