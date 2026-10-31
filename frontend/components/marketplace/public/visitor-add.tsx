'use client'

/**
 * PRD-270 US-270-08b, US-270-09a (§5.7, K21) — "Add to my workspace" for a signed-out visitor. It
 * adds nothing: it remembers the item's Marketplace link for the way back
 * (lib/marketplace/return-after-sign-in.ts) and goes to sign-in carrying the same link as
 * `redirect_url` (`/sign-in?redirect_url=/marketplace?tab=<tab>&item=<key>&add=1`, the PRD's URL
 * contract), so the person comes back to that item. A secondary button: the visitor header's
 * Start free is the view's one ink primary (K29).
 */
import Link from 'next/link'
import { Download } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { marketplaceItemPath, remember, signInUrl } from '@/lib/marketplace/return-after-sign-in'

import { ADD_TO_WORKSPACE, THEMES_TAB_VALUE } from '../themes/listing-format'
import type { AddControlProps } from '../themes/listing-source'

/** Sign-in, then back on `item` on `tab`. */
export function VisitorAdd({ tab, item }: { tab: string; item: string }) {
  const returnPath = marketplaceItemPath(tab, item)
  return (
    <Button asChild variant="secondary">
      <Link href={signInUrl(returnPath)} onClick={() => remember(returnPath)}>
        <Download className="mr-2 h-4 w-4" aria-hidden />
        {ADD_TO_WORKSPACE}
      </Link>
    </Button>
  )
}

/** The storefront's Add control on the visitor's page (its listing source's `AddControl`). */
export function VisitorThemeAdd({ listing }: AddControlProps) {
  return <VisitorAdd tab={THEMES_TAB_VALUE} item={listing.listingId} />
}
