'use client'

/**
 * PRD-270 US-270-08b (§5.7, K21) — "Add to my workspace" for a signed-out visitor. It adds nothing:
 * it goes to sign-in carrying the item's Marketplace link as `redirect_url`
 * (`/sign-in?redirect_url=/marketplace?tab=<tab>&item=<key>&add=1`, the PRD's URL contract), so
 * the person comes back to that item; US-270-09a remembers the link for the way back. A secondary
 * button: the visitor header's Start free is the view's one ink primary (K29).
 */
import Link from 'next/link'
import { Download } from 'lucide-react'

import { Button } from '@/components/ui/button'

import { ADD_TO_WORKSPACE, THEMES_TAB_VALUE } from '../themes/listing-format'
import type { AddControlProps } from '../themes/listing-source'

export const SIGN_IN_PATH = '/sign-in'
const MARKETPLACE_PATH = '/marketplace'
const ADD_PARAM = '1'

/** The Marketplace link that opens `item` on `tab` with its add dialog ready (US-270-09b). */
export function marketplaceItemLink(tab: string, item: string): string {
  return `${MARKETPLACE_PATH}?${new URLSearchParams({ tab, item, add: ADD_PARAM })}`
}

/** Sign-in, then back on `item`. */
export function signInToAdd(tab: string, item: string): string {
  return `${SIGN_IN_PATH}?${new URLSearchParams({ redirect_url: marketplaceItemLink(tab, item) })}`
}

export function VisitorAdd({ tab, item }: { tab: string; item: string }) {
  return (
    <Button asChild variant="secondary">
      <Link href={signInToAdd(tab, item)}>
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
