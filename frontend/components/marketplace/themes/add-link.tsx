'use client'

/**
 * PRD-270 US-270-09b (§5.7, §8-Q8, K22) — the Themes tab's side of a Marketplace link,
 * `?tab=themes&item=<listing id>&add=1`, as a member lands on it after sign-in or sign-up.
 *
 * Once the listings answer, the listing is picked in the hero (the tab's View). With `add=1` its
 * detail dialog opens too, with focus on Add to my workspace; a member who can't manage themes sees
 * AddToWorkspace's own MANAGERS_ADD sentence instead. A listing that isn't listed opens nothing.
 * Nothing installs until the click: Add is the same control, with its route, permission,
 * who-installs and step-up rules unchanged. Closing the dialog strips `item` and `add` from the
 * address, as the homepage's modal does, so a reload doesn't reopen it.
 *
 * MarketplaceThemesTab renders this only while the tab strip hands it a Themes link, so the tab
 * reads no router otherwise.
 */
import { useEffect, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import type { ThemeListing } from '@/lib/api-client'

import { pathWithoutLink } from '../deep-link/deep-link-params'
import type { MarketplaceLink } from '../deep-link/deep-link-params'
import { DetailDialog } from './detail-dialog'

interface ThemeAddLinkProps {
  readonly link: MarketplaceLink
  /** The approved listings, once they answer. */
  readonly listings: readonly ThemeListing[] | undefined
  readonly canManage: boolean
  /** Shows a listing in the hero. */
  readonly onSelect: (listingId: string) => void
}

export function ThemeAddLink({ link, listings, canManage, onSelect }: ThemeAddLinkProps) {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [closed, setClosed] = useState(false)
  const selected = useRef(false)
  const listed = listings?.some((listing) => listing.listingId === link.item) ?? false

  useEffect(() => {
    if (!listed || selected.current) return
    selected.current = true
    onSelect(link.item)
  }, [listed, link.item, onSelect])

  if (!link.add) return null
  const close = () => {
    setClosed(true)
    router.replace(pathWithoutLink(searchParams))
  }
  return (
    <DetailDialog listingId={listed && !closed ? link.item : null} canManage={canManage} focusAdd onClose={close} />
  )
}
