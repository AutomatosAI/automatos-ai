'use client'

/**
 * One Marketplace page for everyone (PRD-270 §5.7) — a signed-out visitor's Marketplace link,
 * `/marketplace?tab=<tab>&item=<key>` (a share card's URL, US-270-10), opens that item's detail:
 * - an agent or a Playbook: the member item dialog, read through the Marketplace seam;
 * - a package, an application, an LLM, a plugin or a skill: the public detail (public-item-detail.tsx);
 * - a theme: the Themes tab shows it in the hero itself (visitor-themes.tsx).
 *
 * The member page's deep link (deep-link/marketplace-deep-link.tsx) reads member routes, so it
 * mounts only signed in. Reading a link adds nothing; `add=1` changes nothing here, since every Add
 * goes to sign-in. Closing the dialog strips `item` and `add` from the address, as the member's does.
 */
import { useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { publicReadOfTab } from '../category-tabs'
import { marketplaceLink, pathWithoutLink } from '../deep-link/deep-link-params'
import type { MarketplaceLink } from '../deep-link/deep-link-params'
import { MarketplaceItemModal } from '../marketplace-item-modal'
import { AGENT_TYPE, PLAYBOOK_TYPE } from '../access/public-shapes'
import { THEMES_TAB_VALUE } from '../themes/listing-format'
import { PublicItemDetail } from './public-item-detail'

/** The item dialog's type for an agent's or a Playbook's tab. */
const ITEM_TYPES: Readonly<Record<string, string>> = { agents: AGENT_TYPE, recipes: PLAYBOOK_TYPE }
const CATALOGUE_ID = /^[1-9]\d{0,9}$/

function LinkedDetail({ link, onClose }: { link: MarketplaceLink; onClose: () => void }) {
  const itemType = ITEM_TYPES[link.tab]
  if (itemType) {
    return CATALOGUE_ID.test(link.item) ? (
      <MarketplaceItemModal itemId={Number(link.item)} itemType={itemType} onClose={onClose} />
    ) : null
  }
  const type = publicReadOfTab(link.tab)
  return type ? <PublicItemDetail type={type} tab={link.tab} itemKey={link.item} onClose={onClose} /> : null
}

/** Signed out: the link's item, once, until the visitor closes it. */
export function PublicDeepLink() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const link = marketplaceLink(searchParams)
  const [closed, setClosed] = useState(false)
  if (closed || link === null || link.tab === THEMES_TAB_VALUE) return null
  const close = () => {
    setClosed(true)
    router.replace(pathWithoutLink(searchParams))
  }
  return <LinkedDetail link={link} onClose={close} />
}
