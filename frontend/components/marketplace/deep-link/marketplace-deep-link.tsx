'use client'

/**
 * PRD-270 US-270-09b (§5.7, §8-Q8, K22) — the member Marketplace's one deep link, mounted once in
 * MarketplaceHomepage, for every tab but Themes (whose tab opens its own, themes/add-link.tsx).
 *
 * A `?tab=<tab>&item=<key>&add=1` link, as a member lands on it after sign-in or sign-up
 * (US-270-09a), opens the item's detail modal, the one its tab opens on a click, with the item read
 * through the route that tab already calls (deep-link-openers.tsx):
 * - agents and Playbooks: MarketplaceItemModal (the homepage's own `?id=&type=` modal);
 * - packages: PackageDetailModal, by slug;
 * - applications: MarketplaceAppDetailsModal, from the Applications tab's catalogue read;
 * - plugins: MarketplacePluginDetailModal;
 * - LLMs: LLMModelDetailModal, its route from the LLMs tab's catalogue read;
 * - a skill, which has no detail modal: the tab strip opens Capabilities › Skills for `?tab=skills`
 *   (category-tabs.ts), and this sets the Marketplace search to the skill's name.
 *
 * The seven tab components are untouched (K4). An item that isn't found, or a key that isn't one,
 * opens nothing, and nothing installs on load. Closing the modal (or applying a skill's search)
 * strips `item` and `add` from the address, the homepage's handleCloseModal pattern, so a reload
 * doesn't reopen it. With no link in the address it renders nothing and never calls useRouter.
 */
import { useEffect, useRef, useState } from 'react'
import type { ComponentType } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { PLUGINS_SUB_TAB, SKILLS_SUB_TAB } from '../category-tabs'
import { THEMES_TAB_VALUE } from '../themes/listing-format'
import { marketplaceLink, pathWithoutLink } from './deep-link-params'
import type { MarketplaceLink } from './deep-link-params'
import {
  AgentOpener,
  ApplicationOpener,
  LlmOpener,
  PackageOpener,
  PlaybookOpener,
  PluginOpener,
} from './deep-link-openers'
import type { OpenerProps } from './deep-link-openers'

/** Each tab's opener, by the tab's link value (CATEGORY_TABS, and `plugins` for Capabilities'). */
const OPENERS: ReadonlyMap<string, ComponentType<OpenerProps>> = new Map<string, ComponentType<OpenerProps>>([
  ['agents', AgentOpener],
  ['recipes', PlaybookOpener],
  ['packages', PackageOpener],
  ['tools', ApplicationOpener],
  [PLUGINS_SUB_TAB, PluginOpener],
  ['llms', LlmOpener],
])

interface MarketplaceDeepLinkProps {
  /** The workspace the Marketplace installs into. */
  readonly workspaceId: string
  /** Sets the Marketplace search box (a skill's link). */
  readonly onSearch: (query: string) => void
}

/** Once: the Marketplace search set to the skill's name, then the link stripped. */
function SkillLink({ name, onSearch }: { name: string; onSearch: (query: string) => void }) {
  const router = useRouter()
  const searchParams = useSearchParams()
  const applied = useRef(false)

  useEffect(() => {
    if (applied.current) return
    applied.current = true
    onSearch(name)
    router.replace(pathWithoutLink(searchParams))
  }, [name, onSearch, router, searchParams])
  return null
}

function LinkedModal({ link, workspaceId }: { link: MarketplaceLink; workspaceId: string }) {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [closed, setClosed] = useState(false)
  const Opener = OPENERS.get(link.tab)
  if (closed || !Opener) return null

  const close = () => {
    setClosed(true)
    router.replace(pathWithoutLink(searchParams))
  }
  return <Opener item={link.item} workspaceId={workspaceId} onClose={close} />
}

export function MarketplaceDeepLink({ workspaceId, onSearch }: MarketplaceDeepLinkProps) {
  const link = marketplaceLink(useSearchParams())
  if (link === null || link.tab === THEMES_TAB_VALUE) return null
  const key = `${link.tab}:${link.item}`
  if (link.tab === SKILLS_SUB_TAB) return <SkillLink key={key} name={link.item} onSearch={onSearch} />
  return <LinkedModal key={key} link={link} workspaceId={workspaceId} />
}
