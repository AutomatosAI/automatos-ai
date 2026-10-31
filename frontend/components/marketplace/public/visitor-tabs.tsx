'use client'

/**
 * PRD-270 US-270-08b (§5.7, §8-Q9, K20) — the signed-out visitor's Marketplace tabs: the member
 * page's own strip (CATEGORY_TABS: Themes, Packages, Applications, Agents, Playbooks, LLMs,
 * Capabilities with Plugins and Skills), every tab read-only.
 *
 * Themes is the member storefront over the hosted catalogue (visitor-themes.tsx); every other tab is
 * a grid of its public read (visitor-items.tsx). A tab's content mounts only once it is chosen
 * (Radix), so each read is asked for when its tab opens. A `?tab=` link opens the tab it names,
 * as on the member page; a Plugins or Skills link (`?tab=plugins`, `?tab=skills`, the visitor's
 * Add links) opens Capabilities on it. A link's `item` is handed to the tab it names, once: after
 * the visitor changes tab (or Capabilities' sub-tab), no tab reopens it.
 */
import { useState } from 'react'
import { useSearchParams } from 'next/navigation'

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import type { PublicMarketplaceType } from '@/lib/marketplace/public-items'

import { CATEGORY_TABS, TAB_ICONS, initialMarketplaceTab } from '../category-tabs'
import { THEMES_TAB_VALUE } from '../themes/listing-format'
import { CAPABILITY_TABS_LABEL, VISITOR_TABS_LABEL } from './copy'
import { VisitorItemsTab } from './visitor-items'
import { VisitorThemesTab } from './visitor-themes'

interface ItemTab {
  /** Its value in a Marketplace link (`?tab=`). */
  readonly tab: string
  readonly type: PublicMarketplaceType
}

/** The tabs of CATEGORY_TABS that read a public item list, and what each reads. */
const ITEM_TABS: readonly ItemTab[] = [
  { tab: 'packages', type: 'packages' },
  { tab: 'tools', type: 'applications' },
  { tab: 'agents', type: 'agents' },
  { tab: 'recipes', type: 'playbooks' },
  { tab: 'llms', type: 'llms' },
]
const CAPABILITIES_TAB = 'capabilities'
const CAPABILITY_TABS: readonly (ItemTab & { readonly label: string })[] = [
  { tab: 'plugins', type: 'plugins', label: 'Plugins' },
  { tab: 'skills', type: 'skills', label: 'Skills' },
]

interface Opened {
  readonly tab: string
  readonly capability: string
}

/** The tab (and Capabilities sub-tab) a `?tab=` link opens. */
export function visitorTab(param: string | null | undefined): Opened {
  const wanted = param?.trim().toLowerCase()
  const capability = CAPABILITY_TABS.find((entry) => entry.tab === wanted)
  if (capability) return { tab: CAPABILITIES_TAB, capability: capability.tab }
  return { tab: initialMarketplaceTab(param), capability: CAPABILITY_TABS[0].tab }
}

interface Linked {
  /** The tab, or the Capabilities sub-tab, the link names. */
  readonly tab: string
  readonly item: string
}

function linkOf(opened: Opened, item: string | null | undefined): Linked | null {
  if (!item) return null
  return { tab: opened.tab === CAPABILITIES_TAB ? opened.capability : opened.tab, item }
}

interface TabProps {
  readonly searchQuery: string
  /** The link's item, for the tab the link names; null for every other tab. */
  readonly itemFor: (tab: string) => string | null
}

interface CapabilitiesProps extends TabProps {
  readonly initial: string
  /** The visitor chose another sub-tab. */
  readonly onSwitch: () => void
}

function CapabilitiesTabs({ initial, searchQuery, itemFor, onSwitch }: CapabilitiesProps) {
  const [capability, setCapability] = useState(initial)
  const choose = (next: string) => {
    setCapability(next)
    onSwitch()
  }
  return (
    <Tabs value={capability} onValueChange={choose} className="space-y-4">
      <TabsList aria-label={CAPABILITY_TABS_LABEL}>
        {CAPABILITY_TABS.map((entry) => (
          <TabsTrigger key={entry.tab} value={entry.tab}>
            {entry.label}
          </TabsTrigger>
        ))}
      </TabsList>
      {CAPABILITY_TABS.map((entry) => (
        <TabsContent key={entry.tab} value={entry.tab} className="mt-4">
          <VisitorItemsTab
            type={entry.type}
            tab={entry.tab}
            icon={TAB_ICONS[CAPABILITIES_TAB]}
            searchQuery={searchQuery}
            item={itemFor(entry.tab)}
          />
        </TabsContent>
      ))}
    </Tabs>
  )
}

export function VisitorTabs({ searchQuery }: { searchQuery: string }) {
  const searchParams = useSearchParams()
  const [opened] = useState(() => visitorTab(searchParams?.get('tab')))
  const [tab, setTab] = useState(opened.tab)
  const [link, setLink] = useState<Linked | null>(() => linkOf(opened, searchParams?.get('item')))
  const itemFor = (value: string) => (link?.tab === value ? link.item : null)
  const forgetLink = () => setLink(null)
  const choose = (next: string) => {
    setTab(next)
    forgetLink()
  }

  return (
    <Tabs value={tab} onValueChange={choose} className="space-y-6">
      <TabsList aria-label={VISITOR_TABS_LABEL} className="h-auto flex-wrap justify-start bg-secondary/50">
        {CATEGORY_TABS.map((entry) => {
          const Icon = TAB_ICONS[entry.value]
          return (
            <TabsTrigger key={entry.value} value={entry.value} className="flex items-center gap-1.5">
              {Icon && <Icon className="h-4 w-4" aria-hidden />}
              {entry.label}
            </TabsTrigger>
          )
        })}
      </TabsList>
      <TabsContent value={THEMES_TAB_VALUE} className="mt-0">
        <VisitorThemesTab searchQuery={searchQuery} item={itemFor(THEMES_TAB_VALUE)} />
      </TabsContent>
      {ITEM_TABS.map((entry) => (
        <TabsContent key={entry.tab} value={entry.tab} className="mt-0">
          <VisitorItemsTab
            type={entry.type}
            tab={entry.tab}
            icon={TAB_ICONS[entry.tab]}
            searchQuery={searchQuery}
            item={itemFor(entry.tab)}
          />
        </TabsContent>
      ))}
      <TabsContent value={CAPABILITIES_TAB} className="mt-0">
        <CapabilitiesTabs initial={opened.capability} searchQuery={searchQuery} itemFor={itemFor} onSwitch={forgetLink} />
      </TabsContent>
    </Tabs>
  )
}
