/**
 * One Marketplace page for everyone (PRD-270 §5.7) — which component draws a tab, by who is looking.
 *
 * Agents, Playbooks and Packages are one component for everyone: they read through the Marketplace
 * seam (access/source.ts). Themes, Applications, LLMs, Plugins and Skills read workspace state and
 * member routes all through their member components, so a signed-out visitor gets the same tab drawn
 * from the public reads with the member tab's own cards (public/), in the same place with the same
 * props. The member page's deep link reads member routes too; the visitor's opens the public detail.
 */
import type { ComponentType } from 'react'

import { useMarketplaceAccess } from './access/marketplace-access'
import type { MarketplaceLink } from './deep-link/deep-link-params'
import { MarketplaceDeepLink } from './deep-link/marketplace-deep-link'
import { MarketplaceLlmsTab } from './marketplace-llms-tab'
import { MarketplacePluginsTab } from './marketplace-plugins-tab'
import { MarketplaceSkillsTab } from './marketplace-skills-tab'
import { MarketplaceToolsTab } from './marketplace-tools-tab'
import { PublicDeepLink } from './public/public-deep-link'
import { PublicLlmsTab } from './public/public-llms-tab'
import { PublicPluginsTab } from './public/public-plugins-tab'
import { PublicSkillsTab } from './public/public-skills-tab'
import { PublicToolsTab } from './public/public-tools-tab'
import { VisitorThemesTab } from './public/visitor-themes'
import { MarketplaceThemesTab } from './themes/tab'

interface SearchProps {
  searchQuery: string
}

interface WorkspaceTabProps extends SearchProps {
  workspaceId: string
}

export interface MarketplaceTabSet {
  readonly Themes: ComponentType<{ searchQuery?: string; link?: MarketplaceLink | null }>
  readonly Tools: ComponentType<SearchProps>
  readonly Llms: ComponentType<SearchProps>
  readonly Plugins: ComponentType<WorkspaceTabProps>
  readonly Skills: ComponentType<WorkspaceTabProps>
  readonly DeepLink: ComponentType<{ workspaceId: string; onSearch: (query: string) => void }>
}

const MEMBER_TABS: MarketplaceTabSet = {
  Themes: MarketplaceThemesTab,
  Tools: MarketplaceToolsTab,
  Llms: MarketplaceLlmsTab,
  Plugins: MarketplacePluginsTab,
  Skills: MarketplaceSkillsTab,
  DeepLink: MarketplaceDeepLink,
}

const VISITOR_TABS: MarketplaceTabSet = {
  Themes: VisitorThemesTab,
  Tools: PublicToolsTab,
  Llms: PublicLlmsTab,
  Plugins: PublicPluginsTab,
  Skills: PublicSkillsTab,
  DeepLink: PublicDeepLink,
}

/** The member's tabs, or the signed-out visitor's on SaaS. */
export function useMarketplaceTabs(): MarketplaceTabSet {
  return useMarketplaceAccess().signedIn ? MEMBER_TABS : VISITOR_TABS
}
