/**
 * The Marketplace's tabs, in order, with their icons, and the tab a `?tab=` link opens: the one
 * Marketplace page (marketplace-homepage.tsx, PRD-270 §5.7) draws its strip from here, for a member
 * and a signed-out visitor alike. Which public read each tab draws is here too, for the visitor's
 * item detail and shared links (public/public-deep-link.tsx) and a link's share card (US-270-10).
 */
import { BookOpen, Bot, Cpu, Package, Palette, Puzzle, Store } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

import type { PublicMarketplaceType } from '@/lib/marketplace/public-items'

import { THEMES_TAB_LABEL, THEMES_TAB_VALUE } from './themes/listing-format'

export const TAB_ICONS: Record<string, LucideIcon> = {
  packages: Store,
  tools: Package,
  agents: Bot,
  recipes: BookOpen,
  llms: Cpu,
  capabilities: Puzzle,
  [THEMES_TAB_VALUE]: Palette,
}

export const CATEGORY_TABS = [
  // PRD-261: the Themes storefront leads the Marketplace, and the Marketplace opens on it (Gerard,
  // 30 Oct 2026; the tab named "Themes", not "Skins", D3/Q11). Packages follow (PRD-230).
  { value: THEMES_TAB_VALUE, label: THEMES_TAB_LABEL },
  { value: 'packages', label: 'Packages' },
  { value: 'tools', label: 'Applications' },
  { value: 'agents', label: 'Agents' },
  { value: 'recipes', label: 'Playbooks' },
  { value: 'llms', label: 'LLMs' },
  { value: 'capabilities', label: 'Capabilities' },
]

export const CAPABILITIES_TAB_VALUE = 'capabilities'
export const PLUGINS_SUB_TAB = 'plugins'
export const SKILLS_SUB_TAB = 'skills'
/** The Capabilities tab's sub-tabs, in order; a `?tab=` link may name one (PRD-270 K21, K22). */
export const CAPABILITY_SUB_TABS = [PLUGINS_SUB_TAB, SKILLS_SUB_TAB] as const

/** A tab that draws a public read (PRD-270 K10), by its value in a Marketplace link (`?tab=`). */
export interface PublicItemTab {
  readonly tab: string
  readonly type: PublicMarketplaceType
}

/**
 * The tabs of CATEGORY_TABS but Themes and Capabilities, and the public read each draws: a visitor's
 * shared link (public/public-deep-link.tsx) and a link's share card (lib/marketplace/share-card-read.ts).
 */
export const PUBLIC_ITEM_TABS: readonly PublicItemTab[] = [
  { tab: 'packages', type: 'packages' },
  { tab: 'tools', type: 'applications' },
  { tab: 'agents', type: 'agents' },
  { tab: 'recipes', type: 'playbooks' },
  { tab: 'llms', type: 'llms' },
]
/** The Capabilities sub-tabs, with their labels, and the public read each draws. */
export const PUBLIC_CAPABILITY_TABS: readonly (PublicItemTab & { readonly label: string })[] = [
  { tab: PLUGINS_SUB_TAB, type: 'plugins', label: 'Plugins' },
  { tab: SKILLS_SUB_TAB, type: 'skills', label: 'Skills' },
]

/** The public read a link's tab value draws, or null (Themes reads the catalogue). */
export function publicReadOfTab(tab: string): PublicMarketplaceType | null {
  return [...PUBLIC_ITEM_TABS, ...PUBLIC_CAPABILITY_TABS].find((entry) => entry.tab === tab)?.type ?? null
}

/** The tab a `?tab=` link opens, and the Capabilities sub-tab that tab opens on. */
export interface OpenedMarketplaceTab {
  readonly tab: string
  readonly capability: string
}

/**
 * The tab a `?tab=` link names, by its value or its label (`?tab=tools`, `?tab=playbooks`, as the
 * Tools dashboard and the Playbooks grid link here), else Themes, the tab the Marketplace opens on.
 */
function initialMarketplaceTab(param: string | null | undefined): string {
  const wanted = param?.trim().toLowerCase()
  const named = CATEGORY_TABS.find((tab) => tab.value === wanted || tab.label.toLowerCase() === wanted)
  return named?.value ?? THEMES_TAB_VALUE
}

/**
 * The tab, and the Capabilities sub-tab, a `?tab=` link opens: a Plugins or Skills link
 * (`?tab=plugins`, `?tab=skills`, as the visitor's Add links name them) opens Capabilities on it;
 * any other opens the tab it names on Plugins. The member page and the visitor's read it alike.
 */
export function openedMarketplaceTab(param: string | null | undefined): OpenedMarketplaceTab {
  const wanted = param?.trim().toLowerCase()
  const capability = CAPABILITY_SUB_TABS.find((value) => value === wanted)
  if (capability) return { tab: CAPABILITIES_TAB_VALUE, capability }
  return { tab: initialMarketplaceTab(param), capability: CAPABILITY_SUB_TABS[0] }
}
