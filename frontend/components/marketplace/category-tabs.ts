/**
 * The Marketplace's tabs, in order, with their icons, and the tab a `?tab=` link opens: the member
 * page (marketplace-homepage.tsx) and the signed-out visitor's (public/visitor-tabs.tsx, PRD-270
 * K20) draw the same strip from here, so the visitor's page never imports the member tabs.
 */
import { BookOpen, Bot, Cpu, Package, Palette, Puzzle, Store } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

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

/**
 * The tab a `?tab=` link names, by its value or its label (`?tab=tools`, `?tab=playbooks`, as the
 * Tools dashboard and the Playbooks grid link here), else Themes, the tab the Marketplace opens on.
 */
export function initialMarketplaceTab(param: string | null | undefined): string {
  const wanted = param?.trim().toLowerCase()
  const named = CATEGORY_TABS.find((tab) => tab.value === wanted || tab.label.toLowerCase() === wanted)
  return named?.value ?? THEMES_TAB_VALUE
}
