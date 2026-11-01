/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the one seam the Marketplace homepage and its
 * shared tabs read their catalogue through.
 *
 * Signed in (and always in the local edition) it is today's member routes, unchanged
 * (member-source.ts). Signed out on SaaS it is the hosted edition's public reads, mapped to the same
 * shapes (public-source.ts), so the same page renders for a visitor. Nothing in the seam reads a
 * workspace: workspace reads (installed items, connected tools, a workspace's skills or plugins)
 * stay where they are, and run only signed in.
 */
import type { MarketplaceItem } from '@/hooks/use-marketplace-api'

import type { MarketplacePackage } from '../package-types'

export type SourceKind = 'member' | 'public'

/** The filters of the Marketplace's item list (agents and Playbooks). */
export interface MarketplaceItemFilters {
  readonly type?: string
  readonly category?: string
  readonly search?: string
  readonly featured?: boolean
  readonly limit?: number
  readonly offset?: number
}

/** The page header's counts: "100 items · 113 installs". */
export interface MarketplaceCounts {
  readonly totalItems: number
  readonly totalInstalls: number
}

export interface MarketplaceSource {
  readonly kind: SourceKind
  /** Agents and Playbooks, as the Agents tab lists them. */
  readonly items: (filters?: MarketplaceItemFilters) => Promise<MarketplaceItem[]>
  /** One agent (`agent`) or Playbook (`recipe`), for the item dialog. */
  readonly item: (id: number, type: string) => Promise<MarketplaceItem>
  /** The Featured hero and rail. */
  readonly featured: (limit: number) => Promise<MarketplaceItem[]>
  readonly counts: () => Promise<MarketplaceCounts>
  /** "Recommended for You" signed in; Popular signed out (the most installed besides `featured`). */
  readonly popular: (featured: readonly MarketplaceItem[]) => Promise<MarketplaceItem[]>
  readonly packages: () => Promise<MarketplacePackage[]>
}
