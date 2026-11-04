'use client'

/**
 * One Marketplace page for everyone (PRD-270 §5.7) — a signed-out visitor's item detail for a
 * package, an application, an LLM, a plugin or a skill: the item as its tab's public read lists it
 * (React Query's copy, read once for the tab), in the public detail dialog (visitor-item-detail.tsx)
 * with Add to my workspace going to sign-in.
 *
 * Their member dialogs can't serve a visitor: the application's and the plugin's read member routes
 * when they open, and the package's and the LLM's add from inside. An agent or a Playbook opens the
 * member item dialog, which reads through the Marketplace seam (marketplace-item-modal.tsx).
 */
import { useQuery } from '@tanstack/react-query'
import { Puzzle } from 'lucide-react'

import type { PublicMarketplaceType } from '@/lib/marketplace/public-items'

import { publicQueries } from '../access/public-source'
import { TAB_ICONS } from '../category-tabs'
import { VisitorItemDetail } from './visitor-item-detail'

interface PublicItemDetailProps {
  readonly type: PublicMarketplaceType
  /** The item's `?tab=` value: its Add link opens it there after sign-in. */
  readonly tab: string
  readonly itemKey: string
  readonly onClose: () => void
}

export function PublicItemDetail({ type, tab, itemKey, onClose }: PublicItemDetailProps) {
  const read = useQuery(publicQueries.entries(type))
  const item = read.data?.find((entry) => entry.item.key === itemKey)?.item ?? null
  return <VisitorItemDetail item={item} tab={tab} icon={TAB_ICONS[tab] ?? Puzzle} onClose={onClose} />
}
