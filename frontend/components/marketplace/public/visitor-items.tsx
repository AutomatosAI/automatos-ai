'use client'

/**
 * PRD-270 US-270-08b (§5.7, K20) — one of the visitor's read-only tabs: a grid of a public read's
 * items (`GET /api/public/marketplace/{type}`), a page at a time with Show more, the Marketplace's
 * search applied here in the browser, and each item's detail with Add to my workspace. A shared
 * link's `item` opens its detail once that item is loaded. Loading and a refusal read as the Themes
 * tab's do (the server's own sentence); nothing here reads a workspace.
 */
import { useMemo, useState } from 'react'
import type { LucideIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { usePublicItems } from '@/hooks/use-public-marketplace'
import { filterPublicItems, pagedItems } from '@/lib/marketplace/public-items'
import type { PublicItem, PublicMarketplaceType } from '@/lib/marketplace/public-items'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { LOADING_ITEMS, LOADING_MORE, NO_ITEMS, NO_ITEM_MATCHES, SHOW_MORE } from './copy'
import { VisitorItemCard } from './visitor-item-card'
import { VisitorItemDetail } from './visitor-item-detail'

interface VisitorItemsTabProps {
  readonly type: PublicMarketplaceType
  /** The tab's value in a Marketplace link (`?tab=`). */
  readonly tab: string
  readonly icon: LucideIcon
  readonly searchQuery: string
  /** A shared link's item key, opened once it is loaded. */
  readonly item: string | null
}

const NONE: readonly PublicItem[] = []

export function VisitorItemsTab({ type, tab, icon, searchQuery, item }: VisitorItemsTabProps) {
  const read = usePublicItems(type)
  const all = useMemo(() => (read.data ? pagedItems(read.data.pages) : NONE), [read.data])
  const shown = useMemo(() => filterPublicItems(all, searchQuery), [all, searchQuery])
  const [openKey, setOpenKey] = useState<string | null>(item)
  const open = all.find((entry) => entry.key === openKey) ?? null

  if (read.isLoading) return <p className="text-sm text-muted-foreground">{LOADING_ITEMS}</p>
  if (read.isError) {
    return (
      <p className="text-sm text-destructive" role="alert">
        {themeErrorMessage(read.error)}
      </p>
    )
  }
  return (
    <div className="space-y-6" data-testid={`visitor-tab-${type}`}>
      {all.length === 0 && <p className="text-sm text-muted-foreground">{NO_ITEMS}</p>}
      {all.length > 0 && shown.length === 0 && <p className="text-sm text-muted-foreground">{NO_ITEM_MATCHES}</p>}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {shown.map((entry) => (
          <VisitorItemCard key={entry.key} item={entry} icon={icon} onDetails={setOpenKey} />
        ))}
      </div>
      {read.hasNextPage && (
        <Button variant="outline" disabled={read.isFetchingNextPage} onClick={() => void read.fetchNextPage()}>
          {read.isFetchingNextPage ? LOADING_MORE : SHOW_MORE}
        </Button>
      )}
      <VisitorItemDetail item={open} tab={tab} icon={icon} onClose={() => setOpenKey(null)} />
    </div>
  )
}
