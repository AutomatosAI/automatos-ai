'use client'

import { useState, useEffect } from 'react'
import { MarketplaceCard } from './marketplace-card'
import type { MarketplaceItem } from './marketplace-homepage'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/components/workspace-provider'
import { planChipForItem } from '@/lib/marketplace-exposure'

interface MarketplaceGridProps {
  items?: MarketplaceItem[]
  type?: string
  category?: string
  search?: string
  onItemClick: (itemId: number) => void
  isAdmin?: boolean
  onToggleFeatured?: (id: number) => void
}

export function MarketplaceGrid({
  items,
  type,
  category,
  search,
  onItemClick,
  isAdmin,
  onToggleFeatured,
}: MarketplaceGridProps) {
  const [gridItems, setGridItems] = useState<MarketplaceItem[]>(items || [])
  const [loading, setLoading] = useState(!items)
  // US-024: the full catalog stays visible to every tier; items beyond the
  // workspace's marketplace depth get a plan-label chip (no hiding — D5).
  const { workspace } = useWorkspace()
  const marketplaceDepth = workspace?.exposure?.marketplace_depth ?? 1

  useEffect(() => {
    if (items) {
      setGridItems(items)
      setLoading(false)
    } else {
      fetchItems()
    }
  }, [items, type, category, search])

  async function fetchItems() {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (type) params.append('type', type)
      if (category) params.append('category', category)
      if (search) params.append('search', search)
      params.append('limit', '50')

      const data = await apiClient.get(`/api/marketplace/items?${params}`)
      setGridItems(data)
    } catch (error) {
      console.error('Failed to fetch marketplace items:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {[...Array(8)].map((_, i) => (
          <div key={i} className="h-48 bg-secondary animate-pulse rounded-lg" />
        ))}
      </div>
    )
  }

  if (gridItems.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No items found. Try adjusting your search or filters.
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {gridItems.map((item) => {
        const planChip = planChipForItem(item, marketplaceDepth)
        return (
          <div key={item.id} className="relative">
            {planChip && (
              <span
                data-testid="plan-chip"
                className="absolute right-2 top-2 z-10 rounded-full bg-primary/15 px-2 py-0.5 text-xs font-medium text-primary ring-1 ring-primary/30"
                title={`Available on the ${planChip} plan`}
              >
                {planChip}
              </span>
            )}
            <MarketplaceCard
              item={item}
              onClick={() => onItemClick(item.id)}
              isAdmin={isAdmin}
              onToggleFeatured={onToggleFeatured}
            />
          </div>
        )
      })}
    </div>
  )
}
