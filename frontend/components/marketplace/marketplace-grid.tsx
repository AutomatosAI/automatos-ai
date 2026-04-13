'use client'

import { useState, useEffect } from 'react'
import { MarketplaceCard } from './marketplace-card'
import type { MarketplaceItem } from './marketplace-homepage'
import { apiClient } from '@/lib/api-client'

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
      {gridItems.map((item) => (
        <MarketplaceCard
          key={item.id}
          item={item}
          onClick={() => onItemClick(item.id)}
          isAdmin={isAdmin}
          onToggleFeatured={onToggleFeatured}
        />
      ))}
    </div>
  )
}
