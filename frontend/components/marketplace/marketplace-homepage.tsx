'use client'

import { useState, useEffect } from 'react'
import { Search, Download, TrendingUp, Grid3X3, Store } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar, type StatItem } from '@/components/shared/stats-bar'
import { SearchInput } from '@/components/shared/search-input'
import { MarketplaceToolsTab } from './marketplace-tools-tab'
import { MarketplaceAgentsTab } from './marketplace-agents-tab'
import { MarketplaceLlmsTab } from './marketplace-llms-tab'
import { MarketplaceRecipesTab } from './marketplace-recipes-tab'
import { MarketplacePluginsTab } from './marketplace-plugins-tab'
import { apiClient } from '@/lib/api-client'

export interface MarketplaceItem {
  id: number
  type: string
  name: string
  description: string
  creator_name: string
  icon?: string
  category?: string
  tags: string[]
  install_count: number
  is_featured: boolean
  version: string
  metadata: Record<string, any>
  created_at: string
  updated_at: string
}

export function MarketplaceHomepage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTab, setSelectedTab] = useState('tools')
  const [stats, setStats] = useState({
    totalItems: 0,
    categories: 0,
    totalInstalls: 0,
    featuredCount: 0
  })

  useEffect(() => {
    fetchStats()
    apiClient.setCurrentPage('marketplace')
  }, [])

  async function fetchStats() {
    try {
      const items = await apiClient.get('/api/marketplace/items?limit=100')
      const categories = new Set(items.map((i: MarketplaceItem) => i.category).filter(Boolean))
      const totalInstalls = items.reduce((sum: number, i: MarketplaceItem) => sum + i.install_count, 0)
      const featuredCount = items.filter((i: MarketplaceItem) => i.is_featured).length

      setStats({
        totalItems: items.length,
        categories: categories.size,
        totalInstalls,
        featuredCount
      })
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <PageHeader
        title="Community"
        titleAccent="Marketplace"
        subtitle="Discover, install, and manage tools to extend your AI agents' capabilities"
      />

      {/* Stats Cards */}
      <StatsBar
        stats={[
          {
            label: 'Total Items',
            value: stats.totalItems.toString(),
            icon: Store,
            iconColor: 'text-primary',
          },
          {
            label: 'Categories',
            value: stats.categories.toString(),
            icon: Grid3X3,
            iconColor: 'text-[hsl(var(--info))]',
          },
          {
            label: 'Featured',
            value: stats.featuredCount.toString(),
            change: 'Curated',
            icon: TrendingUp,
            iconColor: 'text-[hsl(var(--warning))]',
          },
          {
            label: 'Total Installs',
            value: stats.totalInstalls >= 1000
              ? `${(stats.totalInstalls / 1000).toFixed(1)}k`
              : stats.totalInstalls.toString(),
            icon: Download,
            iconColor: 'text-[hsl(var(--success))]',
          },
        ] satisfies StatItem[]}
      />

      {/* Tabs and Search Bar - Full Width */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between w-full">
        <Tabs value={selectedTab} onValueChange={setSelectedTab}>
          <TabsList>
            <TabsTrigger value="tools">Applications</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="recipes">Recipes</TabsTrigger>
            <TabsTrigger value="llms">LLMs</TabsTrigger>
            <TabsTrigger value="plugins">Capabilities</TabsTrigger>
          </TabsList>
        </Tabs>

        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search marketplace..."
          className="w-full sm:flex-1"
        />
      </div>

      {/* Tab Content */}
      <Tabs value={selectedTab} className="space-y-6">
        <TabsContent value="tools" className="mt-0">
          <MarketplaceToolsTab searchQuery={searchQuery} />
        </TabsContent>

        <TabsContent value="agents" className="mt-0">
          <MarketplaceAgentsTab searchQuery={searchQuery} />
        </TabsContent>

        <TabsContent value="recipes" className="mt-0">
          <MarketplaceRecipesTab searchQuery={searchQuery} />
        </TabsContent>

        <TabsContent value="llms" className="mt-0">
          <MarketplaceLlmsTab searchQuery={searchQuery} />
        </TabsContent>

        <TabsContent value="plugins" className="mt-0">
          <MarketplacePluginsTab searchQuery={searchQuery} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
