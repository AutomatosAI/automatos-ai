'use client'

import { useState, useEffect } from 'react'
import { Search, Download, TrendingUp, Grid3X3, Store } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { MarketplaceToolsTab } from './marketplace-tools-tab'
import { MarketplaceAgentsTab } from './marketplace-agents-tab'
import { MarketplaceLlmsTab } from './marketplace-llms-tab'
import { MarketplaceRecipesTab } from './marketplace-recipes-tab'
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
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold mb-2">
            Community <span className="gradient-text">Marketplace</span>
          </h1>
          <p className="text-muted-foreground mt-1">
            Discover, install, and manage tools to extend your AI agents' capabilities
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{stats.totalItems}</div>
                <div className="text-sm text-muted-foreground">Total Items</div>
              </div>
              <Store className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{stats.categories}</div>
                <div className="text-sm text-muted-foreground">Categories</div>
              </div>
              <Grid3X3 className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{stats.featuredCount}</div>
                <div className="text-sm text-muted-foreground">Featured</div>
              </div>
              <Badge className="bg-orange-500/20 text-orange-500 border-orange-500/30">
                Curated
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">
                  {stats.totalInstalls >= 1000
                    ? `${(stats.totalInstalls / 1000).toFixed(1)}k`
                    : stats.totalInstalls}
                </div>
                <div className="text-sm text-muted-foreground">Total Installs</div>
              </div>
              <Download className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs and Search Bar - Full Width */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between w-full">
        <Tabs value={selectedTab} onValueChange={setSelectedTab}>
          <TabsList className="bg-secondary border border-secondary">
            <TabsTrigger
              value="tools"
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Applications
            </TabsTrigger>
            <TabsTrigger
              value="agents"
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Agents
            </TabsTrigger>
            <TabsTrigger
              value="recipes"
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Recipes
            </TabsTrigger>
            <TabsTrigger
              value="llms"
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              LLMs
            </TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="relative w-full sm:flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            type="text"
            placeholder="Search marketplace..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50 w-full"
          />
        </div>
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
      </Tabs>
    </div>
  )
}
