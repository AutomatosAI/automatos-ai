'use client'

import { useState, useEffect } from 'react'
import { Search, Download, TrendingUp, Grid3X3, Store } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { MarketplaceGrid } from './marketplace-grid'
import { MarketplaceItemModal } from './marketplace-item-modal'
import { MarketplaceToolsTab } from './marketplace-tools-tab'

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
  const [featuredItems, setFeaturedItems] = useState<MarketplaceItem[]>([])
  const [popularItems, setPopularItems] = useState<MarketplaceItem[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTab, setSelectedTab] = useState('all')
  const [selectedItem, setSelectedItem] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState({
    totalItems: 0,
    categories: 0,
    totalInstalls: 0,
    featuredCount: 0
  })

  useEffect(() => {
    fetchFeaturedItems()
    fetchPopularItems()
    fetchStats()
  }, [])

  async function fetchFeaturedItems() {
    try {
      const response = await fetch('/api/marketplace/featured?limit=8')
      if (response.ok) {
        const items = await response.json()
        setFeaturedItems(items)
      }
    } catch (error) {
      console.error('Failed to fetch featured items:', error)
    } finally {
      setLoading(false)
    }
  }

  async function fetchPopularItems() {
    try {
      const response = await fetch('/api/marketplace/items?limit=10')
      if (response.ok) {
        const items = await response.json()
        setPopularItems(items)
      }
    } catch (error) {
      console.error('Failed to fetch popular items:', error)
    }
  }

  async function fetchStats() {
    try {
      const response = await fetch('/api/marketplace/items?limit=1000')
      if (response.ok) {
        const items = await response.json()
        const categories = new Set(items.map((i: MarketplaceItem) => i.category).filter(Boolean))
        const totalInstalls = items.reduce((sum: number, i: MarketplaceItem) => sum + i.install_count, 0)
        const featuredCount = items.filter((i: MarketplaceItem) => i.is_featured).length

        setStats({
          totalItems: items.length,
          categories: categories.size,
          totalInstalls,
          featuredCount
        })
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Page Header - Matching Automatos Style */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">
          Community <span className="text-orange-500">Marketplace</span>
        </h1>
        <p className="text-gray-400">
          Discover, install, and manage tools to extend your AI agents' capabilities
        </p>
      </div>

      {/* Stats Cards - Matching Tools Page Style */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <Card className="bg-[#1a1a1a] border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{stats.totalItems}</div>
                <div className="text-sm text-gray-400">Total Items</div>
              </div>
              <Store className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-[#1a1a1a] border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{stats.categories}</div>
                <div className="text-sm text-gray-400">Categories</div>
              </div>
              <Grid3X3 className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-[#1a1a1a] border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{stats.featuredCount}</div>
                <div className="text-sm text-gray-400">Featured</div>
              </div>
              <Badge className="bg-orange-500/20 text-orange-500 border-orange-500/30">
                Curated
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-[#1a1a1a] border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">
                  {stats.totalInstalls >= 1000
                    ? `${(stats.totalInstalls / 1000).toFixed(1)}k`
                    : stats.totalInstalls}
                </div>
                <div className="text-sm text-gray-400">Total Installs</div>
              </div>
              <Download className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search Bar */}
      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
          <Input
            type="text"
            placeholder="Search marketplace..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-[#1a1a1a] border-gray-800 text-white placeholder:text-gray-500"
          />
        </div>
      </div>

      {/* Tab Navigation - Matching Automatos Style */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="flex-1">
        <TabsList className="bg-[#1a1a1a] border border-gray-800 p-1">
          <TabsTrigger
            value="all"
            className="data-[state=active]:bg-orange-500 data-[state=active]:text-white"
          >
            All Items
          </TabsTrigger>
          <TabsTrigger
            value="tools"
            className="data-[state=active]:bg-orange-500 data-[state=active]:text-white"
          >
            Tools
          </TabsTrigger>
          <TabsTrigger
            value="agents"
            className="data-[state=active]:bg-orange-500 data-[state=active]:text-white"
          >
            Agents
          </TabsTrigger>
          <TabsTrigger
            value="recipes"
            className="data-[state=active]:bg-orange-500 data-[state=active]:text-white"
          >
            Recipes
          </TabsTrigger>
          <TabsTrigger
            value="llms"
            className="data-[state=active]:bg-orange-500 data-[state=active]:text-white"
          >
            LLMs
          </TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="mt-6 space-y-8">
          {/* Featured Section */}
          {!loading && featuredItems.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <h2 className="text-xl font-bold text-white">Featured</h2>
                <Badge className="bg-orange-500/20 text-orange-500 border-orange-500/30">
                  Curated by Automatos
                </Badge>
              </div>
              <MarketplaceGrid
                items={featuredItems}
                onItemClick={setSelectedItem}
              />
            </div>
          )}

          {/* Most Popular Section */}
          {!loading && popularItems.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="h-5 w-5 text-orange-500" />
                <h2 className="text-xl font-bold text-white">Most Popular</h2>
              </div>
              <MarketplaceGrid
                items={popularItems.slice(0, 10)}
                onItemClick={setSelectedItem}
              />
            </div>
          )}
        </TabsContent>

        <TabsContent value="tools" className="mt-6">
          <MarketplaceToolsTab />
        </TabsContent>

        <TabsContent value="agents" className="mt-6">
          <MarketplaceGrid
            type="agent"
            search={searchQuery}
            onItemClick={setSelectedItem}
          />
        </TabsContent>

        <TabsContent value="recipes" className="mt-6">
          <MarketplaceGrid
            type="recipe"
            search={searchQuery}
            onItemClick={setSelectedItem}
          />
        </TabsContent>

        <TabsContent value="llms" className="mt-6">
          <MarketplaceGrid
            type="llm"
            search={searchQuery}
            onItemClick={setSelectedItem}
          />
        </TabsContent>
      </Tabs>

      {/* Item Detail Modal */}
      {selectedItem && (
        <MarketplaceItemModal
          itemId={selectedItem}
          onClose={() => setSelectedItem(null)}
        />
      )}
    </div>
  )
}
