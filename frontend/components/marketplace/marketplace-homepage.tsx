'use client'

import { useState, useEffect } from 'react'
import { Search } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { MarketplaceGrid } from './marketplace-grid'
import { MarketplaceItemModal } from './marketplace-item-modal'

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

  useEffect(() => {
    fetchFeaturedItems()
    fetchPopularItems()
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

  return (
    <div className="flex flex-col h-full">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-8 rounded-lg mb-6">
        <h1 className="text-4xl font-bold mb-4">Community Marketplace</h1>
        <p className="text-lg mb-6">
          Discover agents, recipes, and tools to automate your work
        </p>

        {/* Search Bar */}
        <div className="relative max-w-2xl">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <Input
            type="text"
            placeholder="Search marketplace..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-white text-gray-900"
          />
        </div>
      </div>

      {/* Tab Navigation */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="flex-1">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="tools">Tools</TabsTrigger>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="recipes">Recipes</TabsTrigger>
          <TabsTrigger value="llms">LLMs</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="mt-6 space-y-8">
          {/* Featured Section */}
          {!loading && featuredItems.length > 0 && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Featured</h2>
              <MarketplaceGrid
                items={featuredItems}
                onItemClick={setSelectedItem}
              />
            </div>
          )}

          {/* Most Popular Section */}
          {!loading && popularItems.length > 0 && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Most Popular</h2>
              <MarketplaceGrid
                items={popularItems.slice(0, 10)}
                onItemClick={setSelectedItem}
              />
            </div>
          )}
        </TabsContent>

        <TabsContent value="tools" className="mt-6">
          <MarketplaceGrid
            type="tool"
            search={searchQuery}
            onItemClick={setSelectedItem}
          />
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
