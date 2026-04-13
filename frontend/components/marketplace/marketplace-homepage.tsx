'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Download,
  Star,
  Sparkles,
  Store,
  Package,
  Bot,
  BookOpen,
  Cpu,
  Puzzle,
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { SearchInput } from '@/components/shared/search-input'
import { MarketplaceToolsTab } from './marketplace-tools-tab'
import { MarketplaceAgentsTab } from './marketplace-agents-tab'
import { MarketplaceLlmsTab } from './marketplace-llms-tab'
import { MarketplacePlaybooksTab } from './marketplace-playbooks-tab'
import { MarketplacePluginsTab } from './marketplace-plugins-tab'
import { MarketplaceSkillsTab } from './marketplace-skills-tab'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/hooks/use-workspace'
import { useFeaturedItems, useToggleFeatured } from '@/hooks/use-marketplace-api'
import { useSystemRole } from '@/contexts/role-context'

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

function CapabilitiesTab({ searchQuery, workspaceId }: { searchQuery: string; workspaceId: string }) {
  const [subTab, setSubTab] = useState('plugins')

  return (
    <div className="space-y-4">
      <Tabs value={subTab} onValueChange={setSubTab}>
        <TabsList>
          <TabsTrigger value="plugins">Plugins</TabsTrigger>
          <TabsTrigger value="skills">Skills</TabsTrigger>
        </TabsList>

        <TabsContent value="plugins" className="mt-4">
          <MarketplacePluginsTab searchQuery={searchQuery} />
        </TabsContent>

        <TabsContent value="skills" className="mt-4">
          <MarketplaceSkillsTab searchQuery={searchQuery} workspaceId={workspaceId} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

// ── Featured Banner ─────────────────────────────────────────────────

function FeaturedBanner({ items, isAdmin }: { items: MarketplaceItem[]; isAdmin: boolean }) {
  const toggleFeatured = useToggleFeatured()

  if (!items || items.length === 0) return null

  // Pick the top featured item for the hero card, rest for the rail
  const hero = items[0]
  const rail = items.slice(1, 5)

  const formatInstalls = (n: number) => {
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
    return n.toString()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <div className="flex items-center gap-2 mb-3">
        <Star className="h-5 w-5 text-primary" />
        <h2 className="text-lg font-semibold">Featured</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Hero card */}
        <Card className="lg:col-span-2 overflow-hidden border-primary/20 bg-gradient-to-br from-primary/10 via-card/80 to-card/50 backdrop-blur-sm">
          <CardContent className="p-6 flex flex-col justify-between h-full min-h-[180px]">
            <div>
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-3">
                  {hero.icon && <span className="text-4xl">{hero.icon}</span>}
                  <div>
                    <h3 className="text-xl font-bold">{hero.name}</h3>
                    <p className="text-sm text-muted-foreground">by {hero.creator_name}</p>
                  </div>
                </div>
                {isAdmin && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-primary hover:text-primary/80"
                    onClick={() => toggleFeatured.mutate(hero.id)}
                    disabled={toggleFeatured.isLoading}
                  >
                    <Star className="h-4 w-4 fill-current" />
                  </Button>
                )}
              </div>
              <p className="text-sm text-muted-foreground line-clamp-2 mt-2">{hero.description}</p>
            </div>
            <div className="flex items-center gap-3 mt-4">
              <Badge variant="outline" className="text-xs">{hero.category || hero.type}</Badge>
              <span className="text-xs text-muted-foreground flex items-center gap-1">
                <Download className="h-3 w-3" /> {formatInstalls(hero.install_count)} installs
              </span>
              <span className="text-xs text-muted-foreground">v{hero.version}</span>
            </div>
          </CardContent>
        </Card>

        {/* Side rail */}
        <div className="flex flex-col gap-3">
          {rail.map((item) => (
            <Card key={item.id} className="border-border/40 bg-card/50 backdrop-blur-sm hover:border-primary/20 transition-colors">
              <CardContent className="p-3 flex items-center gap-3">
                {item.icon && <span className="text-2xl shrink-0">{item.icon}</span>}
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm truncate">{item.name}</p>
                  <p className="text-xs text-muted-foreground truncate">{item.description}</p>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  {isAdmin && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-primary/60 hover:text-primary"
                      onClick={() => toggleFeatured.mutate(item.id)}
                      disabled={toggleFeatured.isLoading}
                    >
                      <Star className="h-3 w-3 fill-current" />
                    </Button>
                  )}
                  <span className="text-xs text-muted-foreground">{formatInstalls(item.install_count)}</span>
                </div>
              </CardContent>
            </Card>
          ))}
          {rail.length === 0 && (
            <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground p-4">
              Feature more items to fill this section
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// ── Recommendations Rail ────────────────────────────────────────────

function RecommendationsRail({ items }: { items: MarketplaceItem[] }) {
  if (!items || items.length === 0) return null

  const formatInstalls = (n: number) => {
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
    return n.toString()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div className="flex items-center gap-2 mb-3">
        <Sparkles className="h-5 w-5 text-[hsl(var(--warning))]" />
        <h2 className="text-lg font-semibold">Recommended for You</h2>
      </div>

      <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin">
        {items.map((item) => (
          <Card
            key={item.id}
            className="min-w-[200px] max-w-[220px] shrink-0 border-border/40 bg-card/50 backdrop-blur-sm hover:border-primary/20 transition-colors cursor-pointer"
          >
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                {item.icon && <span className="text-2xl">{item.icon}</span>}
                <div className="min-w-0">
                  <p className="font-medium text-sm truncate">{item.name}</p>
                  <p className="text-xs text-muted-foreground">{item.creator_name}</p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground line-clamp-2 mb-2">{item.description}</p>
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Download className="h-3 w-3" /> {formatInstalls(item.install_count)}
                </span>
                {item.category && (
                  <Badge variant="outline" className="text-[10px] px-1.5 py-0">{item.category}</Badge>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </motion.div>
  )
}

// ── Main Page ───────────────────────────────────────────────────────

const TAB_ICONS: Record<string, any> = {
  tools: Package,
  agents: Bot,
  recipes: BookOpen,
  llms: Cpu,
  capabilities: Puzzle,
}

export function MarketplaceHomepage() {
  const { workspaceId } = useWorkspace()
  const { isAdmin } = useSystemRole()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTab, setSelectedTab] = useState('tools')
  const [stats, setStats] = useState({ totalItems: 0, totalInstalls: 0 })
  const { data: featuredItems, isLoading: featuredLoading } = useFeaturedItems(5)

  // Lightweight stats for pills (no StatsBar)
  useEffect(() => {
    apiClient.setCurrentPage('marketplace')
    apiClient.get('/api/marketplace/items?limit=100')
      .then((items: MarketplaceItem[]) => {
        setStats({
          totalItems: items.length,
          totalInstalls: items.reduce((sum, i) => sum + i.install_count, 0),
        })
      })
      .catch(() => {})
  }, [])

  const [recItems, setRecItems] = useState<MarketplaceItem[]>([])
  useEffect(() => {
    apiClient.get('/api/marketplace/items?limit=20')
      .then((items: MarketplaceItem[]) => {
        const featuredIds = new Set((featuredItems || []).map((f: MarketplaceItem) => f.id))
        const recs = items
          .filter((i: MarketplaceItem) => !featuredIds.has(i.id))
          .sort((a: MarketplaceItem, b: MarketplaceItem) => b.install_count - a.install_count)
          .slice(0, 6)
        setRecItems(recs)
      })
      .catch(() => {})
  }, [featuredItems])

  return (
    <div className="space-y-6">
      {/* ── Storefront Header ───────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col gap-4">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
            <div>
              <h1 className="text-2xl md:text-3xl font-bold">
                <Store className="inline-block h-7 w-7 mr-2 -mt-1 text-primary" />
                Community <span className="gradient-text">Marketplace</span>
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Discover tools, agents, and playbooks to supercharge your AI workspace
              </p>
            </div>

            {/* Stat pills + search */}
            <div className="flex items-center gap-3 shrink-0">
              <Badge variant="outline" className="text-xs px-2.5 py-1 border-border">
                {stats.totalItems} items
              </Badge>
              <Badge variant="outline" className="text-xs px-2.5 py-1 border-border">
                <Download className="h-3 w-3 mr-1" />
                {stats.totalInstalls >= 1000
                  ? `${(stats.totalInstalls / 1000).toFixed(1)}k`
                  : stats.totalInstalls} installs
              </Badge>
            </div>
          </div>

          <div data-tour="marketplace-search" className="max-w-lg">
            <SearchInput
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder="Search the marketplace..."
            />
          </div>
        </div>
      </motion.div>

      {/* ── Featured Banner ─────────────────────────────────── */}
      {!featuredLoading && (featuredItems as MarketplaceItem[] | undefined)?.length ? (
        <FeaturedBanner items={featuredItems as MarketplaceItem[]} isAdmin={isAdmin} />
      ) : null}

      {/* ── Recommendations Rail ────────────────────────────── */}
      <RecommendationsRail items={recItems} />

      {/* ── Category Tabs ───────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
          <TabsList data-tour="marketplace-tabs" className="bg-secondary/50">
            {[
              { value: 'tools', label: 'Applications' },
              { value: 'agents', label: 'Agents' },
              { value: 'recipes', label: 'Playbooks' },
              { value: 'llms', label: 'LLMs' },
              { value: 'capabilities', label: 'Capabilities' },
            ].map((tab) => {
              const Icon = TAB_ICONS[tab.value]
              return (
                <TabsTrigger key={tab.value} value={tab.value} className="flex items-center gap-1.5">
                  {Icon && <Icon className="h-4 w-4" />}
                  {tab.label}
                </TabsTrigger>
              )
            })}
          </TabsList>

          <TabsContent value="tools" className="mt-0" data-tour="marketplace-content">
            <MarketplaceToolsTab searchQuery={searchQuery} />
          </TabsContent>

          <TabsContent value="agents" className="mt-0">
            <MarketplaceAgentsTab searchQuery={searchQuery} />
          </TabsContent>

          <TabsContent value="recipes" className="mt-0">
            <MarketplacePlaybooksTab searchQuery={searchQuery} />
          </TabsContent>

          <TabsContent value="llms" className="mt-0">
            <MarketplaceLlmsTab searchQuery={searchQuery} />
          </TabsContent>

          <TabsContent value="capabilities" className="mt-0">
            <CapabilitiesTab searchQuery={searchQuery} workspaceId={workspaceId || ''} />
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  )
}
