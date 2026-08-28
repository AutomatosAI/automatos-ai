'use client'

import { useState, useEffect, useCallback } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
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
import { PageHeader, SearchInput } from '@/components/shared'
import { MarketplaceToolsTab } from './marketplace-tools-tab'
import { MarketplaceAgentsTab } from './marketplace-agents-tab'
import { MarketplaceLlmsTab } from './marketplace-llms-tab'
import { MarketplacePlaybooksTab } from './marketplace-playbooks-tab'
import { MarketplacePluginsTab } from './marketplace-plugins-tab'
import { MarketplaceSkillsTab } from './marketplace-skills-tab'
import { apiClient, getAdminWorkspaceOverride } from '@/lib/api-client'
import { useWorkspace } from '@/hooks/use-workspace'
import { useFeaturedItems, useToggleFeatured } from '@/hooks/use-marketplace-api'
import { useSystemRole } from '@/contexts/role-context'
import { MarketplaceItemModal } from './marketplace-item-modal'
import { AdminWorkspaceSwitcher } from '@/components/analytics/admin-workspace-switcher'
import { FeaturedShowcaseCard } from './featured-showcase-card'

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
          <MarketplacePluginsTab searchQuery={searchQuery} workspaceId={workspaceId} />
        </TabsContent>

        <TabsContent value="skills" className="mt-4">
          <MarketplaceSkillsTab searchQuery={searchQuery} workspaceId={workspaceId} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

// ── Featured Banner ─────────────────────────────────────────────────

function FeaturedBanner({ items, isAdmin, onItemClick }: { items: MarketplaceItem[]; isAdmin: boolean; onItemClick: (id: number) => void }) {
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
        {/* Hero card — animated Showcase */}
        <div className="lg:col-span-2">
          <FeaturedShowcaseCard
            item={hero}
            isAdmin={isAdmin}
            onItemClick={onItemClick}
            onToggleFeatured={() => toggleFeatured.mutate(hero.id)}
            toggleDisabled={toggleFeatured.isLoading}
          />
        </div>

        {/* Side rail */}
        <div className="flex flex-col gap-3">
          {rail.map((item) => (
            <Card key={item.id} className="border-border/40 bg-card/50 backdrop-blur-sm hover:border-primary/20 transition-colors cursor-pointer" onClick={() => onItemClick(item.id)}>
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
                      onClick={(e) => { e.stopPropagation(); toggleFeatured.mutate(item.id) }}
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

function RecommendationsRail({ items, onItemClick }: { items: MarketplaceItem[]; onItemClick: (id: number) => void }) {
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
            onClick={() => onItemClick(item.id)}
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

  // Admin override: lets a system admin install to a different workspace.
  // AdminWorkspaceSwitcher mutates a module-level override via setAdminWorkspaceOverride().
  // We mirror it in local state so React re-renders and the effective ID propagates to tabs.
  const [adminOverrideId, setAdminOverrideId] = useState<string | null>(() => {
    const v = getAdminWorkspaceOverride()
    // '__all__' is an analytics-only aggregation; don't use it for install targeting.
    return v && v !== '__all__' ? v : null
  })

  const handleAdminWorkspaceChange = useCallback(() => {
    const v = getAdminWorkspaceOverride()
    setAdminOverrideId(v && v !== '__all__' ? v : null)
  }, [])

  const effectiveWorkspaceId = adminOverrideId || workspaceId || ''

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

  const searchParams = useSearchParams()
  const router = useRouter()
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null)

  // Deep-link: ?id=N opens the item modal (used by Recommended strip on /assignments)
  useEffect(() => {
    const idParam = searchParams?.get('id')
    if (!idParam) return
    const parsed = parseInt(idParam, 10)
    if (!Number.isNaN(parsed)) setSelectedItemId(parsed)
  }, [searchParams])

  const handleCloseModal = useCallback(() => {
    setSelectedItemId(null)
    if (searchParams?.get('id')) {
      // Strip ?id= so refresh doesn't re-open
      router.replace('/marketplace')
    }
  }, [router, searchParams])

  const [recItems, setRecItems] = useState<MarketplaceItem[]>([])
  useEffect(() => {
    apiClient.get('/api/marketplace/items?limit=20')
      .then((items: MarketplaceItem[]) => {
        const featuredIds = new Set(((featuredItems as MarketplaceItem[] | undefined) || []).map((f: MarketplaceItem) => f.id))
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
      <PageHeader
        title="Community"
        titleAccent="Marketplace"
        eyebrow="Workspace · what to install"
        lede="Tools, agents, skills, and playbooks shared by the community. Install with one click; everything is sandboxed and reversible."
        actions={
          <>
            {isAdmin && (
              <AdminWorkspaceSwitcher onWorkspaceChange={handleAdminWorkspaceChange} />
            )}
            <Badge variant="outline" className="text-xs px-2.5 py-1 border-border">
              {stats.totalItems} items
            </Badge>
            <Badge variant="outline" className="text-xs px-2.5 py-1 border-border">
              <Download className="h-3 w-3 mr-1" />
              {stats.totalInstalls >= 1000
                ? `${(stats.totalInstalls / 1000).toFixed(1)}k`
                : stats.totalInstalls} installs
            </Badge>
          </>
        }
      />

      <div className="max-w-lg">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search the marketplace..."
        />
      </div>

      {/* ── Featured Banner ─────────────────────────────────── */}
      {!featuredLoading && (featuredItems as MarketplaceItem[] | undefined)?.length ? (
        <FeaturedBanner items={featuredItems as MarketplaceItem[]} isAdmin={isAdmin} onItemClick={setSelectedItemId} />
      ) : null}

      {/* ── Recommendations Rail ────────────────────────────── */}
      <RecommendationsRail items={recItems} onItemClick={setSelectedItemId} />

      {/* ── Category Tabs ───────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
          <TabsList className="bg-secondary/50">
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

          <TabsContent value="tools" className="mt-0">
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
            <CapabilitiesTab searchQuery={searchQuery} workspaceId={effectiveWorkspaceId} />
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Item detail modal for featured/recommended clicks */}
      {selectedItemId !== null && (
        <MarketplaceItemModal
          itemId={selectedItemId}
          onClose={handleCloseModal}
        />
      )}
    </div>
  )
}
