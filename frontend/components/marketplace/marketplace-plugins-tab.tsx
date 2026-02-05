'use client'

import { useState, useMemo, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search,
  Puzzle,
  CheckCircle,
  Loader2,
  Star,
  Download,
  Terminal,
  Zap,
  Coins,
  ArrowUpDown,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useToast } from '@/hooks/use-toast'
import { apiClient } from '@/lib/api-client'
import { MarketplacePluginDetailModal } from './marketplace-plugin-detail-modal'

// ===================================================================
// Types
// ===================================================================

interface PluginSummary {
  id: string
  slug: string
  name: string
  version: string
  description: string | null
  category_id: string | null
  category_name: string | null
  tags: string[]
  skills_count: number
  commands_count: number
  agents_count: number
  hooks_count: number
  token_estimate: number
  enable_count: number
  is_featured: boolean
  security_status: string | null
  original_author: string | null
  license: string | null
  created_at: string | null
}

interface PluginCategory {
  id: string
  slug: string
  name: string
  description: string | null
  icon: string | null
  sort_order: number
  parent_id: string | null
}

interface MarketplacePluginsTabProps {
  searchQuery: string
}

// ===================================================================
// Component
// ===================================================================

export function MarketplacePluginsTab({ searchQuery }: MarketplacePluginsTabProps) {
  const { toast } = useToast()

  // State
  const [plugins, setPlugins] = useState<PluginSummary[]>([])
  const [featuredPlugins, setFeaturedPlugins] = useState<PluginSummary[]>([])
  const [categories, setCategories] = useState<PluginCategory[]>([])
  const [enabledPluginIds, setEnabledPluginIds] = useState<Set<string>>(new Set())
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [sortBy, setSortBy] = useState('popular')
  const [isLoading, setIsLoading] = useState(true)
  const [enablingId, setEnablingId] = useState<string | null>(null)
  const [selectedPluginId, setSelectedPluginId] = useState<string | null>(null)

  // Get workspace ID from localStorage (same pattern as api-client.ts)
  const getWorkspaceId = useCallback((): string | null => {
    if (typeof window === 'undefined') return null
    return localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
  }, [])

  // Fetch categories
  useEffect(() => {
    async function fetchCategories() {
      try {
        const data = await apiClient.get('/api/marketplace/plugins/categories')
        setCategories(data as PluginCategory[])
      } catch (err) {
        console.error('Failed to fetch plugin categories:', err)
      }
    }
    fetchCategories()
  }, [])

  // Fetch enabled plugins for workspace
  useEffect(() => {
    async function fetchEnabledPlugins() {
      const wsId = getWorkspaceId()
      if (!wsId) return

      try {
        const data: any = await apiClient.get(`/api/workspaces/${wsId}/plugins`)
        const ids = new Set<string>(
          (data?.items || []).map((item: any) => item.plugin_id)
        )
        setEnabledPluginIds(ids)
      } catch (err) {
        console.error('Failed to fetch enabled plugins:', err)
      }
    }
    fetchEnabledPlugins()
  }, [getWorkspaceId])

  // Fetch plugins
  useEffect(() => {
    async function fetchPlugins() {
      setIsLoading(true)
      try {
        const params: Record<string, any> = {
          sort: sortBy,
          limit: 100,
        }
        if (selectedCategory !== 'all') {
          params.category = selectedCategory
        }

        const data: any = await apiClient.get('/api/marketplace/plugins', params)
        const items: PluginSummary[] = data?.items || []

        // Separate featured plugins
        const featured = items.filter((p) => p.is_featured)
        setFeaturedPlugins(featured)
        setPlugins(items)
      } catch (err) {
        console.error('Failed to fetch plugins:', err)
      } finally {
        setIsLoading(false)
      }
    }
    fetchPlugins()
  }, [selectedCategory, sortBy])

  // Client-side search filter
  const filteredPlugins = useMemo(() => {
    if (!searchQuery.trim()) return plugins

    const query = searchQuery.toLowerCase()
    return plugins.filter(
      (p) =>
        p.name.toLowerCase().includes(query) ||
        p.description?.toLowerCase().includes(query) ||
        p.slug.toLowerCase().includes(query) ||
        (p.tags || []).some((t) => t.toLowerCase().includes(query))
    )
  }, [plugins, searchQuery])

  // Enable plugin for workspace
  const handleEnable = async (pluginId: string, pluginName: string) => {
    const wsId = getWorkspaceId()
    if (!wsId) {
      toast({
        title: 'No workspace',
        description: 'No active workspace found. Please select a workspace.',
        variant: 'destructive',
      })
      return
    }

    setEnablingId(pluginId)
    try {
      await apiClient.post(`/api/workspaces/${wsId}/plugins`, {
        plugin_id: pluginId,
      })

      // Update local state
      setEnabledPluginIds((prev) => new Set([...prev, pluginId]))

      // Update enable_count in local data
      setPlugins((prev) =>
        prev.map((p) =>
          p.id === pluginId ? { ...p, enable_count: p.enable_count + 1 } : p
        )
      )
      setFeaturedPlugins((prev) =>
        prev.map((p) =>
          p.id === pluginId ? { ...p, enable_count: p.enable_count + 1 } : p
        )
      )

      toast({
        title: 'Plugin Enabled',
        description: `${pluginName} has been enabled for your workspace.`,
      })
    } catch (err: any) {
      const msg = err?.message || 'Failed to enable plugin'
      if (msg.includes('already enabled') || msg.includes('409')) {
        toast({
          title: 'Already Enabled',
          description: `${pluginName} is already enabled for your workspace.`,
        })
        setEnabledPluginIds((prev) => new Set([...prev, pluginId]))
      } else {
        toast({
          title: 'Error',
          description: msg,
          variant: 'destructive',
        })
      }
    } finally {
      setEnablingId(null)
    }
  }

  // Disable plugin for workspace
  const handleDisable = async (pluginId: string, pluginName: string) => {
    const wsId = getWorkspaceId()
    if (!wsId) return

    try {
      await apiClient.delete(`/api/workspaces/${wsId}/plugins/${pluginId}`)

      // Update local state
      setEnabledPluginIds((prev) => {
        const next = new Set(prev)
        next.delete(pluginId)
        return next
      })

      // Decrement enable_count in local data
      setPlugins((prev) =>
        prev.map((p) =>
          p.id === pluginId ? { ...p, enable_count: Math.max(0, p.enable_count - 1) } : p
        )
      )
      setFeaturedPlugins((prev) =>
        prev.map((p) =>
          p.id === pluginId ? { ...p, enable_count: Math.max(0, p.enable_count - 1) } : p
        )
      )

      toast({
        title: 'Plugin Disabled',
        description: `${pluginName} has been disabled for your workspace.`,
      })
    } catch (err: any) {
      toast({
        title: 'Error',
        description: err?.message || 'Failed to disable plugin',
        variant: 'destructive',
      })
    }
  }

  // Get selected plugin name for disable handler
  const selectedPlugin = selectedPluginId
    ? plugins.find((p) => p.id === selectedPluginId) || featuredPlugins.find((p) => p.id === selectedPluginId)
    : null

  // Loading skeleton
  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <Card key={i} className="glass-card animate-pulse">
              <CardHeader className="pb-2">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-secondary/50 rounded-lg" />
                  <div className="space-y-2">
                    <div className="h-4 w-24 bg-secondary/50 rounded" />
                    <div className="h-3 w-32 bg-secondary/50 rounded" />
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-8 w-full bg-secondary/50 rounded mt-2" />
                <div className="h-6 w-20 bg-secondary/50 rounded mt-3" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold">Plugin Marketplace</h3>
          <p className="text-sm text-muted-foreground">
            Extend your agents with community-curated skills, commands, and hooks
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-green-400 border-green-500/30">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
            {enabledPluginIds.size} Enabled
          </Badge>
          <Badge variant="outline" className="text-blue-400 border-blue-500/30">
            {plugins.length} Available
          </Badge>
        </div>
      </div>

      {/* Featured Plugins Section */}
      {featuredPlugins.length > 0 && selectedCategory === 'all' && !searchQuery && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Star className="w-4 h-4 text-orange-400" />
            <h4 className="text-sm font-semibold text-orange-400 uppercase tracking-wider">Featured Plugins</h4>
          </div>
          <div className="flex gap-4 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent">
            {featuredPlugins.map((plugin) => (
              <FeaturedPluginCard
                key={plugin.id}
                plugin={plugin}
                isEnabled={enabledPluginIds.has(plugin.id)}
                isEnabling={enablingId === plugin.id}
                onEnable={() => handleEnable(plugin.id, plugin.name)}
                onClick={() => setSelectedPluginId(plugin.id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Category Filter Pills + Sort */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent flex-1">
          <Button
            variant={selectedCategory === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory('all')}
            className={`whitespace-nowrap flex-shrink-0 ${
              selectedCategory === 'all'
                ? 'bg-primary hover:bg-primary/90 text-primary-foreground'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }`}
          >
            All Plugins
          </Button>
          {categories.map((cat) => (
            <Button
              key={cat.id}
              variant={selectedCategory === cat.slug ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedCategory(cat.slug)}
              className={`whitespace-nowrap flex-shrink-0 ${
                selectedCategory === cat.slug
                  ? 'bg-primary hover:bg-primary/90 text-primary-foreground'
                  : 'border-secondary text-muted-foreground hover:bg-secondary'
              }`}
            >
              {cat.icon && <span className="mr-1">{cat.icon}</span>}
              {cat.name}
            </Button>
          ))}
        </div>

        <Select value={sortBy} onValueChange={setSortBy}>
          <SelectTrigger className="w-[140px] bg-secondary/50 border-secondary flex-shrink-0">
            <ArrowUpDown className="w-3 h-3 mr-1" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="popular">Popular</SelectItem>
            <SelectItem value="newest">Newest</SelectItem>
            <SelectItem value="name">Name</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Plugins Grid */}
      {filteredPlugins.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 rounded-lg bg-secondary/30 flex items-center justify-center mx-auto mb-4">
            <Puzzle className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold mb-2">No plugins found</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery
              ? `No plugins match "${searchQuery}"`
              : 'No plugins available in this category'}
          </p>
          <Button
            variant="outline"
            onClick={() => {
              setSelectedCategory('all')
            }}
          >
            Clear Filters
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <AnimatePresence>
            {filteredPlugins.map((plugin, index) => (
              <PluginCard
                key={plugin.id}
                plugin={plugin}
                index={index}
                isEnabled={enabledPluginIds.has(plugin.id)}
                isEnabling={enablingId === plugin.id}
                onEnable={() => handleEnable(plugin.id, plugin.name)}
                onClick={() => setSelectedPluginId(plugin.id)}
              />
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Plugin Detail Modal */}
      <MarketplacePluginDetailModal
        open={!!selectedPluginId}
        pluginId={selectedPluginId}
        onClose={() => setSelectedPluginId(null)}
        isEnabled={selectedPluginId ? enabledPluginIds.has(selectedPluginId) : false}
        isEnabling={selectedPluginId ? enablingId === selectedPluginId : false}
        onEnable={() => {
          if (selectedPluginId && selectedPlugin) {
            handleEnable(selectedPluginId, selectedPlugin.name)
          }
        }}
        onDisable={() => {
          if (selectedPluginId && selectedPlugin) {
            handleDisable(selectedPluginId, selectedPlugin.name)
            setSelectedPluginId(null)
          }
        }}
      />
    </div>
  )
}

// ===================================================================
// Featured Plugin Card (Horizontal scroll row)
// ===================================================================

interface FeaturedPluginCardProps {
  plugin: PluginSummary
  isEnabled: boolean
  isEnabling: boolean
  onEnable: () => void
  onClick: () => void
}

function FeaturedPluginCard({
  plugin,
  isEnabled,
  isEnabling,
  onEnable,
  onClick,
}: FeaturedPluginCardProps) {
  return (
    <Card
      className="min-w-[300px] max-w-[340px] flex-shrink-0 bg-gradient-to-br from-orange-500/10 to-secondary/30 border-orange-500/20 hover:border-orange-500/40 transition-all duration-200 cursor-pointer hover:shadow-lg hover:shadow-orange-500/10"
      onClick={onClick}
    >
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h4 className="font-semibold text-white truncate">{plugin.name}</h4>
              <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/30 text-[10px] h-5 flex-shrink-0">
                <Star className="w-2.5 h-2.5 mr-0.5" />
                Featured
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">v{plugin.version}</p>
          </div>
        </div>

        <p className="text-sm text-gray-400 line-clamp-2">
          {plugin.description || 'No description available'}
        </p>

        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Terminal className="w-3 h-3" />
            <span>{plugin.skills_count} skills</span>
          </div>
          <div className="flex items-center gap-1">
            <Coins className="w-3 h-3" />
            <span>~{plugin.token_estimate} tokens</span>
          </div>
          <div className="flex items-center gap-1">
            <Download className="w-3 h-3" />
            <span>{plugin.enable_count}</span>
          </div>
        </div>

        <div
          onClick={(e) => e.stopPropagation()}
        >
          {isEnabled ? (
            <Button
              size="sm"
              variant="secondary"
              className="w-full bg-secondary/50 border border-white/10"
              disabled
            >
              <CheckCircle className="w-3 h-3 mr-1" />
              Enabled
            </Button>
          ) : (
            <Button
              size="sm"
              className="w-full bg-orange-600 hover:bg-orange-700 text-white"
              onClick={onEnable}
              disabled={isEnabling}
            >
              {isEnabling ? (
                <>
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  Enabling...
                </>
              ) : (
                'Enable for Workspace'
              )}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// ===================================================================
// Plugin Card (Grid item)
// ===================================================================

interface PluginCardProps {
  plugin: PluginSummary
  index: number
  isEnabled: boolean
  isEnabling: boolean
  onEnable: () => void
  onClick: () => void
}

function PluginCard({
  plugin,
  index,
  isEnabled,
  isEnabling,
  onEnable,
  onClick,
}: PluginCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ delay: Math.min(index * 0.05, 0.5) }}
    >
      <Card
        className="glass-card card-glow hover:border-orange-500/30 hover:shadow-lg hover:shadow-orange-500/20 transition-all duration-300 cursor-pointer"
        onClick={onClick}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-white line-clamp-1">
                  {plugin.name}
                </h3>
                {plugin.is_featured && (
                  <Star className="w-3.5 h-3.5 text-orange-400 flex-shrink-0" />
                )}
              </div>
              <p className="text-xs text-gray-500">
                v{plugin.version}
                {plugin.original_author && (
                  <> &middot; {plugin.original_author}</>
                )}
              </p>
            </div>
            {plugin.security_status === 'safe' && (
              <Badge
                variant="outline"
                className="text-[10px] h-5 border-green-500/30 text-green-400 flex-shrink-0"
              >
                Verified
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          <p className="text-sm text-gray-400 line-clamp-2">
            {plugin.description || 'No description available'}
          </p>

          {/* Category + Tags */}
          <div className="flex flex-wrap gap-1.5">
            {plugin.category_name && (
              <Badge variant="outline" className="text-[10px] h-5 border-gray-700 text-gray-400">
                {plugin.category_name}
              </Badge>
            )}
            {(plugin.tags || []).slice(0, 2).map((tag) => (
              <Badge key={tag} variant="outline" className="text-[10px] h-5 border-gray-700 text-gray-500">
                {tag}
              </Badge>
            ))}
          </div>

          {/* Stats Row */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-1" title="Skills">
              <Terminal className="w-3 h-3" />
              <span>{plugin.skills_count}</span>
            </div>
            <div className="flex items-center gap-1" title="Commands">
              <Zap className="w-3 h-3" />
              <span>{plugin.commands_count}</span>
            </div>
            <div className="flex items-center gap-1" title="Token estimate">
              <Coins className="w-3 h-3" />
              <span>~{plugin.token_estimate}</span>
            </div>
            <div className="flex items-center gap-1 ml-auto" title="Enabled by workspaces">
              <Download className="w-3 h-3" />
              <span>{plugin.enable_count}</span>
            </div>
          </div>

          <Separator />

          {/* Action Section */}
          <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
            <Button
              variant="outline"
              size="sm"
              className="flex-1 border-gray-700 text-gray-300 hover:bg-gray-800"
              onClick={onClick}
            >
              Details
            </Button>

            {isEnabled ? (
              <Button
                size="sm"
                variant="secondary"
                className="flex-1 bg-secondary/50 hover:bg-secondary border border-white/10"
                disabled
              >
                <CheckCircle className="w-3 h-3 mr-1" />
                Enabled
              </Button>
            ) : (
              <Button
                size="sm"
                className="flex-1 bg-orange-600 hover:bg-orange-700 text-white"
                onClick={onEnable}
                disabled={isEnabling}
              >
                {isEnabling ? (
                  <>
                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                    Enabling...
                  </>
                ) : (
                  'Enable'
                )}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
