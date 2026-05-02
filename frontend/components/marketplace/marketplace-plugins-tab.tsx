'use client'

import { useState, useMemo, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Sparkles,
  CheckCircle,
  Loader2,
  Star,
  Download,
  Terminal,
  Zap,
  Cpu,
  ArrowUpDown,
  MoreVertical,
  Eye,
  Github,
  Upload,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { StatusBadge } from '@/components/shared'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Separator } from '@/components/ui/separator'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useToast } from '@/hooks/use-toast'
import { PremiumIcon } from '@/components/shared'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { apiClient } from '@/lib/api-client'
import { useUser } from '@clerk/nextjs'
import { MarketplacePluginDetailModal } from './marketplace-plugin-detail-modal'
import { GitHubImportModal } from './github-import-modal'

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
  approval_status: string | null
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
  workspaceId?: string
}

// ===================================================================
// Component
// ===================================================================

export function MarketplacePluginsTab({ searchQuery, workspaceId }: MarketplacePluginsTabProps) {
  const { toast } = useToast()
  const { user } = useUser()
  const [viewMode, setViewMode] = useViewMode('mp-plugins')
  const { data: iconMappings = {} } = useSystemIcons()
  const globalPluginIcon = iconMappings['global_plugin'] || null

  // Admin check (same pattern as agents tab)
  const isAdmin = user?.emailAddresses?.[0]?.emailAddress?.includes('automatos.app') || false

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
  const [approvingId, setApprovingId] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [deactivatingId, setDeactivatingId] = useState<string | null>(null)
  const [showImportModal, setShowImportModal] = useState(false)

  // Workspace ID: prop (e.g. admin override) wins, else localStorage fallback.
  const getWorkspaceId = useCallback((): string | null => {
    if (workspaceId) return workspaceId
    if (typeof window === 'undefined') return null
    return localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
  }, [workspaceId])

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

  // Fetch plugins (+ pending for admins)
  const fetchAllPlugins = useCallback(async () => {
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
      let items: PluginSummary[] = data?.items || []

      // Admin: also fetch pending plugins and merge
      if (isAdmin) {
        try {
          const pendingData: any = await apiClient.get('/api/admin/plugins/pending', { limit: 100 })
          const pendingItems: any[] = pendingData?.items || []
          const existingIds = new Set(items.map((p) => p.id))
          for (const p of pendingItems) {
            if (!existingIds.has(p.id)) {
              items.push({
                id: p.id,
                slug: p.slug,
                name: p.name,
                version: p.version,
                description: p.description,
                category_id: null,
                category_name: null,
                tags: [],
                skills_count: 0,
                commands_count: 0,
                agents_count: 0,
                hooks_count: 0,
                token_estimate: 0,
                enable_count: 0,
                is_featured: false,
                security_status: p.security_status,
                approval_status: 'pending',
                original_author: null,
                license: null,
                created_at: p.created_at,
              })
            }
          }
        } catch {
          // Non-fatal: admin pending fetch failed
        }
      }

      // Separate featured plugins
      const featured = items.filter((p) => p.is_featured)
      setFeaturedPlugins(featured)
      setPlugins(items)
    } catch (err) {
      console.error('Failed to fetch plugins:', err)
    } finally {
      setIsLoading(false)
    }
  }, [selectedCategory, sortBy, isAdmin])

  useEffect(() => {
    fetchAllPlugins()
  }, [fetchAllPlugins])

  // Admin action handlers
  const handleApprove = async (pluginId: string) => {
    setApprovingId(pluginId)
    try {
      await apiClient.post(`/api/admin/plugins/${pluginId}/approve`)
      toast({ title: 'Capability approved and published to marketplace!' })
      fetchAllPlugins()
    } catch (error: any) {
      toast({
        title: 'Failed to approve capability',
        description: error?.message || 'An error occurred',
        variant: 'destructive',
      })
    } finally {
      setApprovingId(null)
    }
  }

  const handleDeactivate = async (pluginId: string) => {
    setDeactivatingId(pluginId)
    try {
      await apiClient.post(`/api/admin/plugins/${pluginId}/deactivate`)
      toast({ title: 'Capability deactivated' })
      fetchAllPlugins()
    } catch (error: any) {
      toast({
        title: 'Failed to deactivate capability',
        description: error?.message || 'An error occurred',
        variant: 'destructive',
      })
    } finally {
      setDeactivatingId(null)
    }
  }

  const handleDelete = async (pluginId: string) => {
    if (!confirm('Are you sure you want to permanently delete this capability? This cannot be undone.')) {
      return
    }
    setDeletingId(pluginId)
    try {
      await apiClient.delete(`/api/admin/plugins/${pluginId}`)
      toast({ title: 'Capability deleted permanently' })
      fetchAllPlugins()
    } catch (error: any) {
      toast({
        title: 'Failed to delete capability',
        description: error?.message || 'An error occurred',
        variant: 'destructive',
      })
    } finally {
      setDeletingId(null)
    }
  }

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
        title: 'Capability Enabled',
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
        title: 'Capability Disabled',
        description: `${pluginName} has been disabled for your workspace.`,
      })
    } catch (err: any) {
      toast({
        title: 'Error',
        description: err?.message || 'Failed to disable capability',
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
        {viewMode === 'list' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-16 glass-card animate-pulse rounded-xl" />
            ))}
          </div>
        ) : (
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
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold">Capabilities Marketplace</h3>
          <p className="text-sm text-muted-foreground">
            Teach your agents new skills with community-curated capabilities
          </p>
        </div>
        <div className="flex items-center gap-3">
          {isAdmin && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowImportModal(true)}
              className="gap-2"
            >
              <Github className="w-4 h-4" />
              <Upload className="w-3.5 h-3.5" />
              Import from GitHub
            </Button>
          )}
          <StatusBadge size="sm" status="success" dot>
            {enabledPluginIds.size} Enabled
          </StatusBadge>
          <StatusBadge size="sm" status="info">
            {plugins.length} Available
          </StatusBadge>
        </div>
      </div>

      {/* Featured Plugins Section */}
      {featuredPlugins.length > 0 && selectedCategory === 'all' && !searchQuery && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Star className="w-4 h-4 text-primary" />
            <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">Featured Capabilities</h4>
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
                globalIcon={globalPluginIcon}
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
                ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }`}
          >
            All Capabilities
          </Button>
          {categories.map((cat) => (
            <Button
              key={cat.id}
              variant={selectedCategory === cat.slug ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedCategory(cat.slug)}
              className={`whitespace-nowrap flex-shrink-0 ${
                selectedCategory === cat.slug
                  ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                  : 'border-secondary text-muted-foreground hover:bg-secondary'
              }`}
            >
              {cat.icon && <span className="mr-1">{cat.icon}</span>}
              {cat.name}
            </Button>
          ))}
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-[140px] bg-secondary/50 border-secondary">
              <ArrowUpDown className="w-3 h-3 mr-1" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="popular">Popular</SelectItem>
              <SelectItem value="newest">Newest</SelectItem>
              <SelectItem value="name">Name</SelectItem>
            </SelectContent>
          </Select>
          <ViewToggle value={viewMode} onChange={setViewMode} />
        </div>
      </div>

      {/* Plugins Grid */}
      {filteredPlugins.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 rounded-lg bg-secondary/30 flex items-center justify-center mx-auto mb-4">
            {globalPluginIcon ? (
              <PremiumIcon name={globalPluginIcon} size={32} />
            ) : (
              <Sparkles className="w-8 h-8 text-muted-foreground" />
            )}
          </div>
          <h3 className="text-lg font-semibold mb-2">No capabilities found</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery
              ? `No capabilities match "${searchQuery}"`
              : 'No capabilities available in this category'}
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
      ) : viewMode === 'list' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {filteredPlugins.map((plugin) => {
            const isEnabled = enabledPluginIds.has(plugin.id)
            return (
              <Card
                key={plugin.id}
                className="glass-card card-glow hover:border-primary/20 transition-all cursor-pointer"
                onClick={() => setSelectedPluginId(plugin.id)}
              >
                <CardContent className="p-3">
                  <div className="flex items-center gap-3">
                    {globalPluginIcon ? (
                      <PremiumIcon name={globalPluginIcon} size={36} className="shrink-0" />
                    ) : (
                      <Sparkles className="w-9 h-9 text-primary shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-sm truncate">{plugin.name}</span>
                        {plugin.category_name && (
                          <Badge variant="outline" className="text-[10px] shrink-0">{plugin.category_name}</Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                        {plugin.skills_count > 0 && <span>{plugin.skills_count} skills</span>}
                        {plugin.skills_count > 0 && plugin.enable_count > 0 && <span>&middot;</span>}
                        <span>{plugin.enable_count} enabled</span>
                      </div>
                    </div>
                    {isEnabled ? (
                      <Badge variant="secondary" className="text-xs shrink-0">
                        <CheckCircle className="w-3 h-3 mr-1" />
                        On
                      </Badge>
                    ) : (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0 shrink-0"
                        onClick={(e) => { e.stopPropagation(); handleEnable(plugin.id, plugin.name) }}
                        disabled={enablingId === plugin.id}
                      >
                        {enablingId === plugin.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            )
          })}
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
                isAdmin={isAdmin}
                isApproving={approvingId === plugin.id}
                isDeleting={deletingId === plugin.id}
                isDeactivating={deactivatingId === plugin.id}
                onApprove={() => handleApprove(plugin.id)}
                onDeactivate={() => handleDeactivate(plugin.id)}
                onDelete={() => handleDelete(plugin.id)}
                globalIcon={globalPluginIcon}
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
        isAdmin={isAdmin}
        onApproved={fetchAllPlugins}
      />

      {/* GitHub Import Modal (admin only) */}
      {isAdmin && (
        <GitHubImportModal
          open={showImportModal}
          onClose={() => setShowImportModal(false)}
          onImportComplete={fetchAllPlugins}
        />
      )}
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
  globalIcon?: string | null
}

function FeaturedPluginCard({
  plugin,
  isEnabled,
  isEnabling,
  onEnable,
  onClick,
  globalIcon,
}: FeaturedPluginCardProps) {
  return (
    <Card
      className="min-w-[300px] max-w-[340px] flex-shrink-0 bg-gradient-to-br from-primary/10 to-secondary/30 border-primary/20 hover:border-primary/20 transition-all duration-300 cursor-pointer"
      onClick={onClick}
    >
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            {globalIcon ? (
              <PremiumIcon name={globalIcon} size={32} className="shrink-0" />
            ) : (
              <Sparkles className="w-8 h-8 text-primary shrink-0" />
            )}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h4 className="font-semibold text-foreground truncate">{plugin.name}</h4>
                <StatusBadge size="sm" status="primary" className="flex-shrink-0">
                  <Star className="w-2.5 h-2.5 mr-0.5" />
                  Featured
                </StatusBadge>
              </div>
              <p className="text-xs text-muted-foreground">v{plugin.version}</p>
            </div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onClick() }}>
                <Eye className="w-4 h-4 mr-2" />
                View Details
              </DropdownMenuItem>
              {!isEnabled && (
                <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onEnable() }} disabled={isEnabling}>
                  <Sparkles className="w-4 h-4 mr-2" />
                  {isEnabling ? 'Enabling...' : 'Enable for Workspace'}
                </DropdownMenuItem>
              )}
              {isEnabled && (
                <DropdownMenuItem disabled>
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Enabled
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        <p className="text-sm text-muted-foreground line-clamp-2">
          {plugin.description || 'No description available'}
        </p>

        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          {plugin.skills_count > 0 && (
            <div className="flex items-center gap-1">
              <Terminal className="w-3 h-3" />
              <span>{plugin.skills_count} skills</span>
            </div>
          )}
          {plugin.commands_count > 0 && (
            <div className="flex items-center gap-1">
              <Zap className="w-3 h-3" />
              <span>{plugin.commands_count} cmds</span>
            </div>
          )}
          {plugin.agents_count > 0 && (
            <div className="flex items-center gap-1">
              <Cpu className="w-3 h-3" />
              <span>{plugin.agents_count} agents</span>
            </div>
          )}
          <div className="flex items-center gap-1">
            <Download className="w-3 h-3" />
            <span>{plugin.enable_count}</span>
          </div>
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
  isAdmin?: boolean
  isApproving?: boolean
  isDeleting?: boolean
  isDeactivating?: boolean
  onApprove?: () => void
  onDeactivate?: () => void
  onDelete?: () => void
  globalIcon?: string | null
}

function PluginCard({
  plugin,
  index,
  isEnabled,
  isEnabling,
  onEnable,
  onClick,
  isAdmin,
  isApproving,
  isDeleting,
  isDeactivating,
  onApprove,
  onDeactivate,
  onDelete,
  globalIcon,
}: PluginCardProps) {
  const isPending = plugin.approval_status === 'pending'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ delay: Math.min(index * 0.05, 0.5) }}
    >
      <Card
        className="glass-card card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer"
        onClick={onClick}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              {globalIcon ? (
                <PremiumIcon name={globalIcon} size={40} className="shrink-0" />
              ) : (
                <Sparkles className="w-10 h-10 text-primary shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold text-foreground line-clamp-1">
                    {plugin.name}
                  </h3>
                  {plugin.is_featured && (
                    <Star className="w-3.5 h-3.5 text-primary flex-shrink-0" />
                  )}
                </div>
                <p className="text-xs text-muted-foreground">
                  v{plugin.version}
                  {plugin.original_author && (
                    <> &middot; {plugin.original_author}</>
                  )}
                </p>
              </div>
            </div>
            {plugin.security_status === 'safe' && (
              <StatusBadge size="sm" status="success" className="flex-shrink-0">
                Verified
              </StatusBadge>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground line-clamp-2">
            {plugin.description || 'No description available'}
          </p>

          {/* Category + Tags */}
          <div className="flex flex-wrap gap-1.5">
            {plugin.category_name && (
              <StatusBadge size="sm" status="neutral">
                {plugin.category_name}
              </StatusBadge>
            )}
            {(plugin.tags || []).slice(0, 2).map((tag) => (
              <StatusBadge key={tag} size="sm" status="neutral">
                {tag}
              </StatusBadge>
            ))}
          </div>

          {/* Stats Row */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            {plugin.skills_count > 0 && (
              <div className="flex items-center gap-1" title="Skills">
                <Terminal className="w-3 h-3" />
                <span>{plugin.skills_count}</span>
              </div>
            )}
            {plugin.commands_count > 0 && (
              <div className="flex items-center gap-1" title="Commands">
                <Zap className="w-3 h-3" />
                <span>{plugin.commands_count}</span>
              </div>
            )}
            {plugin.agents_count > 0 && (
              <div className="flex items-center gap-1" title="Agents">
                <Cpu className="w-3 h-3" />
                <span>{plugin.agents_count}</span>
              </div>
            )}
            <div className="flex items-center gap-1 ml-auto" title="Enabled by workspaces">
              <Download className="w-3 h-3" />
              <span>{plugin.enable_count}</span>
            </div>
          </div>

        </CardContent>
        <Separator />
        <div className="flex items-center justify-between px-6 py-3">
          <Button variant="ghost" size="sm"
            onClick={(e) => { e.stopPropagation(); onClick() }}
            className="text-muted-foreground hover:text-foreground p-0 h-auto">
            Details
          </Button>
          {isEnabled ? (
            <Button size="sm" variant="secondary"
              className="bg-secondary/50 hover:bg-secondary border border-white/10">
              <CheckCircle className="w-3 h-3 mr-2" />
              Added
            </Button>
          ) : (
            <Button size="sm" variant="outline"
              onClick={(e) => { e.stopPropagation(); onEnable() }}
              disabled={isEnabling}>
              {isEnabling ? 'Adding...' : 'Add to Workspace'}
            </Button>
          )}
        </div>
      </Card>
    </motion.div>
  )
}
