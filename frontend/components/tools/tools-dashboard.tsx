
/**
 * Tools Dashboard
 * ===============
 *
 * UI for marketplace + connected integrations.
 * Data source: rewrite `/api/tools/*` endpoints (DB-backed cache + connections).
 */

'use client'

import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'
import {
  Search,
  Filter,
  Grid3X3,
  List,
  Zap,
  Shield,
  Clock,
  CheckCircle,
  AlertTriangle,
  Settings,
  Download,
  Tag,
  Star,
  TrendingUp,
  Users,
  Activity,
  Trash2,
  MoreVertical,
  Eye,
  Store
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar, type StatItem } from '@/components/shared/stats-bar'
import { SearchInput } from '@/components/shared/search-input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { apiClient } from '@/lib/api-client'
import { ToolConfigModal } from './tool-config-modal'
import { ToolDetailsModal } from './tool-details-modal'
// import { AgentToolAssignment } from './agent-tool-assignment'
import { EnhancedPagination } from '@/components/ui/pagination'
import { useTools, useToolsStats, useToolCategories } from '@/hooks/use-tools-api'
import { ToolLogo } from '@/components/ui/tool-logo'
import { ComposioAppsSection } from './composio-apps-section' // PRD-36: Composio Integration
import { ToolActionsModal } from './tool-actions-modal'
import { useInitiateConnection, useDisconnectApp } from '@/hooks/use-composio-api'
import { Loader2, ExternalLink, Wrench } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useToast } from '@/hooks/use-toast'

// Tool Categories are now loaded dynamically from the API
// See toolCategories useMemo below for the dynamic implementation


// Tools data comes from the database via `/api/tools/*`
// No mock data needed - using cached marketplace metadata + connections.

const statusIcons = {
  available: CheckCircle,
  deprecated: AlertTriangle,
  maintenance: Clock,
  beta: Zap
}

const statusColors = {
  available: 'text-[hsl(var(--success))]',
  deprecated: 'text-[hsl(var(--destructive))]',
  maintenance: 'text-[hsl(var(--warning))]',
  beta: 'text-[hsl(var(--info))]'
}

interface Tool {
  id: number
  name: string
  description: string
  category: string
  icon: string
  provider: string
  status: string
  isInstalled: boolean
  isConfigured: boolean
  version: string
  pricing: string
  rating: number
  usageCount: number
  tags: string[]
  permissions: string[]
  requiredCredentials: string[]
  supportedEnvironments: string[]
  lastUpdated: string
  configuration?: Record<string, any>
  metadata?: Record<string, any>
  logo?: string
  composio_app_name?: string
  source?: 'composio' | 'internal'
  integration_url?: string
}

export function ToolsDashboard() {
  const { toast } = useToast()
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [sortBy, setSortBy] = useState('name')
  const [activeTab, setActiveTab] = useState('enabled')
  const [cachedToolsData, setCachedToolsData] = useState<any | null>(null)

  // Pagination state - dynamic page size based on view mode
  const [currentPage, setCurrentPage] = useState(1)
  const pageSize = viewMode === 'list' ? 60 : 20 // 60 for list (20 rows × 3 cols), 20 for grid

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchQuery)
      setCurrentPage(1) // Reset to first page on search
    }, 500)
    return () => clearTimeout(timer)
  }, [searchQuery])

  // Reset to page 1 when view mode changes
  useEffect(() => {
    setCurrentPage(1)
  }, [viewMode])

  // Fetch real data from API with error handling, pagination, and search
  const categoryParam = selectedCategory !== 'all' ? selectedCategory : undefined
  console.log('API call params:', {
    skip: (currentPage - 1) * pageSize,
    limit: pageSize,
    search: debouncedSearch,
    category: categoryParam
  })

  const {
    data: toolsData,
    isLoading: toolsLoading,
    isFetching: toolsFetching,
    error: toolsError
  } = useTools({
    skip: (currentPage - 1) * pageSize,
    limit: pageSize,
    search: debouncedSearch || undefined,
    category: categoryParam
  })
  // Fetch connected tools for the Applications tab (with search support)
  const { data: enabledToolsData, isLoading: enabledToolsLoading, error: enabledToolsError } = useTools({
    status: 'active',
    search: debouncedSearch || undefined,
    limit: 1000
  })
  const { data: statsData, error: statsError } = useToolsStats()
  const { data: categoriesData, error: categoriesError } = useToolCategories()
  const disconnectAppMutation = useDisconnectApp()
  const queryClient = useQueryClient()
  const syncCacheMutation = useMutation({
    mutationFn: async () => apiClient.syncToolsCache('full'),
    onSuccess: () => {
      // Refresh marketplace + stats
      queryClient.invalidateQueries({ queryKey: ['tools'] })
      queryClient.invalidateQueries({ queryKey: ['tools', 'stats'] })
      queryClient.invalidateQueries({ queryKey: ['tools', 'categories'] })
      toast({
        title: 'Sync started/completed',
        description: 'Marketplace cache sync finished. Refreshing tools list…',
      })
    },
    onError: (err: any) => {
      console.error('[TOOLS_SYNC] Failed', err)
      toast({
        title: 'Sync failed',
        description: err?.message || 'Failed to sync tools cache.',
        variant: 'destructive',
      })
    },
  })

  useEffect(() => {
    if (toolsData) setCachedToolsData(toolsData)
  }, [toolsData])

  const effectiveToolsData = toolsData ?? cachedToolsData

  // Extract tools and pagination data
  const rawTools = (effectiveToolsData as any)?.data || []
  const paginationData = (effectiveToolsData as any)?.pagination || { total: 0, pages: 0, current_page: 1 }

  // Build dynamic categories from API data
  const toolCategories = useMemo(() => {
    const categories = [
      {
        id: 'all',
        name: 'All Tools',
        color: 'text-muted-foreground',
        count: paginationData.total || 0
      }
    ]

    if (categoriesData && Array.isArray(categoriesData)) {
      console.log('Categories from API:', categoriesData)
      categoriesData.forEach((cat: any) => {
        categories.push({
          id: cat.name || cat.id, // Use name as ID to match exact backend string
          name: cat.name,
          color: 'text-[hsl(var(--info))]', // Default color
          count: cat.count || 0
        })
      })
    }

    return categories
  }, [categoriesData, paginationData.total])

  // Log any API errors (but don't spam the console)
  if (toolsError && !(toolsError as any).message?.includes('422')) console.error('Tools API Error:', toolsError)
  if (statsError && !(statsError as any).message?.includes('422')) console.error('Stats API Error:', statsError)
  if (categoriesError && !(categoriesError as any).message?.includes('422')) console.error('Categories API Error:', categoriesError)
  if (enabledToolsError && !(enabledToolsError as any).message?.includes('422')) console.error('Enabled Tools API Error:', enabledToolsError)

  const [loading, setLoading] = useState(false)
  const [toolModifications, setToolModifications] = useState<Record<number, Partial<Tool>>>({})

  // Modal states
  const [configModalOpen, setConfigModalOpen] = useState(false)

  const [detailsModalOpen, setDetailsModalOpen] = useState(false)
  const [detailsInitialTab, setDetailsInitialTab] = useState<'features' | 'triggers'>('features')
  const [selectedTool, setSelectedTool] = useState<any | null>(null)

  const [actionModalOpen, setActionModalOpen] = useState(false)

  // Connection state
  const initiateConnection = useInitiateConnection()
  const [connectingTool, setConnectingTool] = useState<string | null>(null)

  // Listen for popup messages
  useEffect(() => {
    const handleMessage = async (event: MessageEvent) => {
      if (event.data?.type === 'COMPOSIO_CONNECTED') {
        console.log('✅ Connection successful message received!', event.data)
        setConnectingTool(null)

        // Invalidate and refetch ALL tools queries to update connection status immediately
        console.log('[AUTO-REFRESH] Invalidating and refetching tools queries...')
        await queryClient.invalidateQueries({ queryKey: ['tools'] })
        await queryClient.refetchQueries({ queryKey: ['tools'] })
        await queryClient.invalidateQueries({ queryKey: ['composio', 'connections'] })
        console.log('[AUTO-REFRESH] Queries refetched successfully')

        toast({
          title: 'Connected!',
          description: `${event.data.app_name || 'App'} is now connected and ready to use.`,
        })
      }
    }
    window.addEventListener('message', handleMessage)
    return () => window.removeEventListener('message', handleMessage)
  }, [queryClient, toast])


  // Convert raw tools to match Tool interface for UI compatibility
  const normalizeTools = useCallback(
    (rawTools: any[]) => {
      try {
        return (rawTools || []).map((tool: any) => {
          const baseTool = {
            ...tool,
            isInstalled: tool.status === 'active',
            // Configuration lives in Composio connection + agent app assignments.
            // The Tools marketplace doesn't track "configured" state per app here.
            isConfigured: false,
            rating: 0,
            usageCount: 0,
            permissions: [],
            requiredCredentials: Object.keys(tool.credentials_schema?.required || {}),
            supportedEnvironments: ['production', 'staging'],
            lastUpdated: tool.updated_at,
            pricing: 'Free'
          }

          const modifications = toolModifications[tool.id]
          return modifications ? { ...baseTool, ...modifications } : baseTool
        })
      } catch (error) {
        console.error('Error converting tools:', error)
        return []
      }
    },
    [toolModifications]
  )

  const tools = useMemo(() => normalizeTools(rawTools as any[]), [normalizeTools, rawTools])
  const enabledTools = useMemo(
    () => normalizeTools((enabledToolsData as any)?.data || []),
    [normalizeTools, enabledToolsData]
  )
  const enabledToolsCount = useMemo(
    () => enabledTools.filter((tool) => tool?.isInstalled).length,
    [enabledTools]
  )

  // Filter and sort tools using useMemo to avoid infinite loops
  const filteredTools = useMemo(() => {
    let filtered = [...tools]

    // Filter by category
    if (selectedCategory !== 'all') {
      const selected = selectedCategory.toLowerCase()
      filtered = filtered.filter(tool =>
        (tool?.category || '').toLowerCase() === selected ||
        (tool?.tags || []).some((tag: string) => (tag || '').toLowerCase() === selected)
      )
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(tool =>
        tool?.name?.toLowerCase()?.includes(query) ||
        tool?.description?.toLowerCase()?.includes(query) ||
        tool?.tags?.some((tag: any) => tag?.toLowerCase()?.includes(query)) ||
        tool?.provider?.toLowerCase()?.includes(query)
      )
    }

    // Sort tools
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return (a?.name || '').localeCompare(b?.name || '')
        case 'updated':
          return new Date(b?.lastUpdated || 0).getTime() - new Date(a?.lastUpdated || 0).getTime()
        default:
          return 0
      }
    })

    return filtered
  }, [tools, selectedCategory, searchQuery, sortBy])

  const getToolStats = () => {
    // Calculate from enabled tools data (workspace apps)
    const connectedApps = enabledTools.filter(tool => tool?.isInstalled).length
    const workspaceApps = enabledTools.length
    const toolsAvailable = enabledTools
      .filter(tool => tool?.isInstalled)
      .reduce((acc, tool: any) => acc + (tool?.metadata?.action_count || 0), 0)
    const triggersAvailable = enabledTools
      .filter(tool => tool?.isInstalled)
      .reduce((acc, tool: any) => acc + (tool?.metadata?.trigger_count || 0), 0)

    return {
      connectedApps,
      workspaceApps,
      toolsAvailable,
      triggersAvailable
    }
  }

  const getCategoryCount = (categoryId: string) => {
    if (categoryId === 'all') return paginationData.total || 0

    // Use real category counts from API if available
    if (categoriesData && Array.isArray(categoriesData)) {
      const category = (categoriesData as any[]).find((c: any) =>
        c.id === categoryId || c.name?.toLowerCase().replace(/\s+/g, '_') === categoryId
      )
      if (category && typeof category.count === 'number') {
        return category.count
      }
    }

    // Fallback: This will be inaccurate as it only counts current page
    // TODO: Fetch all tools without pagination for accurate counts
    return tools.filter(tool => tool?.category === categoryId)?.length || 0
  }

  const isInitialLoading = toolsLoading && !effectiveToolsData

  // Show loading state with skeleton components only on first load
  if (isInitialLoading) {
    return (
      <div className="space-y-6">
        {/* Header skeleton */}
        <div className="flex items-center justify-between">
          <div>
            <div className="h-8 w-48 bg-secondary/50 rounded mb-2" />
            <div className="h-5 w-80 bg-secondary/50 rounded" />
          </div>
          <div className="h-10 w-32 bg-secondary/50 rounded" />
        </div>

        {/* Stats skeleton */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="glass-card">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="h-4 w-24 bg-secondary/50 rounded mb-2" />
                    <div className="h-8 w-16 bg-secondary/50 rounded mb-1" />
                    <div className="h-3 w-20 bg-secondary/50 rounded" />
                  </div>
                  <div className="w-10 h-10 bg-secondary/50 rounded-lg" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Tools grid skeleton */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="glass-card card-glow">
              <CardContent className="p-5 space-y-4">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 bg-secondary/50 rounded-lg shrink-0" />
                  <div className="flex-1">
                    <div className="h-4 w-3/4 bg-secondary/50 rounded mb-2" />
                    <div className="h-4 w-16 bg-secondary/50 rounded" />
                  </div>
                </div>
                <div className="h-3 w-1/2 bg-secondary/50 rounded" />
                <div className="flex gap-2">
                  <div className="h-8 flex-1 bg-secondary/50 rounded" />
                  <div className="h-8 flex-1 bg-secondary/50 rounded" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  const handleToolConfigure = (tool: Tool) => {
    setSelectedTool(tool)
    // For Composio tools, the configuration (Feature Toggles) is now in the Details modal
    // For legacy/other tools, keep using the Config modal
    if (tool.provider === 'Composio') {
      setDetailsInitialTab('features')
      setDetailsModalOpen(true)
    } else {
      setConfigModalOpen(true)
    }
  }

  const handleToolDetails = (tool: Tool) => {
    setSelectedTool(tool)
    setDetailsInitialTab('features')
    setDetailsModalOpen(true)
  }



  const handleToolConnect = async (tool: Tool) => {
    setConnectingTool(tool.name)

    // Open popup IMMEDIATELY on user click to avoid popup blocker
    // The popup must be opened synchronously from the user gesture
    const width = 450
    const height = 600
    const left = window.screenX + (window.outerWidth - width) / 2
    const top = window.screenY + (window.outerHeight - height) / 2

    const popup = window.open(
      'about:blank',  // Start with blank page
      `Connect ${tool.name}`,
      `width=${width},height=${height},left=${left},top=${top}`
    )

    if (!popup) {
      // If popup STILL blocked, fall back to same-window redirect
      console.warn('[CONNECT] Popup blocked, using same-window redirect')
    }

    try {
      const callbackPath = '/tools/callback'
      console.log(`[CONNECT] Initiating connection for ${tool.name}...`)

      const redirectUrlResult = await initiateConnection.mutateAsync({
        appName: tool.name,
        callbackUrl: `${window.location.origin}${callbackPath}`,
      })

      console.log('[CONNECT] API Response:', redirectUrlResult)

      if (!redirectUrlResult) {
        throw new Error('Invalid response from server')
      }

      // NO_AUTH apps (e.g. composio_search) are activated immediately — no OAuth redirect needed
      if (!redirectUrlResult.redirect_url) {
        console.log('[CONNECT] NO_AUTH app — activated immediately, refreshing...')
        popup?.close()
        setConnectingTool(null)
        await queryClient.invalidateQueries({ queryKey: ['tools'] })
        await queryClient.refetchQueries({ queryKey: ['tools'], type: 'active' })
        await queryClient.invalidateQueries({ queryKey: ['composio', 'connections'] })
        return
      }

      console.log('[CONNECT] Starting OAuth flow:', redirectUrlResult.redirect_url)

      if (popup && !popup.closed) {
        // Navigate the popup to the redirect URL
        popup.location.href = redirectUrlResult.redirect_url

        // Poll for popup close check
        const checkClosed = setInterval(async () => {
          if (popup?.closed) {
            clearInterval(checkClosed)
            setConnectingTool(null)
            // Invalidate and refetch ALL tools queries to update connection status immediately
            console.log('[AUTO-REFRESH] Popup closed, invalidating queries...')
            await queryClient.invalidateQueries({ queryKey: ['tools'] })
            await queryClient.refetchQueries({ queryKey: ['tools'], type: 'active' })
            await queryClient.invalidateQueries({ queryKey: ['composio', 'connections'] })
            console.log('[AUTO-REFRESH] Queries refreshed after popup close')
          }
        }, 1000)
      } else {
        // Fallback: Redirect in same window if popup was blocked
        setConnectingTool(null)
        window.location.href = redirectUrlResult.redirect_url
      }

    } catch (error) {
      console.error("[CONNECT] Connection initiation failed:", error)
      popup?.close()
      setConnectingTool(null)
      const msg = error instanceof Error ? error.message : "Unknown error"
      alert(`Failed to start connection flow: ${msg}\n\nPlease check console for details.`)
    }
  }


  const handleToolManage = (tool: Tool) => {
    setSelectedTool(tool)
    setActionModalOpen(true)
  }

  const handleToolDelete = async (tool: Tool) => {
    setLoading(true)
    try {
      const integrationUrl = (tool as any).integration_url || tool.metadata?.integration_url
      const composioFromUrl = integrationUrl?.startsWith('composio://')
        ? integrationUrl.replace('composio://', '').split('/')[0]
        : null
      const composioAppName =
        tool.composio_app_name ||
        tool.metadata?.composio_app_name ||
        tool.metadata?.app_name ||
        composioFromUrl ||
        tool.name
      const isComposioTool =
        tool.source === 'composio' ||
        tool.provider === 'Composio' ||
        !!tool.metadata?.composio_app_name ||
        !!composioFromUrl

      if (isComposioTool) {
        if (!composioAppName) {
          throw new Error('Missing Composio app name for disconnect')
        }
        console.log(`Disconnecting Composio app: ${composioAppName}`)
        await disconnectAppMutation.mutateAsync(composioAppName)
        console.log('Composio app disconnected successfully')
        return
      }
      throw new Error('Non-Composio tools are not supported in the new tools catalog.')

    } catch (error) {
      console.error('Failed to delete tool:', error)
      alert(`❌ Failed to delete ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const handleRemoveFromWorkspace = async (tool: Tool) => {
    setLoading(true)
    try {
      // Resolve the canonical Composio app name (e.g. "GOOGLEDRIVE"), not the
      // display name ("Google Drive") — backend looks up by ComposioConnection.app_name.
      const integrationUrl = (tool as any).integration_url || tool.metadata?.integration_url
      const composioFromUrl = integrationUrl?.startsWith('composio://')
        ? integrationUrl.replace('composio://', '').split('/')[0]
        : null
      const appName =
        tool.composio_app_name ||
        tool.metadata?.composio_app_name ||
        tool.metadata?.app_name ||
        composioFromUrl ||
        tool.name

      console.log(`Removing ${appName} from workspace`)

      await apiClient.delete(`/api/tools/remove-from-workspace/${encodeURIComponent(appName)}`)

      // Close modal and refresh tools list
      setDetailsModalOpen(false)
      queryClient.invalidateQueries({ queryKey: ['tools'] })
      queryClient.refetchQueries({ queryKey: ['tools'] })

      toast({
        title: 'Removed from Workspace',
        description: `${tool.name} has been removed from your workspace.`,
      })
    } catch (error) {
      console.error('Failed to remove from workspace:', error)
      toast({
        title: 'Error',
        description: `Failed to remove ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        variant: 'destructive',
      })
    } finally {
      setLoading(false)
    }
  }

  const stats = getToolStats()

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div data-tour="tools-page-header">
      <PageHeader
        title="Tools &"
        titleAccent="Integrations"
        subtitle="Discover, install, and manage tools to extend your AI agents' capabilities"
        actions={
          <Badge variant="outline" className="text-primary border-primary/30">
            <div className="w-2 h-2 bg-[hsl(var(--success))] rounded-full animate-pulse mr-2" />
            {toolsLoading ? 'Loading...' : `${paginationData.total || 0} Total Tools`}
          </Badge>
        }
      />
      </div>

      {/* Statistics Cards */}
      <StatsBar
        stats={[
          {
            label: 'Connected Apps',
            value: stats.connectedApps.toString(),
            change: 'Active',
            icon: CheckCircle,
            iconColor: 'text-[hsl(var(--success))]',
            globalIconKey: 'global_channel',

          },
          {
            label: 'In Workspace',
            value: stats.workspaceApps.toString(),
            change: 'Total apps',
            icon: Grid3X3,
            iconColor: 'text-[hsl(var(--info))]',
          },
          {
            label: 'Tools Available',
            value: stats.toolsAvailable.toString(),
            change: 'Actions',
            icon: Wrench,
            iconColor: 'text-[hsl(var(--agent))]',
            globalIconKey: 'global_tool',

          },
          {
            label: 'Triggers',
            value: stats.triggersAvailable.toString(),
            change: 'Available',
            icon: Zap,
            iconColor: 'text-primary',
            globalIconKey: 'global_trigger',
          }
        ] satisfies StatItem[]}
        loading={toolsLoading}
      />

      {/* Consolidated Controls Bar */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          {/* Line 1: Tabs + Sort + Search + View Toggle */}
          <div className="w-full grid grid-cols-1 lg:grid-cols-[auto,minmax(0,1fr),auto] gap-3 items-center">
            <div className="flex items-center gap-3 flex-wrap">
              <TabsList className="bg-secondary/50">
                <TabsTrigger value="enabled" className="flex items-center gap-1.5 px-3">
                  <CheckCircle className="w-4 h-4" />
                  <span>Applications ({enabledToolsCount})</span>
                </TabsTrigger>
              </TabsList>

              {/* Sort Dropdown - Pill shaped */}
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-32 h-9 bg-secondary/50 rounded-full border-border/40 hover:border-primary/50 transition-colors">
                  <SelectValue placeholder="Sort" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="updated">Updated</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Search - Full Width */}
            <div data-tour="tools-search">
            <SearchInput
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder="Search tools..."
              loading={toolsFetching}
            />
            </div>

            {/* View Mode Toggle */}
            <div className="flex items-center space-x-1 bg-secondary/30 rounded-lg p-1 justify-self-end">
              <Button
                variant={viewMode === 'grid' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('grid')}
                className="h-7 w-7 p-0"
              >
                <Grid3X3 className="w-4 h-4" />
              </Button>
              <Button
                variant={viewMode === 'list' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('list')}
                className="h-7 w-7 p-0"
              >
                <List className="w-4 h-4" />
              </Button>
            </div>
          </div>

          <TabsContent value="enabled" className="space-y-6" data-tour="tools-connected-section">
            {/* Enabled Tools Management */}
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">Applications</h3>
              <div className={viewMode === 'grid'
                ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6'
                : 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3'
              }>
                <AnimatePresence>
                  {enabledTools.map((tool, index) => (
                      <ToolCard
                        key={tool?.id}
                        tool={tool}
                        viewMode={viewMode}
                        index={index}
                        onInstall={() => {
                          if (tool.provider === 'Composio') {
                            handleToolConnect(tool)
                          }
                        }}
                        onDetails={() => handleToolDetails(tool)}
                        onUninstall={() => handleToolConfigure(tool)}
                        onConfigure={() => handleToolConfigure(tool)}
                        loading={loading}
                        showMenu={true}
                      />
                    ))}
                </AnimatePresence>
              </div>
            </div>
          </TabsContent>


          {/* Installed tab removed; Enabled is now first */}

          {/* App Integrations tab removed - not used */}

          <TabsContent value="security" className="space-y-6">
            {/* Security Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Shield className="w-5 h-5 text-[hsl(var(--success))]" />
                    <span>Security Status</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Active Tools</span>
                    <Badge className="bg-[hsl(var(--success))]/10 text-[hsl(var(--success))] border-[hsl(var(--success))]/20">
                      {tools.filter(t => t?.status === 'active').length}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Assigned Tools</span>
                    <Badge className="bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20">
                      {tools.filter(t => t?.isInstalled).length}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Configured Tools</span>
                    <Badge className="bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/20">
                      {tools.filter(t => t?.isConfigured).length}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Total Agents</span>
                    <Badge className="bg-primary/10 text-primary border-primary/20">
                      —
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Activity className="w-5 h-5 text-[hsl(var(--info))]" />
                    <span>Tool Assignments</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="text-center py-4">
                      <p className="text-sm text-muted-foreground">
                        Agent↔tool assignments are managed on the Agents page.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Modals */}
      <ToolDetailsModal
        open={detailsModalOpen}
        onClose={() => setDetailsModalOpen(false)}
        tool={selectedTool}
        onInstall={() => selectedTool && handleToolConnect(selectedTool)}
        onUninstall={() => selectedTool && handleToolDelete(selectedTool)}
        onRemoveFromWorkspace={() => selectedTool && handleRemoveFromWorkspace(selectedTool)}
        onConfigure={() => {
          setDetailsModalOpen(false)
          handleToolConfigure(selectedTool)
        }}
        loading={loading || connectingTool === selectedTool?.name}
        initialTab={detailsInitialTab}
      />

      <ToolConfigModal
        open={configModalOpen}
        onClose={() => setConfigModalOpen(false)}
        tool={selectedTool}
      />

      <ToolActionsModal
        open={actionModalOpen}
        onClose={() => setActionModalOpen(false)}
        tool={selectedTool}
      />

      {/* <AgentToolAssignment
        open={assignmentModalOpen}
        onClose={() => setAssignmentModalOpen(false)}
        tools={tools}
      /> */}

    </div>
  )
}

// Tool Card Component
interface ToolCardProps {
  tool: Tool
  viewMode: 'grid' | 'list'
  index: number
  onInstall: () => void
  onDetails: () => void
  onUninstall: () => void
  onConfigure: () => void
  loading: boolean
  showMenu?: boolean
}

function ToolCard({
  tool,
  viewMode,
  index,
  onInstall, // This will be handleToolConnect for Connect
  onDetails,
  onUninstall, // This will be handleToolManage for Manage
  onConfigure,
  loading,
  showMenu = false
}: ToolCardProps) {
  const StatusIcon = statusIcons[tool?.status as keyof typeof statusIcons] || CheckCircle
  const statusColor = statusColors[tool?.status as keyof typeof statusColors] || 'text-muted-foreground'
  const authSchemes = Array.isArray(tool?.metadata?.auth_schemes) ? tool.metadata.auth_schemes : []
  const toolsCount = tool?.metadata?.action_count ?? tool?.metadata?.actions_count ?? tool?.metadata?.tools_count
  const triggersCount = tool?.metadata?.trigger_count ?? (Array.isArray(tool?.metadata?.triggers) ? tool.metadata.triggers.length : undefined)

  const formatAuthScheme = (scheme: string) => {
    const normalized = scheme.toLowerCase()
    if (normalized.includes('oauth')) return 'OAuth2'
    if (normalized.includes('bearer')) return 'Bearer Token'
    if (normalized.includes('api_key') || normalized.includes('apikey')) return 'API Key'
    if (normalized.includes('basic')) return 'Basic Auth'
    return scheme.replace(/_/g, ' ').toUpperCase()
  }

  // Use isInstalled to determine if we show "Manage" or "Connect"
  // isInstalled comes from the API which checks Composio connection status
  const isConnected = tool?.isInstalled

  if (viewMode === 'list') {
    // Compact list card — matches agent roster list style
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 10 }}
        transition={{ delay: index * 0.03 }}
      >
        <Card className="glass-card hover:border-primary/20 transition-all h-full">
          <CardContent className="p-3 flex items-center gap-3 h-full">
            <ToolLogo
              logo={tool?.logo}
              name={tool?.name}
              size={36}
              fallbackIcon={tool?.icon}
              showBackground={true}
            />
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-sm truncate">{tool?.name}</h3>
              <p className="text-xs text-muted-foreground truncate">
                {tool?.category || tool?.provider || 'Integration'}
              </p>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <Button variant="outline" size="sm" onClick={onDetails} className="h-7 text-xs px-2">
                View
              </Button>
              {isConnected ? (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 text-xs px-2 text-[hsl(var(--info))] border border-[hsl(var(--info))]/30 hover:bg-[hsl(var(--info))]/10"
                  onClick={onUninstall}
                >
                  <Settings className="w-3 h-3 mr-1" />
                  Manage
                </Button>
              ) : (
                <Button size="sm" variant="outline" onClick={onInstall} className="h-7 text-xs px-2">
                  <ExternalLink className="w-3 h-3 mr-1" />
                  Connect
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ delay: index * 0.1 }}
    >
      <Card className={`glass-card card-glow transition-all duration-300 flex flex-col h-full group ${
        isConnected
          ? 'border-[hsl(var(--success))]/30 hover:border-[hsl(var(--success))]/50'
          : 'hover:border-primary/20'
      }`}>
        <CardContent className="p-5 flex flex-col flex-1">
          {/* Header row: logo + name + category */}
          <div className="flex items-center gap-3">
            <ToolLogo
              logo={tool?.logo}
              name={tool?.name}
              size={40}
              fallbackIcon={tool?.icon}
              showBackground={true}
            />
            <div className="min-w-0 flex-1">
              <h3 className="font-semibold text-sm leading-tight truncate">{tool?.name}</h3>
              <Badge variant="outline" className="mt-1 capitalize text-[10px] h-5">
                {tool?.category || tool?.provider || 'Integration'}
              </Badge>
            </div>
            {isConnected && (
              <Badge className="bg-[hsl(var(--success))]/20 text-[hsl(var(--success))] border-none text-xs px-2 py-0.5 shrink-0">
                Connected
              </Badge>
            )}
          </div>

          {/* Stats row */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground mt-4">
            <div className="flex items-center gap-1">
              <Wrench className="w-3 h-3" />
              <span>{typeof toolsCount === 'number' ? toolsCount : 0} Tools</span>
            </div>
            <span className="text-border">|</span>
            <div className="flex items-center gap-1">
              <Zap className="w-3 h-3" />
              <span>{typeof triggersCount === 'number' ? triggersCount : 0} Triggers</span>
            </div>
          </div>

          {/* Action buttons — pushed to bottom */}
          <div className="flex items-center gap-2 pt-4 mt-auto">
            <Button variant="outline" size="sm" onClick={onDetails} className="flex-1 h-8">
              View
            </Button>
            {isConnected ? (
              <Button
                size="sm"
                variant="secondary"
                className="flex-1 h-8 bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border border-[hsl(var(--info))]/30 hover:bg-[hsl(var(--info))]/20"
                onClick={onUninstall}
              >
                <Settings className="w-3 h-3 mr-1.5" />
                Manage
              </Button>
            ) : (
              <Button
                size="sm"
                variant="outline"
                className="flex-1 h-8"
                onClick={onInstall}
              >
                <ExternalLink className="w-3 h-3 mr-1.5" />
                Connect
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
