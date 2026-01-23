
/**
 * Tools Dashboard - MCP Tools Management
 * ========================================
 * 
 * Phase 3: Updated to use REAL API data from MCP Tools endpoints
 * - Replaced mock data with useMCPTools() hook
 * - Real-time stats from useMCPToolsStats()
 * - Category counts from useMCPToolCategories()
 * 
 * TODO: Complete implementation of:
 * - Installation status tracking
 * - Tool configuration management  
 * - Ratings system
 * - Usage analytics integration
 */

'use client'

import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
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
  Eye
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
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
import { useMCPTools, useMCPToolsStats, useMCPToolCategories, useMCPToolAssignments, useUpdateMCPTool } from '@/hooks/use-mcp-tools-api'
import { ToolLogo } from '@/components/ui/tool-logo'
import { ComposioAppsSection } from './composio-apps-section' // PRD-36: Composio Integration
import { ToolActionsModal } from './tool-actions-modal'
import { useInitiateConnection, useDisconnectApp } from '@/hooks/use-composio-api'
import { Loader2, ExternalLink, Wrench } from 'lucide-react'

// Tool Categories are now loaded dynamically from the API
// See toolCategories useMemo below for the dynamic implementation


// MCP Tools data comes from the database via useMCPTools hook
// No mock data needed - using real MCP server metadata

const statusIcons = {
  available: CheckCircle,
  deprecated: AlertTriangle,
  maintenance: Clock,
  beta: Zap
}

const statusColors = {
  available: 'text-green-400',
  deprecated: 'text-red-400',
  maintenance: 'text-yellow-400',
  beta: 'text-blue-400'
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
  source?: 'mcp' | 'composio'
  mcp_server_url?: string
}

export function ToolsDashboard() {
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
    data: mcpToolsData,
    isLoading: toolsLoading,
    isFetching: toolsFetching,
    error: toolsError
  } = useMCPTools({
    skip: (currentPage - 1) * pageSize,
    limit: pageSize,
    search: debouncedSearch || undefined,
    category: categoryParam
  })
  const { data: enabledToolsData, isLoading: enabledToolsLoading, error: enabledToolsError } = useMCPTools({
    status: 'active',
    limit: 1000
  })
  const { data: statsData, error: statsError } = useMCPToolsStats()
  const { data: categoriesData, error: categoriesError } = useMCPToolCategories()
  // Temporarily disable assignments API until backend endpoint is ready
  // const { data: toolAssignments = [], error: assignmentsError } = useMCPToolAssignments(false)
  const toolAssignments: any[] = [] // Mock empty assignments for now
  const updateToolMutation = useUpdateMCPTool()
  const disconnectAppMutation = useDisconnectApp()

  useEffect(() => {
    if (mcpToolsData) setCachedToolsData(mcpToolsData)
  }, [mcpToolsData])

  const effectiveToolsData = mcpToolsData ?? cachedToolsData

  // Extract tools and pagination data
  const mcpTools = (effectiveToolsData as any)?.data || []
  const paginationData = (effectiveToolsData as any)?.pagination || { total: 0, pages: 0, current_page: 1 }

  // Build dynamic categories from API data
  const toolCategories = useMemo(() => {
    const categories = [
      {
        id: 'all',
        name: 'All Tools',
        color: 'text-gray-400',
        count: paginationData.total || 0
      }
    ]

    if (categoriesData && Array.isArray(categoriesData)) {
      console.log('Categories from API:', categoriesData)
      categoriesData.forEach((cat: any) => {
        categories.push({
          id: cat.name || cat.id, // Use name as ID to match exact backend string
          name: cat.name,
          color: 'text-blue-400', // Default color
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
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === 'COMPOSIO_CONNECTED') {
        console.log('✅ Connection successful message received!', event.data)
        setConnectingTool(null)
        // Trigger a refetch of tools to update status
        // We can treat this as "connected" optimistically too if needed
        // But invalidating query is safer
        // queryClient.invalidateQueries(['mcp-tools']) (need queryClient access or just wait for poll)
        window.location.reload() // Brute force refresh for now to ensure state sync, or use queryClient if available
      }
    }
    window.addEventListener('message', handleMessage)
    return () => window.removeEventListener('message', handleMessage)
  }, [])


  // Convert MCP tools to match Tool interface for UI compatibility
  const normalizeTools = useCallback(
    (rawTools: any[]) => {
      try {
        return (rawTools || []).map((tool: any) => {
          const isAssigned = (toolAssignments as any[]).some((assignment: any) =>
            assignment.tool_id === tool.id && assignment.enabled
          )

          const baseTool = {
            ...tool,
            isInstalled: tool.status === 'active',
            isConfigured: isAssigned,
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
    [toolAssignments, toolModifications]
  )

  const tools = useMemo(() => normalizeTools(mcpTools as any[]), [normalizeTools, mcpTools])
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
        case 'rating':
          return (b?.rating || 0) - (a?.rating || 0)
        case 'usage':
          return (b?.usageCount || 0) - (a?.usageCount || 0)
        case 'updated':
          return new Date(b?.lastUpdated || 0).getTime() - new Date(a?.lastUpdated || 0).getTime()
        default:
          return 0
      }
    })

    return filtered
  }, [tools, selectedCategory, searchQuery, sortBy])

  const getToolStats = () => {
    // Use real stats from API if available and has the expected structure
    if (statsData && typeof statsData === 'object' && 'total_tools' in statsData) {
      console.log('Stats API data:', statsData)
      return {
        connectedApps: (statsData as any).connected_apps ?? (statsData as any).active_tools ?? 0,
        categories: (Array.isArray(categoriesData) && categoriesData.length > 0)
          ? categoriesData.length
          : (statsData as any).categories ?? 0,
        appsAvailable: (statsData as any).apps_available ?? (statsData as any).total_tools ?? 0,
        totalTools: (statsData as any).tools_available ?? 0
      }
    }

    console.log('Stats API not available, using fallback. Tools on page:', tools.length)

    // Fallback: Use pagination total for overall count, but this won't give accurate enabled count
    // The enabled count will be wrong unless we fetch all tools
    const connectedApps = tools.filter(tool => tool?.isInstalled)?.length || 0
    const categories = (categoriesData && Array.isArray(categoriesData)) ? categoriesData.length : 0
    const appsAvailable = paginationData.total || tools?.length || 0
    const totalTools = tools.reduce((acc, tool: any) => acc + (tool?.metadata?.action_count || 0), 0)

    return {
      connectedApps, // This will be inaccurate - only counts current page
      categories,
      appsAvailable,
      totalTools
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
            <Card key={i} className="glass-card">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="h-8 w-8 bg-secondary/50 rounded" />
                  <div className="h-6 w-16 bg-secondary/50 rounded" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-5 w-3/4 bg-secondary/50 rounded mb-2" />
                <div className="h-4 w-1/2 bg-secondary/50 rounded mb-4" />
                <div className="flex justify-between">
                  <div className="h-4 w-16 bg-secondary/50 rounded" />
                  <div className="h-4 w-20 bg-secondary/50 rounded" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  const handleToolToggle = async (tool: Tool, enabled: boolean) => {
    setLoading(true)
    try {
      // Update local state immediately for better UX
      setToolModifications(prev => ({
        ...prev,
        [tool.id]: { isInstalled: enabled, isConfigured: enabled }
      }))

      // Update the tool status in the database
      await updateToolMutation.mutateAsync({
        id: tool.id,
        data: {
          status: enabled ? 'active' : 'inactive',
          // Add metadata to track installation state
          metadata: {
            ...(tool.metadata || {}),
            is_installed: enabled,
            installed_at: enabled ? new Date().toISOString() : null
          }
        }
      })

      console.log(`✅ Successfully ${enabled ? 'enabled' : 'disabled'} tool: ${tool.name}`)
      if (enabled) {
        setSelectedTool(tool)
        setConfigModalOpen(true)
      }

    } catch (error) {
      console.error(`❌ Failed to ${enabled ? 'enable' : 'disable'} tool:`, error)
      // Revert local state on error
      setToolModifications(prev => ({
        ...prev,
        [tool.id]: { isInstalled: !enabled, isConfigured: !enabled }
      }))
      alert(`❌ Failed to ${enabled ? 'enable' : 'disable'} ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
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

      if (!redirectUrlResult || !redirectUrlResult.redirect_url) {
        throw new Error('Invalid response from server: Missing redirect_url')
      }

      console.log('[CONNECT] Starting OAuth flow:', redirectUrlResult.redirect_url)

      if (popup && !popup.closed) {
        // Navigate the popup to the redirect URL
        popup.location.href = redirectUrlResult.redirect_url

        // Poll for popup close check
        const checkClosed = setInterval(() => {
          if (popup?.closed) {
            clearInterval(checkClosed)
            setConnectingTool(null)
            // Refresh connections after popup closes
            window.location.reload()
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
      const mcpServerUrl = tool.mcp_server_url || tool.metadata?.mcp_server_url
      const composioFromUrl = mcpServerUrl?.startsWith('composio://')
        ? mcpServerUrl.replace('composio://', '').split('/')[0]
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

      console.log(`Deleting tool: ${tool.name} (ID: ${tool.id})`)

      // Use the mutation hook which automatically invalidates queries
      await updateToolMutation.mutateAsync({
        id: tool.id,
        data: { status: 'inactive' }
      })

      console.log('Tool deleted successfully')

    } catch (error) {
      console.error('Failed to delete tool:', error)
      alert(`❌ Failed to delete ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const stats = getToolStats()

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex justify-between items-start"
      >
        <div>
          <h1 className="text-3xl font-bold mb-2">
            Tools & <span className="gradient-text">Integrations</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Discover, install, and manage tools to extend your AI agents' capabilities
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-brand-primary border-brand-primary/30">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
            {toolsLoading ? 'Loading...' : `${paginationData.total || 0} Total Tools`}
          </Badge>

          {/* Agent Assignment Button Removed */}
        </div>
      </motion.div>

      {/* Statistics Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {[
          {
            label: 'Connected Apps',
            value: stats.connectedApps.toString(),
            change: 'Connected',
            icon: CheckCircle,
            color: 'text-green-400'
          },
          {
            label: 'Categories',
            value: stats.categories.toString(),
            change: 'Available',
            icon: Tag,
            color: 'text-blue-400'
          },
          {
            label: 'Apps Available',
            value: stats.appsAvailable.toString(),
            change: 'In marketplace',
            icon: Grid3X3,
            color: 'text-purple-400'
          },
          {
            label: 'Total Tools',
            value: stats.totalTools.toString(),
            change: 'Across apps',
            icon: TrendingUp,
            color: 'text-orange-400'
          }
        ].map((stat, index) => (
          <Card key={stat.label} className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-2xl bg-black/20 border border-orange-500/10 flex items-center justify-center shrink-0">
                    <stat.icon
                      className={`w-5 h-5 ${index === 0 ? 'text-green-400' :
                        index === 1 ? 'text-blue-400' :
                          index === 2 ? 'text-purple-400' :
                            index === 3 ? 'text-orange-400' :
                              'text-white'
                        }`}
                    />
                  </div>
                  <div className="min-w-0">
                    <div className="text-2xl font-bold leading-none">
                      {toolsLoading ? '…' : stat.value}
                    </div>
                    <div className="text-sm text-muted-foreground truncate">{stat.label}</div>
                  </div>
                </div>
                <div className={`shrink-0 text-xs ${stat.color}`}>
                  {stat.change}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </motion.div>

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
                  <span>Connected ({enabledToolsCount})</span>
                </TabsTrigger>
                <TabsTrigger value="marketplace" className="flex items-center gap-1.5 px-3">
                  <Grid3X3 className="w-4 h-4" />
                  <span>Marketplace</span>
                </TabsTrigger>
              </TabsList>

              {/* Sort Dropdown */}
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-32 h-9 bg-secondary/50">
                  <SelectValue placeholder="Sort" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="rating">Rating</SelectItem>
                  <SelectItem value="usage">Usage</SelectItem>
                  <SelectItem value="updated">Updated</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Search - Full Width */}
            <div className="relative w-full">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
              <Input
                placeholder="Search tools..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 pr-9 h-9 bg-secondary/50 border-secondary focus:border-primary/50"
              />
              {toolsFetching && (
                <Loader2 className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 animate-spin text-muted-foreground" />
              )}
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

          <TabsContent value="enabled" className="space-y-6">
            {/* Enabled Tools Management */}
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">Connected Tools</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <AnimatePresence>
                  {enabledTools
                    .filter(tool => tool?.isInstalled)
                    .map((tool, index) => (
                      <ToolCard
                        key={tool?.id}
                        tool={tool}
                        viewMode="grid"
                        index={index}
                        onInstall={() => {
                          if (tool.provider === 'Composio') {
                            handleToolConnect(tool)
                          } else {
                            handleToolToggle(tool, true)
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

          <TabsContent value="marketplace" className="space-y-4">

            {/* Categories */}
            <div className="flex overflow-x-auto pb-4 gap-2 no-scrollbar scroll-smooth">
              {toolCategories.map((category) => (
                <Button
                  key={category.id}
                  variant={selectedCategory === category.id ? 'default' : 'outline'}
                  onClick={() => setSelectedCategory(category.id)}
                  className={`flex-shrink-0 flex items-center space-x-2 ${selectedCategory === category.id
                    ? 'bg-gray-800 border-orange-400/50 text-white'
                    : 'hover:border-orange-500/50'
                    }`}
                >
                  <span>{category.name}</span>
                  <Badge variant="outline" className="ml-1 text-xs">
                    {getCategoryCount(category.id)}
                  </Badge>
                </Button>
              ))}
            </div>

            {/* Tools Grid/List */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <AnimatePresence>
                {filteredTools.map((tool, index) => (
                  <ToolCard
                    key={tool?.id}
                    tool={tool}
                    viewMode={viewMode}
                    index={index}
                    onInstall={() => {
                      if (tool.provider === 'Composio') {
                        handleToolConnect(tool)
                      } else {
                        handleToolToggle(tool, true)
                      }
                    }}
                    onDetails={() => handleToolDetails(tool)}
                    onUninstall={() => handleToolConfigure(tool)}
                    onConfigure={() => handleToolConfigure(tool)}
                    loading={loading}
                  />
                ))}
              </AnimatePresence>
            </div>

            {/* Pagination */}
            {paginationData.total > pageSize && (
              <EnhancedPagination
                data={paginationData}
                onPageChange={(page) => setCurrentPage(page)}
                className="mt-6"
              />
            )}

            {/* Empty State */}
            {filteredTools?.length === 0 && (
              <div className="text-center py-12">
                <div className="w-16 h-16 rounded-lg bg-secondary/30 flex items-center justify-center mx-auto mb-4">
                  <Search className="w-8 h-8 text-muted-foreground" />
                </div>
                <h3 className="text-lg font-semibold mb-2">No tools found</h3>
                <p className="text-muted-foreground mb-4">
                  Try adjusting your search or category filter
                </p>
                <Button variant="outline" onClick={() => {
                  setSearchQuery('')
                  setSelectedCategory('all')
                }}>
                  Clear Filters
                </Button>
              </div>
            )}
          </TabsContent>

          {/* Installed tab removed; Enabled is now first */}

          {/* App Integrations tab removed - not used */}

          <TabsContent value="security" className="space-y-6">
            {/* Security Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Shield className="w-5 h-5 text-green-400" />
                    <span>Security Status</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Active Tools</span>
                    <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                      {tools.filter(t => t?.status === 'active').length}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Assigned Tools</span>
                    <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                      {tools.filter(t => t?.isInstalled).length}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Configured Tools</span>
                    <Badge className="bg-purple-500/10 text-purple-400 border-purple-500/20">
                      {tools.filter(t => t?.isConfigured).length}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Total Agents</span>
                    <Badge className="bg-orange-500/10 text-orange-400 border-orange-500/20">
                      {new Set((toolAssignments as any[]).map((a: any) => a.agent_id)).size}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Activity className="w-5 h-5 text-blue-400" />
                    <span>Tool Assignments</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {(toolAssignments as any[]).slice(0, 5).map((assignment: any, index: number) => (
                      <div key={index} className="flex items-center space-x-3">
                        <div className="w-2 h-2 rounded-full bg-green-400" />
                        <div className="flex-1">
                          <p className="text-sm">
                            {assignment.tool?.name || `Tool ${assignment.tool_id}`} → Agent {assignment.agent_id}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {assignment.enabled ? 'Active' : 'Disabled'}
                          </p>
                        </div>
                      </div>
                    ))}
                    {(toolAssignments as any[]).length === 0 && (
                      <div className="text-center py-4">
                        <p className="text-sm text-muted-foreground">No tool assignments yet</p>
                      </div>
                    )}
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
  const statusColor = statusColors[tool?.status as keyof typeof statusColors] || 'text-gray-400'
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
    // Compact list view
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 20 }}
        transition={{ delay: index * 0.05 }}
      >
        <Card className="glass-card hover:border-primary/20 transition-all">
          <CardContent className="p-4">
            <div className="flex items-center gap-3 justify-between">
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <ToolLogo
                  logo={tool?.logo}
                  name={tool?.name}
                  size={40}
                  fallbackIcon={tool?.icon}
                  showBackground={true}
                />
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold truncate">{tool?.name}</h3>
                  <Badge variant="outline" className="mt-1 capitalize text-xs">
                    {tool?.category || 'Uncategorized'}
                  </Badge>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onDetails}
                >
                  View
                </Button>
                {isConnected ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-blue-400 border border-blue-400/30 hover:bg-blue-400/10"
                    onClick={onUninstall} // Mapped to Manage
                  >
                    <Settings className="w-4 h-4 mr-2" />
                    Manage
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    className="bg-orange-600 hover:bg-orange-700 text-white"
                    onClick={onInstall} // Mapped to Connect
                  >
                    <ExternalLink className="w-4 h-4 mr-2" />
                    Connect
                  </Button>
                )}
              </div>
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
      <Card className={`bg-secondary/30 border-border/30 hover:border-orange-500/30 transition-all duration-200 group hover:shadow-lg ${isConnected ? 'border-green-500/30 bg-green-500/5' : ''}`}>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-3">
              <ToolLogo
                logo={tool?.logo}
                name={tool?.name}
                size={40}
                fallbackIcon={tool?.icon}
                showBackground={true}
              />
              <div>
                <h3 className="font-semibold">{tool?.name}</h3>
                <p className="text-xs text-muted-foreground">{tool?.provider}</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              {isConnected && <Badge className="bg-green-500/20 text-green-400 border-none text-xs px-2 py-0.5">Connected</Badge>}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground line-clamp-2 min-h-[40px]">
            {tool?.description}
          </p>

          <div className="flex items-center justify-between text-xs">
            {tool?.rating ? (
              <div className="flex items-center space-x-1">
                <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                <span>{tool.rating}</span>
              </div>
            ) : (
              <div />
            )}
            <div className="flex gap-1">
              {tool?.tags?.slice(0, 2)?.map((tag) => (
                <Badge key={tag} variant="outline" className="text-[10px] h-5">
                  {tag}
                </Badge>
              )) || []}
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <Wrench className="w-3 h-3" />
              <span>{typeof toolsCount === 'number' ? toolsCount : 0} Tools</span>
            </div>
            <div className="flex items-center gap-1">
              <Zap className="w-3 h-3" />
              <span>{typeof triggersCount === 'number' ? triggersCount : 0} Triggers</span>
            </div>
          </div>

          {authSchemes.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {authSchemes.slice(0, 2).map((scheme: string) => (
                <Badge
                  key={scheme}
                  variant="outline"
                  className="text-[10px] h-5 border-orange-500/40 text-orange-300"
                >
                  {formatAuthScheme(scheme)}
                </Badge>
              ))}
            </div>
          )}

          <Separator />
          {/* Action Section */}
          <div className="flex items-center justify-between">
            <Button variant="ghost" size="sm" onClick={onDetails} className="text-muted-foreground hover:text-white p-0 h-auto">
              Details
            </Button>

            {isConnected ? (
              <Button
                size="sm"
                variant="secondary"
                className="w-24 bg-secondary/50 hover:bg-secondary border border-white/10"
                onClick={onUninstall} // Mapped to Manage
              >
                <Settings className="w-3 h-3 mr-2" />
                Manage
              </Button>
            ) : (
              <Button
                size="sm"
                className="w-24 bg-orange-600 hover:bg-orange-700 text-white shadow-lg shadow-orange-900/20"
                onClick={onInstall} // Mapped to Connect
                style={{ height: '32px' }}
              >
                Connect
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
