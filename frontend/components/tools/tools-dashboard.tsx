
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
}

export function ToolsDashboard() {
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [sortBy, setSortBy] = useState('name')

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

  const { data: mcpToolsData, isLoading: toolsLoading, error: toolsError } = useMCPTools({
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

  // Extract tools and pagination data
  const mcpTools = mcpToolsData?.data || []
  const paginationData = mcpToolsData?.pagination || { total: 0, pages: 0, current_page: 1 }

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
          id: cat.id || cat.name?.toLowerCase().replace(/\s+/g, '_'),
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
  const [selectedTool, setSelectedTool] = useState<any | null>(null)
  // const [assignmentModalOpen, setAssignmentModalOpen] = useState(false)

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
      filtered = filtered.filter(tool => tool?.category === selectedCategory)
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
        installed: (statsData as any).assigned_tools || (statsData as any).enabled_tools || 0,
        configured: (statsData as any).configured_tools || (statsData as any).assigned_tools || 0,
        available: (statsData as any).active_tools || (statsData as any).available_tools || 0,
        total: (statsData as any).total_tools || 0
      }
    }

    console.log('Stats API not available, using fallback. Tools on page:', tools.length)

    // Fallback: Use pagination total for overall count, but this won't give accurate enabled count
    // The enabled count will be wrong unless we fetch all tools
    const installed = tools.filter(tool => tool?.isInstalled)?.length || 0
    const configured = tools.filter(tool => tool?.isConfigured)?.length || 0
    const available = tools.filter(tool => tool?.status === 'active')?.length || 0

    return {
      installed, // This will be inaccurate - only counts current page
      configured,
      available,
      total: paginationData.total || tools?.length || 0
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

  // Show loading state with skeleton components
  if (toolsLoading) {
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
    setConfigModalOpen(true)
  }

  const handleToolDetails = (tool: Tool) => {
    setSelectedTool(tool)
    setDetailsModalOpen(true)
  }

  const handleToolDelete = async (tool: Tool) => {
    setLoading(true)
    try {
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

  const stats = {
    total_tools: paginationData.total || 0, // Use total from pagination
    active_tools: (mcpTools as any[]).filter((tool: any) => tool.status === 'active').length,
    assigned_tools: (toolAssignments as any[]).filter((assignment: any) => assignment.enabled).length,
    total_agents: (toolAssignments as any[]).reduce((acc: any, assignment: any) => {
      if (assignment.enabled && !acc.includes(assignment.agent_id)) {
        acc.push(assignment.agent_id)
      }
      return acc
    }, []).length,
    installed: enabledToolsCount || 0, // Use full enabled tools count
    configured: (toolAssignments as any[]).filter((assignment: any) => assignment.enabled).length,
    available: paginationData.total || 0, // Use total from pagination for available
    total: paginationData.total || 0 // Use total from pagination
  }

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
            label: 'Enabled Tools',
            value: stats.installed.toString(),
            change: `+${stats.installed} this week`,
            icon: CheckCircle,
            color: 'text-green-400'
          },
          {
            label: 'Configured',
            value: stats.configured.toString(),
            change: `+${stats.configured} ready`,
            icon: Settings,
            color: 'text-blue-400'
          },
          {
            label: 'Available',
            value: stats.available.toString(),
            change: 'Ready to install',
            icon: Zap,
            color: 'text-purple-400'
          },
          {
            label: 'Total Tools',
            value: stats.total.toString(),
            change: 'In marketplace',
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

      {/* Search and Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="flex flex-col sm:flex-row gap-4"
      >
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
          <Input
            placeholder="Search tools by name, provider, or tags..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
          />
        </div>


      </motion.div>

      {/* Main Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <Tabs defaultValue="enabled" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="enabled" className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4" />
              <span className="hidden sm:inline">Enabled ({enabledToolsCount})</span>
            </TabsTrigger>
            <TabsTrigger value="marketplace" className="flex items-center space-x-2">
              <Grid3X3 className="w-4 h-4" />
              <span className="hidden sm:inline">Marketplace</span>
            </TabsTrigger>
            {/* Security tab hidden - not implemented yet */}
            {/* <TabsTrigger value="security" className="flex items-center space-x-2">
              <Shield className="w-4 h-4" />
              <span className="hidden sm:inline">Security</span>
            </TabsTrigger> */}
          </TabsList>

          <TabsContent value="enabled" className="space-y-6">
            {/* Enabled Tools Management */}
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">Enabled Tools</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <AnimatePresence>
                  {enabledTools
                    .filter(tool => tool?.isInstalled)
                    .map((tool, index) => (
                      <ToolCard
                        key={tool?.id}
                        tool={tool}
                        viewMode="grid"
                        index={index}
                        onInstall={() => handleToolToggle(tool, true)}
                        onDetails={() => handleToolDetails(tool)}
                        onUninstall={() => handleToolToggle(tool, false)}
                        onConfigure={() => handleToolConfigure(tool)}
                        loading={loading}
                        showMenu={true}
                      />
                    ))}
                </AnimatePresence>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="marketplace" className="space-y-6">
            {/* Additional Filters and View Options */}
            <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
              <div className="flex flex-col md:flex-row gap-4 flex-1">
                {/* Sort */}
                <Select value={sortBy} onValueChange={setSortBy}>
                  <SelectTrigger className="w-48">
                    <SelectValue placeholder="Sort by..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="name">Name</SelectItem>
                    <SelectItem value="rating">Rating</SelectItem>
                    <SelectItem value="usage">Usage</SelectItem>
                    <SelectItem value="updated">Last Updated</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* View Mode Toggle */}
              <div className="flex items-center space-x-2 bg-secondary/30 rounded-lg p-1">
                <Button
                  variant={viewMode === 'grid' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('grid')}
                  className="h-8 w-8 p-0"
                >
                  <Grid3X3 className="w-4 h-4" />
                </Button>
                <Button
                  variant={viewMode === 'list' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('list')}
                  className="h-8 w-8 p-0"
                >
                  <List className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Categories */}
            <div className="flex flex-wrap gap-2">
              {toolCategories.map((category) => (
                <Button
                  key={category.id}
                  variant={selectedCategory === category.id ? 'default' : 'outline'}
                  onClick={() => setSelectedCategory(category.id)}
                  className={`flex items-center space-x-2 ${selectedCategory === category.id
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
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <AnimatePresence>
                {filteredTools.map((tool, index) => (
                  <ToolCard
                    key={tool?.id}
                    tool={tool}
                    viewMode={viewMode}
                    index={index}
                    onInstall={() => handleToolToggle(tool, true)}
                    onDetails={() => handleToolDetails(tool)}
                    onUninstall={() => handleToolToggle(tool, false)}
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
        onInstall={() => {
          if (selectedTool) {
            handleToolToggle(selectedTool, true)
            setDetailsModalOpen(false)
          }
        }}
        onConfigure={() => {
          setDetailsModalOpen(false)
          setConfigModalOpen(true)
        }}
        onUninstall={() => {
          if (selectedTool) {
            handleToolToggle(selectedTool, false)
            setDetailsModalOpen(false)
          }
        }}
        loading={loading}
      />

      <ToolConfigModal
        open={configModalOpen}
        onClose={() => setConfigModalOpen(false)}
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
  onInstall,
  onDetails,
  onUninstall,
  onConfigure,
  loading,
  showMenu = false
}: ToolCardProps) {
  const StatusIcon = statusIcons[tool?.status as keyof typeof statusIcons] || CheckCircle
  const statusColor = statusColors[tool?.status as keyof typeof statusColors] || 'text-gray-400'

  if (viewMode === 'list') {
    // Compact list view - still 3 columns but minimal card content
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
              {/* Toggle button */}
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onDetails}
                >
                  View
                </Button>
                <button
                  onClick={() => {
                    console.log('Toggle clicked:', tool.name, !tool?.isInstalled)
                    tool?.isInstalled ? onUninstall() : onInstall()
                  }}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 ${tool?.isInstalled ? 'bg-orange-500' : 'bg-gray-600'
                    }`}
                >
                  <span className="sr-only">Toggle tool</span>
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${tool?.isInstalled ? 'translate-x-6' : 'translate-x-1'
                      }`}
                  />
                </button>
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
      <Card className="bg-secondary/30 border-border/30 hover:border-orange-500/30 transition-all duration-200 group hover:shadow-lg">
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
              <StatusIcon className={`w-4 h-4 ${statusColor}`} />
              {showMenu && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <MoreVertical className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={onDetails}>
                      <Eye className="w-4 h-4 mr-2" />
                      View Details
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={onConfigure}>
                      <Settings className="w-4 h-4 mr-2" />
                      Configure
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground line-clamp-2">
            {tool?.description}
          </p>

          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-1">
              <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
              <span>{tool?.rating}</span>
            </div>
            <span className="text-muted-foreground">{tool?.pricing}</span>
          </div>

          <div className="flex flex-wrap gap-1">
            {tool?.tags?.slice(0, 2)?.map((tag) => (
              <Badge key={tag} variant="outline" className="text-xs">
                {tag}
              </Badge>
            )) || []}
          </div>

          <Separator />
          {/* Action Section */}
          <div className="flex items-center justify-between">
            <Button variant="outline" size="sm" onClick={onDetails}>
              View Details
            </Button>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  console.log('Toggle clicked:', tool.name, !tool?.isInstalled)
                  tool?.isInstalled ? onUninstall() : onInstall()
                }}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 ${tool?.isInstalled ? 'bg-orange-500' : 'bg-gray-600'
                  }`}
              >
                <span className="sr-only">Toggle tool</span>
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${tool?.isInstalled ? 'translate-x-6' : 'translate-x-1'
                    }`}
                />
              </button>
              <span className="text-sm text-muted-foreground">
                {tool?.isInstalled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
