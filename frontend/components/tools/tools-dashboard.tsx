
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

import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, 
  Filter, 
  Grid3X3, 
  List, 
  Plus,
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
  Edit,
  Power,
  PowerOff
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
import { AgentToolAssignment } from './agent-tool-assignment'
import { CreateToolModal } from './create-tool-modal'
import { useMCPTools, useMCPToolsStats, useMCPToolCategories, useMCPToolAssignments, useUpdateMCPTool } from '@/hooks/use-mcp-tools-api'

// Tool Categories
const toolCategories = [
  {
    id: 'all',
    name: 'All Tools',
    color: 'text-gray-400',
    count: 0
  },
  {
    id: 'developer',
    name: 'Developer Tools',
    color: 'text-blue-400',
    description: 'Code repositories, project management, CI/CD',
    count: 12
  },
  {
    id: 'communication',
    name: 'Communication',
    color: 'text-green-400',
    description: 'Team chat, email, video conferencing',
    count: 8
  },
  {
    id: 'cloud',
    name: 'Cloud Services',
    color: 'text-purple-400',
    description: 'AWS, Azure, GCP infrastructure tools',
    count: 15
  },
  {
    id: 'analytics',
    name: 'Analytics',
    color: 'text-orange-400',
    description: 'Data analysis, monitoring, reporting',
    count: 10
  },
  {
    id: 'productivity',
    name: 'Productivity',
    color: 'text-pink-400',
    description: 'Task management, documentation, scheduling',
    count: 7
  },
  {
    id: 'security',
    name: 'Security',
    color: 'text-red-400',
    description: 'Security scanning, compliance, monitoring',
    count: 6
  }
]

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
}

export function ToolsDashboard() {
  // Fetch real data from API with error handling
  const { data: mcpTools = [], isLoading: toolsLoading, error: toolsError } = useMCPTools({ limit: 100 })
  const { data: statsData, error: statsError } = useMCPToolsStats()
  const { data: categoriesData, error: categoriesError } = useMCPToolCategories()
  // Temporarily disable assignments API until backend endpoint is ready
  // const { data: toolAssignments = [], error: assignmentsError } = useMCPToolAssignments(false)
  const toolAssignments: any[] = [] // Mock empty assignments for now
  const updateToolMutation = useUpdateMCPTool()

  // Log any API errors (but don't spam the console)
  if (toolsError && !toolsError.message?.includes('422')) console.error('Tools API Error:', toolsError)
  if (statsError && !statsError.message?.includes('422')) console.error('Stats API Error:', statsError)
  if (categoriesError && !categoriesError.message?.includes('422')) console.error('Categories API Error:', categoriesError)

  const [selectedCategory, setSelectedCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [sortBy, setSortBy] = useState('name')
  const [loading, setLoading] = useState(false)
  const [toolModifications, setToolModifications] = useState<Record<number, Partial<Tool>>>({})
  
  // Modal states
  const [configModalOpen, setConfigModalOpen] = useState(false)
  const [selectedTool, setSelectedTool] = useState<any | null>(null)
  const [assignmentModalOpen, setAssignmentModalOpen] = useState(false)
  const [createToolModalOpen, setCreateToolModalOpen] = useState(false) // Phase 3

  // Convert MCP tools to match Tool interface for UI compatibility
  const tools = useMemo(() => {
    try {
      return (mcpTools as any[]).map((tool: any) => {
        // Check if this tool is assigned to any agent
        const isAssigned = (toolAssignments as any[]).some((assignment: any) => 
          assignment.tool_id === tool.id && assignment.enabled
        )
        
        const baseTool = {
          ...tool,
          isInstalled: isAssigned, // Use real assignment status
          isConfigured: isAssigned, // Assume configured if assigned
          rating: 0, // TODO: Add ratings system
          usageCount: 0, // TODO: Track from usage logs
          permissions: [], // TODO: Map from tool data
          requiredCredentials: Object.keys(tool.credentials_schema?.required || {}),
          supportedEnvironments: ['production', 'staging'], // TODO: Get from tool metadata
          lastUpdated: tool.updated_at,
          pricing: 'Free' // TODO: Add pricing info to backend
        }
        
        // Apply any local modifications
        const modifications = toolModifications[tool.id]
        return modifications ? { ...baseTool, ...modifications } : baseTool
      })
    } catch (error) {
      console.error('Error converting tools:', error)
      return []
    }
  }, [mcpTools, toolAssignments, toolModifications])

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
      return {
        installed: (statsData as any).assigned_tools || 0,
        configured: (statsData as any).assigned_tools || 0,
        available: (statsData as any).active_tools || 0,
        total: (statsData as any).total_tools || 0
      }
    }
    
    // Fallback to calculated stats
    const installed = tools.filter(tool => tool?.isInstalled)?.length || 0
    const configured = tools.filter(tool => tool?.isConfigured)?.length || 0
    const available = tools.filter(tool => tool?.status === 'active')?.length || 0
    
    return { installed, configured, available, total: tools?.length || 0 }
  }

  const getCategoryCount = (categoryId: string) => {
    if (categoryId === 'all') return tools?.length || 0
    
    // Use real category counts from API if available
    if (categoriesData && Array.isArray(categoriesData) && categoryId !== 'all') {
      const category = (categoriesData as any[]).find((c: any) => c.name === categoryId)
      if (category) return category.count
    }
    
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
      
      // For now, just update the tool status in the database
      // TODO: Later we should manage agent-tool assignments instead
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
    total_tools: (mcpTools as any[]).length,
    active_tools: (mcpTools as any[]).filter((tool: any) => tool.status === 'active').length,
    assigned_tools: (toolAssignments as any[]).filter((assignment: any) => assignment.enabled).length,
    total_agents: (toolAssignments as any[]).reduce((acc: any, assignment: any) => {
      if (assignment.enabled && !acc.includes(assignment.agent_id)) {
        acc.push(assignment.agent_id)
      }
      return acc
    }, []).length,
    installed: tools.filter(tool => tool?.isInstalled)?.length || 0, // Use actual enabled tools count
    configured: (toolAssignments as any[]).filter((assignment: any) => assignment.enabled).length,
    available: (mcpTools as any[]).filter((tool: any) => tool.status === 'available').length,
    total: (mcpTools as any[]).length
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
            {toolsLoading ? 'Loading...' : `${(mcpTools as any[])?.length || 0} Tools`}
          </Badge>
          
          <Button
            variant="outline"
            onClick={() => setAssignmentModalOpen(true)}
            className="hover:border-blue-500/50"
          >
            <Users className="w-4 h-4 mr-2" />
            Agent Assignment
          </Button>
          
          <Button 
            className="bg-brand-primary hover:bg-brand-primary/90"
            onClick={() => setCreateToolModalOpen(true)}
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Custom Tool
          </Button>
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
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                  <stat.icon className={`w-5 h-5 ${
                    index === 0 ? 'text-green-400' :
                    index === 1 ? 'text-blue-400' :
                    index === 2 ? 'text-purple-400' :
                    index === 3 ? 'text-orange-400' :
                    'text-white'
                  }`} />
                </div>
              </div>
              <div className="space-y-1">
                <h3 className="text-2xl font-bold">
                  {toolsLoading ? '...' : stat.value}
                </h3>
                <div className="text-muted-foreground text-sm">{stat.label}</div>
                <div className={`text-xs ${stat.color}`}>
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
        
        <div className="flex gap-2">
          <Button
            variant={selectedCategory === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory('all')}
          >
            All
          </Button>
          <Button
            variant={selectedCategory === 'developer' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory('developer')}
          >
            Developer
          </Button>
          <Button
            variant={selectedCategory === 'communication' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory('communication')}
          >
            Communication
          </Button>
          <Button
            variant={selectedCategory === 'cloud' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory('cloud')}
          >
            Cloud
          </Button>
        </div>
      </motion.div>

      {/* Main Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <Tabs defaultValue="marketplace" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="marketplace" className="flex items-center space-x-2">
              <Grid3X3 className="w-4 h-4" />
              <span className="hidden sm:inline">Marketplace</span>
            </TabsTrigger>
            <TabsTrigger value="installed" className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4" />
              <span className="hidden sm:inline">Enabled ({stats.installed})</span>
            </TabsTrigger>
            <TabsTrigger value="security" className="flex items-center space-x-2">
              <Shield className="w-4 h-4" />
              <span className="hidden sm:inline">Security</span>
            </TabsTrigger>
          </TabsList>

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
                className={`flex items-center space-x-2 ${
                  selectedCategory === category.id 
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
                <motion.div
                  key={tool?.id}
                  className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  {/* Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                        <span className="text-lg">{tool?.icon}</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold truncate">{tool?.name}</h3>
                        <p className="text-xs text-muted-foreground">
                          {tool?.provider} • {tool?.category}
                        </p>
                      </div>
                    </div>
                    
                    {/* Context Menu - Top Right */}
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => handleToolConfigure(tool)}>
                          <Edit className="w-4 h-4 mr-2" />
                          Edit
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => handleToolDelete(tool)}
                          className="text-red-400"
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  {/* Status */}
                  <div className="flex items-center justify-between mb-4">
                    <Badge className={tool?.isInstalled 
                      ? 'bg-green-500/10 text-green-400 border-green-500/20' 
                      : 'bg-gray-500/10 text-gray-400 border-gray-500/20'
                    }>
                      {tool?.isInstalled ? 'installed' : 'available'}
                    </Badge>
                  </div>

                  {/* Description */}
                  <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                    {tool?.description}
                  </p>

                  {/* Tool Info */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-sm font-medium">Version</p>
                      <p className="text-xs text-muted-foreground">{tool?.version}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Rating</p>
                      <p className="text-xs text-muted-foreground">
                        <div className="flex items-center space-x-1">
                          <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                          <span>{tool?.rating}</span>
                        </div>
                      </p>
                    </div>
                  </div>

                  {/* Tags */}
                  <div className="flex flex-wrap gap-1 mb-4">
                    {tool?.tags?.slice(0, 2)?.map((tag: string) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    )) || []}
                  </div>

                  {/* Action Buttons */}
                  <div className="flex items-center justify-between">
                    {/* Custom Toggle Button */}
                    <button
                      onClick={() => {
                        console.log('Toggle clicked:', tool.name, !tool?.isInstalled)
                        handleToolToggle(tool, !tool?.isInstalled)
                      }}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 ${
                        tool?.isInstalled ? 'bg-orange-500' : 'bg-gray-600'
                      }`}
                    >
                      <span className="sr-only">Toggle tool</span>
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                          tool?.isInstalled ? 'translate-x-6' : 'translate-x-1'
                        }`}
                      />
                    </button>
                    <span className="text-sm text-muted-foreground">
                      {tool?.isInstalled ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

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

        <TabsContent value="installed" className="space-y-6">
          {/* Installed Tools Management */}
          <div className="space-y-4">
            <h3 className="text-xl font-semibold">Installed Tools</h3>
            <div className="grid gap-4">
              {tools.filter(tool => tool?.isInstalled).map((tool) => (
                <Card key={tool?.id} className="bg-secondary/30 border-border/30">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                  <span className="text-lg">{tool?.icon}</span>
                </div>
                        <div>
                          <h4 className="font-semibold">{tool?.name}</h4>
                          <p className="text-sm text-muted-foreground">{tool?.provider}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge className={tool?.isConfigured 
                          ? 'bg-green-500/10 text-green-400 border-green-500/20' 
                          : 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
                        }>
                          {tool?.isConfigured ? 'Configured' : 'Needs Config'}
                        </Badge>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => handleToolConfigure(tool)}
                        >
                          Configure
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

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
      <ToolConfigModal
        open={configModalOpen}
        onClose={() => setConfigModalOpen(false)}
        tool={selectedTool}
        onSave={(toolId, config) => {
          setToolModifications(prev => ({
            ...prev,
            [toolId]: { ...prev[toolId], isConfigured: true, configuration: config }
          }))
        }}
      />

      <AgentToolAssignment
        open={assignmentModalOpen}
        onClose={() => setAssignmentModalOpen(false)}
        tools={tools}
      />

      <CreateToolModal
        open={createToolModalOpen}
        onClose={() => setCreateToolModalOpen(false)}
      />
    </div>
  )
}

// Tool Card Component
interface ToolCardProps {
  tool: Tool
  viewMode: 'grid' | 'list'
  index: number
  onInstall: () => void
  onConfigure: () => void
  onUninstall: () => void
  loading: boolean
}

function ToolCard({ tool, viewMode, index, onInstall, onConfigure, onUninstall, loading }: ToolCardProps) {
  const StatusIcon = statusIcons[tool?.status as keyof typeof statusIcons] || CheckCircle
  const statusColor = statusColors[tool?.status as keyof typeof statusColors] || 'text-gray-400'

  if (viewMode === 'list') {
    return (
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ delay: index * 0.1 }}
      >
        <Card className="bg-secondary/30 border-border/30 hover:border-orange-500/30 transition-colors">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 flex-1">
                <div className="w-12 h-12 rounded-lg bg-secondary/50 flex items-center justify-center">
                  <span className="text-lg">{tool?.icon}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold">{tool?.name}</h3>
                    <Badge variant="outline">{tool?.category}</Badge>
                    <StatusIcon className={`w-4 h-4 ${statusColor}`} />
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">{tool?.description}</div>
                  <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                    <span>{tool?.provider}</span>
                    <span>v{tool?.version}</span>
                    <span>{tool?.pricing}</span>
                    <div className="flex items-center space-x-1">
                      <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                      <span>{tool?.rating}</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {tool?.isInstalled ? (
                  <>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={onConfigure}
                      className="hover:border-blue-500/50"
                    >
                      <Settings className="w-4 h-4 mr-1" />
                      Configure
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={onUninstall}
                      className="hover:border-red-500/50 text-red-400"
                      disabled={loading}
                    >
                      Remove
                    </Button>
                  </>
                ) : (
                  <Button 
                    onClick={onInstall}
                    disabled={loading}
                    className="bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
                    size="sm"
                  >
                    <Download className="w-4 h-4 mr-1" />
                    Install
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
      <Card className="bg-secondary/30 border-border/30 hover:border-orange-500/30 transition-all duration-200 group hover:shadow-lg">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <span className="text-lg">{tool?.icon}</span>
              </div>
              <div>
                <h3 className="font-semibold">{tool?.name}</h3>
                <p className="text-xs text-muted-foreground">{tool?.provider}</p>
              </div>
            </div>
            <div className="flex items-center space-x-1">
              <StatusIcon className={`w-4 h-4 ${statusColor}`} />
              {tool?.isInstalled && (
                <Badge className="bg-green-500/10 text-green-400 border-green-500/20 text-xs">
                  Installed
                </Badge>
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

          <div className="flex space-x-2">
            {tool?.isInstalled ? (
              <>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex-1 hover:border-blue-500/50"
                  onClick={onConfigure}
                >
                  <Settings className="w-4 h-4 mr-1" />
                  Config
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={onUninstall}
                  className="hover:border-red-500/50 text-red-400"
                  disabled={loading}
                >
                  Remove
                </Button>
              </>
            ) : (
              <Button 
                onClick={onInstall}
                disabled={loading}
                className="w-full bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
                size="sm"
              >
                <Download className="w-4 h-4 mr-2" />
                Install
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
