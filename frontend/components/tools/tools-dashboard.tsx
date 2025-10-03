
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

import React, { useState, useEffect } from 'react'
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
  Activity
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
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
    icon: Grid3X3,
    color: 'text-gray-400',
    count: 0
  },
  {
    id: 'developer',
    name: 'Developer Tools',
    icon: '⚡',
    color: 'text-blue-400',
    description: 'Code repositories, project management, CI/CD',
    count: 12
  },
  {
    id: 'communication',
    name: 'Communication',
    icon: '💬',
    color: 'text-green-400',
    description: 'Team chat, email, video conferencing',
    count: 8
  },
  {
    id: 'cloud',
    name: 'Cloud Services',
    icon: '☁️',
    color: 'text-purple-400',
    description: 'AWS, Azure, GCP infrastructure tools',
    count: 15
  },
  {
    id: 'analytics',
    name: 'Analytics',
    icon: '📊',
    color: 'text-orange-400',
    description: 'Data analysis, monitoring, reporting',
    count: 10
  },
  {
    id: 'productivity',
    name: 'Productivity',
    icon: '✨',
    color: 'text-pink-400',
    description: 'Task management, documentation, scheduling',
    count: 7
  },
  {
    id: 'security',
    name: 'Security',
    icon: '🛡️',
    color: 'text-red-400',
    description: 'Security scanning, compliance, monitoring',
    count: 6
  }
]

// Mock tools data (in real app, this would come from API)
const mockTools = [
  // Developer Tools
  {
    id: 1,
    name: 'GitHub',
    description: 'Code repository management and version control',
    category: 'developer',
    icon: '⚡',
    provider: 'GitHub Inc.',
    status: 'available',
    isInstalled: true,
    isConfigured: true,
    version: '2.0.1',
    pricing: 'Free/Pro',
    rating: 4.9,
    usageCount: 1250,
    tags: ['git', 'repository', 'collaboration'],
    permissions: ['code_architect', 'custom'],
    requiredCredentials: ['api_token'],
    supportedEnvironments: ['development', 'staging', 'production'],
    lastUpdated: '2024-01-15',
    configuration: {
      webhook_url: 'configured',
      branch_protection: true,
      auto_merge: false
    }
  },
  {
    id: 2,
    name: 'Jira',
    description: 'Project management and issue tracking',
    category: 'developer',
    icon: '📋',
    provider: 'Atlassian',
    status: 'available',
    isInstalled: false,
    isConfigured: false,
    version: '1.8.3',
    pricing: 'Free/Pro',
    rating: 4.6,
    usageCount: 890,
    tags: ['project-management', 'tracking', 'agile'],
    permissions: ['code_architect', 'custom'],
    requiredCredentials: ['api_token', 'domain'],
    supportedEnvironments: ['development', 'production'],
    lastUpdated: '2024-01-10'
  },
  // Communication Tools
  {
    id: 3,
    name: 'Slack',
    description: 'Team communication and collaboration platform',
    category: 'communication',
    icon: '💬',
    provider: 'Slack Technologies',
    status: 'available',
    isInstalled: true,
    isConfigured: true,
    version: '3.2.1',
    pricing: 'Free/Pro',
    rating: 4.8,
    usageCount: 2100,
    tags: ['chat', 'team', 'notifications'],
    permissions: ['custom', 'infrastructure_manager'],
    requiredCredentials: ['bot_token', 'webhook_url'],
    supportedEnvironments: ['production'],
    lastUpdated: '2024-01-12',
    configuration: {
      channels_configured: 3,
      notifications_enabled: true,
      bot_active: true
    }
  },
  {
    id: 4,
    name: 'Gmail',
    description: 'Email management and automation',
    category: 'communication',
    icon: '📧',
    provider: 'Google',
    status: 'available',
    isInstalled: false,
    isConfigured: false,
    version: '2.1.0',
    pricing: 'Free',
    rating: 4.7,
    usageCount: 560,
    tags: ['email', 'automation', 'notifications'],
    permissions: ['custom'],
    requiredCredentials: ['oauth2_token'],
    supportedEnvironments: ['production'],
    lastUpdated: '2024-01-08'
  },
  // Cloud Services
  {
    id: 5,
    name: 'AWS S3',
    description: 'Cloud object storage and file management',
    category: 'cloud',
    icon: '☁️',
    provider: 'Amazon Web Services',
    status: 'available',
    isInstalled: true,
    isConfigured: false,
    version: '1.9.2',
    pricing: 'Pay-per-use',
    rating: 4.9,
    usageCount: 1800,
    tags: ['storage', 'cloud', 'files'],
    permissions: ['infrastructure_manager'],
    requiredCredentials: ['access_key', 'secret_key', 'region'],
    supportedEnvironments: ['development', 'staging', 'production'],
    lastUpdated: '2024-01-14'
  },
  {
    id: 6,
    name: 'Docker',
    description: 'Container management and deployment',
    category: 'cloud',
    icon: '🐳',
    provider: 'Docker Inc.',
    status: 'available',
    isInstalled: true,
    isConfigured: true,
    version: '4.1.0',
    pricing: 'Free/Pro',
    rating: 4.8,
    usageCount: 920,
    tags: ['containers', 'deployment', 'orchestration'],
    permissions: ['infrastructure_manager', 'code_architect'],
    requiredCredentials: ['registry_token'],
    supportedEnvironments: ['development', 'staging', 'production'],
    lastUpdated: '2024-01-13',
    configuration: {
      registries_connected: 2,
      images_managed: 15,
      auto_deployment: true
    }
  },
  // Analytics Tools
  {
    id: 7,
    name: 'Google Analytics',
    description: 'Web analytics and user behavior tracking',
    category: 'analytics',
    icon: '📊',
    provider: 'Google',
    status: 'available',
    isInstalled: false,
    isConfigured: false,
    version: '2.3.1',
    pricing: 'Free/Pro',
    rating: 4.5,
    usageCount: 340,
    tags: ['analytics', 'tracking', 'insights'],
    permissions: ['data_analyst'],
    requiredCredentials: ['property_id', 'service_account_key'],
    supportedEnvironments: ['production'],
    lastUpdated: '2024-01-11'
  },
  {
    id: 8,
    name: 'DataDog',
    description: 'Application performance monitoring and observability',
    category: 'analytics',
    icon: '🔍',
    provider: 'DataDog Inc.',
    status: 'available',
    isInstalled: false,
    isConfigured: false,
    version: '3.0.2',
    pricing: 'Pro',
    rating: 4.7,
    usageCount: 180,
    tags: ['monitoring', 'apm', 'logs'],
    permissions: ['infrastructure_manager', 'performance_optimizer'],
    requiredCredentials: ['api_key', 'app_key'],
    supportedEnvironments: ['staging', 'production'],
    lastUpdated: '2024-01-09'
  }
]

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
}

export function ToolsDashboard() {
  // Fetch real data from API
  const { data: mcpTools = [], isLoading: toolsLoading } = useMCPTools({ limit: 100 })
  const { data: statsData } = useMCPToolsStats()
  const { data: categoriesData } = useMCPToolCategories()
  const { data: toolAssignments = [] } = useMCPToolAssignments()
  const updateToolMutation = useUpdateMCPTool()

  const [filteredTools, setFilteredTools] = useState<any[]>([])
  const [tools, setTools] = useState<any[]>([])
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [sortBy, setSortBy] = useState('name')
  const [loading, setLoading] = useState(false)
  
  // Modal states
  const [configModalOpen, setConfigModalOpen] = useState(false)
  const [selectedTool, setSelectedTool] = useState<any | null>(null)
  const [assignmentModalOpen, setAssignmentModalOpen] = useState(false)
  const [createToolModalOpen, setCreateToolModalOpen] = useState(false) // Phase 3

  // Convert MCP tools to match Tool interface for UI compatibility
  useEffect(() => {
    const convertedTools = (mcpTools as any[]).map((tool: any) => {
      // Check if this tool is assigned to any agent
      const isAssigned = (toolAssignments as any[]).some((assignment: any) => 
        assignment.tool_id === tool.id && assignment.enabled
      )
      
      return {
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
    })
    setTools(convertedTools)
  }, [mcpTools, toolAssignments])

  useEffect(() => {
    filterTools()
  }, [selectedCategory, searchQuery, sortBy, tools])

  const filterTools = () => {
    let filtered = tools

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

    setFilteredTools(filtered)
  }

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

  // Show loading state
  if (toolsLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading tools...</p>
        </div>
      </div>
    )
  }

  const handleToolInstall = async (tool: Tool) => {
    setLoading(true)
    try {
      console.log(`Installing tool: ${tool.name} (ID: ${tool.id})`)
      
      // Use the mutation hook which automatically invalidates queries
      await updateToolMutation.mutateAsync({
        id: tool.id,
        data: { status: 'active' }
      })
      
      console.log('Tool installed successfully')
      
    } catch (error) {
      console.error('Failed to install tool:', error)
      alert(`❌ Failed to install ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const handleToolConfigure = (tool: Tool) => {
    setSelectedTool(tool)
    setConfigModalOpen(true)
  }

  const handleToolUninstall = async (tool: Tool) => {
    setLoading(true)
    try {
      console.log(`Uninstalling tool: ${tool.name} (ID: ${tool.id})`)
      
      // Use the mutation hook which automatically invalidates queries
      await updateToolMutation.mutateAsync({
        id: tool.id,
        data: { status: 'inactive' }
      })
      
      console.log('Tool uninstalled successfully')
      
    } catch (error) {
      console.error('Failed to uninstall tool:', error)
      alert(`❌ Failed to uninstall ${tool.name}: ${error instanceof Error ? error.message : 'Unknown error'}`)
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
    installed: (mcpTools as any[]).filter((tool: any) => tool.status === 'active').length,
    configured: (toolAssignments as any[]).filter((assignment: any) => assignment.enabled).length,
    available: (mcpTools as any[]).filter((tool: any) => tool.status === 'available').length,
    total: (mcpTools as any[]).length
  }

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold mb-2">
            Tools & <span className="gradient-text">Integrations</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Discover, install, and manage tools to extend your AI agents' capabilities
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Button
            variant="outline"
            onClick={() => setAssignmentModalOpen(true)}
            className="hover:border-blue-500/50"
          >
            <Users className="w-4 h-4 mr-2" />
            Agent Assignment
          </Button>
          <Button 
            className="gradient-accent hover:opacity-90 transition-opacity"
            onClick={() => setCreateToolModalOpen(true)}
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Custom Tool
          </Button>
        </div>
      </motion.div>

      {/* Statistics Cards */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
          <motion.div
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-green-400" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{stats.installed}</h3>
              <p className="text-muted-foreground text-sm">Installed</p>
              <p className="text-xs text-green-400">+{stats.installed} this week</p>
            </div>
          </motion.div>
          
          <motion.div
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <Settings className="w-5 h-5 text-blue-400" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{stats.configured}</h3>
              <p className="text-muted-foreground text-sm">Configured</p>
              <p className="text-xs text-blue-400">+{stats.configured} ready</p>
            </div>
          </motion.div>
          
          <motion.div
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <Zap className="w-5 h-5 text-purple-400" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{stats.available}</h3>
              <p className="text-muted-foreground text-sm">Available</p>
              <p className="text-xs text-purple-400">Ready to install</p>
            </div>
          </motion.div>
          
          <motion.div
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-orange-400" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{stats.total}</h3>
              <p className="text-muted-foreground text-sm">Total Tools</p>
              <p className="text-xs text-orange-400">In marketplace</p>
            </div>
          </motion.div>
        </motion.div>

      {/* Main Content */}
      <Tabs defaultValue="marketplace" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 bg-secondary/50">
          <TabsTrigger value="marketplace" className="flex items-center space-x-2">
            <Grid3X3 className="w-4 h-4" />
            <span>Marketplace</span>
          </TabsTrigger>
          <TabsTrigger value="installed" className="flex items-center space-x-2">
            <CheckCircle className="w-4 h-4" />
            <span>Installed ({stats.installed})</span>
          </TabsTrigger>
          <TabsTrigger value="security" className="flex items-center space-x-2">
            <Shield className="w-4 h-4" />
            <span>Security</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="marketplace" className="space-y-6">
          {/* Filters and Search */}
          <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
            <div className="flex flex-col md:flex-row gap-4 flex-1">
              {/* Search */}
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search tools, providers, or tags..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>

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
                {typeof category.icon === 'string' ? (
                  <span className="text-sm">{category.icon}</span>
                ) : (
                  <category.icon className="w-4 h-4" />
                )}
                <span>{category.name}</span>
                <Badge variant="outline" className="ml-1 text-xs">
                  {getCategoryCount(category.id)}
                </Badge>
              </Button>
            ))}
          </div>

          {/* Tools Grid/List */}
          <div className={`grid gap-4 ${
            viewMode === 'grid' 
              ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4' 
              : 'grid-cols-1'
          }`}>
            <AnimatePresence>
              {filteredTools.map((tool, index) => (
                <ToolCard
                  key={tool?.id}
                  tool={tool}
                  viewMode={viewMode}
                  index={index}
                  onInstall={() => handleToolInstall(tool)}
                  onConfigure={() => handleToolConfigure(tool)}
                  onUninstall={() => handleToolUninstall(tool)}
                  loading={loading}
                />
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
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
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

      {/* Modals */}
      <ToolConfigModal
        open={configModalOpen}
        onClose={() => setConfigModalOpen(false)}
        tool={selectedTool}
        onSave={(toolId, config) => {
          setTools(prevTools => 
            prevTools.map(t => 
              t?.id === toolId 
                ? { ...t, isConfigured: true, configuration: config }
                : t
            )
          )
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
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                  <span className="text-lg">{tool?.icon}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold">{tool?.name}</h3>
                    <Badge variant="outline">{tool?.category}</Badge>
                    <StatusIcon className={`w-4 h-4 ${statusColor}`} />
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">{tool?.description}</p>
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
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
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
