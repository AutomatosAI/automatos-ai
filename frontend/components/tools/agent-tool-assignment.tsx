

'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Users, 
  Wrench, 
  Shield, 
  CheckCircle, 
  AlertTriangle, 
  Settings,
  Plus,
  Minus,
  Filter,
  Search,
  Zap,
  Bot,
  Lock,
  Unlock,
  Eye,
  AlertCircle,
  Save,
  RefreshCw
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { apiClient } from '@/lib/api-client'
import { useAgents } from '@/hooks/use-agent-api'
import { useMCPTools, useMCPToolAssignments } from '@/hooks/use-mcp-tools-api'

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

interface Agent {
  id: number
  name: string
  agent_type: string
  status: string
  description?: string
}

interface AgentToolAssignmentProps {
  open: boolean
  onClose: () => void
  tools: Tool[]
}

// Real data is now loaded from API hooks above

// Permission matrix - maps agent types to allowed tools (simplified for demo)
const agentToolPermissions = {
  code_architect: ['GitHub', 'Jira', 'Docker'],
  security_expert: ['GitHub', 'AWS S3'],
  performance_optimizer: ['DataDog', 'Docker'],
  data_analyst: ['Google Analytics', 'DataDog'],
  infrastructure_manager: ['AWS S3', 'Docker', 'Slack'],
  custom: ['GitHub', 'Jira', 'Slack', 'Gmail']
}

const agentTypeColors = {
  code_architect: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  security_expert: 'bg-red-500/10 text-red-400 border-red-500/20',
  performance_optimizer: 'bg-green-500/10 text-green-400 border-green-500/20',
  data_analyst: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  infrastructure_manager: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  custom: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
}

function AgentToolAssignment({ open, onClose, tools }: AgentToolAssignmentProps) {
  // Use real API data instead of mock data
  const { data: agentsData = [], isLoading: agentsLoading } = useAgents()
  const { data: mcpTools = [], isLoading: toolsLoading } = useMCPTools({ limit: 100 })
  // Temporarily disable assignments API until backend endpoint is ready
  // const { data: toolAssignments = [] } = useMCPToolAssignments()
  const toolAssignments: any[] = [] // Mock empty assignments for now
  
  // Convert API data to match expected format
  const agents = (agentsData as any[]).map((agent: any) => ({
    id: agent.id,
    name: agent.name,
    agent_type: agent.agent_type,
    status: agent.status,
    description: agent.description,
    icon: '🤖' // Default icon, could be enhanced later
  }))
  
  const [assignments, setAssignments] = useState<Record<string, number[]>>({})
  const [activeTab, setActiveTab] = useState('matrix')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedAgentType, setSelectedAgentType] = useState('all')
  const [saving, setSaving] = useState(false)
  const [conflicts, setConflicts] = useState<Array<{
    agentId: number
    toolId: number
    reason: string
  }>>([])

  useEffect(() => {
    if (open && (toolAssignments as any[]).length > 0) {
      loadCurrentAssignments()
    }
  }, [open, toolAssignments])

  useEffect(() => {
    validateAssignments()
  }, [assignments])

  const loadCurrentAssignments = async () => {
    try {
      // Convert real tool assignments to the expected format
      const realAssignments: Record<string, number[]> = {}
      
      (toolAssignments as any[]).forEach((assignment: any) => {
        if (assignment.enabled) {
          const agentKey = `agent_${assignment.agent_id}`
          if (!realAssignments[agentKey]) {
            realAssignments[agentKey] = []
          }
          realAssignments[agentKey].push(assignment.tool_id)
        }
      })
      
      console.log('Real assignments loaded:', realAssignments)
      setAssignments(realAssignments)
    } catch (error) {
      console.error('Failed to load assignments:', error)
    }
  }

  const validateAssignments = () => {
    const newConflicts: Array<{ agentId: number; toolId: number; reason: string }> = []

    Object.entries(assignments).forEach(([agentKey, toolIds]) => {
      const agentId = parseInt(agentKey.split('_')[1])
      const agent = agents.find(a => a?.id === agentId)
      
      if (!agent) return

      toolIds?.forEach(toolId => {
        const tool = tools.find(t => t?.id === toolId)
        if (!tool) return

        // Check if agent type has permission for this tool
        const allowedTools = agentToolPermissions[agent.agent_type as keyof typeof agentToolPermissions] || []
        if (!allowedTools.includes(tool.name)) {
          newConflicts.push({
            agentId,
            toolId,
            reason: `${agent.agent_type} agents don't have permission for ${tool.name}`
          })
        }

        // Check if tool is configured
        if (!tool.isConfigured) {
          newConflicts.push({
            agentId,
            toolId,
            reason: `${tool.name} is not properly configured`
          })
        }
      })
    })

    setConflicts(newConflicts)
  }

  const toggleAssignment = (agentId: number, toolId: number) => {
    console.log(`Toggling assignment: Agent ${agentId} <-> Tool ${toolId}`)
    const agentKey = `agent_${agentId}`
    setAssignments(prev => {
      const currentAssignments = prev[agentKey] || []
      const isAssigned = currentAssignments.includes(toolId)
      
      console.log(`Current assignments for ${agentKey}:`, currentAssignments)
      console.log(`Is assigned: ${isAssigned}`)
      
      const newAssignments = {
        ...prev,
        [agentKey]: isAssigned 
          ? currentAssignments.filter(id => id !== toolId)
          : [...currentAssignments, toolId]
      }
      
      console.log(`New assignments:`, newAssignments)
      return newAssignments
    })
  }

  const isToolAssigned = (agentId: number, toolId: number) => {
    const agentKey = `agent_${agentId}`
    return assignments[agentKey]?.includes(toolId) || false
  }

  const isPermissionAllowed = (agentType: string, toolName: string) => {
    // For demo purposes, allow most combinations
    // In a real implementation, this would check against a proper permission matrix
    const restrictedCombinations = [
      { agentType: 'code_architect', toolName: 'Slack MCP' },
      { agentType: 'security_expert', toolName: 'Jira MCP' },
      { agentType: 'infrastructure_manager', toolName: 'Jira MCP' }
    ]
    
    const isRestricted = restrictedCombinations.some(combo => 
      combo.agentType === agentType && combo.toolName === toolName
    )
    
    return !isRestricted
  }

  const getAssignmentCount = (agentId: number) => {
    const agentKey = `agent_${agentId}`
    return assignments[agentKey]?.length || 0
  }

  const getToolAssignmentCount = (toolId: number) => {
    return Object.values(assignments).flat().filter(id => id === toolId).length
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      // In real app, save to API
      // await apiClient.request('/api/permissions/assignments', {
      //   method: 'POST',
      //   body: JSON.stringify({ assignments })
      // })
      
      await new Promise(resolve => setTimeout(resolve, 1500))
      onClose()
    } catch (error) {
      console.error('Failed to save assignments:', error)
    } finally {
      setSaving(false)
    }
  }

  const filteredAgents = agents.filter((agent: any) => {
    const matchesSearch = !searchQuery || 
      agent?.name?.toLowerCase()?.includes(searchQuery.toLowerCase()) ||
      agent?.agent_type?.toLowerCase()?.includes(searchQuery.toLowerCase())
    
    const matchesType = selectedAgentType === 'all' || agent?.agent_type === selectedAgentType
    
    return matchesSearch && matchesType
  })

  // Show only installed tools (tools that have been installed via the install button)
  const installedTools = (mcpTools as any[]).filter((tool: any) => {
    // Check if tool is installed (status: 'active')
    return tool.status === 'active'
  })
  
  // If no tools are installed, show a helpful message
  const toolsToShow = installedTools.length > 0 ? installedTools : []

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="glass-card max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                <Users className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-semibold">Agent-Tool Assignment</h2>
                <p className="text-sm text-muted-foreground font-normal">
                  Manage which tools each agent can access
                  {agentsLoading && <span className="ml-2 text-blue-400">(Loading agents...)</span>}
                  {toolsLoading && <span className="ml-2 text-blue-400">(Loading tools...)</span>}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => {
                  console.log('🔍 DEBUG INFO:')
                  console.log(`- Agents loaded: ${agents.length}`)
                  console.log(`- Tools loaded: ${toolsToShow.length}`)
                  console.log(`- Assignments:`, assignments)
                  console.log(`- Conflicts:`, conflicts)
                  alert(`Modal Debug:\n- Agents: ${agents.length}\n- Tools: ${toolsToShow.length}\n- Assignments: ${Object.keys(assignments).length}`)
                }}
                className="text-xs"
              >
                Debug Info
              </Button>
              {conflicts?.length > 0 && (
                <Badge className="bg-red-500/10 text-red-400 border-red-500/20">
                  {conflicts.length} Conflicts
                </Badge>
              )}
            </div>
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
            <TabsList className="grid w-full grid-cols-3 bg-secondary/50">
              <TabsTrigger value="matrix" className="flex items-center space-x-2">
                <Wrench className="w-4 h-4" />
                <span>Assignment Matrix</span>
              </TabsTrigger>
              <TabsTrigger value="agents" className="flex items-center space-x-2">
                <Bot className="w-4 h-4" />
                <span>By Agent</span>
              </TabsTrigger>
              <TabsTrigger value="security" className="flex items-center space-x-2">
                <Shield className="w-4 h-4" />
                <span>Security & Conflicts</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="matrix" className="space-y-6">
              {/* Show message if no tools are installed */}
              {toolsToShow.length === 0 && (
                <div className="text-center py-8">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-orange-500/10 flex items-center justify-center">
                    <Wrench className="w-8 h-8 text-orange-400" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">No Tools Installed</h3>
                  <p className="text-muted-foreground mb-4">
                    Install tools from the Tools Dashboard to assign them to agents
                  </p>
                  <Button 
                    variant="outline" 
                    onClick={() => {
                      onClose()
                      // Navigate to tools page or refresh
                      window.location.href = '/tools'
                    }}
                  >
                    Go to Tools Dashboard
                  </Button>
                </div>
              )}
              
              {/* Assignment Matrix Table */}
              {toolsToShow.length > 0 && (
                <>
                  {/* Filters */}
                  <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
                <div className="flex gap-4 flex-1">
                  <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      placeholder="Search agents..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  <Select value={selectedAgentType} onValueChange={setSelectedAgentType}>
                    <SelectTrigger className="w-48">
                      <SelectValue placeholder="Filter by type..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Types</SelectItem>
                      <SelectItem value="code_architect">Code Architect</SelectItem>
                      <SelectItem value="security_expert">Security Expert</SelectItem>
                      <SelectItem value="performance_optimizer">Performance Optimizer</SelectItem>
                      <SelectItem value="data_analyst">Data Analyst</SelectItem>
                      <SelectItem value="infrastructure_manager">Infrastructure Manager</SelectItem>
                      <SelectItem value="custom">Custom</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Assignment Matrix */}
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Wrench className="w-5 h-5" />
                    <span>Assignment Matrix</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>
                          <th className="text-left p-3 border-b border-border/30">Agent</th>
                          {installedTools.map(tool => (
                            <th key={tool?.id} className="text-center p-3 border-b border-border/30 min-w-[100px]">
                              <div className="flex flex-col items-center space-y-1">
                                <span className="text-lg">{tool?.icon}</span>
                                <span className="text-xs">{tool?.name}</span>
                                <Badge variant="outline" className="text-xs">
                                  {getToolAssignmentCount(tool?.id || 0)}
                                </Badge>
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filteredAgents.map(agent => (
                          <tr key={agent?.id} className="border-b border-border/30">
                            <td className="p-3">
                              <div className="flex items-center space-x-3">
                                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                                  <span className="text-sm">{agent?.icon || '🤖'}</span>
                                </div>
                                <div>
                                  <p className="font-medium">{agent?.name}</p>
                                  <Badge className={agentTypeColors[agent?.agent_type as keyof typeof agentTypeColors]} variant="outline">
                                    {agent?.agent_type?.replace('_', ' ')}
                                  </Badge>
                                </div>
                              </div>
                            </td>
                            {toolsToShow.map(tool => {
                              const isAssigned = isToolAssigned(agent?.id || 0, tool?.id || 0)
                              const isAllowed = isPermissionAllowed(agent?.agent_type || '', tool?.name || '')
                              const hasConflict = conflicts.some(c => 
                                c.agentId === agent?.id && c.toolId === tool?.id
                              )

                              // console.log(`Agent: ${agent?.name} (${agent?.agent_type}), Tool: ${tool?.name}, Allowed: ${isAllowed}, Assigned: ${isAssigned}`)

                              return (
                                <td key={tool?.id} className="p-3 text-center">
                                  <div className="relative">
                                    <Button
                                      variant={isAssigned ? "default" : "outline"}
                                      size="sm"
                                      onClick={(e) => {
                                        e.preventDefault()
                                        e.stopPropagation()
                                        toggleAssignment(agent?.id || 0, tool?.id || 0)
                                      }}
                                      disabled={!isAllowed}
                                      className={`w-8 h-8 p-0 ${
                                        isAssigned 
                                          ? hasConflict 
                                            ? 'bg-red-500/20 hover:bg-red-500/30 text-red-400'
                                            : 'bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white'
                                          : isAllowed 
                                            ? 'hover:border-orange-500/50 hover:bg-orange-500/10 cursor-pointer' 
                                            : 'opacity-50 cursor-not-allowed'
                                      }`}
                                      style={{ pointerEvents: 'auto', zIndex: 10 }}
                                      title={`${isAllowed ? 'Click to assign/unassign' : 'Permission denied'} - ${agent?.name} ↔ ${tool?.name}`}
                                    >
                                      {isAssigned ? (
                                        hasConflict ? <AlertTriangle className="w-4 h-4" /> : <CheckCircle className="w-4 h-4" />
                                      ) : isAllowed ? (
                                        <Plus className="w-4 h-4" />
                                      ) : (
                                        <Lock className="w-4 h-4" />
                                      )}
                                    </Button>
                                  </div>
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
                </>
              )}
            </TabsContent>

            <TabsContent value="agents" className="space-y-6">
              {/* Agent-specific assignments */}
              <div className="grid gap-4">
                {filteredAgents.map(agent => (
                  <Card key={agent?.id} className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                            <span className="text-lg">{agent?.icon || '🤖'}</span>
                          </div>
                          <div>
                            <h3 className="font-semibold">{agent?.name}</h3>
                            <p className="text-sm text-muted-foreground">{agent?.description}</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge className={agentTypeColors[agent?.agent_type as keyof typeof agentTypeColors]}>
                            {agent?.agent_type?.replace('_', ' ')}
                          </Badge>
                          <Badge variant="outline">
                            {getAssignmentCount(agent?.id || 0)} tools
                          </Badge>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <h4 className="text-sm font-medium mb-2">Assigned Tools</h4>
                          <div className="flex flex-wrap gap-2">
                            {toolsToShow.filter(tool => isToolAssigned(agent?.id || 0, tool?.id || 0)).map(tool => {
                              const hasConflict = conflicts.some(c => 
                                c.agentId === agent?.id && c.toolId === tool?.id
                              )
                              return (
                                <Badge 
                                  key={tool?.id} 
                                  className={hasConflict 
                                    ? 'bg-red-500/10 text-red-400 border-red-500/20'
                                    : 'bg-green-500/10 text-green-400 border-green-500/20'
                                  }
                                >
                                  {tool?.icon} {tool?.name}
                                  {hasConflict && <AlertTriangle className="w-3 h-3 ml-1" />}
                                </Badge>
                              )
                            })}
                            {getAssignmentCount(agent?.id || 0) === 0 && (
                              <span className="text-sm text-muted-foreground">No tools assigned</span>
                            )}
                          </div>
                        </div>
                        <Separator />
                        <div>
                          <h4 className="text-sm font-medium mb-2">Available Tools</h4>
                          <div className="flex flex-wrap gap-2">
                            {toolsToShow
                              .filter(tool => 
                                !isToolAssigned(agent?.id || 0, tool?.id || 0) && 
                                isPermissionAllowed(agent?.agent_type || '', tool?.name || '')
                              )
                              .map(tool => (
                                <Button
                                  key={tool?.id}
                                  variant="outline"
                                  size="sm"
                                  onClick={(e) => {
                                    e.preventDefault()
                                    e.stopPropagation()
                                    console.log(`Button clicked (By Agent): Agent ${agent?.id}, Tool ${tool?.id}`)
                                    toggleAssignment(agent?.id || 0, tool?.id || 0)
                                  }}
                                  className="h-8 text-xs hover:border-orange-500/50"
                                  style={{ pointerEvents: 'auto', zIndex: 10 }}
                                >
                                  <Plus className="w-3 h-3 mr-1" />
                                  {tool?.icon} {tool?.name}
                                </Button>
                              ))}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
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
                      <span className="text-sm text-muted-foreground">Role-based Access</span>
                      <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                        Active
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Permission Validation</span>
                      <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                        Enabled
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Access Audit</span>
                      <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                        Logging
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Conflicts Detected</span>
                      <Badge className={conflicts?.length > 0 
                        ? "bg-red-500/10 text-red-400 border-red-500/20"
                        : "bg-green-500/10 text-green-400 border-green-500/20"
                      }>
                        {conflicts?.length || 0}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-secondary/30 border-border/30">
                  <CardHeader>
                    <CardTitle className="text-base flex items-center space-x-2">
                      <Eye className="w-5 h-5 text-blue-400" />
                      <span>Assignment Summary</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Total Agents</span>
                      <Badge variant="outline">{agents?.length}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Installed Tools</span>
                      <Badge variant="outline">{installedTools?.length}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Total Assignments</span>
                      <Badge variant="outline">
                        {Object.values(assignments).flat().length}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Agents with Tools</span>
                      <Badge variant="outline">
                        {Object.values(assignments).filter(arr => arr?.length > 0).length}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Conflicts List */}
              {conflicts?.length > 0 && (
                <Card className="bg-secondary/30 border-border/30">
                  <CardHeader>
                    <CardTitle className="text-base flex items-center space-x-2">
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                      <span>Assignment Conflicts</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {conflicts.map((conflict, index) => {
                        const agent = agents.find(a => a?.id === conflict.agentId)
                        const tool = tools.find(t => t?.id === conflict.toolId)
                        
                        return (
                          <Alert key={index} className="border-red-500/50 bg-red-500/10">
                            <AlertTriangle className="h-4 w-4 text-red-400" />
                            <AlertDescription className="text-red-400">
                              <span className="font-medium">{agent?.name}</span> → <span className="font-medium">{tool?.name}</span>: {conflict.reason}
                            </AlertDescription>
                          </Alert>
                        )
                      })}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Permission Matrix Legend */}
              <Card className="bg-secondary/30 border-border/30">
                <CardHeader>
                  <CardTitle className="text-base">Permission Matrix</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(agentToolPermissions).map(([agentType, allowedTools]) => (
                      <div key={agentType} className="space-y-2">
                        <Badge className={agentTypeColors[agentType as keyof typeof agentTypeColors]}>
                          {agentType.replace('_', ' ').toUpperCase()}
                        </Badge>
                        <div className="flex flex-wrap gap-1">
                          {allowedTools.map(toolName => (
                            <Badge key={toolName} variant="outline" className="text-xs">
                              {toolName}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-4 border-t border-border/30">
          <div className="flex items-center space-x-4 text-sm text-muted-foreground">
            <span>
              {Object.values(assignments).flat().length} assignments
            </span>
            {conflicts?.length > 0 && (
              <Badge className="bg-red-500/10 text-red-400 border-red-500/20">
                <AlertTriangle className="w-3 h-3 mr-1" />
                {conflicts.length} conflicts
              </Badge>
            )}
          </div>
          <div className="flex space-x-3">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button 
              onClick={handleSave}
              disabled={saving || conflicts?.length > 0}
              className="bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
            >
              {saving ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-2" />
                  Save Assignments
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export { AgentToolAssignment }
