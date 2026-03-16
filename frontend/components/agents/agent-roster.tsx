
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Search,
  Filter,
  MoreVertical,
  Play,
  Pause,
  Settings,
  Eye,
  CheckCircle,
  Clock,
  AlertCircle,
  Bot,
  Trash2,
  UserCircle,
  Headphones,
  Terminal,
  Share2,
  Calculator,
  ShoppingBag,
  PenTool,
  Users,
  BarChart3,
  Wrench,
  Shield,
  Zap,
  Cloud,
  Pin,
  PinOff
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { StatusBadge, PremiumIcon } from '@/components/shared'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useAgents, useStartAgent, useStopAgent } from '@/hooks/use-agent-api'
import { usePinnedAgents } from '@/hooks/use-pinned-agents'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { AgentConfigurationModal } from './agent-configuration-modal'
import { AgentStatusControlModal } from './agent-status-control-modal'
import { AgentConfirmDeleteModal } from './agent-confirm-delete-modal'
import { ToolLogo } from '@/components/ui/tool-logo'

// Agent type/category icons mapping - SIMPLE clean icons only!
const getAgentIcon = (agentType: string, category?: string) => {
  // First try category (marketplace agents)
  const categoryMap: Record<string, any> = {
    'Personal Assistant': UserCircle,
    'Customer Support': Headphones,
    'DevOps': Terminal,
    'Social Media': Share2,
    'Accounting': Calculator,
    'E-commerce': ShoppingBag,
    'Content Creation': PenTool,
    'HR': Users,
    'Data Analysis': BarChart3,
    'Custom': Bot,
    'specialized': Bot,
    'general': Wrench,
  }

  // Then try agent_type (workspace agents)
  const typeMap: Record<string, any> = {
    code_architect: Terminal,  // Changed from Code jellyfish!
    security_expert: Shield,
    performance_optimizer: Zap,
    data_analyst: BarChart3,
    infrastructure_manager: Cloud,
    custom: Bot,  // Simple bot icon
    devops: Terminal,
    support: Headphones,
    assistant: UserCircle,
  }

  if (category && categoryMap[category]) {
    return categoryMap[category]
  }

  return typeMap[agentType] || Bot  // Default to Bot, not jellyfish!
}

// Get color for each agent type/category
const getAgentIconColor = (agentType: string, category?: string) => {
  // Category-based colors using semantic tokens
  const categoryColorMap: Record<string, string> = {
    'Personal Assistant': 'text-[hsl(var(--info))]',
    'Customer Support': 'text-[hsl(var(--success))]',
    'DevOps': 'text-[hsl(var(--agent))]',
    'Social Media': 'text-primary',
    'Accounting': 'text-[hsl(var(--warning))]',
    'E-commerce': 'text-[hsl(var(--info))]',
    'Content Creation': 'text-[hsl(var(--agent))]',
    'HR': 'text-[hsl(var(--success))]',
    'Data Analysis': 'text-[hsl(var(--destructive))]',
    'Custom': 'text-primary'
  }

  // Agent type-based colors using semantic tokens
  const typeColorMap: Record<string, string> = {
    code_architect: 'text-[hsl(var(--agent))]',
    security_expert: 'text-[hsl(var(--destructive))]',
    performance_optimizer: 'text-[hsl(var(--warning))]',
    data_analyst: 'text-[hsl(var(--destructive))]',
    infrastructure_manager: 'text-[hsl(var(--info))]',
    custom: 'text-primary',
    devops: 'text-[hsl(var(--agent))]',
    support: 'text-[hsl(var(--success))]',
    assistant: 'text-[hsl(var(--info))]',
  }

  // Try category first
  if (category && categoryColorMap[category]) {
    return categoryColorMap[category]
  }

  // Then try agent type
  return typeColorMap[agentType] || 'text-primary'
}

// Real agent data from API - no more mock data
interface AgentWithPerformance {
  id: string
  name: string
  agent_type: string
  status: string
  description?: string
  created_at?: string
  skills?: Array<{ id: string; name: string }>
  plugins?: Array<{ plugin_id: string; slug: string; name: string; skills_count: number; commands_count: number }>
  performance_metrics?: {
    success_rate?: number
    tasks_completed?: number
  }
  performance?: number
  tasksCompleted?: number
  specializations?: string[]
  lastActive?: string
  avatar?: string
  premium_icon?: string | null
  agent_model_config?: {
    provider?: string
  }
  tools?: Array<{
    id: number
    name: string // tool_id like 'slack'
    icon?: string
  }>
}

const statusIcons: Record<string, any> = {
  active: CheckCircle,
  idle: Clock,
  maintenance: AlertCircle
}

// PRD-15: Model display helper
const getModelDisplayName = (modelId?: string): string => {
  if (!modelId) return 'GPT-4'

  const modelNames: Record<string, string> = {
    'gpt-4': 'GPT-4',
    'gpt-4-turbo': 'GPT-4 Turbo',
    'gpt-4-turbo-preview': 'GPT-4 Turbo',
    'gpt-3.5-turbo': 'GPT-3.5',
    'claude-3-opus-20240229': 'Claude 3 Opus',
    'claude-3-sonnet-20240229': 'Claude 3 Sonnet',
    'claude-3-haiku-20240307': 'Claude 3 Haiku',
  }

  return modelNames[modelId] || modelId.split('-')[0].toUpperCase()
}

interface AgentRosterProps {
  agents: any[]
  loading: boolean
  searchTerm: string
  statusFilter: string
  onAgentSelect: (agentId: string | null) => void
  onViewDetails: (agentId: string | null) => void
  selectedAgentId: string | null
  onRefresh: () => void
  setSearchTerm?: ((term: string) => void) | undefined
  viewMode?: 'grid' | 'list'
}

export function AgentRoster({
  agents,
  loading,
  searchTerm,
  statusFilter,
  onAgentSelect,
  onViewDetails,
  selectedAgentId,
  onRefresh,
  setSearchTerm,
  viewMode: viewModeProp
}: AgentRosterProps) {
  const [viewModeLocal, setViewModeLocal] = useViewMode('agents')
  const viewMode = viewModeProp || viewModeLocal

  // Modal states
  const [configModalAgentId, setConfigModalAgentId] = useState<number | null>(null)
  const [statusModalAgentId, setStatusModalAgentId] = useState<number | null>(null)
  const [currentAgentStatus, setCurrentAgentStatus] = useState<string>('')
  const [deleteModalAgentId, setDeleteModalAgentId] = useState<number | null>(null)

  // Pinned agents
  const workspaceId = typeof window !== 'undefined'
    ? localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org') || ''
    : ''
  const { pinnedIds, pin, unpin, isPinned } = usePinnedAgents(workspaceId)

  // System icon mappings
  const { data: iconMappings = {} } = useSystemIcons()

  // Use real API hooks for agent actions
  const startAgentMutation = useStartAgent()
  const stopAgentMutation = useStopAgent()

  // Context menu handlers
  const handleViewDetails = (agentId: string) => {
    onViewDetails(agentId) // Use the dedicated View Details handler
  }

  const handleConfigure = (agentId: string) => {
    setConfigModalAgentId(Number(agentId))
  }

  const handleToggleStatus = async (agentId: string, currentStatus: string) => {
    setStatusModalAgentId(Number(agentId))
    setCurrentAgentStatus(currentStatus)
  }

  const handleDelete = (agentId: string) => {
    setDeleteModalAgentId(Number(agentId))
  }

  const handleStatusChange = async (agentId: number, newStatus: string) => {
    try {
      if (newStatus === 'inactive') {
        await (stopAgentMutation.mutateAsync as any)(agentId.toString())
      } else if (newStatus === 'active') {
        await (startAgentMutation.mutateAsync as any)(agentId.toString())
      }

      // Refresh the agents list
      onRefresh()

    } catch (error) {
      console.error('Error updating agent status:', error)
    }
  }

  const filteredAgents = agents.filter(agent => {
    const matchesSearch = (agent.name && agent.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (agent.agent_type && agent.agent_type.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (agent.skills && agent.skills.some((skill: any) => skill.name && skill.name.toLowerCase().includes(searchTerm.toLowerCase())))

    const matchesStatus = statusFilter === 'all' || agent.status === statusFilter

    return matchesSearch && matchesStatus
  })

  if (loading) {
    return (
      <div className="space-y-6">
        {viewMode === 'list' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-16 glass-card animate-pulse rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-muted-foreground">Loading agents...</p>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Agent Grid / List */}
      {viewMode === 'list' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {filteredAgents.map((agent) => {
            const StatusIcon = statusIcons[agent.status || 'active'] || CheckCircle
            const premiumIconName = agent.premium_icon || iconMappings[(agent as any).marketplace_category] || iconMappings[(agent as any).configuration?.category] || iconMappings[agent.agent_type] || null
            const AgentIcon = getAgentIcon(agent.agent_type || 'custom', (agent as any).marketplace_category)
            const iconColor = getAgentIconColor(agent.agent_type || 'custom', (agent as any).marketplace_category)

            return (
              <div
                key={agent.id}
                data-testid="agent-card"
                className="glass-card p-3 hover:border-primary/20 transition-all cursor-pointer"
                onClick={() => onViewDetails(agent.id.toString())}
              >
                <div className="flex items-center gap-3">
                  {premiumIconName ? (
                    <PremiumIcon name={premiumIconName} size={36} className={`shrink-0 ${iconColor}`} />
                  ) : (
                    <AgentIcon className={`w-9 h-9 ${iconColor} shrink-0`} />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm truncate">{agent.name || 'Unknown Agent'}</span>
                      <StatusBadge size="sm" status={
                        agent.status === 'idle' ? 'warning' :
                          agent.status === 'maintenance' ? 'error' : 'success'
                      }>
                        <StatusIcon className="w-2.5 h-2.5 mr-0.5" />
                        {agent.status || 'active'}
                      </StatusBadge>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                      <span>{agent.agent_type ? agent.agent_type.replace('_', ' ') : 'Unknown'}</span>
                      <span>&middot;</span>
                      <span>{getModelDisplayName(agent.agent_model_config?.model_id)}</span>
                      <span>&middot;</span>
                      <span>{agent.performance_metrics?.tasks_completed || 0} tasks</span>
                    </div>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={(e) => e.stopPropagation()}>
                        <MoreVertical className="w-4 h-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleViewDetails(agent.id.toString()) }}>
                        <Eye className="w-4 h-4 mr-2" />
                        View Details
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleConfigure(agent.id) }}>
                        <Settings className="w-4 h-4 mr-2" />
                        Configure
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        disabled={!isPinned(Number(agent.id)) && pinnedIds.length >= 6}
                        title={!isPinned(Number(agent.id)) && pinnedIds.length >= 6 ? 'Max 6 agents' : undefined}
                        onClick={(e) => {
                          e.stopPropagation()
                          if (isPinned(Number(agent.id))) {
                            unpin(Number(agent.id))
                          } else {
                            pin(Number(agent.id))
                          }
                        }}
                      >
                        {isPinned(Number(agent.id)) ? (
                          <><PinOff className="w-4 h-4 mr-2" />Unpin from Chat</>
                        ) : (
                          <><Pin className="w-4 h-4 mr-2" />Pin to Chat</>
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleToggleStatus(agent.id, agent.status) }}>
                        {agent.status === 'active' ? (
                          <><Pause className="w-4 h-4 mr-2" />Pause Agent</>
                        ) : (
                          <><Play className="w-4 h-4 mr-2" />Start Agent</>
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={(e) => { e.stopPropagation(); handleDelete(agent.id) }}
                        className="text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))] hover:bg-[hsl(var(--destructive))]/10"
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredAgents.map((agent, index) => {
            const StatusIcon = statusIcons[agent.status || 'active'] || CheckCircle
            const premiumIconName = agent.premium_icon || iconMappings[(agent as any).marketplace_category] || iconMappings[(agent as any).configuration?.category] || iconMappings[agent.agent_type] || null
            const AgentIcon = getAgentIcon(agent.agent_type || 'custom', (agent as any).marketplace_category)
            const iconColor = getAgentIconColor(agent.agent_type || 'custom', (agent as any).marketplace_category)

            return (
              <motion.div
                key={agent.id}
                data-testid="agent-card"
                className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300 flex flex-col h-full"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                {/* Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {premiumIconName ? (
                      <PremiumIcon name={premiumIconName} size={40} className={`shrink-0 ${iconColor}`} />
                    ) : (
                      <AgentIcon className={`w-10 h-10 ${iconColor} shrink-0`} />
                    )}
                    <div>
                      <h3 className="font-semibold">{agent.name || 'Unknown Agent'}</h3>
                      <p className="text-xs text-muted-foreground">{agent.agent_type ? agent.agent_type.replace('_', ' ') : 'Unknown Type'}</p>
                    </div>
                  </div>

                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8" data-tour="agent-card-menu">
                        <MoreVertical className="w-4 h-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={(e) => {
                        e.stopPropagation(); // Prevent card click
                        handleViewDetails(agent.id.toString());
                      }}>
                        <Eye className="w-4 h-4 mr-2" />
                        View Details
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={(e) => {
                        e.stopPropagation(); // Prevent card click
                        handleConfigure(agent.id);
                      }}>
                        <Settings className="w-4 h-4 mr-2" />
                        Configure
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        disabled={!isPinned(Number(agent.id)) && pinnedIds.length >= 6}
                        title={!isPinned(Number(agent.id)) && pinnedIds.length >= 6 ? 'Max 6 agents' : undefined}
                        onClick={(e) => {
                          e.stopPropagation()
                          if (isPinned(Number(agent.id))) {
                            unpin(Number(agent.id))
                          } else {
                            pin(Number(agent.id))
                          }
                        }}
                      >
                        {isPinned(Number(agent.id)) ? (
                          <><PinOff className="w-4 h-4 mr-2" />Unpin from Chat</>
                        ) : (
                          <><Pin className="w-4 h-4 mr-2" />Pin to Chat</>
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={(e) => {
                        e.stopPropagation(); // Prevent card click
                        handleToggleStatus(agent.id, agent.status);
                      }}>
                        {agent.status === 'active' ? (
                          <>
                            <Pause className="w-4 h-4 mr-2" />
                            Pause Agent
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4 mr-2" />
                            Start Agent
                          </>
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={(e) => {
                          e.stopPropagation(); // Prevent card click
                          handleDelete(agent.id);
                        }}
                        className="text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))] hover:bg-[hsl(var(--destructive))]/10"
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                {/* Status & Model */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <StatusBadge size="sm" status={
                      agent.status === 'idle' ? 'warning' :
                        agent.status === 'maintenance' ? 'error' : 'success'
                    }>
                      <StatusIcon className="w-2.5 h-2.5 mr-0.5" />
                      {agent.status || 'active'}
                    </StatusBadge>
                    {/* PRD-15: Model Badge */}
                    <StatusBadge size="sm" status={
                      (() => {
                        const p = agent.agent_model_config?.provider?.toLowerCase()
                        if (p === 'anthropic') return 'purple' as const
                        if (p === 'openai') return 'success' as const
                        if (p === 'google' || p === 'azure') return 'info' as const
                        if (p === 'huggingface') return 'warning' as const
                        if (p === 'aws_bedrock' || p === 'bedrock') return 'primary' as const
                        return 'info' as const
                      })()
                    }>
                      <Bot className="w-2.5 h-2.5 mr-0.5" />
                      {getModelDisplayName(agent.agent_model_config?.model_id)}
                    </StatusBadge>
                  </div>
                </div>

                {/* Performance Bar */}
                <div className="mb-4">
                  <div className="flex justify-between text-xs text-muted-foreground mb-1">
                    <span>Success Rate</span>
                    <span>
                      {agent.performance_metrics?.success_rate != null ?
                        `${(agent.performance_metrics.success_rate * 100).toFixed(1)}%` :
                        'N/A'
                      }
                    </span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-primary to-[hsl(var(--destructive))] h-2 rounded-full transition-all duration-300"
                      style={{
                        width: `${agent.performance_metrics?.success_rate != null ?
                          agent.performance_metrics.success_rate * 100 : 0}%`
                      }}
                    />
                  </div>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-lg font-bold">
                      {agent.performance_metrics?.tasks_completed || 0}
                    </p>
                    <p className="text-xs text-muted-foreground">Tasks Completed</p>
                  </div>
                  <div>
                    <p className="text-lg font-bold">{(agent.plugins?.length || 0) + (agent.skills?.length || 0)}</p>
                    <p className="text-xs text-muted-foreground">Capabilities</p>
                  </div>
                </div>

                {/* Plugins Preview */}
                <div className="mb-4">
                  <p className="text-xs text-muted-foreground mb-2">Capabilities</p>
                  <div className="flex flex-wrap gap-1">
                    {((agent.plugins?.length || 0) + (agent.skills?.length || 0)) > 0 ? (
                      <>
                        {agent.skills?.slice(0, 3).map((skill: any) => (
                          <Badge key={`skill-${skill.id}`} variant="secondary" className="text-xs">
                            {skill.name}
                          </Badge>
                        ))}
                        {agent.plugins?.slice(0, Math.max(0, 3 - (agent.skills?.length || 0))).map((plugin: any) => (
                          <Badge key={`plugin-${plugin.plugin_id}`} variant="secondary" className="text-xs">
                            {plugin.name}
                          </Badge>
                        ))}
                        {(agent.plugins?.length || 0) + (agent.skills?.length || 0) > 3 && (
                          <Badge variant="secondary" className="text-xs">
                            +{(agent.plugins?.length || 0) + (agent.skills?.length || 0) - 3} more
                          </Badge>
                        )}
                      </>
                    ) : (
                      <span className="text-xs text-muted-foreground italic">No capabilities assigned</span>
                    )}
                  </div>
                </div>

                {/* Tools Preview */}
                {agent.tools && agent.tools.length > 0 && (
                  <div className="mb-4 pt-3 border-t border-border/50">
                    <div className="flex flex-wrap gap-2">
                      {agent.tools.slice(0, 5).map((tool: any) => (
                        <div key={tool.id} title={tool.name}>
                          <ToolLogo
                            name={tool.name}
                            logo={tool.icon}
                            size={24}
                            showBackground={true}
                            className="bg-secondary/30 border border-border/50"
                          />
                        </div>
                      ))}
                      {agent.tools.length > 5 && (
                        <div className="bg-secondary/30 px-1.5 h-[24px] flex items-center justify-center rounded-md border border-border/50 text-[10px] text-muted-foreground">
                          +{agent.tools.length - 5}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Created Date */}
                <p className="text-xs text-muted-foreground mt-auto pt-3">
                  Created: {agent.created_at ? new Date(agent.created_at).toLocaleDateString() : 'Unknown'}
                </p>
              </motion.div>
            )
          })}
        </div>
      )}

      {filteredAgents.length === 0 && (
        <motion.div
          className="text-center py-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Bot className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">No agents found</h3>
          <p className="text-muted-foreground">
            Try adjusting your search or create a new agent.
          </p>
        </motion.div>
      )}

      {/* Agent Configuration Modal */}
      <AgentConfigurationModal
        agentId={configModalAgentId}
        open={configModalAgentId !== null}
        onClose={() => setConfigModalAgentId(null)}
        onSave={(agentId, config) => {
          onRefresh();
        }}
      />

      {/* Agent Status Control Modal */}
      <AgentStatusControlModal
        agentId={statusModalAgentId}
        currentStatus={currentAgentStatus}
        open={statusModalAgentId !== null}
        onClose={() => setStatusModalAgentId(null)}
        onStatusChanged={handleStatusChange}
      />

      {/* Agent Confirm Delete Modal */}
      <AgentConfirmDeleteModal
        agentId={deleteModalAgentId}
        open={deleteModalAgentId !== null}
        onClose={() => setDeleteModalAgentId(null)}
        onDeleted={() => {
          onRefresh();
        }}
      />
    </div>
  )
}
