
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
  Cloud
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useAgents, useStartAgent, useStopAgent } from '@/hooks/use-agent-api'
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
  // Category-based colors (matching create-agent-modal)
  const categoryColorMap: Record<string, string> = {
    'Personal Assistant': 'text-blue-500',
    'Customer Support': 'text-green-500',
    'DevOps': 'text-purple-500',
    'Social Media': 'text-pink-500',
    'Accounting': 'text-amber-500',
    'E-commerce': 'text-cyan-500',
    'Content Creation': 'text-indigo-500',
    'HR': 'text-teal-500',
    'Data Analysis': 'text-rose-500',
    'Custom': 'text-orange-500'
  }

  // Agent type-based colors
  const typeColorMap: Record<string, string> = {
    code_architect: 'text-purple-500',
    security_expert: 'text-red-500',
    performance_optimizer: 'text-yellow-500',
    data_analyst: 'text-rose-500',
    infrastructure_manager: 'text-cyan-500',
    custom: 'text-orange-500',
    devops: 'text-purple-500',
    support: 'text-green-500',
    assistant: 'text-blue-500',
  }

  // Try category first
  if (category && categoryColorMap[category]) {
    return categoryColorMap[category]
  }

  // Then try agent type
  return typeColorMap[agentType] || 'text-orange-500'
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
  performance_metrics?: {
    success_rate?: number
    tasks_completed?: number
  }
  performance?: number
  tasksCompleted?: number
  specializations?: string[]
  lastActive?: string
  avatar?: string
  agent_model_config?: {
    provider?: string
  }
  tools?: Array<{
    id: number
    name: string // tool_id like 'slack'
    icon?: string
  }>
}

const statusStyles: Record<string, string> = {
  active: 'bg-green-500/10 text-green-400 border-green-500/20',
  idle: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  maintenance: 'bg-red-500/10 text-red-400 border-red-500/20'
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

const getProviderColor = (provider?: string): string => {
  const colors: Record<string, string> = {
    openai: 'bg-green-500/10 text-green-400 border-green-500/20',
    anthropic: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    google: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    azure: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    huggingface: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    aws_bedrock: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
    bedrock: 'bg-orange-500/10 text-orange-400 border-orange-500/20'
  }
  return colors[provider?.toLowerCase() || 'openai'] || 'bg-blue-500/10 text-blue-400 border-blue-500/20'
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
  setSearchTerm
}: AgentRosterProps) {
  // Modal states
  const [configModalAgentId, setConfigModalAgentId] = useState<number | null>(null)
  const [statusModalAgentId, setStatusModalAgentId] = useState<number | null>(null)
  const [currentAgentStatus, setCurrentAgentStatus] = useState<string>('')
  const [deleteModalAgentId, setDeleteModalAgentId] = useState<number | null>(null)

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
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading agents...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {filteredAgents.map((agent, index) => {
          const StatusIcon = statusIcons[agent.status || 'active'] || CheckCircle
          const AgentIcon = getAgentIcon(agent.agent_type || 'custom', (agent as any).marketplace_category)
          const iconColor = getAgentIconColor(agent.agent_type || 'custom', (agent as any).marketplace_category)

          return (
            <motion.div
              key={agent.id}
              className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <AgentIcon className={`w-10 h-10 ${iconColor} shrink-0`} />
                  <div>
                    <h3 className="font-semibold">{agent.name || 'Unknown Agent'}</h3>
                    <p className="text-xs text-muted-foreground">{agent.agent_type ? agent.agent_type.replace('_', ' ') : 'Unknown Type'}</p>
                  </div>
                </div>

                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
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
                      className="text-red-500 hover:text-red-600 hover:bg-red-100/10"
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
                  <Badge className={statusStyles[agent.status || 'active'] || statusStyles.active}>
                    <StatusIcon className="w-3 h-3 mr-1" />
                    {agent.status || 'active'}
                  </Badge>
                  {/* PRD-15: Model Badge */}
                  <Badge className={getProviderColor(agent.agent_model_config?.provider)}>
                    <Bot className="w-3 h-3 mr-1" />
                    {getModelDisplayName(agent.agent_model_config?.model_id)}
                  </Badge>
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
                    className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-300"
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
                  <p className="text-lg font-bold">{agent.skills ? agent.skills.length : 0}</p>
                  <p className="text-xs text-muted-foreground">Skills</p>
                </div>
              </div>

              {/* Skills Preview */}
              <div className="mb-4">
                <p className="text-xs text-muted-foreground mb-2">Primary Skills</p>
                <div className="flex flex-wrap gap-1">
                  {agent.skills && agent.skills.slice(0, 3).map((skill: any) => (
                    <Badge key={skill.id} variant="secondary" className="text-xs">
                      {skill.name ? skill.name.replace('_', ' ') : 'Unknown Skill'}
                    </Badge>
                  ))}
                  {agent.skills && agent.skills.length > 3 && (
                    <Badge variant="secondary" className="text-xs">
                      +{agent.skills.length - 3} more
                    </Badge>
                  )}
                </div>
              </div>

              {/* Tools Preview */}
              {agent.tools && agent.tools.length > 0 && (
                <div className="mb-4 pt-3 border-t border-white/5">
                  <div className="flex flex-wrap gap-2">
                    {agent.tools.slice(0, 5).map((tool: any) => (
                      <div key={tool.id} title={tool.name}>
                        <ToolLogo
                          name={tool.name}
                          logo={tool.icon}
                          size={24}
                          showBackground={true}
                          className="bg-secondary/30 border border-white/5"
                        />
                      </div>
                    ))}
                    {agent.tools.length > 5 && (
                      <div className="bg-secondary/30 px-1.5 h-[24px] flex items-center justify-center rounded-md border border-white/5 text-[10px] text-muted-foreground">
                        +{agent.tools.length - 5}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Created Date */}
              <p className="text-xs text-muted-foreground">
                Created: {agent.created_at ? new Date(agent.created_at).toLocaleDateString() : 'Unknown'}
              </p>
            </motion.div>
          )
        })}
      </div>

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
