
'use client'

import { useState, useMemo } from 'react'
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
  Edit,
  Trash2,
  Activity
} from 'lucide-react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'react-hot-toast'

// API hooks
import { useUpdateAgent, useDeleteAgent } from "@/hooks/use-agent-api"

// Agent type icons mapping
const agentTypeIcons: Record<string, string> = {
  code_architect: '🏗️',
  security_expert: '🛡️',
  performance_optimizer: '⚡',
  data_analyst: '📊',
  infrastructure_manager: '☁️',
  system: '🔧',
  specialized: '🎯',
  custom: '🤖'
}

const statusStyles: Record<string, string> = {
  active: 'bg-green-500/10 text-green-400 border-green-500/20',
  idle: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  inactive: 'bg-red-500/10 text-red-400 border-red-500/20',
  maintenance: 'bg-blue-500/10 text-blue-400 border-blue-500/20'
}

const statusIcons: Record<string, any> = {
  active: CheckCircle,
  idle: Clock,
  inactive: AlertCircle,
  maintenance: Settings
}

interface RealAgentRosterProps {
  agents: any[]
  loading: boolean
  searchTerm: string
  statusFilter: string
  selectedAgentId: string | null
  onAgentSelect: (agentId: string | null) => void
  onRefresh: () => void
}

export function RealAgentRoster({
  agents,
  loading,
  searchTerm,
  statusFilter,
  selectedAgentId,
  onAgentSelect,
  onRefresh
}: RealAgentRosterProps) {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')

  // API mutations
  const updateAgentMutation = useUpdateAgent()
  const deleteAgentMutation = useDeleteAgent()

  // Filter agents based on search and status
  const filteredAgents = useMemo(() => {
    return agents.filter(agent => {
      const matchesSearch = !searchTerm || 
        agent.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        agent.agent_type?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        agent.description?.toLowerCase().includes(searchTerm.toLowerCase())
      
      const matchesStatus = statusFilter === 'all' || agent.status === statusFilter
      
      return matchesSearch && matchesStatus
    })
  }, [agents, searchTerm, statusFilter])

  // Handle agent actions
  const handleStartAgent = async (agentId: string) => {
    try {
      await updateAgentMutation.mutateAsync({
        id: agentId,
        data: { status: 'active' }
      })
      toast.success('Agent started successfully')
      onRefresh()
    } catch (error) {
      toast.error('Failed to start agent')
    }
  }

  const handlePauseAgent = async (agentId: string) => {
    try {
      await updateAgentMutation.mutateAsync({
        id: agentId,
        data: { status: 'idle' }
      })
      toast.success('Agent paused successfully')
      onRefresh()
    } catch (error) {
      toast.error('Failed to pause agent')
    }
  }

  const handleDeleteAgent = async (agentId: string, agentName: string) => {
    if (!confirm(`Are you sure you want to delete agent "${agentName}"? This action cannot be undone.`)) {
      return
    }

    try {
      await deleteAgentMutation.mutateAsync(agentId)
      toast.success('Agent deleted successfully')
      onRefresh()
    } catch (error) {
      toast.error('Failed to delete agent')
    }
  }

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <Card key={i} className="glass-card">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <Skeleton className="w-12 h-12 rounded-full" />
                <div className="flex-1">
                  <Skeleton className="h-4 w-24 mb-2" />
                  <Skeleton className="h-3 w-16" />
                </div>
                <Skeleton className="h-6 w-16" />
              </div>
              <Skeleton className="h-3 w-full mb-2" />
              <Skeleton className="h-3 w-3/4" />
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* View Controls */}
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">
            {filteredAgents.length} of {agents.length} agents
          </span>
          {searchTerm && (
            <Badge variant="outline">
              Filtered by: "{searchTerm}"
            </Badge>
          )}
          {statusFilter !== 'all' && (
            <Badge variant="outline">
              Status: {statusFilter}
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === 'grid' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('grid')}
          >
            Grid
          </Button>
          <Button
            variant={viewMode === 'list' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('list')}
          >
            List
          </Button>
        </div>
      </div>

      {/* Agents Grid/List */}
      <div className={viewMode === 'grid' ? 
        'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' : 
        'space-y-4'
      }>
        {filteredAgents.map((agent, index) => {
          const StatusIcon = statusIcons[agent.status] || Bot
          const agentIcon = agentTypeIcons[agent.agent_type] || '🤖'
          
          return (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              <Card 
                className={`glass-card cursor-pointer transition-all duration-200 hover:shadow-lg hover:border-brand-primary/30 ${
                  selectedAgentId === agent.id ? 'border-brand-primary shadow-lg' : ''
                }`}
                onClick={() => onAgentSelect(agent.id)}
              >
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <Avatar className="w-12 h-12">
                        <AvatarFallback className="text-lg">
                          {agentIcon}
                        </AvatarFallback>
                      </Avatar>
                      
                      <div className="flex-1">
                        <h3 className="font-semibold text-lg leading-tight truncate" title={agent.name}>
                          {agent.name}
                        </h3>
                        <p className="text-sm text-muted-foreground capitalize">
                          {agent.agent_type?.replace('_', ' ')}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <Badge 
                        variant="outline" 
                        className={`text-xs ${statusStyles[agent.status] || statusStyles.inactive}`}
                      >
                        <StatusIcon className="w-3 h-3 mr-1" />
                        {agent.status}
                      </Badge>

                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                            <MoreVertical className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-48">
                          <DropdownMenuItem onClick={(e) => {
                            e.stopPropagation()
                            onAgentSelect(agent.id)
                          }}>
                            <Eye className="w-4 h-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                          
                          <DropdownMenuItem onClick={(e) => {
                            e.stopPropagation()
                            onAgentSelect(agent.id)
                          }}>
                            <Edit className="w-4 h-4 mr-2" />
                            Edit Agent
                          </DropdownMenuItem>
                          
                          <DropdownMenuSeparator />
                          
                          {agent.status === 'active' ? (
                            <DropdownMenuItem onClick={(e) => {
                              e.stopPropagation()
                              handlePauseAgent(agent.id)
                            }}>
                              <Pause className="w-4 h-4 mr-2" />
                              Pause Agent
                            </DropdownMenuItem>
                          ) : (
                            <DropdownMenuItem onClick={(e) => {
                              e.stopPropagation()
                              handleStartAgent(agent.id)
                            }}>
                              <Play className="w-4 h-4 mr-2" />
                              Start Agent
                            </DropdownMenuItem>
                          )}
                          
                          <DropdownMenuSeparator />
                          
                          <DropdownMenuItem 
                            className="text-red-600"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeleteAgent(agent.id, agent.name)
                            }}
                          >
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete Agent
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </div>

                  {/* Agent Description */}
                  <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                    {agent.description || 'No description available'}
                  </p>

                  {/* Skills */}
                  {agent.skills && agent.skills.length > 0 && (
                    <div className="mb-4">
                      <div className="flex flex-wrap gap-1">
                        {agent.skills && Array.isArray(agent.skills) && agent.skills.slice(0, 3).map((skill: any) => (
                          <Badge key={skill.id} variant="secondary" className="text-xs">
                            {skill.name}
                          </Badge>
                        ))}
                        {agent.skills && Array.isArray(agent.skills) && agent.skills.length > 3 && (
                          <Badge variant="secondary" className="text-xs">
                            +{agent.skills.length - 3} more
                          </Badge>
                        )}
                        )}
                      </div>
                    </div>
                  )}

                  {/* Performance Indicator */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Performance</span>
                      <span className="font-medium">
                        {agent.performance_score || '95'}%
                      </span>
                    </div>
                    <Progress 
                      value={agent.performance_score || 95} 
                      className="h-2"
                    />
                  </div>

                  {/* Last Activity */}
                  <div className="flex items-center justify-between mt-4 pt-4 border-t border-border/50">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Activity className="w-3 h-3" />
                      Last active: {agent.last_active || '2 minutes ago'}
                    </div>
                    
                    <div className="text-xs text-muted-foreground">
                      Tasks: {agent.total_tasks || 0}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )
        })}
      </div>

      {/* No Results */}
      {filteredAgents.length === 0 && !loading && (
        <Card className="glass-card">
          <CardContent className="p-12 text-center">
            <Bot className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No agents found</h3>
            <p className="text-muted-foreground">
              {searchTerm || statusFilter !== 'all' 
                ? 'Try adjusting your search or filters' 
                : 'Create your first agent to get started'
              }
            </p>
            {!searchTerm && statusFilter === 'all' && (
              <Button className="mt-4" onClick={() => onAgentSelect(null)}>
                Create Agent
              </Button>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

