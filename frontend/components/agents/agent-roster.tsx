
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
  Bot
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
import { apiClient, Agent } from '@/lib/api'

// Agent type icons mapping
const agentTypeIcons: Record<string, string> = {
  code_architect: '🏗️',
  security_expert: '🛡️',
  performance_optimizer: '⚡',
  data_analyst: '📊',
  infrastructure_manager: '☁️',
  custom: '🤖'
}

// Real agent data from API - no more mock data
interface RealAgent extends Agent {
  performance?: number
  tasksCompleted?: number
  specializations?: string[]
  lastActive?: string
  avatar?: string
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

export function AgentRoster() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch agents from API
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setLoading(true)
        const data = await apiClient.getAgents()
        setAgents(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch agents')
        console.error('Error fetching agents:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchAgents()
  }, [])

  // Context menu handlers
  const handleViewDetails = (agentId: string) => {
    console.log('View details for agent:', agentId)
    // You can implement a modal or navigate to a details page
    alert(`Viewing details for agent ${agentId}`)
  }

  const handleConfigure = (agentId: string) => {
    console.log('Configure agent:', agentId)
    // Navigate to configuration tab or open configuration modal
    // For now, we'll show an alert
    alert(`Opening configuration for agent ${agentId}. This would typically switch to the Configuration tab.`)
  }

  const handleToggleStatus = async (agentId: number, currentStatus: string) => {
    try {
      const newStatus = currentStatus === 'active' ? 'inactive' : 'active'
      const updatedAgent = await apiClient.request(`/api/agents/${agentId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      })
      console.log('Agent status updated:', updatedAgent)
      
      // Update the local state
      setAgents(prevAgents => 
        prevAgents.map((agent: any) => 
          agent.id === agentId ? { ...agent, status: newStatus } : agent
        )
      )
      
      alert(`Agent status changed to ${newStatus}`)
      
    } catch (error) {
      console.error('Error updating agent status:', error)
      alert(`Error updating agent status: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  
  const filteredAgents = agents.filter(agent =>
    (agent.name && agent.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (agent.agent_type && agent.agent_type.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (agent.skills && agent.skills.some((skill: any) => skill.name && skill.name.toLowerCase().includes(searchTerm.toLowerCase())))
  )

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

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertCircle className="h-8 w-8 text-red-500 mx-auto mb-4" />
          <p className="text-red-500 mb-2">Error loading agents</p>
          <p className="text-sm text-muted-foreground">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search agents by name, type, or skills..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
          />
        </div>
        <Button variant="outline" className="shrink-0">
          <Filter className="w-4 h-4 mr-2" />
          Filters
        </Button>
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {filteredAgents.map((agent, index) => {
          const StatusIcon = statusIcons[agent.status || 'active'] || CheckCircle
          const agentIcon = agentTypeIcons[agent.agent_type || 'custom'] || '🤖'
          
          return (
            <motion.div
              key={agent.id}
              className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              onClick={() => setSelectedAgent(agent.id.toString())}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center text-lg">
                    {agentIcon}
                  </div>
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
                    <DropdownMenuItem onClick={() => handleViewDetails(agent.id)}>
                      <Eye className="w-4 h-4 mr-2" />
                      View Details
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleConfigure(agent.id)}>
                      <Settings className="w-4 h-4 mr-2" />
                      Configure
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleToggleStatus(agent.id, agent.status)}>
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
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              {/* Status */}
              <div className="flex items-center justify-between mb-4">
                <Badge className={statusStyles[agent.status || 'active'] || statusStyles.active}>
                  <StatusIcon className="w-3 h-3 mr-1" />
                  {agent.status || 'active'}
                </Badge>
                <span className="text-sm text-muted-foreground">
                  {agent.performance_metrics?.success_rate ? 
                    `${(agent.performance_metrics.success_rate * 100).toFixed(1)}%` : 
                    'N/A'
                  }
                </span>
              </div>

              {/* Performance Bar */}
              <div className="mb-4">
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>Success Rate</span>
                  <span>
                    {agent.performance_metrics?.success_rate ? 
                      `${(agent.performance_metrics.success_rate * 100).toFixed(1)}%` : 
                      'N/A'
                    }
                  </span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-300"
                    style={{ 
                      width: `${agent.performance_metrics?.success_rate ? 
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
                  {agent.skills agent.skills && agent.skills.sliceagent.skills && agent.skills.slice Array.isArray(agent.skills) ? agent.skills.slice(0, 3).map((skill: any) => (
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
    </div>
  )
}
