'use client'

/**
 * Agent Status Grid - Shows live agent states from factory
 */

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Bot, 
  Activity, 
  Zap, 
  Clock,
  TrendingUp,
  AlertCircle
} from 'lucide-react'

interface Agent {
  agent_id: string
  name: string
  agent_type: string
  execution_count: number
  total_tokens_used: number
  success_rate: number
  avg_execution_time: number
  status: string
  memory_items_count: number
  collaboration_count: number
  last_active: string | null
}

interface AgentStatusGridProps {
  agents: Agent[]
}

export function AgentStatusGrid({ agents }: AgentStatusGridProps) {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState<'activity' | 'performance' | 'tokens'>('activity')

  // Sort agents based on selected criteria
  const sortedAgents = [...agents].sort((a, b) => {
    switch (sortBy) {
      case 'activity':
        return b.execution_count - a.execution_count
      case 'performance':
        return b.success_rate - a.success_rate
      case 'tokens':
        return b.total_tokens_used - a.total_tokens_used
      default:
        return 0
    }
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-500'
      case 'idle':
        return 'bg-yellow-500'
      case 'error':
        return 'bg-red-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getAgentTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'orchestrator':
        return <Activity className="w-4 h-4" />
      case 'worker':
        return <Bot className="w-4 h-4" />
      case 'specialist':
        return <Zap className="w-4 h-4" />
      default:
        return <Bot className="w-4 h-4" />
    }
  }

  const formatLastActive = (timestamp: string | null) => {
    if (!timestamp) return 'Never'
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    
    if (minutes < 1) return 'Just now'
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  }

  return (
    <div className="space-y-4">
      {/* Header with sorting options */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Bot className="w-5 h-5" />
          Agent Status ({agents.length} agents)
        </h3>
        <div className="flex gap-2">
          <Badge 
            variant={sortBy === 'activity' ? 'default' : 'outline'}
            className="cursor-pointer"
            onClick={() => setSortBy('activity')}
          >
            Activity
          </Badge>
          <Badge 
            variant={sortBy === 'performance' ? 'default' : 'outline'}
            className="cursor-pointer"
            onClick={() => setSortBy('performance')}
          >
            Performance
          </Badge>
          <Badge 
            variant={sortBy === 'tokens' ? 'default' : 'outline'}
            className="cursor-pointer"
            onClick={() => setSortBy('tokens')}
          >
            Tokens
          </Badge>
        </div>
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <AnimatePresence>
          {sortedAgents.slice(0, 12).map((agent) => (
            <motion.div
              key={agent.agent_id}
              layout
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              whileHover={{ scale: 1.02 }}
              onClick={() => setSelectedAgent(agent.agent_id === selectedAgent ? null : agent.agent_id)}
            >
              <Card className="cursor-pointer hover:shadow-lg transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getAgentTypeIcon(agent.agent_type)}
                      <div>
                        <p className="font-semibold text-sm">{agent.name}</p>
                        <p className="text-xs text-muted-foreground">{agent.agent_type}</p>
                      </div>
                    </div>
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(agent.status)}`} />
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {/* Success Rate */}
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span>Success Rate</span>
                      <span>{agent.success_rate.toFixed(1)}%</span>
                    </div>
                    <Progress value={agent.success_rate} className="h-1" />
                  </div>

                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <p className="text-muted-foreground">Executions</p>
                      <p className="font-semibold">{agent.execution_count}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Avg Time</p>
                      <p className="font-semibold">{agent.avg_execution_time.toFixed(1)}s</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Tokens Used</p>
                      <p className="font-semibold">
                        {agent.total_tokens_used > 1000 
                          ? `${(agent.total_tokens_used / 1000).toFixed(1)}k`
                          : agent.total_tokens_used
                        }
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Memories</p>
                      <p className="font-semibold">{agent.memory_items_count}</p>
                    </div>
                  </div>

                  {/* Expanded Details */}
                  <AnimatePresence>
                    {selectedAgent === agent.agent_id && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="pt-2 border-t space-y-2"
                      >
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Collaborations</span>
                          <span>{agent.collaboration_count}</span>
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Last Active</span>
                          <span>{formatLastActive(agent.last_active)}</span>
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Status</span>
                          <Badge variant="outline" className="text-xs">
                            {agent.status}
                          </Badge>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Summary Stats */}
      <Card>
        <CardContent className="pt-6">
          <div className="grid grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-green-500">
                {agents.filter(a => a.status === 'active').length}
              </p>
              <p className="text-xs text-muted-foreground">Active</p>
            </div>
            <div>
              <p className="text-2xl font-bold">
                {agents.reduce((sum, a) => sum + a.execution_count, 0)}
              </p>
              <p className="text-xs text-muted-foreground">Total Executions</p>
            </div>
            <div>
              <p className="text-2xl font-bold">
                {(agents.reduce((sum, a) => sum + a.success_rate, 0) / agents.length).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">Avg Success</p>
            </div>
            <div>
              <p className="text-2xl font-bold">
                {(agents.reduce((sum, a) => sum + a.total_tokens_used, 0) / 1000).toFixed(1)}k
              </p>
              <p className="text-xs text-muted-foreground">Total Tokens</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

