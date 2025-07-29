
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Play, 
  Pause, 
  Square, 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  Activity,
  Users,
  TrendingUp,
  Eye,
  MoreVertical,
  RefreshCw
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu'
import { apiClient } from '@/lib/api'
import { LiveProgressPanel } from './live-progress-panel'

interface ActiveWorkflow {
  id: number;
  name: string;
  description: string;
  status: string;
  agents: Array<{
    id: number;
    name: string;
    agent_type: string;
    status: string;
  }>;
  current_execution: {
    id: number | null;
    status: string;
    progress: number;
    current_step: string;
    started_at: string | null;
    estimated_completion: string | null;
  };
  metrics: {
    total_executions: number;
    successful_executions: number;
    success_rate: number;
    avg_duration: string;
    last_execution: string | null;
  };
  recent_executions: Array<{
    id: number;
    status: string;
    started_at: string;
    completed_at: string | null;
    duration: string | null;
  }>;
  created_at: string;
  updated_at: string;
}

interface ActiveWorkflowsData {
  active_workflows: ActiveWorkflow[];
  total_active: number;
  system_load: number;
  last_updated: string;
}

export function ActiveWorkflowsPanel() {
  const [workflowsData, setWorkflowsData] = useState<ActiveWorkflowsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [selectedWorkflow, setSelectedWorkflow] = useState<{id: number, name: string} | null>(null)

  useEffect(() => {
    loadActiveWorkflows()
    
    if (autoRefresh) {
      const interval = setInterval(loadActiveWorkflows, 5000) // Refresh every 5 seconds
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const loadActiveWorkflows = async () => {
    try {
      setError(null)
      const data = await apiClient.request<ActiveWorkflowsData>('/api/workflows/active')
      setWorkflowsData(data)
    } catch (err) {
      console.error('Error loading active workflows:', err)
      setError(err instanceof Error ? err.message : 'Failed to load active workflows')
    } finally {
      setLoading(false)
    }
  }

  const handleExecuteWorkflow = async (workflowId: number) => {
    try {
      await apiClient.request(`/api/workflows/${workflowId}/execute-advanced`, {
        method: 'POST',
        body: JSON.stringify({
          options: {
            priority: 'normal',
            timeout: 300
          }
        })
      })
      
      // Refresh data after execution
      setTimeout(loadActiveWorkflows, 1000)
    } catch (err) {
      console.error('Error executing workflow:', err)
      setError(err instanceof Error ? err.message : 'Failed to execute workflow')
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return Activity
      case 'completed': return CheckCircle
      case 'failed': return AlertTriangle
      case 'idle': return Clock
      default: return Clock
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-blue-400'
      case 'completed': return 'text-green-400'
      case 'failed': return 'text-red-400'
      case 'idle': return 'text-gray-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusBadgeStyle = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-500/10 text-blue-400 border-blue-500/20'
      case 'completed': return 'bg-green-500/10 text-green-400 border-green-500/20'
      case 'failed': return 'bg-red-500/10 text-red-400 border-red-500/20'
      case 'idle': return 'bg-gray-500/10 text-gray-400 border-gray-500/20'
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading active workflows...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
          <p className="text-red-400 mb-4">Error: {error}</p>
          <Button onClick={loadActiveWorkflows} variant="outline">
            Try Again
          </Button>
        </div>
      </div>
    )
  }

  if (!workflowsData || workflowsData.active_workflows.length === 0) {
    return (
      <div className="text-center py-12">
        <Activity className="h-8 w-8 text-muted-foreground mx-auto mb-4" />
        <p className="text-muted-foreground">No active workflows found</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Active Workflows</h3>
          <p className="text-sm text-muted-foreground">
            {workflowsData.total_active} active • {workflowsData.system_load}% system load
          </p>
        </div>
        
        <Button
          variant="outline"
          size="sm"
          onClick={() => setAutoRefresh(!autoRefresh)}
          className={autoRefresh ? 'bg-green-500/10 border-green-500/20' : ''}
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
          Auto Refresh
        </Button>
      </div>

      {/* Active Workflows Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {workflowsData.active_workflows.map((workflow, index) => {
          const StatusIcon = getStatusIcon(workflow.current_execution.status)
          const statusColor = getStatusColor(workflow.current_execution.status)
          
          return (
            <motion.div
              key={workflow.id}
              className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-lg mb-1">{workflow.name}</h4>
                  <p className="text-sm text-muted-foreground line-clamp-2">
                    {workflow.description}
                  </p>
                </div>
                
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <MoreVertical className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => setSelectedWorkflow({id: workflow.id, name: workflow.name})}>
                      <Eye className="w-4 h-4 mr-2" />
                      View Live Progress
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleExecuteWorkflow(workflow.id)}>
                      <Play className="w-4 h-4 mr-2" />
                      Execute Workflow
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              {/* Current Execution Status */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <StatusIcon className={`w-4 h-4 ${statusColor}`} />
                    <span className="text-sm font-medium">{workflow.current_execution.current_step}</span>
                  </div>
                  <Badge className={getStatusBadgeStyle(workflow.current_execution.status)}>
                    {workflow.current_execution.status}
                  </Badge>
                </div>
                
                {workflow.current_execution.status === 'running' && (
                  <>
                    <Progress value={workflow.current_execution.progress} className="h-2 mb-2" />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>{workflow.current_execution.progress}% complete</span>
                      <span>
                        ETA: {workflow.current_execution.estimated_completion ? 
                          new Date(workflow.current_execution.estimated_completion).toLocaleTimeString() : 
                          'Calculating...'
                        }
                      </span>
                    </div>
                  </>
                )}
              </div>

              {/* Agents */}
              <div className="mb-4">
                <div className="text-sm text-muted-foreground mb-2">
                  Agents ({workflow.agents.length})
                </div>
                <div className="flex flex-wrap gap-1">
                  {workflow.agents.map(agent => (
                    <Badge key={agent.id} variant="outline" className="text-xs">
                      {agent.name}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-4 mb-4 text-xs">
                <div>
                  <p className="text-muted-foreground">Total Runs</p>
                  <p className="font-medium">{workflow.metrics.total_executions}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Success Rate</p>
                  <p className="font-medium">{workflow.metrics.success_rate}%</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Avg Duration</p>
                  <p className="font-medium">{workflow.metrics.avg_duration}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Last Run</p>
                  <p className="font-medium">
                    {workflow.metrics.last_execution ? 
                      new Date(workflow.metrics.last_execution).toLocaleDateString() : 
                      'Never'
                    }
                  </p>
                </div>
              </div>

              {/* Actions */}
              <div className="flex justify-between items-center pt-4 border-t border-border/30">
                <div className="text-xs text-muted-foreground">
                  Created: {new Date(workflow.created_at).toLocaleDateString()}
                </div>
                
                <div className="flex space-x-2">
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => setSelectedWorkflow({id: workflow.id, name: workflow.name})}
                  >
                    <Eye className="w-3 h-3 mr-1" />
                    View Progress
                  </Button>
                  
                  {workflow.current_execution.status === 'idle' && (
                    <Button 
                      size="sm" 
                      className="gradient-accent hover:opacity-90"
                      onClick={() => handleExecuteWorkflow(workflow.id)}
                    >
                      <Play className="w-3 h-3 mr-1" />
                      Execute
                    </Button>
                  )}
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Live Progress Panel */}
      {selectedWorkflow && (
        <LiveProgressPanel
          workflowId={selectedWorkflow.id}
          workflowName={selectedWorkflow.name}
          isOpen={!!selectedWorkflow}
          onClose={() => setSelectedWorkflow(null)}
        />
      )}
    </div>
  )
}
