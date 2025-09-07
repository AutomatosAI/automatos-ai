
'use client'

import * as React from 'react'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Bot, 
  Calendar, 
  Activity, 
  Settings, 
  CheckCircle, 
  AlertTriangle, 
  Clock, 
  Zap,
  Database,
  Eye,
  Edit,
  Pause,
  Play,
  Trash2,
  RefreshCw,
  TrendingUp,
  Users,
  Brain
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { apiClient } from '@/lib/api'

interface AgentDetailsModalProps {
  agentId: number | null
  open: boolean
  onClose: () => void
  onEdit?: (agentId: number) => void
  onToggleStatus?: (agentId: number, currentStatus: string) => void
  onDelete?: (agentId: number) => void
}

interface AgentDetails {
  id: number
  name: string
  description?: string
  agent_type: string
  status: string
  configuration?: Record<string, any>
  performance_metrics?: {
    success_rate?: number
    avg_response_time?: number
    tasks_completed?: number
    tasks_failed?: number
    uptime_hours?: number
    efficiency_score?: number
  }
  skills?: Array<{
    id: number
    name: string
    description?: string
    skill_type: string
    category?: string
    is_active: boolean
  }>
  created_at?: string
  updated_at?: string
  current_workload?: {
    active_workflows: number
    queued_tasks: number
    processing_capacity: number
    current_utilization: number
  }
  activity_timeline?: Array<{
    timestamp: string
    event_type: string
    description: string
    status: 'success' | 'warning' | 'error' | 'info'
  }>
  resource_usage?: {
    memory_mb: number
    cpu_percent: number
    network_io: number
    storage_mb: number
  }
}

const statusStyles: Record<string, string> = {
  active: 'bg-green-500/10 text-green-400 border-green-500/20',
  inactive: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
  training: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  maintenance: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  error: 'bg-red-500/10 text-red-400 border-red-500/20'
}

const statusIcons = {
  active: CheckCircle,
  inactive: Clock,
  training: RefreshCw,
  maintenance: Settings,
  error: AlertTriangle
}

const agentTypeIcons: Record<string, string> = {
  code_architect: '🏗️',
  security_expert: '🛡️',
  performance_optimizer: '⚡',
  data_analyst: '📊',
  infrastructure_manager: '☁️',
  custom: '🤖'
}

export function AgentDetailsModal({ 
  agentId, 
  open, 
  onClose, 
  onEdit,
  onToggleStatus,
  onDelete 
}: AgentDetailsModalProps) {
  const [agent, setAgent] = useState<AgentDetails | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    if (open && agentId) {
      loadAgentDetails()
    }
  }, [open, agentId])

  const loadAgentDetails = async () => {
    if (!agentId) return
    
    setLoading(true)
    setError(null)
    
    try {
      // Try to fetch real agent details
      const details = await apiClient.request<AgentDetails>(`/api/agents/${agentId}`)
      
      // Enhance with mock data for demonstration
      const enhancedDetails: AgentDetails = {
        ...details,
        current_workload: details?.current_workload || {
          active_workflows: Math.floor(Math.random() * 10) + 1,
          queued_tasks: Math.floor(Math.random() * 20),
          processing_capacity: 100,
          current_utilization: Math.floor(Math.random() * 80) + 10
        },
        activity_timeline: details?.activity_timeline || [
          {
            timestamp: new Date(Date.now() - 300000).toISOString(),
            event_type: 'task_completed',
            description: 'Completed code analysis task',
            status: 'success'
          },
          {
            timestamp: new Date(Date.now() - 600000).toISOString(),
            event_type: 'workflow_started',
            description: 'Started new workflow execution',
            status: 'info'
          },
          {
            timestamp: new Date(Date.now() - 1200000).toISOString(),
            event_type: 'skill_applied',
            description: 'Applied architecture design skill',
            status: 'success'
          }
        ],
        resource_usage: details?.resource_usage || {
          memory_mb: Math.floor(Math.random() * 1024) + 256,
          cpu_percent: Math.floor(Math.random() * 60) + 20,
          network_io: Math.floor(Math.random() * 1000),
          storage_mb: Math.floor(Math.random() * 512) + 128
        }
      }
      
      setAgent(enhancedDetails)
    } catch (err) {
      console.error('Error loading agent details:', err)
      setError(err instanceof Error ? err?.message : 'Failed to load agent details')
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleString()
  }

  const formatUptime = (hours?: number) => {
    if (!hours) return 'N/A'
    const days = Math.floor(hours / 24)
    const remainingHours = hours % 24
    return days > 0 ? `${days}d ${remainingHours}h` : `${remainingHours}h`
  }

  const handleEdit = () => {
    if (agent && onEdit) {
      onEdit(agent.id)
    }
  }

  const handleToggleStatus = () => {
    if (agent && onToggleStatus) {
      onToggleStatus(agent.id, agent.status)
    }
  }

  const handleDelete = () => {
    if (agent && onDelete) {
      onDelete(agent.id)
      onClose()
    }
  }

  if (!open || !agentId) return null

  return (
    <AnimatePresence>
      <motion.div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        exit={{ opacity: 0 }} 
        onClick={onClose}
      />
      <motion.div 
        className="fixed inset-0 z-50 flex items-center justify-center p-4" 
        initial={{ opacity: 0, scale: 0.95 }} 
        animate={{ opacity: 1, scale: 1 }} 
        exit={{ opacity: 0, scale: 0.95 }}
      >
        <Card className="glass-card w-full max-w-6xl max-h-[90vh] overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
            <CardTitle className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                {agent?.agent_type ? agentTypeIcons[agent.agent_type] || '🤖' : <Bot className="w-5 h-5" />}
              </div>
              <div>
                <span className="text-xl">Agent Details</span>
                <p className="text-sm text-muted-foreground font-normal">
                  {agent?.name || 'Loading...'}
                </p>
              </div>
            </CardTitle>
            <div className="flex items-center space-x-2">
              {agent && (
                <>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={handleEdit}
                    className="hover:border-blue-500/50"
                  >
                    <Edit className="w-4 h-4 mr-2" />
                    Edit
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={handleToggleStatus}
                    className={agent.status === 'active' ? "hover:border-yellow-500/50" : "hover:border-green-500/50"}
                  >
                    {agent.status === 'active' ? (
                      <>
                        <Pause className="w-4 h-4 mr-2" />
                        Pause
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Start
                      </>
                    )}
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={handleDelete}
                    className="hover:border-red-500/50 text-red-400"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete
                  </Button>
                </>
              )}
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="w-5 h-5" />
              </Button>
            </div>
          </CardHeader>
          
          <CardContent className="overflow-y-auto p-0">
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading agent details...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
                  <p className="text-red-400 mb-4">Error: {error}</p>
                  <Button onClick={loadAgentDetails} variant="outline">
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {agent && (
              <Tabs value={activeTab} onValueChange={setActiveTab} className="p-6">
                <TabsList className="grid w-full grid-cols-4 bg-secondary/50">
                  <TabsTrigger value="overview" className="flex items-center space-x-2">
                    <Eye className="w-4 h-4" />
                    <span>Overview</span>
                  </TabsTrigger>
                  <TabsTrigger value="performance" className="flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4" />
                    <span>Performance</span>
                  </TabsTrigger>
                  <TabsTrigger value="workload" className="flex items-center space-x-2">
                    <Activity className="w-4 h-4" />
                    <span>Workload</span>
                  </TabsTrigger>
                  <TabsTrigger value="skills" className="flex items-center space-x-2">
                    <Brain className="w-4 h-4" />
                    <span>Skills</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-6 mt-6">
                  {/* Agent Information */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Agent Information</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Name</p>
                          <p className="font-semibold">{agent?.name}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Type</p>
                          <Badge variant="outline">
                            {agent?.agent_type?.replace('_', ' ').toUpperCase() || 'UNKNOWN'}
                          </Badge>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Status</p>
                          <Badge className={statusStyles[agent?.status?.toLowerCase()] || statusStyles.inactive}>
                            <CheckCircle className="w-3 h-3 mr-1" />
                            {agent?.status || 'inactive'}
                          </Badge>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Description</p>
                          <p className="text-sm">{agent?.description || 'No description available'}</p>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">System Info</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Created</p>
                          <p className="font-semibold">{formatDate(agent?.created_at)}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Last Updated</p>
                          <p className="font-semibold">{formatDate(agent?.updated_at)}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Uptime</p>
                          <p className="font-semibold">{formatUptime(agent?.performance_metrics?.uptime_hours)}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Agent ID</p>
                          <p className="font-semibold text-orange-400">#{agent?.id}</p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Quick Stats */}
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Performance Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center p-3 bg-background/50 rounded-lg">
                          <p className="text-2xl font-bold text-green-400">
                            {agent?.performance_metrics?.success_rate ? 
                              `${(agent.performance_metrics.success_rate * 100).toFixed(1)}%` : 
                              'N/A'
                            }
                          </p>
                          <p className="text-sm text-muted-foreground">Success Rate</p>
                        </div>
                        <div className="text-center p-3 bg-background/50 rounded-lg">
                          <p className="text-2xl font-bold text-blue-400">
                            {agent?.performance_metrics?.tasks_completed || 0}
                          </p>
                          <p className="text-sm text-muted-foreground">Tasks Done</p>
                        </div>
                        <div className="text-center p-3 bg-background/50 rounded-lg">
                          <p className="text-2xl font-bold text-purple-400">
                            {agent?.performance_metrics?.avg_response_time ? 
                              `${agent.performance_metrics.avg_response_time}ms` : 
                              'N/A'
                            }
                          </p>
                          <p className="text-sm text-muted-foreground">Avg Response</p>
                        </div>
                        <div className="text-center p-3 bg-background/50 rounded-lg">
                          <p className="text-2xl font-bold text-orange-400">
                            {agent?.skills?.length || 0}
                          </p>
                          <p className="text-sm text-muted-foreground">Skills</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="performance" className="space-y-6 mt-6">
                  {/* Performance Metrics */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Task Performance</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Success Rate</span>
                            <span>{agent?.performance_metrics?.success_rate ? 
                              `${(agent.performance_metrics.success_rate * 100).toFixed(1)}%` : 'N/A'}
                            </span>
                          </div>
                          <Progress 
                            value={agent?.performance_metrics?.success_rate ? 
                              agent.performance_metrics.success_rate * 100 : 0} 
                            className="h-2"
                          />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Efficiency Score</span>
                            <span>{agent?.performance_metrics?.efficiency_score ? 
                              `${(agent.performance_metrics.efficiency_score * 100).toFixed(1)}%` : 'N/A'}
                            </span>
                          </div>
                          <Progress 
                            value={agent?.performance_metrics?.efficiency_score ? 
                              agent.performance_metrics.efficiency_score * 100 : 75} 
                            className="h-2"
                          />
                        </div>
                        <div className="pt-4 grid grid-cols-2 gap-4 text-center">
                          <div>
                            <p className="text-lg font-semibold text-green-400">
                              {agent?.performance_metrics?.tasks_completed || 0}
                            </p>
                            <p className="text-xs text-muted-foreground">Completed</p>
                          </div>
                          <div>
                            <p className="text-lg font-semibold text-red-400">
                              {agent?.performance_metrics?.tasks_failed || 0}
                            </p>
                            <p className="text-xs text-muted-foreground">Failed</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Resource Usage</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>CPU Usage</span>
                            <span>{agent?.resource_usage?.cpu_percent}%</span>
                          </div>
                          <Progress value={agent?.resource_usage?.cpu_percent || 0} className="h-2" />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Memory Usage</span>
                            <span>{agent?.resource_usage?.memory_mb} MB</span>
                          </div>
                          <Progress value={(agent?.resource_usage?.memory_mb || 0) / 10.24} className="h-2" />
                        </div>
                        <div className="pt-4 grid grid-cols-2 gap-4 text-center">
                          <div>
                            <p className="text-lg font-semibold text-blue-400">
                              {agent?.resource_usage?.network_io || 0} KB/s
                            </p>
                            <p className="text-xs text-muted-foreground">Network I/O</p>
                          </div>
                          <div>
                            <p className="text-lg font-semibold text-purple-400">
                              {agent?.resource_usage?.storage_mb || 0} MB
                            </p>
                            <p className="text-xs text-muted-foreground">Storage</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Activity Timeline */}
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Recent Activity</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {agent?.activity_timeline?.map((activity, index) => (
                          <div key={index} className="flex items-start space-x-3 p-3 bg-background/50 rounded-lg">
                            <div className={`w-2 h-2 rounded-full mt-2 ${
                              activity.status === 'success' ? 'bg-green-400' :
                              activity.status === 'warning' ? 'bg-yellow-400' :
                              activity.status === 'error' ? 'bg-red-400' : 'bg-blue-400'
                            }`} />
                            <div className="flex-1">
                              <p className="text-sm font-medium">{activity.description}</p>
                              <p className="text-xs text-muted-foreground">
                                {formatDate(activity.timestamp)} • {activity.event_type.replace('_', ' ')}
                              </p>
                            </div>
                          </div>
                        )) || (
                          <p className="text-center text-muted-foreground py-4">No recent activity</p>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="workload" className="space-y-6 mt-6">
                  {/* Current Workload */}
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Current Workload</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <div className="text-center p-4 bg-background/50 rounded-lg">
                          <Users className="w-8 h-8 text-blue-400 mx-auto mb-2" />
                          <p className="text-2xl font-bold">{agent?.current_workload?.active_workflows || 0}</p>
                          <p className="text-sm text-muted-foreground">Active Workflows</p>
                        </div>
                        <div className="text-center p-4 bg-background/50 rounded-lg">
                          <Clock className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
                          <p className="text-2xl font-bold">{agent?.current_workload?.queued_tasks || 0}</p>
                          <p className="text-sm text-muted-foreground">Queued Tasks</p>
                        </div>
                        <div className="text-center p-4 bg-background/50 rounded-lg">
                          <Database className="w-8 h-8 text-green-400 mx-auto mb-2" />
                          <p className="text-2xl font-bold">{agent?.current_workload?.processing_capacity || 0}</p>
                          <p className="text-sm text-muted-foreground">Capacity</p>
                        </div>
                        <div className="text-center p-4 bg-background/50 rounded-lg">
                          <TrendingUp className="w-8 h-8 text-purple-400 mx-auto mb-2" />
                          <p className="text-2xl font-bold">{agent?.current_workload?.current_utilization || 0}%</p>
                          <p className="text-sm text-muted-foreground">Utilization</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Utilization Chart */}
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Capacity Utilization</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Current Utilization</span>
                            <span>{agent?.current_workload?.current_utilization || 0}%</span>
                          </div>
                          <Progress value={agent?.current_workload?.current_utilization || 0} className="h-3" />
                        </div>
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Idle</span>
                          <span>Optimal</span>
                          <span>Overloaded</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="skills" className="space-y-6 mt-6">
                  {/* Skills Overview */}
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Assigned Skills</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {agent?.skills && agent.skills.length > 0 ? (
                        <div className="space-y-4">
                          {agent.skills.map((skill) => (
                            <div key={skill.id} className="flex items-center justify-between p-3 bg-background/50 rounded-lg">
                              <div className="flex items-center space-x-3">
                                <div className={`w-3 h-3 rounded-full ${skill.is_active ? 'bg-green-400' : 'bg-gray-400'}`} />
                                <div>
                                  <h4 className="font-medium">{skill.name}</h4>
                                  <p className="text-sm text-muted-foreground">
                                    {skill.description || 'No description available'}
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-center space-x-2">
                                <Badge variant="outline" className="text-xs">
                                  {skill.skill_type?.replace('_', ' ') || 'Unknown'}
                                </Badge>
                                {skill.category && (
                                  <Badge variant="secondary" className="text-xs">
                                    {skill.category}
                                  </Badge>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-8">
                          <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2">No Skills Assigned</h3>
                          <p className="text-muted-foreground">
                            This agent doesn't have any skills assigned yet.
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Skills Performance */}
                  {agent?.skills && agent.skills.length > 0 && (
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Skills Performance</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {agent.skills.slice(0, 5).map((skill) => (
                            <div key={skill.id} className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span>{skill.name}</span>
                                <span>{Math.floor(Math.random() * 30) + 70}% efficiency</span>
                              </div>
                              <Progress value={Math.floor(Math.random() * 30) + 70} className="h-2" />
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
