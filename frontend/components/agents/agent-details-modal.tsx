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
  Brain,
  Share2,
  MoreVertical
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { useAgent, useAgentStats, useAgentLogs, useAgentPerformance } from '@/hooks/use-agent-api'
import { apiClient } from '@/lib/api-client'
import { useSubmitToMarketplace } from '@/hooks/use-marketplace-api'

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
  active: 'bg-[hsl(var(--success))]/10 text-[hsl(var(--success))] border-[hsl(var(--success))]/20',
  inactive: 'bg-secondary/50 text-muted-foreground border-border/30',
  training: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20',
  maintenance: 'bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/20',
  error: 'bg-[hsl(var(--destructive))]/10 text-[hsl(var(--destructive))] border-[hsl(var(--destructive))]/20'
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
  const [activeTab, setActiveTab] = useState('overview')
  const [showShareDialog, setShowShareDialog] = useState(false)

  // Use real API hooks
  const { data: agent, isLoading: loading, error: agentError } = useAgent(agentId?.toString() || '')
  const { data: agentPerformance } = useAgentPerformance(agentId?.toString() || '')
  const { data: agentLogs } = useAgentLogs(agentId?.toString() || '')
  // Plugin state for the Plugins tab
  const [plugins, setPlugins] = useState<any[]>([])
  const [pluginsLoading, setPluginsLoading] = useState(false)

  const error = agentError?.message || null

  // Fetch assigned plugins when modal opens
  useEffect(() => {
    if (!open || !agentId) return
    let mounted = true
    setPluginsLoading(true)
    apiClient.request<any>(`/api/agents/${agentId}/plugins`, { method: 'GET' })
      .then((data) => {
        if (!mounted) return
        const items = Array.isArray(data) ? data : (data?.items || data?.plugins || [])
        setPlugins(items)
      })
      .catch((err) => {
        console.error('Failed to fetch agent plugins:', err)
        if (mounted) setPlugins([])
      })
      .finally(() => { if (mounted) setPluginsLoading(false) })
    return () => { mounted = false }
  }, [open, agentId])

  // Marketplace submission
  const { mutate: submitToMarketplace, isPending: isSubmitting } = useSubmitToMarketplace()


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

  const handleShare = () => {
    setShowShareDialog(true)
  }

  const confirmShare = () => {
    if (!agent) return

    submitToMarketplace({
      item_type: 'agent',
      category: agent.agent_type,
      metadata: {
        agent_id: agent.id,
        icon: agentTypeIcons[agent.agent_type] || '🤖'
      }
    }, {
      onSuccess: () => {
        setShowShareDialog(false)
      }
    })
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
        <Card className="glass-card card-glow w-full max-w-6xl max-h-[90vh] overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
            <CardTitle className="flex items-center space-x-3">
              <Eye className="w-6 h-6 text-primary" />
              <div>
                <span className="text-xl">Agent <span className="gradient-text">Details</span></span>
                <p className="text-sm text-muted-foreground font-normal">
                  {agent?.name || 'Loading...'}
                </p>
              </div>
            </CardTitle>
            <div className="flex items-center space-x-2">
              {agent && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <MoreVertical className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={handleEdit}>
                      <Edit className="w-4 h-4 mr-2" />
                      Edit
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={handleShare} disabled={isSubmitting}>
                      <Share2 className="w-4 h-4 mr-2" />
                      {isSubmitting ? 'Sharing...' : 'Share to Marketplace'}
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={handleToggleStatus}>
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
                      onClick={handleDelete}
                      className="text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))] hover:bg-[hsl(var(--destructive))]/10"
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="w-5 h-5" />
              </Button>
            </div>
          </CardHeader>
          
          <CardContent className="overflow-y-auto p-6">
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
                  <AlertTriangle className="h-8 w-8 text-[hsl(var(--destructive))] mx-auto mb-4" />
                  <p className="text-[hsl(var(--destructive))] mb-4">Error: {error}</p>
                  <Button onClick={() => window.location.reload()} variant="outline">
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {agent && (
              <Tabs value={activeTab} onValueChange={setActiveTab}>
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
                  <TabsTrigger value="plugins" className="flex items-center space-x-2">
                    <Puzzle className="w-4 h-4" />
                    <span>Plugins</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-6 mt-6">
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
                          <p className="font-semibold text-primary">#{agent?.id}</p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Performance Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center p-3 bg-background/50 rounded-lg">
                          <p className="text-2xl font-bold text-[hsl(var(--success))]">
                            {agentPerformance?.success_rate != null ? 
                              `${agentPerformance.success_rate}%` : 
                              'N/A'
                            }
                          </p>
                          <p className="text-sm text-muted-foreground">Success Rate</p>
                        </div>
                        <div className="text-center p-3 bg-background/50 rounded-lg">
                          <p className="text-2xl font-bold text-[hsl(var(--info))]">
                            {agentPerformance?.completed_tasks || 0}
                          </p>
                          <p className="text-sm text-muted-foreground">Tasks Done</p>
                        </div>
                        <div className="text-center p-3 bg-background/50 rounded-lg">
                          <p className="text-2xl font-bold text-[hsl(var(--agent))]">
                            {agentPerformance?.average_response_time != null ? 
                              `${agentPerformance.average_response_time}s` : 
                              'N/A'
                            }
                          </p>
                          <p className="text-sm text-muted-foreground">Avg Response</p>
                        </div>
                        <div className="text-center p-3 bg-background/50 rounded-lg">
                          <p className="text-2xl font-bold text-primary">
                            {skills?.length || 0}
                          </p>
                          <p className="text-sm text-muted-foreground">Plugins</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="performance" className="space-y-6 mt-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="bg-secondary/30 border-border/30">
                      <CardHeader>
                        <CardTitle className="text-base">Task Performance</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Success Rate</span>
                            <span>{agentPerformance?.success_rate != null ? 
                              `${agentPerformance.success_rate}%` : 'N/A'}
                            </span>
                          </div>
                          <Progress 
                            value={agentPerformance?.success_rate || 0} 
                            className="h-2"
                          />
                        </div>
                        <div className="pt-4 grid grid-cols-2 gap-4 text-center">
                          <div>
                            <p className="text-lg font-semibold text-[hsl(var(--success))]">
                              {agentPerformance?.completed_tasks || 0}
                            </p>
                            <p className="text-xs text-muted-foreground">Completed</p>
                          </div>
                          <div>
                            <p className="text-lg font-semibold text-[hsl(var(--destructive))]">
                              {agentPerformance?.failed_tasks || 0}
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
                            <span>Total Tokens Used</span>
                            <span>{agentPerformance?.total_tokens_used?.toLocaleString() || 0}</span>
                          </div>
                          <Progress 
                            value={Math.min((agentPerformance?.total_tokens_used || 0) / 100000 * 100, 100)} 
                            className="h-2" 
                          />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Total Cost</span>
                            <span>${agentPerformance?.total_cost?.toFixed(2) || '0.00'}</span>
                          </div>
                          <Progress 
                            value={Math.min((agentPerformance?.total_cost || 0) / 10 * 100, 100)} 
                            className="h-2" 
                          />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Avg Tokens per Task</span>
                            <span>{agentPerformance?.average_tokens_per_task || 0}</span>
                          </div>
                          <Progress 
                            value={Math.min((agentPerformance?.average_tokens_per_task || 0) / 5000 * 100, 100)} 
                            className="h-2" 
                          />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Total Requests</span>
                            <span>{agentPerformance?.total_requests || 0}</span>
                          </div>
                          <Progress 
                            value={Math.min((agentPerformance?.total_requests || 0) / 1000 * 100, 100)} 
                            className="h-2" 
                          />
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="workload" className="space-y-6 mt-6">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Current Workload</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <div className="text-center p-4 bg-background/50 rounded-lg">
                          <Users className="w-8 h-8 text-[hsl(var(--info))] mx-auto mb-2" />
                          <p className="text-2xl font-bold">{agentPerformance?.total_tasks || 0}</p>
                          <p className="text-sm text-muted-foreground">Total Tasks</p>
                        </div>
                        <div className="text-center p-4 bg-background/50 rounded-lg">
                          <Clock className="w-8 h-8 text-[hsl(var(--warning))] mx-auto mb-2" />
                          <p className="text-2xl font-bold">{agentPerformance?.total_requests || 0}</p>
                          <p className="text-sm text-muted-foreground">Total Requests</p>
                        </div>
                        <div className="text-center p-4 bg-background/50 rounded-lg">
                          <Database className="w-8 h-8 text-[hsl(var(--success))] mx-auto mb-2" />
                          <p className="text-2xl font-bold">{agentPerformance?.completed_tasks || 0}</p>
                          <p className="text-sm text-muted-foreground">Completed</p>
                        </div>
                        <div className="text-center p-4 bg-background/50 rounded-lg">
                          <TrendingUp className="w-8 h-8 text-[hsl(var(--agent))] mx-auto mb-2" />
                          <p className="text-2xl font-bold">{agentPerformance?.uptime_percentage || 0}%</p>
                          <p className="text-sm text-muted-foreground">Uptime</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  {/* Activity Timeline */}
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base">Recent Activity</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {agentLogs && agentLogs.length > 0 ? (
                        <div className="space-y-3">
                          {agentLogs.slice(0, 5).map((log: any, index: number) => (
                            <div key={index} className="flex items-center space-x-3 p-3 bg-background/50 rounded-lg">
                              <div className={`w-2 h-2 rounded-full ${
                                log.status === 'completed' ? 'bg-[hsl(var(--success))]' :
                                log.status === 'failed' ? 'bg-[hsl(var(--destructive))]' :
                                'bg-[hsl(var(--warning))]'
                              }`} />
                              <div className="flex-1">
                                <p className="text-sm font-medium">{log.message}</p>
                                <p className="text-xs text-muted-foreground">
                                  {log.timestamp ? new Date(log.timestamp).toLocaleString() : 'Recent'}
                                  {log.tokens_used && ` • ${log.tokens_used} tokens`}
                                  {log.execution_time_ms && ` • ${log.execution_time_ms}ms`}
                                </p>
                              </div>
                              <Badge variant="outline" className="text-xs">
                                {log.status || 'unknown'}
                              </Badge>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-8">
                          <Activity className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2">No Recent Activity</h3>
                          <p className="text-muted-foreground">
                            This agent hasn't executed any tasks recently.
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="plugins" className="space-y-6 mt-6">
                  <Card className="bg-secondary/30 border-border/30">
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Puzzle className="h-5 w-5 text-orange-400" />
                        Assigned Plugins
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {pluginsLoading ? (
                        <div className="flex items-center justify-center py-8">
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
                        </div>
                      ) : plugins && plugins.length > 0 ? (
                        <div className="space-y-4">
                          {skills.map((skill: any) => (
                            <div key={skill.id} className="flex items-center justify-between p-3 bg-background/50 rounded-lg">
                              <div className="flex items-center space-x-3">
                                <div className="w-3 h-3 rounded-full bg-[hsl(var(--success))]" />
                                <div>
                                  <h4 className="font-medium">{skill.name}</h4>
                                  <p className="text-sm text-muted-foreground">
                                    {skill.description || 'No description available'}
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-center gap-3 shrink-0 ml-4 text-xs text-muted-foreground">
                                <span className="flex items-center gap-1">
                                  <Terminal className="w-3 h-3" />
                                  {plugin.skills_count || 0} skills
                                </span>
                                <span className="flex items-center gap-1">
                                  <Zap className="w-3 h-3" />
                                  {plugin.commands_count || 0} cmds
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-8">
                          <Puzzle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2">No Plugins Assigned</h3>
                          <p className="text-muted-foreground">
                            This agent doesn't have any plugins assigned yet. Use the configuration modal to assign plugins.
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Share to Marketplace Confirmation Dialog */}
      <AlertDialog open={showShareDialog} onOpenChange={setShowShareDialog}>
        <AlertDialogContent className="glass-card card-glow">
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center space-x-2">
              <Share2 className="w-5 h-5 text-primary" />
              <span>Share Agent to Marketplace</span>
            </AlertDialogTitle>
            <AlertDialogDescription className="space-y-3">
              <p>
                Are you sure you want to share <strong className="text-foreground">{agent?.name}</strong> to the marketplace?
              </p>
              <div className="bg-primary/10 border border-primary/20 rounded-lg p-3 text-sm">
                <p className="text-primary font-semibold mb-1">📝 What happens next:</p>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Your agent will be cloned to the marketplace</li>
                  <li>• Other users can browse and install it</li>
                  <li>• Your original agent stays in your workspace</li>
                  <li>• You'll be credited as the creator</li>
                </ul>
              </div>
              <div className="bg-primary/10 border border-primary/20 rounded-lg p-3 text-sm">
                <p className="text-primary font-semibold mb-1">⚡ Approval Status:</p>
                <p className="text-muted-foreground">
                  Trusted creators (5+ approved items) get instant approval. New contributors go through a quick review process.
                </p>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isSubmitting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmShare}
              disabled={isSubmitting}
              className="bg-gradient-to-r from-primary to-[hsl(var(--destructive))] hover:opacity-90"
            >
              {isSubmitting ? 'Sharing...' : 'Share to Marketplace'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AnimatePresence>
  )
}
