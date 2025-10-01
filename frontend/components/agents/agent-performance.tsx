
'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown,
  Activity,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Target,
  Zap,
  Cpu,
  MemoryStick,
  HardDrive,
  RefreshCw,
  Calendar,
  Filter
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'

// API hooks
import { 
  useAgentPerformance, 
  useAgentLogs, 
  useAgent,
  useAgents 
} from '@/hooks/use-agent-api'

interface AgentPerformanceProps {
  agents: any[]
  agentStats: any
  selectedAgentId: string | null
  onAgentSelect?: (agentId: string | null) => void
  showAnalytics?: boolean
}

export function AgentPerformance({ 
  agents, 
  agentStats, 
  selectedAgentId,
  onAgentSelect,
  showAnalytics = false 
}: AgentPerformanceProps) {
  const [timeRange, setTimeRange] = useState('24h')
  const [metricFilter, setMetricFilter] = useState('all')

  // Fetch performance data
  const { data: selectedAgent } = useAgent(selectedAgentId)
  const { data: agentPerformance, isLoading: performanceLoading } = useAgentPerformance(selectedAgentId)
  const { data: agentLogs, isLoading: logsLoading } = useAgentLogs(selectedAgentId)

  // Calculate performance metrics
  const performanceMetrics = useMemo(() => {
    if (!agentPerformance) return null

    return {
      overallScore: agentPerformance.overall_score || 95,
      successRate: agentPerformance.success_rate || 92,
      averageResponseTime: agentPerformance.average_response_time || 1.2,
      totalTasks: agentPerformance.total_tasks || 156,
      completedTasks: agentPerformance.completed_tasks || 144,
      failedTasks: agentPerformance.failed_tasks || 12,
      averageMemoryUsage: agentPerformance.average_memory_usage || 65,
      averageCpuUsage: agentPerformance.average_cpu_usage || 35,
      uptime: agentPerformance.uptime_percentage || 99.8
    }
  }, [agentPerformance])

  // Calculate system-wide analytics
  const systemAnalytics = useMemo(() => {
    if (!agentStats) return null

    const totalAgents = agentStats.total_agents || 0
    const activeAgents = agentStats.active_agents || 0
    const averagePerformance = agentStats.average_performance || 0

    return {
      totalAgents,
      activeAgents,
      inactiveAgents: totalAgents - activeAgents,
      averagePerformance,
      utilizationRate: totalAgents > 0 ? (activeAgents / totalAgents) * 100 : 0,
      agentsByType: agentStats.agents_by_type || {},
      totalExecutions: agentStats.total_executions || 0,
      successfulExecutions: agentStats.successful_executions || 0,
      failedExecutions: agentStats.failed_executions || 0
    }
  }, [agentStats])

  // TODO: Fetch performance trend data from API endpoint
  // For now, using empty array - will be populated when API endpoint is ready
  const performanceTrend: Array<{time: string, performance: number}> = []

  if (showAnalytics) {
    // Show system-wide analytics
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">System Analytics</h2>
            <p className="text-muted-foreground">
              Comprehensive performance analytics across all agents
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">Last Hour</SelectItem>
                <SelectItem value="24h">Last 24h</SelectItem>
                <SelectItem value="7d">Last 7 Days</SelectItem>
                <SelectItem value="30d">Last 30 Days</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* System Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-transparent">
                  <Activity className="w-5 h-5 text-blue-400" />
                </div>
              </div>
              <div className="space-y-1">
                <h3 className="text-2xl font-bold">{systemAnalytics?.totalAgents || 0}</h3>
                <p className="text-muted-foreground text-sm">Total Agents</p>
                <p className="text-xs text-muted-foreground">
                  {systemAnalytics?.activeAgents || 0} active
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-transparent">
                  <TrendingUp className="w-5 h-5 text-green-400" />
                </div>
              </div>
              <div className="space-y-1">
                <h3 className="text-2xl font-bold">{systemAnalytics?.averagePerformance.toFixed(1) || '0'}%</h3>
                <p className="text-muted-foreground text-sm">Avg Performance</p>
                <p className="text-xs text-green-400">+2.3%</p>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-transparent">
                  <Target className="w-5 h-5 text-orange-400" />
                </div>
              </div>
              <div className="space-y-1">
                <h3 className="text-2xl font-bold">{systemAnalytics?.utilizationRate.toFixed(0) || 0}%</h3>
                <p className="text-muted-foreground text-sm">Utilization</p>
                <p className="text-xs text-orange-400">CPU usage based</p>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-transparent">
                  <CheckCircle className="w-5 h-5 text-purple-400" />
                </div>
              </div>
              <div className="space-y-1">
                <h3 className="text-2xl font-bold">
                  {systemAnalytics?.totalExecutions > 0 
                    ? ((systemAnalytics.successfulExecutions / systemAnalytics.totalExecutions) * 100).toFixed(1)
                    : '0'
                  }%
                </h3>
                <p className="text-muted-foreground text-sm">Success Rate</p>
                <p className="text-xs text-purple-400">
                  {systemAnalytics?.successfulExecutions || 0} / {systemAnalytics?.totalExecutions || 0} tasks
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Agent Types Distribution */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Agent Distribution by Type</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(systemAnalytics?.agentsByType || {}).map(([type, count]) => (
                <div key={type} className="text-center">
                  <div className="text-2xl font-bold text-primary">{count as number}</div>
                  <div className="text-sm text-muted-foreground capitalize">
                    {type.replace('_', ' ')}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!selectedAgentId) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-orange-400" />
            <span>Agent Performance</span>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Select an agent to view detailed performance metrics and analytics
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Select Agent</Label>
            <Select onValueChange={(value) => onAgentSelect?.(value)}>
              <SelectTrigger>
                <SelectValue placeholder="Choose an agent to view performance" />
              </SelectTrigger>
              <SelectContent>
                {agents.map((agent) => (
                  <SelectItem key={agent.id} value={agent.id.toString()}>
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${
                        agent.status === "active" ? "bg-green-400" : "bg-gray-400"
                      }`} />
                      <span>{agent.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {agent.agent_type?.replace("_", " ") || "Unknown"}
                      </Badge>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {agents.length === 0 && (
            <div className="text-center py-8">
              <BarChart3 className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Agents Available</h3>
              <p className="text-muted-foreground">
                Create an agent first to view performance metrics
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Agent Performance</h2>
          <p className="text-muted-foreground">
            Detailed performance analytics for {selectedAgent?.name || 'selected agent'}
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="24h">Last 24h</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
          
          <Button variant="outline" size="sm" disabled={performanceLoading}>
            <RefreshCw className={`w-4 h-4 mr-2 ${performanceLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Agent Info */}
      {selectedAgent && (
        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center text-white text-xl">
                🤖
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold">{selectedAgent.name}</h3>
                <p className="text-sm text-muted-foreground capitalize">
                  {selectedAgent.agent_type?.replace('_', ' ')} • Created {new Date(selectedAgent.created_at).toLocaleDateString()}
                </p>
              </div>
              <Badge variant="outline" className="capitalize">
                {selectedAgent.status}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Metrics */}
      {performanceLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="glass-card">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <Skeleton className="h-4 w-24 mb-2" />
                    <Skeleton className="h-8 w-16 mb-2" />
                    <Skeleton className="h-3 w-20" />
                  </div>
                  <Skeleton className="w-12 h-12 rounded-xl" />
            </div>
              </CardContent>
            </Card>
        ))}
      </div>
      ) : performanceMetrics ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Overall Score</p>
                  <p className="text-2xl font-bold">{performanceMetrics.overallScore}%</p>
                  <div className="flex items-center gap-1 mt-1">
                    <TrendingUp className="w-3 h-3 text-green-500" />
                    <span className="text-xs text-green-500">Excellent</span>
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-gradient-to-br from-green-500 to-green-600">
                  <Target className="w-6 h-6 text-white" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold">{performanceMetrics.successRate}%</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {performanceMetrics.completedTasks}/{performanceMetrics.totalTasks} tasks
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600">
                  <CheckCircle className="w-6 h-6 text-white" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Avg Response</p>
                  <p className="text-2xl font-bold">{performanceMetrics.averageResponseTime}s</p>
                  <div className="flex items-center gap-1 mt-1">
                    <Clock className="w-3 h-3 text-blue-500" />
                    <span className="text-xs text-blue-500">Fast</span>
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500 to-purple-600">
                  <Clock className="w-6 h-6 text-white" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Uptime</p>
                  <p className="text-2xl font-bold">{performanceMetrics.uptime}%</p>
                  <Progress value={performanceMetrics.uptime} className="mt-2 h-2" />
                </div>
                <div className="p-3 rounded-xl bg-gradient-to-br from-orange-500 to-orange-600">
                  <Activity className="w-6 h-6 text-white" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : null}

      {/* Detailed Performance Tabs */}
      <Tabs defaultValue="metrics" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="metrics">Resource Usage</TabsTrigger>
          <TabsTrigger value="tasks">Task History</TabsTrigger>
          <TabsTrigger value="trends">Performance Trends</TabsTrigger>
          <TabsTrigger value="logs">Activity Logs</TabsTrigger>
        </TabsList>

        {/* Resource Usage */}
        <TabsContent value="metrics" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="w-5 h-5" />
                  CPU Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-3xl font-bold">
                    {performanceMetrics?.averageCpuUsage || 0}%
                  </div>
                  <Progress value={performanceMetrics?.averageCpuUsage || 0} className="h-3" />
                  <p className="text-sm text-muted-foreground">
                    Average over {timeRange}
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MemoryStick className="w-5 h-5" />
                  Memory Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-3xl font-bold">
                    {performanceMetrics?.averageMemoryUsage || 0}%
                  </div>
                  <Progress value={performanceMetrics?.averageMemoryUsage || 0} className="h-3" />
                  <p className="text-sm text-muted-foreground">
                    512MB allocated
                  </p>
                </div>
              </CardContent>
            </Card>

        <Card className="glass-card">
          <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <HardDrive className="w-5 h-5" />
                  Storage Usage
                </CardTitle>
          </CardHeader>
          <CardContent>
                <div className="space-y-4">
                  <div className="text-3xl font-bold">23%</div>
                  <Progress value={23} className="h-3" />
                  <p className="text-sm text-muted-foreground">
                    1.2GB used of 5GB
                  </p>
            </div>
          </CardContent>
        </Card>
          </div>
        </TabsContent>

        {/* Task History */}
        <TabsContent value="tasks" className="space-y-6">
        <Card className="glass-card">
          <CardHeader>
              <CardTitle>Recent Task Executions</CardTitle>
          </CardHeader>
          <CardContent>
              <div className="space-y-4">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="flex items-center gap-4 p-3 rounded-lg border border-border">
                    <div className={`w-3 h-3 rounded-full ${i % 3 === 0 ? 'bg-red-500' : 'bg-green-500'}`} />
                    <div className="flex-1">
                      <div className="font-medium">Task #{1000 + i}</div>
                      <div className="text-sm text-muted-foreground">
                        {i % 3 === 0 ? 'Failed: Timeout error' : 'Completed successfully'}
                          </div>
                        </div>
                    <div className="text-sm text-muted-foreground">
                      {i}h ago
                    </div>
                    <Badge variant={i % 3 === 0 ? 'destructive' : 'default'}>
                      {i % 3 === 0 ? 'Failed' : 'Success'}
                        </Badge>
                        </div>
                  ))}
            </div>
          </CardContent>
        </Card>
        </TabsContent>

        {/* Performance Trends */}
        <TabsContent value="trends" className="space-y-6">
        <Card className="glass-card">
          <CardHeader>
              <CardTitle>Performance Over Time</CardTitle>
          </CardHeader>
          <CardContent>
              <div className="h-64 flex items-end justify-between gap-2">
                {performanceTrend.map((point, index) => (
                  <div key={index} className="flex-1 flex flex-col items-center gap-2">
                    <div 
                      className="w-full bg-gradient-to-t from-primary to-primary/50 rounded-t"
                      style={{ height: `${(point.performance / 100) * 200}px` }}
                    />
                    <div className="text-xs text-muted-foreground">{point.time}</div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
        </TabsContent>

        {/* Activity Logs */}
        <TabsContent value="logs" className="space-y-6">
          <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
            </CardHeader>
            <CardContent>
              {logsLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="flex items-center gap-4">
                      <Skeleton className="w-4 h-4 rounded-full" />
                      <div className="flex-1">
                        <Skeleton className="h-4 w-48 mb-1" />
                        <Skeleton className="h-3 w-32" />
                      </div>
                      <Skeleton className="h-3 w-16" />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-4">
                  {(Array.isArray(agentLogs) ? agentLogs : []).slice(0, 10).map((log: any, index: number) => (
                    <div key={index} className="flex items-center gap-4 p-3 rounded-lg border border-border">
                      <div className={`w-3 h-3 rounded-full ${
                        log.level === 'error' ? 'bg-red-500' : 
                        log.level === 'warning' ? 'bg-yellow-500' : 'bg-green-500'
                      }`} />
                      <div className="flex-1">
                        <div className="font-medium">{log.message || `Log entry ${index + 1}`}</div>
                        <div className="text-sm text-muted-foreground">
                          {log.details || 'System activity logged'}
                        </div>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : `${index + 1}m ago`}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

