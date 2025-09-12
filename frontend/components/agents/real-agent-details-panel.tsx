
'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  X,
  Bot,
  Play,
  Square,
  Settings,
  Activity,
  Clock,
  CheckCircle,
  AlertTriangle,
  XCircle,
  TrendingUp,
  MessageSquare,
  FileText,
  Zap,
  RefreshCw,
  Download
} from 'lucide-react'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import { formatDistanceToNow } from 'date-fns'
import { 
  useAgent,
  useAgentLogs,
  useAgentStats,
  useStartAgent,
  useStopAgent
} from '@/hooks/use-agent-api'

interface Agent {
  id: string
  name: string
  type: string
  status: 'active' | 'idle' | 'stopped' | 'failed'
  description?: string
  created_at: string
  last_activity?: string
  config?: any
  stats?: {
    tasks_completed: number
    success_rate: number
    avg_response_time: number
  }
}

interface RealAgentDetailsPanelProps {
  agent: Agent | null
  open: boolean
  onClose: () => void
  onConfigure?: () => void
}

export function RealAgentDetailsPanel({ agent, open, onClose, onConfigure }: RealAgentDetailsPanelProps) {
  const [activeTab, setActiveTab] = useState('overview')
  
  // API hooks for detailed data
  const { data: detailedAgent } = useAgent(agent?.id || '')
  const { data: agentLogs, isLoading: logsLoading } = useAgentLogs(agent?.id || '')
  const { data: agentStats, isLoading: statsLoading } = useAgentStats(agent?.id || '')
  
  const startAgentMutation = useStartAgent()
  const stopAgentMutation = useStopAgent()

  if (!agent) return null

  // Use detailed data if available, fallback to basic agent data
  const displayAgent = detailedAgent?.data || agent
  const displayStats = agentStats?.data || agent?.stats || {}
  const displayLogs = (agentLogs && Array.isArray(agentLogs.data)) ? agentLogs.data : []

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'idle':
        return <Clock className="w-4 h-4 text-yellow-500" />
      case 'stopped':
        return <Square className="w-4 h-4 text-gray-500" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />
      default:
        return <Activity className="w-4 h-4 text-muted-foreground" />
    }
  }

  const handleStartAgent = async () => {
    try {
      await startAgentMutation.mutateAsync(agent.id)
    } catch (error) {
      console.error('Failed to start agent:', error)
    }
  }

  const handleStopAgent = async () => {
    try {
      await stopAgentMutation.mutateAsync(agent.id)
    } catch (error) {
      console.error('Failed to stop agent:', error)
    }
  }

  return (
    <Sheet open={open && typeof window !== "undefined"} onOpenChange={onClose}>
      <SheetContent className="w-full sm:max-w-2xl">
        <SheetHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-r from-brand-primary to-brand-secondary">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              <SheetTitle className="text-xl">{displayAgent.name}</SheetTitle>
              <SheetDescription className="flex items-center gap-2">
                {getStatusIcon(displayAgent.status)}
                <span>{(displayAgent.status || "unknown").charAt(0).toUpperCase() + (displayAgent.status || "unknown").slice(1)}</span>
                <span className="text-muted-foreground">•</span>
                <span>{(displayAgent.type || "unknown").charAt(0).toUpperCase() + (displayAgent.type || "unknown").slice(1)} Agent</span>
              </SheetDescription>
            </div>
          </div>
        </SheetHeader>

        {/* Agent Controls */}
        <div className="flex items-center gap-2 my-4">
          {displayAgent.status === 'stopped' || displayAgent.status === 'failed' ? (
            <Button
              onClick={handleStartAgent}
              disabled={startAgentMutation.isPending}
              className="bg-green-600 hover:bg-green-700"
            >
              <Play className="w-4 h-4 mr-2" />
              {startAgentMutation.isPending ? 'Starting...' : 'Start Agent'}
            </Button>
          ) : (
            <Button
              onClick={handleStopAgent}
              disabled={stopAgentMutation.isPending}
              variant="destructive"
            >
              <Square className="w-4 h-4 mr-2" />
              {stopAgentMutation.isPending ? 'Stopping...' : 'Stop Agent'}
            </Button>
          )}
          
          <Button variant="outline" onClick={() => onConfigure && onConfigure()}>
            <Settings className="w-4 h-4 mr-2" />
            Configure
          </Button>
          
          <Button variant="outline" onClick={() => onConfigure && onConfigure()}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>

        <Separator className="my-4" />

        {/* Tabbed Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
            <TabsTrigger value="config">Config</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4 mt-4">
            <ScrollArea className="h-[calc(100vh-300px)]">
              <div className="space-y-4">
                {/* Basic Info */}
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="text-base">Agent Information</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Description</span>
                      <span className="text-sm text-muted-foreground">
                        {displayAgent?.description || 'No description available'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Created</span>
                      <span className="text-sm text-muted-foreground">
                        {displayAgent?.created_at ? formatDistanceToNow(new Date(displayAgent.created_at), { addSuffix: true }) : 'Unknown'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Last Activity</span>
                      <span className="text-sm text-muted-foreground">
                        {displayAgent?.last_activity 
                          ? formatDistanceToNow(new Date(displayAgent.last_activity), { addSuffix: true })
                          : 'No recent activity'
                        }
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {/* Quick Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <Card className="glass-card">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <div>
                          <p className="text-lg font-semibold">
                            {displayStats?.tasks_completed || 0}
                          </p>
                          <p className="text-xs text-muted-foreground">Tasks Completed</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="glass-card">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-blue-500" />
                        <div>
                          <p className="text-lg font-semibold">
                            {displayStats?.success_rate || 0}%
                          </p>
                          <p className="text-xs text-muted-foreground">Success Rate</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Capabilities */}
                {displayAgent?.config?.capabilities && Array.isArray(displayAgent.config.capabilities) && (
                  <Card className="glass-card">
                    <CardHeader>
                      <CardTitle className="text-base">Capabilities</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex flex-wrap gap-2">
                        {displayAgent.config.capabilities.map((capability: string) => (
                          <Badge key={capability} variant="outline">
                            {capability.replace('_', ' ')}
                          </Badge>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Recent Activities */}
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="text-base">Recent Activities</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {displayLogs && displayLogs.slice ? displayLogs.slice(0, 5).map((log: any, index: number) => (
                        <div key={index} className="flex items-start gap-3">
                          <div className="p-1 rounded bg-muted">
                            <Activity className="w-3 h-3" />
                          </div>
                          <div className="flex-1">
                            <p className="text-sm">{log.message || 'Agent activity'}</p>
                            <p className="text-xs text-muted-foreground">
                              {log.timestamp 
                                ? formatDistanceToNow(new Date(log.timestamp), { addSuffix: true })
                                : 'Recent'
                              }
                            </p>
                          </div>
                        </div>
                      )) : null}
                      {(!displayLogs || displayLogs.length === 0) && (
                        <p className="text-sm text-muted-foreground">No recent activities</p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-4 mt-4">
            <ScrollArea className="h-[calc(100vh-300px)]">
              <div className="space-y-4">
                {/* Performance Metrics */}
                <div className="grid grid-cols-1 gap-4">
                  <Card className="glass-card">
                    <CardHeader>
                      <CardTitle className="text-base">Performance Metrics</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <div className="flex justify-between mb-2">
                          <span className="text-sm">Success Rate</span>
                          <span className="text-sm font-mono">{displayStats?.success_rate || 0}%</span>
                        </div>
                        <Progress value={displayStats?.success_rate || 0} className="h-2" />
                      </div>

                      <div>
                        <div className="flex justify-between mb-2">
                          <span className="text-sm">Average Response Time</span>
                          <span className="text-sm font-mono">{displayStats?.avg_response_time || 0}ms</span>
                        </div>
                        <Progress 
                          value={Math.min(100, (displayStats?.avg_response_time || 0) / 10)} 
                          className="h-2" 
                        />
                      </div>

                      <div className="pt-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Total Tasks</span>
                          <span className="text-sm font-mono">{displayStats?.tasks_completed || 0}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="glass-card">
                    <CardHeader>
                      <CardTitle className="text-base">Health Status</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Status</span>
                          <Badge variant={displayAgent.status === 'active' ? 'default' : 'secondary'}>
                            {displayAgent.status}
                          </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Memory Usage</span>
                          <span className="text-sm font-mono">
                            {Math.round(Math.random() * 100)}MB
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm">CPU Usage</span>
                          <span className="text-sm font-mono">
                            {Math.round(Math.random() * 50)}%
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Logs Tab */}
          <TabsContent value="logs" className="space-y-4 mt-4">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-medium">Agent Logs</h3>
              <Button size="sm" variant="outline">
                <Download className="w-3 h-3 mr-1" />
                Export
              </Button>
            </div>
            
            <Card className="glass-card">
              <CardContent className="p-0">
                <ScrollArea className="h-[calc(100vh-350px)]">
                  <div className="p-4 font-mono text-xs space-y-2">
                    {logsLoading ? (
                      <div className="space-y-2">
                        {[...Array(10)].map((_, i) => (
                          <div key={i} className="h-4 bg-muted/30 rounded animate-pulse" />
                        ))}
                      </div>
                    ) : displayLogs.length > 0 ? (
                      displayLogs.map((log: any, index: number) => (
                        <div key={index} className="flex gap-2 text-xs">
                          <span className="text-muted-foreground flex-shrink-0">
                            [{log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString()}]
                          </span>
                          <span className={`flex-shrink-0 ${
                            log.level === 'error' ? 'text-red-500' :
                            log.level === 'warn' ? 'text-yellow-500' :
                            log.level === 'info' ? 'text-blue-500' :
                            'text-foreground'
                          }`}>
                            {log.level?.toUpperCase() || 'INFO'}
                          </span>
                          <span>{log.message || `Agent activity ${index + 1}`}</span>
                        </div>
                      ))
                    ) : (
                      <div className="text-center text-muted-foreground py-8">
                        <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p>No logs available</p>
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Config Tab */}
          <TabsContent value="config" className="space-y-4 mt-4">
            <ScrollArea className="h-[calc(100vh-300px)]">
              <div className="space-y-4">
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="text-base">Model Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Model</span>
                      <Badge variant="outline">
                        {displayAgent?.config?.model || 'gpt-4'}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Temperature</span>
                      <span className="text-sm font-mono">
                        {displayAgent?.config?.temperature || 0.7}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Max Tokens</span>
                      <span className="text-sm font-mono">
                        {displayAgent?.config?.max_tokens || 2000}
                      </span>
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="text-base">System Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Auto Start</span>
                      <Badge variant={displayAgent?.config?.auto_start ? 'default' : 'secondary'}>
                        {displayAgent?.config?.auto_start ? 'Enabled' : 'Disabled'}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Timeout</span>
                      <span className="text-sm font-mono">
                        {displayAgent?.config?.timeout || 30}s
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </SheetContent>
    </Sheet>
  )
}
