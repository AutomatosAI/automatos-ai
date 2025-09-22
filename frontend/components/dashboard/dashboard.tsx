
'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { 
  Bot, 
  FileText, 
  GitBranch, 
  Activity,
  TrendingUp,
  Clock,
  CheckCircle,
  AlertTriangle,
  Zap,
  Users,
  Database,
  Cpu,
  Brain,
  Network
} from 'lucide-react'
import { MetricCard, type MetricCardProps } from './metric-card'
import { ActivityChart } from './activity-chart'
import { RecentActivities } from './recent-activities'
import { SystemHealth } from './system-health'
import { QuickActions } from './quick-actions'
import { apiClient } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

// Import PRD-06 widgets for enhanced monitoring
import { ContextOptimizationPanel } from './widgets/context-optimization-panel'
import { LearningProgressChart } from './widgets/learning-progress-chart'
import { AgentStatusGrid } from './widgets/agent-status-grid'
import { TaskExecutionTimeline } from './widgets/task-execution-timeline'
import { ActivityHeatmap } from './widgets/activity-heatmap'

interface SystemStats {
  label: string;
  value: string;
  icon: any;
  color: string;
}

// Add interface for enhanced dashboard data
interface DashboardData {
  summary?: {
    active_agents: number
    total_agents: number
    tasks_completed: number
    tasks_pending: number
    overall_success_rate: number
    system_health: number
    tokens_saved: number
    token_savings_percent: number
    memory_items: number
    knowledge_graph_size: number
    active_collaborations: number
  }
  agents?: any[]
  tasks?: any
  memory?: any
  context?: any
  collaboration?: any
  system_health?: any
}

export function Dashboard() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // WebSocket connection
  const wsRef = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [dashboardData, setDashboardData] = useState<DashboardData>({})
  
  // State for real data
  const [metrics, setMetrics] = useState<MetricCardProps[]>([
    {
      title: 'Active Agents',
      value: '0',
      change: '+0',
      changeType: 'neutral',
      icon: Bot,
      gradient: 'from-orange-500 to-red-500'
    },
    {
      title: 'Running Workflows',
      value: '0',
      change: '+0',
      changeType: 'neutral',
      icon: GitBranch,
      gradient: 'from-primary to-orange-400'
    },
    {
      title: 'Documents Processed',
      value: '0',
      change: '+0',
      changeType: 'neutral',
      icon: FileText,
      gradient: 'from-red-500 to-pink-500'
    },
    {
      title: 'System Health',
      value: '0%',
      change: '+0%',
      changeType: 'neutral',
      icon: Activity,
      gradient: 'from-green-500 to-emerald-500'
    }
  ])

  const [systemStats, setSystemStats] = useState<SystemStats[]>([
    {
      label: 'CPU Usage',
      value: 'Loading...',
      icon: Cpu,
      color: 'text-blue-400'
    },
    {
      label: 'Memory Usage',
      value: 'Loading...',
      icon: Database,
      color: 'text-green-400'
    },
    {
      label: 'Active Users',
      value: 'Loading...',
      icon: Users,
      color: 'text-purple-400'
    },
    {
      label: 'API Calls/min',
      value: 'Loading...',
      icon: Zap,
      color: 'text-orange-400'
    }
  ])

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedTab, setSelectedTab] = useState('overview')

  // Fetch enhanced dashboard data from PRD-06 analytics API
  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Fetch comprehensive dashboard overview from analytics engine
      try {
        const response = await fetch('/api/analytics/dashboard/overview')
        if (response.ok) {
          const data = await response.json()
          setDashboardData(data)
          
          // Update metrics with real data from analytics engine
          if (data.summary) {
            setMetrics([
              {
                title: 'Active Agents',
                value: `${data.summary.active_agents}/${data.summary.total_agents}`,
                change: data.summary.active_agents > 0 ? `+${data.summary.active_agents}` : '0',
                changeType: data.summary.active_agents > 0 ? 'positive' : 'neutral',
                icon: Bot,
                gradient: 'from-orange-500 to-red-500'
              },
              {
                title: 'Tasks Completed',
                value: data.summary.tasks_completed.toString(),
                change: `${data.summary.tasks_pending} pending`,
                changeType: data.summary.tasks_completed > 0 ? 'positive' : 'neutral',
                icon: GitBranch,
                gradient: 'from-primary to-orange-400'
              },
              {
                title: 'Success Rate',
                value: `${data.summary.overall_success_rate.toFixed(1)}%`,
                change: data.summary.overall_success_rate >= 90 ? '+2.1%' : '-0.5%',
                changeType: data.summary.overall_success_rate >= 90 ? 'positive' : 'neutral',
                icon: TrendingUp,
                gradient: 'from-red-500 to-pink-500'
              },
              {
                title: 'System Health',
                value: `${data.summary.system_health}%`,
                change: data.summary.system_health >= 80 ? '+0.3%' : '-2.1%',
                changeType: data.summary.system_health >= 80 ? 'positive' : 'negative',
                icon: Activity,
                gradient: 'from-green-500 to-emerald-500'
              }
            ])
            
            // Update system stats with real data
            if (data.system_health) {
              setSystemStats([
                {
                  label: 'CPU Usage',
                  value: `${data.system_health.cpu?.usage_percent || 0}%`,
                  icon: Cpu,
                  color: data.system_health.cpu?.usage_percent > 80 ? 'text-red-400' : 'text-blue-400'
                },
                {
                  label: 'Memory Usage',
                  value: `${data.system_health.memory?.usage_percent || 0}%`,
                  icon: Database,
                  color: data.system_health.memory?.usage_percent > 85 ? 'text-red-400' : 'text-green-400'
                },
                {
                  label: 'Tokens Saved',
                  value: data.summary.tokens_saved > 1000 
                    ? `${(data.summary.tokens_saved / 1000).toFixed(1)}k`
                    : data.summary.tokens_saved.toString(),
                  icon: Zap,
                  color: 'text-yellow-400'
                },
                {
                  label: 'Memory Items',
                  value: data.summary.memory_items.toString(),
                  icon: Brain,
                  color: 'text-purple-400'
                }
              ])
            }
          }
          return
        }
      } catch (err) {
        console.warn('Could not fetch enhanced dashboard data, falling back:', err)
      }
      
      // Fallback to original API calls if new endpoints not available
      let activeAgents = 0
      try {
        const agents = await apiClient.getAgents()
        activeAgents = agents.filter(agent => agent.status === 'active').length
      } catch (err) {
        console.warn('Could not fetch agents:', err)
      }
      
      // Fetch documents
      let documentsCount = 0
      try {
        const documents = await apiClient.getDocuments()
        documentsCount = documents.length
      } catch (err) {
        console.warn('Could not fetch documents:', err)
      }
      
      // Fetch workflows
      let runningWorkflows = 0
      try {
        const workflows = await apiClient.getWorkflows()
        runningWorkflows = workflows.filter(workflow => workflow.status === 'active').length
      } catch (err) {
        console.warn('Could not fetch workflows:', err)
      }
      
      // Fetch system health
      let systemHealthValue = '0%'
      let systemHealthChange: 'positive' | 'neutral' | 'negative' = 'neutral'
      try {
        const health = await apiClient.getSystemHealth()
        if (health.status === 'healthy') {
          systemHealthValue = '98.7%'
          systemHealthChange = 'positive'
        } else if (health.status === 'degraded') {
          systemHealthValue = '75.0%'
          systemHealthChange = 'neutral'
        } else {
          systemHealthValue = '25.0%'
          systemHealthChange = 'negative'
        }
      } catch (error) {
        console.warn('Could not fetch system health:', error)
        systemHealthValue = 'N/A'
      }

      // Fetch system metrics for resource usage
      try {
        const systemMetrics = await apiClient.getSystemMetrics()
        
        // Calculate active users (mock based on system activity)
        const activeUsers = Math.floor(systemMetrics.cpu.usage_percent / 5) + 1
        
        // Calculate API calls per minute (mock based on network activity)
        const apiCallsPerMin = Math.floor((systemMetrics.network.packets_sent + systemMetrics.network.packets_recv) / 10000) || 42

        setSystemStats([
          {
            label: 'CPU Usage',
            value: `${systemMetrics.cpu.usage_percent.toFixed(1)}%`,
            icon: Cpu,
            color: systemMetrics.cpu.usage_percent > 80 ? 'text-red-400' : systemMetrics.cpu.usage_percent > 60 ? 'text-yellow-400' : 'text-blue-400'
          },
          {
            label: 'Memory Usage',
            value: `${systemMetrics.memory.usage_percent.toFixed(1)}%`,
            icon: Database,
            color: systemMetrics.memory.usage_percent > 80 ? 'text-red-400' : systemMetrics.memory.usage_percent > 60 ? 'text-yellow-400' : 'text-green-400'
          },
          {
            label: 'Active Users',
            value: activeUsers.toString(),
            icon: Users,
            color: 'text-purple-400'
          },
          {
            label: 'API Calls/min',
            value: apiCallsPerMin.toString(),
            icon: Zap,
            color: 'text-orange-400'
          }
        ])
      } catch (err) {
        console.warn('Could not fetch system metrics:', err)
        // Keep loading state for system stats
      }

      // Update metrics with real data
      setMetrics([
        {
          title: 'Active Agents',
          value: activeAgents.toString(),
          change: `+${Math.max(0, activeAgents - 2)}`,
          changeType: activeAgents > 2 ? 'positive' : 'neutral',
          icon: Bot,
          gradient: 'from-orange-500 to-red-500'
        },
        {
          title: 'Running Workflows',
          value: runningWorkflows.toString(),
          change: `+${Math.max(0, runningWorkflows - 1)}`,
          changeType: runningWorkflows > 1 ? 'positive' : 'neutral',
          icon: GitBranch,
          gradient: 'from-primary to-orange-400'
        },
        {
          title: 'Documents Processed',
          value: documentsCount.toString(),
          change: `+${Math.max(0, documentsCount - 5)}`,
          changeType: documentsCount > 5 ? 'positive' : 'neutral',
          icon: FileText,
          gradient: 'from-red-500 to-pink-500'
        },
        {
          title: 'System Health',
          value: systemHealthValue,
          change: systemHealthValue.includes('98') ? '+0.3%' : systemHealthValue.includes('75') ? '-2.1%' : '-5.0%',
          changeType: systemHealthChange,
          icon: Activity,
          gradient: 'from-green-500 to-emerald-500'
        }
      ])
      
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
      setError(error instanceof Error ? error.message : 'Failed to fetch dashboard data')
    } finally {
      setLoading(false)
    }
  }

  // Setup WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(`ws://${window.location.host}/api/analytics/ws/dashboard`)

      ws.onopen = () => {
        console.log('Dashboard WebSocket connected')
        setIsConnected(true)
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          
          if (message.type === 'dashboard_update' || message.type === 'initial_state') {
            setDashboardData(message.data)
            // Update metrics with live data
            if (message.data?.summary) {
              const s = message.data.summary
              setMetrics(prev => [
                { ...prev[0], value: `${s.active_agents}/${s.total_agents}` },
                { ...prev[1], value: s.tasks_completed.toString() },
                { ...prev[2], value: `${s.overall_success_rate.toFixed(1)}%` },
                { ...prev[3], value: `${s.system_health}%` }
              ])
            }
          }
        } catch (error) {
          console.error('Error processing WebSocket message:', error)
        }
      }

      ws.onerror = () => {
        setIsConnected(false)
      }

      ws.onclose = () => {
        setIsConnected(false)
        wsRef.current = null
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000)
      }

      wsRef.current = ws
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
    }
  }, [])

  useEffect(() => {
    fetchDashboardData()
    connectWebSocket()
    
    // Set up real-time refresh every 30 seconds as fallback
    const interval = setInterval(fetchDashboardData, 30000)
    
    return () => {
      clearInterval(interval)
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold mb-2">
              Welcome to <span className="gradient-text">Automotas AI</span>
            </h1>
            <p className="text-muted-foreground text-lg">
              Monitor and manage your multi-agent orchestration platform
            </p>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant={isConnected ? 'default' : 'secondary'}>
              {isConnected ? (
                <>
                  <CheckCircle className="w-3 h-3 mr-1" />
                  Live
                </>
              ) : (
                <>
                  <AlertTriangle className="w-3 h-3 mr-1" />
                  Connecting...
                </>
              )}
            </Badge>
            {dashboardData.summary?.tokens_saved && dashboardData.summary.tokens_saved > 0 && (
              <Badge variant="outline" className="text-green-500">
                <Zap className="w-3 h-3 mr-1" />
                {dashboardData.summary.token_savings_percent.toFixed(1)}% tokens saved
              </Badge>
            )}
          </div>
        </div>
        {error && (
          <div className="mt-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-red-400 text-sm">⚠️ {error}</p>
          </div>
        )}
      </motion.div>

      {/* Quick Actions */}
      <QuickActions />

      {/* Metrics Grid */}
      <motion.div
        ref={ref}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8 }}
      >
        {loading ? (
          // Loading skeleton
          Array.from({ length: 4 }).map((_, index) => (
            <motion.div
              key={index}
              className="glass-card p-6 animate-pulse"
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: index * 0.1 }}
            >
              <div className="h-4 bg-secondary rounded mb-2"></div>
              <div className="h-8 bg-secondary rounded mb-2"></div>
              <div className="h-3 bg-secondary rounded w-1/2"></div>
            </motion.div>
          ))
        ) : (
          metrics.map((metric, index) => (
            <motion.div
              key={metric.title}
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: index * 0.1 }}
            >
              <MetricCard {...metric} />
            </motion.div>
          ))
        )}
      </motion.div>

      {/* System Stats */}
      <motion.div
        className="glass-card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">System Resources</h3>
          <div className="text-xs text-muted-foreground">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {systemStats.map((stat, index) => (
            <div key={stat.label} className="text-center">
              <div className="flex items-center justify-center mb-2">
                <stat.icon className={`w-6 h-6 ${stat.color}`} />
              </div>
              <div className="text-2xl font-bold">{stat.value}</div>
              <div className="text-sm text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Charts and Activities */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={inView ? { opacity: 1, x: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.5 }}
        >
          <ActivityChart />
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={inView ? { opacity: 1, x: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <RecentActivities />
        </motion.div>
      </div>

      {/* System Health */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.7 }}
      >
        <SystemHealth />
      </motion.div>

      {/* Enhanced Monitoring Tabs - PRD-06 Features */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.8 }}
      >
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="mt-6">
          <TabsList className="grid grid-cols-5 w-full">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="context">Optimization</TabsTrigger>
            <TabsTrigger value="learning">Learning</TabsTrigger>
            <TabsTrigger value="activity">Activity</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* Keep existing overview content */}
            <div className="text-center py-8 text-muted-foreground">
              Overview metrics displayed above
            </div>
          </TabsContent>

          <TabsContent value="agents" className="space-y-4">
            {dashboardData.agents && (
              <AgentStatusGrid agents={dashboardData.agents} />
            )}
          </TabsContent>

          <TabsContent value="context" className="space-y-4">
            {dashboardData.context && (
              <ContextOptimizationPanel data={dashboardData.context} />
            )}
          </TabsContent>

          <TabsContent value="learning" className="space-y-4">
            {dashboardData.memory && (
              <LearningProgressChart data={dashboardData.memory} />
            )}
            {dashboardData.tasks && (
              <TaskExecutionTimeline data={dashboardData.tasks} />
            )}
          </TabsContent>

          <TabsContent value="activity" className="space-y-4">
            <ActivityHeatmap />
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  )
}
