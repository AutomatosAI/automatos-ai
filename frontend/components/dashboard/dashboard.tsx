
'use client'

import { useState, useEffect, useRef } from 'react'
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
  MemoryStick,
  HardDrive,
  Wifi,
  Brain,
  Target,
  BarChart3
} from 'lucide-react'

// Import PRD06 widgets
import { ContextOptimizationPanel } from './widgets/context-optimization-panel'
import { LearningProgressChart } from './widgets/learning-progress-chart'
import { AgentStatusGrid } from './widgets/agent-status-grid'
import { TaskExecutionTimeline } from './widgets/task-execution-timeline'
import { ActivityHeatmap } from './widgets/activity-heatmap'

interface DashboardData {
  systemHealth?: {
    cpuUsage: number
    memoryUsage: number
    diskUsage: number
    databaseStatus: string
    redisStatus: string
    uptime: string
  }
  agentMetrics?: {
    activeAgents: number
    totalAgents: number
    successRate: number
    avgExecutionTime: number
    totalTokensUsed: number
    recentExecutions: number
  }
  workflowMetrics?: {
    totalWorkflows: number
    completedWorkflows: number
    pendingWorkflows: number
    completionRate: number
    totalExecutions: number
    successfulExecutions: number
    successRate: number
    recentWorkflows: number
  }
  contextMetrics?: {
    tokensSaved: number
    avgCompressionRatio: number
    totalOptimizations: number
    efficiency: number
  }
  learningMetrics?: {
    totalMemoryItems: number
    recentMemoryItems: number
    knowledgeNodes: number
    activeCollaborations: number
    totalCollaborations: number
    knowledgeGrowth: number
    memoryConsolidations: number
    avgImprovement: number
  }
  timestamp?: string
}

export function Dashboard() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        
        // Use mock data for now since API endpoints don't exist
        const mockData: DashboardData = {
          systemHealth: {
            cpuUsage: 45,
            memoryUsage: 67,
            diskUsage: 23,
            databaseStatus: 'healthy',
            redisStatus: 'healthy',
            uptime: '7d 12h 34m'
          },
          agentMetrics: {
            activeAgents: 12,
            totalAgents: 15,
            successRate: 94.5,
            avgExecutionTime: 2.3,
            totalTokensUsed: 125000,
            recentExecutions: 45
          },
          workflowMetrics: {
            totalWorkflows: 8,
            completedWorkflows: 6,
            pendingWorkflows: 2,
            completionRate: 75,
            totalExecutions: 156,
            successfulExecutions: 142,
            successRate: 91,
            recentWorkflows: 3
          },
          contextMetrics: {
            tokensSaved: 12500,
            avgCompressionRatio: 0.65,
            totalOptimizations: 89,
            efficiency: 87.3
          },
          learningMetrics: {
            totalMemoryItems: 234,
            recentMemoryItems: 12,
            knowledgeNodes: 45,
            activeCollaborations: 8,
            totalCollaborations: 23,
            knowledgeGrowth: 15.2,
            memoryConsolidations: 67,
            avgImprovement: 12.8
          },
          timestamp: new Date().toISOString()
        }
        
        setData(mockData)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data')
        console.error('Dashboard error:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    
    // Set up WebSocket for real-time updates
    const connectWebSocket = () => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`
        
        wsRef.current = new WebSocket(wsUrl)
        
        wsRef.current.onopen = () => {
          console.log('Dashboard WebSocket connected')
        }
        
        wsRef.current.onmessage = (event) => {
          try {
            const update = JSON.parse(event.data)
            console.log('Real-time update:', update)
            
            // Refresh data on updates
            if (update.type === 'metrics_update' || update.type === 'agent_update') {
              fetchData()
            }
          } catch (err) {
            console.error('WebSocket message error:', err)
          }
        }
        
        wsRef.current.onerror = (error) => {
          console.error('WebSocket error:', error)
        }
        
        wsRef.current.onclose = () => {
          console.log('WebSocket disconnected, reconnecting...')
          setTimeout(connectWebSocket, 5000)
        }
      } catch (err) {
        console.error('WebSocket connection error:', err)
      }
    }
    
    connectWebSocket()
    
    // Fallback polling every 30 seconds
    const interval = setInterval(fetchData, 30000)
    
    return () => {
      clearInterval(interval)
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen gradient-bg">
        <div className="max-w-7xl mx-auto p-8">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-400 mx-auto mb-4"></div>
            <h1 className="text-3xl font-bold text-white mb-2">🚀 Automatos AI Dashboard</h1>
            <p className="text-gray-300">Loading real-time analytics...</p>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen gradient-bg">
        <div className="max-w-7xl mx-auto p-8">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-white mb-4">🚀 Automatos AI Dashboard</h1>
            <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-6 max-w-md mx-auto">
              <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-2" />
              <p className="text-red-300">Error: {error}</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
        {/* Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center"
        >
          <h1 className="text-4xl font-bold text-white mb-2">
            🚀 Automatos AI Dashboard
          </h1>
          <p className="text-gray-300 text-lg">
            Real-time monitoring and analytics for your multi-agent platform
          </p>
          <div className="flex items-center justify-center gap-4 mt-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-400 text-sm font-medium">System Online</span>
            </div>
            <div className="text-gray-400 text-sm">
              Last updated: {data?.timestamp ? new Date(data.timestamp).toLocaleTimeString() : 'Unknown'}
            </div>
          </div>
        </motion.div>

        {/* Main Metrics Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {/* Active Agents */}
          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <Bot className="w-8 h-8 text-blue-400" />
              <TrendingUp className="w-5 h-5 text-green-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-200 mb-2">Active Agents</h3>
            <p className="text-3xl font-bold text-blue-400">
              {data?.agentMetrics?.activeAgents || 0}
              <span className="text-lg text-gray-400">/{data?.agentMetrics?.totalAgents || 0}</span>
            </p>
            <p className="text-sm text-gray-400 mt-1">
              {data?.agentMetrics?.successRate?.toFixed(1) || 0}% success rate
            </p>
          </div>

          {/* Workflows */}
          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <GitBranch className="w-8 h-8 text-purple-400" />
              <BarChart3 className="w-5 h-5 text-purple-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-200 mb-2">Workflows</h3>
            <p className="text-3xl font-bold text-purple-400">
              {data?.workflowMetrics?.totalWorkflows || 0}
            </p>
            <p className="text-sm text-gray-400 mt-1">
              {data?.workflowMetrics?.totalExecutions || 0} executions
            </p>
          </div>

          {/* System Health */}
          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-green-400" />
              <CheckCircle className="w-5 h-5 text-green-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-200 mb-2">System Health</h3>
            <p className="text-3xl font-bold text-green-400">
              {data?.systemHealth?.cpuUsage?.toFixed(1) || 0}%
            </p>
            <p className="text-sm text-gray-400 mt-1">
              CPU Usage
            </p>
          </div>

          {/* PRD06 Features */}
          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <Brain className="w-8 h-8 text-orange-400" />
              <Target className="w-5 h-5 text-orange-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-200 mb-2">AI Learning</h3>
            <p className="text-3xl font-bold text-orange-400">
              {data?.learningMetrics?.totalMemoryItems || 0}
            </p>
            <p className="text-sm text-gray-400 mt-1">
              Memory items
            </p>
          </div>
        </motion.div>

        {/* Detailed Analytics Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* System Resources */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="glass-card p-6"
          >
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <Cpu className="w-6 h-6 text-blue-400" />
              System Resources
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Cpu className="w-5 h-5 text-blue-400" />
                  <span className="text-gray-300">CPU Usage</span>
                </div>
                <span className="text-blue-400 font-mono">
                  {data?.systemHealth?.cpuUsage?.toFixed(1) || 0}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-blue-400 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${data?.systemHealth?.cpuUsage || 0}%` }}
                ></div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <MemoryStick className="w-5 h-5 text-green-400" />
                  <span className="text-gray-300">Memory Usage</span>
                </div>
                <span className="text-green-400 font-mono">
                  {data?.systemHealth?.memoryUsage?.toFixed(1) || 0}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-green-500 to-green-400 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${data?.systemHealth?.memoryUsage || 0}%` }}
                ></div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <HardDrive className="w-5 h-5 text-orange-400" />
                  <span className="text-gray-300">Disk Usage</span>
                </div>
                <span className="text-orange-400 font-mono">
                  {data?.systemHealth?.diskUsage?.toFixed(1) || 0}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-orange-500 to-orange-400 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${data?.systemHealth?.diskUsage || 0}%` }}
                ></div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Wifi className="w-5 h-5 text-purple-400" />
                  <span className="text-gray-300">Network</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm">Online</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* PRD06 Advanced Features */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="glass-card p-6"
          >
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <Brain className="w-6 h-6 text-purple-400" />
              PRD06 Advanced Features
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Memory Items</span>
                <span className="text-blue-400 font-mono">
                  {data?.learningMetrics?.totalMemoryItems || 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Knowledge Nodes</span>
                <span className="text-blue-400 font-mono">
                  {data?.learningMetrics?.knowledgeNodes || 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Tokens Saved</span>
                <span className="text-green-400 font-mono">
                  {data?.contextMetrics?.tokensSaved || 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Optimizations</span>
                <span className="text-purple-400 font-mono">
                  {data?.contextMetrics?.totalOptimizations || 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Active Collaborations</span>
                <span className="text-orange-400 font-mono">
                  {data?.learningMetrics?.activeCollaborations || 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Avg Improvement</span>
                <span className="text-green-400 font-mono">
                  {(data?.learningMetrics?.avgImprovement || 0).toFixed(1)}%
                </span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* PRD06 Widgets Section 1: Context Optimization & Learning Progress */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <ContextOptimizationPanel 
              contextData={data?.contextMetrics || {}}
              overview={data || {}}
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            <LearningProgressChart 
              learningData={data?.learningMetrics || {}}
              overview={data || {}}
            />
          </motion.div>
        </div>

        {/* PRD06 Widget: Agent Status Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <AgentStatusGrid 
            agentMetrics={data?.agentMetrics || {}}
            agents={data?.agents || []}
          />
        </motion.div>

        {/* PRD06 Widgets Section 2: Task Execution & Activity Heatmap */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.7 }}
          >
            <TaskExecutionTimeline 
              tasks={data?.recentTasks || []}
              workflows={data?.workflowMetrics || {}}
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            <ActivityHeatmap 
              activityData={data?.activityData || {}}
              agents={data?.agents || []}
            />
          </motion.div>
        </div>

        {/* Database & Service Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.9 }}
          className="glass-card p-6"
        >
          <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Database className="w-6 h-6 text-green-400" />
            Service Status
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Database className="w-8 h-8 text-green-400" />
              </div>
              <h4 className="text-lg font-semibold text-gray-200">Database</h4>
              <p className="text-green-400 font-medium">
                {data?.systemHealth?.databaseStatus || 'Unknown'}
              </p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Zap className="w-8 h-8 text-red-400" />
              </div>
              <h4 className="text-lg font-semibold text-gray-200">Redis Cache</h4>
              <p className="text-green-400 font-medium">
                {data?.systemHealth?.redisStatus || 'Unknown'}
              </p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Clock className="w-8 h-8 text-blue-400" />
              </div>
              <h4 className="text-lg font-semibold text-gray-200">Uptime</h4>
              <p className="text-blue-400 font-medium">
                {data?.systemHealth?.uptime || 'Unknown'}
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
