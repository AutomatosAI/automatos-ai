
'use client'

import { useState, useEffect } from 'react'
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

export function EnhancedDashboard() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch('/api/analytics/dashboard/overview')
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        
        const result = await response.json()
        setData(result)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data')
        console.error('Dashboard error:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
            <h1 className="text-3xl font-bold text-white mb-2">🚀 Automatos AI Dashboard</h1>
            <p className="text-gray-300">Loading real-time analytics...</p>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-red-900 to-purple-900 p-8">
        <div className="max-w-7xl mx-auto">
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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
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
          <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 backdrop-blur-sm border border-blue-500/30 rounded-xl p-6 hover:border-blue-400/50 transition-all duration-300">
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
          <div className="bg-gradient-to-br from-purple-600/20 to-purple-800/20 backdrop-blur-sm border border-purple-500/30 rounded-xl p-6 hover:border-purple-400/50 transition-all duration-300">
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
          <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 backdrop-blur-sm border border-green-500/30 rounded-xl p-6 hover:border-green-400/50 transition-all duration-300">
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
          <div className="bg-gradient-to-br from-orange-600/20 to-orange-800/20 backdrop-blur-sm border border-orange-500/30 rounded-xl p-6 hover:border-orange-400/50 transition-all duration-300">
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
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* System Resources */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="bg-gradient-to-br from-gray-800/40 to-gray-900/40 backdrop-blur-sm border border-gray-600/30 rounded-xl p-6"
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
            className="bg-gradient-to-br from-gray-800/40 to-gray-900/40 backdrop-blur-sm border border-gray-600/30 rounded-xl p-6"
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

        {/* Database & Service Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="bg-gradient-to-br from-gray-800/40 to-gray-900/40 backdrop-blur-sm border border-gray-600/30 rounded-xl p-6"
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
