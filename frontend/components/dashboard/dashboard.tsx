'use client'

import React from "react"
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

// Import API hooks
import { useSystemHealth, useSystemMetrics, useAgents, useDocuments, useWorkflows } from '@/hooks/use-api'

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

  // Use real API hooks
  const { data: systemHealth, isLoading: healthLoading, error: healthError } = useSystemHealth()
  const { data: systemMetrics, isLoading: metricsLoading, error: metricsError } = useSystemMetrics()
  const { data: agents, isLoading: agentsLoading, error: agentsError } = useAgents()
  const { data: documents, isLoading: documentsLoading, error: documentsError } = useDocuments()
  const { data: workflows, isLoading: workflowsLoading, error: workflowsError } = useWorkflows()

  // Calculate loading and error states
  const loading = healthLoading || metricsLoading || agentsLoading || documentsLoading || workflowsLoading
  const error = healthError || metricsError || agentsError || documentsError || workflowsError

  // Process real data into dashboard format
  const data: DashboardData | null = React.useMemo(() => {
    if (!systemHealth && !systemMetrics && !agents && !documents && !workflows) {
      // Return default empty data structure instead of null
      return {
        systemHealth: {
          cpuUsage: 0,
          memoryUsage: 0,
          diskUsage: 0,
          databaseStatus: 'unknown',
          redisStatus: 'unknown',
          uptime: 'unknown'
        },
        agentMetrics: {
          activeAgents: 0,
          totalAgents: 0,
          successRate: 0,
          avgExecutionTime: 0,
          totalTokensUsed: 0,
          recentExecutions: 0
        },
        workflowMetrics: {
          totalWorkflows: 0,
          completedWorkflows: 0,
          pendingWorkflows: 0,
          completionRate: 0,
          successRate: 0,
          recentWorkflows: 0
        },
        contextMetrics: {
          tokensSaved: 0,
          avgCompressionRatio: 1.0,
          totalOptimizations: 0,
          efficiency: 0
        },
        learningMetrics: {
          totalMemoryItems: 0,
          recentMemoryItems: 0,
          knowledgeNodes: 0,
          activeCollaborations: 0,
          totalCollaborations: 0,
          knowledgeGrowth: 0,
          memoryConsolidations: 0,
          avgImprovement: 0
        },
        timestamp: new Date().toISOString()
      }
    }

    // Calculate agent metrics from real data - safely handle missing/null data
    const agentsArray = Array.isArray(agents) ? agents : []
    const activeAgents = agentsArray.filter((agent: any) => agent?.status === 'active').length
    const totalAgents = agentsArray.length
    const successRate = agentsArray.length > 0 
      ? agentsArray.reduce((sum: number, agent: any) => sum + (agent?.success_rate || 0), 0) / agentsArray.length
      : 0
    const avgExecutionTime = agentsArray.length > 0
      ? agentsArray.reduce((sum: number, agent: any) => sum + (agent?.avg_execution_time || 0), 0) / agentsArray.length
      : 0
    const totalTokensUsed = agentsArray.reduce((sum: number, agent: any) => sum + (agent?.total_tokens_used || 0), 0)
    const recentExecutions = agentsArray.reduce((sum: number, agent: any) => sum + (agent?.execution_count || 0), 0)

    // Calculate workflow metrics from real data - safely handle missing/null data
    const workflowsArray = Array.isArray(workflows) ? workflows : []
    const completedWorkflows = workflowsArray.filter((workflow: any) => workflow?.status === 'completed').length
    const pendingWorkflows = workflowsArray.filter((workflow: any) => workflow?.status === 'pending').length
    const totalWorkflows = workflowsArray.length
    const completionRate = totalWorkflows > 0 ? (completedWorkflows / totalWorkflows) * 100 : 0
    const workflowSuccessRate = totalWorkflows > 0
      ? workflowsArray.reduce((sum: number, workflow: any) => sum + (workflow?.success_rate || 0), 0) / totalWorkflows
      : 0

    return {
      systemHealth: {
        cpuUsage: systemMetrics?.cpu?.average_usage || 0,
        memoryUsage: systemMetrics?.memory?.percent || 0,
        diskUsage: systemMetrics?.disk?.percent || 0,
        databaseStatus: systemHealth?.services?.database || 'unknown',
        redisStatus: systemHealth?.services?.redis || 'unknown',
        uptime: systemHealth?.uptime || 'unknown'
      },
      agentMetrics: {
        activeAgents,
        totalAgents,
        successRate,
        avgExecutionTime,
        totalTokensUsed,
        recentExecutions
      },
      workflowMetrics: {
        totalWorkflows,
        completedWorkflows,
        pendingWorkflows,
        completionRate,
        successRate: workflowSuccessRate,
        recentWorkflows: pendingWorkflows
      },
      contextMetrics: {
        tokensSaved: systemMetrics?.context_optimization?.tokens_saved || 0,
        avgCompressionRatio: systemMetrics?.context_optimization?.compression_ratio || 1.0,
        totalOptimizations: systemMetrics?.context_optimization?.total_optimizations || 0,
        efficiency: systemMetrics?.context_optimization?.efficiency || 0
      },
      learningMetrics: {
        totalMemoryItems: systemMetrics?.learning?.total_memories || 0,
        recentMemoryItems: systemMetrics?.learning?.recent_memories || 0,
        knowledgeNodes: systemMetrics?.learning?.knowledge_nodes || 0,
        activeCollaborations: systemMetrics?.learning?.active_collaborations || 0,
        totalCollaborations: systemMetrics?.learning?.total_collaborations || 0,
        knowledgeGrowth: systemMetrics?.learning?.knowledge_growth || 0,
        memoryConsolidations: systemMetrics?.learning?.memory_consolidations || 0,
        avgImprovement: systemMetrics?.learning?.avg_improvement || 0
      },
      timestamp: new Date().toISOString()
    }
  }, [systemHealth, systemMetrics, agents, documents, workflows])

  if (loading) {
    return (
      <div className="min-h-screen gradient-bg">
        <div className="max-w-7xl mx-auto p-8">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <h1 className="text-3xl font-bold text-foreground mb-2">🚀 Automatos AI Dashboard</h1>
            <p className="text-muted-foreground">Loading real-time analytics...</p>
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
            <h1 className="text-3xl font-bold text-foreground mb-4">🚀 Automatos AI Dashboard</h1>
            <div className="bg-destructive/20 border border-destructive/50 rounded-lg p-6 max-w-md mx-auto">
              <AlertTriangle className="w-8 h-8 text-destructive mx-auto mb-2" />
              <p className="text-destructive-foreground">Error: {error}</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen gradient-bg">
      <div className="max-w-7xl mx-auto p-8 space-y-8">
        {/* Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center"
        >
          <h1 className="text-4xl font-bold text-foreground mb-2">
            🚀 Automatos AI Dashboard
          </h1>
          <p className="text-muted-foreground text-lg">
            Real-time monitoring and analytics for your multi-agent platform
          </p>
          <div className="flex items-center justify-center gap-4 mt-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-400 text-sm font-medium">System Online</span>
            </div>
            <div className="text-muted-foreground text-sm">
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
            <h3 className="text-lg font-semibold text-card-foreground mb-2">Active Agents</h3>
            <p className="text-3xl font-bold text-blue-400">
              {data?.agentMetrics?.activeAgents || 0}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              of {data?.agentMetrics?.totalAgents || 0} total
            </p>
          </div>

          {/* Success Rate */}
          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <CheckCircle className="w-8 h-8 text-green-400" />
              <TrendingUp className="w-5 h-5 text-green-400" />
            </div>
            <h3 className="text-lg font-semibold text-card-foreground mb-2">Success Rate</h3>
            <p className="text-3xl font-bold text-green-400">
              {data?.agentMetrics?.successRate?.toFixed(1) || 0}%
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              {data?.agentMetrics?.recentExecutions || 0} recent executions
            </p>
          </div>

          {/* Token Usage */}
          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <FileText className="w-8 h-8 text-purple-400" />
              <Activity className="w-5 h-5 text-purple-400" />
            </div>
            <h3 className="text-lg font-semibold text-card-foreground mb-2">Token Usage</h3>
            <p className="text-3xl font-bold text-purple-400">
              {data?.agentMetrics?.totalTokensUsed ? 
                (data.agentMetrics.totalTokensUsed > 1000000 ? 
                  `${(data.agentMetrics.totalTokensUsed / 1000000).toFixed(1)}M` :
                  data.agentMetrics.totalTokensUsed > 1000 ?
                  `${(data.agentMetrics.totalTokensUsed / 1000).toFixed(1)}k` :
                  data.agentMetrics.totalTokensUsed
                ) : 0
              }
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Avg: {data?.agentMetrics?.avgExecutionTime?.toFixed(1) || 0}s per task
            </p>
          </div>

          {/* Workflows */}
          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <GitBranch className="w-8 h-8 text-primary" />
              <Clock className="w-5 h-5 text-primary" />
            </div>
            <h3 className="text-lg font-semibold text-card-foreground mb-2">Workflows</h3>
            <p className="text-3xl font-bold text-primary">
              {data?.workflowMetrics?.completedWorkflows || 0}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              {data?.workflowMetrics?.pendingWorkflows || 0} pending
            </p>
          </div>
        </motion.div>

        {/* Advanced Widgets Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-8">
            {/* Context Optimization Panel */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="glass-card p-6"
            >
              <h2 className="text-2xl font-bold text-card-foreground mb-6 flex items-center gap-2">
                <Zap className="w-6 h-6 text-primary" />
                Context Optimization
              </h2>
              <ContextOptimizationPanel 
                contextData={data?.contextMetrics || {}}
                overview={systemMetrics || {}}
              />
            </motion.div>

            {/* Agent Status Grid */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="glass-card p-6"
            >
              <h2 className="text-2xl font-bold text-card-foreground mb-6 flex items-center gap-2">
                <Users className="w-6 h-6 text-blue-400" />
                Agent Status
              </h2>
              <AgentStatusGrid agentMetrics={data?.agentMetrics || {}} agents={agents || []} />
            </motion.div>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            {/* Learning Progress Chart */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="glass-card p-6"
            >
              <h2 className="text-2xl font-bold text-card-foreground mb-6 flex items-center gap-2">
                <Brain className="w-6 h-6 text-green-400" />
                Learning Progress
              </h2>
              <LearningProgressChart learningData={data?.learningMetrics || {}} overview={systemMetrics || {}} />
            </motion.div>

            {/* Task Execution Timeline */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="glass-card p-6"
            >
              <h2 className="text-2xl font-bold text-card-foreground mb-6 flex items-center gap-2">
                <Target className="w-6 h-6 text-purple-400" />
                Task Timeline
              </h2>
              <TaskExecutionTimeline tasks={[]} workflows={data?.workflowMetrics || {}} />
            </motion.div>
          </div>
        </div>

        {/* Activity Heatmap - Full Width */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="glass-card p-6"
        >
          <h2 className="text-2xl font-bold text-card-foreground mb-6 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-primary" />
            Activity Heatmap
          </h2>
          <ActivityHeatmap activityData={systemMetrics || {}} agents={agents || []} />
        </motion.div>

        {/* System Health */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="glass-card p-6"
        >
          <h2 className="text-2xl font-bold text-card-foreground mb-6 flex items-center gap-2">
            <Activity className="w-6 h-6 text-green-400" />
            System Health
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6">
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Cpu className="w-8 h-8 text-blue-400" />
              </div>
              <h4 className="text-lg font-semibold text-card-foreground">CPU Usage</h4>
              <p className="text-blue-400 font-medium">
                {data?.systemHealth?.cpuUsage || 0}%
              </p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <MemoryStick className="w-8 h-8 text-green-400" />
              </div>
              <h4 className="text-lg font-semibold text-card-foreground">Memory</h4>
              <p className="text-green-400 font-medium">
                {data?.systemHealth?.memoryUsage || 0}%
              </p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <HardDrive className="w-8 h-8 text-yellow-400" />
              </div>
              <h4 className="text-lg font-semibold text-card-foreground">Disk Usage</h4>
              <p className="text-yellow-400 font-medium">
                {data?.systemHealth?.diskUsage || 0}%
              </p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Database className="w-8 h-8 text-purple-400" />
              </div>
              <h4 className="text-lg font-semibold text-card-foreground">Database</h4>
              <p className="text-green-400 font-medium">
                {data?.systemHealth?.databaseStatus || 'Unknown'}
              </p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Clock className="w-8 h-8 text-blue-400" />
              </div>
              <h4 className="text-lg font-semibold text-card-foreground">Uptime</h4>
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