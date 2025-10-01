
'use client'

import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { logger } from '../../lib/logger'
import { useQuery } from '@tanstack/react-query'
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
  Wifi
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Badge } from '../ui/badge'
import { Progress } from '../ui/progress'
import { Separator } from '../ui/separator'
import { 
  useSystemHealth,
  useSystemMetrics,
  useSystemAgentStatistics,
  useSystemActivities
} from '../../hooks/use-system-config-api'

import { 
  useAgents,
  useDocuments,
  useWorkflows
} from '../../hooks/use-api'

import {
  useAllMetrics
} from '../../hooks/use-analytics-api'

import {
  usePerformanceAnalytics
} from '../../hooks/use-performance-api'

// Client-side only component to avoid hydration issues with time
const TimeDisplay = dynamic(() => Promise.resolve(() => {
  return <span>{new Date().toLocaleTimeString()}</span>;
}), { ssr: false });

import { MetricCards } from './metric-cards'
import { ActivityFeed } from './activity-feed'
import { SystemHealth } from './system-health'
import { QuickActions } from './quick-actions'
import { PerformanceChart } from './performance-chart'


export function Dashboard() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Fetch real data from APIs
  const { data: systemHealth, isLoading: healthLoading } = useSystemHealth()
  const { data: systemMetrics, isLoading: metricsLoading } = useSystemMetrics()
  const { data: activities, isLoading: activitiesLoading } = useSystemActivities()
  const { data: agents, isLoading: agentsLoading } = useAgents()
  const { data: documents, isLoading: documentsLoading } = useDocuments()
  const { data: workflows, isLoading: workflowsLoading } = useWorkflows()
  
  // Additional metrics and stats
  const { data: agentStats, isLoading: agentStatsLoading } = useSystemAgentStatistics()
  const { data: allMetrics, isLoading: allMetricsLoading } = useAllMetrics()
  const { data: performanceData, isLoading: performanceLoading } = usePerformanceAnalytics('24h') // Last 24 hours

  // Loading states
  const isLoading = healthLoading || metricsLoading || activitiesLoading || 
                  agentStatsLoading || allMetricsLoading || performanceLoading

  return (
    <div ref={ref} className="space-y-6 p-4 container mx-auto max-w-[1300px]">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold gradient-text">Dashboard</h1>
            <p className="text-muted-foreground mt-1">
              Monitor your multi-agent orchestration platform in real-time
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-brand-primary border-brand-primary/30">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
              System Online
            </Badge>
            <Badge variant="secondary">
              Last Updated: <TimeDisplay />
            </Badge>
          </div>
        </div>
      </motion.div>

      {/* System Status Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
            <MetricCards />
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Charts and Analytics */}
        <div className="lg:col-span-2 space-y-6">
          {/* Performance Chart */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <PerformanceChart />
          </motion.div>

          {/* System Health Overview */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <SystemHealth />
          </motion.div>

          {/* Agent Status Grid */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bot className="w-5 h-5 text-brand-primary" />
                  Agent Status Overview
                </CardTitle>
              </CardHeader>
              <CardContent>
                {agentsLoading ? (
                  <div className="space-y-3">
                    {[...Array(3)].map((_, i) => (
                      <div key={i} className="h-4 bg-muted rounded animate-pulse" />
                    ))}
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Active Agents</span>
                      <span className="text-sm font-mono text-brand-primary">
                        {(agents || []).filter((a: any) => a.status === 'active').length || 0}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Idle Agents</span>
                      <span className="text-sm font-mono text-yellow-500">
                        {(agents || []).filter((a: any) => a.status === 'idle').length || 0}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Failed Agents</span>
                      <span className="text-sm font-mono text-red-500">
                        {(agents || []).filter((a: any) => a.status === 'failed').length || 0}
                      </span>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span>System Load</span>
                        <span>{agentStats?.utilization ? Math.round(agentStats.utilization * 100) : 0}%</span>
                      </div>
                      <Progress 
                        value={agentStats?.utilization ? Math.round(agentStats.utilization * 100) : 0} 
                        className="h-2"
                      />
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Right Column - Activities and Quick Actions */}
        <div className="space-y-6">
          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="w-full"
          >
            <QuickActions />
          </motion.div>

          {/* Recent Activities */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <ActivityFeed />
          </motion.div>

          {/* System Resources */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-brand-secondary" />
                  System Resources
                </CardTitle>
              </CardHeader>
              <CardContent>
                {metricsLoading ? (
                  <div className="space-y-3">
                    {[...Array(4)].map((_, i) => (
                      <div key={i} className="h-4 bg-muted rounded animate-pulse" />
                    ))}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* CPU Usage */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Cpu className="w-4 h-4 text-blue-500" />
                          <span className="text-sm">CPU</span>
                        </div>
                        <span className="text-sm font-mono">
                          {systemMetrics?.cpu?.average_usage || 0}%
                        </span>
                      </div>
                      <Progress 
                        value={systemMetrics?.cpu?.average_usage || 0} 
                        className="h-1.5"
                      />
                    </div>

                    {/* Memory Usage */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <MemoryStick className="w-4 h-4 text-green-500" />
                          <span className="text-sm">Memory</span>
                        </div>
                        <span className="text-sm font-mono">
                          {systemMetrics?.memory?.percent || 0}%
                        </span>
                      </div>
                      <Progress 
                        value={systemMetrics?.memory?.percent || 0} 
                        className="h-1.5"
                      />
                    </div>

                    {/* Disk Usage */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <HardDrive className="w-4 h-4 text-orange-500" />
                          <span className="text-sm">Disk</span>
                        </div>
                        <span className="text-sm font-mono">
                          {systemMetrics?.disk?.percent || 0}%
                        </span>
                      </div>
                      <Progress 
                        value={systemMetrics?.disk?.percent || 0} 
                        className="h-1.5"
                      />
                    </div>

                    {/* Network Status */}
                    <div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Wifi className="w-4 h-4 text-purple-500" />
                          <span className="text-sm">Network</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                          <span className="text-sm font-mono">Online</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* API Health Status */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="w-5 h-5 text-brand-accent" />
                  API Health
                </CardTitle>
              </CardHeader>
              <CardContent>
                {healthLoading ? (
                  <div className="space-y-3">
                    {[...Array(4)].map((_, i) => (
                      <div key={i} className="h-4 bg-muted rounded animate-pulse" />
                    ))}
                  </div>
                ) : (
                  <div className="space-y-3">
                    {systemHealth?.endpoints?.map((endpoint: any) => (
                      <div key={endpoint.name} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full ${
                            endpoint.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                          }`} />
                          <span className="text-sm">{endpoint.name}</span>
                        </div>
                        <span className="text-xs font-mono text-muted-foreground">
                          {endpoint.latency}ms
                        </span>
                      </div>
                    ))}
                    
                    {!systemHealth?.endpoints && (
                      <div className="text-sm text-muted-foreground">
                        No API health data available
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  )
}