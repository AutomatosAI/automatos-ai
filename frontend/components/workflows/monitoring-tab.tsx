'use client'

import { motion } from 'framer-motion'
import { 
  Activity,
  TrendingUp,
  TrendingDown,
  CheckCircle,
  AlertTriangle,
  Clock
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { useWorkflowStatsDashboard } from '@/hooks/use-workflow-api'

export function MonitoringTab() {
  const { data: stats, isLoading, error } = useWorkflowStatsDashboard()
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading monitoring data...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
          <p className="text-red-400">Error loading monitoring data</p>
        </div>
      </div>
    )
  }

  const metrics = [
    {
      title: 'Total Workflows',
      value: stats?.totalWorkflows || 0,
      change: '+12%',
      trend: 'up',
      icon: Activity,
      color: 'text-blue-400'
    },
    {
      title: 'Active Executions',
      value: stats?.activeExecutions || 0,
      change: '+5%',
      trend: 'up',
      icon: CheckCircle,
      color: 'text-green-400'
    },
    {
      title: 'Success Rate',
      value: `${stats?.successRate || 0}%`,
      change: '+3%',
      trend: 'up',
      icon: TrendingUp,
      color: 'text-green-400'
    },
    {
      title: 'Avg Duration',
      value: stats?.avgDuration || '0m',
      change: '-8%',
      trend: 'down',
      icon: Clock,
      color: 'text-orange-400'
    }
  ]

  return (
    <div className="space-y-6">
      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => {
          const Icon = metric.icon
          const TrendIcon = metric.trend === 'up' ? TrendingUp : TrendingDown
          
          return (
            <motion.div
              key={metric.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card className="glass-card card-glow">
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center`}>
                      <Icon className={`w-5 h-5 ${metric.color}`} />
                    </div>
                    <div className={`flex items-center text-xs ${metric.trend === 'up' ? 'text-green-400' : 'text-orange-400'}`}>
                      <TrendIcon className="w-3 h-3 mr-1" />
                      {metric.change}
                    </div>
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold mb-1">{metric.value}</h3>
                    <p className="text-sm text-muted-foreground">{metric.title}</p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )
        })}
      </div>

      {/* System Health */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Activity className="w-5 h-5 mr-2 text-green-400" />
              System Health
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm">CPU Usage</span>
                <span className="text-sm font-medium">{stats?.cpuUsage || 0}%</span>
              </div>
              <Progress value={stats?.cpuUsage || 0} className="h-2" />
            </div>
            
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm">Memory Usage</span>
                <span className="text-sm font-medium">{stats?.memoryUsage || 0}%</span>
              </div>
              <Progress value={stats?.memoryUsage || 0} className="h-2" />
            </div>
            
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm">Agent Utilization</span>
                <span className="text-sm font-medium">{stats?.agentUtilization || 0}%</span>
              </div>
              <Progress value={stats?.agentUtilization || 0} className="h-2" />
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Recent Activity */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Real-time Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <div>
                    <p className="text-sm font-medium">Workflow execution started</p>
                    <p className="text-xs text-muted-foreground">2 minutes ago</p>
                  </div>
                </div>
                <CheckCircle className="w-4 h-4 text-green-400" />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  <div>
                    <p className="text-sm font-medium">Agent coordination in progress</p>
                    <p className="text-xs text-muted-foreground">5 minutes ago</p>
                  </div>
                </div>
                <Activity className="w-4 h-4 text-blue-400" />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                  <div>
                    <p className="text-sm font-medium">Task decomposition completed</p>
                    <p className="text-xs text-muted-foreground">10 minutes ago</p>
                  </div>
                </div>
                <CheckCircle className="w-4 h-4 text-orange-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}

