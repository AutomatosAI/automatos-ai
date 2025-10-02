
'use client'

import { motion } from 'framer-motion'
import { 
  Bot, 
  FileText, 
  GitBranch, 
  TrendingUp,
  Activity,
  Zap,
  Clock,
  CheckCircle
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { useAgents, useDocuments, useWorkflows } from '@/hooks/use-api'
import { useSystemMetrics, useApiHealth } from '@/hooks/use-system-config-api'

interface MetricCardProps {
  title: string
  value: string | number
  change?: string
  changeType?: 'positive' | 'negative' | 'neutral'
  icon: React.ComponentType<any>
  gradient: string
  badge?: string
}

function MetricCard({ title, value, change, changeType, icon: Icon, gradient, badge }: MetricCardProps) {
  return (
    <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300 overflow-hidden">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-transparent">
            <Icon className={`w-5 h-5 ${
              gradient.includes('orange') ? 'text-orange-400' :
              gradient.includes('blue') ? 'text-blue-400' :
              gradient.includes('green') ? 'text-green-400' :
              gradient.includes('purple') ? 'text-purple-400' :
              'text-white'
            }`} />
          </div>
        </div>
        <div className="space-y-1">
          <h3 className="text-2xl font-bold">{value}</h3>
          <p className="text-muted-foreground text-sm">{title}</p>
          {change && (
            <p className={`text-xs ${
              changeType === 'positive' 
                ? 'text-green-400' 
                : changeType === 'negative' 
                ? 'text-red-400' 
                : 'text-muted-foreground'
            }`}>
              {change}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export function MetricCards() {
  // Use real API hooks
  const { data: agents, isLoading: agentsLoading } = useAgents()
  const { data: documents, isLoading: documentsLoading } = useDocuments()
  const { data: workflows, isLoading: workflowsLoading } = useWorkflows()
  const { data: systemMetrics, isLoading: metricsLoading } = useSystemMetrics()
  const { data: apiHealth, isLoading: apiHealthLoading } = useApiHealth()
  
  const isLoading = agentsLoading || documentsLoading || workflowsLoading || metricsLoading || apiHealthLoading
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[...Array(4)].map((_, index) => (
          <Card key={index} className="glass-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <Skeleton className="h-4 w-20 mb-2" />
                  <Skeleton className="h-8 w-16" />
                </div>
                <Skeleton className="w-12 h-12 rounded-xl" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  // Calculate real metrics from API data - safely handle missing/null data
  const agentsArray = Array.isArray(agents) ? agents : []
  const workflowsArray = Array.isArray(workflows) ? workflows : []
  const documentsArray = Array.isArray(documents) ? documents : []
  
  const activeAgents = agentsArray.filter((a: any) => a?.status === 'active').length || 0
  const totalAgents = agentsArray.length || 0
  const documentsProcessed = documentsArray.length || 0
  const runningWorkflows = workflowsArray.filter((w: any) => w?.status === 'running').length || 0
  const totalWorkflows = workflowsArray.length || 0
  const apiCalls = apiHealth?.summary?.total_calls || 0

  const metrics = [
    {
      title: 'Active Agents',
      value: activeAgents,
      change: totalAgents > 0 ? `+${Math.round((activeAgents / totalAgents) * 100)}%` : '+0%',
      changeType: activeAgents > 0 ? 'positive' : 'neutral' as const,
      icon: Bot,
      gradient: 'from-orange-500 to-red-500',
      badge: 'Live'
    },
    {
      title: 'Documents Processed',
      value: documentsProcessed,
      change: documentsProcessed > 0 ? `+${documentsProcessed}` : '+0',
      changeType: documentsProcessed > 0 ? 'positive' : 'neutral' as const,
      icon: FileText,
      gradient: 'from-blue-500 to-purple-500',
    },
    {
      title: 'Active Workflows',
      value: runningWorkflows,
      change: totalWorkflows > 0 ? `${totalWorkflows} total` : '0 total',
      changeType: runningWorkflows > 0 ? 'positive' : 'neutral' as const,
      icon: GitBranch,
      gradient: 'from-green-500 to-teal-500',
    },
    {
      title: 'API Calls',
      value: apiCalls,
      change: apiCalls > 1000 ? `${Math.round(apiCalls / 1000)}K today` : `${apiCalls} today`,
      changeType: apiCalls > 0 ? 'positive' : 'neutral' as const,
      icon: Activity,
      gradient: 'from-purple-500 to-pink-500',
      badge: 'Real-time'
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric, index) => (
        <motion.div
          key={metric.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: index * 0.1 }}
        >
          <MetricCard {...metric} />
        </motion.div>
      ))}
    </div>
  )
}
