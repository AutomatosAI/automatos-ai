'use client'

import { motion } from 'framer-motion'
import {
  Bot,
  GitBranch,
  FileText,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Brain,
  Sparkles,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { useAnalyticsOverview, usePlanUsage, useRecommendations, useWorkspaceMemory } from '@/hooks/use-unified-analytics'
import { AnalyticsRecommendations } from './analytics-recommendations'
import { AnalyticsPlanUsage } from './analytics-plan-usage'
import { AnalyticsMemory } from './analytics-memory'

interface OverviewProps {
  days: number
}

export function AnalyticsOverview({ days }: OverviewProps) {
  const { data: overview, isLoading } = useAnalyticsOverview(days)
  const { data: planUsage, isLoading: planLoading } = usePlanUsage()
  const { data: recommendations, isLoading: recsLoading } = useRecommendations()
  const { data: memory, isLoading: memoryLoading } = useWorkspaceMemory()

  const summaryCards = [
    {
      label: 'Agents',
      value: overview?.agents.total || 0,
      sub: `${overview?.agents.active || 0} active`,
      icon: Bot,
      color: 'text-orange-400',
      bgColor: 'from-orange-500/20 to-orange-500/5',
    },
    {
      label: 'Workflows',
      value: overview?.workflows.total || 0,
      sub: `${overview?.workflows.successRate?.toFixed(0) || 0}% success rate`,
      icon: GitBranch,
      color: 'text-purple-400',
      bgColor: 'from-purple-500/20 to-purple-500/5',
    },
    {
      label: 'Documents',
      value: overview?.documents.total || 0,
      sub: `${overview?.documents.storageMb?.toFixed(1) || 0} MB stored`,
      icon: FileText,
      color: 'text-green-400',
      bgColor: 'from-green-500/20 to-green-500/5',
    },
    {
      label: 'Monthly Cost',
      value: `$${(overview?.cost.currentPeriod || 0).toFixed(2)}`,
      sub: overview?.cost.previousPeriod
        ? `${overview.cost.currentPeriod > overview.cost.previousPeriod ? '+' : ''}${(((overview.cost.currentPeriod - overview.cost.previousPeriod) / (overview.cost.previousPeriod || 1)) * 100).toFixed(0)}% vs last period`
        : 'Current period',
      icon: DollarSign,
      color: 'text-blue-400',
      bgColor: 'from-blue-500/20 to-blue-500/5',
      trend: overview?.cost.previousPeriod
        ? overview.cost.currentPeriod <= overview.cost.previousPeriod ? 'down' : 'up'
        : undefined,
    },
  ]

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {isLoading
          ? Array.from({ length: 4 }).map((_, i) => (
              <Card key={i} className="glass-card">
                <CardContent className="p-6">
                  <Skeleton className="h-4 w-20 mb-3" />
                  <Skeleton className="h-8 w-16 mb-2" />
                  <Skeleton className="h-3 w-24" />
                </CardContent>
              </Card>
            ))
          : summaryCards.map((card, index) => (
              <motion.div
                key={card.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className={`glass-card card-glow hover:border-primary/20 transition-all duration-300 bg-gradient-to-br ${card.bgColor}`}>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                        <card.icon className={`w-5 h-5 ${card.color}`} />
                      </div>
                      {card.trend && (
                        <div className={`flex items-center text-xs ${card.trend === 'down' ? 'text-green-400' : 'text-orange-400'}`}>
                          {card.trend === 'down' ? <TrendingDown className="w-3 h-3 mr-1" /> : <TrendingUp className="w-3 h-3 mr-1" />}
                        </div>
                      )}
                    </div>
                    <h3 className="text-2xl font-bold mb-1">{card.value}</h3>
                    <p className="text-sm text-muted-foreground">{card.label}</p>
                    <p className="text-xs text-muted-foreground mt-1">{card.sub}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
      </div>

      {/* Plan Usage */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <AnalyticsPlanUsage data={planUsage} isLoading={planLoading} />
      </motion.div>

      {/* Memory Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
      >
        <AnalyticsMemory data={memory} isLoading={memoryLoading} />
      </motion.div>

      {/* AI Recommendations */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <AnalyticsRecommendations recommendations={recommendations || []} isLoading={recsLoading} />
      </motion.div>
    </div>
  )
}
