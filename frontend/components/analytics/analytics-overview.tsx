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
import { StatsBar } from '@/components/shared/stats-bar'
import { useAnalyticsOverview, usePlanUsage, useRecommendations, useWorkspaceMemory, useChartPresets } from '@/hooks/use-unified-analytics'
import { AnalyticsRecommendations } from './analytics-recommendations'
import { AnalyticsPlanUsage } from './analytics-plan-usage'
import { AnalyticsMemory } from './analytics-memory'
import { AnalyticsPandasChart } from './analytics-pandas-chart'

interface OverviewProps {
  days: number
}

export function AnalyticsOverview({ days }: OverviewProps) {
  const { data: overview, isLoading } = useAnalyticsOverview(days)
  const { data: planUsage, isLoading: planLoading } = usePlanUsage()
  const { data: recommendations, isLoading: recsLoading } = useRecommendations()
  const { data: memory, isLoading: memoryLoading } = useWorkspaceMemory()
  const { data: chartPresets } = useChartPresets()

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
      <StatsBar stats={summaryCards.map(card => ({
        label: card.label,
        value: card.value,
        change: card.sub,
        icon: card.icon,
        iconColor: card.color.replace('text-orange-400', 'text-primary')
          .replace('text-purple-400', 'text-[hsl(var(--info))]')
          .replace('text-green-400', 'text-[hsl(var(--success))]')
          .replace('text-blue-400', 'text-[hsl(var(--agent))]'),
      }))} loading={isLoading} />

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

      {/* AI-Generated Insights — only render if presets are available */}
      {chartPresets && chartPresets.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.7 }}
        >
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Sparkles className="w-4 h-4 text-purple-400" />
                AI-Generated Insights
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <AnalyticsPandasChart presetId="cost-by-model" />
                <AnalyticsPandasChart presetId="tokens-over-time" />
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  )
}
