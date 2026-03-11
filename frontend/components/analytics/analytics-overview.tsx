'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import {
  Bot,
  GitBranch,
  FileText,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Brain,
  Sparkles,
  Activity,
  MessageSquare,
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

  // PRD-55: Heartbeat and Channel analytics
  const [heartbeatStats, setHeartbeatStats] = useState<any>({ total_heartbeats: 0, total_tokens: 0, successes: 0, errors: 0, recent_events: [] })
  const [channelStats, setChannelStats] = useState<any>({})

  useEffect(() => {
    import('@/lib/api-client').then(({ apiClient }) => {
      apiClient.request('/api/heartbeat/analytics')
        .then((data: any) => {
          setHeartbeatStats({
            total_heartbeats: data.today?.total_heartbeats || 0,
            total_tokens: data.today?.total_tokens || 0,
            successes: data.today?.successes || 0,
            errors: data.today?.errors || 0,
            recent_events: data.recent_events || [],
          })
        })
        .catch((err: any) => console.warn('[Analytics] Heartbeat fetch failed:', err?.message || err))

      apiClient.request('/api/channels/analytics')
        .then((data: any) => setChannelStats(data.today_by_source || {}))
        .catch((err: any) => console.warn('[Analytics] Channel fetch failed:', err?.message || err))
    })
  }, [days])

  const summaryCards = [
    {
      label: 'Agents',
      value: overview?.agents.total || 0,
      sub: `${overview?.agents.active || 0} active`,
      icon: Bot,
      color: 'text-orange-400',
      bgColor: 'from-orange-500/20 to-orange-500/5',
      tooltipId: 'agents.roster.stats.total_agents',
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
      tooltipId: 'knowledge.library.stats.total_documents',
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
      tooltipId: 'analytics.costs.total_cost',
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
        tooltipId: card.tooltipId,
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

      {/* PRD-55: Heartbeat Activity */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.65 }}>
        <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5" /> Heartbeat Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <p className="text-xl font-bold">{heartbeatStats.total_heartbeats}</p>
                <p className="text-xs text-muted-foreground">Today</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <p className="text-xl font-bold">{heartbeatStats.successes}</p>
                <p className="text-xs text-muted-foreground">Successes</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <p className="text-xl font-bold">{heartbeatStats.errors}</p>
                <p className="text-xs text-muted-foreground">Errors</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-muted/50">
                <p className="text-xl font-bold">{heartbeatStats.total_tokens?.toLocaleString() || 0}</p>
                <p className="text-xs text-muted-foreground">Tokens</p>
              </div>
            </div>
            {heartbeatStats.recent_events?.length > 0 && (
              <div className="mt-4 space-y-2">
                {heartbeatStats.recent_events.slice(0, 5).map((evt: any, i: number) => (
                  <div key={i} className="flex items-center gap-2 text-sm p-2 rounded bg-muted/30">
                    <span className={`h-2 w-2 rounded-full ${evt.status === 'success' ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="text-muted-foreground">{evt.source_type}</span>
                    <span className="flex-1 truncate">{evt.source_id}</span>
                    <span className="text-xs text-muted-foreground">{new Date(evt.created_at).toLocaleTimeString()}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* PRD-55: Channel Activity */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.7 }}>
        <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><MessageSquare className="h-5 w-5" /> Channel Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-5 gap-3">
              {['web', 'telegram', 'slack', 'discord', 'webhook'].map(ch => (
                <div key={ch} className="text-center p-3 rounded-lg bg-muted/50">
                  <p className="text-lg font-bold">{channelStats[ch] || 0}</p>
                  <p className="text-xs text-muted-foreground capitalize">{ch}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* AI-Generated Insights */}
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
    </div>
  )
}
