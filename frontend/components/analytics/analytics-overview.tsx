'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import {
  Bot,
  GitBranch,
  FileText,
  DollarSign,
  Activity,
  MessageSquare,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { StatsBar } from '@/components/shared/stats-bar'
import { useAnalyticsOverview, usePlanUsage, useRecommendations } from '@/hooks/use-unified-analytics'
import { AnalyticsRecommendations } from './analytics-recommendations'
import { AnalyticsPlanUsage } from './analytics-plan-usage'

interface OverviewProps {
  days: number
}

export function AnalyticsOverview({ days }: OverviewProps) {
  const { data: overview, isLoading } = useAnalyticsOverview(days)
  const { data: planUsage, isLoading: planLoading } = usePlanUsage()
  const { data: recommendations, isLoading: recsLoading } = useRecommendations()

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
        .then((data: any) => {
          // Merge today_by_source (routing_decisions) + channel_connections total messages
          // Normalize keys to lowercase for matching
          const merged: Record<string, number> = {}
          for (const [key, val] of Object.entries(data.today_by_source || {})) {
            merged[key.toLowerCase()] = (merged[key.toLowerCase()] || 0) + (val as number)
          }
          // Add channel_connections totals for channels with no routing_decisions today
          for (const ch of (data.channels || [])) {
            const key = (ch.platform || '').toLowerCase()
            if (key && !merged[key]) {
              merged[key] = ch.total_messages || 0
            }
          }
          setChannelStats(merged)
        })
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
    },
    {
      label: 'Missions / Runs',
      value: overview?.workflows.total || 0,
      sub: `${overview?.workflows.successRate?.toFixed(0) || 0}% success rate`,
      icon: GitBranch,
      color: 'text-purple-400',
      bgColor: 'from-purple-500/20 to-purple-500/5',
    },
    {
      label: 'Documents / Usage',
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
      <StatsBar stats={summaryCards.map((card, idx) => ({
        label: card.label,
        value: card.value,
        change: card.sub,
        icon: card.icon,
        iconColor: card.color.replace('text-orange-400', 'text-primary')
          .replace('text-purple-400', 'text-[hsl(var(--info))]')
          .replace('text-green-400', 'text-[hsl(var(--success))]')
          .replace('text-blue-400', 'text-[hsl(var(--agent))]'),
        globalIconKey: (['global_agent', 'global_workflow', 'global_document', 'global_cost'] as const)[idx],
      }))} loading={isLoading} />

      {/* Plan Usage */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <AnalyticsPlanUsage data={planUsage} isLoading={planLoading} />
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
            <div className="grid grid-cols-4 gap-3">
              {[
                { label: 'Today', value: heartbeatStats.total_heartbeats ?? 0 },
                { label: 'Successes', value: heartbeatStats.successes ?? 0, accent: (heartbeatStats.successes ?? 0) > 0 ? 'text-green-400' : '' },
                { label: 'Errors', value: heartbeatStats.errors ?? 0, accent: (heartbeatStats.errors ?? 0) === 0 ? 'text-muted-foreground' : 'text-red-400' },
                { label: 'Tokens', value: heartbeatStats.total_tokens?.toLocaleString() || 0 },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="rounded-xl px-4 py-3 bg-muted/30 border border-border/40"
                >
                  <p className={`text-xl font-semibold tabular-nums ${stat.accent ?? ''}`}>{stat.value}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{stat.label}</p>
                </div>
              ))}
            </div>
            {heartbeatStats.recent_events?.length > 0 && (() => {
              // Group consecutive pings from the same source — a single
              // chatty agent shouldn't fill the panel with identical rows.
              type Event = { source_type: string; source_id: string; status: string; created_at: string }
              const events: Event[] = heartbeatStats.recent_events.slice(0, 25)
              const groups: Array<{ key: string; type: string; id: string; count: number; lastAt: string; statuses: string[] }> = []
              for (const evt of events) {
                const key = `${evt.source_type}::${evt.source_id}`
                const last = groups[groups.length - 1]
                if (last && last.key === key) {
                  last.count += 1
                  last.lastAt = evt.created_at
                  last.statuses.push(evt.status)
                } else {
                  groups.push({ key, type: evt.source_type, id: evt.source_id, count: 1, lastAt: evt.created_at, statuses: [evt.status] })
                }
              }
              return (
                <div className="mt-4 space-y-1.5">
                  {groups.slice(0, 5).map((g) => {
                    const allOk = g.statuses.every((s) => s === 'success')
                    return (
                      <div key={g.key + g.lastAt} className="flex items-center gap-3 text-sm py-2 border-t border-border/30 first:border-t-0">
                        <span className={`h-2 w-2 rounded-full shrink-0 ${allOk ? 'bg-green-500 ring-4 ring-green-500/10' : 'bg-red-500 ring-4 ring-red-500/10'}`} />
                        <span className="text-muted-foreground text-xs uppercase tracking-wide">{g.type}</span>
                        <span className="flex-1 truncate font-medium">{g.id}</span>
                        {g.count > 1 && (
                          <span className="text-[11px] px-2 py-0.5 rounded-full bg-secondary/40 border border-border/40 text-muted-foreground">
                            {g.count} pings
                          </span>
                        )}
                        <span className="text-xs text-muted-foreground tabular-nums shrink-0">
                          {new Date(g.lastAt).toLocaleTimeString()}
                        </span>
                      </div>
                    )
                  })}
                </div>
              )
            })()}
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

    </div>
  )
}
