'use client'

import { useState, Fragment } from 'react'
import {
  Wrench,
  AppWindow,
  Zap,
  Link2,
  ChevronDown,
  ChevronRight,
  Bot,
  FileText,
  Activity,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Timer,
  Gauge,
  TrendingUp,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { StatsBar } from '@/components/shared/stats-bar'
import { Progress } from '@/components/ui/progress'
import {
  useComposioApps,
  useComposioActions,
  useComposioAgentTools,
  useComposioExecStats,
  useComposioPerformance,
  useComposioDailyVolume,
  useComposioRecentExecs,
  useComposioErrors,
} from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return 'Never'
  const d = new Date(dateStr)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

/** Format Composio app names: GOOGLEDRIVE → Google Drive */
function formatAppName(name: string): string {
  const known: Record<string, string> = {
    GOOGLEDRIVE: 'Google Drive',
    GOOGLECALENDAR: 'Google Calendar',
    GOOGLEDOCS: 'Google Docs',
    GOOGLESHEETS: 'Google Sheets',
    GOOGLEGMAIL: 'Gmail',
    GITHUB: 'GitHub',
    SLACK: 'Slack',
    DROPBOX: 'Dropbox',
    TELEGRAM: 'Telegram',
    NOTION: 'Notion',
    JIRA: 'Jira',
    TRELLO: 'Trello',
    DISCORD: 'Discord',
    LINEAR: 'Linear',
    ASANA: 'Asana',
    WHATSAPP: 'WhatsApp',
  }
  if (known[name]) return known[name]
  // Fallback: capitalize first, lowercase rest
  return name.charAt(0) + name.slice(1).toLowerCase()
}

function getStatusVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status.toLowerCase()) {
    case 'active':
    case 'connected':
      return 'default'
    case 'error':
    case 'failed':
      return 'destructive'
    case 'disconnected':
    case 'inactive':
      return 'secondary'
    default:
      return 'outline'
  }
}

type SortKey = 'total_usage_count' | 'agent_count' | 'action_name' | 'app_name'
type SortDir = 'asc' | 'desc'

export function AnalyticsComposio({ days }: Props) {
  const { data: apps, isLoading: appsLoading } = useComposioApps(days)
  const { data: actions, isLoading: actionsLoading } = useComposioActions(days)
  const { data: agentTools, isLoading: agentToolsLoading } = useComposioAgentTools(days)
  const { data: execStats, isLoading: execStatsLoading } = useComposioExecStats(days)
  const { data: perfByAction, isLoading: perfLoading } = useComposioPerformance(days)
  const { data: dailyVolume } = useComposioDailyVolume(days)
  const { data: recentExecs } = useComposioRecentExecs()
  const { data: errors } = useComposioErrors(days)
  const [expandedAgents, setExpandedAgents] = useState<Set<number>>(new Set())
  const [sortKey, setSortKey] = useState<SortKey>('total_usage_count')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const isLoading = appsLoading || actionsLoading || agentToolsLoading

  const toggleAgent = (agentId: number) => {
    setExpandedAgents(prev => {
      const next = new Set(prev)
      if (next.has(agentId)) next.delete(agentId)
      else next.add(agentId)
      return next
    })
  }

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(prev => prev === 'desc' ? 'asc' : 'desc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const sortedActions = actions ? [...actions].sort((a, b) => {
    const aVal = a[sortKey]
    const bVal = b[sortKey]
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
    }
    const aNum = Number(aVal) || 0
    const bNum = Number(bVal) || 0
    return sortDir === 'asc' ? aNum - bNum : bNum - aNum
  }) : []

  // Stats calculations
  const connectedAppsCount = apps?.length ?? 0
  const totalActionsUsed = apps?.reduce((sum, app) => sum + app.total_actions_used, 0) ?? 0
  const mostUsedApp = apps && apps.length > 0
    ? [...apps].sort((a, b) => b.total_actions_used - a.total_actions_used)[0]
    : null
  const activeIntegrations = apps?.filter(a =>
    a.status.toLowerCase() === 'active' || a.status.toLowerCase() === 'connected'
  ).length ?? 0

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="glass-card"><CardContent className="p-6"><Skeleton className="h-8 w-16 mb-2" /><Skeleton className="h-4 w-24" /></CardContent></Card>
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StatsBar stats={[
        { label: 'Connected Apps', value: connectedAppsCount, change: 'Composio integrations', icon: AppWindow, iconColor: 'text-primary' },
        { label: 'Total Actions Used', value: totalActionsUsed, change: `Last ${days} days`, icon: Zap, iconColor: 'text-[hsl(var(--success))]' },
        { label: 'Most Used App', value: mostUsedApp?.total_actions_used ?? 0, change: mostUsedApp ? formatAppName(mostUsedApp.app_name) : 'No data', icon: Wrench, iconColor: 'text-[hsl(var(--info))]' },
        { label: 'Active Integrations', value: activeIntegrations, change: `of ${connectedAppsCount} total`, icon: Link2, iconColor: 'text-[hsl(var(--agent))]' },
      ]} loading={isLoading} />

      {/* API Monitoring Overview */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" />
            API Execution Monitor
          </CardTitle>
        </CardHeader>
        <CardContent>
          {execStatsLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-16" />)}
            </div>
          ) : !execStats || execStats.total_executions === 0 ? (
            <div className="text-center py-8">
              <Activity className="w-10 h-10 mx-auto mb-3 opacity-50 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No tool executions recorded yet</p>
              <p className="text-xs text-muted-foreground mt-1">Execute Composio tools through your agents to see metrics here</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="rounded-lg border border-border/30 p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Gauge className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-xs text-muted-foreground">Total Calls</span>
                  </div>
                  <p className="text-xl font-bold">{execStats.total_executions.toLocaleString()}</p>
                  <p className="text-[10px] text-muted-foreground">{execStats.unique_actions} actions across {execStats.unique_apps} apps</p>
                </div>
                <div className="rounded-lg border border-border/30 p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                    <span className="text-xs text-muted-foreground">Success Rate</span>
                  </div>
                  <p className="text-xl font-bold">{execStats.success_rate}%</p>
                  <Progress value={execStats.success_rate} className="h-1.5 mt-1" />
                </div>
                <div className="rounded-lg border border-border/30 p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <XCircle className="w-3.5 h-3.5 text-red-400" />
                    <span className="text-xs text-muted-foreground">Errors</span>
                  </div>
                  <p className="text-xl font-bold">{execStats.error_count}</p>
                  <p className="text-[10px] text-muted-foreground">{execStats.timeout_count} timeouts</p>
                </div>
                <div className="rounded-lg border border-border/30 p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Timer className="w-3.5 h-3.5 text-yellow-400" />
                    <span className="text-xs text-muted-foreground">Avg Latency</span>
                  </div>
                  <p className="text-xl font-bold">{execStats.avg_latency_ms < 1000 ? `${execStats.avg_latency_ms}ms` : `${(execStats.avg_latency_ms / 1000).toFixed(1)}s`}</p>
                  <p className="text-[10px] text-muted-foreground">
                    p50: {execStats.p50_latency_ms != null ? `${execStats.p50_latency_ms}ms` : '—'} · p95: {execStats.p95_latency_ms != null ? `${execStats.p95_latency_ms}ms` : '—'}
                  </p>
                </div>
              </div>

              {/* Cache Hit Rate */}
              {execStats.cache_hit_rate > 0 && (
                <div className="flex items-center gap-3 text-sm px-1">
                  <span className="text-muted-foreground">Cache Hit Rate:</span>
                  <Progress value={execStats.cache_hit_rate} className="h-1.5 flex-1 max-w-xs" />
                  <span className="font-medium">{execStats.cache_hit_rate}%</span>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Daily Volume Chart */}
      {dailyVolume && dailyVolume.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-indigo-400" />
              Daily Execution Volume
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {(() => {
                const maxVal = Math.max(...dailyVolume.map(d => d.total), 1)
                return dailyVolume.slice(-14).map((d) => (
                  <div key={d.date} className="flex items-center gap-3 text-xs">
                    <span className="w-20 text-muted-foreground shrink-0">
                      {new Date(d.date + 'T00:00:00').toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                    </span>
                    <div className="flex-1 flex items-center gap-0.5 h-5">
                      <div
                        className="bg-green-500/70 h-full rounded-l"
                        style={{ width: `${(d.successes / maxVal) * 100}%` }}
                      />
                      {d.errors > 0 && (
                        <div
                          className="bg-red-500/70 h-full rounded-r"
                          style={{ width: `${(d.errors / maxVal) * 100}%` }}
                        />
                      )}
                    </div>
                    <span className="w-12 text-right font-medium tabular-nums">{d.total}</span>
                    <span className="w-16 text-right text-muted-foreground tabular-nums">{d.avg_latency_ms}ms</span>
                  </div>
                ))
              })()}
              <div className="flex items-center gap-4 text-[10px] text-muted-foreground pt-2 px-1">
                <div className="flex items-center gap-1"><div className="w-2.5 h-2.5 rounded bg-green-500/70" /> Success</div>
                <div className="flex items-center gap-1"><div className="w-2.5 h-2.5 rounded bg-red-500/70" /> Error</div>
                <div className="ml-auto">Rightmost column = avg latency</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Action Performance Breakdown */}
      {perfByAction && perfByAction.length > 0 && (
        <Card className="glass-card overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-orange-400" />
              Action Performance
            </CardTitle>
          </CardHeader>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border/50">
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">Action</th>
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">App</th>
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">Calls</th>
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">Success</th>
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">Err %</th>
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">Avg Latency</th>
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">Max</th>
                </tr>
              </thead>
              <tbody>
                {perfByAction.map((a) => (
                  <tr key={`${a.app_name}-${a.action_name}`} className="border-b border-border/30 hover:bg-secondary/20 transition-colors">
                    <td className="p-4">
                      <Badge variant="secondary" className="font-mono text-xs">{a.action_name}</Badge>
                    </td>
                    <td className="p-4 text-sm">{formatAppName(a.app_name)}</td>
                    <td className="p-4 text-sm font-medium tabular-nums">{a.total_calls}</td>
                    <td className="p-4 text-sm tabular-nums text-green-400">{a.success_count}</td>
                    <td className="p-4 text-sm tabular-nums">
                      <span className={a.error_rate > 10 ? 'text-red-400 font-medium' : a.error_rate > 0 ? 'text-yellow-400' : 'text-muted-foreground'}>
                        {a.error_rate}%
                      </span>
                    </td>
                    <td className="p-4 text-sm tabular-nums">
                      <span className={a.avg_latency_ms > 2000 ? 'text-red-400' : a.avg_latency_ms > 500 ? 'text-yellow-400' : ''}>
                        {a.avg_latency_ms < 1000 ? `${a.avg_latency_ms}ms` : `${(a.avg_latency_ms / 1000).toFixed(1)}s`}
                      </span>
                    </td>
                    <td className="p-4 text-sm text-muted-foreground tabular-nums">
                      {a.max_latency_ms < 1000 ? `${a.max_latency_ms}ms` : `${(a.max_latency_ms / 1000).toFixed(1)}s`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Recent Executions */}
      {recentExecs && recentExecs.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" />
              Recent Executions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {recentExecs.slice(0, 15).map((exec) => (
                <div key={exec.id} className="flex items-center gap-3 text-sm p-2 rounded bg-muted/30">
                  <span className={`h-2 w-2 rounded-full shrink-0 ${exec.status === 'success' ? 'bg-green-500' : exec.status === 'timeout' ? 'bg-yellow-500' : 'bg-red-500'}`} />
                  <span className="text-muted-foreground shrink-0 w-16 text-xs">{formatAppName(exec.app_name)}</span>
                  <Badge variant="secondary" className="font-mono text-[10px] shrink-0">{exec.action_name}</Badge>
                  {exec.agent_name && <span className="text-xs text-muted-foreground truncate">by {exec.agent_name}</span>}
                  <span className="flex-1" />
                  {exec.execution_time_ms != null && (
                    <span className="text-xs text-muted-foreground tabular-nums shrink-0">
                      {exec.execution_time_ms < 1000 ? `${exec.execution_time_ms}ms` : `${(exec.execution_time_ms / 1000).toFixed(1)}s`}
                    </span>
                  )}
                  {exec.cache_hit && <Badge variant="outline" className="text-[9px] px-1 py-0 shrink-0">cached</Badge>}
                  {exec.executed_at && (
                    <span className="text-[10px] text-muted-foreground shrink-0">
                      {new Date(exec.executed_at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error Breakdown */}
      {errors && errors.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-red-400" />
              Error Breakdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {errors.map((err, i) => (
                <div key={i} className="rounded-lg border border-red-500/20 bg-red-500/5 p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant="destructive" className="text-[10px]">{err.count}x</Badge>
                    <span className="text-sm font-medium">{formatAppName(err.app_name)}</span>
                    <Badge variant="secondary" className="font-mono text-[10px]">{err.action_name}</Badge>
                    {err.error_code && <span className="text-xs text-muted-foreground">({err.error_code})</span>}
                  </div>
                  <p className="text-xs text-muted-foreground line-clamp-2">{err.error_message}</p>
                  {err.last_seen && (
                    <p className="text-[10px] text-muted-foreground mt-1">
                      Last seen: {new Date(err.last_seen).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Connected Apps Grid */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AppWindow className="w-5 h-5 text-blue-400" />
            Connected Apps
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!apps || apps.length === 0 ? (
            <div className="text-center py-8">
              <AppWindow className="w-10 h-10 mx-auto mb-3 opacity-50 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No Composio apps connected yet</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {apps.map((app) => (
                <div
                  key={app.app_name}
                  className="rounded-lg border border-border/30 p-4 hover:border-border/60 transition-colors"
                >
                  <div className="flex items-center justify-between mb-3">
                    <p className="font-medium text-sm">{formatAppName(app.app_name)}</p>
                    <Badge variant={getStatusVariant(app.status)} className="text-[10px]">
                      {app.status}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs text-muted-foreground">
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-foreground font-medium tabular-nums">{app.total_actions_used}</span>
                      <span className="truncate">Actions Used</span>
                    </div>
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-foreground font-medium tabular-nums">{app.agent_count}</span>
                      <span className="truncate">Agents</span>
                    </div>
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-foreground font-medium tabular-nums">{app.documents_synced}</span>
                      <span className="truncate">Docs Synced</span>
                    </div>
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-foreground font-medium">{formatDate(app.last_used_at)}</span>
                      <span className="truncate">Last Used</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Action Leaderboard */}
      <Card className="glass-card overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Action Leaderboard
          </CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th
                  className="text-left p-4 text-xs font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort('action_name')}
                >
                  Action Name {sortKey === 'action_name' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th
                  className="text-left p-4 text-xs font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort('app_name')}
                >
                  App {sortKey === 'app_name' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th
                  className="text-left p-4 text-xs font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort('total_usage_count')}
                >
                  Usage Count {sortKey === 'total_usage_count' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th
                  className="text-left p-4 text-xs font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort('agent_count')}
                >
                  Agents Using {sortKey === 'agent_count' && (sortDir === 'asc' ? '↑' : '↓')}
                </th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Last Used</th>
              </tr>
            </thead>
            <tbody>
              {sortedActions.length === 0 ? (
                <tr>
                  <td colSpan={5} className="p-12 text-center text-muted-foreground">
                    <Zap className="w-10 h-10 mx-auto mb-3 opacity-50" />
                    <p>No action usage data yet</p>
                  </td>
                </tr>
              ) : (
                sortedActions.map((action) => (
                  <tr key={`${action.app_name}-${action.action_name}`} className="border-b border-border/30 hover:bg-secondary/20 transition-colors">
                    <td className="p-4">
                      <Badge variant="secondary" className="font-mono text-xs">{action.action_name}</Badge>
                    </td>
                    <td className="p-4 text-sm">{formatAppName(action.app_name)}</td>
                    <td className="p-4 text-sm font-medium">{action.total_usage_count}</td>
                    <td className="p-4 text-sm">{action.agent_count}</td>
                    <td className="p-4 text-sm text-muted-foreground">{formatDate(action.last_used_at)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Agent Tool Mapping */}
      <Card className="glass-card overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-green-400" />
            Agent Tool Mapping
          </CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          {!agentTools || agentTools.length === 0 ? (
            <div className="text-center py-12">
              <Bot className="w-10 h-10 mx-auto mb-3 opacity-50 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No agent tool mappings found</p>
            </div>
          ) : (
            <table className="w-full">
              <thead>
                <tr className="border-b border-border/50">
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground w-8"></th>
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">Agent</th>
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">Tools</th>
                  <th className="text-left p-4 text-xs font-medium text-muted-foreground">Total Usage</th>
                </tr>
              </thead>
              <tbody>
                {agentTools.map((agent) => {
                  const isExpanded = expandedAgents.has(agent.agent_id)
                  const totalUsage = agent.tools.reduce((sum, t) => sum + t.usage_count, 0)
                  return (
                    <Fragment key={agent.agent_id}>
                      <tr
                        className="border-b border-border/30 hover:bg-secondary/20 transition-colors cursor-pointer"
                        onClick={() => toggleAgent(agent.agent_id)}
                      >
                        <td className="p-4">
                          {isExpanded
                            ? <ChevronDown className="w-4 h-4 text-muted-foreground" />
                            : <ChevronRight className="w-4 h-4 text-muted-foreground" />
                          }
                        </td>
                        <td className="p-4 font-medium text-sm">{agent.agent_name}</td>
                        <td className="p-4 text-sm">{agent.tools.length} tool{agent.tools.length !== 1 ? 's' : ''}</td>
                        <td className="p-4 text-sm font-medium">{totalUsage}</td>
                      </tr>
                      {isExpanded && agent.tools.map((tool) => (
                        <tr
                          key={`${agent.agent_id}-${tool.tool_name}`}
                          className="border-b border-border/20 bg-secondary/10"
                        >
                          <td className="p-4"></td>
                          <td className="p-4 pl-8">
                            <div className="flex items-center gap-2">
                              <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                              <Badge variant="secondary" className="font-mono text-xs">{tool.tool_name}</Badge>
                            </div>
                          </td>
                          <td className="p-4 text-sm text-muted-foreground">{formatAppName(tool.app_name)}</td>
                          <td className="p-4 text-sm">
                            {tool.usage_count}
                            {!tool.enabled && (
                              <Badge variant="outline" className="ml-2 text-[10px] px-1.5 py-0">disabled</Badge>
                            )}
                          </td>
                        </tr>
                      ))}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </Card>
    </div>
  )
}
