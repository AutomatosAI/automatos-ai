'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Shield,
  Users,
  DollarSign,
  Zap,
  Activity,
  Download,
  AlertTriangle,
  Building,
  Server,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { StatsBar } from '@/components/shared/stats-bar'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from 'recharts'
import { useAdminWorkspaceAnalytics, useAdminCostAnalytics } from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

const PROVIDER_COLORS = ['#60B5FF', '#72BF78', '#ff6b35', '#a78bfa', '#f472b6', '#38bdf8']

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}

export function AnalyticsAdmin({ days }: Props) {
  const { data, isLoading } = useAdminWorkspaceAnalytics(days)
  const [costPeriod, setCostPeriod] = useState('30d')
  const { data: costData, isLoading: costLoading } = useAdminCostAnalytics(costPeriod)
  const [costSortDir, setCostSortDir] = useState<'asc' | 'desc'>('desc')

  const escapeCsvField = (value: string): string => {
    // Neutralize formula-characters to prevent CSV injection
    let escaped = value
    if (/^[=+\-@]/.test(escaped)) {
      escaped = "'" + escaped
    }
    // Double internal quotes and wrap in quotes
    return '"' + escaped.replace(/"/g, '""') + '"'
  }

  const handleExport = () => {
    if (!data?.workspaces) return

    const headers = ['Workspace,Plan,Users,Agents,Workflows,API Calls,Tokens,Cost,Status']
    const rows = data.workspaces.map(ws =>
      [
        escapeCsvField(ws.name),
        escapeCsvField(ws.plan),
        ws.users,
        ws.agents,
        ws.workflows,
        ws.apiCalls,
        ws.tokens,
        `$${ws.cost.toFixed(2)}`,
        escapeCsvField(ws.status),
      ].join(',')
    )
    const csv = [...headers, ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `workspace-analytics-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
          {Array.from({ length: 5 }).map((_, i) => (
            <Card key={i} className="glass-card"><CardContent className="p-6"><Skeleton className="h-8 w-16 mb-2" /><Skeleton className="h-4 w-24" /></CardContent></Card>
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Admin Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-red-400" />
          <span className="text-sm font-medium text-muted-foreground">Super Admin View</span>
        </div>
        <Button variant="outline" size="sm" onClick={handleExport}>
          <Download className="w-4 h-4 mr-2" />
          Export CSV
        </Button>
      </div>

      <StatsBar stats={[
        { label: 'Workspaces', value: data?.platformSummary?.totalWorkspaces || 0, icon: Building, iconColor: 'text-primary' },
        { label: 'Total Users', value: data?.platformSummary?.totalUsers || 0, icon: Users, iconColor: 'text-[hsl(var(--success))]' },
        { label: 'API Calls', value: formatNumber(data?.platformSummary?.totalApiCalls || 0), icon: Activity, iconColor: 'text-[hsl(var(--info))]' },
        { label: 'Total Tokens', value: formatNumber(data?.platformSummary?.totalTokens || 0), icon: Zap, iconColor: 'text-[hsl(var(--agent))]' },
      ]} loading={isLoading} />

      {/* Workspace Table */}
      <Card className="glass-card overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building className="w-5 h-5 text-blue-400" />
            Workspace Usage
          </CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Workspace</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Plan</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden md:table-cell">Users</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Agents</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden md:table-cell">Workflows</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">API Calls</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden lg:table-cell">Tokens</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Cost</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden lg:table-cell">Status</th>
              </tr>
            </thead>
            <tbody>
              {(!data?.workspaces || data.workspaces.length === 0) ? (
                <tr>
                  <td colSpan={9} className="p-12 text-center text-muted-foreground">
                    <Building className="w-10 h-10 mx-auto mb-3 opacity-50" />
                    <p>No workspace data available</p>
                  </td>
                </tr>
              ) : (
                data.workspaces.map((ws) => (
                  <tr key={ws.id} className="border-b border-border/30 hover:bg-secondary/20 transition-colors">
                    <td className="p-4 font-medium">{ws.name}</td>
                    <td className="p-4">
                      <Badge variant="outline" className="text-xs">{ws.plan}</Badge>
                    </td>
                    <td className="p-4 text-sm hidden md:table-cell">{ws.users}</td>
                    <td className="p-4 text-sm">{ws.agents}</td>
                    <td className="p-4 text-sm hidden md:table-cell">{ws.workflows}</td>
                    <td className="p-4 text-sm">{formatNumber(ws.apiCalls)}</td>
                    <td className="p-4 text-sm hidden lg:table-cell">{formatNumber(ws.tokens)}</td>
                    <td className="p-4 text-sm font-medium">${ws.cost.toFixed(2)}</td>
                    <td className="p-4 hidden lg:table-cell">
                      <Badge variant="outline" className={ws.status === 'active' ? 'text-green-400 border-green-400/30' : ''}>
                        {ws.status}
                      </Badge>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* ──────────── Platform Costs ──────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="space-y-6"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-green-400" />
            <span className="text-lg font-semibold">Platform Costs</span>
          </div>
          <select
            value={costPeriod}
            onChange={(e) => setCostPeriod(e.target.value)}
            className="text-xs bg-secondary/50 border border-border/50 rounded-md px-2 py-1 text-foreground"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
        </div>

        {costLoading ? (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {Array.from({ length: 4 }).map((_, i) => (
                <Card key={i} className="glass-card"><CardContent className="p-6"><Skeleton className="h-8 w-16 mb-2" /><Skeleton className="h-4 w-24" /></CardContent></Card>
              ))}
            </div>
            <Skeleton className="h-64 w-full" />
          </div>
        ) : costData ? (
          <>
            <StatsBar stats={[
              { label: 'Total Platform Cost', value: `$${costData.total_platform_cost.toFixed(2)}`, icon: DollarSign, iconColor: 'text-[hsl(var(--success))]' },
              { label: 'Total Tokens', value: formatNumber(costData.total_tokens), icon: Zap, iconColor: 'text-primary' },
              { label: 'Total API Requests', value: formatNumber(costData.total_requests), icon: Activity, iconColor: 'text-[hsl(var(--info))]' },
              { label: 'Active Providers', value: costData.cost_by_provider.length, icon: Server, iconColor: 'text-[hsl(var(--agent))]' },
            ]} loading={false} />

            {/* Daily Cost Trend - Stacked Area Chart */}
            {costData.daily_cost_trend && costData.daily_cost_trend.length > 0 && (
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-blue-400" />
                    Daily Cost Trend
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={costData.daily_cost_trend}>
                        <XAxis
                          dataKey="date"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                        />
                        <YAxis
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                          tickFormatter={(v: number) => `$${v.toFixed(2)}`}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px',
                            fontSize: '12px',
                          }}
                          formatter={(value: number) => [`$${value.toFixed(4)}`, 'Cost']}
                        />
                        <Area
                          type="monotone"
                          dataKey="cost"
                          stroke="#72BF78"
                          fill="#72BF78"
                          fillOpacity={0.3}
                          name="Daily Cost ($)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Provider Distribution + Per-Workspace Cost Table row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Provider Distribution Pie Chart */}
              {costData.cost_by_provider && costData.cost_by_provider.length > 0 && (
                <Card className="glass-card lg:col-span-1">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Server className="w-5 h-5 text-purple-400" />
                      Cost by Provider
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={costData.cost_by_provider.map(p => ({
                              name: p.key,
                              value: p.total_cost,
                            }))}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            labelLine={{ stroke: 'hsl(var(--muted-foreground))' }}
                          >
                            {costData.cost_by_provider.map((_entry, index) => (
                              <Cell key={index} fill={PROVIDER_COLORS[index % PROVIDER_COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'hsl(var(--card))',
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                            formatter={(value: number) => [`$${value.toFixed(4)}`, 'Cost']}
                          />
                          <Legend wrapperStyle={{ fontSize: '11px' }} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Per-Workspace Cost Table */}
              <Card className={`glass-card overflow-hidden ${costData.cost_by_provider && costData.cost_by_provider.length > 0 ? 'lg:col-span-2' : 'lg:col-span-3'}`}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      <Building className="w-5 h-5 text-blue-400" />
                      Cost by Workspace
                    </span>
                    <button
                      onClick={() => setCostSortDir(prev => prev === 'desc' ? 'asc' : 'desc')}
                      className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      Sort: {costSortDir === 'desc' ? 'High → Low' : 'Low → High'}
                    </button>
                  </CardTitle>
                </CardHeader>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-border/50">
                        <th className="text-left p-4 text-xs font-medium text-muted-foreground">Workspace</th>
                        <th className="text-left p-4 text-xs font-medium text-muted-foreground">Plan</th>
                        <th className="text-left p-4 text-xs font-medium text-muted-foreground">Total Cost</th>
                        <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden md:table-cell">Total Tokens</th>
                        <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden md:table-cell">Requests</th>
                        <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden lg:table-cell">Top Model</th>
                      </tr>
                    </thead>
                    <tbody>
                      {costData.cost_by_workspace.length === 0 ? (
                        <tr>
                          <td colSpan={6} className="p-12 text-center text-muted-foreground">
                            <Building className="w-10 h-10 mx-auto mb-3 opacity-50" />
                            <p>No workspace cost data available</p>
                          </td>
                        </tr>
                      ) : (
                        [...costData.cost_by_workspace]
                          .sort((a, b) => costSortDir === 'desc' ? b.total_cost - a.total_cost : a.total_cost - b.total_cost)
                          .map((ws) => (
                            <tr key={ws.workspace_id} className="border-b border-border/30 hover:bg-secondary/20 transition-colors">
                              <td className="p-4 font-medium text-sm">{ws.workspace_name}</td>
                              <td className="p-4">
                                <Badge variant="outline" className="text-xs">{ws.plan}</Badge>
                              </td>
                              <td className="p-4 text-sm font-medium">${ws.total_cost.toFixed(2)}</td>
                              <td className="p-4 text-sm hidden md:table-cell">{formatNumber(ws.total_tokens)}</td>
                              <td className="p-4 text-sm hidden md:table-cell">{formatNumber(ws.total_requests)}</td>
                              <td className="p-4 text-sm hidden lg:table-cell">
                                {ws.top_model ? (
                                  <Badge variant="secondary" className="font-mono text-xs">{ws.top_model}</Badge>
                                ) : (
                                  <span className="text-muted-foreground">—</span>
                                )}
                              </td>
                            </tr>
                          ))
                      )}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          </>
        ) : (
          <Card className="glass-card">
            <CardContent className="p-12 text-center">
              <AlertTriangle className="w-10 h-10 mx-auto mb-3 opacity-50 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Platform cost data unavailable — admin access required</p>
            </CardContent>
          </Card>
        )}
      </motion.div>
    </div>
  )
}
