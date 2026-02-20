'use client'

import { useState, useMemo, Fragment } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Shield,
  DollarSign,
  Zap,
  Activity,
  Download,
  Building,
  Server,
  TrendingUp,
  ChevronDown,
  ChevronRight,
  Bot,
  ChefHat,
  Play,
  Key,
  Crown,
  Layers,
  BarChart3,
  PieChart as PieChartIcon,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Progress } from '@/components/ui/progress'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts'
import {
  useAdminDashboard,
  useAdminWorkspaceAnalytics,
  useAdminCostAnalytics,
} from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

// Vibrant distinct colors — same palette as costs page for consistency
const PROVIDER_COLORS = [
  '#60B5FF', // sky blue
  '#72BF78', // green
  '#ff6b35', // orange
  '#a78bfa', // purple
  '#f472b6', // pink
  '#fbbf24', // amber
  '#34d399', // emerald
  '#f87171', // red
  '#38bdf8', // cyan
  '#c084fc', // violet
]

const PLAN_COLORS: Record<string, string> = {
  starter: 'text-gray-400 border-gray-400/30 bg-gray-400/5',
  pilot: 'text-blue-400 border-blue-400/30 bg-blue-400/5',
  pro: 'text-purple-400 border-purple-400/30 bg-purple-400/5',
  enterprise: 'text-amber-400 border-amber-400/30 bg-amber-400/5',
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}

function formatCost(n: number): string {
  if (n >= 100) return `$${n.toFixed(0)}`
  if (n >= 1) return `$${n.toFixed(2)}`
  if (n >= 0.01) return `$${n.toFixed(3)}`
  return `$${n.toFixed(4)}`
}

function shortenModelName(name: string): string {
  const parts = name.split('/')
  return parts[parts.length - 1]
}

// Period selector — matches costs page
function PeriodToggle({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const periods = [
    { key: '7d', label: '7D' },
    { key: '30d', label: '30D' },
    { key: '90d', label: '90D' },
  ]
  return (
    <div className="flex items-center rounded-lg border border-border/50 bg-secondary/30 p-0.5">
      {periods.map((p) => (
        <button
          key={p.key}
          onClick={() => onChange(p.key)}
          className={`px-3 py-1 text-xs font-medium rounded-md transition-all duration-200 ${
            value === p.key
              ? 'bg-primary text-primary-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}

// Custom tooltip for stacked area chart
function ProviderCostTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  const total = payload.reduce((s: number, p: any) => s + (p.value || 0), 0)
  return (
    <div className="rounded-xl border border-border/50 bg-card/95 backdrop-blur-lg px-4 py-3 shadow-2xl">
      <p className="text-xs text-muted-foreground mb-2 font-medium">{label}</p>
      <div className="space-y-1.5">
        {payload
          .filter((p: any) => p.value > 0)
          .sort((a: any, b: any) => b.value - a.value)
          .map((p: any) => (
            <div key={p.dataKey} className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: p.color }} />
                <span className="text-xs text-foreground capitalize">{p.dataKey}</span>
              </div>
              <span className="text-xs font-mono font-medium">{formatCost(p.value)}</span>
            </div>
          ))}
      </div>
      <div className="mt-2 pt-2 border-t border-border/30 flex justify-between">
        <span className="text-[10px] text-muted-foreground">Total</span>
        <span className="text-xs font-mono font-bold">{formatCost(total)}</span>
      </div>
    </div>
  )
}

export function AnalyticsAdmin({ days }: Props) {
  const [period, setPeriod] = useState('30d')
  const { data: dashboard, isLoading: dashLoading } = useAdminDashboard(period)

  // Keep legacy hooks as fallback
  const { data: legacyData, isLoading: legacyLoading } = useAdminWorkspaceAnalytics(days)
  const { data: legacyCostData, isLoading: legacyCostLoading } = useAdminCostAnalytics(period)

  // Expandable workspace rows
  const [expandedWs, setExpandedWs] = useState<Set<string>>(new Set())
  const toggleExpand = (id: string) => {
    setExpandedWs((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  // Sorting for workspace table
  const [wsSortField, setWsSortField] = useState<'cost' | 'requests' | 'agents' | 'name'>('cost')
  const [wsSortDir, setWsSortDir] = useState<'asc' | 'desc'>('desc')

  const toggleSort = (field: typeof wsSortField) => {
    if (wsSortField === field) {
      setWsSortDir((d) => (d === 'desc' ? 'asc' : 'desc'))
    } else {
      setWsSortField(field)
      setWsSortDir('desc')
    }
  }

  const sortedWorkspaces = useMemo(() => {
    if (!dashboard?.workspaces) return []
    return [...dashboard.workspaces].sort((a, b) => {
      const dir = wsSortDir === 'desc' ? -1 : 1
      if (wsSortField === 'name') return dir * a.name.localeCompare(b.name) * -1
      return dir * ((a[wsSortField] || 0) - (b[wsSortField] || 0))
    })
  }, [dashboard?.workspaces, wsSortField, wsSortDir])

  // Max cost for proportional bars
  const maxWsCost = useMemo(() => {
    if (!dashboard?.workspaces?.length) return 1
    return Math.max(...dashboard.workspaces.map((w) => w.cost), 0.001)
  }, [dashboard?.workspaces])

  const maxModelCost = useMemo(() => {
    if (!dashboard?.models?.length) return 1
    return Math.max(...dashboard.models.map((m) => m.cost), 0.001)
  }, [dashboard?.models])

  // CSV export
  const escapeCsvField = (value: string): string => {
    let escaped = value
    if (/^[=+\-@]/.test(escaped)) escaped = "'" + escaped
    return '"' + escaped.replace(/"/g, '""') + '"'
  }

  const handleExport = () => {
    const workspaces = dashboard?.workspaces || legacyData?.workspaces || []
    if (!workspaces.length) return

    const headers = ['Workspace,Plan,Agents,Recipes,Executions,Requests,Tokens,Cost']
    const rows = workspaces.map((ws: any) =>
      [
        escapeCsvField(ws.name),
        escapeCsvField(ws.plan || 'unknown'),
        ws.agents ?? 0,
        ws.recipes ?? 0,
        ws.executions ?? 0,
        ws.requests ?? ws.apiCalls ?? 0,
        ws.tokens ?? 0,
        `$${(ws.cost || 0).toFixed(4)}`,
      ].join(',')
    )
    const csv = [...headers, ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `admin-analytics-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Determine data source
  const isLoading = dashLoading && legacyLoading
  const hasDashboard = !!dashboard?.overview

  // BYOK donut chart data
  const byokDonutData = useMemo(() => {
    if (!dashboard?.byok_split) return []
    const { platform_cost, byok_cost } = dashboard.byok_split
    if (platform_cost === 0 && byok_cost === 0) return []
    return [
      { name: 'Platform', value: platform_cost, color: '#60B5FF' },
      { name: 'BYOK', value: byok_cost, color: '#72BF78' },
    ].filter((d) => d.value > 0)
  }, [dashboard?.byok_split])

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="glass-card">
              <CardContent className="p-5">
                <Skeleton className="h-4 w-20 mb-3" />
                <Skeleton className="h-8 w-24 mb-2" />
                <Skeleton className="h-3 w-16" />
              </CardContent>
            </Card>
          ))}
        </div>
        <Skeleton className="h-80 w-full rounded-2xl" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* ─── Admin Header ─── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-red-500/10 flex items-center justify-center">
            <Shield className="w-4 h-4 text-red-400" />
          </div>
          <div>
            <h2 className="text-sm font-semibold">Super Admin Dashboard</h2>
            <p className="text-xs text-muted-foreground">Platform-wide monitoring & revenue insights</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <PeriodToggle value={period} onChange={setPeriod} />
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* ─── Hero Stats ─── */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 md:gap-4">
        {[
          {
            label: 'Total Platform Cost',
            value: formatCost(dashboard?.overview?.total_cost || legacyCostData?.total_platform_cost || 0),
            sub: 'All workspaces',
            icon: DollarSign,
            iconBg: 'bg-emerald-500/10',
            iconColor: 'text-emerald-400',
            accent: 'border-l-emerald-500',
          },
          {
            label: 'Projected Monthly',
            value: formatCost(dashboard?.overview?.projected_monthly || 0),
            sub: 'Based on daily avg',
            icon: TrendingUp,
            iconBg: 'bg-blue-500/10',
            iconColor: 'text-blue-400',
            accent: 'border-l-blue-500',
          },
          {
            label: 'Workspaces',
            value: dashboard?.overview?.total_workspaces || legacyData?.platformSummary?.totalWorkspaces || 0,
            sub: 'Active tenants',
            icon: Building,
            iconBg: 'bg-purple-500/10',
            iconColor: 'text-purple-400',
            accent: 'border-l-purple-500',
          },
          {
            label: 'API Requests',
            value: formatNumber(dashboard?.overview?.total_requests || legacyData?.platformSummary?.totalApiCalls || 0),
            sub: 'This period',
            icon: Activity,
            iconBg: 'bg-orange-500/10',
            iconColor: 'text-orange-400',
            accent: 'border-l-orange-500',
          },
          {
            label: 'Total Tokens',
            value: formatNumber(dashboard?.overview?.total_tokens || legacyData?.platformSummary?.totalTokens || 0),
            sub: 'Input + Output',
            icon: Zap,
            iconBg: 'bg-cyan-500/10',
            iconColor: 'text-cyan-400',
            accent: 'border-l-cyan-500',
          },
          {
            label: 'Daily Average',
            value: formatCost(dashboard?.overview?.daily_average || 0),
            sub: 'Revenue baseline',
            icon: BarChart3,
            iconBg: 'bg-pink-500/10',
            iconColor: 'text-pink-400',
            accent: 'border-l-pink-500',
          },
        ].map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.06 }}
          >
            <Card className={`glass-card border-l-2 ${stat.accent} hover:border-l-4 transition-all duration-300`}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="space-y-1 min-w-0">
                    <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider truncate">{stat.label}</p>
                    <p className="text-xl font-bold leading-none">{stat.value}</p>
                    <p className="text-[10px] text-muted-foreground mt-1">{stat.sub}</p>
                  </div>
                  <div className={`w-8 h-8 rounded-lg ${stat.iconBg} flex items-center justify-center shrink-0`}>
                    <stat.icon className={`w-4 h-4 ${stat.iconColor}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* ─── Cost by Provider Chart + BYOK Split ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stacked Area Chart */}
        <motion.div
          className="lg:col-span-2"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.35 }}
        >
          <Card className="glass-card h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-400" />
                Daily Cost by Provider
              </CardTitle>
            </CardHeader>
            <CardContent>
              {dashboard?.daily_by_provider?.series?.length ? (
                <>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={dashboard.daily_by_provider.series} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                        <defs>
                          {dashboard.daily_by_provider.providers.map((prov, idx) => (
                            <linearGradient key={prov} id={`admin-grad-${idx}`} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor={PROVIDER_COLORS[idx % PROVIDER_COLORS.length]} stopOpacity={0.35} />
                              <stop offset="100%" stopColor={PROVIDER_COLORS[idx % PROVIDER_COLORS.length]} stopOpacity={0.02} />
                            </linearGradient>
                          ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.3} vertical={false} />
                        <XAxis
                          dataKey="date"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                          tickFormatter={(v: string) => {
                            const d = new Date(v)
                            return `${d.getMonth() + 1}/${d.getDate()}`
                          }}
                        />
                        <YAxis
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                          tickFormatter={(v: number) => formatCost(v)}
                          width={60}
                        />
                        <Tooltip content={<ProviderCostTooltip />} />
                        {dashboard.daily_by_provider.providers.map((prov, idx) => (
                          <Area
                            key={prov}
                            type="monotone"
                            dataKey={prov}
                            stackId="1"
                            stroke={PROVIDER_COLORS[idx % PROVIDER_COLORS.length]}
                            strokeWidth={2}
                            fill={`url(#admin-grad-${idx})`}
                            dot={false}
                            activeDot={{ r: 4, strokeWidth: 2, fill: 'hsl(var(--card))' }}
                          />
                        ))}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  {/* Legend */}
                  <div className="flex flex-wrap gap-3 mt-3 px-1">
                    {dashboard.daily_by_provider.providers.map((prov, idx) => (
                      <div key={prov} className="flex items-center gap-1.5">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ background: PROVIDER_COLORS[idx % PROVIDER_COLORS.length] }}
                        />
                        <span className="text-xs text-muted-foreground capitalize">{prov}</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : legacyCostData?.daily_cost_trend?.length ? (
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={legacyCostData.daily_cost_trend}>
                      <defs>
                        <linearGradient id="adminCostGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#60B5FF" stopOpacity={0.35} />
                          <stop offset="100%" stopColor="#60B5FF" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.3} vertical={false} />
                      <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                      <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} tickFormatter={(v: number) => formatCost(v)} width={60} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '12px', fontSize: '12px' }}
                        formatter={(value: number) => [formatCost(value), 'Daily Cost']}
                      />
                      <Area type="monotone" dataKey="cost" stroke="#60B5FF" strokeWidth={2} fill="url(#adminCostGrad)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-72 flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="w-12 h-12 mx-auto mb-3 text-muted-foreground/30" />
                    <p className="text-sm text-muted-foreground">No cost trend data yet</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* BYOK vs Platform Split */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Card className="glass-card h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="w-5 h-5 text-amber-400" />
                Cost Source Split
              </CardTitle>
            </CardHeader>
            <CardContent>
              {byokDonutData.length > 0 ? (
                <div className="space-y-4">
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={byokDonutData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={75}
                          paddingAngle={4}
                          strokeWidth={0}
                        >
                          {byokDonutData.map((d, i) => (
                            <Cell key={i} fill={d.color} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '12px', fontSize: '12px' }}
                          formatter={(value: number) => [formatCost(value), 'Cost']}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  {/* Legend / breakdown */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-[#60B5FF]" />
                        <span className="text-xs text-foreground">Platform Keys</span>
                      </div>
                      <div className="text-right">
                        <span className="text-xs font-mono font-medium">{formatCost(dashboard?.byok_split?.platform_cost || 0)}</span>
                        <span className="text-[10px] text-muted-foreground ml-2">{formatNumber(dashboard?.byok_split?.platform_requests || 0)} req</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-[#72BF78]" />
                        <span className="text-xs text-foreground">User BYOK</span>
                      </div>
                      <div className="text-right">
                        <span className="text-xs font-mono font-medium">{formatCost(dashboard?.byok_split?.byok_cost || 0)}</span>
                        <span className="text-[10px] text-muted-foreground ml-2">{formatNumber(dashboard?.byok_split?.byok_requests || 0)} req</span>
                      </div>
                    </div>
                  </div>
                  {/* Revenue insight */}
                  {(dashboard?.byok_split?.platform_cost || 0) > 0 && (
                    <div className="rounded-lg bg-emerald-500/5 border border-emerald-500/20 p-3">
                      <p className="text-[10px] text-emerald-400 uppercase tracking-wider font-medium mb-1">Your Revenue Exposure</p>
                      <p className="text-lg font-bold text-emerald-400">{formatCost(dashboard?.byok_split?.platform_cost || 0)}</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        This is what you pay for users on platform keys. BYOK users ({formatCost(dashboard?.byok_split?.byok_cost || 0)}) use their own keys.
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-48 flex items-center justify-center">
                  <div className="text-center">
                    <PieChartIcon className="w-10 h-10 mx-auto mb-3 text-muted-foreground/30" />
                    <p className="text-sm text-muted-foreground">No BYOK split data</p>
                    <p className="text-xs text-muted-foreground/60 mt-1">Usage data will appear here</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* ─── Workspace Drill-Down Table ─── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.45 }}
      >
        <Card className="glass-card overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Building className="w-5 h-5 text-blue-400" />
                Workspace Deep Dive
              </span>
              <span className="text-xs text-muted-foreground font-normal">
                {sortedWorkspaces.length} workspace{sortedWorkspaces.length !== 1 ? 's' : ''}
              </span>
            </CardTitle>
          </CardHeader>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border/50">
                  <th className="w-8 p-4" />
                  <SortHeader field="name" label="Workspace" current={wsSortField} dir={wsSortDir} onToggle={toggleSort} />
                  <th className="text-left p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Plan</th>
                  <SortHeader field="agents" label="Agents" current={wsSortField} dir={wsSortDir} onToggle={toggleSort} align="right" />
                  <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden md:table-cell">Recipes</th>
                  <SortHeader field="requests" label="Requests" current={wsSortField} dir={wsSortDir} onToggle={toggleSort} align="right" className="hidden md:table-cell" />
                  <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden lg:table-cell">Tokens</th>
                  <SortHeader field="cost" label="Cost" current={wsSortField} dir={wsSortDir} onToggle={toggleSort} align="right" />
                  <th className="p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden lg:table-cell w-36">Share</th>
                </tr>
              </thead>
              <tbody>
                {sortedWorkspaces.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="p-12 text-center text-muted-foreground">
                      <Building className="w-10 h-10 mx-auto mb-3 opacity-30" />
                      <p className="text-sm">No workspace data available</p>
                    </td>
                  </tr>
                ) : (
                  sortedWorkspaces.map((ws) => (
                    <Fragment key={ws.id}>
                      <tr
                        className="border-b border-border/20 hover:bg-secondary/10 transition-colors cursor-pointer"
                        onClick={() => toggleExpand(ws.id)}
                      >
                        <td className="p-4 w-8">
                          <motion.div
                            animate={{ rotate: expandedWs.has(ws.id) ? 90 : 0 }}
                            transition={{ duration: 0.2 }}
                          >
                            <ChevronRight className="w-4 h-4 text-muted-foreground" />
                          </motion.div>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-sm">{ws.name}</span>
                            {ws.is_personal && (
                              <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-blue-400/30 text-blue-400">personal</Badge>
                            )}
                          </div>
                        </td>
                        <td className="p-4">
                          <Badge variant="outline" className={`text-[10px] uppercase ${PLAN_COLORS[ws.plan] || PLAN_COLORS.starter}`}>
                            {ws.plan}
                          </Badge>
                        </td>
                        <td className="p-4 text-sm text-right tabular-nums">{ws.agents}</td>
                        <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell">{ws.recipes}</td>
                        <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell">{formatNumber(ws.requests)}</td>
                        <td className="p-4 text-sm text-right tabular-nums hidden lg:table-cell text-muted-foreground">{formatNumber(ws.tokens)}</td>
                        <td className="p-4 text-sm text-right tabular-nums font-medium">{formatCost(ws.cost)}</td>
                        <td className="p-4 hidden lg:table-cell">
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 flex-1 rounded-full bg-secondary/30 overflow-hidden">
                              <motion.div
                                className="h-full rounded-full bg-blue-400"
                                initial={{ width: 0 }}
                                animate={{ width: `${(ws.cost / maxWsCost) * 100}%` }}
                                transition={{ duration: 0.6, delay: 0.1 }}
                              />
                            </div>
                            <span className="text-[10px] text-muted-foreground w-8 text-right tabular-nums">
                              {(dashboard?.overview?.total_cost || 0) > 0
                                ? `${((ws.cost / (dashboard?.overview?.total_cost || 1)) * 100).toFixed(0)}%`
                                : '0%'}
                            </span>
                          </div>
                        </td>
                      </tr>
                      {/* Expanded row */}
                      <AnimatePresence>
                        {expandedWs.has(ws.id) && (
                          <tr>
                            <td colSpan={9} className="p-0">
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.3 }}
                                className="overflow-hidden"
                              >
                                <div className="px-6 py-4 bg-secondary/5 border-b border-border/20">
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div className="rounded-lg bg-secondary/20 border border-border/20 p-3">
                                      <div className="flex items-center gap-2 mb-1">
                                        <Bot className="w-3.5 h-3.5 text-blue-400" />
                                        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Agents</span>
                                      </div>
                                      <p className="text-lg font-bold">{ws.agents}</p>
                                    </div>
                                    <div className="rounded-lg bg-secondary/20 border border-border/20 p-3">
                                      <div className="flex items-center gap-2 mb-1">
                                        <ChefHat className="w-3.5 h-3.5 text-orange-400" />
                                        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Recipes</span>
                                      </div>
                                      <p className="text-lg font-bold">{ws.recipes}</p>
                                    </div>
                                    <div className="rounded-lg bg-secondary/20 border border-border/20 p-3">
                                      <div className="flex items-center gap-2 mb-1">
                                        <Play className="w-3.5 h-3.5 text-green-400" />
                                        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Executions</span>
                                      </div>
                                      <p className="text-lg font-bold">{ws.executions}</p>
                                    </div>
                                    <div className="rounded-lg bg-secondary/20 border border-border/20 p-3">
                                      <div className="flex items-center gap-2 mb-1">
                                        <Zap className="w-3.5 h-3.5 text-cyan-400" />
                                        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Tokens</span>
                                      </div>
                                      <p className="text-lg font-bold">{formatNumber(ws.tokens)}</p>
                                    </div>
                                  </div>
                                  {/* Revenue insight per workspace */}
                                  <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                                    <span>Created: {ws.created_at ? new Date(ws.created_at).toLocaleDateString() : 'N/A'}</span>
                                    <span>Cost/Request: {ws.requests > 0 ? formatCost(ws.cost / ws.requests) : '--'}</span>
                                    <span>Tokens/Request: {ws.requests > 0 ? formatNumber(ws.tokens / ws.requests) : '--'}</span>
                                  </div>
                                </div>
                              </motion.div>
                            </td>
                          </tr>
                        )}
                      </AnimatePresence>
                    </Fragment>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </motion.div>

      {/* ─── Top Models Platform-Wide ─── */}
      {dashboard?.models && dashboard.models.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <Card className="glass-card overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="w-5 h-5 text-purple-400" />
                Platform Model Usage
              </CardTitle>
            </CardHeader>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border/50">
                    <th className="text-left p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Model</th>
                    <th className="text-left p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Provider</th>
                    <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Requests</th>
                    <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden md:table-cell">Tokens</th>
                    <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Cost</th>
                    <th className="p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden lg:table-cell w-36">Share</th>
                    <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden md:table-cell">Workspaces</th>
                  </tr>
                </thead>
                <tbody>
                  {dashboard.models.map((model, idx) => (
                    <tr key={model.model_id} className="border-b border-border/20 hover:bg-secondary/10 transition-colors">
                      <td className="p-4">
                        <div className="flex items-center gap-2.5">
                          <span
                            className="w-2.5 h-2.5 rounded-full shrink-0"
                            style={{ background: PROVIDER_COLORS[idx % PROVIDER_COLORS.length] }}
                          />
                          <Badge variant="secondary" className="font-mono text-xs">{shortenModelName(model.model_id)}</Badge>
                        </div>
                      </td>
                      <td className="p-4 text-xs text-muted-foreground capitalize">{model.provider}</td>
                      <td className="p-4 text-sm text-right tabular-nums">{formatNumber(model.requests)}</td>
                      <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell text-muted-foreground">{formatNumber(model.tokens)}</td>
                      <td className="p-4 text-sm text-right tabular-nums font-medium">{formatCost(model.cost)}</td>
                      <td className="p-4 hidden lg:table-cell">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 rounded-full bg-secondary/30 overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${(model.cost / maxModelCost) * 100}%`,
                                background: PROVIDER_COLORS[idx % PROVIDER_COLORS.length],
                              }}
                            />
                          </div>
                          <span className="text-[10px] text-muted-foreground w-8 text-right tabular-nums">
                            {(dashboard.overview?.total_cost || 0) > 0
                              ? `${((model.cost / (dashboard.overview.total_cost || 1)) * 100).toFixed(0)}%`
                              : '0%'}
                          </span>
                        </div>
                      </td>
                      <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell">
                        <div className="flex items-center justify-end gap-1.5">
                          <Building className="w-3 h-3 text-muted-foreground" />
                          <span className="text-muted-foreground">{model.workspace_count}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </motion.div>
      )}

      {/* ─── Provider Cost Breakdown (Bar Chart) ─── */}
      {legacyCostData?.cost_by_provider && legacyCostData.cost_by_provider.length > 0 && !dashboard?.daily_by_provider?.series?.length && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.55 }}
        >
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="w-5 h-5 text-purple-400" />
                Cost by Provider
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={legacyCostData.cost_by_provider.map((p) => ({
                      name: p.key,
                      cost: p.total_cost,
                      requests: p.request_count,
                    }))}
                    margin={{ top: 5, right: 10, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.3} vertical={false} />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} tickFormatter={(v: number) => formatCost(v)} width={60} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '12px', fontSize: '12px' }}
                      formatter={(value: number, name: string) => [formatCost(value), name === 'cost' ? 'Total Cost' : name]}
                    />
                    <Bar dataKey="cost" radius={[6, 6, 0, 0]}>
                      {legacyCostData.cost_by_provider.map((_, index) => (
                        <Cell key={index} fill={PROVIDER_COLORS[index % PROVIDER_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  )
}

// ─── Sortable Table Header ────────────────────────────────────────────

function SortHeader({
  field,
  label,
  current,
  dir,
  onToggle,
  align = 'left',
  className = '',
}: {
  field: string
  label: string
  current: string
  dir: 'asc' | 'desc'
  onToggle: (field: any) => void
  align?: 'left' | 'right'
  className?: string
}) {
  return (
    <th
      className={`p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:text-foreground transition-colors select-none ${
        align === 'right' ? 'text-right' : 'text-left'
      } ${className}`}
      onClick={() => onToggle(field)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {current === field && (
          <ChevronDown className={`w-3 h-3 transition-transform ${dir === 'asc' ? 'rotate-180' : ''}`} />
        )}
      </span>
    </th>
  )
}
