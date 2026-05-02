'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  Shield,
  DollarSign,
  Zap,
  Activity,
  Download,
  Building,
  TrendingUp,
  ChevronDown,
  Key,
  BarChart3,
  PieChart as PieChartIcon,
  AlertTriangle,
  CreditCard,
  Users,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
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
  CartesianGrid,
} from 'recharts'
import {
  useAdminDashboard,
  useAdminWorkspaceAnalytics,
  useAdminCostAnalytics,
} from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

// ─── Constants ────────────────────────────────────────────────────────

const PROVIDER_COLORS = [
  '#60B5FF', '#72BF78', '#ff6b35', '#a78bfa', '#f472b6',
  '#fbbf24', '#34d399', '#f87171', '#38bdf8', '#c084fc',
]

const PLAN_COLORS: Record<string, string> = {
  starter: 'text-muted-foreground border-border/30 bg-secondary/30',
  pilot: 'text-info border-info/30 bg-info/5',
  pro: 'text-agent border-agent/30 bg-agent/5',
  enterprise: 'text-warning border-warning/30 bg-warning/5',
}

const PLAN_DONUT_COLORS: Record<string, string> = {
  starter: '#9ca3af',
  pilot: '#60a5fa',
  pro: '#a78bfa',
  enterprise: '#fbbf24',
}

// ─── Helpers ──────────────────────────────────────────────────────────

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

// ─── Period Toggle ────────────────────────────────────────────────────

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

// ─── Custom Tooltips ──────────────────────────────────────────────────

function ProviderCostTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  const total = payload.reduce((s: number, p: any) => s + (p.value || 0), 0)
  return (
    <div className="rounded-xl border border-border/50 bg-card/95 backdrop-blur-lg px-4 py-3 shadow-2xl">
      <p className="text-xs text-muted-foreground mb-2 font-medium">{label}</p>
      <div className="space-y-1.5">
        {payload.filter((p: any) => p.value > 0).sort((a: any, b: any) => b.value - a.value).map((p: any) => (
          <div key={p.dataKey} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: p.color }} />
              <span className="text-xs text-foreground capitalize">{p.dataKey}</span>
            </div>
            <span className="text-xs font-mono font-medium">{formatCost(p.value)}</span>
          </div>
        ))}
      </div>
      {total > 0 && (
        <div className="mt-2 pt-2 border-t border-border/30 flex justify-between">
          <span className="text-[10px] text-muted-foreground">Total</span>
          <span className="text-xs font-mono font-bold">{formatCost(total)}</span>
        </div>
      )}
    </div>
  )
}

// ─── Sortable Header ──────────────────────────────────────────────────

function SortHeader({
  field, label, current, dir, onToggle, align = 'left', className = '',
}: {
  field: string; label: string; current: string; dir: 'asc' | 'desc'
  onToggle: (f: any) => void; align?: 'left' | 'right'; className?: string
}) {
  return (
    <th
      className={`p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:text-foreground transition-colors select-none ${align === 'right' ? 'text-right' : 'text-left'} ${className}`}
      onClick={() => onToggle(field)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {current === field && <ChevronDown className={`w-3 h-3 transition-transform ${dir === 'asc' ? 'rotate-180' : ''}`} />}
      </span>
    </th>
  )
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════

export function AnalyticsAdmin({ days }: Props) {
  const [period, setPeriod] = useState('30d')

  // ─── Data hooks ─────────────────────────────────────────────────────
  const { data: dashboard, isLoading: dashLoading } = useAdminDashboard(period)
  const { data: legacyData, isLoading: legacyLoading } = useAdminWorkspaceAnalytics(days)
  const { data: legacyCostData } = useAdminCostAnalytics(period)

  // ─── Top spenders sorting ──────────────────────────────────────────
  const [spenderSort, setSpenderSort] = useState<'cost' | 'requests' | 'agents' | 'name'>('cost')
  const [spenderDir, setSpenderDir] = useState<'asc' | 'desc'>('desc')
  const toggleSpenderSort = (field: typeof spenderSort) => {
    if (spenderSort === field) setSpenderDir((d) => (d === 'desc' ? 'asc' : 'desc'))
    else { setSpenderSort(field); setSpenderDir('desc') }
  }

  // ─── Computed data ──────────────────────────────────────────────────

  // Merge workspace data from dashboard or legacy
  const allWorkspaces = useMemo(() => {
    if (dashboard?.workspaces?.length) return dashboard.workspaces
    if (legacyData?.workspaces?.length) {
      return legacyData.workspaces.map((ws: any) => ({
        id: ws.id, name: ws.name, plan: ws.plan || 'pilot',
        is_personal: false, created_at: null,
        agents: ws.agents || 0, recipes: 0, executions: 0,
        cost: ws.cost || 0, tokens: ws.tokens || 0, requests: ws.apiCalls || 0,
      }))
    }
    return []
  }, [dashboard?.workspaces, legacyData?.workspaces])

  const totalCost = dashboard?.overview?.total_cost || legacyCostData?.total_platform_cost || 0

  // Plan distribution
  const planDistribution = useMemo(() => {
    const counts: Record<string, number> = {}
    allWorkspaces.forEach((ws) => {
      const plan = (ws.plan || 'starter').toLowerCase()
      counts[plan] = (counts[plan] || 0) + 1
    })
    return Object.entries(counts)
      .map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
        color: PLAN_DONUT_COLORS[name] || '#9ca3af',
      }))
      .sort((a, b) => b.value - a.value)
  }, [allWorkspaces])

  // Revenue per workspace (sorted by cost)
  const sortedSpenders = useMemo(() => {
    return [...allWorkspaces].sort((a, b) => {
      const dir = spenderDir === 'desc' ? -1 : 1
      if (spenderSort === 'name') return dir * a.name.localeCompare(b.name) * -1
      return dir * ((a[spenderSort] || 0) - (b[spenderSort] || 0))
    })
  }, [allWorkspaces, spenderSort, spenderDir])

  // Cost anomalies: workspaces with cost > 2x the average
  const costAnomalies = useMemo(() => {
    if (allWorkspaces.length < 2) return []
    const avgCost = allWorkspaces.reduce((s, w) => s + w.cost, 0) / allWorkspaces.length
    if (avgCost <= 0) return []
    return allWorkspaces
      .filter((ws) => ws.cost > avgCost * 2)
      .sort((a, b) => b.cost - a.cost)
      .map((ws) => ({
        ...ws,
        ratio: ws.cost / avgCost,
      }))
  }, [allWorkspaces])

  const maxSpenderCost = useMemo(() => Math.max(...allWorkspaces.map((w) => w.cost), 0.001), [allWorkspaces])

  // MRR projection: simple estimate from daily average
  const dailyAvg = dashboard?.overview?.daily_average || 0
  const projectedMrr = dashboard?.overview?.projected_monthly || dailyAvg * 30

  // ─── CSV Export ─────────────────────────────────────────────────────
  const escapeCsv = (v: string) => {
    let s = v; if (/^[=+\-@]/.test(s)) s = "'" + s
    return '"' + s.replace(/"/g, '""') + '"'
  }
  const handleExport = () => {
    if (!allWorkspaces.length) return
    const headers = ['Workspace,Plan,Agents,Requests,Tokens,Cost']
    const rows = allWorkspaces.map((ws) =>
      [escapeCsv(ws.name), escapeCsv(ws.plan || 'unknown'), ws.agents, ws.requests, ws.tokens, `$${ws.cost.toFixed(4)}`].join(',')
    )
    const blob = new Blob([[...headers, ...rows].join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url
    a.download = `admin-billing-${new Date().toISOString().split('T')[0]}.csv`
    a.click(); URL.revokeObjectURL(url)
  }

  // ─── Loading ────────────────────────────────────────────────────────
  if (dashLoading && legacyLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="glass-card"><CardContent className="p-4"><Skeleton className="h-4 w-20 mb-3" /><Skeleton className="h-7 w-24 mb-2" /><Skeleton className="h-3 w-16" /></CardContent></Card>
          ))}
        </div>
        <Skeleton className="h-80 w-full rounded-2xl" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* ═══ HEADER ═══ */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-red-500/10 flex items-center justify-center">
            <Shield className="w-4 h-4 text-red-400" />
          </div>
          <div>
            <h2 className="text-sm font-semibold">Billing & Revenue</h2>
            <p className="text-xs text-muted-foreground">Plans, costs, and revenue management</p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <PeriodToggle value={period} onChange={setPeriod} />
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="w-4 h-4 mr-2" />Export
          </Button>
        </div>
      </div>

      {/* ═══ HERO STATS ═══ */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 md:gap-4">
        {[
          { label: 'Total Revenue', value: formatCost(totalCost), sub: 'Platform cost this period', icon: DollarSign, iconBg: 'bg-emerald-500/10', iconColor: 'text-emerald-400', accent: 'border-l-emerald-500' },
          { label: 'MRR Projection', value: formatCost(projectedMrr), sub: `${formatCost(dailyAvg)}/day avg`, icon: TrendingUp, iconBg: 'bg-blue-500/10', iconColor: 'text-blue-400', accent: 'border-l-blue-500' },
          { label: 'Workspaces', value: dashboard?.overview?.total_workspaces || legacyData?.platformSummary?.totalWorkspaces || 0, sub: 'Active tenants', icon: Building, iconBg: 'bg-purple-500/10', iconColor: 'text-purple-400', accent: 'border-l-purple-500' },
          { label: 'API Requests', value: formatNumber(dashboard?.overview?.total_requests || legacyData?.platformSummary?.totalApiCalls || 0), sub: 'This period', icon: Activity, iconBg: 'bg-orange-500/10', iconColor: 'text-orange-400', accent: 'border-l-orange-500' },
          { label: 'Total Tokens', value: formatNumber(dashboard?.overview?.total_tokens || legacyData?.platformSummary?.totalTokens || 0), sub: 'Input + Output', icon: Zap, iconBg: 'bg-cyan-500/10', iconColor: 'text-cyan-400', accent: 'border-l-cyan-500' },
          { label: 'BYOK Savings', value: formatCost(dashboard?.byok_split?.byok_cost || 0), sub: 'User-provided keys', icon: Key, iconBg: 'bg-amber-500/10', iconColor: 'text-amber-400', accent: 'border-l-amber-500' },
        ].map((stat, index) => (
          <motion.div key={stat.label} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: index * 0.06 }}>
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

      {/* ═══ COST BY PROVIDER CHART + BYOK SPLIT ═══ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stacked Area Chart */}
        <motion.div className="lg:col-span-2" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.35 }}>
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
                            <linearGradient key={prov} id={`adm-g-${idx}`} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor={PROVIDER_COLORS[idx % PROVIDER_COLORS.length]} stopOpacity={0.35} />
                              <stop offset="100%" stopColor={PROVIDER_COLORS[idx % PROVIDER_COLORS.length]} stopOpacity={0.02} />
                            </linearGradient>
                          ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.3} vertical={false} />
                        <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} tickFormatter={(v: string) => { const d = new Date(v); return `${d.getMonth() + 1}/${d.getDate()}` }} />
                        <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} tickFormatter={(v: number) => formatCost(v)} width={60} />
                        <Tooltip content={<ProviderCostTooltip />} />
                        {dashboard.daily_by_provider.providers.map((prov, idx) => (
                          <Area key={prov} type="monotone" dataKey={prov} stackId="1" stroke={PROVIDER_COLORS[idx % PROVIDER_COLORS.length]} strokeWidth={2} fill={`url(#adm-g-${idx})`} dot={false} activeDot={{ r: 4, strokeWidth: 2, fill: 'hsl(var(--card))' }} />
                        ))}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex flex-wrap gap-3 mt-3 px-1">
                    {dashboard.daily_by_provider.providers.map((prov, idx) => (
                      <div key={prov} className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ background: PROVIDER_COLORS[idx % PROVIDER_COLORS.length] }} />
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
                        <linearGradient id="admCostGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#60B5FF" stopOpacity={0.35} />
                          <stop offset="100%" stopColor="#60B5FF" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.3} vertical={false} />
                      <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                      <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} tickFormatter={(v: number) => formatCost(v)} width={60} />
                      <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '12px', fontSize: '12px' }} formatter={(value: number) => [formatCost(value), 'Daily Cost']} />
                      <Area type="monotone" dataKey="cost" stroke="#60B5FF" strokeWidth={2} fill="url(#admCostGrad)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-72 flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="w-12 h-12 mx-auto mb-3 text-muted-foreground/30" />
                    <p className="text-sm text-muted-foreground">No cost trend data yet</p>
                    <p className="text-xs text-muted-foreground/60 mt-1">LLM usage will generate cost data</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* BYOK vs Platform Split */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.4 }}>
          <Card className="glass-card h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="w-5 h-5 text-amber-400" />
                Cost Source Split
              </CardTitle>
            </CardHeader>
            <CardContent>
              {(() => {
                const byok = dashboard?.byok_split
                const platformCost = byok?.platform_cost || 0
                const byokCost = byok?.byok_cost || 0
                const hasSplit = platformCost > 0 || byokCost > 0
                const donutData = [
                  { name: 'Platform', value: platformCost, color: '#60B5FF' },
                  { name: 'BYOK', value: byokCost, color: '#72BF78' },
                ].filter((d) => d.value > 0)

                if (!hasSplit) return (
                  <div className="h-48 flex items-center justify-center">
                    <div className="text-center">
                      <PieChartIcon className="w-10 h-10 mx-auto mb-3 text-muted-foreground/30" />
                      <p className="text-sm text-muted-foreground">No BYOK split data</p>
                      <p className="text-xs text-muted-foreground/60 mt-1">Usage with API keys will show here</p>
                    </div>
                  </div>
                )

                return (
                  <div className="space-y-4">
                    <div className="h-44">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={donutData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={70} paddingAngle={4} strokeWidth={0}>
                            {donutData.map((d, i) => <Cell key={i} fill={d.color} />)}
                          </Pie>
                          <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '12px', fontSize: '12px' }} formatter={(value: number) => [formatCost(value), 'Cost']} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="space-y-2.5">
                      {[
                        { label: 'Platform Keys', color: '#60B5FF', cost: platformCost, req: byok?.platform_requests || 0 },
                        { label: 'User BYOK', color: '#72BF78', cost: byokCost, req: byok?.byok_requests || 0 },
                      ].map((row) => (
                        <div key={row.label} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="w-3 h-3 rounded-full" style={{ background: row.color }} />
                            <span className="text-xs">{row.label}</span>
                          </div>
                          <div className="text-right">
                            <span className="text-xs font-mono font-medium">{formatCost(row.cost)}</span>
                            <span className="text-[10px] text-muted-foreground ml-2">{formatNumber(row.req)} req</span>
                          </div>
                        </div>
                      ))}
                    </div>
                    {platformCost > 0 && (
                      <div className="rounded-lg bg-emerald-500/5 border border-emerald-500/20 p-3">
                        <p className="text-[10px] text-emerald-400 uppercase tracking-wider font-medium mb-1">Your Revenue Exposure</p>
                        <p className="text-lg font-bold text-emerald-400">{formatCost(platformCost)}</p>
                        <p className="text-[10px] text-muted-foreground mt-0.5">
                          Cost on platform keys. BYOK users ({formatCost(byokCost)}) use their own.
                        </p>
                      </div>
                    )}
                  </div>
                )
              })()}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* ═══ BILLING OVERVIEW + PLAN DISTRIBUTION ═══ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Billing Overview */}
        <motion.div className="lg:col-span-2" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.45 }}>
          <Card className="glass-card h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CreditCard className="w-5 h-5 text-emerald-400" />
                Billing Overview
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Revenue summary cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[
                    { label: 'Platform Revenue', value: formatCost(dashboard?.byok_split?.platform_cost || totalCost), color: 'text-emerald-400' },
                    { label: 'Avg per Workspace', value: allWorkspaces.length > 0 ? formatCost(totalCost / allWorkspaces.length) : '$0', color: 'text-blue-400' },
                    { label: 'Cost per Request', value: (dashboard?.overview?.total_requests || 0) > 0 ? formatCost(totalCost / (dashboard?.overview?.total_requests || 1)) : '$0', color: 'text-purple-400' },
                    { label: 'Daily Run Rate', value: formatCost(dailyAvg), color: 'text-orange-400' },
                  ].map((item) => (
                    <div key={item.label} className="rounded-lg bg-secondary/20 border border-border/20 p-3">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">{item.label}</p>
                      <p className={`text-lg font-bold ${item.color}`}>{item.value}</p>
                    </div>
                  ))}
                </div>

                {/* Revenue per workspace */}
                {allWorkspaces.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">Revenue by Workspace</p>
                    <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                      {[...allWorkspaces].sort((a, b) => b.cost - a.cost).slice(0, 10).map((ws) => (
                        <div key={ws.id} className="flex items-center gap-3">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-medium truncate">{ws.name}</span>
                              <span className="text-xs font-mono font-medium ml-2">{formatCost(ws.cost)}</span>
                            </div>
                            <div className="h-1.5 w-full rounded-full bg-secondary/30 overflow-hidden">
                              <motion.div
                                className="h-full rounded-full bg-emerald-400"
                                initial={{ width: 0 }}
                                animate={{ width: `${(ws.cost / maxSpenderCost) * 100}%` }}
                                transition={{ duration: 0.6 }}
                              />
                            </div>
                          </div>
                          <Badge variant="outline" className={`text-[10px] uppercase shrink-0 ${PLAN_COLORS[(ws.plan || 'starter').toLowerCase()] || PLAN_COLORS.starter}`}>
                            {ws.plan}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Plan Distribution Donut */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.5 }}>
          <Card className="glass-card h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="w-5 h-5 text-purple-400" />
                Plan Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              {planDistribution.length === 0 ? (
                <div className="h-48 flex items-center justify-center">
                  <div className="text-center">
                    <PieChartIcon className="w-10 h-10 mx-auto mb-3 text-muted-foreground/30" />
                    <p className="text-sm text-muted-foreground">No plan data</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="h-44">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={planDistribution}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={45}
                          outerRadius={70}
                          paddingAngle={4}
                          strokeWidth={0}
                        >
                          {planDistribution.map((d, i) => <Cell key={i} fill={d.color} />)}
                        </Pie>
                        <Tooltip
                          contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '12px', fontSize: '12px' }}
                          formatter={(value: number, name: string) => [`${value} workspace${value !== 1 ? 's' : ''}`, name]}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="space-y-2">
                    {planDistribution.map((plan) => (
                      <div key={plan.name} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ background: plan.color }} />
                          <span className="text-xs font-medium">{plan.name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono font-medium">{plan.value}</span>
                          <span className="text-[10px] text-muted-foreground">
                            ({allWorkspaces.length > 0 ? ((plan.value / allWorkspaces.length) * 100).toFixed(0) : 0}%)
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* ═══ COST ALERTS / ANOMALIES ═══ */}
      {costAnomalies.length > 0 && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.55 }}>
          <Card className="glass-card border-l-2 border-l-amber-500">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
                Cost Anomalies
                <Badge variant="outline" className="text-[10px] text-amber-400 border-amber-400/30 ml-2">
                  {costAnomalies.length} alert{costAnomalies.length !== 1 ? 's' : ''}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground mb-3">
                Workspaces spending more than 2x the average ({formatCost(totalCost / Math.max(allWorkspaces.length, 1))}/workspace)
              </p>
              <div className="space-y-2">
                {costAnomalies.map((ws) => (
                  <div key={ws.id} className="flex items-center justify-between rounded-lg bg-amber-500/5 border border-amber-500/20 p-3">
                    <div className="flex items-center gap-3 min-w-0">
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                      <div className="min-w-0">
                        <span className="text-sm font-medium truncate block">{ws.name}</span>
                        <span className="text-[10px] text-muted-foreground">{ws.ratio.toFixed(1)}x avg cost</span>
                      </div>
                    </div>
                    <div className="text-right shrink-0 ml-3">
                      <span className="text-sm font-mono font-bold text-amber-400">{formatCost(ws.cost)}</span>
                      <div className="flex items-center gap-1 justify-end">
                        <Badge variant="outline" className={`text-[10px] uppercase ${PLAN_COLORS[(ws.plan || 'starter').toLowerCase()] || PLAN_COLORS.starter}`}>
                          {ws.plan}
                        </Badge>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* ═══ TOP SPENDERS TABLE ═══ */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.6 }}>
        <Card className="glass-card overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Building className="w-5 h-5 text-blue-400" />
                Top Spenders
              </span>
              <span className="text-xs text-muted-foreground font-normal">
                {allWorkspaces.length} workspace{allWorkspaces.length !== 1 ? 's' : ''}
              </span>
            </CardTitle>
          </CardHeader>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border/50">
                  <SortHeader field="name" label="Workspace" current={spenderSort} dir={spenderDir} onToggle={toggleSpenderSort} />
                  <th className="text-left p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Plan</th>
                  <SortHeader field="agents" label="Agents" current={spenderSort} dir={spenderDir} onToggle={toggleSpenderSort} align="right" />
                  <SortHeader field="requests" label="Requests" current={spenderSort} dir={spenderDir} onToggle={toggleSpenderSort} align="right" className="hidden md:table-cell" />
                  <SortHeader field="cost" label="Cost" current={spenderSort} dir={spenderDir} onToggle={toggleSpenderSort} align="right" />
                  <th className="p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden lg:table-cell w-36">Share</th>
                </tr>
              </thead>
              <tbody>
                {sortedSpenders.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="p-12 text-center text-muted-foreground">
                      <Building className="w-10 h-10 mx-auto mb-3 opacity-30" />
                      <p className="text-sm">No workspace data available</p>
                    </td>
                  </tr>
                ) : sortedSpenders.map((ws) => (
                  <tr key={ws.id} className="border-b border-border/20 hover:bg-secondary/10 transition-colors">
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm">{ws.name}</span>
                        {ws.is_personal && <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-blue-400/30 text-blue-400">personal</Badge>}
                      </div>
                    </td>
                    <td className="p-4">
                      <Badge variant="outline" className={`text-[10px] uppercase ${PLAN_COLORS[(ws.plan || 'starter').toLowerCase()] || PLAN_COLORS.starter}`}>{ws.plan}</Badge>
                    </td>
                    <td className="p-4 text-sm text-right tabular-nums">{ws.agents}</td>
                    <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell">{formatNumber(ws.requests)}</td>
                    <td className="p-4 text-sm text-right tabular-nums font-medium">{formatCost(ws.cost)}</td>
                    <td className="p-4 hidden lg:table-cell">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 flex-1 rounded-full bg-secondary/30 overflow-hidden">
                          <motion.div className="h-full rounded-full bg-blue-400" initial={{ width: 0 }} animate={{ width: `${(ws.cost / maxSpenderCost) * 100}%` }} transition={{ duration: 0.6 }} />
                        </div>
                        <span className="text-[10px] text-muted-foreground w-8 text-right tabular-nums">
                          {totalCost > 0 ? `${((ws.cost / totalCost) * 100).toFixed(0)}%` : '0%'}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}
