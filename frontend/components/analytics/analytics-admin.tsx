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
  Layers,
  BarChart3,
  PieChart as PieChartIcon,
  Search,
  Filter,
  AppWindow,
  Wrench,
  Link2,
  FileText,
  X,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
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
} from 'recharts'
import {
  useAdminDashboard,
  useAdminWorkspaceAnalytics,
  useAdminCostAnalytics,
  useComposioApps,
  useComposioActions,
  useComposioAgentTools,
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
  starter: 'text-gray-400 border-gray-400/30 bg-gray-400/5',
  pilot: 'text-blue-400 border-blue-400/30 bg-blue-400/5',
  pro: 'text-purple-400 border-purple-400/30 bg-purple-400/5',
  enterprise: 'text-amber-400 border-amber-400/30 bg-amber-400/5',
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

function shortenModelName(name: string): string {
  const parts = name.split('/')
  return parts[parts.length - 1]
}

function formatAppName(name: string): string {
  const known: Record<string, string> = {
    GOOGLEDRIVE: 'Google Drive', GOOGLECALENDAR: 'Google Calendar',
    GOOGLEDOCS: 'Google Docs', GOOGLESHEETS: 'Google Sheets',
    GOOGLEGMAIL: 'Gmail', GITHUB: 'GitHub', SLACK: 'Slack',
    DROPBOX: 'Dropbox', TELEGRAM: 'Telegram', NOTION: 'Notion',
    JIRA: 'Jira', TRELLO: 'Trello', DISCORD: 'Discord',
    LINEAR: 'Linear', ASANA: 'Asana', WHATSAPP: 'WhatsApp',
  }
  return known[name] || name.charAt(0) + name.slice(1).toLowerCase()
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return 'Never'
  return new Date(dateStr).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
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

// ─── Filter Dropdown ──────────────────────────────────────────────────

function FilterDropdown({
  label,
  icon: Icon,
  value,
  options,
  onChange,
}: {
  label: string
  icon: any
  value: string
  options: { key: string; label: string; sub?: string }[]
  onChange: (v: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const filtered = options.filter(
    (o) => o.label.toLowerCase().includes(search.toLowerCase()) || o.key.toLowerCase().includes(search.toLowerCase())
  )
  const selectedLabel = value === 'all' ? `All ${label}` : options.find((o) => o.key === value)?.label || value

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg border border-border/50 bg-secondary/30 hover:bg-secondary/50 transition-all"
      >
        <Icon className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-foreground font-medium truncate max-w-[140px]">{selectedLabel}</span>
        <ChevronDown className="w-3 h-3 text-muted-foreground" />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute z-50 mt-1 w-72 max-h-72 rounded-xl border border-border bg-card shadow-xl overflow-hidden">
            {options.length > 5 && (
              <div className="p-2 border-b border-border/50">
                <div className="relative">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
                  <input
                    autoFocus
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder={`Search ${label.toLowerCase()}...`}
                    className="w-full pl-8 pr-3 py-1.5 text-xs bg-secondary/30 border border-border/30 rounded-lg outline-none focus:border-primary/50 text-foreground placeholder:text-muted-foreground"
                  />
                </div>
              </div>
            )}
            <div className="max-h-56 overflow-y-auto">
              <button
                className={`w-full text-left px-4 py-2.5 text-xs hover:bg-secondary/50 transition-colors flex items-center justify-between ${value === 'all' ? 'bg-primary/10 text-primary' : ''}`}
                onClick={() => { onChange('all'); setOpen(false); setSearch('') }}
              >
                <span>All {label}</span>
                {value === 'all' && <span className="text-[10px] text-primary">Active</span>}
              </button>
              {filtered.map((o) => (
                <button
                  key={o.key}
                  className={`w-full text-left px-4 py-2.5 text-xs hover:bg-secondary/50 transition-colors flex items-center justify-between ${value === o.key ? 'bg-primary/10 text-primary' : ''}`}
                  onClick={() => { onChange(o.key); setOpen(false); setSearch('') }}
                >
                  <span>{o.label}</span>
                  {o.sub && <span className="text-[10px] text-muted-foreground">{o.sub}</span>}
                </button>
              ))}
              {filtered.length === 0 && (
                <p className="px-4 py-3 text-xs text-muted-foreground">No matches</p>
              )}
            </div>
          </div>
        </>
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
  const periodDays = period === '7d' ? 7 : period === '90d' ? 90 : 30

  // ─── Data hooks ─────────────────────────────────────────────────────
  const { data: dashboard, isLoading: dashLoading } = useAdminDashboard(period)
  const { data: legacyData, isLoading: legacyLoading } = useAdminWorkspaceAnalytics(days)
  const { data: legacyCostData, isLoading: legacyCostLoading } = useAdminCostAnalytics(period)

  // Composio
  const { data: composioApps } = useComposioApps(periodDays)
  const { data: composioActions } = useComposioActions(periodDays)
  const { data: composioAgentTools } = useComposioAgentTools(periodDays)

  // ─── Filters ────────────────────────────────────────────────────────
  const [wsFilter, setWsFilter] = useState('all')
  const [providerFilter, setProviderFilter] = useState('all')

  // ─── Expandable rows ───────────────────────────────────────────────
  const [expandedWs, setExpandedWs] = useState<Set<string>>(new Set())
  const toggleExpand = (id: string) => {
    setExpandedWs((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  // ─── Sorting ────────────────────────────────────────────────────────
  const [wsSortField, setWsSortField] = useState<'cost' | 'requests' | 'agents' | 'name'>('cost')
  const [wsSortDir, setWsSortDir] = useState<'asc' | 'desc'>('desc')
  const toggleSort = (field: typeof wsSortField) => {
    if (wsSortField === field) setWsSortDir((d) => (d === 'desc' ? 'asc' : 'desc'))
    else { setWsSortField(field); setWsSortDir('desc') }
  }

  // Composio action sort
  const [actionSort, setActionSort] = useState<'total_usage_count' | 'agent_count' | 'action_name'>('total_usage_count')
  const [actionSortDir, setActionSortDir] = useState<'asc' | 'desc'>('desc')

  // ─── Computed data ──────────────────────────────────────────────────
  const hasDashboard = !!dashboard?.overview

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

  // Extract unique providers from models
  const providers = useMemo(() => {
    if (!dashboard?.models) return []
    const set = new Set(dashboard.models.map((m) => m.provider))
    return Array.from(set).sort()
  }, [dashboard?.models])

  // Filter + sort workspaces
  const sortedWorkspaces = useMemo(() => {
    let ws = [...allWorkspaces]
    if (wsFilter !== 'all') ws = ws.filter((w) => w.id === wsFilter)
    return ws.sort((a, b) => {
      const dir = wsSortDir === 'desc' ? -1 : 1
      if (wsSortField === 'name') return dir * a.name.localeCompare(b.name) * -1
      return dir * ((a[wsSortField] || 0) - (b[wsSortField] || 0))
    })
  }, [allWorkspaces, wsFilter, wsSortField, wsSortDir])

  // Filter models by provider
  const filteredModels = useMemo(() => {
    if (!dashboard?.models) return []
    if (providerFilter === 'all') return dashboard.models
    return dashboard.models.filter((m) => m.provider === providerFilter)
  }, [dashboard?.models, providerFilter])

  const maxWsCost = useMemo(() => Math.max(...allWorkspaces.map((w) => w.cost), 0.001), [allWorkspaces])
  const maxModelCost = useMemo(() => Math.max(...(filteredModels.map((m) => m.cost) || [0.001])), [filteredModels])

  const totalCost = dashboard?.overview?.total_cost || legacyCostData?.total_platform_cost || 0

  // Composio aggregates
  const composioTotalActions = composioApps?.reduce((s, a) => s + a.total_actions_used, 0) ?? 0
  const composioActiveApps = composioApps?.filter((a) => a.status === 'active' || a.status === 'connected').length ?? 0
  const composioTopApp = composioApps?.length ? [...composioApps].sort((a, b) => b.total_actions_used - a.total_actions_used)[0] : null

  const sortedComposioActions = useMemo(() => {
    if (!composioActions) return []
    return [...composioActions].sort((a, b) => {
      const av = a[actionSort], bv = b[actionSort]
      if (typeof av === 'string' && typeof bv === 'string')
        return actionSortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      return actionSortDir === 'asc' ? (Number(av) || 0) - (Number(bv) || 0) : (Number(bv) || 0) - (Number(av) || 0)
    })
  }, [composioActions, actionSort, actionSortDir])

  // Workspace dropdown options
  const wsOptions = useMemo(() =>
    allWorkspaces.map((w) => ({ key: w.id, label: w.name, sub: formatCost(w.cost) })),
    [allWorkspaces]
  )

  // Provider dropdown options
  const providerOptions = useMemo(() =>
    providers.map((p) => ({
      key: p,
      label: p.charAt(0).toUpperCase() + p.slice(1),
      sub: `${dashboard?.models?.filter((m) => m.provider === p).length || 0} models`,
    })),
    [providers, dashboard?.models]
  )

  // ─── CSV Export ─────────────────────────────────────────────────────
  const escapeCsv = (v: string) => {
    let s = v; if (/^[=+\-@]/.test(s)) s = "'" + s
    return '"' + s.replace(/"/g, '""') + '"'
  }
  const handleExport = () => {
    if (!allWorkspaces.length) return
    const headers = ['Workspace,Plan,Agents,Recipes,Executions,Requests,Tokens,Cost']
    const rows = allWorkspaces.map((ws) =>
      [escapeCsv(ws.name), escapeCsv(ws.plan || 'unknown'), ws.agents, ws.recipes ?? 0, ws.executions ?? 0, ws.requests, ws.tokens, `$${ws.cost.toFixed(4)}`].join(',')
    )
    const blob = new Blob([[...headers, ...rows].join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url
    a.download = `admin-analytics-${new Date().toISOString().split('T')[0]}.csv`
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
            <h2 className="text-sm font-semibold">Super Admin Dashboard</h2>
            <p className="text-xs text-muted-foreground">Platform-wide monitoring & revenue insights</p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <PeriodToggle value={period} onChange={setPeriod} />
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="w-4 h-4 mr-2" />Export
          </Button>
        </div>
      </div>

      {/* ═══ FILTERS ═══ */}
      <div className="flex items-center gap-2 flex-wrap">
        <Filter className="w-4 h-4 text-muted-foreground" />
        <FilterDropdown label="Workspaces" icon={Building} value={wsFilter} options={wsOptions} onChange={setWsFilter} />
        {providers.length > 0 && (
          <FilterDropdown label="Providers" icon={Server} value={providerFilter} options={providerOptions} onChange={setProviderFilter} />
        )}
        {(wsFilter !== 'all' || providerFilter !== 'all') && (
          <button
            onClick={() => { setWsFilter('all'); setProviderFilter('all') }}
            className="inline-flex items-center gap-1 px-2 py-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-3 h-3" /> Clear filters
          </button>
        )}
      </div>

      {/* ═══ HERO STATS ═══ */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 md:gap-4">
        {[
          { label: 'Total Platform Cost', value: formatCost(totalCost), sub: 'All workspaces', icon: DollarSign, iconBg: 'bg-emerald-500/10', iconColor: 'text-emerald-400', accent: 'border-l-emerald-500' },
          { label: 'Projected Monthly', value: formatCost(dashboard?.overview?.projected_monthly || 0), sub: `${formatCost(dashboard?.overview?.daily_average || 0)}/day avg`, icon: TrendingUp, iconBg: 'bg-blue-500/10', iconColor: 'text-blue-400', accent: 'border-l-blue-500' },
          { label: 'Workspaces', value: dashboard?.overview?.total_workspaces || legacyData?.platformSummary?.totalWorkspaces || 0, sub: 'Active tenants', icon: Building, iconBg: 'bg-purple-500/10', iconColor: 'text-purple-400', accent: 'border-l-purple-500' },
          { label: 'API Requests', value: formatNumber(dashboard?.overview?.total_requests || legacyData?.platformSummary?.totalApiCalls || 0), sub: 'This period', icon: Activity, iconBg: 'bg-orange-500/10', iconColor: 'text-orange-400', accent: 'border-l-orange-500' },
          { label: 'Total Tokens', value: formatNumber(dashboard?.overview?.total_tokens || legacyData?.platformSummary?.totalTokens || 0), sub: 'Input + Output', icon: Zap, iconBg: 'bg-cyan-500/10', iconColor: 'text-cyan-400', accent: 'border-l-cyan-500' },
          { label: 'Composio Actions', value: formatNumber(composioTotalActions), sub: `${composioActiveApps} active apps`, icon: AppWindow, iconBg: 'bg-pink-500/10', iconColor: 'text-pink-400', accent: 'border-l-pink-500' },
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
                        {dashboard.daily_by_provider.providers
                          .filter((p) => providerFilter === 'all' || p === providerFilter)
                          .map((prov, idx) => (
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

      {/* ═══ WORKSPACE DEEP DIVE TABLE ═══ */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.45 }}>
        <Card className="glass-card overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Building className="w-5 h-5 text-blue-400" />
                Workspace Deep Dive
              </span>
              <span className="text-xs text-muted-foreground font-normal">
                {sortedWorkspaces.length} workspace{sortedWorkspaces.length !== 1 ? 's' : ''}
                {wsFilter !== 'all' && <span className="text-primary ml-1">(filtered)</span>}
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
                      <p className="text-sm">{wsFilter !== 'all' ? 'No matching workspaces' : 'No workspace data available'}</p>
                    </td>
                  </tr>
                ) : sortedWorkspaces.map((ws) => (
                  <Fragment key={ws.id}>
                    <tr className="border-b border-border/20 hover:bg-secondary/10 transition-colors cursor-pointer" onClick={() => toggleExpand(ws.id)}>
                      <td className="p-4 w-8">
                        <motion.div animate={{ rotate: expandedWs.has(ws.id) ? 90 : 0 }} transition={{ duration: 0.2 }}>
                          <ChevronRight className="w-4 h-4 text-muted-foreground" />
                        </motion.div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm">{ws.name}</span>
                          {ws.is_personal && <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-blue-400/30 text-blue-400">personal</Badge>}
                        </div>
                      </td>
                      <td className="p-4">
                        <Badge variant="outline" className={`text-[10px] uppercase ${PLAN_COLORS[ws.plan] || PLAN_COLORS.starter}`}>{ws.plan}</Badge>
                      </td>
                      <td className="p-4 text-sm text-right tabular-nums">{ws.agents}</td>
                      <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell">{ws.recipes ?? 0}</td>
                      <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell">{formatNumber(ws.requests)}</td>
                      <td className="p-4 text-sm text-right tabular-nums hidden lg:table-cell text-muted-foreground">{formatNumber(ws.tokens)}</td>
                      <td className="p-4 text-sm text-right tabular-nums font-medium">{formatCost(ws.cost)}</td>
                      <td className="p-4 hidden lg:table-cell">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 rounded-full bg-secondary/30 overflow-hidden">
                            <motion.div className="h-full rounded-full bg-blue-400" initial={{ width: 0 }} animate={{ width: `${(ws.cost / maxWsCost) * 100}%` }} transition={{ duration: 0.6 }} />
                          </div>
                          <span className="text-[10px] text-muted-foreground w-8 text-right tabular-nums">
                            {totalCost > 0 ? `${((ws.cost / totalCost) * 100).toFixed(0)}%` : '0%'}
                          </span>
                        </div>
                      </td>
                    </tr>
                    <AnimatePresence>
                      {expandedWs.has(ws.id) && (
                        <tr>
                          <td colSpan={9} className="p-0">
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.3 }} className="overflow-hidden">
                              <div className="px-6 py-4 bg-secondary/5 border-b border-border/20">
                                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                                  {[
                                    { icon: Bot, color: 'text-blue-400', label: 'Agents', value: ws.agents },
                                    { icon: ChefHat, color: 'text-orange-400', label: 'Recipes', value: ws.recipes ?? 0 },
                                    { icon: Play, color: 'text-green-400', label: 'Executions', value: ws.executions ?? 0 },
                                    { icon: Zap, color: 'text-cyan-400', label: 'Tokens', value: formatNumber(ws.tokens) },
                                    { icon: DollarSign, color: 'text-emerald-400', label: 'Cost/Req', value: ws.requests > 0 ? formatCost(ws.cost / ws.requests) : '--' },
                                  ].map((m) => (
                                    <div key={m.label} className="rounded-lg bg-secondary/20 border border-border/20 p-3">
                                      <div className="flex items-center gap-2 mb-1">
                                        <m.icon className={`w-3.5 h-3.5 ${m.color}`} />
                                        <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{m.label}</span>
                                      </div>
                                      <p className="text-lg font-bold">{m.value}</p>
                                    </div>
                                  ))}
                                </div>
                                <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                                  <span>Created: {ws.created_at ? new Date(ws.created_at).toLocaleDateString() : 'N/A'}</span>
                                  <span>Tokens/Req: {ws.requests > 0 ? formatNumber(ws.tokens / ws.requests) : '--'}</span>
                                </div>
                              </div>
                            </motion.div>
                          </td>
                        </tr>
                      )}
                    </AnimatePresence>
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </motion.div>

      {/* ═══ PLATFORM MODEL USAGE ═══ */}
      {filteredModels.length > 0 && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.5 }}>
          <Card className="glass-card overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Layers className="w-5 h-5 text-purple-400" />
                  Platform Model Usage
                </span>
                {providerFilter !== 'all' && (
                  <Badge variant="outline" className="text-[10px] capitalize">{providerFilter}</Badge>
                )}
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
                  {filteredModels.map((model, idx) => (
                    <tr key={model.model_id} className="border-b border-border/20 hover:bg-secondary/10 transition-colors">
                      <td className="p-4">
                        <div className="flex items-center gap-2.5">
                          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: PROVIDER_COLORS[idx % PROVIDER_COLORS.length] }} />
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
                            <div className="h-full rounded-full transition-all duration-500" style={{ width: `${(model.cost / maxModelCost) * 100}%`, background: PROVIDER_COLORS[idx % PROVIDER_COLORS.length] }} />
                          </div>
                          <span className="text-[10px] text-muted-foreground w-8 text-right tabular-nums">
                            {totalCost > 0 ? `${((model.cost / totalCost) * 100).toFixed(0)}%` : '0%'}
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

      {/* ═══ COMPOSIO API ANALYTICS ═══ */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.55 }}>
        <div className="space-y-6">
          {/* Section header */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center">
              <AppWindow className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">Composio API & Tool Analytics</h3>
              <p className="text-xs text-muted-foreground">Connected apps, action usage, and agent tool mappings</p>
            </div>
          </div>

          {/* Composio Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Connected Apps', value: composioApps?.length ?? 0, sub: `${composioActiveApps} active`, icon: AppWindow, color: 'text-indigo-400', bg: 'bg-indigo-500/10', accent: 'border-l-indigo-500' },
              { label: 'Total API Actions', value: composioTotalActions, sub: `Last ${periodDays}d`, icon: Zap, color: 'text-yellow-400', bg: 'bg-yellow-500/10', accent: 'border-l-yellow-500' },
              { label: 'Most Used App', value: composioTopApp ? formatAppName(composioTopApp.app_name) : 'None', sub: composioTopApp ? `${composioTopApp.total_actions_used} actions` : 'No data', icon: Wrench, color: 'text-green-400', bg: 'bg-green-500/10', accent: 'border-l-green-500' },
              { label: 'Agent Integrations', value: composioAgentTools?.length ?? 0, sub: 'Agents with tools', icon: Link2, color: 'text-cyan-400', bg: 'bg-cyan-500/10', accent: 'border-l-cyan-500' },
            ].map((stat, idx) => (
              <motion.div key={stat.label} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.6 + idx * 0.05 }}>
                <Card className={`glass-card border-l-2 ${stat.accent}`}>
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1 min-w-0">
                        <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider truncate">{stat.label}</p>
                        <p className="text-lg font-bold leading-none truncate">{stat.value}</p>
                        <p className="text-[10px] text-muted-foreground mt-1">{stat.sub}</p>
                      </div>
                      <div className={`w-7 h-7 rounded-lg ${stat.bg} flex items-center justify-center shrink-0`}>
                        <stat.icon className={`w-3.5 h-3.5 ${stat.color}`} />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          {/* Connected Apps Grid + Action Leaderboard side by side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Connected Apps */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AppWindow className="w-5 h-5 text-indigo-400" />
                  Connected Apps
                </CardTitle>
              </CardHeader>
              <CardContent>
                {!composioApps || composioApps.length === 0 ? (
                  <div className="text-center py-8">
                    <AppWindow className="w-10 h-10 mx-auto mb-3 text-muted-foreground/30" />
                    <p className="text-sm text-muted-foreground">No Composio apps connected</p>
                    <p className="text-xs text-muted-foreground/60 mt-1">Connect apps in Settings to enable tool analytics</p>
                  </div>
                ) : (
                  <div className="space-y-2.5 max-h-80 overflow-y-auto pr-1">
                    {[...composioApps].sort((a, b) => b.total_actions_used - a.total_actions_used).map((app) => {
                      const maxActions = Math.max(...composioApps.map((a) => a.total_actions_used), 1)
                      return (
                        <div key={app.app_name} className="rounded-lg border border-border/20 p-3 hover:border-border/40 transition-colors">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium">{formatAppName(app.app_name)}</span>
                            <Badge
                              variant="outline"
                              className={`text-[10px] ${app.status === 'active' || app.status === 'connected' ? 'text-green-400 border-green-400/30' : app.status === 'error' ? 'text-red-400 border-red-400/30' : ''}`}
                            >
                              {app.status}
                            </Badge>
                          </div>
                          <div className="flex items-center justify-between text-xs text-muted-foreground mb-1.5">
                            <span>{app.total_actions_used} actions &middot; {app.agent_count} agent{app.agent_count !== 1 ? 's' : ''}</span>
                            {app.documents_synced > 0 && <span>{app.documents_synced} docs</span>}
                          </div>
                          <div className="h-1.5 w-full rounded-full bg-secondary/30 overflow-hidden">
                            <motion.div
                              className="h-full rounded-full bg-indigo-400"
                              initial={{ width: 0 }}
                              animate={{ width: `${(app.total_actions_used / maxActions) * 100}%` }}
                              transition={{ duration: 0.6 }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Action Leaderboard */}
            <Card className="glass-card overflow-hidden">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-yellow-400" />
                  Top API Actions
                </CardTitle>
              </CardHeader>
              {!sortedComposioActions.length ? (
                <CardContent>
                  <div className="text-center py-8">
                    <Zap className="w-10 h-10 mx-auto mb-3 text-muted-foreground/30" />
                    <p className="text-sm text-muted-foreground">No action usage data yet</p>
                    <p className="text-xs text-muted-foreground/60 mt-1">Actions will appear as agents use Composio tools</p>
                  </div>
                </CardContent>
              ) : (
                <div className="overflow-x-auto max-h-80">
                  <table className="w-full">
                    <thead className="sticky top-0 bg-card">
                      <tr className="border-b border-border/50">
                        <th
                          className="text-left p-3 text-[11px] font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:text-foreground"
                          onClick={() => { actionSort === 'action_name' ? setActionSortDir((d) => d === 'desc' ? 'asc' : 'desc') : (setActionSort('action_name'), setActionSortDir('desc')) }}
                        >
                          Action {actionSort === 'action_name' && <ChevronDown className={`inline w-3 h-3 ${actionSortDir === 'asc' ? 'rotate-180' : ''}`} />}
                        </th>
                        <th className="text-left p-3 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">App</th>
                        <th
                          className="text-right p-3 text-[11px] font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:text-foreground"
                          onClick={() => { actionSort === 'total_usage_count' ? setActionSortDir((d) => d === 'desc' ? 'asc' : 'desc') : (setActionSort('total_usage_count'), setActionSortDir('desc')) }}
                        >
                          Usage {actionSort === 'total_usage_count' && <ChevronDown className={`inline w-3 h-3 ${actionSortDir === 'asc' ? 'rotate-180' : ''}`} />}
                        </th>
                        <th
                          className="text-right p-3 text-[11px] font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:text-foreground hidden md:table-cell"
                          onClick={() => { actionSort === 'agent_count' ? setActionSortDir((d) => d === 'desc' ? 'asc' : 'desc') : (setActionSort('agent_count'), setActionSortDir('desc')) }}
                        >
                          Agents {actionSort === 'agent_count' && <ChevronDown className={`inline w-3 h-3 ${actionSortDir === 'asc' ? 'rotate-180' : ''}`} />}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedComposioActions.slice(0, 15).map((action) => (
                        <tr key={`${action.app_name}-${action.action_name}`} className="border-b border-border/20 hover:bg-secondary/10 transition-colors">
                          <td className="p-3">
                            <Badge variant="secondary" className="font-mono text-[10px]">{action.action_name}</Badge>
                          </td>
                          <td className="p-3 text-xs text-muted-foreground">{formatAppName(action.app_name)}</td>
                          <td className="p-3 text-sm text-right tabular-nums font-medium">{action.total_usage_count}</td>
                          <td className="p-3 text-sm text-right tabular-nums text-muted-foreground hidden md:table-cell">{action.agent_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          </div>

          {/* Agent Tool Mapping */}
          {composioAgentTools && composioAgentTools.length > 0 && (
            <Card className="glass-card overflow-hidden">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bot className="w-5 h-5 text-green-400" />
                  Agent Tool Usage
                </CardTitle>
              </CardHeader>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border/50">
                      <th className="w-8 p-4" />
                      <th className="text-left p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Agent</th>
                      <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Tools</th>
                      <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Total Usage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {composioAgentTools.map((agent) => {
                      const totalUsage = agent.tools.reduce((s, t) => s + t.usage_count, 0)
                      const isExp = expandedWs.has(`agent-${agent.agent_id}`)
                      return (
                        <Fragment key={agent.agent_id}>
                          <tr
                            className="border-b border-border/20 hover:bg-secondary/10 transition-colors cursor-pointer"
                            onClick={() => toggleExpand(`agent-${agent.agent_id}`)}
                          >
                            <td className="p-4 w-8">
                              <motion.div animate={{ rotate: isExp ? 90 : 0 }} transition={{ duration: 0.2 }}>
                                <ChevronRight className="w-4 h-4 text-muted-foreground" />
                              </motion.div>
                            </td>
                            <td className="p-4 font-medium text-sm">{agent.agent_name}</td>
                            <td className="p-4 text-sm text-right tabular-nums">{agent.tools.length}</td>
                            <td className="p-4 text-sm text-right tabular-nums font-medium">{totalUsage}</td>
                          </tr>
                          <AnimatePresence>
                            {isExp && (
                              <tr>
                                <td colSpan={4} className="p-0">
                                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2 }} className="overflow-hidden">
                                    <div className="bg-secondary/5">
                                      {agent.tools.map((tool) => (
                                        <div key={tool.tool_name} className="flex items-center justify-between px-6 py-2.5 border-b border-border/10">
                                          <div className="flex items-center gap-2">
                                            <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                                            <Badge variant="secondary" className="font-mono text-[10px]">{tool.tool_name}</Badge>
                                            <span className="text-xs text-muted-foreground">{formatAppName(tool.app_name)}</span>
                                          </div>
                                          <div className="flex items-center gap-2">
                                            <span className="text-xs tabular-nums">{tool.usage_count}</span>
                                            {!tool.enabled && <Badge variant="outline" className="text-[10px] px-1.5 py-0">disabled</Badge>}
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  </motion.div>
                                </td>
                              </tr>
                            )}
                          </AnimatePresence>
                        </Fragment>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      </motion.div>

      {/* ═══ PROVIDER BAR CHART (legacy fallback) ═══ */}
      {!dashboard?.daily_by_provider?.series?.length && legacyCostData?.cost_by_provider?.length ? (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.6 }}>
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
                  <BarChart data={legacyCostData.cost_by_provider.map((p) => ({ name: p.key, cost: p.total_cost }))} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.3} vertical={false} />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} tickFormatter={(v: number) => formatCost(v)} width={60} />
                    <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '12px', fontSize: '12px' }} formatter={(value: number) => [formatCost(value), 'Cost']} />
                    <Bar dataKey="cost" radius={[6, 6, 0, 0]}>
                      {legacyCostData.cost_by_provider.map((_, i) => <Cell key={i} fill={PROVIDER_COLORS[i % PROVIDER_COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      ) : null}
    </div>
  )
}
