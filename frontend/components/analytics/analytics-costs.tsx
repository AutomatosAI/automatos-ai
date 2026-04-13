'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  DollarSign,
  Zap,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Bot,
  GitCompare,
  X,
  ChevronDown,
  ArrowUpDown,
  Layers,
  BarChart3,
  Activity,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { StatsBar, type StatItem } from '@/components/shared/stats-bar'
import { Progress } from '@/components/ui/progress'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
  Cell,
} from 'recharts'
import {
  useCostAnalyticsUnified,
  useModelComparison,
  useCostProjections,
  useDailyCostByModel,
} from '@/hooks/use-unified-analytics'
import { useWorkspaceModels } from '@/hooks/use-model-api'

interface Props {
  days: number
}

// Vibrant distinct colors for model lines — designed for dark backgrounds
const MODEL_COLORS = [
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

const COMPARISON_COLORS = ['#60B5FF', '#72BF78', '#ff6b35', '#a78bfa']

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
  // Strip provider prefix for legend/tooltip readability
  const parts = name.split('/')
  return parts[parts.length - 1]
}

// Period selector component
function PeriodToggle({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const periods = [
    { key: '24h', label: '1D' },
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

// Custom tooltip for multi-line chart
function ModelCostTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
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
                <span className="text-xs text-foreground">{shortenModelName(p.dataKey)}</span>
              </div>
              <span className="text-xs font-mono font-medium">{formatCost(p.value)}</span>
            </div>
          ))}
      </div>
    </div>
  )
}

export function AnalyticsCosts({ days }: Props) {
  const { data, isLoading } = useCostAnalyticsUnified(days)
  const [chartPeriod, setChartPeriod] = useState('30d')
  const { data: dailyByModel, isLoading: dailyLoading } = useDailyCostByModel(chartPeriod)
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([])
  const [comparisonPeriod, setComparisonPeriod] = useState('30d')
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false)
  const [projectionPeriod, setProjectionPeriod] = useState('30d')
  const { data: comparisonData, isLoading: comparisonLoading } = useModelComparison(selectedModelIds, comparisonPeriod)
  const { data: projections, isLoading: projectionsLoading } = useCostProjections(projectionPeriod)
  const { data: workspaceModels = [] } = useWorkspaceModels()

  // Build a merged list of all models for the comparison dropdown:
  // workspace models (all installed) + usage models (have cost data).
  // Workspace models appear even when they have zero usage.
  const allAvailableModels = useMemo(() => {
    const seen = new Set<string>()
    const result: Array<{ model: string; displayName: string; requests: number; totalCost: number }> = []

    // Usage-based models first (they have stats to show)
    for (const m of data?.byModel || []) {
      const key = m.model.toLowerCase()
      if (!seen.has(key)) {
        seen.add(key)
        result.push({ model: m.model, displayName: shortenModelName(m.model), requests: m.requests, totalCost: m.totalCost })
      }
    }

    // Workspace-installed models that have no usage yet
    for (const wm of workspaceModels) {
      const key = wm.model_id.toLowerCase()
      if (!seen.has(key)) {
        seen.add(key)
        result.push({ model: wm.model_id, displayName: wm.display_name || shortenModelName(wm.model_id), requests: 0, totalCost: 0 })
      }
    }

    return result
  }, [data?.byModel, workspaceModels])

  // Compute max cost share for proportional bars in model table
  const maxModelCost = useMemo(() => {
    if (!data?.byModel?.length) return 1
    return Math.max(...data.byModel.map((m) => m.totalCost), 0.001)
  }, [data?.byModel])

  if (isLoading) {
    return (
      <div className="space-y-6">
        <StatsBar stats={[
          { label: 'Total Cost', value: '...', icon: DollarSign, iconColor: 'text-[hsl(var(--success))]' },
          { label: 'Total Tokens', value: '...', icon: Zap, iconColor: 'text-[hsl(var(--info))]' },
          { label: 'Cost per Request', value: '...', icon: Activity, iconColor: 'text-[hsl(var(--agent))]' },
          { label: 'Top Spender', value: '...', icon: AlertTriangle, iconColor: 'text-primary' },
        ]} loading={true} />
        <Skeleton className="h-80 w-full rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* ─── Hero Stats ─── */}
      <StatsBar stats={[
        {
          label: 'Total Cost',
          value: formatCost(data?.summary?.totalCost || 0),
          change: 'This period',
          icon: DollarSign,
          iconColor: 'text-[hsl(var(--success))]',
          globalIconKey: 'global_cost',
        },
        {
          label: 'Total Tokens',
          value: formatNumber(data?.summary?.totalTokens || 0),
          change: 'Input + Output',
          icon: Zap,
          iconColor: 'text-[hsl(var(--info))]',
        },
        {
          label: 'Cost per Request',
          value: formatCost(data?.summary?.costPerTask || 0),
          change: `${formatNumber(data?.summary?.totalRequests || 0)} requests`,
          icon: Activity,
          iconColor: 'text-[hsl(var(--agent))]',
        },
        {
          label: 'Top Spender',
          value: data?.summary?.mostExpensiveAgent?.name || 'None',
          change: data?.summary?.mostExpensiveAgent
            ? `${formatCost(data.summary.mostExpensiveAgent.cost)} on ${shortenModelName(data.summary.mostExpensiveAgent.model)}`
            : 'No data',
          icon: AlertTriangle,
          iconColor: 'text-primary',
        },
      ] satisfies StatItem[]} />

      {/* ─── Multi-Line Cost by Model Chart ─── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-400" />
                Cost by Model Over Time
              </span>
              <PeriodToggle value={chartPeriod} onChange={setChartPeriod} />
            </CardTitle>
          </CardHeader>
          <CardContent>
            {dailyLoading ? (
              <Skeleton className="h-80 w-full rounded-xl" />
            ) : dailyByModel && dailyByModel.series?.length > 0 ? (
              <>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={dailyByModel.series} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                      <defs>
                        {dailyByModel.models.map((model, idx) => (
                          <linearGradient key={model} id={`grad-${idx}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={MODEL_COLORS[idx % MODEL_COLORS.length]} stopOpacity={0.3} />
                            <stop offset="100%" stopColor={MODEL_COLORS[idx % MODEL_COLORS.length]} stopOpacity={0.02} />
                          </linearGradient>
                        ))}
                      </defs>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="hsl(var(--border))"
                        strokeOpacity={0.3}
                        vertical={false}
                      />
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
                      <Tooltip content={<ModelCostTooltip />} />
                      {dailyByModel.models.map((model, idx) => (
                        <Area
                          key={model}
                          type="monotone"
                          dataKey={model}
                          stroke={MODEL_COLORS[idx % MODEL_COLORS.length]}
                          strokeWidth={2}
                          fill={`url(#grad-${idx})`}
                          dot={false}
                          activeDot={{ r: 4, strokeWidth: 2, fill: 'hsl(var(--card))' }}
                        />
                      ))}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                {/* Legend */}
                <div className="flex flex-wrap gap-3 mt-4 px-1">
                  {dailyByModel.models.map((model, idx) => (
                    <div key={model} className="flex items-center gap-1.5">
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ background: MODEL_COLORS[idx % MODEL_COLORS.length] }}
                      />
                      <span className="text-xs text-muted-foreground">{shortenModelName(model)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              /* Fallback to aggregate cost trend if no per-model data */
              data?.costTrend?.length > 0 ? (
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data.costTrend} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#60B5FF" stopOpacity={0.35} />
                          <stop offset="100%" stopColor="#60B5FF" stopOpacity={0.02} />
                        </linearGradient>
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
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'hsl(var(--card))',
                          border: '1px solid hsl(var(--border))',
                          borderRadius: '12px',
                          fontSize: '12px',
                        }}
                        formatter={(value: number) => [formatCost(value), 'Daily Cost']}
                      />
                      <Area
                        type="monotone"
                        dataKey="total_cost"
                        stroke="#60B5FF"
                        strokeWidth={2}
                        fill="url(#costGrad)"
                        dot={false}
                        activeDot={{ r: 4, strokeWidth: 2, fill: 'hsl(var(--card))' }}
                        name="Daily Cost"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-80 flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="w-12 h-12 mx-auto mb-3 text-muted-foreground/30" />
                    <p className="text-sm text-muted-foreground">No cost data for this period</p>
                    <p className="text-xs text-muted-foreground/60 mt-1">Run some agent conversations to generate usage data</p>
                  </div>
                </div>
              )
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* ─── Cost Projections ─── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-purple-400" />
                Cost Projections
              </span>
              <PeriodToggle value={projectionPeriod} onChange={setProjectionPeriod} />
            </CardTitle>
          </CardHeader>
          <CardContent>
            {projectionsLoading ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-24 w-full rounded-xl" />
                ))}
              </div>
            ) : projections ? (
              <div className="space-y-6">
                {/* Projection metric cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="rounded-xl bg-secondary/20 border border-border/20 p-4 space-y-1">
                    <p className="text-[11px] text-muted-foreground uppercase tracking-wider">Current Period</p>
                    <p className="text-xl font-bold">{formatCost(projections.current_period_cost)}</p>
                  </div>
                  <div className="rounded-xl bg-secondary/20 border border-border/20 p-4 space-y-1">
                    <p className="text-[11px] text-muted-foreground uppercase tracking-wider">Daily Average</p>
                    <p className="text-xl font-bold">{formatCost(projections.daily_average)}</p>
                  </div>
                  <div className="rounded-xl bg-primary/5 border border-primary/20 p-4 space-y-1">
                    <p className="text-[11px] text-muted-foreground uppercase tracking-wider">Projected Monthly</p>
                    <p className="text-2xl font-bold text-primary">{formatCost(projections.projected_monthly)}</p>
                  </div>
                  <div className="rounded-xl bg-secondary/20 border border-border/20 p-4 space-y-1">
                    <p className="text-[11px] text-muted-foreground uppercase tracking-wider">vs Previous</p>
                    {projections.change_percent != null ? (
                      <div className="flex items-center gap-1.5">
                        {projections.change_percent > 0 ? (
                          <TrendingUp className="w-4 h-4 text-red-400" />
                        ) : projections.change_percent < 0 ? (
                          <TrendingDown className="w-4 h-4 text-green-400" />
                        ) : null}
                        <p className={`text-xl font-bold ${
                          projections.change_percent > 0 ? 'text-red-400' :
                          projections.change_percent < 0 ? 'text-green-400' : 'text-muted-foreground'
                        }`}>
                          {projections.change_percent > 0 ? '+' : ''}{projections.change_percent.toFixed(1)}%
                        </p>
                      </div>
                    ) : (
                      <p className="text-xl font-bold text-muted-foreground">--</p>
                    )}
                  </div>
                </div>

                {/* Projected by model — horizontal bars with visual proportion */}
                {projections.projected_by_model && projections.projected_by_model.length > 0 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-3 uppercase tracking-wider">Projected Monthly by Model</p>
                    <div className="space-y-2.5">
                      {projections.projected_by_model
                        .sort((a, b) => b.projected_monthly_cost - a.projected_monthly_cost)
                        .map((item, idx) => {
                          const maxProjected = Math.max(
                            ...projections.projected_by_model!.map((m) => m.projected_monthly_cost),
                            0.001
                          )
                          const pct = (item.projected_monthly_cost / maxProjected) * 100
                          return (
                            <div key={item.key} className="group">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-foreground">{shortenModelName(item.key)}</span>
                                <span className="text-xs font-mono font-medium">{formatCost(item.projected_monthly_cost)}/mo</span>
                              </div>
                              <div className="h-2 w-full rounded-full bg-secondary/30 overflow-hidden">
                                <motion.div
                                  className="h-full rounded-full"
                                  style={{ background: MODEL_COLORS[idx % MODEL_COLORS.length] }}
                                  initial={{ width: 0 }}
                                  animate={{ width: `${pct}%` }}
                                  transition={{ duration: 0.8, delay: idx * 0.05 }}
                                />
                              </div>
                            </div>
                          )
                        })}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12">
                <TrendingUp className="w-10 h-10 mx-auto mb-3 text-muted-foreground/30" />
                <p className="text-sm text-muted-foreground">No projection data available</p>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* ─── Token Usage by Model — Enhanced Table ─── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
      >
        <Card className="glass-card overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-purple-400" />
              Token Usage by Model
            </CardTitle>
          </CardHeader>
          <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
            <table className="w-full">
              <thead className="sticky top-0 bg-card z-10">
                <tr className="border-b border-border/50">
                  <th className="text-left p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Model</th>
                  <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Requests</th>
                  <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden md:table-cell">Input</th>
                  <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden md:table-cell">Output</th>
                  <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Cost</th>
                  <th className="p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden lg:table-cell w-48">Share</th>
                  <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden lg:table-cell">Avg/Req</th>
                </tr>
              </thead>
              <tbody>
                {(!data?.byModel || data.byModel.length === 0) ? (
                  <tr>
                    <td colSpan={7} className="p-12 text-center text-muted-foreground">
                      <Zap className="w-10 h-10 mx-auto mb-3 opacity-30" />
                      <p className="text-sm">No token usage data yet</p>
                    </td>
                  </tr>
                ) : (
                  data.byModel.map((model, idx) => (
                    <tr key={model.model} className="border-b border-border/20 hover:bg-secondary/10 transition-colors">
                      <td className="p-4">
                        <div className="flex items-center gap-2.5">
                          <span
                            className="w-2.5 h-2.5 rounded-full shrink-0"
                            style={{ background: MODEL_COLORS[idx % MODEL_COLORS.length] }}
                          />
                          <Badge variant="secondary" className="font-mono text-xs">{shortenModelName(model.model)}</Badge>
                        </div>
                      </td>
                      <td className="p-4 text-sm text-right tabular-nums">{formatNumber(model.requests)}</td>
                      <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell text-muted-foreground">{formatNumber(model.inputTokens)}</td>
                      <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell text-muted-foreground">{formatNumber(model.outputTokens)}</td>
                      <td className="p-4 text-sm text-right tabular-nums font-medium">{formatCost(model.totalCost)}</td>
                      <td className="p-4 hidden lg:table-cell">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 rounded-full bg-secondary/30 overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${(model.totalCost / maxModelCost) * 100}%`,
                                background: MODEL_COLORS[idx % MODEL_COLORS.length],
                              }}
                            />
                          </div>
                          <span className="text-[10px] text-muted-foreground w-8 text-right">
                            {data.summary.totalCost > 0 ? `${((model.totalCost / data.summary.totalCost) * 100).toFixed(0)}%` : '0%'}
                          </span>
                        </div>
                      </td>
                      <td className="p-4 text-sm text-right tabular-nums hidden lg:table-cell text-muted-foreground font-mono">
                        {formatCost(model.avgCostPerRequest)}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </motion.div>

      {/* ─── Per-Agent Cost Breakdown ─── */}
      {data?.byAgent && data.byAgent.filter(a => a.cost > 0 || a.tokens > 0).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <Card className="glass-card overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bot className="w-5 h-5 text-orange-400" />
                Cost by Agent
              </CardTitle>
            </CardHeader>
            <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
              <table className="w-full">
                <thead className="sticky top-0 bg-card z-10">
                  <tr className="border-b border-border/50">
                    <th className="text-left p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Agent</th>
                    <th className="text-left p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Model</th>
                    <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Tokens</th>
                    <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Cost</th>
                    <th className="text-right p-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wider hidden md:table-cell">Requests</th>
                  </tr>
                </thead>
                <tbody>
                  {data.byAgent
                    .filter(a => a.cost > 0 || a.tokens > 0)
                    .map((agent) => (
                      <tr key={agent.id} className="border-b border-border/20 hover:bg-secondary/10 transition-colors">
                        <td className="p-4 font-medium text-sm">{agent.name}</td>
                        <td className="p-4">
                          <Badge variant="secondary" className="font-mono text-xs">{shortenModelName(agent.model)}</Badge>
                        </td>
                        <td className="p-4 text-sm text-right tabular-nums">{formatNumber(agent.tokens)}</td>
                        <td className="p-4 text-sm text-right tabular-nums font-medium">{formatCost(agent.cost)}</td>
                        <td className="p-4 text-sm text-right tabular-nums hidden md:table-cell text-muted-foreground">{agent.requests}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </Card>
        </motion.div>
      )}

      {/* ─── Model Comparison ─── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.7 }}
      >
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <GitCompare className="w-5 h-5 text-blue-400" />
                Model Comparison
              </span>
              <PeriodToggle value={comparisonPeriod} onChange={setComparisonPeriod} />
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Model selector */}
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">Select up to 4 models to compare</p>
              <div className="flex flex-wrap gap-2 items-center">
                {selectedModelIds.map((id, idx) => (
                  <Badge
                    key={id}
                    variant="secondary"
                    className="font-mono text-xs flex items-center gap-1.5 py-1"
                    style={{ borderColor: COMPARISON_COLORS[idx] }}
                  >
                    <span className="w-2 h-2 rounded-full" style={{ background: COMPARISON_COLORS[idx] }} />
                    {shortenModelName(id)}
                    <button
                      onClick={() => setSelectedModelIds(prev => prev.filter(m => m !== id))}
                      className="ml-0.5 hover:text-destructive transition-colors"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </Badge>
                ))}
                {selectedModelIds.length < 4 && (
                  <div className="relative">
                    <button
                      onClick={() => setModelDropdownOpen(prev => !prev)}
                      className="inline-flex items-center gap-1 text-xs px-3 py-1.5 rounded-lg border border-dashed border-border/50 text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all"
                    >
                      Add Model
                      <ChevronDown className="w-3 h-3" />
                    </button>
                    {modelDropdownOpen && (
                      <div className="absolute z-50 mt-1 w-72 max-h-56 overflow-y-auto rounded-xl border border-border bg-card shadow-xl">
                        {allAvailableModels
                          .filter(m => !selectedModelIds.includes(m.model))
                          .map(m => (
                            <button
                              key={m.model}
                              className="w-full text-left px-4 py-2.5 text-xs hover:bg-secondary/50 transition-colors font-mono flex items-center justify-between"
                              onClick={() => {
                                setSelectedModelIds(prev => [...prev, m.model])
                                setModelDropdownOpen(false)
                              }}
                            >
                              <span>{m.displayName}</span>
                              <span className="text-muted-foreground">
                                {m.requests > 0 ? `${m.requests} req · ${formatCost(m.totalCost)}` : 'No usage yet'}
                              </span>
                            </button>
                          ))}
                        {allAvailableModels.filter(m => !selectedModelIds.includes(m.model)).length === 0 && (
                          <p className="px-4 py-3 text-xs text-muted-foreground">No more models available</p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Comparison content */}
            {selectedModelIds.length === 0 ? (
              <div className="text-center py-12">
                <GitCompare className="w-10 h-10 mx-auto mb-3 text-muted-foreground/30" />
                <p className="text-sm text-muted-foreground">Select models above to compare cost, performance, and capabilities</p>
              </div>
            ) : comparisonLoading ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Array.from({ length: 8 }).map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full rounded-xl" />
                ))}
              </div>
            ) : comparisonData && comparisonData.length > 0 ? (
              <>
                {/* Comparison Table */}
                <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
                  <table className="w-full">
                    <thead className="sticky top-0 bg-card z-10">
                      <tr className="border-b border-border/50">
                        <th className="text-left p-3 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Metric</th>
                        {comparisonData.map((model, idx) => (
                          <th key={model.model_id} className="text-right p-3 text-[11px] font-medium uppercase tracking-wider" style={{ color: COMPARISON_COLORS[idx] }}>
                            {model.display_name || shortenModelName(model.model_id)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { label: 'Provider', render: (m: any) => m.provider || '--' },
                        { label: 'Input Cost/1K', render: (m: any) => m.input_cost_per_1k != null ? `$${m.input_cost_per_1k.toFixed(4)}` : '--' },
                        { label: 'Output Cost/1K', render: (m: any) => m.output_cost_per_1k != null ? `$${m.output_cost_per_1k.toFixed(4)}` : '--' },
                        { label: 'Total Requests', render: (m: any) => formatNumber(m.total_requests) },
                        { label: 'Total Tokens', render: (m: any) => formatNumber(m.total_tokens) },
                        { label: 'Total Cost', render: (m: any) => formatCost(m.total_cost), bold: true },
                        { label: 'Avg Latency', render: (m: any) => m.avg_latency_ms != null ? `${m.avg_latency_ms.toFixed(0)}ms` : '--' },
                        {
                          label: 'Error Rate',
                          render: (m: any) => (
                            <span className={m.error_rate > 5 ? 'text-red-400' : m.error_rate > 1 ? 'text-yellow-400' : 'text-green-400'}>
                              {m.error_rate.toFixed(1)}%
                            </span>
                          ),
                        },
                        { label: 'Success Rate', render: (m: any) => `${m.success_rate.toFixed(1)}%` },
                      ].map((row) => (
                        <tr key={row.label} className="border-b border-border/20">
                          <td className="p-3 text-xs text-muted-foreground">{row.label}</td>
                          {comparisonData.map(m => (
                            <td key={m.model_id} className={`p-3 text-sm text-right tabular-nums ${row.bold ? 'font-medium' : ''}`}>
                              {row.render(m)}
                            </td>
                          ))}
                        </tr>
                      ))}
                      <tr>
                        <td className="p-3 text-xs text-muted-foreground">Capabilities</td>
                        {comparisonData.map(m => (
                          <td key={m.model_id} className="p-3 text-right">
                            <div className="flex flex-wrap gap-1 justify-end">
                              {m.capabilities && Object.entries(m.capabilities)
                                .filter(([, v]) => v === true || (typeof v === 'number' && v > 0))
                                .map(([cap]) => (
                                  <Badge key={cap} variant="outline" className="text-[10px] px-1.5 py-0">
                                    {cap}
                                  </Badge>
                                ))}
                              {(!m.capabilities || Object.keys(m.capabilities).length === 0) && (
                                <span className="text-xs text-muted-foreground">--</span>
                              )}
                            </div>
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Radar Chart */}
                {(() => {
                  const allCaps = new Set<string>()
                  comparisonData.forEach(m => {
                    if (m.capabilities) {
                      Object.entries(m.capabilities).forEach(([k, v]) => {
                        if (typeof v === 'number' || v === true) allCaps.add(k)
                      })
                    }
                  })
                  const capsList = Array.from(allCaps)
                  if (capsList.length < 3) return null

                  const radarData = capsList.map(cap => {
                    const entry: Record<string, any> = { capability: cap }
                    comparisonData.forEach(m => {
                      const val = m.capabilities?.[cap]
                      entry[m.model_id] = typeof val === 'number' ? val : val === true ? 1 : 0
                    })
                    return entry
                  })

                  return (
                    <div className="mt-4">
                      <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Capability Comparison</p>
                      <div className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart data={radarData}>
                            <PolarGrid stroke="hsl(var(--border))" strokeOpacity={0.5} />
                            <PolarAngleAxis
                              dataKey="capability"
                              tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                            />
                            <PolarRadiusAxis tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} />
                            {comparisonData.map((m, idx) => (
                              <Radar
                                key={m.model_id}
                                name={m.display_name || shortenModelName(m.model_id)}
                                dataKey={m.model_id}
                                stroke={COMPARISON_COLORS[idx]}
                                fill={COMPARISON_COLORS[idx]}
                                fillOpacity={0.1}
                                strokeWidth={2}
                              />
                            ))}
                            <Legend wrapperStyle={{ fontSize: '11px' }} />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: 'hsl(var(--card))',
                                border: '1px solid hsl(var(--border))',
                                borderRadius: '12px',
                                fontSize: '12px',
                              }}
                            />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )
                })()}
              </>
            ) : (
              <div className="text-center py-8">
                <p className="text-sm text-muted-foreground">No comparison data available for selected models</p>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
