'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  DollarSign,
  Zap,
  Clock,
  TrendingDown,
  Lightbulb,
  BarChart3,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { apiClient } from '@/lib/api-client'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Period = '1h' | '24h' | '7d' | '30d'

interface CostTrendPoint {
  date: string
  cost: number
}

interface TopModelSummary {
  model_id: string
  total_cost: number
  request_count: number
}

interface LLMSummary {
  total_requests: number
  total_tokens: number
  total_cost: number
  avg_latency_ms: number
  error_rate: number
  top_models: TopModelSummary[]
  cost_trend: CostTrendPoint[]
}

interface UsageByModel {
  key: string
  request_count: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
  total_cost: number
}

interface Recommendation {
  type: string
  title: string
  description: string
  potential_savings: number
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toString()
}

function formatCost(n: number): string {
  return `$${n.toFixed(2)}`
}

function formatLatency(ms: number): string {
  if (ms >= 1_000) return `${(ms / 1_000).toFixed(1)}s`
  return `${Math.round(ms)}ms`
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AnalyticsLLMUsage() {
  const [period, setPeriod] = useState<Period>('24h')

  // --- Queries ---

  const {
    data: summary,
    isLoading: summaryLoading,
  } = useQuery<LLMSummary>({
    queryKey: ['llm-summary', period],
    queryFn: () => apiClient.get(`/api/analytics/llm/summary?period=${period}`),
  })

  const {
    data: usageByModel,
    isLoading: usageLoading,
  } = useQuery<UsageByModel[]>({
    queryKey: ['llm-usage', period],
    queryFn: () => apiClient.get(`/api/analytics/llm/usage?period=${period}&group_by=model`),
  })

  const {
    data: recommendations,
    isLoading: recsLoading,
  } = useQuery<Recommendation[]>({
    queryKey: ['llm-recommendations'],
    queryFn: () => apiClient.get('/api/analytics/llm/recommendations'),
  })

  // --- Derived ---

  const costTrend = summary?.cost_trend ?? []
  const maxCost = Math.max(...costTrend.map((d) => d.cost), 0.01)

  const sortedModels = [...(usageByModel ?? [])].sort(
    (a, b) => b.total_cost - a.total_cost,
  )

  const isLoading = summaryLoading

  // --- Period selector ---

  const periods: { value: Period; label: string }[] = [
    { value: '1h', label: '1h' },
    { value: '24h', label: '24h' },
    { value: '7d', label: '7d' },
    { value: '30d', label: '30d' },
  ]

  // --- Render ---

  return (
    <div className="space-y-6">
      {/* Period selector */}
      <div className="flex items-center gap-2">
        <BarChart3 className="h-5 w-5 text-muted-foreground" />
        <span className="text-sm font-medium text-muted-foreground mr-2">
          Period
        </span>
        {periods.map((p) => (
          <Button
            key={p.value}
            size="sm"
            variant={period === p.value ? 'default' : 'outline'}
            onClick={() => setPeriod(p.value)}
            className="min-w-[48px]"
          >
            {p.label}
          </Button>
        ))}
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <SummaryCard
          icon={Activity}
          iconColor="text-blue-400"
          label="Total Requests"
          value={isLoading ? '...' : (summary?.total_requests ?? 0).toLocaleString()}
        />
        <SummaryCard
          icon={Zap}
          iconColor="text-purple-400"
          label="Total Tokens"
          value={isLoading ? '...' : formatTokens(summary?.total_tokens ?? 0)}
        />
        <SummaryCard
          icon={DollarSign}
          iconColor="text-green-400"
          label="Total Cost"
          value={isLoading ? '...' : formatCost(summary?.total_cost ?? 0)}
        />
        <SummaryCard
          icon={Clock}
          iconColor="text-warning"
          label="Avg Latency"
          value={isLoading ? '...' : formatLatency(summary?.avg_latency_ms ?? 0)}
        />
      </div>

      {/* Cost Trend bar chart */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <DollarSign className="h-5 w-5 text-green-400" />
            Cost Trend
          </CardTitle>
        </CardHeader>
        <CardContent>
          {summaryLoading ? (
            <div className="h-48 flex items-center justify-center text-muted-foreground text-sm">
              Loading chart data...
            </div>
          ) : costTrend.length === 0 ? (
            <div className="h-48 flex flex-col items-center justify-center text-muted-foreground">
              <BarChart3 className="h-10 w-10 mb-3 opacity-40" />
              <p className="text-sm">No cost data for this period</p>
            </div>
          ) : (
            <div className="flex items-end gap-1 h-48">
              {costTrend.map((point, i) => {
                const heightPct = (point.cost / maxCost) * 100
                return (
                  <div
                    key={i}
                    className="flex-1 flex flex-col items-center justify-end h-full group"
                  >
                    {/* Tooltip */}
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity text-xs text-muted-foreground mb-1 whitespace-nowrap">
                      {formatCost(point.cost)}
                    </div>
                    {/* Bar */}
                    <div
                      className="w-full rounded-t bg-gradient-to-t from-green-500/80 to-green-400/60 hover:from-green-400 hover:to-green-300 transition-all duration-220 min-h-[2px]"
                      style={{ height: `${Math.max(heightPct, 1)}%` }}
                    />
                    {/* Date label */}
                    <span className="text-[10px] text-muted-foreground mt-1.5 truncate w-full text-center">
                      {point.date.slice(5)}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Top Models table */}
      <Card className="glass-card overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Zap className="h-5 w-5 text-purple-400" />
            Top Models
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {usageLoading ? (
            <div className="p-8 text-center text-muted-foreground text-sm">
              Loading model data...
            </div>
          ) : sortedModels.length === 0 ? (
            <div className="p-12 text-center text-muted-foreground">
              <Zap className="h-10 w-10 mx-auto mb-3 opacity-40" />
              <p className="text-sm">No model usage data for this period</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="border-border/50">
                  <TableHead className="text-xs">Model</TableHead>
                  <TableHead className="text-xs">Provider</TableHead>
                  <TableHead className="text-xs text-right">Requests</TableHead>
                  <TableHead className="text-xs text-right hidden md:table-cell">Tokens</TableHead>
                  <TableHead className="text-xs text-right">Cost</TableHead>
                  <TableHead className="text-xs text-right hidden lg:table-cell">Avg Latency</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedModels.map((model) => {
                  const provider = deriveProvider(model.key)
                  return (
                    <TableRow key={model.key} className="border-border/30">
                      <TableCell>
                        <Badge variant="secondary" className="font-mono text-xs">
                          {model.key}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {provider}
                      </TableCell>
                      <TableCell className="text-sm text-right">
                        {model.request_count.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-sm text-right hidden md:table-cell">
                        {formatTokens(model.total_tokens)}
                      </TableCell>
                      <TableCell className="text-sm text-right font-medium">
                        {formatCost(model.total_cost)}
                      </TableCell>
                      <TableCell className="text-sm text-right hidden lg:table-cell text-muted-foreground">
                        {model.request_count > 0
                          ? formatLatency(
                              (summary?.avg_latency_ms ?? 0)
                            )
                          : '-'}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Optimization Recommendations */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Lightbulb className="h-5 w-5 text-yellow-400" />
            Optimization Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          {recsLoading ? (
            <div className="text-center text-muted-foreground text-sm py-8">
              Loading recommendations...
            </div>
          ) : !recommendations || recommendations.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-muted-foreground">
              <Lightbulb className="h-10 w-10 mb-3 opacity-40" />
              <p className="text-sm">No optimization recommendations at this time</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendations.map((rec, i) => (
                <div
                  key={i}
                  className="rounded-xl border border-border/40 bg-secondary/20 p-4 space-y-2"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <TrendingDown className="h-4 w-4 text-green-400 shrink-0 mt-0.5" />
                      <span className="text-sm font-medium leading-tight">
                        {rec.title}
                      </span>
                    </div>
                    {rec.potential_savings > 0 && (
                      <Badge
                        variant="secondary"
                        className="shrink-0 bg-green-500/10 text-green-400 border-green-500/20"
                      >
                        Save {formatCost(rec.potential_savings)}
                      </Badge>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {rec.description}
                  </p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SummaryCard({
  icon: Icon,
  iconColor,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>
  iconColor: string
  label: string
  value: string
}) {
  return (
    <Card className="glass-card">
      <CardContent className="p-5">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-secondary/40 p-2.5">
            <Icon className={`h-5 w-5 ${iconColor}`} />
          </div>
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground truncate">{label}</p>
            <p className="text-xl font-semibold tracking-tight">{value}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function deriveProvider(modelId: string): string {
  const id = modelId.toLowerCase()
  if (id.startsWith('gpt-') || id.startsWith('o1') || id.startsWith('o3') || id.startsWith('o4')) return 'OpenAI'
  if (id.startsWith('claude')) return 'Anthropic'
  if (id.startsWith('gemini') || id.startsWith('palm')) return 'Google'
  if (id.startsWith('llama') || id.startsWith('meta')) return 'Meta'
  if (id.startsWith('mistral') || id.startsWith('mixtral')) return 'Mistral'
  if (id.startsWith('command') || id.startsWith('cohere')) return 'Cohere'
  if (id.startsWith('deepseek')) return 'DeepSeek'
  return 'Other'
}
