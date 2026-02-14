'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  DollarSign,
  Zap,
  TrendingUp,
  AlertTriangle,
  Bot,
  GitCompare,
  X,
  ChevronDown,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { StatsBar } from '@/components/shared/stats-bar'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
  Cell,
} from 'recharts'
import { useCostAnalyticsUnified, useModelComparison, useCostProjections } from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

const COMPARISON_COLORS = ['#60B5FF', '#72BF78', '#ff6b35', '#a78bfa']

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}

export function AnalyticsCosts({ days }: Props) {
  const { data, isLoading } = useCostAnalyticsUnified(days)
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([])
  const [comparisonPeriod, setComparisonPeriod] = useState('30d')
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false)
  const [projectionPeriod, setProjectionPeriod] = useState('30d')
  const { data: comparisonData, isLoading: comparisonLoading } = useModelComparison(selectedModelIds, comparisonPeriod)
  const { data: projections, isLoading: projectionsLoading } = useCostProjections(projectionPeriod)

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="glass-card"><CardContent className="p-6"><Skeleton className="h-8 w-16 mb-2" /><Skeleton className="h-4 w-24" /></CardContent></Card>
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StatsBar stats={[
        { label: 'Total Tokens', value: formatNumber(data?.summary?.totalTokens || 0), change: 'Input + Output', icon: Zap, iconColor: 'text-primary' },
        { label: 'Total LLM Cost', value: `$${(data?.summary?.totalCost || 0).toFixed(2)}`, change: 'This period', icon: DollarSign, iconColor: 'text-[hsl(var(--success))]' },
        { label: 'Cost per Task', value: `$${(data?.summary?.costPerTask || 0).toFixed(4)}`, change: `${data?.summary?.totalRequests || 0} requests`, icon: TrendingUp, iconColor: 'text-[hsl(var(--info))]' },
        { label: 'Most Expensive', value: data?.summary?.mostExpensiveAgent?.name || 'None', change: data?.summary?.mostExpensiveAgent ? `$${data.summary.mostExpensiveAgent.cost.toFixed(2)}` : 'No data', icon: AlertTriangle, iconColor: 'text-[hsl(var(--agent))]' },
      ]} loading={isLoading} />

      {/* Cost Trend Chart */}
      {data && data.costTrend?.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DollarSign className="w-5 h-5 text-green-400" />
              Cost Over Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.costTrend}>
                  <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Area type="monotone" dataKey="total_cost" stroke="#72BF78" fill="#72BF78" fillOpacity={0.3} name="Daily Cost ($)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Cost Projections */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-400" />
              Cost Projections
            </span>
            <select
              value={projectionPeriod}
              onChange={(e) => setProjectionPeriod(e.target.value)}
              className="text-xs bg-secondary/50 border border-border/50 rounded-md px-2 py-1 text-foreground"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
            </select>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {projectionsLoading ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-20 w-full" />
                ))}
              </div>
              <Skeleton className="h-48 w-full" />
            </div>
          ) : projections ? (
            <>
              {/* Projection Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="rounded-lg border border-border/30 p-4">
                  <p className="text-xs text-muted-foreground mb-1">Current Period Cost</p>
                  <p className="text-xl font-semibold">${projections.current_period_cost.toFixed(2)}</p>
                </div>
                <div className="rounded-lg border border-border/30 p-4">
                  <p className="text-xs text-muted-foreground mb-1">Daily Average</p>
                  <p className="text-xl font-semibold">${projections.daily_average.toFixed(4)}</p>
                </div>
                <div className="rounded-lg border border-border/30 p-4">
                  <p className="text-xs text-muted-foreground mb-1">Projected Monthly</p>
                  <p className="text-2xl font-bold text-primary">${projections.projected_monthly.toFixed(2)}</p>
                </div>
                <div className="rounded-lg border border-border/30 p-4">
                  <p className="text-xs text-muted-foreground mb-1">vs Previous Period</p>
                  {projections.change_percent != null ? (
                    <p className={`text-xl font-semibold ${projections.change_percent > 0 ? 'text-red-400' : projections.change_percent < 0 ? 'text-green-400' : 'text-muted-foreground'}`}>
                      {projections.change_percent > 0 ? '+' : ''}{projections.change_percent.toFixed(1)}%
                    </p>
                  ) : (
                    <p className="text-xl font-semibold text-muted-foreground">—</p>
                  )}
                </div>
              </div>

              {/* Projected Cost by Model - Horizontal Bar Chart */}
              {projections.projected_by_model && projections.projected_by_model.length > 0 && (
                <div>
                  <p className="text-xs text-muted-foreground mb-3">Projected Monthly Cost by Model</p>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={projections.projected_by_model}
                        layout="vertical"
                        margin={{ left: 20, right: 20, top: 5, bottom: 5 }}
                      >
                        <XAxis
                          type="number"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                          tickFormatter={(v: number) => `$${v.toFixed(2)}`}
                        />
                        <YAxis
                          type="category"
                          dataKey="key"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                          width={140}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px',
                            fontSize: '12px',
                          }}
                          formatter={(value: number) => [`$${value.toFixed(4)}`, 'Projected Monthly']}
                        />
                        <Bar dataKey="projected_monthly_cost" radius={[0, 4, 4, 0]}>
                          {projections.projected_by_model.map((_entry, index) => (
                            <Cell key={index} fill={COMPARISON_COLORS[index % COMPARISON_COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-8">
              <TrendingUp className="w-10 h-10 mx-auto mb-3 opacity-50 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No projection data available — usage data needed</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Token Usage by Model */}
      <Card className="glass-card overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-purple-400" />
            Token Usage by LLM Model
          </CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">LLM Model</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Requests</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden md:table-cell">Input Tokens</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden md:table-cell">Output Tokens</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Total Cost</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden lg:table-cell">Avg Cost/Req</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden lg:table-cell">Used By</th>
              </tr>
            </thead>
            <tbody>
              {(!data?.byModel || data.byModel.length === 0) ? (
                <tr>
                  <td colSpan={7} className="p-12 text-center text-muted-foreground">
                    <Zap className="w-10 h-10 mx-auto mb-3 opacity-50" />
                    <p>No token usage data yet</p>
                  </td>
                </tr>
              ) : (
                data.byModel.map((model) => (
                  <tr key={model.model} className="border-b border-border/30 hover:bg-secondary/20 transition-colors">
                    <td className="p-4">
                      <Badge variant="secondary" className="font-mono text-xs">{model.model}</Badge>
                    </td>
                    <td className="p-4 text-sm">{formatNumber(model.requests)}</td>
                    <td className="p-4 text-sm hidden md:table-cell">{formatNumber(model.inputTokens)}</td>
                    <td className="p-4 text-sm hidden md:table-cell">{formatNumber(model.outputTokens)}</td>
                    <td className="p-4 text-sm font-medium">${model.totalCost.toFixed(2)}</td>
                    <td className="p-4 text-sm hidden lg:table-cell">${model.avgCostPerRequest.toFixed(4)}</td>
                    <td className="p-4 text-sm hidden lg:table-cell">{model.agentCount} agent{model.agentCount !== 1 ? 's' : ''}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Per-Agent Cost Breakdown */}
      <Card className="glass-card overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-orange-400" />
            Cost by Agent
          </CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Agent</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Model</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Tokens</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Cost</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden md:table-cell">Requests</th>
              </tr>
            </thead>
            <tbody>
              {(!data?.byAgent || data.byAgent.length === 0) ? (
                <tr>
                  <td colSpan={5} className="p-8 text-center text-muted-foreground text-sm">No agent cost data</td>
                </tr>
              ) : (
                data.byAgent.filter(a => a.cost > 0 || a.tokens > 0).map((agent) => (
                  <tr key={agent.id} className="border-b border-border/30 hover:bg-secondary/20 transition-colors">
                    <td className="p-4 font-medium text-sm">{agent.name}</td>
                    <td className="p-4">
                      <Badge variant="secondary" className="font-mono text-xs">{agent.model}</Badge>
                    </td>
                    <td className="p-4 text-sm">{formatNumber(agent.tokens)}</td>
                    <td className="p-4 text-sm font-medium">${agent.cost.toFixed(2)}</td>
                    <td className="p-4 text-sm hidden md:table-cell">{agent.requests}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Model Comparison */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <GitCompare className="w-5 h-5 text-blue-400" />
              Model Comparison
            </span>
            <select
              value={comparisonPeriod}
              onChange={(e) => setComparisonPeriod(e.target.value)}
              className="text-xs bg-secondary/50 border border-border/50 rounded-md px-2 py-1 text-foreground"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
            </select>
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
                  className="font-mono text-xs flex items-center gap-1"
                  style={{ borderColor: COMPARISON_COLORS[idx] }}
                >
                  <span className="w-2 h-2 rounded-full" style={{ background: COMPARISON_COLORS[idx] }} />
                  {id}
                  <button
                    onClick={() => setSelectedModelIds(prev => prev.filter(m => m !== id))}
                    className="ml-1 hover:text-destructive"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </Badge>
              ))}
              {selectedModelIds.length < 4 && (
                <div className="relative">
                  <button
                    onClick={() => setModelDropdownOpen(prev => !prev)}
                    className="inline-flex items-center gap-1 text-xs px-3 py-1.5 rounded-md border border-dashed border-border/50 text-muted-foreground hover:text-foreground hover:border-border transition-colors"
                  >
                    Add Model
                    <ChevronDown className="w-3 h-3" />
                  </button>
                  {modelDropdownOpen && (
                    <div className="absolute z-50 mt-1 w-64 max-h-48 overflow-y-auto rounded-md border border-border bg-card shadow-lg">
                      {data?.byModel
                        ?.filter(m => !selectedModelIds.includes(m.model))
                        .map(m => (
                          <button
                            key={m.model}
                            className="w-full text-left px-3 py-2 text-xs hover:bg-secondary/50 transition-colors font-mono"
                            onClick={() => {
                              setSelectedModelIds(prev => [...prev, m.model])
                              setModelDropdownOpen(false)
                            }}
                          >
                            {m.model}
                            <span className="text-muted-foreground ml-2">({m.requests} req)</span>
                          </button>
                        ))}
                      {(!data?.byModel || data.byModel.filter(m => !selectedModelIds.includes(m.model)).length === 0) && (
                        <p className="px-3 py-2 text-xs text-muted-foreground">No more models available</p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Comparison content */}
          {selectedModelIds.length === 0 ? (
            <div className="text-center py-8">
              <GitCompare className="w-10 h-10 mx-auto mb-3 opacity-50 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Select models above to compare cost, performance, and capabilities</p>
            </div>
          ) : comparisonLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-48 w-full" />
              <Skeleton className="h-64 w-full" />
            </div>
          ) : comparisonData && comparisonData.length > 0 ? (
            <>
              {/* Comparison Table */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border/50">
                      <th className="text-left p-3 text-xs font-medium text-muted-foreground">Metric</th>
                      {comparisonData.map((model, idx) => (
                        <th key={model.model_id} className="text-left p-3 text-xs font-medium" style={{ color: COMPARISON_COLORS[idx] }}>
                          {model.display_name || model.model_id}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border/30">
                      <td className="p-3 text-xs text-muted-foreground">Provider</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3 text-sm">{m.provider || '—'}</td>
                      ))}
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="p-3 text-xs text-muted-foreground">Input Cost / 1K</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3 text-sm font-mono">
                          {m.input_cost_per_1k != null ? `$${m.input_cost_per_1k.toFixed(4)}` : '—'}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="p-3 text-xs text-muted-foreground">Output Cost / 1K</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3 text-sm font-mono">
                          {m.output_cost_per_1k != null ? `$${m.output_cost_per_1k.toFixed(4)}` : '—'}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="p-3 text-xs text-muted-foreground">Total Requests</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3 text-sm">{formatNumber(m.total_requests)}</td>
                      ))}
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="p-3 text-xs text-muted-foreground">Total Tokens</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3 text-sm">{formatNumber(m.total_tokens)}</td>
                      ))}
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="p-3 text-xs text-muted-foreground">Total Cost</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3 text-sm font-medium">${m.total_cost.toFixed(2)}</td>
                      ))}
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="p-3 text-xs text-muted-foreground">Avg Latency</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3 text-sm">
                          {m.avg_latency_ms != null ? `${m.avg_latency_ms.toFixed(0)}ms` : '—'}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="p-3 text-xs text-muted-foreground">Error Rate</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3 text-sm">
                          <span className={m.error_rate > 5 ? 'text-red-400' : m.error_rate > 1 ? 'text-yellow-400' : 'text-green-400'}>
                            {m.error_rate.toFixed(1)}%
                          </span>
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="p-3 text-xs text-muted-foreground">Success Rate</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3 text-sm">{m.success_rate.toFixed(1)}%</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 text-xs text-muted-foreground">Capabilities</td>
                      {comparisonData.map(m => (
                        <td key={m.model_id} className="p-3">
                          <div className="flex flex-wrap gap-1">
                            {m.capabilities && Object.entries(m.capabilities)
                              .filter(([, v]) => v === true || (typeof v === 'number' && v > 0))
                              .map(([cap]) => (
                                <Badge key={cap} variant="outline" className="text-[10px] px-1.5 py-0">
                                  {cap}
                                </Badge>
                              ))}
                            {(!m.capabilities || Object.keys(m.capabilities).length === 0) && (
                              <span className="text-xs text-muted-foreground">—</span>
                            )}
                          </div>
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Radar Chart for capabilities */}
              {(() => {
                // Build radar data from capabilities across selected models
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
                    <p className="text-xs text-muted-foreground mb-2">Capability Comparison</p>
                    <div className="h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart data={radarData}>
                          <PolarGrid stroke="hsl(var(--border))" />
                          <PolarAngleAxis
                            dataKey="capability"
                            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                          />
                          <PolarRadiusAxis tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} />
                          {comparisonData.map((m, idx) => (
                            <Radar
                              key={m.model_id}
                              name={m.display_name || m.model_id}
                              dataKey={m.model_id}
                              stroke={COMPARISON_COLORS[idx]}
                              fill={COMPARISON_COLORS[idx]}
                              fillOpacity={0.15}
                            />
                          ))}
                          <Legend
                            wrapperStyle={{ fontSize: '11px' }}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'hsl(var(--card))',
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px',
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
    </div>
  )
}
