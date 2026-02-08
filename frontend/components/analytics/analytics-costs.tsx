'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  DollarSign,
  Zap,
  TrendingUp,
  AlertTriangle,
  Bot,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { StatsBar } from '@/components/shared/stats-bar'
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts'
import { useCostAnalyticsUnified } from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}

export function AnalyticsCosts({ days }: Props) {
  const { data, isLoading } = useCostAnalyticsUnified(days)

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
      {data?.costTrend?.length > 0 && (
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
    </div>
  )
}
