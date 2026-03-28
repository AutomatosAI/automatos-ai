'use client'

import { DollarSign, Loader2, TrendingUp, TrendingDown } from 'lucide-react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Tooltip,
  XAxis,
} from 'recharts'
import { useCostTracker } from '@/hooks/use-kpi-api'
import { cn } from '@/lib/utils'

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.[0]) return null
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="text-muted-foreground">{label}</p>
      <p className="font-medium">${payload[0].value.toFixed(4)}</p>
    </div>
  )
}

interface CostTrackerWidgetProps {
  period: string
  className?: string
}

export function CostTrackerWidget({ period, className }: CostTrackerWidgetProps) {
  const { data, isLoading } = useCostTracker(period)

  const totalCost = data?.total_cost ?? 0
  const changePct = data?.change_pct ?? 0
  const trend = data?.daily_trend ?? []
  const topAgents = data?.top_agents ?? []
  const maxAgentCost = Math.max(...topAgents.map((a) => a.cost), 0.01)

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-semibold">Cost Tracker</h3>
        </div>
        {!isLoading && (
          <div className="flex items-center gap-1 text-xs">
            {changePct !== 0 && (
              changePct > 0 ? (
                <TrendingUp className="w-3 h-3 text-red-400" />
              ) : (
                <TrendingDown className="w-3 h-3 text-emerald-400" />
              )
            )}
            <span className={cn(
              'font-medium',
              changePct > 0 ? 'text-red-400' : changePct < 0 ? 'text-emerald-400' : 'text-muted-foreground'
            )}>
              {changePct > 0 ? '+' : ''}{changePct}%
            </span>
          </div>
        )}
      </div>

      <div className="flex-1 px-4 py-3 space-y-3 overflow-hidden">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            {/* Big number */}
            <div className="text-center">
              <span className="text-3xl font-bold">${totalCost.toFixed(2)}</span>
              <p className="text-[10px] text-muted-foreground mt-0.5">Total spend ({period})</p>
            </div>

            {/* Sparkline */}
            {trend.length > 1 && (
              <div className="h-[80px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trend}>
                    <defs>
                      <linearGradient id="costGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--success))" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="hsl(var(--success))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="date" hide />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="cost"
                      stroke="hsl(var(--success))"
                      fill="url(#costGradient)"
                      strokeWidth={1.5}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Top agents */}
            {topAgents.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Top spenders</p>
                {topAgents.map((agent) => (
                  <div key={agent.name} className="flex items-center gap-2">
                    <span className="text-xs truncate flex-1 min-w-0">{agent.name}</span>
                    <div className="w-24 h-1.5 bg-secondary/50 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-400/60 rounded-full"
                        style={{ width: `${(agent.cost / maxAgentCost) * 100}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-muted-foreground shrink-0 w-12 text-right">
                      ${agent.cost.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
