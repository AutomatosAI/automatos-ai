'use client'

import { useMemo } from 'react'
import { BarChart3, Loader2 } from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from 'recharts'
import { useBoardStats } from '@/hooks/use-activity-api'
import { cn } from '@/lib/utils'

const PRIORITY_COLORS: Record<string, string> = {
  urgent: 'hsl(var(--destructive))',
  high: 'hsl(var(--warning))',
  medium: 'hsl(var(--info))',
  low: 'hsl(var(--muted-foreground))',
}

const PRIORITY_ORDER = ['urgent', 'high', 'medium', 'low']

interface PriorityBreakdownWidgetProps {
  period: string
  className?: string
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.[0]) return null
  const { name, value } = payload[0].payload
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 text-xs shadow-lg">
      <span className="font-medium capitalize">{name}</span>: {value} tasks
    </div>
  )
}

export function PriorityBreakdownWidget({ period, className }: PriorityBreakdownWidgetProps) {
  const { data, isLoading } = useBoardStats(period)

  const chartData = useMemo(() => {
    if (!data?.priorities) return []
    const byPriority = new Map(data.priorities.map((p) => [p.priority, p.count]))
    return PRIORITY_ORDER.map((p) => ({
      name: p,
      value: byPriority.get(p) ?? 0,
      color: PRIORITY_COLORS[p],
    }))
  }, [data])

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-[hsl(var(--warning))]" />
          <h3 className="text-sm font-semibold">Priority Breakdown</h3>
        </div>
      </div>

      <div className="flex-1 px-4 py-3">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : chartData.every((d) => d.value === 0) ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <BarChart3 className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">No priority data yet</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
                strokeOpacity={0.3}
                vertical={false}
              />
              <XAxis
                dataKey="name"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))', textTransform: 'capitalize' } as any}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                allowDecimals={false}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'hsl(var(--secondary))', opacity: 0.3 }} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={48}>
                {chartData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
