'use client'

import { useMemo } from 'react'
import { PieChart as PieChartIcon, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
} from 'recharts'
import { useBoardStats } from '@/hooks/use-activity-api'
import { cn } from '@/lib/utils'

const STATUS_COLORS: Record<string, string> = {
  inbox: 'hsl(var(--muted-foreground))',
  assigned: 'hsl(var(--agent))',
  in_progress: 'hsl(var(--info))',
  review: 'hsl(var(--warning))',
  done: 'hsl(var(--success))',
}

const STATUS_LABELS: Record<string, string> = {
  inbox: 'Inbox',
  assigned: 'Assigned',
  in_progress: 'In Progress',
  review: 'In Review',
  done: 'Done',
}

interface StatusOverviewWidgetProps {
  period: string
  onViewAll?: () => void
  className?: string
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.[0]) return null
  const { name, value } = payload[0]
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 text-xs shadow-lg">
      <span className="font-medium">{name}</span>: {value}
    </div>
  )
}

export function StatusOverviewWidget({ period, onViewAll, className }: StatusOverviewWidgetProps) {
  const { data, isLoading } = useBoardStats(period)

  const chartData = useMemo(() => {
    if (!data?.columns) return []
    return data.columns
      .filter((c) => c.count > 0)
      .map((c) => ({
        name: STATUS_LABELS[c.status] ?? c.status,
        value: c.count,
        color: STATUS_COLORS[c.status] ?? 'hsl(var(--muted-foreground))',
      }))
  }, [data])

  const totalTasks = data?.total_tasks ?? 0

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <PieChartIcon className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold">Status Overview</h3>
        </div>
        {onViewAll && (
          <Button variant="ghost" size="sm" className="text-xs h-6" onClick={onViewAll}>
            View all →
          </Button>
        )}
      </div>

      <div className="flex-1 flex items-center px-4 py-3">
        {isLoading ? (
          <div className="flex items-center justify-center w-full py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex flex-col items-center justify-center w-full py-8 text-muted-foreground">
            <PieChartIcon className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">No tasks yet</p>
          </div>
        ) : (
          <div className="flex items-center gap-6 w-full">
            {/* Donut chart */}
            <div className="relative w-[160px] h-[160px] shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={72}
                    paddingAngle={2}
                    dataKey="value"
                    stroke="none"
                  >
                    {chartData.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              {/* Center label */}
              <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                <span className="text-2xl font-bold leading-none">{totalTasks}</span>
                <span className="text-[10px] text-muted-foreground">Total tasks</span>
              </div>
            </div>

            {/* Legend */}
            <div className="flex flex-col gap-2 flex-1 min-w-0">
              {(data?.columns ?? []).map((c) => (
                <div key={c.status} className="flex items-center gap-2">
                  <div
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: STATUS_COLORS[c.status] }}
                  />
                  <span className="text-xs text-muted-foreground truncate flex-1">
                    {STATUS_LABELS[c.status] ?? c.status}
                  </span>
                  <span className="text-xs font-medium">{c.count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
