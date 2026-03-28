'use client'

import { TrendingUp, Loader2, Bot } from 'lucide-react'
import { useAgentPerformance } from '@/hooks/use-kpi-api'
import { cn } from '@/lib/utils'

interface AgentPerformanceWidgetProps {
  period: string
  className?: string
}

function rateColor(rate: number): string {
  if (rate >= 90) return 'bg-emerald-400'
  if (rate >= 70) return 'bg-amber-400'
  return 'bg-red-400'
}

function rateTextColor(rate: number): string {
  if (rate >= 90) return 'text-emerald-400'
  if (rate >= 70) return 'text-amber-400'
  return 'text-red-400'
}

export function AgentPerformanceWidget({ period, className }: AgentPerformanceWidgetProps) {
  const { data, isLoading } = useAgentPerformance(period)

  const agents = data?.agents ?? []

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-[hsl(var(--info))]" />
          <h3 className="text-sm font-semibold">Agent Performance</h3>
        </div>
        {!isLoading && agents.length > 0 && (
          <span className="text-[10px] text-muted-foreground">{agents.length} active</span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-1.5">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : agents.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <Bot className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">No agent activity</p>
          </div>
        ) : (
          agents.map((agent) => (
            <div key={agent.agent_id} className="flex items-center gap-2 py-1">
              <span className="text-xs truncate flex-1 min-w-0">{agent.name}</span>
              <div className="w-20 h-1.5 bg-secondary/50 rounded-full overflow-hidden shrink-0">
                <div
                  className={cn('h-full rounded-full', rateColor(agent.success_rate))}
                  style={{ width: `${agent.success_rate}%` }}
                />
              </div>
              <span className={cn('text-[11px] font-medium w-10 text-right shrink-0', rateTextColor(agent.success_rate))}>
                {agent.success_rate}%
              </span>
              <span className="text-[10px] text-muted-foreground w-8 text-right shrink-0">
                {agent.tasks_completed}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
