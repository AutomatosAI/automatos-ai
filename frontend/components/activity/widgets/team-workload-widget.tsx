'use client'

import { Users, Loader2, Bot } from 'lucide-react'
import { PremiumIcon } from '@/components/shared'
import { useBoardStats } from '@/hooks/use-activity-api'
import { cn } from '@/lib/utils'

interface TeamWorkloadWidgetProps {
  period: string
  className?: string
}

export function TeamWorkloadWidget({ period, className }: TeamWorkloadWidgetProps) {
  const { data, isLoading } = useBoardStats(period)
  const workload = data?.workload ?? []
  const maxTasks = Math.max(...workload.map((w) => w.task_count), 1)

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-[hsl(var(--agent))]" />
          <h3 className="text-sm font-semibold">Team Workload</h3>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : workload.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <Users className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">No workload data yet</p>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Header row */}
            <div className="flex items-center text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
              <span className="flex-1">Assignee</span>
              <span className="w-32 text-right mr-8">Work distribution</span>
            </div>

            {workload.map((agent) => {
              const pct = Math.round((agent.task_count / maxTasks) * 100)

              return (
                <div key={agent.agent_id} className="flex items-center gap-3">
                  <div className="w-6 h-6 flex items-center justify-center shrink-0">
                    {agent.agent_icon ? (
                      <PremiumIcon name={agent.agent_icon} size={20} />
                    ) : (
                      <Bot className="w-4 h-4 text-primary" />
                    )}
                  </div>
                  <span className="text-sm font-medium w-24 truncate shrink-0">
                    {agent.agent_name}
                  </span>
                  <div className="flex-1 h-5 bg-secondary/30 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-[hsl(var(--agent))] transition-all duration-500"
                      style={{ width: `${Math.max(pct, 4)}%`, opacity: 0.7 }}
                    />
                  </div>
                  <span className="text-xs font-medium w-6 text-right shrink-0">
                    {agent.task_count}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
