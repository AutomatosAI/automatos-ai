'use client'

import { BookOpen, Loader2 } from 'lucide-react'
import { usePlaybookMetrics } from '@/hooks/use-kpi-api'
import { cn } from '@/lib/utils'

interface PlaybookMetricsWidgetProps {
  period: string
  className?: string
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  if (mins < 60) return `${mins}m`
  return `${Math.floor(mins / 60)}h ${mins % 60}m`
}

export function PlaybookMetricsWidget({ period, className }: PlaybookMetricsWidgetProps) {
  const { data, isLoading } = usePlaybookMetrics(period)

  const playbooks = data?.playbooks ?? []

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-violet-400" />
          <h3 className="text-sm font-semibold">Playbook Metrics</h3>
        </div>
        {!isLoading && playbooks.length > 0 && (
          <span className="text-[10px] text-muted-foreground">{playbooks.length} playbooks</span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-2">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : playbooks.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <BookOpen className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">No playbook runs yet</p>
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-muted-foreground border-b border-border/30">
                <th className="text-left py-1.5 font-medium">Name</th>
                <th className="text-right py-1.5 font-medium w-12">Runs</th>
                <th className="text-right py-1.5 font-medium w-14">Success</th>
                <th className="text-right py-1.5 font-medium w-14">Avg Time</th>
              </tr>
            </thead>
            <tbody>
              {playbooks.map((pb) => (
                <tr key={pb.workflow_id} className="border-b border-border/10">
                  <td className="py-1.5 truncate max-w-[120px]">{pb.name}</td>
                  <td className="py-1.5 text-right text-muted-foreground">{pb.runs}</td>
                  <td className={cn(
                    'py-1.5 text-right font-medium',
                    pb.success_pct >= 90 ? 'text-emerald-400' : pb.success_pct >= 70 ? 'text-amber-400' : 'text-red-400'
                  )}>
                    {pb.success_pct}%
                  </td>
                  <td className="py-1.5 text-right text-muted-foreground">
                    {formatDuration(pb.avg_duration_seconds)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
