'use client'

import { Layers, Loader2, RefreshCw, CheckSquare, FolderKanban } from 'lucide-react'
import { useBoardStats } from '@/hooks/use-activity-api'
import { cn } from '@/lib/utils'

const TYPE_CONFIG: Record<string, { icon: typeof CheckSquare; color: string; label: string }> = {
  routine: { icon: RefreshCw, color: 'hsl(var(--agent))', label: 'Routine' },
  recipe: { icon: CheckSquare, color: 'hsl(var(--info))', label: 'Task' },
  mission: { icon: FolderKanban, color: 'hsl(var(--success))', label: 'Project' },
}

interface TypesOfWorkWidgetProps {
  period: string
  className?: string
}

export function TypesOfWorkWidget({ period, className }: TypesOfWorkWidgetProps) {
  const { data, isLoading } = useBoardStats(period)
  const types = data?.types ?? []

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-[hsl(var(--info))]" />
          <h3 className="text-sm font-semibold">Types of Work</h3>
        </div>
      </div>

      <div className="flex-1 px-4 py-3">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : types.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <Layers className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">No work items yet</p>
          </div>
        ) : (
          <div className="space-y-4 pt-2">
            {/* Header row */}
            <div className="flex items-center text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
              <span className="flex-1">Type</span>
              <span className="w-32 text-right mr-12">Distribution</span>
            </div>

            {types.map((t) => {
              const config = TYPE_CONFIG[t.type] ?? TYPE_CONFIG.recipe
              const TypeIcon = config.icon
              const pct = Math.round(t.percentage)

              return (
                <div key={t.type} className="flex items-center gap-3">
                  <TypeIcon className="w-4 h-4 shrink-0" style={{ color: config.color }} />
                  <span className="text-sm font-medium w-20 shrink-0">{config.label}</span>
                  <div className="flex-1 h-5 bg-secondary/30 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.max(pct, 2)}%`,
                        backgroundColor: config.color,
                        opacity: 0.7,
                      }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-10 text-right shrink-0">
                    {pct}%
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
