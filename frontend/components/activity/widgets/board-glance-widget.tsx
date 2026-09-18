'use client'

/**
 * PRD-244 review batch 2 — "Board at a glance": the four slices of one read
 * (`useBoardStats`) that used to be four widgets — status, priority, type of
 * work, workload per agent. The Board tab is the full thing.
 */
import { useMemo } from 'react'
import { Bot, CheckSquare, FolderKanban, Loader2, PieChart as PieChartIcon, RefreshCw } from 'lucide-react'
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts'
import { Button } from '@/components/ui/button'
import { PremiumIcon } from '@/components/shared'
import { useBoardStats } from '@/hooks/use-activity-api'
import { cn } from '@/lib/utils'

const STATUS_COLORS: Record<string, string> = {
  inbox: 'hsl(var(--muted-foreground))',
  assigned: 'hsl(var(--agent))',
  in_progress: 'hsl(var(--info))',
  review: 'hsl(var(--warning))',
  blocked: 'hsl(var(--destructive))',
  done: 'hsl(var(--success))',
  failed: 'hsl(var(--destructive))',
  cancelled: 'hsl(var(--muted-foreground))',
}
const STATUS_LABELS: Record<string, string> = {
  inbox: 'Inbox', assigned: 'Assigned', in_progress: 'In Progress', review: 'In Review',
  blocked: 'Blocked', done: 'Done', failed: 'Failed', cancelled: 'Cancelled',
}
const PRIORITY_ORDER = ['urgent', 'high', 'medium', 'low'] as const
const PRIORITY_COLORS: Record<string, string> = {
  urgent: 'hsl(var(--destructive))', high: 'hsl(var(--warning))', medium: 'hsl(var(--info))', low: 'hsl(var(--muted-foreground))',
}
const TYPE_CONFIG: Record<string, { icon: typeof CheckSquare; color: string; label: string }> = {
  routine: { icon: RefreshCw, color: 'hsl(var(--agent))', label: 'Routine' },
  recipe: { icon: CheckSquare, color: 'hsl(var(--info))', label: 'Task' },
  mission: { icon: FolderKanban, color: 'hsl(var(--success))', label: 'Project' },
}

function Facet({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2" aria-label={title}>
      <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">{title}</p>
      {children}
    </div>
  )
}

function Bar({ label, count, max, color, icon }: { label: string; count: number; max: number; color: string; icon?: React.ReactNode }) {
  const pct = max > 0 ? Math.round((count / max) * 100) : 0
  return (
    <div className="flex items-center gap-2">
      {icon}
      <span className="text-xs w-20 truncate shrink-0">{label}</span>
      <div className="flex-1 h-3 bg-secondary/30 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${count > 0 ? Math.max(pct, 3) : 0}%`, backgroundColor: color, opacity: 0.75 }} />
      </div>
      <span className="text-xs font-medium w-6 text-right shrink-0">{count}</span>
    </div>
  )
}

function DonutTooltip({ active, payload }: any) {
  if (!active || !payload?.[0]) return null
  const { name, value } = payload[0]
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 text-xs shadow-lg">
      <span className="font-medium">{name}</span>: {value}
    </div>
  )
}

interface BoardGlanceWidgetProps {
  period: string
  onViewAll?: () => void
  className?: string
}

export function BoardGlanceWidget({ period, onViewAll, className }: BoardGlanceWidgetProps) {
  const { data, isLoading } = useBoardStats(period)
  const columns = data?.columns ?? []
  const total = data?.total_tasks ?? 0
  const chart = useMemo(
    () => columns.filter((c) => c.count > 0).map((c) => ({ name: STATUS_LABELS[c.status] ?? c.status, value: c.count, color: STATUS_COLORS[c.status] ?? 'hsl(var(--muted-foreground))' })),
    [columns],
  )
  const priorities = new Map((data?.priorities ?? []).map((p) => [p.priority, p.count]))
  const priorityMax = Math.max(...PRIORITY_ORDER.map((p) => priorities.get(p) ?? 0), 1)
  const types = data?.types ?? []
  const typeMax = Math.max(...types.map((t) => t.count), 1)
  const workload = data?.workload ?? []
  const workMax = Math.max(...workload.map((w) => w.task_count), 1)

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <PieChartIcon className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold">Board at a glance</h3>
          {!isLoading && total > 0 && <span className="text-xs text-muted-foreground">{total} task{total === 1 ? '' : 's'}</span>}
        </div>
        {onViewAll && (
          <Button variant="ghost" size="sm" className="text-xs h-6" onClick={onViewAll}>
            View board →
          </Button>
        )}
      </div>

      <div className="flex-1 px-4 py-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : total === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <PieChartIcon className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">No tasks yet</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5">
            <Facet title="Status">
              <div className="flex items-center gap-4">
                <div className="relative w-[120px] h-[120px] shrink-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={chart} cx="50%" cy="50%" innerRadius={38} outerRadius={54} paddingAngle={2} dataKey="value" stroke="none">
                        {chart.map((entry, idx) => <Cell key={idx} fill={entry.color} />)}
                      </Pie>
                      <Tooltip content={<DonutTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                    <span className="text-xl font-bold leading-none">{total}</span>
                  </div>
                </div>
                <div className="flex flex-col gap-1 flex-1 min-w-0">
                  {columns.filter((c) => c.count > 0).map((c) => (
                    <div key={c.status} className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: STATUS_COLORS[c.status] ?? 'hsl(var(--muted-foreground))' }} />
                      <span className="text-xs text-muted-foreground truncate flex-1">{STATUS_LABELS[c.status] ?? c.status}</span>
                      <span className="text-xs font-medium">{c.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </Facet>

            <Facet title="Priority">
              {PRIORITY_ORDER.map((p) => (
                <Bar key={p} label={p.charAt(0).toUpperCase() + p.slice(1)} count={priorities.get(p) ?? 0} max={priorityMax} color={PRIORITY_COLORS[p]} />
              ))}
            </Facet>

            <Facet title="Type of work">
              {types.length === 0 ? (
                <p className="text-xs text-muted-foreground">No work items yet</p>
              ) : (
                types.map((t) => {
                  const conf = TYPE_CONFIG[t.type] ?? TYPE_CONFIG.recipe
                  const TypeIcon = conf.icon
                  return <Bar key={t.type} label={conf.label} count={t.count} max={typeMax} color={conf.color} icon={<TypeIcon className="w-3.5 h-3.5 shrink-0" style={{ color: conf.color }} />} />
                })
              )}
            </Facet>

            <Facet title="Workload">
              {workload.length === 0 ? (
                <p className="text-xs text-muted-foreground">No workload data yet</p>
              ) : (
                workload.map((a) => (
                  <Bar
                    key={a.agent_id}
                    label={a.agent_name}
                    count={a.task_count}
                    max={workMax}
                    color="hsl(var(--agent))"
                    icon={a.agent_icon ? <PremiumIcon name={a.agent_icon} size={16} /> : <Bot className="w-3.5 h-3.5 shrink-0 text-primary" />}
                  />
                ))
              )}
            </Facet>
          </div>
        )}
      </div>
    </div>
  )
}
