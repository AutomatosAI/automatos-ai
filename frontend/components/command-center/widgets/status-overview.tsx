'use client'

/**
 * StatusOverview — board status breakdown panel for the Summary tab.
 * Live data via `useBoardStats('1d')`. Renders each kanban column with a
 * count + proportional bar relative to the largest count.
 */

import { useMemo } from 'react'
import { useBoardStats } from '@/hooks/use-activity-api'

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  inbox:       { label: 'Inbox',       color: 'hsl(30 14% 12%)' },
  assigned:    { label: 'Assigned',    color: 'hsl(213 51% 35%)' },
  in_progress: { label: 'In Progress', color: 'hsl(38 78% 27%)' },
  review:      { label: 'In Review',   color: 'hsl(45 80% 60%)' },
  blocked:     { label: 'Blocked',     color: 'hsl(15 76% 44%)' },
  done:        { label: 'Done',        color: 'hsl(82 30% 33%)' },
}
const STATUS_ORDER = ['inbox', 'assigned', 'in_progress', 'review', 'blocked', 'done']

export function StatusOverview() {
  const { data, isLoading } = useBoardStats('1d')

  const rows = useMemo(() => {
    if (!data?.columns) return []
    const map: Record<string, number> = {}
    data.columns.forEach((c) => {
      map[c.status] = c.count
    })
    return STATUS_ORDER.map((s) => ({
      key: s,
      label: STATUS_LABELS[s]?.label ?? s,
      color: STATUS_LABELS[s]?.color ?? 'hsl(var(--muted-foreground))',
      count: map[s] ?? 0,
    }))
  }, [data])

  if (isLoading) {
    return <div className="cc-panel-empty">Loading status…</div>
  }

  const max = Math.max(1, ...rows.map((r) => r.count))
  const total = data?.total_tasks ?? rows.reduce((s, r) => s + r.count, 0)

  if (total === 0) {
    return (
      <div className="cc-panel-empty">
        No tasks on the board today. New work will appear here as it arrives.
      </div>
    )
  }

  return (
    <div className="cc-status">
      {rows.map((r) => (
        <div key={r.key} className="cc-status-row">
          <span
            aria-hidden
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: r.color,
              display: 'inline-block',
            }}
          />
          <span className="nm">{r.label}</span>
          <span className="v">{r.count}</span>
          <div className="bar">
            <span style={{ width: `${(r.count / max) * 100}%`, background: r.color }} />
          </div>
        </div>
      ))}
    </div>
  )
}
