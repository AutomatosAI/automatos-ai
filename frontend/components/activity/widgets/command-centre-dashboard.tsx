'use client'

import { useState, useCallback, useEffect } from 'react'
import { Settings2, RotateCcw, GripVertical } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ActiveNowWidget } from './active-now-widget'
import { ScheduleWidget } from './schedule-widget'
import { AgentReportsWidget } from './agent-reports-widget'
import { RecentActivityWidget } from './recent-activity-widget'
import { StatusOverviewWidget } from './status-overview-widget'
import { PriorityBreakdownWidget } from './priority-breakdown-widget'
import { TypesOfWorkWidget } from './types-of-work-widget'
import { TeamWorkloadWidget } from './team-workload-widget'
import { cn } from '@/lib/utils'

// ── Widget Order Storage ────────────────────────────────────

const ORDER_STORAGE_KEY = 'automatos:command-centre-order-v2'
const DEFAULT_ORDER = [
  'active-now', 'status-overview',
  'schedule', 'priority-breakdown',
  'recent-activity', 'types-of-work',
  'agent-reports', 'team-workload',
]

function loadSavedOrder(): string[] | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(ORDER_STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

function saveOrder(order: string[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(ORDER_STORAGE_KEY, JSON.stringify(order))
}

// ── Dashboard Component ─────────────────────────────────────

interface CommandCentreDashboardProps {
  period: string
  onViewAllActivity?: () => void
  onViewCalendar?: () => void
}

export function CommandCentreDashboard({ period, onViewAllActivity, onViewCalendar }: CommandCentreDashboardProps) {
  const [widgetOrder, setWidgetOrder] = useState<string[]>(DEFAULT_ORDER)
  const [isCustomizing, setIsCustomizing] = useState(false)
  const [draggedWidget, setDraggedWidget] = useState<string | null>(null)

  useEffect(() => {
    const saved = loadSavedOrder()
    if (saved && saved.length === DEFAULT_ORDER.length) {
      setWidgetOrder(saved)
    }
  }, [])

  const handleReset = useCallback(() => {
    setWidgetOrder(DEFAULT_ORDER)
    saveOrder(DEFAULT_ORDER)
  }, [])

  const toggleCustomize = useCallback(() => {
    setIsCustomizing((prev) => !prev)
  }, [])

  // Simple drag-and-drop reorder
  const handleDragStart = useCallback((widgetId: string) => {
    setDraggedWidget(widgetId)
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent, targetId: string) => {
    e.preventDefault()
    if (!draggedWidget || draggedWidget === targetId) return

    setWidgetOrder((prev) => {
      const newOrder = [...prev]
      const fromIdx = newOrder.indexOf(draggedWidget)
      const toIdx = newOrder.indexOf(targetId)
      if (fromIdx === -1 || toIdx === -1) return prev
      newOrder.splice(fromIdx, 1)
      newOrder.splice(toIdx, 0, draggedWidget)
      return newOrder
    })
  }, [draggedWidget])

  const handleDragEnd = useCallback(() => {
    setDraggedWidget(null)
    setWidgetOrder((current) => {
      saveOrder(current)
      return current
    })
  }, [])

  // Render widget by ID
  const renderWidget = (widgetId: string) => {
    switch (widgetId) {
      case 'active-now':
        return <ActiveNowWidget period={period} />
      case 'status-overview':
        return <StatusOverviewWidget period={period} onViewAll={onViewAllActivity} />
      case 'schedule':
        return <ScheduleWidget onViewAll={onViewCalendar} />
      case 'priority-breakdown':
        return <PriorityBreakdownWidget period={period} />
      case 'recent-activity':
        return <RecentActivityWidget period={period} onViewAll={onViewAllActivity} />
      case 'types-of-work':
        return <TypesOfWorkWidget period={period} />
      case 'agent-reports':
        return <AgentReportsWidget />
      case 'team-workload':
        return <TeamWorkloadWidget period={period} />
      default:
        return null
    }
  }

  // Grid span config for each widget
  const gridSpan: Record<string, string> = {
    'active-now': 'lg:col-span-6',
    'status-overview': 'lg:col-span-6',
    'schedule': 'lg:col-span-6',
    'priority-breakdown': 'lg:col-span-6',
    'recent-activity': 'lg:col-span-6',
    'types-of-work': 'lg:col-span-6',
    'agent-reports': 'lg:col-span-6',
    'team-workload': 'lg:col-span-6',
  }

  // Height config
  const gridHeight: Record<string, string> = {
    'active-now': 'min-h-[320px]',
    'status-overview': 'min-h-[320px]',
    'schedule': 'min-h-[320px]',
    'priority-breakdown': 'min-h-[320px]',
    'recent-activity': 'min-h-[280px]',
    'types-of-work': 'min-h-[280px]',
    'agent-reports': 'min-h-[280px]',
    'team-workload': 'min-h-[280px]',
  }

  return (
    <div className="space-y-3">
      {/* Customize Controls */}
      <div className="flex items-center justify-end gap-2">
        {isCustomizing && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="text-xs h-7"
          >
            <RotateCcw className="w-3 h-3 mr-1" />
            Reset Layout
          </Button>
        )}
        <Button
          variant={isCustomizing ? 'secondary' : 'ghost'}
          size="sm"
          onClick={toggleCustomize}
          className="text-xs h-7"
        >
          <Settings2 className="w-3 h-3 mr-1" />
          {isCustomizing ? 'Done' : 'Customize'}
        </Button>
      </div>

      {/* Widget Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {widgetOrder.map((widgetId) => (
          <div
            key={widgetId}
            draggable={isCustomizing}
            onDragStart={() => handleDragStart(widgetId)}
            onDragOver={(e) => handleDragOver(e, widgetId)}
            onDragEnd={handleDragEnd}
            className={cn(
              'glass-card overflow-hidden rounded-xl border col-span-1 relative',
              gridSpan[widgetId],
              gridHeight[widgetId],
              isCustomizing && 'ring-1 ring-dashed ring-primary/30',
              isCustomizing && draggedWidget === widgetId && 'opacity-50',
            )}
          >
            {isCustomizing && (
              <div className="absolute top-2 left-2 z-10 cursor-grab active:cursor-grabbing p-1 rounded bg-background/80 backdrop-blur-sm border border-border/50">
                <GripVertical className="w-3.5 h-3.5 text-muted-foreground" />
              </div>
            )}
            {renderWidget(widgetId)}
          </div>
        ))}
      </div>
    </div>
  )
}
