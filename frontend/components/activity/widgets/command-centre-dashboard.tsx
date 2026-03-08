'use client'

import { useState, useCallback, useMemo, useEffect } from 'react'
import { Settings2, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ActiveNowWidget } from './active-now-widget'
import { ScheduleWidget } from './schedule-widget'
import { AgentReportsWidget } from './agent-reports-widget'
import { RecentActivityWidget } from './recent-activity-widget'
import { cn } from '@/lib/utils'

// react-grid-layout uses CommonJS — require for SSR-safe dynamic import
// eslint-disable-next-line @typescript-eslint/no-require-imports
const RGL = require('react-grid-layout')
const ResponsiveGridLayout = RGL.WidthProvider(RGL.Responsive)

type Layout = { i: string; x: number; y: number; w: number; h: number; minW?: number; minH?: number }
type Layouts = Record<string, Layout[]>

// ── Layout Storage ──────────────────────────────────────────

const LAYOUT_STORAGE_KEY = 'automatos:command-centre-layout'

const DEFAULT_LAYOUTS: Layouts = {
  lg: [
    { i: 'active-now', x: 0, y: 0, w: 5, h: 5, minW: 3, minH: 3 },
    { i: 'schedule', x: 5, y: 0, w: 7, h: 5, minW: 4, minH: 3 },
    { i: 'agent-reports', x: 0, y: 5, w: 12, h: 5, minW: 6, minH: 3 },
    { i: 'recent-activity', x: 0, y: 10, w: 12, h: 4, minW: 6, minH: 3 },
  ],
  md: [
    { i: 'active-now', x: 0, y: 0, w: 5, h: 5, minW: 3, minH: 3 },
    { i: 'schedule', x: 5, y: 0, w: 5, h: 5, minW: 3, minH: 3 },
    { i: 'agent-reports', x: 0, y: 5, w: 10, h: 5, minW: 5, minH: 3 },
    { i: 'recent-activity', x: 0, y: 10, w: 10, h: 4, minW: 5, minH: 3 },
  ],
  sm: [
    { i: 'active-now', x: 0, y: 0, w: 6, h: 5, minW: 3, minH: 3 },
    { i: 'schedule', x: 0, y: 5, w: 6, h: 5, minW: 3, minH: 3 },
    { i: 'agent-reports', x: 0, y: 10, w: 6, h: 6, minW: 3, minH: 3 },
    { i: 'recent-activity', x: 0, y: 16, w: 6, h: 4, minW: 3, minH: 3 },
  ],
}

function loadSavedLayouts(): Layouts | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(LAYOUT_STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

function saveLayouts(layouts: Layouts) {
  if (typeof window === 'undefined') return
  localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(layouts))
}

// ── Dashboard Component ─────────────────────────────────────

interface CommandCentreDashboardProps {
  period: string
  onViewAllActivity?: () => void
}

export function CommandCentreDashboard({ period, onViewAllActivity }: CommandCentreDashboardProps) {
  const [layouts, setLayouts] = useState<Layouts>(DEFAULT_LAYOUTS)
  const [isCustomizing, setIsCustomizing] = useState(false)
  const [mounted, setMounted] = useState(false)

  // Load saved layouts on mount
  useEffect(() => {
    const saved = loadSavedLayouts()
    if (saved) {
      setLayouts(saved)
    }
    setMounted(true)
  }, [])

  const handleLayoutChange = useCallback((_layout: Layout[], allLayouts: Layouts) => {
    setLayouts(allLayouts)
    saveLayouts(allLayouts)
  }, [])

  const handleReset = useCallback(() => {
    setLayouts(DEFAULT_LAYOUTS)
    saveLayouts(DEFAULT_LAYOUTS)
  }, [])

  const toggleCustomize = useCallback(() => {
    setIsCustomizing((prev) => !prev)
  }, [])

  // Widget wrapper with glass-card styling
  const widgetClass = useMemo(() => cn(
    'glass-card overflow-hidden rounded-xl border',
    isCustomizing && 'ring-1 ring-dashed ring-primary/30'
  ), [isCustomizing])

  if (!mounted) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="h-[280px] bg-secondary/20 rounded-xl animate-pulse" />
          <div className="h-[280px] bg-secondary/20 rounded-xl animate-pulse" />
        </div>
        <div className="h-[280px] bg-secondary/20 rounded-xl animate-pulse" />
        <div className="h-[220px] bg-secondary/20 rounded-xl animate-pulse" />
      </div>
    )
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

      {/* Grid Layout */}
      <ResponsiveGridLayout
        className="layout"
        layouts={layouts}
        breakpoints={{ lg: 1200, md: 996, sm: 0 }}
        cols={{ lg: 12, md: 10, sm: 6 }}
        rowHeight={56}
        margin={[16, 16]}
        containerPadding={[0, 0]}
        onLayoutChange={handleLayoutChange}
        isDraggable={isCustomizing}
        isResizable={isCustomizing}
        draggableHandle=".widget-drag-handle"
        useCSSTransforms={true}
        compactType="vertical"
      >
        <div key="active-now" className={widgetClass}>
          {isCustomizing && <DragHandle />}
          <ActiveNowWidget period={period} />
        </div>

        <div key="schedule" className={widgetClass}>
          {isCustomizing && <DragHandle />}
          <ScheduleWidget />
        </div>

        <div key="agent-reports" className={widgetClass}>
          {isCustomizing && <DragHandle />}
          <AgentReportsWidget />
        </div>

        <div key="recent-activity" className={widgetClass}>
          {isCustomizing && <DragHandle />}
          <RecentActivityWidget period={period} onViewAll={onViewAllActivity} />
        </div>
      </ResponsiveGridLayout>
    </div>
  )
}

// ── Drag Handle ─────────────────────────────────────────────

function DragHandle() {
  return (
    <div className="widget-drag-handle absolute top-0 left-0 right-0 h-8 flex items-center justify-center cursor-grab active:cursor-grabbing z-10 bg-gradient-to-b from-background/60 to-transparent">
      <div className="flex gap-0.5">
        <div className="w-1 h-1 rounded-full bg-muted-foreground/40" />
        <div className="w-1 h-1 rounded-full bg-muted-foreground/40" />
        <div className="w-1 h-1 rounded-full bg-muted-foreground/40" />
      </div>
    </div>
  )
}
