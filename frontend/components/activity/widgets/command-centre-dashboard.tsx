'use client'

import { useState, useCallback, useEffect, useMemo, type ReactNode } from 'react'
import {
  Settings2, RotateCcw, GripVertical, Eye, EyeOff,
  Play, PieChart as PieChartIcon, Calendar, BarChart3,
  Activity, Layers, FileText, Users,
  DollarSign, TrendingUp, BookOpen, ShieldCheck,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { ActiveNowWidget } from './active-now-widget'
import { ScheduleWidget } from './schedule-widget'
import { AgentReportsWidget } from './agent-reports-widget'
import { RecentActivityWidget } from './recent-activity-widget'
import { StatusOverviewWidget } from './status-overview-widget'
import { PriorityBreakdownWidget } from './priority-breakdown-widget'
import { TypesOfWorkWidget } from './types-of-work-widget'
import { TeamWorkloadWidget } from './team-workload-widget'
import { CostTrackerWidget } from './cost-tracker-widget'
import { AgentPerformanceWidget } from './agent-performance-widget'
import { PlaybookMetricsWidget } from './playbook-metrics-widget'
import { ApprovalGatesWidget } from './approval-gates-widget'
import { cn } from '@/lib/utils'

// ── Widget Registry ─────────────────────────────────────────

type WidgetSize = 'third' | 'half' | 'two-thirds' | 'full'

const SIZE_TO_SPAN: Record<WidgetSize, string> = {
  third:      'lg:col-span-4',   // 3-up — compact KPI tiles, sparkline strips
  half:       'lg:col-span-6',   // 2-up — default for lists and small charts
  'two-thirds': 'lg:col-span-8', // 1.5-up — wide chart paired with a compact list
  full:       'lg:col-span-12',  // 1-up — hero charts, long-title lists
}

const SIZE_LABEL: Record<WidgetSize, string> = {
  third: 'Third', half: 'Half', 'two-thirds': '2 / 3', full: 'Full',
}

interface WidgetDef {
  id: string
  label: string
  icon: ReactNode
  defaultVisible: boolean
  size: WidgetSize
  height: string
}

const WIDGET_REGISTRY: WidgetDef[] = [
  { id: 'active-now',          label: 'Active Now',          icon: <Play className="w-3.5 h-3.5" />,          defaultVisible: true,  size: 'half',  height: 'min-h-[320px]' },
  { id: 'status-overview',     label: 'Status Overview',     icon: <PieChartIcon className="w-3.5 h-3.5" />,  defaultVisible: true,  size: 'half',  height: 'min-h-[320px]' },
  { id: 'schedule',            label: 'Schedule',            icon: <Calendar className="w-3.5 h-3.5" />,      defaultVisible: true,  size: 'half',  height: 'min-h-[320px]' },
  { id: 'priority-breakdown',  label: 'Priority Breakdown',  icon: <BarChart3 className="w-3.5 h-3.5" />,     defaultVisible: true,  size: 'half',  height: 'min-h-[320px]' },
  // Long task titles get truncated hard at half-width — give them the full row.
  { id: 'recent-activity',     label: 'Recent Activity',     icon: <Activity className="w-3.5 h-3.5" />,      defaultVisible: true,  size: 'full',  height: 'min-h-[280px]' },
  { id: 'types-of-work',       label: 'Types of Work',       icon: <Layers className="w-3.5 h-3.5" />,        defaultVisible: true,  size: 'half',  height: 'min-h-[280px]' },
  { id: 'agent-reports',       label: 'Agent Reports',       icon: <FileText className="w-3.5 h-3.5" />,      defaultVisible: true,  size: 'half',  height: 'min-h-[280px]' },
  { id: 'team-workload',       label: 'Team Workload',       icon: <Users className="w-3.5 h-3.5" />,         defaultVisible: true,  size: 'half',  height: 'min-h-[280px]' },
  // Cost over time benefits from chart width — full row.
  { id: 'cost-tracker',        label: 'Cost Tracker',        icon: <DollarSign className="w-3.5 h-3.5" />,    defaultVisible: true,  size: 'full',  height: 'min-h-[320px]' },
  { id: 'agent-performance',   label: 'Agent Performance',   icon: <TrendingUp className="w-3.5 h-3.5" />,    defaultVisible: true,  size: 'half',  height: 'min-h-[320px]' },
  { id: 'playbook-metrics',    label: 'Playbook Metrics',    icon: <BookOpen className="w-3.5 h-3.5" />,      defaultVisible: true,  size: 'half',  height: 'min-h-[280px]' },
  // Pending approval rows often have long mission titles — full row.
  { id: 'approval-gates',      label: 'Approval Gates',      icon: <ShieldCheck className="w-3.5 h-3.5" />,   defaultVisible: true,  size: 'full',  height: 'min-h-[280px]' },
]

const ALL_IDS = WIDGET_REGISTRY.map((w) => w.id)
const DEFAULT_ORDER = ALL_IDS
const DEFAULT_HIDDEN: string[] = []

// ── Persistence ─────────────────────────────────────────────

// v4: introduces per-widget size override. v3 entries stay readable —
// missing `sizes` field falls back to registry defaults.
const STORAGE_KEY = 'automatos:command-centre-v4'
const LEGACY_STORAGE_KEY = 'automatos:command-centre-v3'

interface DashboardState {
  order: string[]
  hidden: string[]
  sizes?: Record<string, WidgetSize>
}

function loadState(): DashboardState | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(LEGACY_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as DashboardState
    if (!Array.isArray(parsed.order) || !Array.isArray(parsed.hidden)) return null
    return parsed
  } catch {
    return null
  }
}

function saveState(state: DashboardState) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
}

// ── Dashboard Component ─────────────────────────────────────

interface CommandCentreDashboardProps {
  period: string
  onViewAllActivity?: () => void
  onViewCalendar?: () => void
}

export function CommandCentreDashboard({ period, onViewAllActivity, onViewCalendar }: CommandCentreDashboardProps) {
  const [widgetOrder, setWidgetOrder] = useState<string[]>(DEFAULT_ORDER)
  const [hiddenWidgets, setHiddenWidgets] = useState<string[]>(DEFAULT_HIDDEN)
  const [sizeOverrides, setSizeOverrides] = useState<Record<string, WidgetSize>>({})
  const [isCustomizing, setIsCustomizing] = useState(false)
  const [draggedWidget, setDraggedWidget] = useState<string | null>(null)

  // Load saved state, merging in any new widgets that didn't exist before
  useEffect(() => {
    const saved = loadState()
    if (saved) {
      // Merge: keep saved order, append any new registry widgets at the end
      const knownIds = new Set(saved.order)
      const newIds = ALL_IDS.filter((id) => !knownIds.has(id))
      const mergedOrder = [...saved.order.filter((id) => ALL_IDS.includes(id)), ...newIds]
      setWidgetOrder(mergedOrder)
      setHiddenWidgets(saved.hidden.filter((id) => ALL_IDS.includes(id)))
      if (saved.sizes) {
        // Drop any sizes for widgets no longer in the registry
        const cleaned: Record<string, WidgetSize> = {}
        for (const [id, size] of Object.entries(saved.sizes)) {
          if (ALL_IDS.includes(id) && (size === 'third' || size === 'half' || size === 'two-thirds' || size === 'full')) {
            cleaned[id] = size
          }
        }
        setSizeOverrides(cleaned)
      }
    }
  }, [])

  const persist = useCallback(
    (order: string[], hidden: string[], sizes: Record<string, WidgetSize>) => {
      saveState({ order, hidden, sizes })
    },
    [],
  )

  const handleReset = useCallback(() => {
    setWidgetOrder(DEFAULT_ORDER)
    setHiddenWidgets(DEFAULT_HIDDEN)
    setSizeOverrides({})
    persist(DEFAULT_ORDER, DEFAULT_HIDDEN, {})
  }, [persist])

  const toggleCustomize = useCallback(() => {
    setIsCustomizing((prev) => !prev)
  }, [])

  const toggleWidget = useCallback((widgetId: string) => {
    setHiddenWidgets((prev) => {
      const next = prev.includes(widgetId)
        ? prev.filter((id) => id !== widgetId)
        : [...prev, widgetId]
      setWidgetOrder((currentOrder) => {
        persist(currentOrder, next, sizeOverrides)
        return currentOrder
      })
      return next
    })
  }, [persist, sizeOverrides])

  const setWidgetSize = useCallback((widgetId: string, size: WidgetSize) => {
    setSizeOverrides((prev) => {
      const def = WIDGET_REGISTRY.find((w) => w.id === widgetId)
      const next = { ...prev }
      if (def && size === def.size) {
        // Setting to the registry default — drop the override entirely
        delete next[widgetId]
      } else {
        next[widgetId] = size
      }
      persist(widgetOrder, hiddenWidgets, next)
      return next
    })
  }, [hiddenWidgets, persist, widgetOrder])

  // Drag and drop
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
      persist(current, hiddenWidgets, sizeOverrides)
      return current
    })
  }, [hiddenWidgets, persist, sizeOverrides])

  const effectiveSize = useCallback(
    (def: WidgetDef): WidgetSize => sizeOverrides[def.id] ?? def.size,
    [sizeOverrides],
  )

  // Registry lookup
  const registryMap = useMemo(() => {
    const map = new Map<string, WidgetDef>()
    for (const w of WIDGET_REGISTRY) map.set(w.id, w)
    return map
  }, [])

  // Visible widgets in order
  const visibleWidgets = useMemo(
    () => widgetOrder.filter((id) => !hiddenWidgets.includes(id)),
    [widgetOrder, hiddenWidgets],
  )

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
      case 'cost-tracker':
        return <CostTrackerWidget period={period} />
      case 'agent-performance':
        return <AgentPerformanceWidget period={period} />
      case 'playbook-metrics':
        return <PlaybookMetricsWidget period={period} />
      case 'approval-gates':
        return <ApprovalGatesWidget period={period} />
      default:
        return null
    }
  }

  return (
    <div className="space-y-3">
      {/* Customize Controls */}
      <div className="flex items-center justify-end gap-2">
        {isCustomizing && (
          <>
            {/* Widget Picker */}
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="ghost" size="sm" className="text-xs h-7">
                  <Eye className="w-3 h-3 mr-1" />
                  Widgets ({visibleWidgets.length}/{WIDGET_REGISTRY.length})
                </Button>
              </PopoverTrigger>
              <PopoverContent align="end" className="w-64 p-2">
                <p className="text-xs font-medium text-muted-foreground px-2 py-1">
                  Show / Hide Widgets
                </p>
                <div className="space-y-0.5 mt-1 max-h-[320px] overflow-y-auto">
                  {WIDGET_REGISTRY.map((w) => {
                    const isHidden = hiddenWidgets.includes(w.id)
                    return (
                      <button
                        key={w.id}
                        type="button"
                        onClick={() => toggleWidget(w.id)}
                        className={cn(
                          'flex items-center gap-2 w-full px-2 py-1.5 rounded-md text-xs transition-colors',
                          isHidden
                            ? 'text-muted-foreground hover:bg-secondary/50'
                            : 'text-foreground hover:bg-secondary/50',
                        )}
                      >
                        <span className={cn('shrink-0', isHidden && 'opacity-40')}>
                          {w.icon}
                        </span>
                        <span className={cn('flex-1 text-left', isHidden && 'line-through opacity-50')}>
                          {w.label}
                        </span>
                        {isHidden ? (
                          <EyeOff className="w-3 h-3 text-muted-foreground" />
                        ) : (
                          <Eye className="w-3 h-3 text-success" />
                        )}
                      </button>
                    )
                  })}
                </div>
              </PopoverContent>
            </Popover>

            <Button
              variant="ghost"
              size="sm"
              onClick={handleReset}
              className="text-xs h-7"
            >
              <RotateCcw className="w-3 h-3 mr-1" />
              Reset
            </Button>
          </>
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
        {visibleWidgets.map((widgetId) => {
          const def = registryMap.get(widgetId)
          if (!def) return null

          const size = effectiveSize(def)
          return (
            <div
              key={widgetId}
              draggable={isCustomizing}
              onDragStart={() => handleDragStart(widgetId)}
              onDragOver={(e) => handleDragOver(e, widgetId)}
              onDragEnd={handleDragEnd}
              className={cn(
                'glass-card overflow-hidden rounded-xl border col-span-1 relative',
                SIZE_TO_SPAN[size],
                def.height,
                isCustomizing && 'ring-1 ring-dashed ring-primary/30',
                isCustomizing && draggedWidget === widgetId && 'opacity-50',
              )}
            >
              {isCustomizing && (
                <>
                  <div className="absolute top-2 left-2 z-10 cursor-grab active:cursor-grabbing p-1 rounded bg-background/80 backdrop-blur-sm border border-border/50">
                    <GripVertical className="w-3.5 h-3.5 text-muted-foreground" />
                  </div>
                  <div className="absolute top-2 right-2 z-10 flex items-center gap-0.5 p-0.5 rounded-md bg-background/80 backdrop-blur-sm border border-border/50 text-[10px]">
                    {(['third', 'half', 'two-thirds', 'full'] as WidgetSize[]).map((opt) => (
                      <button
                        key={opt}
                        type="button"
                        onClick={() => setWidgetSize(widgetId, opt)}
                        className={cn(
                          'px-1.5 py-0.5 rounded-sm font-medium transition-colors',
                          size === opt
                            ? 'bg-primary/20 text-primary'
                            : 'text-muted-foreground hover:bg-secondary/50',
                        )}
                        title={`${SIZE_LABEL[opt]} width`}
                      >
                        {SIZE_LABEL[opt]}
                      </button>
                    ))}
                  </div>
                </>
              )}
              {renderWidget(widgetId)}
            </div>
          )
        })}
      </div>
    </div>
  )
}
