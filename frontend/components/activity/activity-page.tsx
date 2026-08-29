'use client'

import { useState, useCallback, useEffect } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'
import {
  Activity,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Users,
  ListTodo,
  LayoutDashboard,
  Columns,
  Calendar,
  Rss,
  History,
  Zap,
  MessageSquare,
  Boxes,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { PageHeader } from '@/components/shared/page-header'
import { AutosRead } from './autos-read'
import { StatsBar } from '@/components/shared/stats-bar'
import { FilterTabs, TabsContent } from '@/components/shared/filter-tabs'
import { ActivityFeed } from './activity-feed'
import { CommandCenterHistory } from './command-center-history'
import { CalendarTab } from '@/components/command-center/calendar-tab'
import { useActivityStats } from '@/hooks/use-activity-api'
import {
  useActivationMetrics,
  useMissionSuccessRate,
  useErrorsBySubsystem,
  useWidgetEngagement,
} from '@/hooks/use-analytics-api'
import type { StatItem } from '@/components/shared/stats-bar'
import { cn } from '@/lib/utils'

// Lazy-load SSR-unfriendly components (react-grid-layout, @hello-pangea/dnd)
const CommandCentreDashboard = dynamic(
  () => import('./widgets/command-centre-dashboard').then((m) => m.CommandCentreDashboard),
  { ssr: false, loading: () => <DashboardSkeleton /> }
)

const BoardView = dynamic(
  () => import('./board').then((m) => m.BoardView),
  { ssr: false, loading: () => <BoardSkeleton /> }
)

function DashboardSkeleton() {
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

function BoardSkeleton() {
  return (
    <div className="space-y-3">
      <div className="h-8 bg-secondary/20 rounded-lg animate-pulse w-96" />
      <div className="flex gap-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="flex-1 min-w-[240px] space-y-2">
            <div className="h-6 w-24 bg-secondary/30 rounded animate-pulse" />
            <div className="h-32 bg-secondary/20 rounded-lg animate-pulse" />
            <div className="h-28 bg-secondary/20 rounded-lg animate-pulse" />
          </div>
        ))}
      </div>
    </div>
  )
}

const PERIOD_OPTIONS = [
  { value: '1d', label: '1 Day' },
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' },
]

const TAB_DEFS = [
  { value: 'summary', label: 'Summary', icon: LayoutDashboard },
  { value: 'board', label: 'Board', icon: Columns },
  { value: 'calendar', label: 'Calendar', icon: Calendar },
  { value: 'feed', label: 'Feed', icon: Rss },
  { value: 'history', label: 'History', icon: History },
]

export function ActivityPage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('summary')
  const [period, setPeriod] = useState('1d')

  // Deep-link: /activity?tab=board&task_id=123
  // (Live execution deep-link is /activity/execution?id=<execId>&recipeId=<recipeId>)
  const searchParams = useSearchParams()
  const openExecution = searchParams.get('openExecution')
  const deepLinkRecipeId = searchParams.get('recipeId')
  const tabParam = searchParams.get('tab')
  const agentIdParam = searchParams.get('agent_id')
  const taskIdParam = searchParams.get('task_id')

  useEffect(() => {
    if (tabParam && TAB_DEFS.some((t) => t.value === tabParam)) {
      setActiveTab(tabParam)
    } else if (openExecution) {
      // Legacy deep-link from old /activity?openExecution=... URLs that
      // got redirected here. Land on the feed; ActivityFeed will open the
      // slide-over for the matching item as a fallback.
      setActiveTab('feed')
    }
  }, [tabParam, openExecution])

  // Navigate to Board tab from Summary widgets
  const handleViewAllActivity = useCallback(() => {
    setActiveTab('board')
  }, [])

  // Navigate to Calendar tab from Schedule widget
  const handleViewCalendar = useCallback(() => {
    setActiveTab('calendar')
  }, [])

  const { data: liveStats } = useActivityStats(period)

  const stats: StatItem[] = [
    { label: 'Working Now', value: liveStats?.working_now ?? 0, icon: Activity, iconColor: 'text-[hsl(var(--info))]', globalIconKey: 'global_activity' },
    { label: 'Agents Active', value: liveStats?.agents_active ?? 0, icon: Users, iconColor: 'text-[hsl(var(--agent))]' },
    { label: 'Tasks in Queue', value: liveStats?.tasks_in_queue ?? 0, icon: ListTodo, iconColor: 'text-[hsl(var(--info))]' },
    { label: 'Needs Attention', value: liveStats?.needs_attention ?? 0, icon: AlertTriangle, iconColor: 'text-destructive' },
  ]

  // PRD-142 Wave 0 (US-007) — "Is it working?" platform vitals over the real
  // measurement endpoints. Per-primitive health (US-006) has no honest data
  // source yet and renders as an explicit placeholder, not a fake green.
  const { data: activation } = useActivationMetrics()
  const { data: mission } = useMissionSuccessRate()
  const { data: errors } = useErrorsBySubsystem('24h')
  const { data: widget } = useWidgetEngagement('7d')

  const totalWs = activation?.total_workspaces ?? 0
  const activationPct = totalWs > 0 ? Math.round((activation?.rate ?? 0) * 100) : 0

  const missionTotal = mission?.total_executions ?? 0
  const missionPct = missionTotal > 0 ? Math.round(mission?.value ?? 0) : 0

  const errorTotal = errors?.total ?? 0
  const worstSubsystem = errors?.by_subsystem?.length
    ? [...errors.by_subsystem].sort((a, b) => b.count - a.count)[0]
    : null

  const widgetSessions = widget?.sessions ?? 0
  const widgetEvents = widget?.by_event_type?.reduce((sum, ev) => sum + ev.count, 0) ?? 0

  const vitals: StatItem[] = [
    {
      label: 'Activation',
      value: totalWs > 0 ? `${activationPct}%` : '—',
      change: totalWs > 0 ? `${activation?.activated ?? 0}/${totalWs} workspaces` : 'No workspaces yet',
      icon: Zap,
      iconColor: 'text-primary',
    },
    {
      label: 'Mission success rate',
      value: missionTotal > 0 ? `${missionPct}%` : '—',
      change: missionTotal > 0 ? `${mission?.successful_executions ?? 0}/${missionTotal} missions` : 'No missions yet',
      icon: CheckCircle2,
      iconColor: 'text-[hsl(var(--success))]',
    },
    {
      label: 'Error rate by subsystem',
      value: errorTotal,
      change: errorTotal > 0 && worstSubsystem ? `${worstSubsystem.subsystem} (${worstSubsystem.count}) · 24h` : 'None · 24h',
      icon: AlertTriangle,
      iconColor: errorTotal > 0 ? 'text-destructive' : 'text-[hsl(var(--success))]',
    },
    {
      label: 'Widget engagement',
      value: widgetSessions,
      change: widgetSessions > 0 ? `${widgetEvents} events · 7d` : 'No widget activity',
      icon: MessageSquare,
      iconColor: 'text-[hsl(var(--info))]',
    },
    {
      label: 'Per-primitive health',
      value: '—',
      change: 'Not yet measured',
      icon: Boxes,
      iconColor: 'text-muted-foreground',
    },
  ]

  return (
    <div className="space-y-4 sm:space-y-6">
      <div>
        <PageHeader
          title="Command"
          titleAccent="Centre"
          eyebrow="Operations · daily glance"
          lede="What your workforce is running right now, what cleared, and what needs your eyes. Everything streams live; replay any window to the second."
          actions={
            <>
              <Select value={period} onValueChange={setPeriod}>
                <SelectTrigger className="w-28 bg-secondary/50 min-h-[44px] sm:min-h-0">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PERIOD_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Button variant="outline" size="sm" className="min-h-[44px] sm:min-h-0">
                <RefreshCw className="w-4 h-4 mr-2" />
                <span className="hidden sm:inline">Refresh</span>
              </Button>
            </>
          }
        />
      </div>

      <div>
        <StatsBar stats={stats} className="grid gap-3 md:gap-4" />
      </div>

      <div>
        <StatsBar stats={vitals} glow={false} className="grid gap-3 md:gap-4 lg:grid-cols-5" />
      </div>

      <div>
        <FilterTabs tabs={TAB_DEFS} value={activeTab} onValueChange={setActiveTab}>
          <TabsContent value="summary">
            <div className="space-y-4">
              <AutosRead period={period} />
              <CommandCentreDashboard
                period={period}
                onViewAllActivity={handleViewAllActivity}
                onViewCalendar={handleViewCalendar}
              />
            </div>
          </TabsContent>

          <TabsContent value="board">
            <div>
              <BoardView period={period} />
            </div>
          </TabsContent>

          <TabsContent value="calendar">
            <div>
              <CalendarTab />
            </div>
          </TabsContent>

          <TabsContent value="feed">
            <div>
              <ActivityFeed
                period={period}
                openExecution={openExecution}
                deepLinkRecipeId={deepLinkRecipeId}
              />
            </div>
          </TabsContent>

          <TabsContent value="history">
            <div>
              <CommandCenterHistory period={period} />
            </div>
          </TabsContent>

        </FilterTabs>
      </div>
    </div>
  )
}
