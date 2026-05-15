'use client'

/**
 * CommandCenterShell — Studio wrapper for the existing Command Centre.
 *
 * Reuses every working piece: PageHeader, StatsBar, FilterTabs, the live
 * `CommandCentreDashboard` (Summary widgets), `BoardView` (with real DnD),
 * `ActivityCalendar` (with working time grid + clickable events), and
 * `ActivityFeed` (which merges feed + history per CD's round-4 spec).
 *
 * The Studio rebrand is a re-shelve, not a rebuild — every existing widget
 * and interaction is preserved. The only structural change is dropping the
 * legacy History tab; Activity now hosts the merged stream.
 *
 * Toolbar matches the classic ActivityPage exactly: period selector +
 * Refresh. No speculative CTAs (mission creation lives on /assignments,
 * task scheduling on /chat?mode=plan, books on /analytics).
 */

import { useCallback, useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import dynamic from 'next/dynamic'
import {
  Activity,
  AlertTriangle,
  Calendar,
  Columns,
  LayoutDashboard,
  ListTodo,
  RefreshCw,
  Rss,
  Users,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar } from '@/components/shared/stats-bar'
import { FilterTabs, TabsContent } from '@/components/shared/filter-tabs'
import { ActivityFeed } from '@/components/activity/activity-feed'
import { ActivityCalendar } from '@/components/activity/calendar'
import { useActivityStats } from '@/hooks/use-activity-api'
import type { StatItem } from '@/components/shared/stats-bar'

const CommandCentreDashboard = dynamic(
  () =>
    import('@/components/activity/widgets/command-centre-dashboard').then(
      (m) => m.CommandCentreDashboard,
    ),
  { ssr: false, loading: () => <DashboardSkeleton /> },
)

const BoardView = dynamic(
  () => import('@/components/activity/board').then((m) => m.BoardView),
  { ssr: false, loading: () => <BoardSkeleton /> },
)

function DashboardSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="h-[280px] bg-secondary/30 rounded-xl animate-pulse" />
        <div className="h-[280px] bg-secondary/30 rounded-xl animate-pulse" />
      </div>
      <div className="h-[280px] bg-secondary/30 rounded-xl animate-pulse" />
      <div className="h-[220px] bg-secondary/30 rounded-xl animate-pulse" />
    </div>
  )
}

function BoardSkeleton() {
  return (
    <div className="space-y-3">
      <div className="h-8 bg-secondary/30 rounded-lg animate-pulse w-96" />
      <div className="flex gap-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="flex-1 min-w-[240px] space-y-2">
            <div className="h-6 w-24 bg-secondary/40 rounded animate-pulse" />
            <div className="h-32 bg-secondary/30 rounded-lg animate-pulse" />
            <div className="h-28 bg-secondary/30 rounded-lg animate-pulse" />
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

// CD round 4: five tabs become four — Feed and History merge into Activity.
const TAB_DEFS = [
  { value: 'summary', label: 'Summary', icon: LayoutDashboard },
  { value: 'board', label: 'Board', icon: Columns },
  { value: 'calendar', label: 'Calendar', icon: Calendar },
  { value: 'activity', label: 'Activity', icon: Rss },
]
const VALID_TABS = new Set(TAB_DEFS.map((t) => t.value))

export function CommandCenterShell() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const initialTab = (() => {
    const raw = searchParams?.get('tab') ?? 'summary'
    // Legacy feed/history URLs → activity tab
    if (raw === 'feed' || raw === 'history') return 'activity'
    return VALID_TABS.has(raw) ? raw : 'summary'
  })()

  const [activeTab, setActiveTab] = useState(initialTab)
  const [period, setPeriod] = useState('1d')

  const openExecution = searchParams?.get('openExecution') ?? null
  const deepLinkRecipeId = searchParams?.get('recipeId') ?? null
  const tabParam = searchParams?.get('tab') ?? null

  useEffect(() => {
    if (tabParam === 'feed' || tabParam === 'history') {
      setActiveTab('activity')
    } else if (tabParam && VALID_TABS.has(tabParam)) {
      setActiveTab(tabParam)
    } else if (openExecution) {
      setActiveTab('activity')
    }
  }, [tabParam, openExecution])

  const handleTabChange = (next: string) => {
    setActiveTab(next)
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.set('tab', next)
    router.replace(`/command-center?${params.toString()}` as any, { scroll: false })
  }

  const handleViewAllActivity = useCallback(() => handleTabChange('board'), [])
  const handleViewCalendar = useCallback(() => handleTabChange('calendar'), [])

  const { data: liveStats, refetch } = useActivityStats(period)

  const stats: StatItem[] = [
    {
      label: 'Working Now',
      value: liveStats?.working_now ?? 0,
      icon: Activity,
      iconColor: 'text-[hsl(var(--info))]',
      globalIconKey: 'global_activity',
    },
    {
      label: 'Agents Active',
      value: liveStats?.agents_active ?? 0,
      icon: Users,
      iconColor: 'text-[hsl(var(--agent))]',
    },
    {
      label: 'Tasks in Queue',
      value: liveStats?.tasks_in_queue ?? 0,
      icon: ListTodo,
      iconColor: 'text-[hsl(var(--info))]',
    },
    {
      label: 'Needs Attention',
      value: liveStats?.needs_attention ?? 0,
      icon: AlertTriangle,
      iconColor: 'text-destructive',
    },
  ]

  return (
    <div className="space-y-4 sm:space-y-6 px-4 py-4 md:px-8 md:py-6 lg:px-12 lg:py-8 max-w-[1720px] mx-auto w-full">
      <div data-tour="activity-page-header">
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

              <Button
                variant="outline"
                size="sm"
                className="min-h-[44px] sm:min-h-0"
                onClick={() => refetch()}
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                <span className="hidden sm:inline">Refresh</span>
              </Button>
            </>
          }
        />
      </div>

      <div data-tour="activity-stats">
        <StatsBar stats={stats} className="grid gap-3 md:gap-4" />
      </div>

      <div>
        <FilterTabs
          tabs={TAB_DEFS}
          value={activeTab}
          onValueChange={handleTabChange}
          dataTour="activity-tabs"
        >
          <TabsContent value="summary">
            <div data-tour="activity-summary">
              <CommandCentreDashboard
                period={period}
                onViewAllActivity={handleViewAllActivity}
                onViewCalendar={handleViewCalendar}
              />
            </div>
          </TabsContent>

          <TabsContent value="board">
            <div data-tour="activity-board">
              <BoardView period={period} />
            </div>
          </TabsContent>

          <TabsContent value="calendar">
            <div data-tour="activity-calendar">
              <ActivityCalendar />
            </div>
          </TabsContent>

          <TabsContent value="activity">
            <div data-tour="activity-feed">
              <ActivityFeed
                period={period}
                openExecution={openExecution}
                deepLinkRecipeId={deepLinkRecipeId}
              />
            </div>
          </TabsContent>
        </FilterTabs>
      </div>
    </div>
  )
}
