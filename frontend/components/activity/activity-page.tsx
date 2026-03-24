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
  Brain,
  Target,
  BookOpen,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar } from '@/components/shared/stats-bar'
import { FilterTabs, TabsContent } from '@/components/shared/filter-tabs'
import { MissionList } from '@/components/missions/mission-list'
import { ActivityMemory } from './activity-memory'
import { ActivityBlog } from './activity-blog'
import { ActivityCalendar } from './calendar'
import { useActivityStats } from '@/hooks/use-activity-api'
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
  { value: 'memory', label: 'Memory', icon: Brain },
  { value: 'missions', label: 'Missions', icon: Target },
  { value: 'blog', label: 'Blog', icon: BookOpen },
]

export function ActivityPage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('summary')
  const [period, setPeriod] = useState('1d')

  // Deep-link: /activity?tab=board&task_id=123 or /activity?openExecution=X&recipeId=Y
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
      setActiveTab('board')
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

  return (
    <div className="space-y-4 sm:space-y-6">
      <div data-tour="activity-page-header">
        <PageHeader
          title="Command"
          titleAccent="Centre"
          subtitle="Your AI workforce at a glance"
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

      <div data-tour="activity-stats">
        <StatsBar stats={stats} className="grid gap-3 md:gap-4" />
      </div>

      <div data-tour="activity-tabs">
        <FilterTabs tabs={TAB_DEFS} value={activeTab} onValueChange={setActiveTab}>
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

          <TabsContent value="memory">
            <div data-tour="activity-memory">
              <ActivityMemory period={period} />
            </div>
          </TabsContent>

          <TabsContent value="missions">
            <div data-tour="activity-missions">
              <MissionList />
            </div>
          </TabsContent>

          <TabsContent value="blog">
            <div data-tour="activity-blog">
              <ActivityBlog period={period} />
            </div>
          </TabsContent>
        </FilterTabs>
      </div>
    </div>
  )
}
