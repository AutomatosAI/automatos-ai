'use client'

import { useState, useCallback } from 'react'
import dynamic from 'next/dynamic'
import {
  Activity,
  RefreshCw,
  ChefHat,
  Rocket,
  CheckCircle2,
  AlertTriangle,
  Radio,
  LayoutDashboard,
  List,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar } from '@/components/shared/stats-bar'
import { FilterTabs, TabsContent } from '@/components/shared/filter-tabs'
import { RecipesTab } from '@/components/workflows/recipes-tab'
import { ActivityMissions } from './activity-missions'
import { ActivityFeed } from './activity-feed'
import { useActivityStats } from '@/hooks/use-activity-api'
import type { StatItem } from '@/components/shared/stats-bar'
import { cn } from '@/lib/utils'

// Lazy-load the dashboard grid (SSR-unfriendly due to react-grid-layout)
const CommandCentreDashboard = dynamic(
  () => import('./widgets/command-centre-dashboard').then((m) => m.CommandCentreDashboard),
  { ssr: false, loading: () => <DashboardSkeleton /> }
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

const PERIOD_OPTIONS = [
  { value: '1d', label: '1 Day' },
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' },
]

const TAB_DEFS = [
  { value: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { value: 'feed', label: 'Feed', icon: List },
  { value: 'recipes', label: 'Recipes', icon: ChefHat },
  { value: 'missions', label: 'Missions', icon: Rocket },
]

export function ActivityPage() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [period, setPeriod] = useState('1d')

  // RecipesTab requires onUseRecipe — no-op in Activity context (full edit is in /workflows)
  const handleUseRecipe = useCallback(() => {}, [])

  // Switch to feed tab when "View All" is clicked in recent activity widget
  const handleViewAllActivity = useCallback(() => {
    setActiveTab('feed')
  }, [])

  const { data: liveStats } = useActivityStats(period)

  const stats: StatItem[] = [
    { label: 'Working Now', value: liveStats?.working_now ?? 0, icon: Activity, iconColor: 'text-[hsl(var(--info))]' },
    { label: 'Channels Live', value: liveStats?.channels_live ?? 0, icon: Radio, iconColor: 'text-[hsl(var(--info))]' },
    { label: 'Completed Today', value: liveStats?.completed_today ?? 0, icon: CheckCircle2, iconColor: 'text-[hsl(var(--success))]' },
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
          <TabsContent value="dashboard">
            <div data-tour="activity-dashboard">
              <CommandCentreDashboard
                period={period}
                onViewAllActivity={handleViewAllActivity}
              />
            </div>
          </TabsContent>

          <TabsContent value="feed">
            <div data-tour="activity-content">
              <ActivityFeed period={period} />
            </div>
          </TabsContent>

          <TabsContent value="recipes">
            <RecipesTab onUseRecipe={handleUseRecipe} />
          </TabsContent>

          <TabsContent value="missions">
            <ActivityMissions />
          </TabsContent>
        </FilterTabs>
      </div>
    </div>
  )
}
