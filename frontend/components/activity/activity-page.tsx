'use client'

import { useState } from 'react'
import {
  Activity,
  RefreshCw,
  ChefHat,
  Rocket,
  CheckCircle2,
  AlertTriangle,
  Radio,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { PageHeader } from '@/components/shared/page-header'
import { StatsBar } from '@/components/shared/stats-bar'
import { FilterTabs, TabsContent } from '@/components/shared/filter-tabs'
import type { StatItem } from '@/components/shared/stats-bar'

const PERIOD_OPTIONS = [
  { value: '1d', label: '1 Day' },
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' },
]

const TAB_DEFS = [
  { value: 'feed', label: 'Feed', icon: Activity },
  { value: 'routines', label: 'Routines', icon: RefreshCw },
  { value: 'recipes', label: 'Recipes', icon: ChefHat },
  { value: 'missions', label: 'Missions', icon: Rocket },
]

export function ActivityPage() {
  const [activeTab, setActiveTab] = useState('feed')
  const [period, setPeriod] = useState('1d')

  const stats: StatItem[] = [
    { label: 'Working Now', value: 0, icon: Activity, iconColor: 'text-[hsl(var(--info))]' },
    { label: 'Channels Live', value: 0, icon: Radio, iconColor: 'text-[hsl(var(--info))]' },
    { label: 'Completed Today', value: 0, icon: CheckCircle2, iconColor: 'text-[hsl(var(--success))]' },
    { label: 'Needs Attention', value: 0, icon: AlertTriangle, iconColor: 'text-destructive' },
  ]

  return (
    <div className="space-y-6">
      <div data-tour="activity-page-header">
        <PageHeader
          title="Command"
          titleAccent="Centre"
          subtitle="Your AI workforce at a glance"
          actions={
            <>
              <Select value={period} onValueChange={setPeriod}>
                <SelectTrigger className="w-28 bg-secondary/50">
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

              <Button variant="outline" size="sm">
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </>
          }
        />
      </div>

      <div data-tour="activity-stats">
        <StatsBar stats={stats} />
      </div>

      <div data-tour="activity-tabs">
        <FilterTabs tabs={TAB_DEFS} value={activeTab} onValueChange={setActiveTab}>
          <TabsContent value="feed">
            <div data-tour="activity-content" className="glass-card p-8 text-center text-muted-foreground">
              <Activity className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p className="font-medium">Feed coming soon</p>
              <p className="text-sm mt-1">All your AI workforce activity in one timeline</p>
            </div>
          </TabsContent>

          <TabsContent value="routines">
            <div className="glass-card p-8 text-center text-muted-foreground">
              <RefreshCw className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p className="font-medium">Routines coming soon</p>
              <p className="text-sm mt-1">Your agent heartbeat routines will appear here</p>
            </div>
          </TabsContent>

          <TabsContent value="recipes">
            <div className="glass-card p-8 text-center text-muted-foreground">
              <ChefHat className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p className="font-medium">Recipes coming soon</p>
              <p className="text-sm mt-1">Your workflow recipes will be wired in next</p>
            </div>
          </TabsContent>

          <TabsContent value="missions">
            <div className="glass-card p-8 text-center text-muted-foreground">
              <Rocket className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p className="font-medium">Missions coming soon</p>
              <p className="text-sm mt-1">Multi-agent projects — coming in a future release</p>
            </div>
          </TabsContent>
        </FilterTabs>
      </div>
    </div>
  )
}
