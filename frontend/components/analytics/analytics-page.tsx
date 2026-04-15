'use client'

import { useState, useCallback } from 'react'
import {
  BarChart3,
  Bot,
  GitBranch,
  FileText,
  DollarSign,
  Wrench,
  Shield,
  RefreshCw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useSystemRole } from '@/contexts/role-context'
import { useQueryClient } from '@tanstack/react-query'
import { PageHeader } from '@/components/shared/page-header'
import { FilterTabs, TabsContent } from '@/components/shared/filter-tabs'
import { AnalyticsOverview } from './analytics-overview'
import { AnalyticsAgents } from './analytics-agents'
import { AnalyticsWorkflows } from './analytics-workflows'
import { AnalyticsDocuments } from './analytics-documents'
import { AnalyticsCosts } from './analytics-costs'
import { AnalyticsAdmin } from './analytics-admin'
import { AnalyticsComposio } from './analytics-composio'
import { AdminWorkspaceSwitcher } from './admin-workspace-switcher'

const TIME_RANGES = [
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
  { value: '90', label: '90 days' },
]

export function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('30')
  const [activeTab, setActiveTab] = useState('overview')
  const [wsLabel, setWsLabel] = useState<string | null>(null)
  const { isAdmin } = useSystemRole()
  const queryClient = useQueryClient()
  const days = parseInt(timeRange)

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['unified-analytics'] })
    queryClient.refetchQueries({ queryKey: ['unified-analytics'] })
  }

  const handleWorkspaceChange = useCallback((label: string) => {
    setWsLabel(label === 'My Workspace' ? null : label)
  }, [])

  const tabDefs = [
    { value: 'overview', label: 'Overview', icon: BarChart3 },
    { value: 'agents', label: 'Agents', icon: Bot },
    { value: 'workflows', label: 'Missions', icon: GitBranch },
    { value: 'documents', label: 'Documents', icon: FileText },
    { value: 'costs', label: 'LLM & Costs', icon: DollarSign },
    { value: 'tools', label: 'Tools & Integrations', icon: Wrench },
    ...(isAdmin ? [{ value: 'admin', label: 'Admin', icon: Shield }] : []),
  ]

  return (
    <div className="space-y-6">
      <div data-tour="analytics-page-header">
      <PageHeader
        title=""
        titleAccent="Analytics"
        subtitle={wsLabel ? `Viewing: ${wsLabel}` : 'Your workspace performance, costs, and insights at a glance'}
        actions={
          <>
            {isAdmin && <AdminWorkspaceSwitcher onWorkspaceChange={handleWorkspaceChange} />}

            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger data-tour="analytics-time-range" className="w-28 bg-secondary/50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TIME_RANGES.map(tr => (
                  <SelectItem key={tr.value} value={tr.value}>{tr.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Button variant="outline" size="sm" onClick={handleRefresh}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </>
        }
      />
      </div>

      <div>
      <FilterTabs tabs={tabDefs} value={activeTab} onValueChange={setActiveTab} dataTour="analytics-tabs">
        <TabsContent value="overview">
          <AnalyticsOverview days={days} />
        </TabsContent>

        <TabsContent value="agents">
          <AnalyticsAgents days={days} />
        </TabsContent>

        <TabsContent value="workflows">
          <AnalyticsWorkflows days={days} />
        </TabsContent>

        <TabsContent value="documents">
          <AnalyticsDocuments days={days} />
        </TabsContent>

        <TabsContent value="costs">
          <AnalyticsCosts days={days} />
        </TabsContent>

        <TabsContent value="tools">
          <AnalyticsComposio days={days} />
        </TabsContent>

        {isAdmin && (
          <TabsContent value="admin">
            <AnalyticsAdmin days={days} />
          </TabsContent>
        )}
      </FilterTabs>
      </div>
    </div>
  )
}
