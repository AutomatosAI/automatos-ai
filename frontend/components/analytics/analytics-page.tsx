'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  BarChart3,
  Bot,
  GitBranch,
  FileText,
  DollarSign,
  Shield,
  RefreshCw,
  Download,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useSystemRole } from '@/contexts/role-context'
import { useQueryClient } from '@tanstack/react-query'
import { AnalyticsOverview } from './analytics-overview'
import { AnalyticsAgents } from './analytics-agents'
import { AnalyticsWorkflows } from './analytics-workflows'
import { AnalyticsDocuments } from './analytics-documents'
import { AnalyticsCosts } from './analytics-costs'
import { AnalyticsAdmin } from './analytics-admin'

const TIME_RANGES = [
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
  { value: '90', label: '90 days' },
]

export function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('30')
  const [activeTab, setActiveTab] = useState('overview')
  const { isAdmin } = useSystemRole()
  const queryClient = useQueryClient()
  const days = parseInt(timeRange)

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['unified-analytics'] })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold mb-1">
            <span className="gradient-text">Analytics</span>
          </h1>
          <p className="text-muted-foreground">
            Your workspace performance, costs, and insights at a glance
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-28 bg-secondary/50">
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
        </div>
      </motion.div>

      {/* Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className={`grid w-full ${isAdmin ? 'grid-cols-6' : 'grid-cols-5'} lg:w-auto lg:inline-grid bg-secondary/50`}>
            <TabsTrigger value="overview" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              <span className="hidden sm:inline">Overview</span>
            </TabsTrigger>
            <TabsTrigger value="agents" className="flex items-center gap-2">
              <Bot className="w-4 h-4" />
              <span className="hidden sm:inline">Agents</span>
            </TabsTrigger>
            <TabsTrigger value="workflows" className="flex items-center gap-2">
              <GitBranch className="w-4 h-4" />
              <span className="hidden sm:inline">Workflows</span>
            </TabsTrigger>
            <TabsTrigger value="documents" className="flex items-center gap-2">
              <FileText className="w-4 h-4" />
              <span className="hidden sm:inline">Documents</span>
            </TabsTrigger>
            <TabsTrigger value="costs" className="flex items-center gap-2">
              <DollarSign className="w-4 h-4" />
              <span className="hidden sm:inline">LLM & Costs</span>
            </TabsTrigger>
            {isAdmin && (
              <TabsTrigger value="admin" className="flex items-center gap-2">
                <Shield className="w-4 h-4" />
                <span className="hidden sm:inline">Admin</span>
              </TabsTrigger>
            )}
          </TabsList>

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

          {isAdmin && (
            <TabsContent value="admin">
              <AnalyticsAdmin days={days} />
            </TabsContent>
          )}
        </Tabs>
      </motion.div>
    </div>
  )
}
