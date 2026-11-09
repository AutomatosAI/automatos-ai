'use client'

import { useState } from 'react'
import { Cpu } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  CommunicationTab,
  LearningTab,
  MemoryTab,
  MetricsOverviewCard,
  OverviewTab
} from './enhanced-orchestrator-sections'
import { useEnhancedOrchestratorData } from './use-enhanced-orchestrator-data'

// The view's data types and steps live in enhanced-orchestrator-data.ts; they stay
// importable from here, where the sections and tests have always found them.
export { memoryActivityType } from './enhanced-orchestrator-data'
export type {
  CommunicationEvent,
  ExecutionOption,
  LearningInsight,
  MemoryActivity,
  OrchestrationMetrics
} from './enhanced-orchestrator-data'

interface EnhancedOrchestratorViewProps {
  workflowId: number
  executionId?: string
  isExecuting: boolean
}

export function EnhancedOrchestratorView({ 
  workflowId, 
  executionId,
  isExecuting 
}: EnhancedOrchestratorViewProps) {
  const data = useEnhancedOrchestratorData(workflowId, executionId, isExecuting)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['memory', 'communication']))
  const [selectedTab, setSelectedTab] = useState('overview')

  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(section)) {
        next.delete(section)
      } else {
        next.add(section)
      }
      return next
    })
  }

  if (data.isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Cpu className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading orchestration data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <MetricsOverviewCard
        metrics={data.metrics}
        isExecuting={isExecuting}
        historicalData={data.historicalData}
        executionOptions={data.executionOptions}
        selectedExecutionId={data.selectedExecutionId}
        selectedExecution={data.selectedExecution}
        onSelectExecution={data.chooseExecution}
      />

      {/* Tabbed Content */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-4">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="memory">Memory</TabsTrigger>
          <TabsTrigger value="communication">Communication</TabsTrigger>
          <TabsTrigger value="learning">Learning</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <OverviewTab
            memoryActivities={data.memoryActivities}
            communications={data.communications}
            expandedSections={expandedSections}
            onToggleSection={toggleSection}
          />
        </TabsContent>

        <TabsContent value="memory">
          <MemoryTab memoryActivities={data.memoryActivities} />
        </TabsContent>

        <TabsContent value="communication">
          <CommunicationTab communications={data.communications} />
        </TabsContent>

        <TabsContent value="learning">
          <LearningTab learningInsights={data.learningInsights} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

