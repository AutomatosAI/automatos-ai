'use client'

// Presentational sections extracted from EnhancedOrchestratorView so that
// component itself stays under the code-shape ratchet's line limit. Pure
// extraction: no behavior change from the JSX this replaces.

import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain,
  MessageSquare,
  Clock,
  ChevronRight,
  ChevronDown,
  Sparkles,
  Network,
  Users,
  Cpu,
  Database,
  BarChart3
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type {
  CommunicationEvent,
  ExecutionOption,
  LearningInsight,
  MemoryActivity,
  OrchestrationMetrics
} from './enhanced-orchestrator-view'

function getMemoryIcon(type: MemoryActivity['type']) {
  switch (type) {
    case 'working': return <Cpu className="w-4 h-4" />
    case 'short_term': return <Clock className="w-4 h-4" />
    case 'long_term': return <Database className="w-4 h-4" />
    case 'collective': return <Users className="w-4 h-4" />
  }
}

function getMemoryColor(type: MemoryActivity['type']) {
  switch (type) {
    case 'working': return 'text-info bg-info/10'
    case 'short_term': return 'text-warning bg-warning/10'
    case 'long_term': return 'text-success bg-success/10'
    case 'collective': return 'text-agent bg-agent/10'
  }
}

function getPriorityColor(priority: CommunicationEvent['priority']) {
  switch (priority) {
    case 'low': return 'text-muted-foreground'
    case 'normal': return 'text-info'
    case 'high': return 'text-warning'
    case 'critical': return 'text-destructive'
  }
}

function getImpactColor(impact: LearningInsight['impact']) {
  switch (impact) {
    case 'low': return 'bg-secondary/50 text-muted-foreground'
    case 'medium': return 'bg-warning/10 text-warning'
    case 'high': return 'bg-success/10 text-success'
  }
}

interface MetricsOverviewCardProps {
  metrics: OrchestrationMetrics
  isExecuting: boolean
  historicalData: unknown
  executionOptions: ExecutionOption[]
  selectedExecutionId: string | null
  selectedExecution: ExecutionOption | null | undefined
  onSelectExecution: (id: string) => void
}

export function MetricsOverviewCard({
  metrics,
  isExecuting,
  historicalData,
  executionOptions,
  selectedExecutionId,
  selectedExecution,
  onSelectExecution
}: MetricsOverviewCardProps) {
  return (
    <Card className="glass-card">
      <CardHeader>
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <CardTitle className="flex items-center gap-2">
            <span className="flex items-center">
              <BarChart3 className="w-5 h-5 mr-2" />
              Orchestration Metrics
            </span>
            {!isExecuting && historicalData && (
              <Badge variant="secondary" className="text-xs">
                Historical Data
              </Badge>
            )}
          </CardTitle>
          {executionOptions.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Execution
              </span>
              <Select
                value={selectedExecutionId ?? undefined}
                onValueChange={onSelectExecution}
              >
                <SelectTrigger className="w-[220px]">
                  <SelectValue placeholder="Select execution" />
                </SelectTrigger>
                <SelectContent>
                  {executionOptions.map((option) => (
                    <SelectItem key={option.id} value={option.id}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedExecution?.status && (
                <Badge variant="outline" className="text-xs capitalize">
                  {selectedExecution.status}
                </Badge>
              )}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div>
            <p className="text-sm text-muted-foreground">Tasks</p>
            <p className="text-2xl font-bold">{metrics.completedTasks}/{metrics.totalTasks}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Active Agents</p>
            <p className="text-2xl font-bold">{metrics.activeAgents}</p>
            {metrics.activeAgents > 1 && (
              <Badge variant="secondary" className="mt-1 text-xs">
                Multi-Agent
              </Badge>
            )}
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Memory Ops</p>
            <p className="text-2xl font-bold">{metrics.memoryOperations}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Messages</p>
            <p className="text-2xl font-bold">{metrics.messagesExchanged}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Patterns</p>
            <p className="text-2xl font-bold">{metrics.patternsLearned}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-border/30">
          <div>
            <p className="text-sm text-muted-foreground">Avg Response</p>
            <p className="text-lg font-semibold">{metrics.avgResponseTime.toFixed(0)}ms</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Success Rate</p>
            <p className="text-lg font-semibold">{(metrics.successRate * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Tokens Used</p>
            <p className="text-lg font-semibold">{metrics.tokenUsage.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Est. Cost</p>
            <p className="text-lg font-semibold">${metrics.estimatedCost.toFixed(2)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

interface OverviewTabProps {
  memoryActivities: MemoryActivity[]
  communications: CommunicationEvent[]
  expandedSections: Set<string>
  onToggleSection: (section: string) => void
}

export function OverviewTab({
  memoryActivities,
  communications,
  expandedSections,
  onToggleSection
}: OverviewTabProps) {
  return (
    <>
      {/* Memory Activity Summary */}
      <Card className="glass-card">
        <CardHeader>
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => onToggleSection('memory')}
          >
            <CardTitle className="flex items-center">
              <Brain className="w-5 h-5 mr-2" />
              Memory System Activity
            </CardTitle>
            {expandedSections.has('memory') ?
              <ChevronDown className="w-5 h-5" /> :
              <ChevronRight className="w-5 h-5" />
            }
          </div>
        </CardHeader>
        <AnimatePresence>
          {expandedSections.has('memory') && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
            >
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-2">
                    {memoryActivities.slice(0, 10).map((activity) => (
                      <div
                        key={activity.id}
                        className="flex items-start space-x-3 p-2 rounded-lg hover:bg-accent/5 transition-colors"
                      >
                        <div className={`p-1 rounded ${getMemoryColor(activity.type)}`}>
                          {getMemoryIcon(activity.type)}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-medium">
                              {activity.agentName || 'System'} • {activity.action}
                            </p>
                            <span className="text-xs text-muted-foreground">
                              {new Date(activity.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">
                            {activity.content}
                          </p>
                          {activity.itemCount && (
                            <Badge variant="secondary" className="mt-1">
                              {activity.itemCount} items
                            </Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Communication Summary */}
      <Card className="glass-card">
        <CardHeader>
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => onToggleSection('communication')}
          >
            <CardTitle className="flex items-center">
              <MessageSquare className="w-5 h-5 mr-2" />
              Inter-Agent Communication
            </CardTitle>
            {expandedSections.has('communication') ?
              <ChevronDown className="w-5 h-5" /> :
              <ChevronRight className="w-5 h-5" />
            }
          </div>
        </CardHeader>
        <AnimatePresence>
          {expandedSections.has('communication') && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
            >
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-2">
                    {communications.slice(0, 10).map((event) => (
                      <div
                        key={event.id}
                        className="flex items-start space-x-3 p-2 rounded-lg hover:bg-accent/5 transition-colors"
                      >
                        <Network className={`w-4 h-4 mt-1 ${getPriorityColor(event.priority)}`} />
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-medium">
                              {event.fromAgent} → {event.toAgent}
                            </p>
                            <span className="text-xs text-muted-foreground">
                              {new Date(event.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge variant="outline" className="text-xs">
                              {event.messageType}
                            </Badge>
                            <Badge
                              variant="secondary"
                              className={`text-xs ${getPriorityColor(event.priority)}`}
                            >
                              {event.priority}
                            </Badge>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">
                            {event.content}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </>
  )
}

export function MemoryTab({ memoryActivities }: { memoryActivities: MemoryActivity[] }) {
  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle>Detailed Memory Operations</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96">
          <div className="space-y-3">
            {memoryActivities.map((activity) => (
              <Card key={activity.id} className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <div className={`p-2 rounded ${getMemoryColor(activity.type)}`}>
                      {getMemoryIcon(activity.type)}
                    </div>
                    <div>
                      <p className="font-medium">
                        {activity.type.replace('_', ' ').toUpperCase()} • {activity.action.toUpperCase()}
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        {activity.agentName || 'System'}
                      </p>
                      <p className="text-sm mt-2">{activity.content}</p>
                      {activity.relevance && (
                        <div className="mt-2">
                          <p className="text-xs text-muted-foreground">Relevance</p>
                          <Progress value={activity.relevance * 100} className="h-1 mt-1" />
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-muted-foreground">
                      {new Date(activity.timestamp).toLocaleTimeString()}
                    </p>
                    {activity.itemCount && (
                      <Badge className="mt-2">{activity.itemCount} items</Badge>
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

export function CommunicationTab({ communications }: { communications: CommunicationEvent[] }) {
  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle>Communication Log</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96">
          <div className="space-y-3">
            {communications.map((event) => (
              <Card key={event.id} className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <Network className={`w-5 h-5 mt-1 ${getPriorityColor(event.priority)}`} />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <p className="font-medium">
                          {event.fromAgent} → {event.toAgent}
                        </p>
                        <Badge variant="outline">{event.messageType}</Badge>
                        <Badge className={getPriorityColor(event.priority)}>
                          {event.priority}
                        </Badge>
                      </div>
                      <p className="text-sm mt-2">{event.content}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="secondary" className="text-xs">
                          {event.status}
                        </Badge>
                      </div>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

export function LearningTab({ learningInsights }: { learningInsights: LearningInsight[] }) {
  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle>Learning Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96">
          <div className="space-y-3">
            {learningInsights.map((insight) => (
              <Card key={insight.id} className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <Sparkles className="w-5 h-5 mt-1 text-warning" />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <p className="font-medium">{insight.category}</p>
                        <Badge className={getImpactColor(insight.impact)}>
                          {insight.impact} impact
                        </Badge>
                      </div>
                      <p className="text-sm mt-2">{insight.description}</p>
                      <div className="mt-2">
                        <p className="text-xs text-muted-foreground">Confidence</p>
                        <Progress value={insight.confidence * 100} className="h-1 mt-1" />
                      </div>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {new Date(insight.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
