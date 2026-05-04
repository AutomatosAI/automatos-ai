'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ArrowLeft,
  Play,
  Pause,
  Square,
  Maximize2,
  Minimize2,
  Bot,
  Brain,
  Zap,
  CheckCircle,
  Clock,
  AlertCircle,
  MessageSquare,
  FileText,
  Sparkles,
  Activity,
  Lightbulb,
  Loader2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { apiClient } from '@/lib/api-client'
import { WorkflowStreamViewer } from './workflow-stream-viewer'
import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'
import { toast } from '@/hooks/use-toast'
import { ToastAction } from '@/components/ui/toast'

// New theater sub-components
import { TheaterStageProgress } from './theater/theater-stage-progress'
import { TheaterStepExecution, type StepExecutionData } from './theater/theater-step-execution'
import {
  TheaterSelfLearningPanel,
  type LearningData,
  type QualityData,
  type MemoryData,
} from './theater/theater-self-learning-panel'
import { PlaybookSuggestionsPanel, type SuggestionsData } from './playbook-suggestions-panel'
import { PlaybookStepProgress, type PlaybookStepResult } from './playbook-step-progress'

interface ExecutionKitchenProps {
  workflowId: number
  onBack: () => void
  autoStart?: boolean
  executionType?: 'workflow' | 'recipe'
  recipeExecutionId?: string
  recipeId?: string
  recipeSteps?: Array<{ step_id: string; order: number; prompt_template: string; agent_id: number }>
}

interface LogEntry {
  id: string
  timestamp: Date
  stage: number
  type: 'stage_start' | 'stage_complete' | 'agent_spawn' | 'task_progress' | 'task_complete' | 'task_error' | 'inter_agent' | 'memory_write' | 'info'
  message: string
  agent?: string
  details?: string
  fullResponse?: string
  tokens?: number
  duration?: number
  model?: string
  taskDescription?: string
  toolsUsed?: string[]
}

// 9-Stage names for log display
const STAGE_NAMES = [
  'Task Decomposition',
  'Agent Selection',
  'Context Engineering',
  'Agent Execution',
  'Result Aggregation',
  'Learning Update',
  'Quality Assessment',
  'Memory Storage',
  'Response Generation',
]

const STAGE_SHORT_NAMES = [
  'Decompose', 'Select', 'Context', 'Execute', 'Aggregate',
  'Learn', 'Quality', 'Memory', 'Response',
]

const STAGE_COLORS = [
  '#ff6b35', '#ff8c5a', '#fbbf24', '#ff6b35', '#10b981',
  '#f97316', '#ff6b35', '#f97316', '#10b981',
]

const formatDuration = (ms?: number) => (ms ? `${(ms / 1000).toFixed(1)}s` : '—')

// Streaming Log Component
function StreamingLog({ logs, selectedStage, onClearFilter }: { logs: LogEntry[], selectedStage: number | null, onClearFilter: () => void }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (autoScroll && scrollRef.current) scrollRef.current.scrollIntoView({ behavior: 'smooth' })
  }, [logs, autoScroll])

  const filteredLogs = selectedStage ? logs.filter(log => log.stage === selectedStage) : logs

  const toggleExpanded = (logId: string) => {
    setExpandedLogs(prev => {
      const next = new Set(prev)
      if (next.has(logId)) next.delete(logId)
      else next.add(logId)
      return next
    })
  }

  const getLogIcon = (type: LogEntry['type']) => {
    switch (type) {
      case 'stage_start': return <Zap className="w-4 h-4 text-primary" />
      case 'stage_complete': return <CheckCircle className="w-4 h-4 text-success" />
      case 'agent_spawn': return <Bot className="w-4 h-4 text-cyan-400" />
      case 'task_progress': return <Activity className="w-4 h-4 text-info" />
      case 'task_complete': return <CheckCircle className="w-4 h-4 text-success" />
      case 'task_error': return <AlertCircle className="w-4 h-4 text-destructive" />
      case 'inter_agent': return <MessageSquare className="w-4 h-4 text-pink-400" />
      case 'memory_write': return <Brain className="w-4 h-4 text-orange-400" />
      default: return <Clock className="w-4 h-4 text-muted-foreground" />
    }
  }

  const getLogClass = (type: LogEntry['type']) => {
    switch (type) {
      case 'task_complete': case 'stage_complete': return 'log-entry-success'
      case 'task_error': return 'log-entry-error'
      case 'agent_spawn': case 'inter_agent': return 'log-entry-agent'
      default: return 'log-entry-info'
    }
  }

  const formatTime = (date: Date) => date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + '.' + date.getMilliseconds().toString().padStart(3, '0')

  return (
    <div className="h-full flex flex-col glass-panel rounded-2xl overflow-hidden">
      <div className="flex items-center justify-between p-4 border-b border-border/30">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-primary" />
          <h3 className="font-semibold">Execution Log</h3>
          <Badge variant="outline" className="bg-white/5">{filteredLogs.length} events</Badge>
          {selectedStage && <Badge variant="secondary" className="bg-primary/20 text-primary cursor-pointer" onClick={onClearFilter}>Stage {selectedStage} only ✕</Badge>}
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => setExpandedLogs(new Set(filteredLogs.map(l => l.id)))} className="text-xs text-muted-foreground hover:text-foreground">Expand All</Button>
          <Button variant="ghost" size="sm" onClick={() => setExpandedLogs(new Set())} className="text-xs text-muted-foreground hover:text-foreground">Collapse All</Button>
          <Button variant="ghost" size="sm" onClick={() => setAutoScroll(!autoScroll)} className={cn("text-xs", autoScroll ? "text-success" : "text-muted-foreground")}>{autoScroll ? '⬇ Auto-scroll ON' : '⬇ Auto-scroll OFF'}</Button>
        </div>
      </div>
      <ScrollArea className="flex-1 theater-scrollbar">
        <div className="p-4 space-y-3">
          <AnimatePresence initial={false}>
            {filteredLogs.map((log) => {
              const isExpanded = expandedLogs.has(log.id)
              const hasExpandableContent = log.fullResponse || log.details || log.taskDescription

              return (
                <motion.div
                  key={log.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className={cn("log-entry p-4 cursor-pointer transition-all", getLogClass(log.type), isExpanded && "ring-1 ring-white/10")}
                  onClick={() => hasExpandableContent && toggleExpanded(log.id)}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 flex-shrink-0">{getLogIcon(log.type)}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center flex-wrap gap-2 mb-2">
                        <span className="text-xs text-muted-foreground font-mono-logs">{formatTime(log.timestamp)}</span>
                        <Badge variant="outline" className="text-[10px] h-5" style={{ borderColor: STAGE_COLORS[log.stage - 1] + '40', color: STAGE_COLORS[log.stage - 1] }}>
                          {STAGE_SHORT_NAMES[log.stage - 1] || `S${log.stage}`}
                        </Badge>
                        {log.agent && <Badge variant="secondary" className="text-[10px] h-5 bg-primary/20 text-primary">🤖 {log.agent}</Badge>}
                        {log.model && <Badge variant="outline" className="text-[10px] h-5 text-cyan-400 border-cyan-400/30">{log.model}</Badge>}
                        {hasExpandableContent && (
                          <span className="text-[10px] text-muted-foreground ml-auto">
                            {isExpanded ? '▼ Click to collapse' : '▶ Click for full details'}
                          </span>
                        )}
                      </div>

                      <p className="text-sm font-medium mb-2">{log.message}</p>

                      {log.taskDescription && (
                        <div className="text-sm text-muted-foreground mb-2">
                          <span className="text-xs text-primary/60 uppercase">Task:</span> {log.taskDescription}
                        </div>
                      )}

                      <AnimatePresence>
                        {isExpanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            {log.fullResponse && (
                              <div className="mt-3 p-3 bg-black/30 rounded-lg border border-border/30">
                                <div className="text-xs text-primary/60 uppercase mb-2 flex items-center gap-2">
                                  <FileText className="w-3 h-3" />
                                  Full Response
                                </div>
                                <pre className="text-sm text-foreground/90 font-mono-logs whitespace-pre-wrap break-words max-h-[400px] overflow-y-auto theater-scrollbar">
                                  {log.fullResponse}
                                </pre>
                              </div>
                            )}

                            {log.details && !log.fullResponse && (
                              <div className="mt-3 p-3 bg-black/30 rounded-lg border border-border/30">
                                <div className="text-xs text-primary/60 uppercase mb-2">Details</div>
                                <pre className="text-sm text-foreground/90 font-mono-logs whitespace-pre-wrap break-words">
                                  {log.details}
                                </pre>
                              </div>
                            )}

                            {log.toolsUsed && log.toolsUsed.length > 0 && (
                              <div className="mt-3 flex items-center gap-2 flex-wrap">
                                <span className="text-xs text-primary/60 uppercase">Tools:</span>
                                {log.toolsUsed.map((tool, i) => (
                                  <Badge key={i} variant="outline" className="text-[10px] text-orange-400 border-orange-400/30">{tool}</Badge>
                                ))}
                              </div>
                            )}
                          </motion.div>
                        )}
                      </AnimatePresence>

                      {((log.tokens && log.tokens > 0) || (log.duration && log.duration > 0)) && (
                        <div className="flex items-center gap-4 mt-3 pt-2 border-t border-border/30 text-xs text-muted-foreground">
                          {log.tokens && log.tokens > 0 && <span className="flex items-center gap-1"><Sparkles className="w-3 h-3 text-primary" />{log.tokens.toLocaleString()} tokens</span>}
                          {log.duration && log.duration > 0 && <span className="flex items-center gap-1"><Clock className="w-3 h-3 text-success" />{(log.duration / 1000).toFixed(2)}s</span>}
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </AnimatePresence>
          <div ref={scrollRef} />
        </div>
      </ScrollArea>
    </div>
  )
}

// Metrics Sidebar Component
function MetricsSidebar({ currentStage, executionData, isExecuting }: { currentStage: number, executionData: any, isExecuting: boolean }) {
  const execution = executionData?.execution
  const subtasks = execution?.output_data?.subtasks || []
  const completedTasks = subtasks.filter((s: any) => s.execution_result?.status === 'completed').length
  const totalTokens = subtasks.reduce((sum: number, s: any) => sum + (s.execution_result?.tokens_used || 0), 0)
  const avgQuality = subtasks.length > 0 ? subtasks.reduce((sum: number, s: any) => sum + (s.context_enhancement?.context_quality || s.context_quality || 0), 0) / subtasks.length : 0

  const CircularProgress = ({ value, size = 100, strokeWidth = 8 }: { value: number, size?: number, strokeWidth?: number }) => {
    const radius = (size - strokeWidth) / 2
    const circumference = radius * 2 * Math.PI
    const offset = circumference - (value / 100) * circumference
    return (
      <svg width={size} height={size} className="circular-progress">
        <circle className="circular-progress-bg" strokeWidth={strokeWidth} fill="none" r={radius} cx={size / 2} cy={size / 2} />
        <circle className="circular-progress-fill" strokeWidth={strokeWidth} fill="none" r={radius} cx={size / 2} cy={size / 2} strokeDasharray={circumference} strokeDashoffset={offset} />
      </svg>
    )
  }

  return (
    <div className="h-full flex flex-col gap-4">
      <div className="glass-panel rounded-2xl p-4">
        <div className="text-xs text-muted-foreground mb-2">CURRENT STAGE</div>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: STAGE_COLORS[currentStage - 1] + '20' }}>
            <span className="text-sm font-bold" style={{ color: STAGE_COLORS[currentStage - 1] }}>{currentStage}</span>
          </div>
          <div>
            <div className="font-semibold">{STAGE_NAMES[currentStage - 1]}</div>
            <div className="text-xs text-muted-foreground">Stage {currentStage} of 9</div>
          </div>
        </div>
      </div>

      <div className="glass-panel rounded-2xl p-4 flex flex-col items-center">
        <div className="relative">
          <CircularProgress value={(completedTasks / Math.max(subtasks.length, 1)) * 100} />
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-bold">{subtasks.length > 0 ? Math.round((completedTasks / subtasks.length) * 100) : 0}%</span>
            <span className="text-xs text-muted-foreground">Complete</span>
          </div>
        </div>
        <div className="mt-3 text-sm text-center">
          <span className="text-success font-medium">{completedTasks}</span>
          <span className="text-muted-foreground"> / {subtasks.length} tasks</span>
        </div>
      </div>

      <div className="glass-panel rounded-2xl p-4 flex-1">
        <div className="text-xs text-muted-foreground mb-3">ACTIVE AGENTS</div>
        <div className="space-y-2">
          {subtasks.slice(0, 5).map((subtask: any, i: number) => {
            const agent = subtask.selected_agent
            const status = subtask.execution_result?.status || 'pending'
            const isRunning = status === 'running' || (isExecuting && status === 'pending' && i === 0)
            if (!agent) return null
            return (
              <div key={i} className="flex items-center gap-2">
                <div className={cn("w-8 h-8 rounded-lg agent-avatar flex items-center justify-center text-xs font-bold", isRunning && "agent-avatar-active")}>{agent.agent_name?.charAt(0) || 'A'}</div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">{agent.agent_name || 'Agent'}</div>
                  <div className="flex items-center gap-2">
                    <div className={cn("w-1.5 h-1.5 rounded-full", status === 'completed' ? "bg-success" : isRunning ? "bg-primary animate-pulse" : "bg-gray-500")} />
                    <span className="text-xs text-muted-foreground">{status === 'completed' ? 'Done' : isRunning ? 'Running' : 'Pending'}</span>
                  </div>
                </div>
              </div>
            )
          })}
          {subtasks.length === 0 && <div className="text-center text-muted-foreground text-sm py-4">No agents assigned yet</div>}
        </div>
      </div>

      <div className="glass-panel rounded-2xl p-4">
        <div className="grid grid-cols-2 gap-3">
          <div><div className="text-xs text-muted-foreground">Tokens</div><div className="text-lg font-semibold">{totalTokens.toLocaleString()}</div></div>
          <div><div className="text-xs text-muted-foreground">Quality</div><div className="text-lg font-semibold">{(avgQuality * 100).toFixed(0)}%</div></div>
          <div><div className="text-xs text-muted-foreground">Duration</div><div className="text-lg font-semibold">{execution?.duration || '--'}</div></div>
          <div><div className="text-xs text-muted-foreground">Status</div><Badge variant={isExecuting ? 'default' : 'secondary'} className={cn("mt-1", isExecuting && "bg-primary/20 text-primary")}>{isExecuting ? 'Running' : execution?.status || 'Ready'}</Badge></div>
        </div>
      </div>
    </div>
  )
}

// Build step execution data from workflow execution subtasks
function buildStepExecutionData(execution: any): StepExecutionData[] {
  if (!execution?.output_data?.subtasks) return []

  const subtasks = execution.output_data.subtasks || []
  return subtasks.map((subtask: any, index: number) => {
    const execResult = subtask.execution_result || {}
    const agent = subtask.selected_agent || {}

    let status: StepExecutionData['status'] = 'pending'
    if (execResult.status === 'completed') status = 'success'
    else if (execResult.status === 'failed') status = 'failed'
    else if (execResult.status === 'running') status = 'running'

    return {
      step_id: subtask.id || subtask.subtask_id || `step-${index + 1}`,
      order: index + 1,
      agent_id: agent.agent_id,
      agent_name: agent.agent_name,
      agent_model: agent.model_id || agent.agent_type,
      agent_tool_count: agent.capabilities?.length,
      prompt_template: subtask.description,
      status,
      started_at: execResult.started_at,
      completed_at: execResult.completed_at,
      duration_ms: execResult.execution_time_ms,
      tokens_used: execResult.tokens_used,
      messages: execResult.messages,
      tool_calls: execResult.tool_calls?.map((tc: any) => ({
        app_name: tc.app_name || tc.tool_name || 'Tool',
        action_name: tc.action_name || tc.function_name || tc.name || 'action',
        params: tc.params || tc.arguments,
        result: tc.result || tc.output,
        status: tc.status || (tc.error ? 'error' : 'success'),
        duration_ms: tc.duration_ms,
      })),
      result: execResult.llm_response || execResult.result || execResult.output,
      error: execResult.error,
      retries: execResult.retries || 0,
    } as StepExecutionData
  })
}

// Build self-learning data from execution output
function buildLearningData(execution: any): LearningData | null {
  const learningSystem = execution?.output_data?.learning_system_metadata ||
    execution?.output_data?.learning_system ||
    execution?.output_data?.learning_updates

  if (!learningSystem) return null

  const updates = learningSystem.updates || learningSystem
  if (!updates || (Array.isArray(updates) && updates.length === 0)) return null

  const patterns = Array.isArray(updates)
    ? updates.map((u: any) => ({
        type: u.type || 'learning',
        description: u.summary || u.description || JSON.stringify(u),
        severity: u.severity || 'medium' as const,
        step: u.step_index ?? u.subtask_index ?? null,
      }))
    : Object.entries(updates).map(([key, value]: [string, any]) => ({
        type: key,
        description: typeof value === 'string' ? value : (value?.summary || JSON.stringify(value)),
        severity: 'medium' as const,
        step: null,
      }))

  return {
    patterns,
    suggestions: [],
    performance_metrics: {
      total_duration_ms: execution?.output_data?.total_execution_time_ms,
      success_rate: execution?.output_data?.subtasks?.length
        ? execution.output_data.subtasks.filter((s: any) => s.execution_result?.status === 'completed').length / execution.output_data.subtasks.length
        : undefined,
    },
  }
}

// Build quality data from execution output
function buildQualityData(execution: any): QualityData | null {
  const qualityScores = execution?.output_data?.quality_scores ||
    execution?.output_data?.result_aggregation_metadata?.quality_scores

  if (!qualityScores) return null

  const overall = qualityScores.overall ?? qualityScores.quality_score ?? 0

  return {
    quality_score: overall,
    breakdown: {
      completeness: qualityScores.completeness ?? overall,
      accuracy: qualityScores.accuracy ?? overall,
      efficiency: qualityScores.efficiency ?? overall,
      reliability: qualityScores.reliability ?? overall,
      cost: qualityScores.cost ?? overall,
    },
    grade: overall >= 0.9 ? 'A' : overall >= 0.75 ? 'B' : overall >= 0.6 ? 'C' : overall >= 0.4 ? 'D' : 'F',
    bottlenecks: [],
  }
}

// Build memory data from execution output
function buildMemoryData(execution: any): MemoryData | null {
  const memoryStorage = execution?.output_data?.memory_storage_metadata ||
    execution?.output_data?.memory_storage ||
    execution?.output_data?.memory_integration_summary ||
    execution?.output_data?.memory_consolidation_metadata

  if (!memoryStorage) return null

  return {
    stored_memories: memoryStorage.stored_count || memoryStorage.memories_stored || 0,
    scopes: memoryStorage.scopes || [],
    memories: (memoryStorage.entries || memoryStorage.memories || []).map((m: any) => ({
      scope: m.scope || 'unknown',
      type: m.type || 'execution',
      agent_id: m.agent_id,
      step_index: m.step_index,
    })),
    errors: memoryStorage.errors,
  }
}

// Main Component
export function ExecutionKitchen({
  workflowId,
  onBack,
  autoStart = false,
  executionType = 'workflow',
  recipeExecutionId: initialRecipeExecutionId,
  recipeId,
  recipeSteps,
}: ExecutionKitchenProps) {
  const isRecipeMode = executionType === 'recipe'

  const [isExecuting, setIsExecuting] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [executionData, setExecutionData] = useState<any>(null)
  const [hasAutoStarted, setHasAutoStarted] = useState(false)
  const [currentExecutionId, setCurrentExecutionId] = useState<number | null>(null)
  const [currentStage, setCurrentStage] = useState(1)
  const [completedStages, setCompletedStages] = useState<number[]>([])
  const [selectedStageFilter, setSelectedStageFilter] = useState<number | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [useAISDK, setUseAISDK] = useState(false)

  // Step execution data for TheaterStepExecution components
  const [stepExecutions, setStepExecutions] = useState<StepExecutionData[]>([])

  // Self-learning data for TheaterSelfLearningPanel
  const [learningData, setLearningData] = useState<LearningData | null>(null)
  const [qualityData, setQualityData] = useState<QualityData | null>(null)
  const [memoryData, setMemoryData] = useState<MemoryData | null>(null)

  // Suggestions notification state
  const [showSuggestionsPanel, setShowSuggestionsPanel] = useState(false)
  const [suggestionsData, setSuggestionsData] = useState<any>(null)
  const executionCompleteCountRef = useRef(0)
  const prevIsExecutingRef = useRef(false)

  // Recipe direct execution state
  const [recipeExecId, setRecipeExecId] = useState<string | undefined>(initialRecipeExecutionId)
  const [recipeExecData, setRecipeExecData] = useState<any>(null)
  const [recipeStepResults, setPlaybookStepResults] = useState<PlaybookStepResult[]>([])

  // Recipe execution polling — stops when status is completed/failed
  const recipePollingRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (!isRecipeMode || !recipeId || !recipeExecId) return

    // Set initial executing state
    setIsExecuting(true)
    setExecutionData({ name: 'Recipe Execution', description: 'Direct recipe execution' })

    let stopped = false

    const poll = async () => {
      if (stopped) return
      try {
        const data: any = await apiClient.getRecipeExecution(recipeId, recipeExecId!)
        if (stopped) return
        setRecipeExecData(data)
        setPlaybookStepResults(data.step_results || [])
        setExecutionData((prev: any) => ({
          ...prev,
          name: data.recipe_name || prev?.name || 'Playbook',
          description: `Execution ${data.execution_id}`,
          execution: data,
        }))

        if (data.status === 'completed' || data.status === 'failed') {
          setIsExecuting(false)
          // Stop polling — execution is done
          if (recipePollingRef.current) {
            clearInterval(recipePollingRef.current)
            recipePollingRef.current = null
          }
        }
      } catch (err) {
        console.error('Error polling recipe execution:', err)
      }
    }

    // Initial fetch
    poll()

    // Poll every 3s while running
    recipePollingRef.current = setInterval(poll, 3000)

    return () => {
      stopped = true
      if (recipePollingRef.current) {
        clearInterval(recipePollingRef.current)
        recipePollingRef.current = null
      }
    }
  }, [isRecipeMode, recipeId, recipeExecId])

  const logsEndRef = useRef<HTMLDivElement>(null)
  const execution = executionData?.execution
  const lastExecutionIdRef = useRef<number | null>(null)
  const handleStartExecutionRef = useRef<() => Promise<void>>()

  // Update step execution and self-learning data when execution changes
  const updateDerivedData = useCallback((exec: any) => {
    if (!exec) return
    setStepExecutions(buildStepExecutionData(exec))
    setLearningData(buildLearningData(exec))
    setQualityData(buildQualityData(exec))
    setMemoryData(buildMemoryData(exec))
  }, [])

  const generateLogsFromExecution = useCallback((execution: any) => {
    if (!execution) return

    const output = execution.output_data || {}
    const hasNoData = !output.subtasks && !output.agent_selection_metadata && !output.context_engineering_metadata
    if (hasNoData && execution.status !== 'completed') {
      setCurrentStage(1)
      setCompletedStages([])
      setLogs([{
        id: 'start',
        timestamp: new Date(execution.started_at || Date.now()),
        stage: 1,
        type: 'stage_start',
        message: '🚀 Starting workflow execution...'
      }])
      return
    }

    const subtasks = output.subtasks || []
    const agentSelectionMeta = output.agent_selection_metadata
    const contextMeta = output.context_engineering_metadata || {
      summary: output.context_engineering?.summary,
      enhancements: output.context_engineering?.enhancements,
    }
    const agentExecutionMeta = output.agent_execution_metadata || output.agent_execution
    const resultAggregationMeta = output.result_aggregation_metadata || output.result_aggregation
    const learningSystemMeta = output.learning_system_metadata || output.learning_system
    const learningUpdates = Array.isArray(output.learning_updates) ? output.learning_updates : []
    const memoryStorageMeta = output.memory_storage_metadata || output.memory_storage
    const memoryIntegrationSummary = output.memory_integration_summary
    const newLogs: LogEntry[] = []
    let time = new Date(execution.started_at || Date.now())
    let current = 1

    // Stage 1: Task Decomposition
    if (subtasks.length > 0) {
      newLogs.push({
        id: 'decomp-start', timestamp: time, stage: 1, type: 'stage_start',
        message: `⚡ STAGE 1: TASK DECOMPOSITION`,
        details: `Original query: ${execution.input_data?.user_query || execution.input_data?.query || 'N/A'}`
      })
      time = new Date(time.getTime() + 500)
      const decompositionDetails = subtasks.map((s: any, i: number) =>
        `\n━━━ Subtask ${i + 1} ━━━\nDescription: ${s.description || 'N/A'}\nComplexity: ${s.complexity || 'N/A'}\nRequired Capabilities: ${s.required_capabilities?.join(', ') || 'N/A'}\nPriority: ${s.priority || 'N/A'}`
      ).join('\n')
      newLogs.push({ id: 'decomp-result', timestamp: time, stage: 1, type: 'stage_complete', message: `✓ Decomposed into ${subtasks.length} subtasks`, fullResponse: decompositionDetails || 'No subtasks generated' })
      current = 2
    }

    // Stage 2: Agent Selection
    const assignmentEntries = Object.entries(agentSelectionMeta?.summary?.assignments || agentSelectionMeta?.assignments || {}).filter(([key]) => key.startsWith('subtask_'))
    const hasAgents = assignmentEntries.length > 0 || subtasks.some((s: any) => s.selected_agent)
    if (hasAgents) {
      time = new Date(time.getTime() + 500)
      newLogs.push({ id: 'select-start', timestamp: time, stage: 2, type: 'stage_start', message: `⚡ STAGE 2: AGENT SELECTION` })
      assignmentEntries.forEach(([key, agents]) => {
        const subtaskIndex = parseInt(key.replace('subtask_', ''), 10)
        const subtaskDescription = subtasks[subtaskIndex]?.description || `Subtask ${subtaskIndex + 1}`
        ;(agents as any[]).forEach((agent, idx) => {
          time = new Date(time.getTime() + 200)
          newLogs.push({
            id: `agent-${key}-${idx}`, timestamp: time, stage: 2, type: 'agent_spawn',
            message: `🤖 Selected: ${agent.agent_name}`, agent: agent.agent_name, taskDescription: subtaskDescription,
            fullResponse: `Match Score: ${(agent.match_score * 100).toFixed(1)}%\nReasoning: ${agent.reasoning || 'N/A'}`
          })
        })
      })
      time = new Date(time.getTime() + 300)
      newLogs.push({ id: 'select-complete', timestamp: time, stage: 2, type: 'stage_complete', message: `✓ STAGE 2: AGENT SELECTION COMPLETE` })
      current = 3
    }

    // Stage 3: Context Engineering
    const subtasksWithContext = contextMeta?.summary?.subtasks_with_context ?? 0
    const shouldShowContextStage = !!contextMeta || subtasks.length > 0
    if (shouldShowContextStage) {
      time = new Date(time.getTime() + 300)
      const enhancements = contextMeta?.enhancements || {}
      const contextDetails = subtasks.map((s: any, i: number) => {
        const enhancement = enhancements[`subtask_${i}`] || enhancements[i]
        const quality = enhancement?.context_quality ?? s?.context_quality ?? 0
        const sources = enhancement?.num_sources ?? (s?.retrieved_docs?.length || 0) + (s?.memory_items?.length || 0)
        return `Subtask ${i + 1}:\n  Context Quality: ${(quality * 100).toFixed(2)}%\n  Sources Retrieved: ${sources}`
      }).join('\n')
      newLogs.push({ id: 'context-start', timestamp: time, stage: 3, type: 'stage_start', message: `⚡ STAGE 3: CONTEXT ENGINEERING`, fullResponse: contextDetails || 'Context preparation in progress...' })
      time = new Date(time.getTime() + 300)
      newLogs.push({ id: 'context-complete', timestamp: time, stage: 3, type: 'stage_complete', message: subtasksWithContext > 0 ? `✓ STAGE 3: CONTEXT ENGINEERING COMPLETE` : `⚠️ STAGE 3: CONTEXT ENGINEERING - No context retrieved` })
      current = 4
    }

    // Stage 4: Agent Execution
    const execResults = agentExecutionMeta?.results || {}
    const resultEntries = Object.entries(execResults)
    const hasExecutionResults = resultEntries.length > 0 || subtasks.some((s: any) => s.execution_result)
    const allTasksDone = agentExecutionMeta?.summary?.completed && agentExecutionMeta?.summary?.total_subtasks
      ? agentExecutionMeta.summary.completed >= agentExecutionMeta.summary.total_subtasks
      : resultEntries.length > 0
    if (hasExecutionResults) {
      time = new Date(time.getTime() + 300)
      newLogs.push({ id: 'exec-start', timestamp: time, stage: 4, type: 'stage_start', message: `⚡ STAGE 4: AGENT EXECUTION` })
      resultEntries.forEach(([key, result]: any) => {
        const subtaskIndex = parseInt(key.replace('subtask_', ''), 10)
        const subtaskDescription = subtasks[subtaskIndex]?.description || `Subtask ${subtaskIndex + 1}`
        time = new Date(time.getTime() + 800)
        newLogs.push({
          id: `exec-${key}`, timestamp: time, stage: 4,
          type: result.status === 'completed' ? 'task_complete' : 'task_error',
          message: result.status === 'completed' ? `✓ ${subtaskDescription.slice(0, 80)}` : `✗ ${subtaskDescription.slice(0, 80)} failed: ${result.error || 'Unknown error'}`,
          agent: result.agent_name, tokens: result.tokens_used, duration: result.execution_time_ms, taskDescription: subtaskDescription
        })
      })
      if (allTasksDone && subtasks.length > 0) {
        time = new Date(time.getTime() + 500)
        newLogs.push({ id: 'exec-complete', timestamp: time, stage: 4, type: 'stage_complete', message: `✓ STAGE 4: AGENT EXECUTION COMPLETE` })
        current = 5
      } else {
        current = 4
      }
    }

    // Stage 5: Result Aggregation
    if (resultAggregationMeta || output.aggregated_result || output.final_report) {
      time = new Date(time.getTime() + 500)
      newLogs.push({ id: 'aggregate', timestamp: time, stage: 5, type: 'stage_complete', message: `⚡ STAGE 5: RESULT AGGREGATION`, fullResponse: output.aggregated_result || output.final_report || 'Results aggregated' })
      current = 6
    }

    // Stage 6: Learning Update
    const hasLearningUpdate = !!(learningSystemMeta?.updates && Object.keys(learningSystemMeta.updates).length > 0)
    if (hasLearningUpdate) {
      time = new Date(time.getTime() + 300)
      newLogs.push({ id: 'learning', timestamp: time, stage: 6, type: 'stage_complete', message: `⚡ STAGE 6: LEARNING UPDATE`, fullResponse: JSON.stringify(learningSystemMeta.updates, null, 2) })
      current = 7
    }

    // Stage 7: Quality Assessment
    const qualityScores = resultAggregationMeta?.quality_scores || output.quality_scores
    if (qualityScores) {
      time = new Date(time.getTime() + 300)
      newLogs.push({ id: 'quality', timestamp: time, stage: 7, type: 'stage_complete', message: `⚡ STAGE 7: QUALITY ASSESSMENT - Overall ${(qualityScores.overall * 100).toFixed(1)}%` })
      current = 8
    }

    // Stage 8: Memory Storage
    if (memoryStorageMeta || memoryIntegrationSummary) {
      time = new Date(time.getTime() + 300)
      newLogs.push({ id: 'memory', timestamp: time, stage: 8, type: 'memory_write', message: `⚡ STAGE 8: MEMORY STORAGE`, fullResponse: JSON.stringify(memoryStorageMeta || memoryIntegrationSummary, null, 2) })
      current = 9
    }

    // Stage 9: Final Response
    if (execution.status === 'completed') {
      time = new Date(time.getTime() + 500)
      const finalReport = execution.output_data?.final_report || execution.output_data?.aggregated_result || subtasks[subtasks.length - 1]?.execution_result?.llm_response || 'Workflow completed'
      newLogs.push({ id: 'complete', timestamp: time, stage: 9, type: 'stage_complete', message: `✅ STAGE 9: RESPONSE GENERATION COMPLETE`, fullResponse: finalReport })
      current = 9
    }

    const hasContextRetrieved = subtasksWithContext > 0 || subtasks.some((s: any) => (s.context_enhancement?.num_sources || 0) > 0 || (s.retrieved_docs?.length || 0) > 0 || (s.memory_items?.length || 0) > 0)

    const stageCompletionFlags: Record<number, boolean> = {
      1: subtasks.length > 0, 2: hasAgents,
      3: hasContextRetrieved || current > 3, 4: hasExecutionResults && allTasksDone,
      5: !!(resultAggregationMeta || output.aggregated_result || output.final_report),
      6: hasLearningUpdate || learningUpdates.length > 0, 7: !!qualityScores,
      8: !!(memoryStorageMeta || memoryIntegrationSummary), 9: execution.status === 'completed'
    }

    let highestComplete = 0
    for (let stage = 1; stage <= 9; stage++) {
      if (stageCompletionFlags[stage] && highestComplete === stage - 1) highestComplete = stage
      else break
    }

    const sequentialCompleted = Array.from({ length: highestComplete }, (_, idx) => idx + 1)
    const nextStage = Math.min(highestComplete + 1, 9)

    const execId = execution?.id ?? execution?.execution_id ?? null
    if (execId !== null && execId !== lastExecutionIdRef.current) {
      lastExecutionIdRef.current = execId
    }

    setCompletedStages(sequentialCompleted)
    setCurrentStage(nextStage)
    setLogs(newLogs)

    // Update derived data for new sub-components
    updateDerivedData(execution)
  }, [updateDerivedData])

  const loadExecutionById = useCallback(async (executionId: number) => {
    try {
      const executionDetails = await apiClient.getWorkflowExecution(executionId.toString())
      setExecutionData((prev: any) => ({ ...prev, execution: executionDetails }))
      generateLogsFromExecution(executionDetails)
    } catch (err) { console.error('Error loading execution:', err) }
  }, [generateLogsFromExecution])

  // AISDK stream
  const [aisdkConnected, setAisdkConnected] = useState(false)
  const streamAbortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!currentExecutionId || !useAISDK || isRecipeMode) return

    const connectAISDKStream = async () => {
      try {
        setAisdkConnected(true)
        streamAbortRef.current = new AbortController()

        const apiUrl = process.env.NEXT_PUBLIC_API_URL ||
          (typeof window !== 'undefined' ? window.location.origin : '')

        if (!apiUrl) {
          throw new Error('NEXT_PUBLIC_API_URL not configured.')
        }

        const url = `${apiUrl}/api/workflows/executions/${currentExecutionId}/stream/aisdk`
        const response = await fetch(url, { method: 'GET', signal: streamAbortRef.current.signal })

        if (!response.ok) throw new Error(`Stream failed: ${response.statusText}`)

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        if (!reader) throw new Error('No response body')

        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (!line.trim()) continue

            if (line.startsWith('d:')) {
              try {
                const event = JSON.parse(line.slice(2))

                if (event.type === 'subtask_update') {
                  const update = event.data || event
                  const newLog: LogEntry = {
                    id: `subtask-${update.subtask_id || Date.now()}-${Math.random()}`,
                    timestamp: new Date(update.timestamp || Date.now()),
                    stage: 4,
                    type: update.status === 'completed' ? 'task_complete' : update.status === 'failed' ? 'task_error' : 'task_progress',
                    message: `${update.status === 'completed' ? '✓' : update.status === 'failed' ? '✗' : '⏳'} ${update.subtask_description || 'Processing...'}`,
                    taskDescription: update.subtask_description,
                    agent: update.agent_name,
                    tokens: update.tokens_used,
                    duration: update.execution_time_ms,
                    fullResponse: update.llm_response || update.result_preview
                  }
                  setLogs((prev: LogEntry[]) => [...prev, newLog])
                  if (currentExecutionId) loadExecutionById(currentExecutionId)
                } else if (event.type === 'execution_log') {
                  const logData = event.data || event
                  const newLog: LogEntry = {
                    id: `log-${Date.now()}-${Math.random()}`,
                    timestamp: new Date(logData.timestamp || Date.now()),
                    stage: logData.stage || 4,
                    type: 'info',
                    message: logData.message || logData.data?.message,
                    details: logData.details || logData.data?.details
                  }
                  setLogs((prev: LogEntry[]) => [...prev, newLog])
                } else if (event.type === 'workflow_complete') {
                  setIsExecuting(false)
                  if (currentExecutionId) loadExecutionById(currentExecutionId)
                } else if (event.type === 'error') {
                  const errorLog: LogEntry = {
                    id: `error-${Date.now()}-${Math.random()}`,
                    timestamp: new Date(),
                    stage: 4,
                    type: 'task_error',
                    message: `❌ Error: ${event.message || event.data?.message || 'Unknown error'}`,
                    details: event.details || JSON.stringify(event)
                  }
                  setLogs((prev: LogEntry[]) => [...prev, errorLog])
                }
              } catch (err) {
                console.error('Error parsing AISDK data:', err, line)
              }
            } else if (line.startsWith('e:')) {
              try {
                const errorData = JSON.parse(line.slice(2))
                const errorLog: LogEntry = {
                  id: `error-${Date.now()}-${Math.random()}`,
                  timestamp: new Date(),
                  stage: 4,
                  type: 'task_error',
                  message: `❌ Stream Error: ${errorData.message || 'Unknown error'}`,
                  details: JSON.stringify(errorData)
                }
                setLogs((prev: LogEntry[]) => [...prev, errorLog])
              } catch (err) {
                console.error('Error parsing AISDK error:', err)
              }
            }
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.error('AISDK stream error:', err)
          setAisdkConnected(false)
        }
      } finally {
        setAisdkConnected(false)
      }
    }

    connectAISDKStream()

    return () => {
      if (streamAbortRef.current) streamAbortRef.current.abort()
    }
  }, [currentExecutionId, loadExecutionById, useAISDK, isRecipeMode])

  // Polling for real-time updates during execution
  useEffect(() => {
    if (!isExecuting || !currentExecutionId) return

    const pollInterval = setInterval(() => {
      loadExecutionById(currentExecutionId)
    }, 3000)

    return () => clearInterval(pollInterval)
  }, [isExecuting, currentExecutionId, loadExecutionById])

  // Detect execution completion and check for suggestions
  useEffect(() => {
    if (prevIsExecutingRef.current && !isExecuting && completedStages.includes(9)) {
      executionCompleteCountRef.current += 1

      // Fetch suggestions from the recipe's learning data
      const execRecipeId = executionData?.template_id || executionData?.recipe_id
      if (execRecipeId) {
        apiClient.getRecipeSuggestions(execRecipeId).then((data: any) => {
          if (data?.suggestions?.length > 0) {
            setSuggestionsData(data)

            // Check if quality is below threshold for 3+ consecutive low-quality executions
            const qualityThreshold = 0.7
            const isLowQuality = data.quality_score !== null && data.quality_score < qualityThreshold
            const consecutiveLow = executionCompleteCountRef.current >= 3 || (data.analysis_count >= 3 && isLowQuality)

            if (isLowQuality || data.suggestions.length > 0) {
              const highPriority = data.suggestions.filter((s: any) => s.priority === 'high')
              toast({
                title: `${executionData.name || 'Playbook'} has improvement suggestions`,
                description: highPriority.length > 0
                  ? `${highPriority.length} high-priority suggestion${highPriority.length !== 1 ? 's' : ''} found`
                  : `${data.suggestions.length} suggestion${data.suggestions.length !== 1 ? 's' : ''} based on execution analysis`,
                action: React.createElement(
                  ToastAction,
                  {
                    altText: 'Review suggestions',
                    onClick: () => setShowSuggestionsPanel(true),
                  } as any,
                  'Review'
                ) as any,
              })
            }
          }
        }).catch((err: any) => {
          console.error('Error fetching suggestions:', err)
        })
      }
    }
    prevIsExecutingRef.current = isExecuting
  }, [isExecuting, completedStages, executionData])

  useEffect(() => {
    if (isRecipeMode) return // Recipe mode uses its own polling via useRecipeExecution
    const loadWorkflow = async () => {
      try {
        const workflow = await apiClient.getWorkflow(workflowId.toString())
        const execResponse: any = await apiClient.getWorkflowExecutions(workflowId.toString())
        const executions = (execResponse?.items || execResponse || []) as any[]
        let executionDetails: any = null
        if (executions.length > 0) {
          const latest = executions.sort((a: any, b: any) => b.id - a.id)[0]
          executionDetails = await apiClient.getWorkflowExecution(latest.id.toString())
          setCurrentExecutionId(latest.id)
          if (executionDetails?.status === 'running' || executionDetails?.status === 'pending') setIsExecuting(true)
          generateLogsFromExecution(executionDetails)
        }
        setExecutionData({ ...(workflow as any), execution: executionDetails })
      } catch (error) { console.error('Error loading workflow:', error) }
    }
    loadWorkflow()
  }, [workflowId, generateLogsFromExecution, isRecipeMode])

  const handleStartExecution = async () => {
    try {
      // Guard: check if there's already a running/pending execution for this workflow
      const execResponse: any = await apiClient.getWorkflowExecutions(workflowId.toString())
      const executions = (execResponse?.items || execResponse || []) as any[]
      const activeExec = executions.find((e: any) => e.status === 'running' || e.status === 'pending')
      if (activeExec) {
        setCurrentExecutionId(activeExec.id)
        setIsExecuting(true)
        loadExecutionById(activeExec.id)
        return
      }

      lastExecutionIdRef.current = null
      setIsExecuting(true)
      setCurrentStage(1)
      setCompletedStages([])
      setStepExecutions([])
      setLearningData(null)
      setQualityData(null)
      setMemoryData(null)
      setLogs([{
        id: 'start', timestamp: new Date(), stage: 1, type: 'stage_start',
        message: '🚀 Starting workflow execution...'
      }])

      const result: any = await apiClient.executeWorkflow(workflowId.toString(), {})
      if (result?.execution_id || result?.id) {
        const execId = result.execution_id || result.id
        lastExecutionIdRef.current = execId
        setCurrentExecutionId(execId)
        setTimeout(() => loadExecutionById(execId), 500)
      }
    } catch (error) {
      console.error('Error starting execution:', error)
      setIsExecuting(false)
    }
  }

  // Keep ref in sync so the auto-start effect always calls the latest version
  handleStartExecutionRef.current = handleStartExecution

  // Auto-start effect — uses ref to avoid stale closure over handleStartExecution
  useEffect(() => {
    if (autoStart && !hasAutoStarted && executionData) {
      setHasAutoStarted(true)
      setTimeout(() => handleStartExecutionRef.current?.(), 1000)
    }
  }, [autoStart, hasAutoStarted, executionData])

  const handlePauseExecution = async () => {
    try {
      if (streamAbortRef.current) streamAbortRef.current.abort()
      if (currentExecutionId) {
        try { await apiClient.cancelWorkflowExecution(currentExecutionId.toString()) } catch (apiError) { console.error('Failed to cancel:', apiError) }
      }
      setIsExecuting(false)
      setLogs(prev => [...prev, { id: `pause-${Date.now()}`, timestamp: new Date(), stage: currentStage, type: 'info', message: '⏸️ Execution paused by user' }])
    } catch (error) {
      console.error('Error during pause:', error)
      setIsExecuting(false)
    }
  }

  const handleStopExecution = async () => {
    try {
      if (streamAbortRef.current) streamAbortRef.current.abort()
      if (currentExecutionId) {
        try { await apiClient.cancelWorkflowExecution(currentExecutionId.toString()) } catch (apiError) { console.error('Failed to stop:', apiError) }
      }
      setIsExecuting(false)
      setLogs(prev => [...prev, { id: `stop-${Date.now()}`, timestamp: new Date(), stage: currentStage, type: 'task_error', message: '🛑 Execution stopped by user' }])
    } catch (error) {
      console.error('Error during stop:', error)
      setIsExecuting(false)
    }
  }

  const handleStageClick = (stage: number) => {
    if (completedStages.includes(stage) || stage === currentStage) {
      setSelectedStageFilter(selectedStageFilter === stage ? null : stage)
    }
  }

  if (!executionData) {
    return (
      <div className="h-screen theater-bg flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Loading workflow...</p>
        </div>
      </div>
    )
  }

  const showSelfLearning = currentStage >= 6 || completedStages.includes(6)

  return (
    <div className={cn("h-screen flex flex-col theater-bg", isFullscreen && "fixed inset-0 z-50")}>
      {/* Header */}
      <div className="glass-panel-light border-b border-border/30">
        <div className="flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" onClick={onBack} className="gap-2"><ArrowLeft className="w-4 h-4" />Back</Button>
            <div>
              <h1 className="text-lg font-semibold"><span className="gradient-text">{executionData.name}</span></h1>
              <p className="text-xs text-muted-foreground line-clamp-1">{executionData.description}</p>
            </div>
            <Badge variant="outline" className={cn("text-xs", aisdkConnected ? "bg-success/10 text-success border-success/30" : "bg-secondary/50 text-muted-foreground border-border/30")}>
              <div className={cn("w-2 h-2 rounded-full mr-2", aisdkConnected ? "bg-success animate-pulse" : "bg-gray-400")} />
              {aisdkConnected ? 'Live' : 'Offline'}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            {isRecipeMode ? (
              // Recipe mode: execution already started, just show status
              isExecuting && (
                <Badge variant="outline" className="bg-orange-500/10 text-orange-400 border-orange-500/30">
                  <Loader2 className="w-3 h-3 mr-2 animate-spin" />
                  Cooking...
                </Badge>
              )
            ) : (
              // Workflow mode: full execution controls
              !isExecuting ? (
                <Button onClick={handleStartExecution} className="gap-2 glow-orange"><Play className="w-4 h-4" />Execute</Button>
              ) : (
                <>
                  <Button variant="outline" size="sm" onClick={handlePauseExecution}><Pause className="w-4 h-4 mr-2" />Pause</Button>
                  <Button variant="outline" size="sm" className="text-destructive hover:text-destructive/80" onClick={handleStopExecution}><Square className="w-4 h-4 mr-2" />Stop</Button>
                </>
              )
            )}
            <Button variant="ghost" size="icon" onClick={() => setIsFullscreen(!isFullscreen)}>
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Recipe Mode: Step Progress | Workflow Mode: 9-Stage Pipeline */}
      {isRecipeMode ? (
        <div className="px-6 py-4">
          <PlaybookStepProgress
            currentStep={recipeExecData?.current_step || 0}
            totalSteps={recipeExecData?.total_steps || recipeSteps?.length || 0}
            status={
              recipeExecData?.status === 'completed' ? 'completed' :
              recipeExecData?.status === 'failed' ? 'failed' :
              recipeExecData?.status === 'running' ? 'running' : 'pending'
            }
            stepResults={recipeStepResults}
            steps={recipeSteps}
            recipeId={recipeId}
            executionId={recipeExecId}
          />
        </div>
      ) : (
        <>
          {/* Stage Pipeline - using new TheaterStageProgress component */}
          <div className="px-6 py-4">
            <TheaterStageProgress currentStage={currentStage} onStageClick={handleStageClick} />
          </div>

          {/* Step Execution Timeline - visible when we have step data */}
          {stepExecutions.length > 0 && (
            <div className="px-6 pb-4">
              <div className="glass-panel rounded-2xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Zap className="w-4 h-4 text-primary" />
                  <h3 className="text-sm font-semibold">Step Execution Timeline</h3>
                  <Badge variant="outline" className="text-[10px] h-5 ml-auto">
                    {stepExecutions.filter(s => s.status === 'success').length}/{stepExecutions.length} completed
                  </Badge>
                </div>
                <div className="space-y-3 max-h-[300px] overflow-y-auto theater-scrollbar pr-1">
                  {stepExecutions.map((step, index) => (
                    <TheaterStepExecution key={step.step_id} step={step} index={index} />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Self-Learning Panel - visible after Learn stage (stage 6+) */}
          {showSelfLearning && (learningData || qualityData || memoryData) && (
            <div className="px-6 pb-4">
              <TheaterSelfLearningPanel
                learningData={learningData}
                qualityData={qualityData}
                memoryData={memoryData}
                defaultOpen={false}
              />
            </div>
          )}
        </>
      )}

      {/* Suggestions Panel - shown when user clicks "Review" on notification toast */}
      <AnimatePresence>
        {showSuggestionsPanel && suggestionsData && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="px-6 pb-4 overflow-hidden"
          >
            <div className="glass-panel rounded-2xl p-4 border border-orange-400/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Lightbulb className="w-4 h-4 text-orange-400" />
                  <h3 className="text-sm font-semibold">Improvement Suggestions</h3>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowSuggestionsPanel(false)}
                  className="text-xs text-muted-foreground hover:text-foreground h-7"
                >
                  Dismiss
                </Button>
              </div>
              <PlaybookSuggestionsPanel
                data={suggestionsData}
                playbookName={executionData?.name}
                onApplyPromptRewrite={(suggestion) => {
                  toast({
                    title: 'Prompt Rewrite',
                    description: `Edit recipe step ${suggestion.target_step !== undefined ? suggestion.target_step + 1 : ''} with suggested prompt improvement`,
                  })
                }}
                onApplyModelUpgrade={(suggestion) => {
                  toast({
                    title: 'Model Upgrade Suggested',
                    description: suggestion.description,
                  })
                }}
                onApplyToolAddition={(suggestion) => {
                  toast({
                    title: 'Tool Addition Suggested',
                    description: suggestion.description,
                  })
                }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content: Log (75%) + Sidebar (25%) */}
      <div className="flex-1 flex gap-4 px-6 pb-6 min-h-0">
        <div className="flex-1 flex flex-col gap-3">
          {isRecipeMode ? (
            /* Recipe mode: show execution output or waiting message */
            <div className="flex-1 glass-panel rounded-2xl p-4 overflow-auto theater-scrollbar">
              {recipeExecData?.status === 'completed' && recipeExecData?.output_data?.final_output ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="w-4 h-4 text-success" />
                    <h3 className="text-sm font-semibold text-success">Execution Complete</h3>
                    {recipeExecData?.output_data?.total_duration_ms && (
                      <span className="text-xs text-muted-foreground ml-auto">
                        {(recipeExecData.output_data.total_duration_ms / 1000).toFixed(1)}s total
                      </span>
                    )}
                  </div>
                  <pre className="text-sm text-foreground/90 whitespace-pre-wrap break-words">
                    {recipeExecData.output_data.final_output}
                  </pre>
                </div>
              ) : recipeExecData?.status === 'failed' ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertCircle className="w-4 h-4 text-destructive" />
                    <h3 className="text-sm font-semibold text-destructive">Execution Failed</h3>
                  </div>
                  <p className="text-sm text-destructive/80">{recipeExecData.error_message || 'Unknown error'}</p>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <Loader2 className="w-8 h-8 animate-spin text-primary mb-3" />
                  <p className="text-sm text-muted-foreground">Executing recipe steps...</p>
                  <p className="text-xs text-muted-foreground/60 mt-1">
                    Step {recipeExecData?.current_step || 0} of {recipeExecData?.total_steps || '?'}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <>
              {/* Stream View Toggle */}
              <div className="flex items-center justify-between glass-panel rounded-xl p-2">
                <div className="flex items-center gap-2">
                  <Label htmlFor="stream-toggle" className="text-xs text-muted-foreground cursor-pointer">
                    Stream View:
                  </Label>
                  <div className="flex items-center gap-2 bg-black/20 rounded-lg p-1">
                    <button
                      onClick={() => setUseAISDK(false)}
                      className={cn(
                        "px-3 py-1 text-xs rounded-md transition-all",
                        !useAISDK ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground"
                      )}
                    >
                      Detailed Logs
                    </button>
                    <button
                      onClick={() => setUseAISDK(true)}
                      className={cn(
                        "px-3 py-1 text-xs rounded-md transition-all",
                        useAISDK ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground"
                      )}
                    >
                      AI SDK Stream
                    </button>
                  </div>
                </div>
                {useAISDK && currentExecutionId && (
                  <Badge variant="outline" className="text-xs bg-primary/10 text-primary border-primary/30">
                    <Sparkles className="w-3 h-3 mr-1" />
                    Real-time AI SDK
                  </Badge>
                )}
              </div>

              {/* Conditional View */}
              {useAISDK && currentExecutionId ? (
                <WorkflowStreamViewer
                  executionId={currentExecutionId}
                  className="flex-1"
                  onStageUpdate={(stage, stageName) => {
                    setCurrentStage(stage);
                    if (!completedStages.includes(stage - 1) && stage > 1) {
                      setCompletedStages(prev => [...prev, stage - 1]);
                    }
                  }}
                  onComplete={() => {
                    setIsExecuting(false);
                    setCompletedStages([1, 2, 3, 4, 5, 6, 7, 8, 9]);
                    if (currentExecutionId) loadExecutionById(currentExecutionId);
                  }}
                />
              ) : (
                <StreamingLog logs={logs} selectedStage={selectedStageFilter} onClearFilter={() => setSelectedStageFilter(null)} />
              )}
            </>
          )}
        </div>
        {!isRecipeMode && (
          <div className="w-[300px]">
            <MetricsSidebar currentStage={currentStage} executionData={executionData} isExecuting={isExecuting} />
          </div>
        )}
      </div>
    </div>
  )
}
