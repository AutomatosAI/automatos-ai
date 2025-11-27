'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
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
  Database,
  CheckCircle,
  Clock,
  AlertCircle,
  MessageSquare,
  Layers,
  Search,
  FileText,
  GitBranch,
  Target,
  Sparkles,
  Activity
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { apiClient } from '@/lib/api-client'
import { useWorkflowWebSocket } from '@/hooks/use-workflow-websocket'
import { useWorkflowStream } from '@/hooks/useWorkflowStream'
import { cn } from '@/lib/utils'

interface ExecutionTheaterProps {
  workflowId: number
  onBack: () => void
  autoStart?: boolean
}

interface StageInfo {
  id: number
  name: string
  shortName: string
  icon: React.ReactNode
  description: string
  color: string
}

interface LogEntry {
  id: string
  timestamp: Date
  stage: number
  type: 'stage_start' | 'stage_complete' | 'agent_spawn' | 'task_progress' | 'task_complete' | 'task_error' | 'inter_agent' | 'memory_write' | 'info'
  message: string
  agent?: string
  details?: string
  fullResponse?: string  // Full LLM response, not truncated
  tokens?: number
  duration?: number
  model?: string
  taskDescription?: string
  toolsUsed?: string[]
}

// 9-Stage Configuration - Using platform colors (orange accent, emerald success)
const STAGES: StageInfo[] = [
  { id: 1, name: 'Task Decomposition', shortName: 'Decompose', icon: <GitBranch className="w-4 h-4" />, description: 'Breaking task into subtasks', color: '#ff6b35' },
  { id: 2, name: 'Agent Selection', shortName: 'Select', icon: <Bot className="w-4 h-4" />, description: 'Choosing best agents', color: '#ff8c5a' },
  { id: 3, name: 'Context Engineering', shortName: 'Context', icon: <Search className="w-4 h-4" />, description: 'Optimizing context', color: '#fbbf24' },
  { id: 4, name: 'Agent Execution', shortName: 'Execute', icon: <Zap className="w-4 h-4" />, description: 'Running agents', color: '#ff6b35' },
  { id: 5, name: 'Result Aggregation', shortName: 'Aggregate', icon: <Layers className="w-4 h-4" />, description: 'Combining results', color: '#10b981' },
  { id: 6, name: 'Learning Update', shortName: 'Learn', icon: <Brain className="w-4 h-4" />, description: 'Extracting patterns', color: '#f97316' },
  { id: 7, name: 'Quality Assessment', shortName: 'Quality', icon: <Target className="w-4 h-4" />, description: 'Validating output', color: '#ff6b35' },
  { id: 8, name: 'Memory Storage', shortName: 'Memory', icon: <Database className="w-4 h-4" />, description: 'Storing learnings', color: '#f97316' },
  { id: 9, name: 'Response Generation', shortName: 'Response', icon: <FileText className="w-4 h-4" />, description: 'Final output', color: '#10b981' },
]

const formatPercent = (value?: number) => `${Math.round(((value ?? 0) * 100) || 0)}%`
const formatDuration = (ms?: number) => (ms ? `${(ms / 1000).toFixed(1)}s` : '—')

// Stage Pipeline Component
function StagePipeline({ currentStage, completedStages, onStageClick }: { currentStage: number, completedStages: number[], onStageClick: (stage: number) => void }) {
  return (
    <div className="glass-panel rounded-2xl p-4">
      <div className="flex items-center justify-between gap-2">
        {STAGES.map((stage, index) => {
          const isCompleted = completedStages.includes(stage.id)
          const isActive = currentStage === stage.id
          const isPending = !isCompleted && !isActive

          return (
            <div key={stage.id} className="flex items-center flex-1">
              <motion.button
                onClick={() => onStageClick(stage.id)}
                className={cn(
                  "relative flex flex-col items-center justify-center p-3 rounded-xl border-2 transition-all min-w-[80px]",
                  isCompleted && "stage-completed",
                  isActive && "stage-active",
                  isPending && "stage-pending"
                )}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div className={cn("mb-1 transition-colors", isCompleted && "text-emerald-400", isActive && "text-primary", isPending && "text-gray-500")} style={{ color: isActive ? stage.color : undefined }}>
                  {isCompleted ? <CheckCircle className="w-5 h-5" /> : stage.icon}
                </div>
                <span className={cn("text-xs font-medium", isCompleted && "text-emerald-300", isActive && "text-white", isPending && "text-gray-500")}>{stage.shortName}</span>
                <div className={cn("absolute -top-1 -right-1 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold", isCompleted && "bg-emerald-500 text-white", isActive && "bg-primary text-white", isPending && "bg-gray-700 text-gray-400")}>{stage.id}</div>
                {isActive && <motion.div className="absolute -bottom-1 left-1/2 w-2 h-2 rounded-full bg-primary" animate={{ scale: [1, 1.5, 1] }} transition={{ duration: 1.5, repeat: Infinity }} style={{ marginLeft: -4 }} />}
              </motion.button>
              {index < STAGES.length - 1 && (
                <div className="flex-1 mx-1 relative h-[2px]">
                  <div className={cn("absolute inset-0 rounded-full", isCompleted ? "stage-connector-completed" : isActive ? "stage-connector-active" : "stage-connector")} />
                  {isActive && <div className="absolute inset-0 overflow-hidden"><div className="flow-particle w-4 h-full bg-gradient-to-r from-transparent via-primary to-transparent" /></div>}
                </div>
              )}
            </div>
          )
        })}
      </div>
      <div className="mt-4 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <motion.div className="h-full bg-gradient-to-r from-emerald-500 via-primary to-orange-600 relative" initial={{ width: 0 }} animate={{ width: `${((currentStage - 1 + (completedStages.length / 9)) / 9) * 100}%` }} transition={{ duration: 0.5 }}>
          <div className="absolute inset-0 progress-shimmer" />
        </motion.div>
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
        <span>Stage {currentStage} of 9</span>
        <span>{STAGES[currentStage - 1]?.description}</span>
      </div>
    </div>
  )
}

interface StageDetailPanelProps {
  stage: number
  subtasks: any[]
  execution?: any
  collapsed: boolean
  onToggle: () => void
}

function StageMetric({ label, value, hint }: { label: string, value: React.ReactNode, hint?: string }) {
  return (
    <div className="rounded-xl border border-white/5 bg-white/5 p-3">
      <div className="text-[11px] uppercase text-muted-foreground tracking-wide">{label}</div>
      <div className="text-lg font-semibold mt-1">{value}</div>
      {hint && <div className="text-xs text-muted-foreground mt-0.5">{hint}</div>}
    </div>
  )
}

function StageDetailPanel({ stage, subtasks, execution, collapsed, onToggle }: StageDetailPanelProps) {
  const stageInfo = STAGES[stage - 1]
  if (!stageInfo) return null

  const aggregatedResult = execution?.output_data?.aggregated_result || execution?.output_data?.final_report
  const learningUpdates = execution?.output_data?.learning_updates || []
  const memoryWrites = execution?.output_data?.memory_stored

  const renderStage1 = () => {
    const highPriority = subtasks.filter((s: any) => (s.priority || '').toLowerCase() === 'high').length
    const withoutAgents = subtasks.filter((s: any) => !s.selected_agent).length
    const completedDrafts = subtasks.filter((s: any) => s.execution_result).length

    return (
      <>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StageMetric label="Subtasks" value={subtasks.length} hint="Created in decomposition" />
          <StageMetric label="High priority" value={highPriority} hint="Marked as high urgency" />
          <StageMetric label="Awaiting agent" value={withoutAgents} hint="Need assignment in Stage 2" />
          <StageMetric label="With drafts" value={completedDrafts} hint="Already have agent output" />
        </div>
        <div className="overflow-x-auto mt-4 max-h-64 pr-1">
          <table className="min-w-full text-sm">
            <thead className="text-[11px] uppercase text-muted-foreground">
              <tr>
                <th className="py-2 pr-3 text-left">#</th>
                <th className="py-2 pr-3 text-left">Task</th>
                <th className="py-2 pr-3 text-left">Priority</th>
                <th className="py-2 pr-3 text-left">Assigned Agent</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {subtasks.map((task: any, index: number) => (
                <tr key={`stage1-${index}`} className="align-top">
                  <td className="py-2 pr-3 text-muted-foreground">{index + 1}</td>
                  <td className="py-2 pr-3">{task.description || '—'}</td>
                  <td className="py-2 pr-3 text-xs uppercase text-muted-foreground">{task.priority || 'n/a'}</td>
                  <td className="py-2 pr-3 text-sm text-muted-foreground">{task.selected_agent?.agent_name || 'Not selected'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    )
  }

  const renderStage2 = () => {
    const assigned = subtasks.filter((s: any) => s.selected_agent).length
    const unassigned = subtasks.length - assigned
    return (
      <>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StageMetric label="Agents Assigned" value={`${assigned}/${subtasks.length}`} hint="Tasks with an agent selected" />
          <StageMetric label="Pending Assignment" value={unassigned} hint="Tasks still waiting for an agent" />
          <StageMetric label="Avg. Match Score" value={
            assigned
              ? `${Math.round((subtasks.reduce((sum: number, s: any) => sum + (s.agent_score || 0), 0) / assigned) * 100)}%`
              : '—'
          } hint="Based on selector output" />
        </div>
        <div className="overflow-x-auto mt-4 max-h-64 pr-1">
          <table className="min-w-full text-sm">
            <thead className="text-[11px] uppercase text-muted-foreground">
              <tr>
                <th className="py-2 pr-3 text-left">Task</th>
                <th className="py-2 pr-3 text-left">Agent</th>
                <th className="py-2 pr-3 text-left">Agent Type</th>
                <th className="py-2 pr-3 text-left">Match</th>
                <th className="py-2 pr-3 text-left">Capabilities</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {subtasks.map((task: any, index: number) => (
                <tr key={`stage2-${index}`}>
                  <td className="py-2 pr-3">{task.description || '—'}</td>
                  <td className="py-2 pr-3">{task.selected_agent?.agent_name || <span className="text-muted-foreground">Unassigned</span>}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{task.selected_agent?.agent_type || '—'}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{task.agent_score ? formatPercent(task.agent_score) : 'n/a'}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{task.selected_agent?.capabilities?.join(', ') || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    )
  }

  const renderStage3 = () => {
    // Read from context_enhancement object instead of retrieved_docs/memory_items
    const withContext = subtasks.filter((s: any) => (s.context_enhancement?.num_sources || 0) > 0).length
    const withoutContext = subtasks.length - withContext

    return (
      <>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StageMetric label="Context Ready" value={`${withContext}/${subtasks.length}`} hint="Tasks with retrieved docs or memories" />
          <StageMetric label="Missing Context" value={withoutContext} hint="Needs RAG or memory" />
          <StageMetric label="Avg. Context Quality" value={
            subtasks.length ? formatPercent(subtasks.reduce((sum: number, s: any) => sum + (s.context_enhancement?.context_quality || s.context_quality || 0), 0) / subtasks.length) : '0%'
          } />
        </div>
        {withoutContext === subtasks.length && (
          <div className="mt-4 p-3 rounded-xl border border-amber-500/30 bg-amber-500/10 text-amber-100 text-sm flex items-start gap-2">
            <AlertCircle className="w-4 h-4 mt-0.5" />
            <div>
              <div className="font-semibold text-amber-200">No context retrieved</div>
              <p>RAG and memory lookups returned zero sources for every subtask. Verify embeddings or vector DB connectivity.</p>
            </div>
          </div>
        )}
        <div className="overflow-x-auto mt-4 max-h-64 pr-1">
          <table className="min-w-full text-sm">
            <thead className="text-[11px] uppercase text-muted-foreground">
              <tr>
                <th className="py-2 pr-3 text-left">Task</th>
                <th className="py-2 pr-3 text-left">Context Quality</th>
                <th className="py-2 pr-3 text-left">Docs</th>
                <th className="py-2 pr-3 text-left">Memories</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {subtasks.map((task: any, index: number) => (
                <tr key={`stage3-${index}`}>
                  <td className="py-2 pr-3">{task.description || '—'}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{formatPercent(task.context_enhancement?.context_quality || task.context_quality || 0)}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{task.context_enhancement?.num_sources || 0}</td>
                  <td className="py-2 pr-3 text-muted-foreground">0</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    )
  }

  const renderStage4 = () => {
    const completed = subtasks.filter((s: any) => s.execution_result?.status === 'completed').length
    const failed = subtasks.filter((s: any) => s.execution_result?.status === 'failed').length
    return (
      <>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StageMetric label="Completed" value={`${completed}/${subtasks.length}`} />
          <StageMetric label="Failed" value={failed} />
          <StageMetric label="Avg. Tokens" value={
            completed ? Math.round(subtasks.reduce((sum: number, s: any) => sum + (s.execution_result?.tokens_used || 0), 0) / completed) : 0
          } hint="Per completed task" />
        </div>
        <div className="overflow-x-auto mt-4 max-h-64 pr-1">
          <table className="min-w-full text-sm">
            <thead className="text-[11px] uppercase text-muted-foreground">
              <tr>
                <th className="py-2 pr-3 text-left">Task</th>
                <th className="py-2 pr-3 text-left">Agent</th>
                <th className="py-2 pr-3 text-left">Status</th>
                <th className="py-2 pr-3 text-left">Tokens</th>
                <th className="py-2 pr-3 text-left">Duration</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {subtasks.map((task: any, index: number) => (
                <tr key={`stage4-${index}`} className="align-top">
                  <td className="py-2 pr-3">{task.description || '—'}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{task.selected_agent?.agent_name || '—'}</td>
                  <td className="py-2 pr-3">
                    <Badge variant="outline" className={cn(
                      'text-[11px]',
                      task.execution_result?.status === 'completed' && 'text-emerald-400 border-emerald-400/40',
                      task.execution_result?.status === 'failed' && 'text-red-400 border-red-400/40'
                    )}>
                      {task.execution_result?.status || 'pending'}
                    </Badge>
                  </td>
                  <td className="py-2 pr-3 text-muted-foreground">{task.execution_result?.tokens_used?.toLocaleString() || '—'}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{formatDuration(task.execution_result?.execution_time_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    )
  }

  const renderStage5 = () => (
    aggregatedResult
      ? <pre className="text-sm bg-black/30 border border-white/5 rounded-xl p-4 whitespace-pre-wrap max-h-64 overflow-y-auto">{aggregatedResult}</pre>
      : <p className="text-sm text-muted-foreground">No aggregated result has been generated yet.</p>
  )

  const renderStage6 = () => (
    learningUpdates.length > 0
      ? (
        <ul className="space-y-3">
          {learningUpdates.map((update: any, index: number) => (
            <li key={`learning-${index}`} className="p-3 rounded-xl border border-white/5 bg-white/5">
              <div className="text-xs text-muted-foreground uppercase mb-1">{update.agent_name || 'Agent'}</div>
              <p className="text-sm">{update.summary || JSON.stringify(update, null, 2)}</p>
            </li>
          ))}
        </ul>
      )
      : <p className="text-sm text-muted-foreground">No learning updates were recorded for this execution.</p>
  )

  const renderStage7 = () => {
    const avgQuality = subtasks.length
      ? subtasks.reduce((sum: number, s: any) => sum + (s.context_quality || 0), 0) / subtasks.length
      : 0
    const lowQuality = subtasks.filter((s: any) => (s.context_quality || 0) < 0.3).length
    return (
      <>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StageMetric label="Overall Quality" value={formatPercent(avgQuality)} hint="Proxy from context quality" />
          <StageMetric label="At Risk Tasks" value={lowQuality} hint="<30% quality" />
        </div>
        <p className="text-sm text-muted-foreground mt-3">
          Quality combines completeness, accuracy, efficiency, reliability, and coherence. Improve context quality to raise these scores.
        </p>
      </>
    )
  }

  const renderStage8 = () => (
    memoryWrites
      ? <pre className="text-sm bg-black/30 border border-white/5 rounded-xl p-4 whitespace-pre-wrap max-h-64 overflow-y-auto">{JSON.stringify(memoryWrites, null, 2)}</pre>
      : <p className="text-sm text-muted-foreground">No new memories were stored for this run.</p>
  )

  const renderStage9 = () => (
    execution?.output_data?.final_report
      ? <pre className="text-sm bg-black/30 border border-white/5 rounded-xl p-4 whitespace-pre-wrap max-h-64 overflow-y-auto">{execution.output_data.final_report}</pre>
      : <p className="text-sm text-muted-foreground">Final response has not been generated yet.</p>
  )

  const stageContent = () => {
    switch (stage) {
      case 1: return renderStage1()
      case 2: return renderStage2()
      case 3: return renderStage3()
      case 4: return renderStage4()
      case 5: return renderStage5()
      case 6: return renderStage6()
      case 7: return renderStage7()
      case 8: return renderStage8()
      case 9: return renderStage9()
      default: return <p className="text-sm text-muted-foreground">No diagnostics available for this stage.</p>
    }
  }

  return (
    <div className={cn("glass-panel rounded-2xl p-5 transition-all", collapsed && "py-3")}>
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Stage Insight</p>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold truncate">{stageInfo.name}</h3>
            <Badge variant="outline" className="text-xs flex-shrink-0">{stageInfo.shortName}</Badge>
          </div>
          <p className="text-sm text-muted-foreground truncate">{stageInfo.description}</p>
        </div>
        <Button variant="ghost" size="sm" onClick={onToggle} className="text-xs">
          {collapsed ? (
            <>
              <Maximize2 className="w-3 h-3 mr-2" /> Expand
            </>
          ) : (
            <>
              <Minimize2 className="w-3 h-3 mr-2" /> Collapse
            </>
          )}
        </Button>
      </div>
      {!collapsed && (
        <div className="mt-4 space-y-4">
          {stageContent()}
        </div>
      )}
    </div>
  )
}

// Streaming Log Component with FULL details
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
      case 'stage_complete': return <CheckCircle className="w-4 h-4 text-emerald-400" />
      case 'agent_spawn': return <Bot className="w-4 h-4 text-cyan-400" />
      case 'task_progress': return <Activity className="w-4 h-4 text-blue-400" />
      case 'task_complete': return <CheckCircle className="w-4 h-4 text-emerald-400" />
      case 'task_error': return <AlertCircle className="w-4 h-4 text-red-400" />
      case 'inter_agent': return <MessageSquare className="w-4 h-4 text-pink-400" />
      case 'memory_write': return <Brain className="w-4 h-4 text-orange-400" />
      default: return <Clock className="w-4 h-4 text-gray-400" />
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
      <div className="flex items-center justify-between p-4 border-b border-white/5">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-primary" />
          <h3 className="font-semibold">Execution Log</h3>
          <Badge variant="outline" className="bg-white/5">{filteredLogs.length} events</Badge>
          {selectedStage && <Badge variant="secondary" className="bg-primary/20 text-primary cursor-pointer" onClick={onClearFilter}>Stage {selectedStage} only ✕</Badge>}
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => setExpandedLogs(new Set(filteredLogs.map(l => l.id)))} className="text-xs text-muted-foreground hover:text-foreground">Expand All</Button>
          <Button variant="ghost" size="sm" onClick={() => setExpandedLogs(new Set())} className="text-xs text-muted-foreground hover:text-foreground">Collapse All</Button>
          <Button variant="ghost" size="sm" onClick={() => setAutoScroll(!autoScroll)} className={cn("text-xs", autoScroll ? "text-emerald-400" : "text-muted-foreground")}>{autoScroll ? '⬇ Auto-scroll ON' : '⬇ Auto-scroll OFF'}</Button>
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
                      {/* Header row */}
                      <div className="flex items-center flex-wrap gap-2 mb-2">
                        <span className="text-xs text-muted-foreground font-mono-logs">{formatTime(log.timestamp)}</span>
                        <Badge variant="outline" className="text-[10px] h-5" style={{ borderColor: STAGES[log.stage - 1]?.color + '40', color: STAGES[log.stage - 1]?.color }}>
                          {STAGES[log.stage - 1]?.shortName || `S${log.stage}`}
                        </Badge>
                        {log.agent && <Badge variant="secondary" className="text-[10px] h-5 bg-primary/20 text-primary">🤖 {log.agent}</Badge>}
                        {log.model && <Badge variant="outline" className="text-[10px] h-5 text-cyan-400 border-cyan-400/30">{log.model}</Badge>}
                        {hasExpandableContent && (
                          <span className="text-[10px] text-muted-foreground ml-auto">
                            {isExpanded ? '▼ Click to collapse' : '▶ Click for full details'}
                          </span>
                        )}
                      </div>

                      {/* Message - always visible */}
                      <p className="text-sm font-medium mb-2">{log.message}</p>

                      {/* Task description if available */}
                      {log.taskDescription && (
                        <div className="text-sm text-muted-foreground mb-2">
                          <span className="text-xs text-primary/60 uppercase">Task:</span> {log.taskDescription}
                        </div>
                      )}

                      {/* Expanded content - FULL details */}
                      <AnimatePresence>
                        {isExpanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            {/* Full LLM Response */}
                            {log.fullResponse && (
                              <div className="mt-3 p-3 bg-black/30 rounded-lg border border-white/5">
                                <div className="text-xs text-primary/60 uppercase mb-2 flex items-center gap-2">
                                  <FileText className="w-3 h-3" />
                                  Full Response
                                </div>
                                <pre className="text-sm text-foreground/90 font-mono-logs whitespace-pre-wrap break-words max-h-[400px] overflow-y-auto theater-scrollbar">
                                  {log.fullResponse}
                                </pre>
                              </div>
                            )}

                            {/* Additional details */}
                            {log.details && !log.fullResponse && (
                              <div className="mt-3 p-3 bg-black/30 rounded-lg border border-white/5">
                                <div className="text-xs text-primary/60 uppercase mb-2">Details</div>
                                <pre className="text-sm text-foreground/90 font-mono-logs whitespace-pre-wrap break-words">
                                  {log.details}
                                </pre>
                              </div>
                            )}

                            {/* Tools used */}
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

                      {/* Metrics row - only show if we have actual values */}
                      {((log.tokens && log.tokens > 0) || (log.duration && log.duration > 0)) && (
                        <div className="flex items-center gap-4 mt-3 pt-2 border-t border-white/5 text-xs text-muted-foreground">
                          {log.tokens && log.tokens > 0 && <span className="flex items-center gap-1"><Sparkles className="w-3 h-3 text-primary" />{log.tokens.toLocaleString()} tokens</span>}
                          {log.duration && log.duration > 0 && <span className="flex items-center gap-1"><Clock className="w-3 h-3 text-emerald-400" />{(log.duration / 1000).toFixed(2)}s</span>}
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
          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: STAGES[currentStage - 1]?.color + '20' }}>
            <span style={{ color: STAGES[currentStage - 1]?.color }}>{STAGES[currentStage - 1]?.icon}</span>
          </div>
          <div>
            <div className="font-semibold">{STAGES[currentStage - 1]?.name}</div>
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
          <span className="text-emerald-400 font-medium">{completedTasks}</span>
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
                    <div className={cn("w-1.5 h-1.5 rounded-full", status === 'completed' ? "bg-emerald-400" : isRunning ? "bg-primary animate-pulse" : "bg-gray-500")} />
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

// Main Component
export function ExecutionTheater({ workflowId, onBack, autoStart = false }: ExecutionTheaterProps) {
  const [isExecuting, setIsExecuting] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [executionData, setExecutionData] = useState<any>(null)
  const [hasAutoStarted, setHasAutoStarted] = useState(false)
  const [currentExecutionId, setCurrentExecutionId] = useState<number | null>(null)
  const [currentStage, setCurrentStage] = useState(1)
  const [completedStages, setCompletedStages] = useState<number[]>([])
  const [selectedStageFilter, setSelectedStageFilter] = useState<number | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const execution = executionData?.execution
  const subtasksForInsights = execution?.output_data?.subtasks || []
  const [isStagePanelCollapsed, setIsStagePanelCollapsed] = useState(false)
  const lastExecutionIdRef = useRef<number | null>(null)

  const generateLogsFromExecution = useCallback((execution: any) => {
    if (!execution) return

    const output = execution.output_data || {}

    // If execution is just starting (no output data yet), don't show stage 3 logs
    const hasNoData = !output.subtasks && !output.agent_selection_metadata && !output.context_engineering_metadata
    if (hasNoData && execution.status !== 'completed') {
      // Execution just started, only show stage 1
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
    const agentSelectionMeta = output.agent_selection_metadata || output.agent_selection_metadata
    const contextMeta = output.context_engineering_metadata || {
      summary: output.context_engineering?.summary,
      enhancements: output.context_engineering?.enhancements,
    }
    const agentExecutionMeta = output.agent_execution_metadata || output.agent_execution
    const resultAggregationMeta = output.result_aggregation_metadata || output.result_aggregation
    const learningSystemMeta = output.learning_system_metadata || output.learning_system
    const learningUpdates = Array.isArray(output.learning_updates)
      ? output.learning_updates
      : []
    const memoryStorageMeta = output.memory_storage_metadata || output.memory_storage
    const memoryIntegrationSummary = output.memory_integration_summary
    const memoryConsolidationMeta = output.memory_consolidation_metadata || output.memory_consolidation
    const newLogs: LogEntry[] = []
    let time = new Date(execution.started_at || Date.now())
    let current = 1

    // Stage 1: Task Decomposition - only if subtasks exist
    if (subtasks.length > 0) {
      newLogs.push({
        id: 'decomp-start',
        timestamp: time,
        stage: 1,
        type: 'stage_start',
        message: `⚡ STAGE 1: TASK DECOMPOSITION`,
        details: `Original query: ${execution.input_data?.user_query || execution.input_data?.query || 'N/A'}`
      })
      time = new Date(time.getTime() + 500)

      const decompositionDetails = subtasks.map((s: any, i: number) => {
        return `\n━━━ Subtask ${i + 1} ━━━\n` +
          `Description: ${s.description || 'N/A'}\n` +
          `Complexity: ${s.complexity || 'N/A'}\n` +
          `Required Capabilities: ${s.required_capabilities?.join(', ') || 'N/A'}\n` +
          `Priority: ${s.priority || 'N/A'}`
      }).join('\n')

      newLogs.push({
        id: 'decomp-result',
        timestamp: time,
        stage: 1,
        type: 'stage_complete',
        message: `✓ Decomposed into ${subtasks.length} subtasks`,
        fullResponse: decompositionDetails || 'No subtasks generated'
      })
      current = 2
    }

    // Stage 2: Agent Selection - only if agents are selected
    const assignmentEntries = Object.entries(agentSelectionMeta?.summary?.assignments || agentSelectionMeta?.assignments || {})
      .filter(([key]) => key.startsWith('subtask_'))
    const hasAgents = assignmentEntries.length > 0 || subtasks.some((s: any) => s.selected_agent)
    if (hasAgents) {
      time = new Date(time.getTime() + 500)
      newLogs.push({ id: 'select-start', timestamp: time, stage: 2, type: 'stage_start', message: `⚡ STAGE 2: AGENT SELECTION` })
      assignmentEntries.forEach(([key, agents]) => {
        const subtaskIndex = parseInt(key.replace('subtask_', ''), 10)
        const subtaskDescription = subtasks[subtaskIndex]?.description || `Subtask ${subtaskIndex + 1}`
          ; (agents as any[]).forEach((agent, idx) => {
            time = new Date(time.getTime() + 200)
            newLogs.push({
              id: `agent-${key}-${idx}`,
              timestamp: time,
              stage: 2,
              type: 'agent_spawn',
              message: `🤖 Selected: ${agent.agent_name}`,
              agent: agent.agent_name,
              taskDescription: subtaskDescription,
              fullResponse:
                `Match Score: ${(agent.match_score * 100).toFixed(1)}%\n` +
                `Reasoning: ${agent.reasoning || 'N/A'}`
            })
          })
      })
      time = new Date(time.getTime() + 300)
      newLogs.push({ id: 'select-complete', timestamp: time, stage: 2, type: 'stage_complete', message: `✓ STAGE 2: AGENT SELECTION COMPLETE` })
      current = 3
    }

    // Stage 3: Context Engineering - detect from metadata and actual retrieved context
    const totalContextTasks = contextMeta?.summary?.total_subtasks ?? subtasks.length
    const subtasksWithContext = contextMeta?.summary?.subtasks_with_context ?? 0
    const shouldShowContextStage = !!contextMeta || subtasks.length > 0

    if (shouldShowContextStage) {
      time = new Date(time.getTime() + 300)
      const enhancements = contextMeta?.enhancements || {}
      const contextDetails = subtasks.map((s: any, i: number) => {
        const enhancement = enhancements[`subtask_${i}`] || enhancements[i]
        const quality = enhancement?.context_quality ?? s?.context_quality ?? 0
        const sources = enhancement?.num_sources ?? (s?.retrieved_docs?.length || 0) + (s?.memory_items?.length || 0)
        return `Subtask ${i + 1}:\n` +
          `  Context Quality: ${(quality * 100).toFixed(2)}%\n` +
          `  Sources Retrieved: ${sources}`
      }).join('\n')

      newLogs.push({
        id: 'context-start',
        timestamp: time,
        stage: 3,
        type: 'stage_start',
        message: `⚡ STAGE 3: CONTEXT ENGINEERING`,
        fullResponse: contextDetails || 'Context preparation in progress...'
      })
      time = new Date(time.getTime() + 300)
      newLogs.push({
        id: 'context-complete',
        timestamp: time,
        stage: 3,
        type: 'stage_complete',
        message: subtasksWithContext > 0
          ? `✓ STAGE 3: CONTEXT ENGINEERING COMPLETE`
          : `⚠️ STAGE 3: CONTEXT ENGINEERING - No context retrieved`
      })
      current = 4
    }

    // Stage 4: Agent Execution - only if execution has started
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
          id: `exec-${key}`,
          timestamp: time,
          stage: 4,
          type: result.status === 'completed' ? 'task_complete' : 'task_error',
          message: result.status === 'completed'
            ? `✓ ${subtaskDescription.slice(0, 80)}`
            : `✗ ${subtaskDescription.slice(0, 80)} failed: ${result.error || 'Unknown error'}`,
          agent: result.agent_name,
          tokens: result.tokens_used,
          duration: result.execution_time_ms,
          taskDescription: subtaskDescription
        })
      })
      // Only mark stage 4 complete if all subtasks are done
      if (allTasksDone && subtasks.length > 0) {
        time = new Date(time.getTime() + 500)
        newLogs.push({ id: 'exec-complete', timestamp: time, stage: 4, type: 'stage_complete', message: `✓ STAGE 4: AGENT EXECUTION COMPLETE` })
        current = 5
      } else {
        current = 4 // Still executing
      }
    }

    // Stage 5: Result Aggregation
    if (resultAggregationMeta || output.aggregated_result || output.final_report) {
      time = new Date(time.getTime() + 500)
      newLogs.push({
        id: 'aggregate',
        timestamp: time,
        stage: 5,
        type: 'stage_complete',
        message: `⚡ STAGE 5: RESULT AGGREGATION`,
        fullResponse: output.aggregated_result || output.final_report || 'Results aggregated'
      })
      current = 6
    }

    // Stage 6: Learning Update
    const hasLearningUpdate = !!(learningSystemMeta?.updates && Object.keys(learningSystemMeta.updates).length > 0)
    if (hasLearningUpdate) {
      time = new Date(time.getTime() + 300)
      newLogs.push({
        id: 'learning',
        timestamp: time,
        stage: 6,
        type: 'stage_complete',
        message: `⚡ STAGE 6: LEARNING UPDATE`,
        fullResponse: JSON.stringify(learningSystemMeta.updates, null, 2)
      })
      current = 7
    }

    // Stage 7: Quality Assessment
    const qualityScores = resultAggregationMeta?.quality_scores || output.quality_scores
    if (qualityScores) {
      time = new Date(time.getTime() + 300)
      newLogs.push({
        id: 'quality',
        timestamp: time,
        stage: 7,
        type: 'stage_complete',
        message: `⚡ STAGE 7: QUALITY ASSESSMENT - Overall ${(qualityScores.overall * 100).toFixed(1)}%`
      })
      current = 8
    }

    // Stage 8: Memory Storage
    if (memoryStorageMeta || memoryIntegrationSummary) {
      time = new Date(time.getTime() + 300)
      newLogs.push({
        id: 'memory',
        timestamp: time,
        stage: 8,
        type: 'memory_write',
        message: `⚡ STAGE 8: MEMORY STORAGE`,
        fullResponse: JSON.stringify(memoryStorageMeta || memoryIntegrationSummary, null, 2)
      })
      current = 9
    }

    // Stage 9: Final Response
    if (execution.status === 'completed') {
      time = new Date(time.getTime() + 500)
      const finalReport = execution.output_data?.final_report ||
        execution.output_data?.aggregated_result ||
        subtasks[subtasks.length - 1]?.execution_result?.llm_response ||
        'Workflow completed'

      newLogs.push({
        id: 'complete',
        timestamp: time,
        stage: 9,
        type: 'stage_complete',
        message: `✅ STAGE 9: RESPONSE GENERATION COMPLETE`,
        fullResponse: finalReport
      })
      current = 9
    }

    const aggregationExists = !!(resultAggregationMeta || output.aggregated_result || output.final_report)
    const qualityScoreAvailable = !!qualityScores
    const memoryStoredExists = !!(memoryStorageMeta || memoryIntegrationSummary)

    // Stage 3 completion: only if context was actually retrieved
    const hasContextRetrieved = subtasksWithContext > 0 ||
      subtasks.some((s: any) => (s.context_enhancement?.num_sources || 0) > 0 ||
        (s.retrieved_docs?.length || 0) > 0 ||
        (s.memory_items?.length || 0) > 0)

    const stageCompletionFlags: Record<number, boolean> = {
      1: subtasks.length > 0,
      2: hasAgents,
      3: hasContextRetrieved || currentStage > 3, // Complete if context found OR we moved past it
      4: hasExecutionResults && allTasksDone,
      5: aggregationExists,
      6: hasLearningUpdate || learningUpdates.length > 0,
      7: qualityScoreAvailable,
      8: memoryStoredExists,
      9: execution.status === 'completed'
    }

    let highestComplete = 0
    for (let stage = 1; stage <= 9; stage++) {
      if (stageCompletionFlags[stage] && highestComplete === stage - 1) {
        highestComplete = stage
      } else {
        break
      }
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
  }, [])

  const loadExecutionById = useCallback(async (executionId: number) => {
    try {
      const executionDetails = await apiClient.getWorkflowExecution(executionId.toString())
      setExecutionData((prev: any) => ({ ...prev, execution: executionDetails }))
      // generateLogsFromExecution will set currentStage and completedStages based on actual execution state
      generateLogsFromExecution(executionDetails)
    } catch (err) { console.error('Error loading execution:', err) }
  }, [generateLogsFromExecution])

  const { isConnected: sseConnected } = useWorkflowStream({
    executionId: currentExecutionId,
    onSubtaskUpdate: useCallback((update) => {
      const newLog: LogEntry = {
        id: `sse-${Date.now()}`,
        timestamp: new Date(),
        stage: 4,
        type: update.status === 'completed' ? 'task_complete' : 'task_progress',
        message: `${update.status === 'completed' ? '✓' : '⏳'} ${update.subtask_description || 'Processing...'}`,
        taskDescription: update.subtask_description,
        agent: update.agent_name,
        model: update.model,
        tokens: update.tokens_used,
        duration: update.execution_time_ms,
        // Capture FULL response from SSE
        fullResponse: update.llm_response || update.response || update.output
      }
      setLogs(prev => [...prev, newLog])
      if (currentExecutionId) loadExecutionById(currentExecutionId)
    }, [currentExecutionId, loadExecutionById]),
    onWorkflowComplete: useCallback(() => {
      setIsExecuting(false)
      if (currentExecutionId) loadExecutionById(currentExecutionId) // Will update stages from execution data
    }, [currentExecutionId, loadExecutionById]),
    onLog: useCallback((log) => {
      // Add real-time logs to the display
      const newLog: LogEntry = {
        id: `log-${Date.now()}`,
        timestamp: new Date(),
        stage: log.stage || 4,
        type: 'info',
        message: log.message,
        details: log.details
      }
      setLogs(prev => [...prev, newLog])
    }, []),
    onError: useCallback((error) => {
      const errorLog: LogEntry = {
        id: `error-${Date.now()}`,
        timestamp: new Date(),
        stage: 4,
        type: 'task_error',
        message: `❌ Error: ${error.message || error}`,
        details: error.stack || JSON.stringify(error)
      }
      setLogs(prev => [...prev, errorLog])
    }, [])
  })

  const handleWebSocketMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'execution_started':
        if (message.data?.execution_id) {
          setCurrentExecutionId(message.data.execution_id)
          setIsExecuting(true)
          setCurrentStage(1)
          setCompletedStages([])
          setLogs([{ id: 'start', timestamp: new Date(), stage: 1, type: 'stage_start', message: '🚀 Starting workflow execution...' }])
          loadExecutionById(message.data.execution_id)
        }
        break
      case 'subtask_execution_update':
        if (currentExecutionId) loadExecutionById(currentExecutionId)
        break
      case 'execution_completed':
      case 'execution_failed':
        setIsExecuting(false)
        if (message.data?.execution_id) loadExecutionById(message.data.execution_id)
        break
    }
  }, [currentExecutionId, loadExecutionById])

  const { isConnected } = useWorkflowWebSocket({ workflowId, executionId: currentExecutionId || undefined, onMessage: handleWebSocketMessage, autoConnect: true })

  useEffect(() => {
    const loadWorkflow = async () => {
      try {
        const workflow = await apiClient.getWorkflow(workflowId.toString())
        const execResponse = await apiClient.getWorkflowExecutions(workflowId.toString())
        const executions = execResponse?.items || execResponse || []
        let executionDetails = null
        if (executions.length > 0) {
          const latest = executions.sort((a: any, b: any) => b.id - a.id)[0]
          executionDetails = await apiClient.getWorkflowExecution(latest.id.toString())
          setCurrentExecutionId(latest.id)
          if (executionDetails.status === 'running') setIsExecuting(true)
          // generateLogsFromExecution will set currentStage and completedStages based on actual state
          generateLogsFromExecution(executionDetails)
        }
        setExecutionData({ ...workflow, execution: executionDetails })
      } catch (error) { console.error('Error loading workflow:', error) }
    }
    loadWorkflow()
  }, [workflowId, generateLogsFromExecution])

  useEffect(() => { if (autoStart && !hasAutoStarted && executionData) { setHasAutoStarted(true); setTimeout(handleStartExecution, 1000) } }, [autoStart, hasAutoStarted, executionData])

  const handleStartExecution = async () => {
    try {
      // Reset everything for new execution
      lastExecutionIdRef.current = null
      setIsExecuting(true)
      setCurrentStage(1)
      setCompletedStages([])
      setLogs([{
        id: 'start',
        timestamp: new Date(),
        stage: 1,
        type: 'stage_start',
        message: '🚀 Starting workflow execution...'
      }])
      const result = await apiClient.executeWorkflow(workflowId.toString(), {})
      if (result?.execution_id || result?.id) {
        const execId = result.execution_id || result.id
        lastExecutionIdRef.current = execId
        setCurrentExecutionId(execId)
        // Wait a bit before loading to ensure backend has started processing
        setTimeout(() => loadExecutionById(execId), 500)
      }
    } catch (error) {
      console.error('Error starting execution:', error)
      setIsExecuting(false)
    }
  }

  const handleStageClick = (stage: number) => { if (completedStages.includes(stage) || stage === currentStage) setSelectedStageFilter(selectedStageFilter === stage ? null : stage) }

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

  return (
    <div className={cn("h-screen flex flex-col theater-bg", isFullscreen && "fixed inset-0 z-50")}>
      {/* Header */}
      <div className="glass-panel-light border-b border-white/5">
        <div className="flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" onClick={onBack} className="gap-2"><ArrowLeft className="w-4 h-4" />Back</Button>
            <div>
              <h1 className="text-lg font-semibold"><span className="gradient-text">{executionData.name}</span></h1>
              <p className="text-xs text-muted-foreground line-clamp-1">{executionData.description}</p>
            </div>
            <Badge variant="outline" className={cn("text-xs", (isConnected || sseConnected) ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" : "bg-gray-500/10 text-gray-400 border-gray-500/30")}>
              <div className={cn("w-2 h-2 rounded-full mr-2", (isConnected || sseConnected) ? "bg-emerald-400 animate-pulse" : "bg-gray-400")} />
              {(isConnected || sseConnected) ? 'Live' : 'Offline'}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            {!isExecuting ? (
              <Button onClick={handleStartExecution} className="gap-2 glow-orange"><Play className="w-4 h-4" />Execute</Button>
            ) : (
              <>
                <Button variant="outline" size="sm" onClick={() => setIsExecuting(false)}><Pause className="w-4 h-4 mr-2" />Pause</Button>
                <Button variant="outline" size="sm" className="text-red-400 hover:text-red-300" onClick={() => setIsExecuting(false)}><Square className="w-4 h-4 mr-2" />Stop</Button>
              </>
            )}
            <Button variant="ghost" size="icon" onClick={() => setIsFullscreen(!isFullscreen)}>
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Stage Pipeline */}
      <div className="px-6 py-4">
        <StagePipeline currentStage={currentStage} completedStages={completedStages} onStageClick={handleStageClick} />
      </div>

      <div className="px-6 pb-4">
        <StageDetailPanel
          stage={selectedStageFilter || currentStage}
          subtasks={subtasksForInsights}
          execution={execution}
          collapsed={isStagePanelCollapsed}
          onToggle={() => setIsStagePanelCollapsed(prev => !prev)}
        />
      </div>

      {/* Main Content: Log (75%) + Sidebar (25%) */}
      <div className="flex-1 flex gap-4 px-6 pb-6 min-h-0">
        <div className="flex-1">
          <StreamingLog logs={logs} selectedStage={selectedStageFilter} onClearFilter={() => setSelectedStageFilter(null)} />
        </div>
        <div className="w-[300px]">
          <MetricsSidebar currentStage={currentStage} executionData={executionData} isExecuting={isExecuting} />
        </div>
      </div>
    </div>
  )
}

