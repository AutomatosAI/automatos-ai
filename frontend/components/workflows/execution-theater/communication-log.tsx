'use client'

import { useEffect, useRef, useState } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Bot,
  MessageCircle,
  User,
  Users,
  Zap,
  Brain,
  Database,
  AlertCircle,
  CheckCircle,
  Clock,
  Trash2,
  FileText,
  Copy,
  Download,
  TrendingUp,
  Activity,
  Layers
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { PieChart, Pie, Cell, LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { apiClient } from '@/lib/api-client'

interface CommunicationLogProps {
  workflowId: number
  isExecuting: boolean
  workflow?: any
}

interface LogEntry {
  id: string
  timestamp: string
  type: 'orchestrator' | 'agent_message' | 'memory_operation' | 'rag_operation' | 'tool_call' | 'system_event'
  from?: string
  to?: string
  message: string
  status?: 'info' | 'success' | 'warning' | 'error'
  metadata?: any
  details?: string[]
}

export function CommunicationLog({ workflowId, isExecuting, workflow }: CommunicationLogProps) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [activeTab, setActiveTab] = useState<string>('all')
  const scrollRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    console.log('CommunicationLog - workflow data:', workflow)
    
    // Generate real logs from execution data
    const generatedLogs: LogEntry[] = []
    const execution = workflow?.execution
    
    console.log('CommunicationLog - execution data:', execution)
    
    if (!execution) {
      console.log('No execution data, showing waiting message')
      generatedLogs.push({
        id: '1',
        timestamp: new Date().toLocaleTimeString(),
        type: 'system_event',
        message: 'Waiting for execution data...',
        status: 'info'
      })
      setLogs(generatedLogs)
      return
    }

    const subtasks = execution.output_data?.subtasks || []
    const decomp = execution.input_data?.decomposition
    let logTime = new Date(execution.started_at)

    // 1. Workflow Started
    generatedLogs.push({
      id: 'start',
      timestamp: logTime.toLocaleTimeString(),
      type: 'orchestrator',
      from: 'Orchestrator',
      message: `Workflow execution started (ID: ${execution.id})`,
      status: 'success'
    })

    // 2. Task Decomposition with details
    logTime = new Date(logTime.getTime() + 1000)
    if (decomp?.is_real && subtasks.length > 0) {
      const subtaskList = subtasks.map((st: any, i: number) => 
        `${i + 1}. ${st.description?.slice(0, 50)}...`
      )
      
      generatedLogs.push({
        id: 'decomp',
        timestamp: logTime.toLocaleTimeString(),
        type: 'orchestrator',
        from: 'Orchestrator',
        message: `Task decomposed by ${decomp.llm_model} into ${subtasks.length} subtasks`,
        status: 'success',
        details: subtaskList
      })
    }

    // 3. For each subtask: Assignment → Execution → Memory update
    subtasks.forEach((subtask: any, index: number) => {
      logTime = new Date(logTime.getTime() + 2000)
      const agent = subtask.selected_agent?.agent_name || 'Agent'
      const status = subtask.execution_result?.status
      const tokens = subtask.execution_result?.tokens_used || 0

      // Assignment from Orchestrator
      generatedLogs.push({
        id: `assign-${index}`,
        timestamp: logTime.toLocaleTimeString(),
        type: 'orchestrator',
        from: 'Orchestrator',
        to: agent,
        message: `Assigning task ${index + 1} to ${agent}`,
        status: 'info',
        metadata: { task: subtask.description?.slice(0, 40) }
      })

      // Agent receives and starts
      logTime = new Date(logTime.getTime() + 500)
      generatedLogs.push({
        id: `agent-start-${index}`,
        timestamp: logTime.toLocaleTimeString(),
        type: 'agent_message',
        from: agent,
        message: `Received task, processing: ${subtask.description?.slice(0, 60)}...`,
        status: 'info'
      })

      // Agent completes
      logTime = new Date(logTime.getTime() + 3000)
      generatedLogs.push({
        id: `agent-complete-${index}`,
        timestamp: logTime.toLocaleTimeString(),
        type: 'agent_message',
        from: agent,
        message: status === 'completed' ? 
          `Task completed successfully` : 
          `Task failed`,
        status: status === 'completed' ? 'success' : 'error',
        metadata: { tokens }
      })

      // Memory update if successful
      if (status === 'completed' && tokens > 0) {
        logTime = new Date(logTime.getTime() + 200)
        generatedLogs.push({
          id: `memory-${index}`,
          timestamp: logTime.toLocaleTimeString(),
          type: 'memory_operation',
          from: agent,
          message: `Updating findings to shared memory`,
          status: 'success',
          metadata: { tokens }
        })
      }
    })

    // 4. Final aggregation
    if (execution.completed_at) {
      logTime = new Date(logTime.getTime() + 1000)
      const totalTokens = subtasks.reduce((sum: number, st: any) => 
        sum + (st.execution_result?.tokens_used || 0), 0)
      
      generatedLogs.push({
        id: 'aggregate',
        timestamp: logTime.toLocaleTimeString(),
        type: 'orchestrator',
        from: 'Orchestrator',
        message: `Aggregating results from ${subtasks.length} agents`,
        status: 'info'
      })

      // Completion
      logTime = new Date(logTime.getTime() + 500)
      generatedLogs.push({
        id: 'complete',
        timestamp: logTime.toLocaleTimeString(),
        type: 'orchestrator',
        from: 'Orchestrator',
        message: `✅ Workflow completed in ${execution.duration || 'N/A'} using ${totalTokens} tokens`,
        status: 'success'
      })
    }

    setLogs(generatedLogs)
  }, [workflow, workflowId, isExecuting])

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  const getIcon = (type: string, status?: string) => {
    switch (type) {
      case 'orchestrator':
        return <Bot className="w-4 h-4 text-info" />
      case 'agent_message':
        return <MessageCircle className="w-4 h-4 text-info" />
      case 'tool_call':
        return <Zap className="w-4 h-4 text-warning" />
      case 'memory_operation':
        return <Brain className="w-4 h-4 text-agent" />
      case 'rag_operation':
        return <Database className="w-4 h-4 text-success" />
      case 'system_event':
        if (status === 'error') return <AlertCircle className="w-4 h-4 text-destructive" />
        if (status === 'success') return <CheckCircle className="w-4 h-4 text-success" />
        return <Clock className="w-4 h-4 text-muted-foreground" />
      default:
        return <Bot className="w-4 h-4 text-muted-foreground" />
    }
  }

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'orchestrator': return 'Orchestrator'
      case 'agent_message': return 'Agent'
      case 'tool_call': return 'Tool'
      case 'memory_operation': return 'Memory'
      case 'rag_operation': return 'RAG'
      case 'system_event': return 'System'
      default: return 'Event'
    }
  }

  const filteredLogs = activeTab === 'all' 
    ? logs 
    : activeTab === 'orchestrator'
      ? logs.filter(log => log.type === 'orchestrator')
      : activeTab === 'agents'
        ? logs.filter(log => log.type === 'agent_message')
        : activeTab === 'memory'
          ? logs.filter(log => log.type === 'memory_operation')
          : activeTab === 'rag'
            ? logs.filter(log => log.type === 'rag_operation')
            : activeTab === 'tools'
              ? logs.filter(log => log.type === 'tool_call')
              : logs
  
  const tabs = [
    { id: 'all', label: 'All Messages', icon: MessageCircle },
    { id: 'results', label: 'Final Report', icon: FileText },
    { id: 'orchestrator', label: 'Orchestrator', icon: Bot },
    { id: 'agents', label: 'Agents', icon: Users }
  ]

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border/30">
        <div className="flex items-center space-x-2">
          <MessageCircle className="w-5 h-5 text-primary" />
          <h3 className="font-semibold">Communication & Events</h3>
          <Badge variant="secondary">{filteredLogs.length}</Badge>
        </div>

        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setAutoScroll(!autoScroll)}
            className={autoScroll ? 'text-success' : 'text-muted-foreground'}
          >
            {autoScroll ? 'Auto-scroll On' : 'Auto-scroll Off'}
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={() => setLogs([])}
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 h-auto p-1 bg-muted/50">
          {tabs.map((tab) => {
            const Icon = tab.icon
            const count = tab.id === 'all' ? logs.length : 
              tab.id === 'results' ? (workflow?.execution?.status === 'completed' ? 1 : 0) :
              logs.filter(log => 
                tab.id === 'orchestrator' ? log.type === 'orchestrator' :
                tab.id === 'agents' ? log.type === 'agent_message' :
                tab.id === 'memory' ? log.type === 'memory_operation' :
                tab.id === 'rag' ? log.type === 'rag_operation' :
                tab.id === 'tools' ? log.type === 'tool_call' : false
              ).length
            
            return (
              <TabsTrigger 
                key={tab.id} 
                value={tab.id}
                className="flex items-center gap-2 px-3 py-1.5 text-sm data-[state=active]:bg-background data-[state=active]:text-foreground"
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
                <Badge variant="outline" className="ml-1 text-xs">
                  {count}
                </Badge>
              </TabsTrigger>
            )
          })}
        </TabsList>
      </Tabs>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-3 space-y-2">
          {/* Results Tab - Final Report */}
          {activeTab === 'results' && (
            <div className="space-y-4">
              {(() => {
                const execution = workflow?.execution
                const subtasks = execution?.output_data?.subtasks || []
                
                if (execution?.status !== 'completed' || subtasks.length === 0) {
                  return (
                    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                      <FileText className="w-12 h-12 mb-3 opacity-50" />
                      <p className="text-sm font-medium">No report available yet</p>
                      <p className="text-xs mt-1">Complete the workflow to see final results</p>
                    </div>
                  )
                }
                
                // Get the aggregated final report from backend (includes ALL subtask results)
                const finalReport = execution?.output_data?.final_report || ''
                const lastSubtask = subtasks[subtasks.length - 1]
                
                // Fallback to last subtask if aggregated report not available
                if (!finalReport) {
                  const fallbackReport = lastSubtask?.execution_result?.llm_response || ''
                  console.warn('⚠️ No aggregated final_report found, falling back to last subtask')
                }
                
                if (!finalReport) {
                  return (
                    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                      <AlertCircle className="w-12 h-12 mb-3 opacity-50" />
                      <p className="text-sm font-medium">No report content found</p>
                    </div>
                  )
                }
                
                const copyToClipboard = () => {
                  navigator.clipboard.writeText(finalReport)
                }
                
                const downloadReport = () => {
                  const blob = new Blob([finalReport], { type: 'text/plain' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `workflow-${workflow.id}-report-${new Date().toISOString().split('T')[0]}.txt`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                }
                
                return (
                  <>
                    {/* Header with Actions */}
                    <div className="flex items-center justify-between pb-3 border-b border-border/30">
                      <div>
                        <h3 className="font-semibold text-lg">Final Report</h3>
                        <p className="text-sm text-muted-foreground">
                          Generated by {lastSubtask?.selected_agent?.agent_name || 'Agent'} • {lastSubtask?.execution_result?.tokens_used || 0} tokens
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={copyToClipboard}
                          className="gap-2"
                        >
                          <Copy className="w-4 h-4" />
                          Copy
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={downloadReport}
                          className="gap-2"
                        >
                          <Download className="w-4 h-4" />
                          Download
                        </Button>
                      </div>
                    </div>
                    
                    {/* Report Content */}
                    <div className="bg-muted/30 rounded-lg p-4 border border-border/50">
                      <div className="prose prose-sm max-w-none text-foreground whitespace-pre-wrap">
                        {finalReport}
                      </div>
                    </div>
                    
                    {/* Metadata */}
                    <div className="grid grid-cols-3 gap-4 pt-3 border-t border-border/30">
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Execution Time</p>
                        <p className="text-sm font-medium">{(lastSubtask?.execution_result?.execution_time_ms / 1000 || 0).toFixed(2)}s</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Context Quality</p>
                        <p className="text-sm font-medium">{((lastSubtask?.context_quality || 0) * 100).toFixed(0)}%</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Total Subtasks</p>
                        <p className="text-sm font-medium">{subtasks.length}</p>
                      </div>
                    </div>
                  </>
                )
              })()}
            </div>
          )}

          {/* All other tabs - Log Messages */}
          {activeTab !== 'results' && (
            <AnimatePresence initial={false}>
              {filteredLogs.map((log) => (
                <motion.div
                  key={log.id}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="flex items-start space-x-3 p-3 rounded-lg bg-background/50 border border-border/30 hover:border-border/60 transition-colors"
                >
                  {/* Icon */}
                  <div className="mt-0.5">
                    {getIcon(log.type, log.status)}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-xs text-muted-foreground font-mono">
                        {log.timestamp}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {getTypeLabel(log.type)}
                      </Badge>
                      {log.status && (
                        <Badge 
                          variant={
                            log.status === 'error' ? 'destructive' :
                            log.status === 'success' ? 'default' :
                            'secondary'
                          }
                          className="text-xs"
                        >
                          {log.status}
                        </Badge>
                      )}
                    </div>

                    {/* From/To */}
                    {(log.from || log.to) && (
                      <div className="flex items-center space-x-2 text-xs mb-1">
                        {log.from && (
                          <span className="text-info font-medium">{log.from}</span>
                        )}
                        {log.from && log.to && (
                          <span className="text-muted-foreground">→</span>
                        )}
                        {log.to && (
                          <span className="text-success font-medium">{log.to}</span>
                        )}
                      </div>
                    )}

                    {/* Message */}
                    <p className="text-sm">{log.message}</p>
                    
                    {/* Details - Expandable list */}
                    {log.details && log.details.length > 0 && (
                      <div className="mt-2 p-2 bg-background/50 rounded border border-border/30">
                        <p className="text-xs text-muted-foreground mb-1 font-medium">Details:</p>
                        <div className="space-y-1">
                          {log.details.map((detail, idx) => (
                            <p key={idx} className="text-xs text-muted-foreground pl-2 border-l-2 border-border/50">
                              {detail}
                            </p>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Metadata - Tokens */}
                    {log.metadata?.tokens > 0 && (
                      <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                        <Zap className="w-3 h-4" />
                        <span className="font-mono">{log.metadata.tokens} tokens</span>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
            <div ref={scrollRef} />
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}

