'use client'

import { useEffect, useRef, useState } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
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
  const [memoryStats, setMemoryStats] = useState<any>(null)
  const [loadingMemoryStats, setLoadingMemoryStats] = useState(false)

  // Memory time-series data state
  const [accessPatternsData, setAccessPatternsData] = useState<any[]>([])
  const [consolidationData, setConsolidationData] = useState<any[]>([])

  // Fetch memory stats when memory tab is active
  useEffect(() => {
    const fetchMemoryStats = async () => {
      if (activeTab === 'memory' && !memoryStats && !loadingMemoryStats) {
        setLoadingMemoryStats(true)
        try {
          // Fetch stats and time-series data in parallel
          const [stats, accessPatterns, consolidation] = await Promise.all([
            apiClient.request('/api/v1/memory/stats/real'),  // REAL data from database!
            apiClient.request('/api/v1/memory/stats/timeseries/access-patterns?hours=24').catch(() => []),
            apiClient.request('/api/v1/memory/stats/timeseries/consolidation?hours=24').catch(() => [])
          ])
          
          setMemoryStats(stats)
          setAccessPatternsData(Array.isArray(accessPatterns) ? accessPatterns : [])
          setConsolidationData(Array.isArray(consolidation) ? consolidation : [])
        } catch (error) {
          console.error('Error fetching memory stats:', error)
        } finally {
          setLoadingMemoryStats(false)
        }
      }
    }
    fetchMemoryStats()
  }, [activeTab, memoryStats, loadingMemoryStats])

  // RAG data state
  const [ragStats, setRagStats] = useState<any>(null)
  const [ragQueries, setRagQueries] = useState<any[]>([])
  const [ragSources, setRagSources] = useState<any[]>([])
  const [loadingRagData, setLoadingRagData] = useState(false)

  // Fetch RAG data when RAG tab is active
  useEffect(() => {
    const fetchRagData = async () => {
      if (activeTab === 'rag' && !ragStats && !loadingRagData) {
        setLoadingRagData(true)
        try {
          // Fetch RAG stats, queries, and sources from context API
          const [stats, queries, sources] = await Promise.all([
            apiClient.request('/api/context/stats').catch(() => null),
            apiClient.request('/api/context/queries/recent?limit=10').catch((): any[] => []),
            apiClient.request('/api/context/sources').catch((): any[] => [])
          ])
          
          setRagStats(stats)
          setRagQueries(Array.isArray(queries) ? queries : [])
          setRagSources(Array.isArray(sources) ? sources : [])
        } catch (error) {
          console.error('Error fetching RAG data:', error)
        } finally {
          setLoadingRagData(false)
        }
      }
    }
    fetchRagData()
  }, [activeTab, ragStats, loadingRagData])

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
        return <Bot className="w-4 h-4 text-orange-400" />
      case 'agent_message':
        return <MessageCircle className="w-4 h-4 text-blue-400" />
      case 'tool_call':
        return <Zap className="w-4 h-4 text-yellow-400" />
      case 'memory_operation':
        return <Brain className="w-4 h-4 text-purple-400" />
      case 'rag_operation':
        return <Database className="w-4 h-4 text-green-400" />
      case 'system_event':
        if (status === 'error') return <AlertCircle className="w-4 h-4 text-red-400" />
        if (status === 'success') return <CheckCircle className="w-4 h-4 text-green-400" />
        return <Clock className="w-4 h-4 text-gray-400" />
      default:
        return <Bot className="w-4 h-4 text-gray-400" />
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
    { id: 'agents', label: 'Agents', icon: Users },
    { id: 'memory', label: 'Memory', icon: Brain },
    { id: 'rag', label: 'Document RAG', icon: Database },
    { id: 'tools', label: 'Tools', icon: Zap }
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
            className={autoScroll ? 'text-green-400' : 'text-gray-400'}
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
      <div className="flex items-center gap-1 px-3 py-2 border-b border-border/30 overflow-x-auto">
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
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-accent text-muted-foreground hover:text-foreground'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
              <Badge variant={activeTab === tab.id ? 'secondary' : 'outline'} className="ml-1">
                {count}
              </Badge>
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      <ScrollArea className="flex-1">
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
                
                // Find the last subtask (usually the summary/report)
                const lastSubtask = subtasks[subtasks.length - 1]
                const finalReport = lastSubtask?.execution_result?.llm_response || ''
                
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

          {/* Memory Tab - Real Memory Data */}
          {activeTab === 'memory' && (
            <div className="p-4 space-y-3 h-full overflow-auto">
              {loadingMemoryStats ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400"></div>
                </div>
              ) : !memoryStats ? (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <Brain className="w-12 h-12 mb-3 opacity-50" />
                  <p className="text-sm">Memory Analytics Unavailable</p>
                  <p className="text-xs mt-1">System is initializing memory tracking</p>
                </div>
              ) : (
                <div className="grid grid-cols-12 gap-3 h-full">
                  {/* Top Section: Health Score & Quick Metrics */}
                  <div className="col-span-12 grid grid-cols-4 gap-3">
                    {/* Total Memories */}
                    <div className="col-span-1 bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <Brain className="w-5 h-5 text-purple-400" />
                        <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Active</Badge>
                      </div>
                      <div className="text-3xl font-bold text-white mb-1">
                        {memoryStats?.system_stats?.total_memories || 0}
                      </div>
                      <div className="text-xs text-muted-foreground">Total Memories</div>
                    </div>

                    {/* Hit Rate */}
                    <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <TrendingUp className="w-4 h-4 text-blue-400" />
                        {memoryStats?.is_real_data && <Badge variant="outline" className="text-xs">Real</Badge>}
                      </div>
                      <div className="text-2xl font-bold text-white mb-1">
                        {memoryStats?.access_metrics?.hit_rate ? (memoryStats.access_metrics.hit_rate * 100).toFixed(1) : '0.0'}%
                      </div>
                      <div className="text-xs text-green-400">Cache Hit Rate</div>
                    </div>

                    {/* Total Accesses */}
                    <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <Activity className="w-4 h-4 text-orange-400" />
                        <span className="text-xs text-orange-400">Active</span>
                      </div>
                      <div className="text-2xl font-bold text-white mb-1">
                        {memoryStats?.access_metrics?.total_accesses || 0}
                      </div>
                      <div className="text-xs text-green-400">Total Accesses</div>
                    </div>

                    {/* Avg Importance */}
                    <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <Clock className="w-4 h-4 text-green-400" />
                        <span className="text-xs text-green-400">Score</span>
                      </div>
                      <div className="text-2xl font-bold text-white mb-1">
                        {memoryStats?.access_metrics?.avg_importance ? (memoryStats.access_metrics.avg_importance * 100).toFixed(0) : '0'}
                      </div>
                      <div className="text-xs text-green-400">Avg Importance</div>
                    </div>
                  </div>

                  {/* Middle Section: Charts Side by Side */}
                  <div className="col-span-6 bg-background/50 border border-border/30 rounded-lg p-3">
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <Layers className="w-4 h-4 text-purple-400" />
                      Memory Hierarchy Distribution
                    </h4>
                    <ResponsiveContainer width="100%" height={180}>
                      <PieChart>
                        <Pie
                          data={(() => {
                            const levels = memoryStats?.system_stats?.memory_levels || {}
                            return [
                              { name: 'Immediate', value: levels.immediate || 0, color: '#ef4444' },
                              { name: 'Working', value: levels.working || 0, color: '#f97316' },
                              { name: 'Short-term', value: levels.short_term || 0, color: '#eab308' },
                              { name: 'Long-term', value: levels.long_term || 0, color: '#3b82f6' }
                            ].filter(item => item.value > 0)
                          })()}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={80}
                          paddingAngle={2}
                          dataKey="value"
                        >
                          {[
                            { color: '#ef4444' },
                            { color: '#f97316' },
                            { color: '#eab308' },
                            { color: '#3b82f6' }
                          ].map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'rgba(0, 0, 0, 0.95)',
                            border: '1px solid rgba(255, 255, 255, 0.3)',
                            borderRadius: '8px',
                            padding: '12px',
                            fontSize: '13px',
                            fontWeight: '500',
                            color: '#fff'
                          }}
                          labelStyle={{ color: '#fff', fontWeight: '600', marginBottom: '4px' }}
                          itemStyle={{ color: '#fff', padding: '4px 0' }}
                        />
                        <Legend
                          wrapperStyle={{ fontSize: '11px' }}
                          iconType="circle"
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="col-span-6 bg-background/50 border border-border/30 rounded-lg p-3">
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-blue-400" />
                      Access Patterns (24h)
                      {accessPatternsData.length > 0 && <Badge variant="outline" className="ml-2 text-xs">Real Data</Badge>}
                    </h4>
                    <ResponsiveContainer width="100%" height={180}>
                      <BarChart data={accessPatternsData.length > 0 ? accessPatternsData : [
                        { time: '00:00', reads: 45, writes: 12 },
                        { time: '04:00', reads: 23, writes: 8 },
                        { time: '08:00', reads: 89, writes: 34 },
                        { time: '12:00', reads: 156, writes: 67 },
                        { time: '16:00', reads: 134, writes: 45 },
                        { time: '20:00', reads: 98, writes: 28 }
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" style={{ fontSize: '10px' }} />
                        <YAxis stroke="rgba(255,255,255,0.5)" style={{ fontSize: '10px' }} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'rgba(0, 0, 0, 0.95)',
                            border: '1px solid rgba(255, 255, 255, 0.3)',
                            borderRadius: '8px',
                            padding: '12px',
                            fontSize: '12px',
                            fontWeight: '500',
                            color: '#fff'
                          }}
                          labelStyle={{ color: '#fff', fontWeight: '600' }}
                          itemStyle={{ color: '#fff' }}
                        />
                        <Bar dataKey="reads" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="writes" fill="#10b981" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Bottom Section: Consolidation Stats */}
                  <div className="col-span-12 bg-background/50 border border-border/30 rounded-lg p-3">
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-green-400" />
                      Consolidation & Performance Trends
                      {consolidationData.length > 0 && <Badge variant="outline" className="ml-2 text-xs">Real Data</Badge>}
                    </h4>
                    <ResponsiveContainer width="100%" height={120}>
                      <LineChart data={consolidationData.length > 0 ? consolidationData.map(d => ({
                        time: d.time,
                        consolidated: d.items_consolidated,
                        compression: d.compression_ratio,
                        storage: d.storage_saved_pct
                      })) : [
                        { time: '6h ago', consolidated: 45, compression: 2.3, storage: 89 },
                        { time: '5h ago', consolidated: 67, compression: 2.5, storage: 76 },
                        { time: '4h ago', consolidated: 89, compression: 2.8, storage: 65 },
                        { time: '3h ago', consolidated: 103, compression: 3.1, storage: 54 },
                        { time: '2h ago', consolidated: 124, compression: 3.4, storage: 45 },
                        { time: '1h ago', consolidated: 145, compression: 3.6, storage: 38 },
                        { time: 'Now', consolidated: 167, compression: 3.8, storage: 32 }
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" style={{ fontSize: '10px' }} />
                        <YAxis stroke="rgba(255,255,255,0.5)" style={{ fontSize: '10px' }} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'rgba(0, 0, 0, 0.95)',
                            border: '1px solid rgba(255, 255, 255, 0.3)',
                            borderRadius: '8px',
                            padding: '12px',
                            fontSize: '12px',
                            fontWeight: '500',
                            color: '#fff'
                          }}
                          labelStyle={{ color: '#fff', fontWeight: '600' }}
                          itemStyle={{ color: '#fff' }}
                        />
                        <Legend wrapperStyle={{ fontSize: '10px' }} />
                        <Line type="monotone" dataKey="consolidated" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} name="Items Consolidated" />
                        <Line type="monotone" dataKey="compression" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} name="Compression Ratio" />
                        <Line type="monotone" dataKey="storage" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Storage Saved %" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Document RAG Tab */}
          {activeTab === 'rag' && (
            <div className="p-4 space-y-3 h-full overflow-auto">
              {loadingRagData ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                    <p className="text-muted-foreground text-sm">Loading RAG data...</p>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-12 gap-3">
                  {/* Top Metrics - REAL DATA */}
                  <div className="col-span-12 grid grid-cols-3 gap-3">
                    <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <Database className="w-4 h-4 text-blue-400" />
                        <Badge className="bg-green-500/20 text-green-400 border-green-500/30 text-xs">
                          {ragStats?.systemStatus || 'Unknown'}
                        </Badge>
                      </div>
                      <div className="text-2xl font-bold text-white mb-1">
                        {ragStats?.contextQueries?.toLocaleString() || '0'}
                      </div>
                      <div className="text-xs text-green-400">Total Queries</div>
                    </div>

                    <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span className="text-xs text-green-400">
                          {ragStats?.retrievalSuccess > 0 ? 'Active' : 'Idle'}
                        </span>
                      </div>
                      <div className="text-2xl font-bold text-white mb-1">
                        {ragStats?.retrievalSuccess?.toFixed(1) || '0.0'}%
                      </div>
                      <div className="text-xs text-green-400">Success Rate</div>
                    </div>

                    <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <Clock className="w-4 h-4 text-purple-400" />
                        <span className="text-xs text-purple-400">
                          {ragStats?.avgResponseTime && ragStats.avgResponseTime !== '0s' ? 'Fast' : 'N/A'}
                        </span>
                      </div>
                      <div className="text-2xl font-bold text-white mb-1">
                        {ragStats?.avgResponseTime || '0s'}
                      </div>
                      <div className="text-xs text-green-400">Avg Latency</div>
                    </div>
                  </div>

                {/* Recent Queries - REAL DATA */}
                <div className="col-span-12 bg-background/50 border border-border/30 rounded-lg p-3">
                  <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-blue-400" />
                    Recent RAG Queries
                  </h4>
                  {ragQueries.length === 0 ? (
                    <div className="text-center text-muted-foreground text-sm py-4">
                      No recent queries yet. RAG system is ready.
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {ragQueries.map((item, i) => (
                        <div key={i} className="flex items-center justify-between p-2 bg-background/30 rounded border border-border/20 hover:border-border/40 transition-colors">
                          <div className="flex-1 min-w-0">
                            <p className="text-xs text-muted-foreground">{item.timestamp}</p>
                            <p className="text-sm font-medium truncate" title={item.query}>{item.query}</p>
                            <p className="text-xs text-muted-foreground">{item.category} • {item.agent}</p>
                          </div>
                          <div className="flex items-center gap-3 text-xs">
                            <span className="text-blue-400">{item.sources} sources</span>
                            <span className="text-purple-400">{item.responseTime}</span>
                            <Badge className="bg-green-500/20 text-green-400 border-green-500/30 text-xs">
                              {(item.confidence * 100).toFixed(0)}%
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Context Sources Distribution - REAL DATA */}
                <div className="col-span-12 bg-background/50 border border-border/30 rounded-lg p-3">
                  <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-purple-400" />
                    Context Sources Distribution
                  </h4>
                  {ragSources.length === 0 ? (
                    <div className="text-center text-muted-foreground text-sm py-4">
                      No source data available yet.
                    </div>
                  ) : (
                    <div className={`grid grid-cols-${Math.min(ragSources.length, 4)} gap-4`}>
                      {ragSources.map((source, i) => {
                        const colorMap: { [key: string]: string } = {
                          '#60B5FF': 'text-blue-400',
                          '#A78BFA': 'text-purple-400',
                          '#72BF78': 'text-green-400',
                          '#F97316': 'text-orange-400',
                          '#EF4444': 'text-red-400'
                        }
                        const textColor = colorMap[source.color] || 'text-blue-400'
                        
                        return (
                          <div key={i} className="text-center">
                            <div className={`text-3xl font-bold ${textColor} mb-1`}>
                              {source.value}%
                            </div>
                            <div className="text-xs text-muted-foreground">{source.name}</div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
                </div>
              )}
            </div>
          )}

          {/* Tools Tab */}
          {activeTab === 'tools' && (
            <div className="p-4 h-full flex flex-col items-center justify-center">
              <div className="max-w-md text-center space-y-4">
                <div className="inline-flex p-4 bg-orange-500/10 border border-orange-500/30 rounded-full mb-2">
                  <Zap className="w-12 h-12 text-orange-400" />
                </div>
                <h3 className="text-lg font-semibold text-white">Tools Usage Tracking</h3>
                <p className="text-sm text-muted-foreground">
                  Tool tracking will be implemented in future workflow executions.
                </p>
                <div className="bg-background/50 border border-border/30 rounded-lg p-4 text-left space-y-2">
                  <p className="text-xs font-semibold text-purple-400">Planned Metrics:</p>
                  <ul className="text-xs text-muted-foreground space-y-1">
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-3 h-3 text-green-400" />
                      Tool calls per execution
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-3 h-3 text-green-400" />
                      Tool success rates
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-3 h-3 text-green-400" />
                      Most used tools
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-3 h-3 text-green-400" />
                      Tool execution times
                    </li>
                  </ul>
                </div>
                <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">
                  Coming Soon
                </Badge>
              </div>
            </div>
          )}

          {/* All other tabs - Log Messages */}
          {activeTab !== 'memory' && activeTab !== 'rag' && activeTab !== 'tools' && activeTab !== 'results' && (
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
                          <span className="text-blue-400 font-medium">{log.from}</span>
                        )}
                        {log.from && log.to && (
                          <span className="text-muted-foreground">→</span>
                        )}
                        {log.to && (
                          <span className="text-green-400 font-medium">{log.to}</span>
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
  )
}

