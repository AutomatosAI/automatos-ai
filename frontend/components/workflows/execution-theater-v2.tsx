'use client'

import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ArrowLeft,
  Play,
  Pause,
  Square,
  Maximize2,
  Minimize2,
  Settings,
  Activity,
  Brain,
  BarChart3,
  Bot
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
// Removed WorkflowCanvas - was just mock data
import { AgentWorkspaceTabs } from './execution-theater/agent-workspace-tabs'
import { OrchestratorControl } from './execution-theater/orchestrator-control'
import { CommunicationLog } from './execution-theater/communication-log'
import { EnhancedOrchestratorView } from './execution-theater/enhanced-orchestrator-view'
import { apiClient } from '@/lib/api-client'
import { useWorkflowWebSocket } from '@/hooks/use-workflow-websocket'
import { cn } from '@/lib/utils'

interface ExecutionTheaterV2Props {
  workflowId: number
  onBack: () => void
  autoStart?: boolean
}

// Compact Agent List Sidebar Component
function AgentListSidebar({
  workflow,
  selectedAgent,
  onAgentSelect,
  isExecuting
}: {
  workflow: any
  selectedAgent: number | null
  onAgentSelect: (agentId: number) => void
  isExecuting: boolean
}) {
  // Extract real agents from workflow execution subtasks
  const execution = workflow?.execution
  const subtasks = execution?.output_data?.subtasks || []
  
  // Build agent list from subtasks (deduplicate by agent_id)
  const agentMap = new Map()
  subtasks.forEach((subtask: any, index: number) => {
    const agentId = subtask.selected_agent?.agent_id
    const agentName = subtask.selected_agent?.agent_name
    
    if (agentId && !agentMap.has(agentId)) {
      agentMap.set(agentId, {
        id: agentId,
        name: agentName,
        agent_type: subtask.agent_type || 'agent',
        status: execution?.status === 'running' && index === 0 ? 'executing' : 
                subtask.execution_result?.status || 'pending',
        tokensUsed: subtask.execution_result?.tokens_used || 0,
        progress: subtask.execution_result?.status === 'completed' ? 100 : 
                 subtask.execution_result?.status === 'running' ? 50 : 0
      })
    }
  })
  
  const agents = Array.from(agentMap.values())

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'executing': return 'bg-blue-500'
      case 'running': return 'bg-blue-500'
      case 'completed': return 'bg-green-500'
      case 'pending': return 'bg-yellow-500'
      case 'waiting': return 'bg-yellow-500'
      case 'failed': return 'bg-red-500'
      case 'error': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'executing': return 'Running'
      case 'running': return 'Running'
      case 'completed': return 'Done'
      case 'pending': return 'Pending'
      case 'waiting': return 'Waiting'
      case 'failed': return 'Failed'
      case 'error': return 'Error'
      default: return 'Idle'
    }
  }

  if (agents.length === 0) {
    return (
      <div className="text-center text-muted-foreground text-xs py-8">
        <Bot className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p>No agents assigned yet</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {agents.map((agent: any) => {
        const isSelected = selectedAgent === agent.id
        const isActive = agent.status === 'executing' || agent.status === 'running'
        
        return (
          <motion.button
            key={agent.id}
            onClick={() => onAgentSelect(agent.id)}
            className={cn(
              "w-full p-3 rounded-lg border transition-all text-left",
              isSelected
                ? "bg-primary/10 border-primary shadow-sm"
                : "bg-card/50 border-border/50 hover:bg-card hover:border-border",
              isActive && "ring-2 ring-blue-500/30"
            )}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium truncate flex-1">{agent.name}</span>
              <div className={cn(
                "w-2 h-2 rounded-full",
                getStatusColor(agent.status),
                isActive && "animate-pulse"
              )} />
            </div>
            
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span className="truncate">{agent.agent_type}</span>
              <Badge variant="outline" className="text-xs">
                {getStatusLabel(agent.status)}
              </Badge>
            </div>
            
            {agent.progress > 0 && (
              <div className="mt-2 h-1 bg-border rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary transition-all duration-500"
                  style={{ width: `${agent.progress}%` }}
                />
              </div>
            )}
            
            {agent.tokensUsed > 0 && (
              <div className="mt-1 text-xs text-muted-foreground">
                {agent.tokensUsed.toLocaleString()} tokens
              </div>
            )}
          </motion.button>
        )
      })}
    </div>
  )
}

export function ExecutionTheaterV2({ workflowId, onBack, autoStart = false }: ExecutionTheaterV2Props) {
  const [isExecuting, setIsExecuting] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showEnhancedView, setShowEnhancedView] = useState(false)
  const [executionData, setExecutionData] = useState<any>(null)
  const [selectedAgent, setSelectedAgent] = useState<number | null>(null)
  const [hasAutoStarted, setHasAutoStarted] = useState(false)
  const [currentExecutionId, setCurrentExecutionId] = useState<number | null>(null)
  const [activeView, setActiveView] = useState<'orchestrator' | 'analytics'>('orchestrator')

  // WebSocket for real-time updates
  const handleWebSocketMessage = useCallback((message: any) => {
    console.log('🔥 Real-time WebSocket update:', message.type, message)

    switch (message.type) {
      case 'connected':
        console.log('✅ WebSocket connected to Redis channel')
        break
        
      case 'execution_started':
        console.log('🚀 Execution started event received')
        if (message.data?.execution_id) {
          setCurrentExecutionId(message.data.execution_id)
          setIsExecuting(true)
          loadExecutionById(message.data.execution_id)
        }
        break

      case 'subtask_execution_update':
        console.log('📊 SUBTASK UPDATE RECEIVED:', message.data?.subtask_id, message.data?.status)
        const execId = currentExecutionId || message.data?.execution_id
        if (execId) {
          console.log('🔄 Refreshing execution data for ID:', execId)
          loadExecutionById(execId)
        }
        break

      case 'execution_progress':
      case 'subtask_started':
      case 'subtask_completed':
      case 'agent_assigned':
      case 'decomposition_complete':
        console.log('📈 Progress update:', message.type)
        if (currentExecutionId || message.data?.execution_id) {
          loadExecutionById(currentExecutionId || message.data.execution_id)
        }
        break

      case 'execution_completed':
      case 'execution_failed':
        console.log('✅ Execution finished:', message.type)
        setIsExecuting(false)
        if (message.data?.execution_id) {
          loadExecutionById(message.data.execution_id)
        }
        break

      default:
        console.log('❓ Unhandled WebSocket message type:', message.type)
    }
  }, [currentExecutionId])

  // WebSocket connection with proper execution ID handling
  const { isConnected, error: wsError } = useWorkflowWebSocket({
    workflowId,
    executionId: currentExecutionId || undefined,
    onMessage: handleWebSocketMessage,
    autoConnect: !!currentExecutionId
  })

  const loadExecutionById = async (executionId: number) => {
    try {
      console.log('🔄 Fetching execution data for ID:', executionId)
      const executionDetails = await apiClient.getWorkflowExecution(executionId.toString())
      console.log('📦 Received execution data:', {
        status: executionDetails.status,
        subtasks: executionDetails.output_data?.subtasks?.length || 0,
        started_at: executionDetails.started_at
      })
      setExecutionData((prev: any) => ({
        ...prev,
        execution: executionDetails
      }))
    } catch (err) {
      console.error('❌ Error loading execution by ID:', err)
    }
  }

  // Load workflow data
  useEffect(() => {
    loadWorkflowData()
    
    const interval = setInterval(() => {
      if (isExecuting) {
        loadLiveProgress()
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [workflowId, isExecuting])

  // Auto-start execution if requested
  useEffect(() => {
    if (autoStart && !hasAutoStarted && executionData) {
      setHasAutoStarted(true)
      setTimeout(() => {
        handleStartExecution()
      }, 1000)
    }
  }, [autoStart, hasAutoStarted, executionData])

  const loadWorkflowData = async () => {
    try {
      const workflow = await apiClient.getWorkflow(workflowId.toString())
      const progress = await apiClient.getWorkflowLiveProgress(workflowId.toString()).catch(() => ({}))
      
      let executionDetails = null
      if (progress?.current_execution?.id) {
        try {
          executionDetails = await apiClient.getWorkflowExecution(progress.current_execution.id.toString())
          setIsExecuting(true)
        } catch (err) {
          console.error('Error loading current execution:', err)
        }
      } else if (progress?.recent_executions && progress.recent_executions.length > 0) {
        try {
          const recentId = progress.recent_executions[0].id
          executionDetails = await apiClient.getWorkflowExecution(recentId.toString())
        } catch (err) {
          console.error('Error loading recent execution:', err)
        }
      } else {
        try {
          const execResponse = await apiClient.getWorkflowExecutions(workflowId.toString())
          const executions = execResponse?.items || execResponse || []
          if (executions && executions.length > 0) {
            const sortedExecs = executions.sort((a: any, b: any) => b.id - a.id)
            executionDetails = sortedExecs[0]
          }
        } catch (err) {
          console.error('Error loading executions:', err)
        }
      }
      
      setExecutionData({
        ...workflow,
        liveProgress: progress || {},
        execution: executionDetails
      })
    } catch (error) {
      console.error('Error loading workflow:', error)
      setExecutionData({
        id: workflowId,
        name: 'Loading...',
        liveProgress: {},
        execution: null
      })
    }
  }

  const loadLiveProgress = async () => {
    try {
      const progress = await apiClient.getWorkflowLiveProgress(workflowId.toString()).catch(() => ({}))
      
      let executionDetails = null
      if (progress?.current_execution?.id) {
        try {
          executionDetails = await apiClient.getWorkflowExecution(progress.current_execution.id.toString())
        } catch (err) {
          console.error('Error loading execution details:', err)
        }
      } else if (progress?.recent_executions && progress.recent_executions.length > 0) {
        try {
          const recentId = progress.recent_executions[0].id
          executionDetails = await apiClient.getWorkflowExecution(recentId.toString())
          if (executionDetails?.status === 'completed' || executionDetails?.status === 'failed') {
            setIsExecuting(false)
          }
        } catch (err) {
          console.error('Error loading recent execution:', err)
        }
      } else {
        try {
          const execResponse = await apiClient.getWorkflowExecutions(workflowId.toString())
          const executions = execResponse?.items || execResponse || []
          if (executions.length > 0) {
            const latestExec = executions[0]
            executionDetails = await apiClient.getWorkflowExecution(latestExec.id.toString())
            if (executionDetails?.status === 'completed' || executionDetails?.status === 'failed') {
              setIsExecuting(false)
            }
          }
        } catch (err) {
          console.error('Error loading executions:', err)
        }
      }
      
      setExecutionData((prev: any) => ({
        ...prev,
        liveProgress: progress || {},
        execution: executionDetails || prev?.execution
      }))
    } catch (error) {
      console.error('Error loading live progress:', error)
    }
  }

  const handleStartExecution = async () => {
    try {
      setIsExecuting(true)
      const result = await apiClient.executeWorkflow(workflowId.toString(), {})
      console.log('Execution started:', result)
      
      // CRITICAL: Capture the execution ID from the response
      if (result?.execution_id || result?.id) {
        const execId = result.execution_id || result.id
        setCurrentExecutionId(execId)
        console.log('🎯 Execution ID captured:', execId)
        
        // Load execution details immediately
        await loadExecutionById(execId)
      }
      
      setTimeout(() => {
        loadLiveProgress()
      }, 1000)
    } catch (error) {
      console.error('Error starting execution:', error)
      alert(`Failed to start execution: ${error instanceof Error ? error.message : 'Unknown error'}`)
      setIsExecuting(false)
    }
  }

  const handlePauseExecution = () => {
    setIsExecuting(false)
  }

  const handleStopExecution = () => {
    setIsExecuting(false)
  }

  if (!executionData) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-muted-foreground">Loading workflow...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={cn("h-screen flex flex-col bg-background", isFullscreen && "fixed inset-0 z-50")}>
      {/* Header - Compact and Professional */}
      <div className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={onBack}
              className="gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>
            
            <div className="max-w-2xl">
              <h1 className="text-lg font-semibold">
                <span className="gradient-text">{executionData.name}</span>
              </h1>
              <p className="text-xs text-muted-foreground line-clamp-1">
                {executionData.description}
              </p>
            </div>

            <Badge 
              variant={isExecuting ? 'default' : 'secondary'}
              className={cn(
                "font-medium",
                isExecuting && "bg-blue-500/20 text-blue-400 border-blue-500/30"
              )}
            >
              {isExecuting ? 'Running' : 'Ready'}
            </Badge>

            {/* WebSocket Status */}
            <Badge 
              variant="outline"
              className={cn(
                "text-xs",
                isConnected 
                  ? "bg-green-500/10 text-green-400 border-green-500/30" 
                  : "bg-gray-500/10 text-gray-400 border-gray-500/30"
              )}
            >
              <div className={cn(
                "w-2 h-2 rounded-full mr-2",
                isConnected ? "bg-green-400 animate-pulse" : "bg-gray-400"
              )} />
              {isConnected ? 'Live' : 'Offline'}
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            {!isExecuting ? (
              <Button 
                onClick={handleStartExecution} 
                variant="outline"
                size="sm"
              >
                <Play className="w-4 h-4 mr-2" />
                Start Execution
              </Button>
            ) : (
              <>
                <Button onClick={handlePauseExecution} variant="outline" size="sm">
                  <Pause className="w-4 h-4 mr-2" />
                  Pause
                </Button>
                <Button 
                  onClick={handleStopExecution} 
                  variant="outline" 
                  size="sm"
                  className="text-red-500 hover:text-red-600"
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop
                </Button>
              </>
            )}

            {/* View Toggle Buttons */}
            <div className="flex items-center gap-1 border-l border-border ml-2 pl-2">
              <Button
                variant={activeView === 'orchestrator' ? 'secondary' : 'ghost'}
                size="sm"
                onClick={() => setActiveView('orchestrator')}
                className="h-8 px-3"
              >
                <Brain className="w-4 h-4" />
              </Button>
              <Button
                variant={activeView === 'analytics' ? 'secondary' : 'ghost'}
                size="sm"
                onClick={() => setActiveView('analytics')}
                className="h-8 px-3"
              >
                <BarChart3 className="w-4 h-4" />
              </Button>
            </div>

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="h-8 w-8"
            >
              {isFullscreen ? (
                <Minimize2 className="w-4 h-4" />
              ) : (
                <Maximize2 className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {activeView === 'analytics' ? (
          <div className="flex-1 p-4 overflow-auto">
            <EnhancedOrchestratorView
              workflowId={workflowId}
              executionId={currentExecutionId?.toString()}
              isExecuting={isExecuting}
            />
          </div>
        ) : (
          <>
            {/* Top Section: Communication Log (50%) */}
            <div className="h-[50%] border-b border-border">
              <div className="h-full p-4">
                <div className="h-full glass-card overflow-hidden">
                  <CommunicationLog
                    workflowId={workflowId}
                    isExecuting={isExecuting}
                    workflow={executionData}
                  />
                </div>
              </div>
            </div>

            {/* Middle Section: Agent List + Workspace Split (40%) */}
            <div className="h-[40%] border-b border-border">
              <div className="h-full flex">
                {/* Left: Agent List Sidebar (20%) */}
                <div className="w-[20%] border-r border-border p-3 overflow-y-auto">
                  <h3 className="text-sm font-semibold mb-3 text-muted-foreground uppercase tracking-wide">
                    Active Agents
                  </h3>
                  <AgentListSidebar
                    workflow={executionData}
                    selectedAgent={selectedAgent}
                    onAgentSelect={setSelectedAgent}
                    isExecuting={isExecuting}
                  />
                </div>

                {/* Right: Selected Agent Workspace (80%) */}
                <div className="flex-1 p-4">
                  <div className="h-full glass-card overflow-hidden">
                    <AgentWorkspaceTabs
                      workflow={executionData}
                      selectedAgent={selectedAgent}
                      onAgentSelect={setSelectedAgent}
                      isExecuting={isExecuting}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom Section: Progress Bar (10%) */}
            <div className="h-[10%] min-h-[60px]">
              <OrchestratorControl
                workflow={executionData}
                isExecuting={isExecuting}
              />
            </div>
          </>
        )}
      </div>
    </div>
  )
}
