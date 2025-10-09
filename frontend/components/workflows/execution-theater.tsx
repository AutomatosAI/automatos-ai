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
  Activity
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { WorkflowCanvas } from './execution-theater/workflow-canvas'
import { AgentWorkspaceTabs } from './execution-theater/agent-workspace-tabs'
import { OrchestratorControl } from './execution-theater/orchestrator-control'
import { CommunicationLog } from './execution-theater/communication-log'
import { MemoryVisualization } from './execution-theater/memory-visualization'
import { HumanInteractionChat } from './execution-theater/human-interaction-chat'
import { EnhancedOrchestratorView } from './execution-theater/enhanced-orchestrator-view'
import { apiClient } from '@/lib/api-client'
import { useRouter } from 'next/navigation'
import { useWorkflowWebSocket } from '@/hooks/use-workflow-websocket'

interface ExecutionTheaterProps {
  workflowId: number
  onBack: () => void
  autoStart?: boolean
}

export function ExecutionTheater({ workflowId, onBack, autoStart = false }: ExecutionTheaterProps) {
  const router = useRouter()
  const [isExecuting, setIsExecuting] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showMemory, setShowMemory] = useState(false)
  const [showEnhancedView, setShowEnhancedView] = useState(false)
  const [executionData, setExecutionData] = useState<any>(null)
  const [selectedAgent, setSelectedAgent] = useState<number | null>(null)
  const [hasAutoStarted, setHasAutoStarted] = useState(false)
  const [currentExecutionId, setCurrentExecutionId] = useState<number | null>(null)

  // WebSocket for real-time updates
  const handleWebSocketMessage = useCallback((message: any) => {
    console.log('🔥 Real-time WebSocket update:', message.type, message)

    switch (message.type) {
      case 'execution_started':
        console.log('🚀 Execution started event received')
        if (message.data?.execution_id) {
          setCurrentExecutionId(message.data.execution_id)
          setIsExecuting(true)
          // Immediately load execution details
          console.log('📥 Loading execution details for ID:', message.data.execution_id)
          loadExecutionById(message.data.execution_id)
        }
        break

      case 'subtask_execution_update':
        // Real-time subtask progress - update immediately
        console.log('📊 SUBTASK UPDATE RECEIVED:', message.data?.subtask_id, message.data?.status)
        const execId = currentExecutionId || message.data?.execution_id
        if (execId) {
          console.log('🔄 Refreshing execution data for ID:', execId)
          loadExecutionById(execId)
        } else {
          console.warn('⚠️ No execution ID available for subtask update')
        }
        break

      case 'execution_progress':
      case 'subtask_started':
      case 'subtask_completed':
      case 'agent_assigned':
      case 'decomposition_complete':
        console.log('📈 Progress update:', message.type)
        // Reload execution data to show latest progress
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

  const { isConnected, error: wsError } = useWorkflowWebSocket({
    workflowId,
    executionId: currentExecutionId || undefined,
    onMessage: handleWebSocketMessage,
    autoConnect: true
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
      console.log('✅ UI updated with new execution data')
    } catch (err) {
      console.error('❌ Error loading execution by ID:', err)
    }
  }

  // Load workflow data
  useEffect(() => {
    loadWorkflowData()
    
    // Set up polling for live updates
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
      // Small delay to ensure UI is ready
      setTimeout(() => {
        handleStartExecution()
      }, 1000)
    }
  }, [autoStart, hasAutoStarted, executionData])

  const loadWorkflowData = async () => {
    try {
      // Load workflow metadata
      const workflow = await apiClient.getWorkflow(workflowId.toString())
      
      // Load live progress to get current or most recent execution
      const progress = await apiClient.getWorkflowLiveProgress(workflowId.toString()).catch(() => ({}))
      
      // If there's a current execution or recent executions, load the details
      let executionDetails = null
      if (progress?.current_execution?.id) {
        // Load current running execution
        try {
          executionDetails = await apiClient.getWorkflowExecution(progress.current_execution.id.toString())
          setIsExecuting(true)
        } catch (err) {
          console.error('Error loading current execution:', err)
        }
      } else if (progress?.recent_executions && progress.recent_executions.length > 0) {
        // Load most recent completed execution
        try {
          const recentId = progress.recent_executions[0].id
          executionDetails = await apiClient.getWorkflowExecution(recentId.toString())
        } catch (err) {
          console.error('Error loading recent execution:', err)
        }
      } else {
        // Fallback: Query executions for this workflow directly
        try {
          const execResponse = await apiClient.getWorkflowExecutions(workflowId.toString())
          const executions = execResponse?.items || execResponse || []
          if (executions && executions.length > 0) {
            // Sort by ID descending to get most recent
            const sortedExecs = executions.sort((a: any, b: any) => b.id - a.id)
            executionDetails = sortedExecs[0]
            console.log('Loaded most recent execution from executions API:', executionDetails.id)
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
      // Set minimal data so UI doesn't crash
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
      
      // Update execution details if there's an active execution
      let executionDetails = null
      if (progress?.current_execution?.id) {
        try {
          executionDetails = await apiClient.getWorkflowExecution(progress.current_execution.id.toString())
        } catch (err) {
          console.error('Error loading execution details:', err)
        }
      } else if (progress?.recent_executions && progress.recent_executions.length > 0) {
        // Fallback: Load most recent execution
        try {
          const recentId = progress.recent_executions[0].id
          executionDetails = await apiClient.getWorkflowExecution(recentId.toString())
          // If execution is completed, stop polling
          if (executionDetails?.status === 'completed' || executionDetails?.status === 'failed') {
            setIsExecuting(false)
          }
        } catch (err) {
          console.error('Error loading recent execution:', err)
        }
      } else {
        // Fallback: Query executions directly if live-progress returns nothing
        try {
          const execResponse = await apiClient.getWorkflowExecutions(workflowId.toString())
          const executions = execResponse?.items || execResponse || []
          if (executions.length > 0) {
            const latestExec = executions[0]
            executionDetails = await apiClient.getWorkflowExecution(latestExec.id.toString())
            // If execution is completed, stop polling
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
      
      // Immediately load live progress
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
    // TODO: Implement pause API
  }

  const handleStopExecution = () => {
    setIsExecuting(false)
    // TODO: Implement stop API
  }

  if (!executionData) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Loading workflow...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`h-screen flex flex-col ${isFullscreen ? 'fixed inset-0 z-50 bg-background' : ''}`}>
      {/* Header */}
      <div className="border-b border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center space-x-4">
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
              <h1 className="text-xl font-bold">{executionData.name}</h1>
              <p className="text-sm text-muted-foreground line-clamp-2" title={executionData.description}>
                {executionData.description}
              </p>
            </div>

            <Badge variant={isExecuting ? 'default' : 'secondary'}>
              {isExecuting ? 'Running' : 'Ready'}
            </Badge>

            {/* WebSocket Status Indicator */}
            <Badge 
              variant={isConnected ? 'default' : 'secondary'} 
              className={isConnected ? 'bg-green-500/20 text-green-400 border-green-500/30' : 'bg-red-500/20 text-red-400 border-red-500/30'}
            >
              <div className={`w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
              {isConnected ? 'Live' : 'Disconnected'}
            </Badge>
            {wsError && (
              <span className="text-xs text-red-400" title={wsError}>⚠️</span>
            )}
          </div>

          <div className="flex items-center space-x-2">
            {!isExecuting ? (
              <Button onClick={handleStartExecution} className="bg-green-500 hover:bg-green-600">
                <Play className="w-4 h-4 mr-2" />
                Start Execution
              </Button>
            ) : (
              <>
                <Button onClick={handlePauseExecution} variant="outline">
                  <Pause className="w-4 h-4 mr-2" />
                  Pause
                </Button>
                <Button onClick={handleStopExecution} variant="outline" className="text-red-500">
                  <Square className="w-4 h-4 mr-2" />
                  Stop
                </Button>
              </>
            )}

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowMemory(!showMemory)}
              title="Memory Visualization"
            >
              <Settings className="w-4 h-4" />
            </Button>

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowEnhancedView(!showEnhancedView)}
              className={showEnhancedView ? "bg-primary/10" : ""}
              title="Enhanced Orchestrator View"
            >
              <Activity className="w-4 h-4" />
            </Button>

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsFullscreen(!isFullscreen)}
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

      {/* Main Theater Layout */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Show Enhanced View or Regular View */}
        {showEnhancedView ? (
          <div className="flex-1 p-4 overflow-auto">
            <EnhancedOrchestratorView
              workflowId={workflowId}
              executionId={currentExecutionId?.toString()}
              isExecuting={isExecuting}
            />
          </div>
        ) : (
          <>
            {/* Top: Communication & Events (45%) - Full Width with Data Tabs */}
            <div className="h-[45%] p-4 pb-2">
              <div className="h-full border border-border/30 rounded-lg bg-background/30 overflow-hidden">
                <CommunicationLog
                  workflowId={workflowId}
                  isExecuting={isExecuting}
                  workflow={executionData}
                />
              </div>
            </div>

        {/* Middle: Split Layout (45%) - Agent Workspace + Human Chat */}
        <div className="h-[45%] flex gap-4 px-4 pb-2">
          {/* Left: Agent Workspaces (50%) */}
          <div className="w-1/2 border border-border/30 rounded-lg bg-background/30 overflow-hidden">
            <AgentWorkspaceTabs
              workflow={executionData}
              selectedAgent={selectedAgent}
              onAgentSelect={setSelectedAgent}
              isExecuting={isExecuting}
            />
          </div>

          {/* Right: Human Interaction Chat (50%) */}
          <div className="w-1/2 border border-border/30 rounded-lg bg-background/30 overflow-hidden">
            <HumanInteractionChat
              workflowId={workflowId}
              isExecuting={isExecuting}
            />
          </div>
        </div>

            {/* Bottom: Progress & System Status (10%) */}
            <div className="h-[10%] min-h-[60px]">
              <OrchestratorControl
                workflow={executionData}
                isExecuting={isExecuting}
              />
            </div>
          </>
        )}
      </div>

      {/* Memory Visualization Panel (Slide-over) */}
      <AnimatePresence>
        {showMemory && (
          <MemoryVisualization
            workflowId={workflowId}
            agents={executionData.agents || []}
            onClose={() => setShowMemory(false)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

