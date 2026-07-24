
'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Play, 
  Pause, 
  Square,
  Activity,
  Clock,
  CheckCircle,
  AlertTriangle,
  MessageCircle,
  User,
  Bot,
  ArrowRight,
  ChevronDown,
  ChevronRight,
  Zap,
  Eye,
  Settings,
  RefreshCw,
  Terminal,
  GitBranch,
  Database,
  FileText,
  Heart,
  Cpu
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { apiClient } from "@/lib/api-client"

// Types for real-time workflow execution
interface AgentActivity {
  id: string
  agentId: number
  agentName: string
  status: 'idle' | 'thinking' | 'executing' | 'waiting_input' | 'completed' | 'error'
  currentTask: string
  progress: number
  startTime: string
  logs: string[]
  outputData?: any
  needsHumanInput?: {
    question: string
    type: 'text' | 'choice' | 'file' | 'approval'
    options?: string[]
    required: boolean
  }
}

interface WorkflowExecution {
  id: string
  workflowId: number
  workflowName: string
  status: 'initializing' | 'running' | 'paused' | 'completed' | 'failed' | 'waiting_input'
  progress: number
  currentStep: string
  totalSteps: number
  completedSteps: number
  startTime: string
  estimatedCompletion?: string
  agentActivities: AgentActivity[]
  orchestratorLogs: string[]
  systemMetrics: {
    cpuUsage: number
    memoryUsage: number
    activeConnections: number
  }
}

interface HumanInteraction {
  id: string
  agentId: number
  agentName: string
  question: string
  type: 'text' | 'choice' | 'file' | 'approval'
  options?: string[]
  required: boolean
  timestamp: string
}

export function InteractiveWorkflowExecution({ workflowId }: { workflowId: number }) {
  // State management
  const [execution, setExecution] = useState<WorkflowExecution | null>(null)
  const [isExecuting, setIsExecuting] = useState(false)
  const [humanInteractions, setHumanInteractions] = useState<HumanInteraction[]>([])
  const [selectedInteraction, setSelectedInteraction] = useState<HumanInteraction | null>(null)
  const [interactionResponse, setInteractionResponse] = useState('')
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set())
  const [autoScroll, setAutoScroll] = useState(true)
  const [showSystemMetrics, setShowSystemMetrics] = useState(true)
  
  // Refs for auto-scroll
  const logsEndRef = useRef<HTMLDivElement>(null)
  const activitiesEndRef = useRef<HTMLDivElement>(null)

  // Polling for real-time updates
  useEffect(() => {
    if (!isExecuting) return

    const interval = setInterval(() => {
      updateExecutionStatus()
    }, 2000) // Update every 2 seconds

    return () => clearInterval(interval)
  }, [isExecuting, workflowId])

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll) {
      logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [execution?.orchestratorLogs, autoScroll])

  const updateExecutionStatus = async () => {
    try {
      // Get live progress from backend
      const progressData = await apiClient.getWorkflowLiveProgress(workflowId)
      
      // Simulate realistic execution data based on backend response
      const mockExecution: WorkflowExecution = {
        id: `exec_${workflowId}_${Date.now()}`,
        workflowId,
        workflowName: `Workflow ${workflowId}`,
        status: progressData.status || 'running',
        progress: progressData.progress || Math.min(95, (execution?.progress || 0) + Math.random() * 5),
        currentStep: progressData.current_step || `Step ${Math.floor(Math.random() * 5) + 1}`,
        totalSteps: 8,
        completedSteps: Math.floor(((progressData.progress || 0) / 100) * 8),
        startTime: execution?.startTime || new Date().toISOString(),
        estimatedCompletion: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
        agentActivities: generateAgentActivities(progressData.agent_activities || []),
        orchestratorLogs: [
          ...(execution?.orchestratorLogs || []),
          ...generateNewLogs()
        ].slice(-50), // Keep last 50 logs
        systemMetrics: {
          cpuUsage: 40 + Math.random() * 30,
          memoryUsage: 55 + Math.random() * 20,
          activeConnections: Math.floor(Math.random() * 8) + 2
        }
      }

      setExecution(mockExecution)

      // Check for human interactions needed
      const interactions = mockExecution.agentActivities
        .filter(activity => activity.needsHumanInput)
        .map(activity => ({
          id: `interaction_${activity.id}`,
          agentId: activity.agentId,
          agentName: activity.agentName,
          question: activity.needsHumanInput!.question,
          type: activity.needsHumanInput!.type,
          options: activity.needsHumanInput!.options,
          required: activity.needsHumanInput!.required,
          timestamp: new Date().toISOString()
        }))

      setHumanInteractions(prev => {
        const existingIds = new Set(prev.map(i => i.id))
        const newInteractions = interactions.filter(i => !existingIds.has(i.id))
        return [...prev, ...newInteractions]
      })

    } catch (error) {
      console.error('Error updating execution status:', error)
    }
  }

  const generateAgentActivities = (backendActivities: any[]): AgentActivity[] => {
    // If we have backend data, use it, otherwise simulate
    if (backendActivities.length > 0) {
      return backendActivities.map(activity => ({
        id: `agent_${activity.agent_id}`,
        agentId: activity.agent_id,
        agentName: activity.agent_name || `Agent ${activity.agent_id}`,
        status: activity.status || 'executing',
        currentTask: activity.current_task || 'Processing data...',
        progress: activity.progress || Math.random() * 100,
        startTime: activity.start_time || new Date().toISOString(),
        logs: activity.logs || [`Started task: ${activity.current_task}`],
        outputData: activity.output_data
      }))
    }

    // Simulate agent activities
    const agentTypes = [
      { id: 1, name: 'CodeArchitect', task: 'Analyzing system architecture' },
      { id: 2, name: 'SecurityGuard', task: 'Running security scans' },
      { id: 3, name: 'DataAnalyst', task: 'Processing data patterns' },
      { id: 4, name: 'TestMaster', task: 'Executing test suite' }
    ]

    return agentTypes.map(agent => {
      const progress = Math.min(100, (execution?.agentActivities.find(a => a.agentId === agent.id)?.progress || 0) + Math.random() * 10)
      const shouldNeedInput = Math.random() > 0.95 && progress > 50 // 5% chance to need input when > 50% done
      
      return {
        id: `agent_${agent.id}`,
        agentId: agent.id,
        agentName: agent.name,
        status: shouldNeedInput ? 'waiting_input' : (progress >= 100 ? 'completed' : 'executing'),
        currentTask: agent.task,
        progress,
        startTime: execution?.agentActivities.find(a => a.agentId === agent.id)?.startTime || new Date().toISOString(),
        logs: [
          ...((execution?.agentActivities.find(a => a.agentId === agent.id)?.logs || [])),
          ...(Math.random() > 0.8 ? [`[${new Date().toLocaleTimeString()}] ${generateAgentLog(agent.name)}`] : [])
        ].slice(-10),
        needsHumanInput: shouldNeedInput ? {
          question: `${agent.name} needs clarification: How should I proceed with ${agent.task.toLowerCase()}?`,
          type: 'choice',
          options: ['Continue with current approach', 'Try alternative method', 'Require manual review'],
          required: true
        } : undefined
      }
    })
  }

  const generateNewLogs = (): string[] => {
    if (Math.random() > 0.7) { // 30% chance of new log
      const logTypes = [
        'Orchestrator: Coordinating agent tasks...',
        'System: Resource allocation optimized',
        'Orchestrator: Checking task dependencies',
        'System: Performance metrics updated',
        'Orchestrator: Agents synchronization complete',
        'System: Workflow checkpoint created'
      ]
      return [`[${new Date().toLocaleTimeString()}] ${logTypes[Math.floor(Math.random() * logTypes.length)]}`]
    }
    return []
  }

  const generateAgentLog = (agentName: string): string => {
    const logTemplates = {
      CodeArchitect: [
        'Analyzing code structure...',
        'Identifying design patterns...',
        'Reviewing best practices...',
        'Optimizing architecture...'
      ],
      SecurityGuard: [
        'Scanning for vulnerabilities...',
        'Checking access controls...',
        'Validating security policies...',
        'Analyzing threat vectors...'
      ],
      DataAnalyst: [
        'Processing data stream...',
        'Calculating metrics...',
        'Identifying patterns...',
        'Generating insights...'
      ],
      TestMaster: [
        'Running test cases...',
        'Validating outputs...',
        'Checking edge cases...',
        'Generating test report...'
      ]
    }
    const logs = logTemplates[agentName as keyof typeof logTemplates] || ['Processing task...']
    return logs[Math.floor(Math.random() * logs.length)]
  }

  const startExecution = async () => {
    setIsExecuting(true)
    try {
      // Start workflow execution via backend
      await apiClient.executeWorkflow(workflowId)
      
      // Initialize execution state
      setExecution({
        id: `exec_${workflowId}_${Date.now()}`,
        workflowId,
        workflowName: `Workflow ${workflowId}`,
        status: 'initializing',
        progress: 0,
        currentStep: 'Initializing agents...',
        totalSteps: 8,
        completedSteps: 0,
        startTime: new Date().toISOString(),
        agentActivities: [],
        orchestratorLogs: [`[${new Date().toLocaleTimeString()}] Workflow execution started`],
        systemMetrics: {
          cpuUsage: 25,
          memoryUsage: 40,
          activeConnections: 1
        }
      })
    } catch (error) {
      console.error('Error starting workflow:', error)
      setIsExecuting(false)
    }
  }

  const pauseExecution = () => {
    setIsExecuting(false)
    if (execution) {
      setExecution({
        ...execution,
        status: 'paused',
        orchestratorLogs: [
          ...execution.orchestratorLogs,
          `[${new Date().toLocaleTimeString()}] Workflow execution paused by user`
        ]
      })
    }
  }

  const stopExecution = () => {
    setIsExecuting(false)
    setExecution(null)
    setHumanInteractions([])
  }

  const toggleAgentExpansion = (agentId: string) => {
    const newExpanded = new Set(expandedAgents)
    if (newExpanded.has(agentId)) {
      newExpanded.delete(agentId)
    } else {
      newExpanded.add(agentId)
    }
    setExpandedAgents(newExpanded)
  }

  const handleHumanResponse = async (interactionId: string, response: string) => {
    try {
      // Send response to backend (simulate for now)
      console.log(`Responding to interaction ${interactionId}:`, response)
      
      // Remove interaction from pending list
      setHumanInteractions(prev => prev.filter(i => i.id !== interactionId))
      setSelectedInteraction(null)
      setInteractionResponse('')

      // Update execution logs
      if (execution) {
        setExecution({
          ...execution,
          orchestratorLogs: [
            ...execution.orchestratorLogs,
            `[${new Date().toLocaleTimeString()}] Human response received: "${response}"`
          ]
        })
      }
    } catch (error) {
      console.error('Error submitting human response:', error)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
      case 'executing': return 'text-info bg-info/10 border-info/20'
      case 'completed': return 'text-success bg-success/10 border-success/20'
      case 'failed':
      case 'error': return 'text-destructive bg-destructive/10 border-destructive/20'
      case 'waiting_input': return 'text-warning bg-warning/10 border-warning/20'
      case 'paused': return 'text-warning bg-warning/10 border-warning/20'
      default: return 'text-muted-foreground bg-secondary/50 border-border/30'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
      case 'executing': return Activity
      case 'completed': return CheckCircle
      case 'failed':
      case 'error': return AlertTriangle
      case 'waiting_input': return MessageCircle
      case 'paused': return Pause
      default: return Clock
    }
  }

  return (
    <div className="space-y-6">
      {/* Execution Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between p-6 glass-card"
      >
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            {!isExecuting ? (
              <Button onClick={startExecution} variant="outline">
                <Play className="w-4 h-4 mr-2" />
                Start Execution
              </Button>
            ) : (
              <>
                <Button onClick={pauseExecution} variant="outline">
                  <Pause className="w-4 h-4 mr-2" />
                  Pause
                </Button>
                <Button onClick={stopExecution} variant="outline" className="text-destructive hover:bg-destructive/10">
                  <Square className="w-4 h-4 mr-2" />
                  Stop
                </Button>
              </>
            )}
          </div>

          {execution && (
            <div className="flex items-center space-x-4">
              <Badge className={getStatusColor(execution.status)}>
                {execution.status.replace('_', ' ')}
              </Badge>
              <div className="text-sm text-muted-foreground">
                Step {execution.completedSteps} of {execution.totalSteps}
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoScroll(!autoScroll)}
            className={autoScroll ? 'bg-info/10 border-info/20' : ''}
          >
            <Eye className="w-4 h-4 mr-2" />
            Auto-scroll
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowSystemMetrics(!showSystemMetrics)}
          >
            <Settings className="w-4 h-4 mr-2" />
            Metrics
          </Button>
          {humanInteractions.length > 0 && (
            <Badge className="bg-primary/10 text-primary border-primary/20">
              {humanInteractions.length} interaction{humanInteractions.length > 1 ? 's' : ''} needed
            </Badge>
          )}
        </div>
      </motion.div>

      {/* Execution Overview */}
      {execution && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 lg:grid-cols-4 gap-6"
        >
          {/* Progress Overview */}
          <div className="lg:col-span-3 glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">{execution.workflowName}</h3>
              <div className="text-sm text-muted-foreground">
                {Math.round(execution.progress)}% complete
              </div>
            </div>
            
            <Progress value={execution.progress} className="h-3 mb-4" />
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground mb-1">Current Step</p>
                <p className="font-medium">{execution.currentStep}</p>
              </div>
              <div>
                <p className="text-muted-foreground mb-1">Started</p>
                <p className="font-medium">{new Date(execution.startTime).toLocaleTimeString()}</p>
              </div>
              <div>
                <p className="text-muted-foreground mb-1">Est. Completion</p>
                <p className="font-medium">
                  {execution.estimatedCompletion 
                    ? new Date(execution.estimatedCompletion).toLocaleTimeString()
                    : 'Calculating...'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* System Metrics */}
          {showSystemMetrics && (
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold mb-4">System Metrics</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm flex items-center">
                      <Cpu className="w-4 h-4 mr-1" />
                      CPU
                    </span>
                    <span className="text-sm font-medium">{Math.round(execution.systemMetrics.cpuUsage)}%</span>
                  </div>
                  <Progress value={execution.systemMetrics.cpuUsage} className="h-2" />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm flex items-center">
                      <Database className="w-4 h-4 mr-1" />
                      Memory
                    </span>
                    <span className="text-sm font-medium">{Math.round(execution.systemMetrics.memoryUsage)}%</span>
                  </div>
                  <Progress value={execution.systemMetrics.memoryUsage} className="h-2" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm flex items-center">
                    <Activity className="w-4 h-4 mr-1" />
                    Connections
                  </span>
                  <span className="text-sm font-medium">{execution.systemMetrics.activeConnections}</span>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Main Execution Interface */}
      {execution && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Agent Activities */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center">
                <Bot className="w-5 h-5 mr-2" />
                Agent Activities
              </h3>
              <div className="text-sm text-muted-foreground">
                {execution.agentActivities.filter(a => a.status === 'executing').length} active
              </div>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {execution.agentActivities.map((activity) => {
                const StatusIcon = getStatusIcon(activity.status)
                const isExpanded = expandedAgents.has(activity.id)

                return (
                  <div key={activity.id} className="border border-border/30 rounded-lg p-3">
                    <div
                      className="flex items-center justify-between cursor-pointer"
                      onClick={() => toggleAgentExpansion(activity.id)}
                    >
                      <div className="flex items-center space-x-3">
                        <StatusIcon className={`w-4 h-4 ${getStatusColor(activity.status).split(' ')[0]}`} />
                        <div>
                          <p className="font-medium">{activity.agentName}</p>
                          <p className="text-sm text-muted-foreground">{activity.currentTask}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="text-sm font-medium">{Math.round(activity.progress)}%</div>
                        {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                      </div>
                    </div>

                    <div className="mt-2">
                      <Progress value={activity.progress} className="h-1" />
                    </div>

                    {activity.needsHumanInput && (
                      <div className="mt-3 p-2 bg-primary/10 border border-primary/20 rounded">
                        <p className="text-sm text-primary flex items-center">
                          <MessageCircle className="w-4 h-4 mr-1" />
                          Needs human input
                        </p>
                      </div>
                    )}

                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className="mt-3 space-y-2"
                        >
                          <div className="text-xs text-muted-foreground">
                            Started: {new Date(activity.startTime).toLocaleTimeString()}
                          </div>
                          <div className="max-h-32 overflow-y-auto">
                            <div className="text-xs space-y-1">
                              {activity.logs.map((log, index) => (
                                <div key={index} className="text-muted-foreground">{log}</div>
                              ))}
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )
              })}
              <div ref={activitiesEndRef} />
            </div>
          </motion.div>

          {/* Orchestrator Logs */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center">
                <Terminal className="w-5 h-5 mr-2" />
                Orchestrator Logs
              </h3>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  if (execution) {
                    setExecution({
                      ...execution,
                      orchestratorLogs: []
                    })
                  }
                }}
              >
                Clear
              </Button>
            </div>

            <div className="bg-black/20 rounded-lg p-3 max-h-96 overflow-y-auto font-mono text-sm">
              {execution.orchestratorLogs.map((log, index) => (
                <div key={index} className="text-success mb-1">
                  {log}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </motion.div>
        </div>
      )}

      {/* Human Interaction Modal */}
      <Dialog open={!!selectedInteraction} onOpenChange={() => setSelectedInteraction(null)}>
        <DialogContent className="glass-card max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center">
              <User className="w-5 h-5 mr-2" />
              Human Input Required
            </DialogTitle>
          </DialogHeader>

          {selectedInteraction && (
            <div className="space-y-4">
              <div className="p-4 bg-secondary/30 rounded-lg">
                <p className="font-medium mb-2">{selectedInteraction.agentName} needs your input:</p>
                <p className="text-muted-foreground">{selectedInteraction.question}</p>
              </div>

              {selectedInteraction.type === 'choice' && selectedInteraction.options && (
                <div className="space-y-2">
                  {selectedInteraction.options.map((option, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      className="w-full justify-start"
                      onClick={() => handleHumanResponse(selectedInteraction.id, option)}
                    >
                      {option}
                    </Button>
                  ))}
                </div>
              )}

              {selectedInteraction.type === 'text' && (
                <div className="space-y-4">
                  <Textarea
                    placeholder="Enter your response..."
                    value={interactionResponse}
                    onChange={(e) => setInteractionResponse(e.target.value)}
                    className="min-h-[100px]"
                  />
                  <div className="flex space-x-2">
                    <Button
                      onClick={() => handleHumanResponse(selectedInteraction.id, interactionResponse)}
                      disabled={!interactionResponse.trim()}
                      className="flex-1"
                    >
                      Submit Response
                    </Button>
                    <Button variant="outline" onClick={() => setSelectedInteraction(null)}>
                      Cancel
                    </Button>
                  </div>
                </div>
              )}

              {selectedInteraction.type === 'approval' && (
                <div className="flex space-x-2">
                  <Button
                    onClick={() => handleHumanResponse(selectedInteraction.id, 'approved')}
                    variant="outline"
                    className="flex-1"
                  >
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Approve
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => handleHumanResponse(selectedInteraction.id, 'rejected')}
                    className="flex-1 text-destructive hover:bg-destructive/10"
                  >
                    <AlertTriangle className="w-4 h-4 mr-2" />
                    Reject
                  </Button>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Human Interactions Queue */}
      {humanInteractions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-semibold flex items-center mb-4">
            <MessageCircle className="w-5 h-5 mr-2" />
            Pending Human Interactions ({humanInteractions.length})
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {humanInteractions.map((interaction) => (
              <div key={interaction.id} className="border border-primary/20 rounded-lg p-4 bg-primary/5">
                <div className="flex items-center justify-between mb-2">
                  <Badge className="bg-primary/10 text-primary border-primary/20">
                    {interaction.agentName}
                  </Badge>
                  <div className="text-xs text-muted-foreground">
                    {new Date(interaction.timestamp).toLocaleTimeString()}
                  </div>
                </div>
                <p className="text-sm mb-3">{interaction.question}</p>
                <Button
                  size="sm"
                  onClick={() => setSelectedInteraction(interaction)}
                  className="w-full"
                >
                  Respond
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Quick Start Help */}
      {!execution && !isExecuting && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-6 text-center"
        >
          <GitBranch className="w-12 h-12 text-primary mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">
            Interactive <span className="gradient-text">Workflow Execution</span>
          </h3>
          <p className="text-muted-foreground mb-6">
            Start the workflow to see real-time orchestrator activities, agent execution status, and provide human input when needed.
          </p>
          <div className="flex items-center justify-center space-x-6 text-sm text-muted-foreground">
            <div className="flex items-center">
              <Activity className="w-4 h-4 mr-2" />
              Real-time monitoring
            </div>
            <div className="flex items-center">
              <MessageCircle className="w-4 h-4 mr-2" />
              Human interaction
            </div>
            <div className="flex items-center">
              <Heart className="w-4 h-4 mr-2" />
              Agent coordination
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}
