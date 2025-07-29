
'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Play, 
  Pause, 
  Square, 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  Activity,
  Cpu,
  MemoryStick,
  User,
  Zap,
  RefreshCw,
  X
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { apiClient } from '@/lib/api'

interface WorkflowStep {
  name: string;
  status: 'completed' | 'running' | 'pending';
  duration: string;
}

interface LiveProgress {
  workflow_id: number;
  execution_id: number;
  status: string;
  progress: {
    percentage: number;
    current_step: string;
    current_step_index: number;
    total_steps: number;
    estimated_completion: string;
  };
  steps: WorkflowStep[];
  timing: {
    started_at: string;
    elapsed_time: string;
    estimated_total: string;
    estimated_remaining: string;
  };
  resources: {
    agent_id: number;
    memory_usage: string;
    cpu_usage: string;
  };
  log_entries: Array<{
    timestamp: string;
    level: string;
    message: string;
  }>;
  last_updated: string;
}

interface LiveProgressPanelProps {
  workflowId: number;
  workflowName: string;
  isOpen: boolean;
  onClose: () => void;
}

export function LiveProgressPanel({ workflowId, workflowName, isOpen, onClose }: LiveProgressPanelProps) {
  const [progressData, setProgressData] = useState<LiveProgress | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    if (isOpen && workflowId) {
      loadProgressData()
      
      if (autoRefresh) {
        const interval = setInterval(loadProgressData, 2000) // Refresh every 2 seconds
        return () => clearInterval(interval)
      }
    }
  }, [isOpen, workflowId, autoRefresh])

  const loadProgressData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const data = await apiClient.request<LiveProgress>(`/api/workflows/${workflowId}/live-progress`)
      setProgressData(data)
      
      // Stop auto-refresh if workflow is completed or failed
      if (data.status === 'idle' || data.status === 'completed' || data.status === 'failed') {
        setAutoRefresh(false)
      }
    } catch (err) {
      console.error('Error loading progress data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load progress data')
    } finally {
      setLoading(false)
    }
  }

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed': return CheckCircle
      case 'running': return Activity
      case 'pending': return Clock
      default: return Clock
    }
  }

  const getStepColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400'
      case 'running': return 'text-blue-400'
      case 'pending': return 'text-gray-400'
      default: return 'text-gray-400'
    }
  }

  const getLogLevelColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'error': return 'text-red-400'
      case 'warning': return 'text-orange-400'
      case 'info': return 'text-blue-400'
      default: return 'text-muted-foreground'
    }
  }

  if (!isOpen) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="glass-card w-full max-w-4xl max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-border/30">
            <div>
              <h2 className="text-xl font-bold">Live Progress</h2>
              <p className="text-sm text-muted-foreground">{workflowName}</p>
            </div>
            
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={autoRefresh ? 'bg-green-500/10 border-green-500/20' : ''}
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
                Auto Refresh
              </Button>
              
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>

          <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(90vh-120px)]">
            {loading && !progressData && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading progress data...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
                  <p className="text-red-400 mb-4">Error: {error}</p>
                  <Button onClick={loadProgressData} variant="outline">
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {progressData && (
              <>
                {/* Status Overview */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="glass-card">
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                          <Activity className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Status</p>
                          <p className="font-semibold capitalize">{progressData.status}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="glass-card">
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                          <Clock className="w-5 h-5 text-orange-400" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Elapsed</p>
                          <p className="font-semibold">{progressData.timing.elapsed_time}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="glass-card">
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                          <Zap className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">ETA</p>
                          <p className="font-semibold">{progressData.timing.estimated_remaining}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Progress Bar */}
                <Card className="glass-card">
                  <CardContent className="p-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold">{progressData.progress.current_step}</h3>
                          <p className="text-sm text-muted-foreground">
                            Step {progressData.progress.current_step_index + 1} of {progressData.progress.total_steps}
                          </p>
                        </div>
                        <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                          {progressData.progress.percentage}%
                        </Badge>
                      </div>
                      
                      <Progress value={progressData.progress.percentage} className="h-3" />
                      
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Started: {new Date(progressData.timing.started_at).toLocaleTimeString()}</span>
                        <span>ETA: {new Date(progressData.progress.estimated_completion).toLocaleTimeString()}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Steps Timeline */}
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="text-lg">Execution Steps</CardTitle>
                  </CardHeader>
                  <CardContent className="p-6">
                    <div className="space-y-4">
                      {progressData.steps.map((step, index) => {
                        const StepIcon = getStepIcon(step.status)
                        const stepColor = getStepColor(step.status)
                        
                        return (
                          <motion.div
                            key={index}
                            className="flex items-center space-x-4 p-3 bg-secondary/20 rounded-lg"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <div className={`w-8 h-8 rounded-full bg-secondary/50 flex items-center justify-center ${
                              step.status === 'running' ? 'animate-pulse' : ''
                            }`}>
                              <StepIcon className={`w-4 h-4 ${stepColor}`} />
                            </div>
                            
                            <div className="flex-1">
                              <h4 className="font-medium">{step.name}</h4>
                              <p className="text-sm text-muted-foreground">
                                Duration: {step.duration}
                              </p>
                            </div>
                            
                            <Badge className={
                              step.status === 'completed' ? 'bg-green-500/10 text-green-400 border-green-500/20' :
                              step.status === 'running' ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' :
                              'bg-gray-500/10 text-gray-400 border-gray-500/20'
                            }>
                              {step.status}
                            </Badge>
                          </motion.div>
                        )
                      })}
                    </div>
                  </CardContent>
                </Card>

                {/* Resource Usage */}
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="text-lg">Resource Usage</CardTitle>
                  </CardHeader>
                  <CardContent className="p-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="flex items-center space-x-3">
                        <User className="w-5 h-5 text-blue-400" />
                        <div>
                          <p className="text-sm text-muted-foreground">Agent</p>
                          <p className="font-medium">Agent #{progressData.resources.agent_id}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-3">
                        <MemoryStick className="w-5 h-5 text-green-400" />
                        <div>
                          <p className="text-sm text-muted-foreground">Memory</p>
                          <p className="font-medium">{progressData.resources.memory_usage}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-3">
                        <Cpu className="w-5 h-5 text-orange-400" />
                        <div>
                          <p className="text-sm text-muted-foreground">CPU</p>
                          <p className="font-medium">{progressData.resources.cpu_usage}</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Live Logs */}
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="text-lg">Live Logs</CardTitle>
                  </CardHeader>
                  <CardContent className="p-6">
                    <ScrollArea className="h-64 w-full">
                      <div className="space-y-2">
                        {progressData.log_entries.map((log, index) => (
                          <motion.div
                            key={index}
                            className="flex items-start space-x-3 p-2 bg-secondary/10 rounded text-sm font-mono"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.05 }}
                          >
                            <span className="text-xs text-muted-foreground shrink-0">
                              {new Date(log.timestamp).toLocaleTimeString()}
                            </span>
                            <span className={`text-xs font-medium shrink-0 ${getLogLevelColor(log.level)}`}>
                              [{log.level}]
                            </span>
                            <span className="text-xs">{log.message}</span>
                          </motion.div>
                        ))}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
