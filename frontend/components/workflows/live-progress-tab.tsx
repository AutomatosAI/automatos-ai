'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Activity,
  Clock,
  CheckCircle,
  AlertTriangle,
  Play,
  Eye,
  RefreshCw
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useWorkflowExecutions } from '@/hooks/use-workflow-api'
import { LiveProgressPanel } from './live-progress-panel'

export function LiveProgressTab() {
  const { data: executionsData, isLoading, error, refetch } = useWorkflowExecutions()
  const [selectedExecution, setSelectedExecution] = useState<any>(null)
  const [showLivePanel, setShowLivePanel] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Auto-refresh every 3 seconds
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        refetch()
      }, 3000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, refetch])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading executions...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
          <p className="text-red-400">Error loading executions</p>
        </div>
      </div>
    )
  }

  const executions = executionsData?.items || []
  const runningExecutions = executions.filter((ex: any) => ex.status === 'running')
  const recentCompleted = executions.filter((ex: any) => ex.status === 'completed').slice(0, 5)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Live Workflow Executions</h3>
          <p className="text-sm text-muted-foreground">
            {runningExecutions.length} running · {recentCompleted.length} recently completed
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={autoRefresh ? 'border-green-500/50 text-green-400' : ''}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto-refresh
          </Button>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Running Executions */}
      {runningExecutions.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-semibold flex items-center">
            <Activity className="w-4 h-4 mr-2 text-blue-400 animate-pulse" />
            Active Executions ({runningExecutions.length})
          </h4>
          <div className="grid gap-4">
            {runningExecutions.map((execution: any, index: number) => (
              <motion.div
                key={execution.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Card className="glass-card border-blue-500/30">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></div>
                        <div>
                          <CardTitle className="text-base">
                            Workflow #{execution.workflow_id}
                          </CardTitle>
                          <p className="text-xs text-muted-foreground">
                            Execution #{execution.id}
                          </p>
                        </div>
                      </div>
                      <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                        <Play className="w-3 h-3 mr-1" />
                        Running
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {/* Mock Progress (would be real from WebSocket) */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Progress</span>
                        <span>~{Math.min(Math.floor((Date.now() - new Date(execution.started_at).getTime()) / 1000 / 20 * 100), 95)}%</span>
                      </div>
                      <Progress 
                        value={Math.min(Math.floor((Date.now() - new Date(execution.started_at).getTime()) / 1000 / 20 * 100), 95)} 
                        className="h-2"
                      />
                    </div>

                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center text-muted-foreground">
                        <Clock className="w-3 h-3 mr-1" />
                        Started {new Date(execution.started_at).toLocaleTimeString()}
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setSelectedExecution(execution)
                          setShowLivePanel(true)
                        }}
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        View Details
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Recently Completed */}
      {recentCompleted.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-semibold flex items-center">
            <CheckCircle className="w-4 h-4 mr-2 text-green-400" />
            Recently Completed ({recentCompleted.length})
          </h4>
          <div className="grid gap-3">
            {recentCompleted.map((execution: any, index: number) => (
              <motion.div
                key={execution.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Card className="glass-card border-green-500/20 hover:border-green-500/40 transition-all">
                  <CardContent className="pt-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <div>
                          <p className="text-sm font-medium">
                            Workflow #{execution.workflow_id} · Execution #{execution.id}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            Completed {new Date(execution.completed_at).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setSelectedExecution(execution)
                          setShowLivePanel(true)
                        }}
                      >
                        <Eye className="w-3 h-3" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {runningExecutions.length === 0 && recentCompleted.length === 0 && (
        <div className="text-center py-12">
          <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">No active or recent executions</p>
          <p className="text-sm text-muted-foreground mt-2">
            Execute a workflow to see live progress here
          </p>
        </div>
      )}

      {/* Live Progress Panel Modal */}
      {selectedExecution && (
        <LiveProgressPanel
          workflowId={selectedExecution.workflow_id}
          workflowName={`Workflow #${selectedExecution.workflow_id}`}
          isOpen={showLivePanel}
          onClose={() => {
            setShowLivePanel(false)
            setSelectedExecution(null)
          }}
        />
      )}
    </div>
  )
}

