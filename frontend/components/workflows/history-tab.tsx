'use client'

import { motion } from 'framer-motion'
import { 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  Activity,
  Eye
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useWorkflowExecutions } from '@/hooks/use-workflow-api'

const statusStyles: Record<string, string> = {
  draft: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
  active: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  running: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  completed: 'bg-green-500/10 text-green-400 border-green-500/20',
  paused: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  failed: 'bg-red-500/10 text-red-400 border-red-500/20',
  queued: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
}

export function HistoryTab() {
  const { data: executionsData, isLoading, error } = useWorkflowExecutions()
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading execution history...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
          <p className="text-red-400">Error loading history</p>
        </div>
      </div>
    )
  }

  const executions = executionsData?.items || []

  if (executions.length === 0) {
    return (
      <div className="text-center py-12">
        <Clock className="h-8 w-8 text-muted-foreground mx-auto mb-4" />
        <p className="text-muted-foreground">No execution history found</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Execution History</h3>
          <p className="text-sm text-muted-foreground">
            {executionsData?.total || 0} total executions
          </p>
        </div>
      </div>

      <div className="grid gap-4">
        {executions.map((execution: any, index: number) => {
          const StatusIcon = execution.status === 'completed' ? CheckCircle : 
                            execution.status === 'failed' ? AlertTriangle :
                            execution.status === 'running' ? Activity : Clock
          const statusColor = execution.status === 'completed' ? 'text-green-400' :
                             execution.status === 'failed' ? 'text-red-400' :
                             execution.status === 'running' ? 'text-blue-400' : 'text-gray-400'

          return (
            <motion.div
              key={execution.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              <Card className="glass-card hover:border-primary/20 transition-all duration-300">
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <StatusIcon className={`w-5 h-5 ${statusColor}`} />
                        <div>
                          <p className="font-medium">Workflow ID: {execution.workflow_id}</p>
                          <p className="text-sm text-muted-foreground">
                            Execution #{execution.id}
                          </p>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Status</p>
                          <Badge className={statusStyles[execution.status] || statusStyles.draft}>
                            {execution.status}
                          </Badge>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Started</p>
                          <p className="font-medium">
                            {new Date(execution.started_at).toLocaleString()}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Duration</p>
                          <p className="font-medium">
                            {execution.duration || 'In progress'}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Agent ID</p>
                          <p className="font-medium">{execution.agent_id || 'N/A'}</p>
                        </div>
                      </div>
                    </div>
                    
                    <Button variant="ghost" size="icon">
                      <Eye className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}

