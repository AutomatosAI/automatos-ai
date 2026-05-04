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
  draft: 'bg-secondary/50 text-muted-foreground border-border/30',
  active: 'bg-info/10 text-info border-info/20',
  running: 'bg-info/10 text-info border-info/20',
  completed: 'bg-success/10 text-success border-success/20',
  paused: 'bg-warning/10 text-warning border-warning/20',
  failed: 'bg-destructive/10 text-destructive border-destructive/20',
  queued: 'bg-secondary/50 text-muted-foreground border-border/30'
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
          <AlertTriangle className="h-8 w-8 text-destructive mx-auto mb-4" />
          <p className="text-destructive">Error loading history</p>
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
          const statusColor = execution.status === 'completed' ? 'text-success' :
                             execution.status === 'failed' ? 'text-destructive' :
                             execution.status === 'running' ? 'text-info' : 'text-muted-foreground'

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

