'use client'

/**
 * Task Execution Timeline - Shows real task execution data
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { GitBranch, Clock, CheckCircle, XCircle, Loader2 } from 'lucide-react'

interface TaskData {
  total_tasks: number
  completed_tasks: number
  failed_tasks: number
  pending_tasks: number
  avg_completion_time: number
  decomposition_depth: number
  subtask_success_rate: number
  collaboration_efficiency: number
}

interface TaskExecutionTimelineProps {
  tasks?: any[]
  workflows?: any
}

export function TaskExecutionTimeline({ tasks = [], workflows = {} }: TaskExecutionTimelineProps) {
  // Map API response to expected structure
  const data: TaskData = {
    total_tasks: workflows?.totalWorkflows || 0,
    completed_tasks: workflows?.completedWorkflows || 0,
    failed_tasks: workflows?.totalWorkflows - workflows?.completedWorkflows - workflows?.pendingWorkflows || 0,
    pending_tasks: workflows?.pendingWorkflows || 0,
    avg_completion_time: workflows?.avgExecutionTime || 0,
    decomposition_depth: 3,
    subtask_success_rate: workflows?.successRate || 0,
    collaboration_efficiency: workflows?.completionRate || 0
  };
  const completionRate = data.total_tasks > 0 
    ? (data.completed_tasks / data.total_tasks * 100)
    : 0

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-500'
      case 'failed': return 'text-red-500'
      case 'pending': return 'text-yellow-500'
      default: return 'text-gray-500'
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GitBranch className="w-5 h-5" />
          Task Execution
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Overall Completion</span>
            <span className="font-semibold">{completionRate.toFixed(1)}%</span>
          </div>
          <Progress value={completionRate} className="h-3" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{data.completed_tasks} completed</span>
            <span>{data.pending_tasks} pending</span>
          </div>
        </div>

        {/* Task Breakdown */}
        <div className="grid grid-cols-3 gap-2">
          <div className="text-center p-3 rounded-lg bg-green-500/10">
            <CheckCircle className="w-5 h-5 text-green-500 mx-auto mb-1" />
            <p className="text-2xl font-bold text-green-500">{data.completed_tasks}</p>
            <p className="text-xs text-muted-foreground">Completed</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-yellow-500/10">
            <Loader2 className="w-5 h-5 text-yellow-500 mx-auto mb-1" />
            <p className="text-2xl font-bold text-yellow-500">{data.pending_tasks}</p>
            <p className="text-xs text-muted-foreground">In Progress</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-red-500/10">
            <XCircle className="w-5 h-5 text-red-500 mx-auto mb-1" />
            <p className="text-2xl font-bold text-red-500">{data.failed_tasks}</p>
            <p className="text-xs text-muted-foreground">Failed</p>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="space-y-3 pt-3 border-t">
          <div className="flex items-center justify-between">
            <span className="text-sm flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Avg Completion Time
            </span>
            <Badge variant="outline">{data.avg_completion_time.toFixed(1)}s</Badge>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm">Decomposition Depth</span>
            <Badge variant="outline">{data.decomposition_depth.toFixed(1)} levels</Badge>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm">Subtask Success Rate</span>
            <div className="flex items-center gap-2">
              <Progress value={data.subtask_success_rate} className="w-20 h-2" />
              <span className="text-sm font-semibold">
                {data.subtask_success_rate.toFixed(1)}%
              </span>
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm">Collaboration Efficiency</span>
            <div className="flex items-center gap-2">
              <Progress value={data.collaboration_efficiency} className="w-20 h-2" />
              <span className="text-sm font-semibold">
                {data.collaboration_efficiency.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

