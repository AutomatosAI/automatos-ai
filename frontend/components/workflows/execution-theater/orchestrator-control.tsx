'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Activity,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  DollarSign,
  Zap
} from 'lucide-react'

interface OrchestratorControlProps {
  workflow: any
  isExecuting: boolean
}

export function OrchestratorControl({ workflow, isExecuting }: OrchestratorControlProps) {
  const execution = workflow?.execution
  const liveProgress = workflow?.liveProgress || {}
  
  // Calculate progress from execution data if available
  const subtasks = execution?.output_data?.subtasks || []
  const completedSubtasks = subtasks.filter((st: any) => st?.execution_result?.status === 'completed').length
  const totalSubtasks = subtasks.length
  const calculatedProgress = totalSubtasks > 0 ? (completedSubtasks / totalSubtasks) * 100 : 0
  
  const progress = liveProgress.current_execution?.progress ?? calculatedProgress
  const currentStep = liveProgress.current_execution?.current_step || 
                      (execution?.status === 'completed' ? 'Completed' : 
                      execution?.status === 'running' ? 'Processing...' : 
                      'Ready')
  
  const completedSteps = completedSubtasks
  const totalSteps = totalSubtasks || 0
  
  // Calculate tokens and cost
  const totalTokens = subtasks.reduce((sum: number, st: any) => 
    sum + (st.execution_result?.tokens_used || 0), 0)
  const costEstimate = (totalTokens / 1000) * 0.002

  return (
    <div className="h-full flex items-center px-6 py-3 border-t border-border/30 mx-4">
      {/* Left Side: Progress Bar */}
      <div className="flex-1 pr-8">
        <div className="flex items-center gap-4">
          <Badge variant={isExecuting ? 'default' : 'secondary'} className="shrink-0">
            {isExecuting ? 'Running' : 'Ready'}
          </Badge>
          
          <div className="flex-1">
            <div className="flex justify-between text-xs mb-1.5">
              <span className="text-muted-foreground">{currentStep}</span>
              <span className="font-medium">{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
          
          <div className="text-xs text-muted-foreground shrink-0">
            <span className="font-medium">{completedSteps}</span> / {totalSteps} steps
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="h-12 w-px bg-border/50 mx-6" />

      {/* Right Side: System Status */}
      <div className="flex items-center gap-8 text-sm">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-400" />
          <span className="text-muted-foreground">Agents:</span>
          <span className="font-medium">{completedSteps} / {totalSteps}</span>
        </div>

        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span className="text-muted-foreground">Tokens:</span>
          <span className="font-medium font-mono">{totalTokens.toLocaleString()}</span>
        </div>

        <div className="flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-green-400" />
          <span className="text-muted-foreground">Cost:</span>
          <span className="font-medium text-green-400">${costEstimate.toFixed(4)}</span>
        </div>

        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-purple-400" />
          <span className="text-muted-foreground">Memory:</span>
          <span className="font-medium">{liveProgress.memory_usage || '0 MB'}</span>
        </div>
      </div>
    </div>
  )
}

