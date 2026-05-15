
'use client'

import { useState, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  Play,
  Pause,
  Square,
  Clock,
  CheckCircle,
  AlertTriangle,
  Activity,
  Users,
  TrendingUp,
  Eye,
  MoreVertical,
  RefreshCw,
  Trash2,
  AlertCircle,
  Search,
  Filter,
  X,
  Copy,
  ChefHat,
  Zap
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { useActiveWorkflows, useExecuteWorkflowAdvanced } from '@/hooks/use-workflow-api'
import { LiveProgressPanel } from './live-progress-panel'
import { apiClient } from '@/lib/api-client'
import { useQueryClient } from '@tanstack/react-query'
import { useToast } from '@/hooks/use-toast'

interface ActiveWorkflow {
  id: number;
  name: string;
  description: string;
  status: string;
  agents: Array<{
    id: number;
    name: string;
    agent_type: string;
    status: string;
  }>;
  current_execution: {
    id: number | null;
    status: string;
    progress: number;
    current_step: string;
    started_at: string | null;
    estimated_completion: string | null;
  };
  metrics: {
    total_executions: number;
    successful_executions: number;
    success_rate: number;
    avg_duration: string;
    last_execution: string | null;
  };
  recent_executions: Array<{
    id: number;
    status: string;
    started_at: string;
    completed_at: string | null;
    duration: string | null;
  }>;
  created_at: string;
  updated_at: string;
}

interface RecipeRun {
  id: number;
  execution_id: string;
  recipe_id: number;
  recipe_template_id: string | null;
  recipe_name: string;
  type: 'recipe';
  status: string;
  current_step: number;
  total_steps: number;
  step_results: Array<{
    step_id: string;
    order: number;
    agent_name: string;
    status: string;
    output: string | null;
    duration_ms: number;
    tokens_used: number;
    error: string | null;
  }>;
  started_at: string | null;
  completed_at: string | null;
  duration: string | null;
  total_tokens: number;
  total_duration_ms: number;
  error_message: string | null;
}

interface ActiveWorkflowsData {
  active_workflows: ActiveWorkflow[];
  recipe_runs: RecipeRun[];
  total_active: number;
  total_recipe_runs: number;
  system_load: number;
  last_updated: string;
}

interface ActiveWorkflowsPanelProps {
  onWorkflowClick?: (workflowId: number) => void
  onPlaybookRunClick?: (run: RecipeRun) => void
}

export function ActiveWorkflowsPanel({ onWorkflowClick, onPlaybookRunClick: onRecipeRunClick }: ActiveWorkflowsPanelProps) {
  const [selectedWorkflow, setSelectedWorkflow] = useState<{ id: number, name: string } | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [showCleanupDialog, setShowCleanupDialog] = useState(false)
  const [cleanupDays, setCleanupDays] = useState('30')
  const [workflowToDelete, setWorkflowToDelete] = useState<{ id: number, name: string } | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')

  const queryClient = useQueryClient()
  const { toast } = useToast()

  // Use React Query hook for active workflows
  const { data: workflowsData, isLoading: loading, error, refetch } = useActiveWorkflows()

  // Use mutation hook for executing workflows
  const executeWorkflowMutation = useExecuteWorkflowAdvanced()

  // Filter workflows based on search and status
  const filteredWorkflows = useMemo(() => {
    if (!workflowsData?.active_workflows) return []

    let filtered = [...workflowsData.active_workflows]

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter(workflow =>
        workflow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        workflow.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
    }

    // Apply status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(workflow =>
        workflow.current_execution.status === statusFilter
      )
    }

    return filtered
  }, [workflowsData, searchQuery, statusFilter])

  // Auto refresh every 5 seconds if enabled
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        refetch()
      }, 5000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, refetch])

  const handleExecuteWorkflow = async (workflowId: number) => {
    console.log('🔥 Cook button clicked! Workflow ID:', workflowId)
    try {
      executeWorkflowMutation.mutate({
        workflowId: workflowId.toString(),
        options: {
          priority: 'normal',
          timeout: 300
        }
      }, {
        onSuccess: (data) => {
          console.log('✅ Execution started:', data)
          toast({
            title: "Workflow started cooking!",
            description: `Execution ID: ${data.execution_id}`,
          })
        },
        onError: (error: any) => {
          console.error('❌ Execution failed:', error)
          toast({
            title: "Failed to start cooking",
            description: error?.message || "Unknown error",
            variant: "destructive",
          })
        }
      })
    } catch (error) {
      console.error('❌ Exception in handleExecuteWorkflow:', error)
    }
  }

  const handleDuplicateWorkflow = async (workflowId: number, workflowName: string) => {
    try {
      const result = await apiClient.duplicateWorkflow(workflowId)
      await queryClient.invalidateQueries({ queryKey: ['activeWorkflows'] })
      await queryClient.invalidateQueries({ queryKey: ['workflows'] })
      refetch()
      // Show success message (you could add a toast here)
      console.log(`✅ Workflow "${workflowName}" duplicated as "${result.name}"`)
    } catch (error) {
      console.error('Error duplicating workflow:', error)
      alert('Failed to duplicate workflow')
    }
  }

  const handleDeleteWorkflow = async () => {
    if (!workflowToDelete) return

    setIsDeleting(true)
    try {
      await apiClient.deleteWorkflow(workflowToDelete.id)
      await queryClient.invalidateQueries({ queryKey: ['activeWorkflows'] })
      toast({
        title: "Workflow removed",
        description: `"${workflowToDelete.name}" has been removed from the kitchen.`,
      })
      setWorkflowToDelete(null)
      refetch()
    } catch (error: any) {
      console.error('Error deleting workflow:', error)
      console.error('Error details:', error?.response?.data || error?.message)
      toast({
        title: "Failed to remove workflow",
        description: error?.response?.data?.detail || error?.message || "Unknown error occurred",
        variant: "destructive",
      })
    } finally {
      setIsDeleting(false)
    }
  }

  const handleCleanupWorkflows = async () => {
    setIsDeleting(true)
    try {
      const response = await apiClient.cleanupOldWorkflows(parseInt(cleanupDays))
      await queryClient.invalidateQueries({ queryKey: ['activeWorkflows'] })
      setShowCleanupDialog(false)
      alert(`Successfully deleted ${response.deleted_count} workflows`)
      refetch()
    } catch (error) {
      console.error('Error cleaning up workflows:', error)
      alert('Failed to cleanup workflows')
    } finally {
      setIsDeleting(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return Activity
      case 'completed': return CheckCircle
      case 'failed': return AlertTriangle
      case 'idle': return Clock
      default: return Clock
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-info'
      case 'completed': return 'text-success'
      case 'failed': return 'text-destructive'
      case 'idle': return 'text-muted-foreground'
      default: return 'text-muted-foreground'
    }
  }

  const getStatusBadgeStyle = (status: string) => {
    switch (status) {
      case 'running': return 'bg-info/10 text-info border-info/20'
      case 'completed': return 'bg-success/10 text-success border-success/20'
      case 'failed': return 'bg-destructive/10 text-destructive border-destructive/20'
      case 'idle': return 'bg-secondary/50 text-muted-foreground border-border/30'
      default: return 'bg-secondary/50 text-muted-foreground border-border/30'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading active workflows...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-destructive mx-auto mb-4" />
          <p className="text-destructive mb-4">Error loading workflows</p>
          <Button onClick={() => refetch()} variant="outline">
            Try Again
          </Button>
        </div>
      </div>
    )
  }

  const hasWorkflows = (workflowsData?.active_workflows?.length ?? 0) > 0
  const hasRecipeRuns = (workflowsData?.recipe_runs?.length ?? 0) > 0

  if (!workflowsData || (!hasWorkflows && !hasRecipeRuns)) {
    return (
      <div className="text-center py-12">
        <Activity className="h-8 w-8 text-muted-foreground mx-auto mb-4" />
        <p className="text-muted-foreground">No active workflows or recipe runs found</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">
            <span className="text-foreground">What's</span> <span className="text-primary">Cooking</span>
          </h3>
          <p className="text-sm text-muted-foreground">
            {filteredWorkflows.length} workflow{filteredWorkflows.length !== 1 ? 's' : ''}{hasRecipeRuns ? ` • ${workflowsData.recipe_runs.length} recipe run${workflowsData.recipe_runs.length !== 1 ? 's' : ''}` : ''} • {workflowsData.system_load}% system load
          </p>
        </div>

        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowCleanupDialog(true)}
            className="bg-destructive/10 border-destructive/20 hover:bg-destructive/20"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Cleanup Old Workflows
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={autoRefresh ? 'bg-success/10 border-success/20' : ''}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
        </div>
      </div>

      {/* Search and Filter Bar */}
      <div className="flex gap-3 items-center">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search workflows by name or description..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 pr-10"
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSearchQuery('')}
              className="absolute right-1 top-1/2 transform -translate-y-1/2 h-7 w-7 p-0"
            >
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>

        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-[180px]">
            <Filter className="w-4 h-4 mr-2" />
            <SelectValue placeholder="Filter by status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="idle">Idle</SelectItem>
            <SelectItem value="running">Running</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="failed">Failed</SelectItem>
          </SelectContent>
        </Select>

        {(searchQuery || statusFilter !== 'all') && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setSearchQuery('')
              setStatusFilter('all')
            }}
          >
            Clear Filters
          </Button>
        )}
      </div>

      {/* Active Workflows List */}
      {filteredWorkflows.length === 0 ? (
        <div className="text-center py-12">
          <Search className="h-8 w-8 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground mb-2">No workflows match your filters</p>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setSearchQuery('')
              setStatusFilter('all')
            }}
          >
            Clear Filters
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredWorkflows.map((workflow, index) => {
            const StatusIcon = getStatusIcon(workflow.current_execution.status)
            const statusColor = getStatusColor(workflow.current_execution.status)

            return (
              <motion.div
                key={workflow.id}
                className="glass-card p-4 hover:border-primary/20 transition-all duration-300"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <div className="flex items-center gap-4">
                  {/* Status Indicator */}
                  <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-secondary/50">
                    <StatusIcon className={`w-5 h-5 ${statusColor}`} />
                  </div>

                  {/* Workflow Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold text-base">{workflow.name}</h4>
                      <Badge className={`text-xs ${
                        workflow.current_execution.status === 'running'
                          ? 'bg-primary/20 text-primary border-primary/30'
                          : workflow.current_execution.status === 'failed'
                            ? 'bg-destructive/20 text-destructive border-destructive/30'
                            : workflow.metrics.last_execution
                              ? 'bg-success/20 text-success border-success/30'
                              : 'bg-destructive/20 text-destructive border-destructive/30'
                      }`}>
                        {workflow.current_execution.status === 'running'
                          ? 'cooking'
                          : workflow.current_execution.status === 'failed'
                            ? 'burnt'
                            : workflow.metrics.last_execution
                              ? 'cooked'
                              : 'never cooked'
                        }
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground line-clamp-1">
                      {workflow.description}
                    </p>
                  </div>

                  {/* Metrics with Runtime Info */}
                  <div className="hidden lg:flex items-center gap-6 text-sm">
                    {/* Runtime Display - Only if cooked */}
                    {(workflow.current_execution.started_at || workflow.metrics.last_execution) && (
                      <div className="text-left max-w-[200px]">
                        <div className="flex items-center gap-1 text-xs text-muted-foreground">
                          <Clock className="w-3 h-3" />
                          {workflow.current_execution.status === 'running' && workflow.current_execution.started_at ? (
                            <span>{new Date(workflow.current_execution.started_at).toLocaleString()}</span>
                          ) : workflow.metrics.last_execution ? (
                            <span>{new Date(workflow.metrics.last_execution).toLocaleString()}</span>
                          ) : null}
                        </div>
                      </div>
                    )}

                    <div className="text-center">
                      <p className="text-muted-foreground text-xs">Agents</p>
                      <p className="font-medium">{workflow.agents?.length || 0}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-muted-foreground text-xs">Runs</p>
                      <p className="font-medium">{workflow.metrics.total_executions}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-muted-foreground text-xs">
                        {workflow.metrics.success_rate === 100 || workflow.metrics.total_executions === 0 ? 'Success' : 'Failed'}
                      </p>
                      <p className={`font-medium ${workflow.metrics.success_rate === 100 || workflow.metrics.total_executions === 0 ? 'text-success' : 'text-destructive'}`}>
                        {workflow.metrics.success_rate === 100 || workflow.metrics.total_executions === 0
                          ? `${workflow.metrics.success_rate}%`
                          : `${100 - workflow.metrics.success_rate}%`
                        }
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-muted-foreground text-xs">Avg Time</p>
                      <p className="font-medium">{workflow.metrics.avg_duration}</p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation()
                        onWorkflowClick?.(workflow.id)
                      }}
                    >
                      <Eye className="w-4 h-4" />
                      <span className="hidden xl:inline ml-1">Kitchen</span>
                    </Button>

                    <Button
                      size="sm"
                      className="bg-primary hover:bg-primary/90 text-primary-foreground"
                      onClick={(e) => {
                        e.stopPropagation()
                        console.log('🔥 COOK BUTTON CLICKED - Workflow:', workflow.id, 'Status:', workflow.current_execution?.status)
                        handleExecuteWorkflow(workflow.id)
                      }}
                      disabled={false}
                    >
                      <Play className="w-4 h-4" />
                      <span className="hidden xl:inline ml-1">Cook</span>
                    </Button>

                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="sm" onClick={(e) => e.stopPropagation()}>
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={(e) => {
                          e.stopPropagation()
                          handleDuplicateWorkflow(workflow.id, workflow.name)
                        }}>
                          <Copy className="w-4 h-4 mr-2" />
                          Duplicate
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation()
                            setWorkflowToDelete({ id: workflow.id, name: workflow.name })
                          }}
                          className="text-destructive focus:text-destructive focus:bg-destructive/10"
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      )}

      {/* Recipe Runs Section */}
      {hasRecipeRuns && (
        <>
          <div className="flex items-center gap-2 mt-6">
            <ChefHat className="w-5 h-5 text-primary" />
            <h3 className="text-base font-semibold">
              <span className="text-foreground">Playbook</span> <span className="text-primary">Runs</span>
            </h3>
            <Badge variant="outline" className="text-xs">
              {workflowsData.recipe_runs.length}
            </Badge>
          </div>
          <div className="space-y-3">
            {workflowsData.recipe_runs.map((run, index) => {
              const isRunning = run.status === 'running'
              const isFailed = run.status === 'failed'
              const isCompleted = run.status === 'completed'

              return (
                <motion.div
                  key={`recipe-${run.id}`}
                  className="glass-card card-glow p-4 hover:border-primary/20 transition-all duration-300 cursor-pointer"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  onClick={() => onRecipeRunClick?.(run)}
                >
                  <div className="flex items-center gap-4">
                    {/* Status Indicator */}
                    <div className={`flex items-center justify-center w-10 h-10 rounded-lg ${
                      isRunning ? 'bg-info/20' : isFailed ? 'bg-destructive/20' : isCompleted ? 'bg-success/20' : 'bg-secondary/50'
                    }`}>
                      {isRunning ? (
                        <Activity className="w-5 h-5 text-info animate-pulse" />
                      ) : isFailed ? (
                        <AlertTriangle className="w-5 h-5 text-destructive" />
                      ) : isCompleted ? (
                        <CheckCircle className="w-5 h-5 text-success" />
                      ) : (
                        <Clock className="w-5 h-5 text-muted-foreground" />
                      )}
                    </div>

                    {/* Recipe Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold text-base">{run.recipe_name}</h4>
                        <Badge className="text-xs bg-primary/20 text-primary border-primary/30">
                          <Zap className="w-3 h-3 mr-1" />
                          Recipe
                        </Badge>
                        <Badge className={`text-xs ${
                          isRunning
                            ? 'bg-info/20 text-info border-info/30'
                            : isFailed
                              ? 'bg-destructive/20 text-destructive border-destructive/30'
                              : isCompleted
                                ? 'bg-success/20 text-success border-success/30'
                                : 'bg-secondary/50 text-muted-foreground border-border/30'
                        }`}>
                          {isRunning ? 'cooking' : isFailed ? 'burnt' : isCompleted ? 'cooked' : run.status}
                        </Badge>
                      </div>
                      {/* Step progress */}
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>Step {run.current_step} of {run.total_steps}</span>
                        {run.error_message && (
                          <span className="text-destructive truncate max-w-[200px]">{run.error_message}</span>
                        )}
                      </div>
                    </div>

                    {/* Metrics */}
                    <div className="hidden lg:flex items-center gap-6 text-sm">
                      {run.started_at && (
                        <div className="text-left max-w-[200px]">
                          <div className="flex items-center gap-1 text-xs text-muted-foreground">
                            <Clock className="w-3 h-3" />
                            <span>{new Date(run.started_at).toLocaleString()}</span>
                          </div>
                        </div>
                      )}
                      <div className="text-center">
                        <p className="text-muted-foreground text-xs">Steps</p>
                        <p className="font-medium">{run.total_steps}</p>
                      </div>
                      {run.total_tokens > 0 && (
                        <div className="text-center">
                          <p className="text-muted-foreground text-xs">Tokens</p>
                          <p className="font-medium">{run.total_tokens.toLocaleString()}</p>
                        </div>
                      )}
                      {run.total_duration_ms > 0 && (
                        <div className="text-center">
                          <p className="text-muted-foreground text-xs">Duration</p>
                          <p className="font-medium">{(run.total_duration_ms / 1000).toFixed(1)}s</p>
                        </div>
                      )}
                    </div>

                    {/* View button */}
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation()
                        onRecipeRunClick?.(run)
                      }}
                    >
                      <Eye className="w-4 h-4" />
                      <span className="hidden xl:inline ml-1">View</span>
                    </Button>
                  </div>

                  {/* Step Results Preview (collapsed) */}
                  {run.step_results && run.step_results.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-border/30">
                      <div className="flex gap-2 flex-wrap">
                        {run.step_results.map((step: any, stepIdx: number) => (
                          <div
                            key={stepIdx}
                            className={`flex items-center gap-1 text-xs px-2 py-1 rounded-md ${
                              step.status === 'success'
                                ? 'bg-success/10 text-success'
                                : step.status === 'failed'
                                  ? 'bg-destructive/10 text-destructive'
                                  : step.status === 'running'
                                    ? 'bg-info/10 text-info'
                                    : 'bg-secondary/50 text-muted-foreground'
                            }`}
                          >
                            {step.status === 'success' ? (
                              <CheckCircle className="w-3 h-3" />
                            ) : step.status === 'failed' ? (
                              <AlertTriangle className="w-3 h-3" />
                            ) : step.status === 'running' ? (
                              <Activity className="w-3 h-3 animate-pulse" />
                            ) : (
                              <Clock className="w-3 h-3" />
                            )}
                            <span>{step.agent_name}</span>
                            {step.duration_ms > 0 && (
                              <span className="opacity-60">({(step.duration_ms / 1000).toFixed(1)}s)</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </>
      )}

      {/* Live Progress Panel */}
      {selectedWorkflow && (
        <LiveProgressPanel
          workflowId={selectedWorkflow.id}
          workflowName={selectedWorkflow.name}
          isOpen={!!selectedWorkflow}
          onClose={() => setSelectedWorkflow(null)}
        />
      )}

      {/* Remove Workflow Confirmation Dialog */}
      <Dialog open={!!workflowToDelete} onOpenChange={(open) => !open && setWorkflowToDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-destructive" />
              Remove from Kitchen
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to remove <span className="font-semibold text-foreground">"{workflowToDelete?.name}"</span> from the kitchen?
              <br /><br />
              This will permanently delete the workflow and all its execution history. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setWorkflowToDelete(null)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteWorkflow}
              disabled={isDeleting}
              className="bg-destructive hover:bg-destructive/80"
            >
              {isDeleting ? 'Removing...' : 'Remove from Kitchen'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Cleanup Old Workflows Dialog */}
      <Dialog open={showCleanupDialog} onOpenChange={setShowCleanupDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Trash2 className="w-5 h-5 text-primary" />
              Cleanup Old Workflows
            </DialogTitle>
            <DialogDescription>
              Select how old workflows should be before deletion. This will permanently delete workflows and their execution history.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Delete workflows older than:</label>
              <Select value={cleanupDays} onValueChange={setCleanupDays}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="7">7 days</SelectItem>
                  <SelectItem value="14">14 days</SelectItem>
                  <SelectItem value="30">30 days</SelectItem>
                  <SelectItem value="60">60 days</SelectItem>
                  <SelectItem value="90">90 days</SelectItem>
                  <SelectItem value="180">6 months</SelectItem>
                  <SelectItem value="365">1 year</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="rounded-md bg-warning/10 border border-warning/20 p-3">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-warning mt-0.5 flex-shrink-0" />
                <p className="text-sm text-warning/70">
                  This action cannot be undone. All workflows created before {new Date(Date.now() - parseInt(cleanupDays) * 24 * 60 * 60 * 1000).toLocaleDateString()} will be permanently deleted.
                </p>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowCleanupDialog(false)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleCleanupWorkflows}
              disabled={isDeleting}
              className="bg-destructive hover:bg-destructive/80"
            >
              {isDeleting ? 'Cleaning...' : 'Cleanup Workflows'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
