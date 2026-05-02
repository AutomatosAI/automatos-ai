'use client'

import { useState, useRef, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  Plus,
  GitBranch,
  Users,
  CheckCircle,
  ChevronRight,
  Edit,
  Trash2,
  AlertTriangle,
  Search,
  Filter,
  Clock,
  Star,
  Share2,
  Loader2,
  Lightbulb,
  Play,
  Bot,
  Zap,
  Wrench,
  BarChart3,
  ExternalLink
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Input } from '@/components/ui/input'
import { PremiumIcon, EmptyState } from '@/components/shared'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'
import {
  useWorkflowPlaybooks,
  useDeletePlaybook,
  useRecordPlaybookUsage,
  useExecutePlaybook,
  useSubmitPlaybookToMarketplace,
  usePlaybookSuggestions,
  usePlaybookExecutions
} from '@/hooks/use-playbook-api'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { useAgents } from '@/hooks/use-agent-api'
import { useToast } from '@/hooks/use-toast'
import { CreatePlaybookModal } from './create-playbook-modal'
import { ViewPlaybookModal } from './view-playbook-modal'
import { PlaybookSuggestionsPanel } from './playbook-suggestions-panel'
import { PlaybookRunDots } from './playbook-run-dots'

interface PlaybookExecutionInfo {
  recipeExecutionId: string
  recipeId: string
  recipeSteps: Array<{ step_id: string; order: number; prompt_template: string; agent_id: number }>
  recipeName: string
}

interface PlaybooksTabProps {
  searchTerm?: string
  viewMode?: 'grid' | 'list'
  externalCreateOpen?: boolean
  onCreateModalClosed?: () => void
  onUseRecipe: (recipe: any) => void
  onExecuteRecipe?: (workflowId: number, recipeExecInfo?: PlaybookExecutionInfo) => void
  emptyState?: React.ReactNode
}

// Generate a consistent color from agent_id
function agentColor(id: number): string {
  const colors = [
    'from-[hsl(var(--info))]/80 to-[hsl(var(--info))]/60',
    'from-[hsl(var(--agent))]/80 to-[hsl(var(--agent))]/60',
    'from-[hsl(var(--success))]/80 to-[hsl(var(--success))]/60',
    'from-primary/80 to-primary/60',
    'from-[hsl(var(--destructive))]/80 to-[hsl(var(--destructive))]/60',
    'from-[hsl(var(--info))]/80 to-[hsl(var(--info))]/60',
    'from-[hsl(var(--warning))]/80 to-[hsl(var(--warning))]/60',
    'from-[hsl(var(--destructive))]/80 to-[hsl(var(--destructive))]/60',
  ]
  return colors[(id || 0) % colors.length]
}

/** Small icon button linking to the Execution Kitchen for the latest run. */
function LatestRunLink({ recipeId }: { recipeId: string }) {
  const { data } = usePlaybookExecutions(recipeId, { limit: 1 })
  const executions: any[] = (data as any)?.items || (Array.isArray(data) ? data : [])
  const latest = executions[0]
  if (!latest) return null

  const execId = latest.execution_id || String(latest.id)
  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-8 w-8 p-0"
      title="View latest execution"
      onClick={(e) => {
        e.stopPropagation()
        window.location.href = `/activity/execution?id=${execId}&recipeId=${recipeId}`
      }}
    >
      <ExternalLink className="w-3.5 h-3.5" />
    </Button>
  )
}

export function PlaybooksTab({
  searchTerm: externalSearchTerm,
  viewMode: viewModeProp,
  externalCreateOpen,
  onCreateModalClosed,
  onUseRecipe,
  onExecuteRecipe,
  emptyState,
}: PlaybooksTabProps) {
  const [viewModeLocal] = useViewMode('wf-recipes')
  const viewMode = viewModeProp || viewModeLocal
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showViewModal, setShowViewModal] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [selectedPlaybook, setSelectedPlaybook] = useState<any>(null)
  const [sharingPlaybookId, setSharingPlaybookId] = useState<string | null>(null)
  const [cookingPlaybookId, setCookingPlaybookId] = useState<string | null>(null)
  const [editPlaybookData, setEditPlaybookData] = useState<any>(null)
  const [editPlaybookId, setEditPlaybookId] = useState<string | null>(null)
  const { data: iconMappings = {} } = useSystemIcons()
  const { data: agents = [] } = useAgents()

  // Build agent lookup by ID for playbook avatar icons
  const agentMap = useMemo(() => {
    const map = new Map<number, any>()
    for (const agent of agents as any[]) {
      map.set(Number(agent.id), agent)
    }
    return map
  }, [agents])

  // Sync external create modal open state
  useEffect(() => {
    if (externalCreateOpen) {
      setEditPlaybookData(null)
      setEditPlaybookId(null)
      setShowCreateModal(true)
    }
  }, [externalCreateOpen])

  // Fetch playbooks with filtering
  const { data: playbooksData, isLoading, error, refetch } = useWorkflowPlaybooks({
    search: externalSearchTerm || undefined,
    limit: 100
  })

  const deleteMutation = useDeletePlaybook()
  const recordUsageMutation = useRecordPlaybookUsage()
  const executePlaybookMutation = useExecutePlaybook()
  const submitToMarketplaceMutation = useSubmitPlaybookToMarketplace()
  const { toast } = useToast()
  const cookingRef = useRef(false)

  // Fetch suggestions for the selected playbook when viewing
  const { data: selectedPlaybookSuggestions } = usePlaybookSuggestions(
    showViewModal ? selectedPlaybook?.template_id || selectedPlaybook?.recipe_id : undefined
  )

  // Fetch recent executions for the selected playbook when viewing
  const selectedPlaybookId = showViewModal ? selectedPlaybook?.template_id || selectedPlaybook?.recipe_id : undefined
  const { data: selectedPlaybookExecutions, isLoading: executionsLoading } = usePlaybookExecutions(
    selectedPlaybookId,
    { limit: 5 }
  )

  const playbooks = (playbooksData as any)?.items || []

  const handleDeletePlaybook = async () => {
    if (!selectedPlaybook) return
    try {
      await deleteMutation.mutateAsync(selectedPlaybook.template_id || selectedPlaybook.id)
      setShowDeleteDialog(false)
      setSelectedPlaybook(null)
      refetch()
    } catch (error) {
      console.error('Error deleting playbook:', error)
    }
  }

  const handleRunPlaybook = async (playbook: any) => {
    if (cookingRef.current) return
    cookingRef.current = true
    const playbookId = playbook.template_id || playbook.id?.toString()
    setCookingPlaybookId(playbookId)
    try {
      const result: any = await executePlaybookMutation.mutateAsync({ playbookId: playbookId })
      toast({
        title: 'Playbook Started',
        description: `"${playbook.name}" is now running.`,
        variant: 'default',
      })
      if (onExecuteRecipe && result?.recipe_execution_id) {
        onExecuteRecipe(0, {
          recipeExecutionId: result.recipe_execution_id,
          recipeId: playbookId,
          recipeSteps: (playbook.steps || []).map((s: any, i: number) => ({
            step_id: s.step_id || `step-${i + 1}`,
            order: s.order || i + 1,
            prompt_template: s.prompt_template || '',
            agent_id: s.agent_id,
          })),
          recipeName: playbook.name,
        })
      }
    } catch (error: any) {
      toast({
        title: 'Execution Failed',
        description: error?.message || 'Failed to start playbook execution',
        variant: 'destructive',
      })
    } finally {
      setCookingPlaybookId(null)
      cookingRef.current = false
    }
  }

  const handleViewClick = (playbook: any) => {
    setSelectedPlaybook(playbook)
    setShowViewModal(true)
  }

  const handleEditClick = (playbook: any) => {
    const backendConfig = playbook.execution_config || {}
    const backendSchedule = playbook.schedule_config || {}

    const formSteps = (playbook.steps || []).map((step: any, idx: number) => ({
      step_id: step.step_id || `step-${idx + 1}`,
      order: step.order ?? idx + 1,
      agent_id: step.agent_id != null ? String(step.agent_id) : '',
      prompt_template: step.prompt_template || '',
      pass_to: step.pass_to,
      error_handling: step.error_handling || 'stop',
      pre_exec: step.pre_exec || '',
    }))

    const initialData = {
      name: playbook.name || '',
      description: playbook.description || '',
      inputs: typeof playbook.inputs === 'string' ? playbook.inputs : JSON.stringify(playbook.inputs || {}, null, 2),
      outputs: typeof playbook.outputs === 'string' ? playbook.outputs : JSON.stringify(playbook.outputs || {}, null, 2),
      steps: formSteps,
      execution_config: {
        mode: backendConfig.mode || 'sequential',
        max_retries: backendConfig.max_retries ?? 3,
        timeout_per_step: (backendConfig.per_step_timeout ?? 120) * 1000,
        total_timeout: (backendConfig.total_timeout ?? 600) * 1000,
        auto_learning: backendConfig.auto_learning ?? backendConfig.auto_learn ?? true,
        parallel_limit: backendConfig.parallel_limit ?? 5,
        memory_isolation: backendConfig.memory_isolation || 'shared',
      },
      schedule_config: {
        type: backendSchedule.type || 'manual',
        cron_expression: backendSchedule.cron_expression || '',
        trigger_config: backendSchedule.trigger_config || {},
        webhook_id: backendSchedule.webhook_id,
      },
    }

    setEditPlaybookData(initialData)
    setEditPlaybookId(playbook.template_id || playbook.id?.toString())
    setShowCreateModal(true)
  }

  const handleDeleteClick = (playbook: any) => {
    setSelectedPlaybook(playbook)
    setShowDeleteDialog(true)
  }

  const handleShareToMarketplace = async (playbook: any) => {
    setSharingPlaybookId(playbook.template_id || playbook.id?.toString())
    try {
      const result: any = await submitToMarketplaceMutation.mutateAsync({
        recipe_id: playbook.template_id,
        category: playbook.category,
        icon: playbook.icon
      })

      if (result.auto_approved) {
        toast({ title: 'Published to Marketplace', description: 'Your playbook is now live in the marketplace!', variant: 'default' })
      } else {
        toast({ title: 'Submitted for Approval', description: 'Your playbook has been submitted and is awaiting approval.', variant: 'default' })
      }
      refetch()
    } catch (error: any) {
      toast({ title: 'Submission Failed', description: error?.message || 'Failed to submit playbook to marketplace', variant: 'destructive' })
    } finally {
      setSharingPlaybookId(null)
    }
  }

  const closeCreateModal = () => {
    setShowCreateModal(false)
    setEditPlaybookData(null)
    setEditPlaybookId(null)
    onCreateModalClosed?.()
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading playbooks...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-[hsl(var(--destructive))] mx-auto mb-4" />
          <p className="text-[hsl(var(--destructive))]">Error loading playbooks</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Playbooks Grid — 4 per row */}
      {playbooks.length === 0 ? (
        emptyState || (
          <EmptyState
            icon={GitBranch}
            title="No playbooks found"
            description="Create your first playbook to get started"
          />
        )
      ) : viewMode === 'list' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {playbooks.map((playbook: any) => {
            const steps = playbook.steps || []
            const stepCount = steps.length
            const agentIds = [...new Set(steps.map((s: any) => s.agent_id).filter(Boolean))] as number[]
            const isRunning = cookingPlaybookId === (playbook.template_id || playbook.id?.toString())

            return (
              <Card
                key={playbook.id}
                data-testid="workflow-card"
                className="glass-card card-glow hover:border-primary/20 transition-all cursor-pointer"
                onClick={() => handleViewClick(playbook)}
              >
                <CardContent className="p-3">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-xl bg-primary/20 border border-primary/30 flex items-center justify-center shrink-0 text-lg">
                      {(() => {
                        const premiumIconName = iconMappings[playbook.marketplace_category] || iconMappings['global_recipe'] || null
                        return premiumIconName ? (
                          <PremiumIcon name={premiumIconName} size={20} className="text-primary" />
                        ) : (
                          playbook.icon || '🍳'
                        )
                      })()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-sm truncate">{playbook.name}</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                        <span>{stepCount} Steps</span>
                        <span>&middot;</span>
                        <span>{agentIds.length} Agents</span>
                        <span>&middot;</span>
                        <span>{playbook.use_count || 0} Runs</span>
                        <PlaybookRunDots
                          recipeId={playbook.template_id || String(playbook.id || '')}
                          compact
                          onClick={() => handleViewClick(playbook)}
                        />
                      </div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <LatestRunLink recipeId={playbook.template_id || String(playbook.id || '')} />
                      <Button
                        className="h-8 text-xs"
                        size="sm"
                        onClick={(e) => { e.stopPropagation(); handleRunPlaybook(playbook) }}
                        disabled={isRunning}
                      >
                        {isRunning ? (
                          <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                        ) : (
                          <Play className="w-3 h-3 mr-1" />
                        )}
                        Run
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {playbooks.map((playbook: any, index: number) => {
            const steps = playbook.steps || []
            const agentIds = [...new Set(steps.map((s: any) => s.agent_id).filter(Boolean))] as number[]
            const stepCount = steps.length
            const qualityScore = playbook.quality_score
            const qualityPct = qualityScore != null ? Math.round(qualityScore * 100) : null
            const tools = playbook.required_tools || []
            const isRunning = cookingPlaybookId === (playbook.template_id || playbook.id?.toString())
            const isSharing = sharingPlaybookId === (playbook.template_id || playbook.id?.toString())

            return (
              <motion.div
                key={playbook.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.05 }}
              >
                <Card data-testid="workflow-card" className="glass-card card-glow hover:border-primary/20 transition-all duration-300 h-full flex flex-col group">
                  {/* Card Header — icon, name, badges */}
                  <CardHeader className="pb-3">
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 rounded-xl bg-primary/20 border border-primary/30 flex items-center justify-center shrink-0 text-xl">
                        {(() => {
                          const premiumIconName = iconMappings[playbook.marketplace_category] || iconMappings['global_recipe'] || null
                          return premiumIconName ? (
                            <PremiumIcon name={premiumIconName} size={24} className="text-primary" />
                          ) : (
                            playbook.icon || '🍳'
                          )
                        })()}
                      </div>
                      <div className="flex-1 min-w-0">
                        <CardTitle className="text-sm font-semibold leading-tight truncate">{playbook.name}</CardTitle>
                        <p className="text-xs text-muted-foreground mt-1 line-clamp-2 leading-relaxed">
                          {playbook.description || 'No description'}
                        </p>
                      </div>
                    </div>

                    {/* Badges row */}
                    <div className="flex flex-wrap gap-1.5 mt-3">
                      {playbook.marketplace_category && (
                        <Badge variant="outline" className="text-[10px] h-5 bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20">
                          {playbook.marketplace_category}
                        </Badge>
                      )}
                      {playbook.is_system && (
                        <Badge variant="outline" className="text-[10px] h-5 bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/20">
                          System
                        </Badge>
                      )}
                      {playbook.learning_data?.latest_suggestions?.length > 0 && (
                        <Badge
                          variant="outline"
                          className="text-[10px] h-5 bg-primary/10 text-primary border-primary/20 cursor-pointer hover:bg-primary/20"
                          onClick={(e) => { e.stopPropagation(); handleViewClick(playbook) }}
                        >
                          <Lightbulb className="w-2.5 h-2.5 mr-0.5" />
                          {playbook.learning_data.latest_suggestions.length}
                        </Badge>
                      )}
                    </div>
                  </CardHeader>

                  <CardContent className="flex-1 flex flex-col justify-between pt-0 space-y-3">
                    {/* Quality score bar */}
                    {qualityPct != null && (
                      <div>
                        <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
                          <span>Quality Score</span>
                          <span className={qualityPct >= 80 ? 'text-[hsl(var(--success))]' : qualityPct >= 50 ? 'text-primary' : 'text-[hsl(var(--destructive))]'}>
                            {qualityPct}%
                          </span>
                        </div>
                        <div className="h-1.5 rounded-full bg-secondary/80 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${
                              qualityPct >= 80 ? 'bg-gradient-to-r from-[hsl(var(--success))] to-[hsl(var(--success))]/80' :
                              qualityPct >= 50 ? 'bg-gradient-to-r from-primary to-[hsl(var(--warning))]/80' :
                              'bg-gradient-to-r from-[hsl(var(--destructive))] to-[hsl(var(--destructive))]/80'
                            }`}
                            style={{ width: `${qualityPct}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Stats row */}
                    <div className="grid grid-cols-3 gap-2">
                      <div className="text-center">
                        <div className="text-lg font-bold text-foreground">{stepCount}</div>
                        <div className="text-[10px] text-muted-foreground">Steps</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-primary">{agentIds.length}</div>
                        <div className="text-[10px] text-muted-foreground">Agents</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-foreground">{playbook.use_count || 0}</div>
                        <div className="text-[10px] text-muted-foreground">Runs</div>
                      </div>
                    </div>

                    {/* Run history dots */}
                    <PlaybookRunDots
                      recipeId={playbook.template_id || String(playbook.id || '')}
                      useCount={playbook.use_count}
                      onClick={() => handleViewClick(playbook)}
                    />

                    {/* Agent avatars */}
                    {agentIds.length > 0 && (
                      <div className="flex items-center gap-1.5">
                        {agentIds.slice(0, 4).map((aid) => {
                          const agentInfo = agentMap.get(aid)
                          const agentIconName = agentInfo?.premium_icon
                            || iconMappings[agentInfo?.marketplace_category]
                            || iconMappings[agentInfo?.configuration?.category]
                            || iconMappings[agentInfo?.agent_type]
                            || null
                          return agentIconName ? (
                            <div
                              key={aid}
                              className="w-8 h-8 flex items-center justify-center shrink-0"
                              title={agentInfo?.name || `Agent ${aid}`}
                            >
                              <PremiumIcon name={agentIconName} size={24} />
                            </div>
                          ) : (
                            <div
                              key={aid}
                              className={`w-8 h-8 rounded-lg bg-gradient-to-br ${agentColor(aid)} flex items-center justify-center border border-border/10`}
                              title={agentInfo?.name || `Agent ${aid}`}
                            >
                              <Bot className="w-3.5 h-3.5 text-foreground" />
                            </div>
                          )
                        })}
                        {agentIds.length > 4 && (
                          <div className="w-8 h-8 rounded-lg bg-secondary/80 border border-border/10 flex items-center justify-center text-[10px] text-muted-foreground font-medium">
                            +{agentIds.length - 4}
                          </div>
                        )}
                        {/* Tool icons */}
                        {tools.length > 0 && (
                          <>
                            <div className="w-px h-5 bg-border/10 mx-1" />
                            {tools.slice(0, 3).map((tool: string, i: number) => (
                              <div
                                key={i}
                                className="w-8 h-8 rounded-lg bg-secondary/80 border border-border/10 flex items-center justify-center"
                                title={tool}
                              >
                                <Wrench className="w-3 h-3 text-muted-foreground" />
                              </div>
                            ))}
                            {tools.length > 3 && (
                              <span className="text-[10px] text-muted-foreground">+{tools.length - 3}</span>
                            )}
                          </>
                        )}
                      </div>
                    )}

                    {/* Action buttons */}
                    <Separator />
                    <div className="flex items-center justify-between">
                      <Button variant="ghost" size="sm" onClick={() => handleViewClick(playbook)}
                        className="text-muted-foreground hover:text-foreground p-0 h-auto">
                        Details
                      </Button>
                      <div className="flex items-center gap-1">
                        <LatestRunLink recipeId={playbook.template_id || String(playbook.id || '')} />
                        {!playbook.is_system && (
                          <>
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={() => handleEditClick(playbook)} title="Edit">
                              <Edit className="w-3.5 h-3.5" />
                            </Button>
                            <Button
                              variant="ghost" size="sm" className="h-8 w-8 p-0 text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))]/80"
                              onClick={() => handleDeleteClick(playbook)}
                              title="Delete"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </Button>
                          </>
                        )}
                        <Button
                          size="sm"
                          variant="outline"
                          className="w-24"
                          onClick={() => handleRunPlaybook(playbook)}
                          disabled={isRunning}
                        >
                          {isRunning ? (
                            <Loader2 className="w-3 h-3 mr-1.5 animate-spin" />
                          ) : (
                            <Play className="w-3 h-3 mr-1.5" />
                          )}
                          {isRunning ? 'Starting...' : 'Run'}
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </div>
      )}

      {/* Create/Edit Playbook Modal (4-step wizard) */}

      <CreatePlaybookModal
        open={showCreateModal}
        onClose={closeCreateModal}
        onSave={() => {
          closeCreateModal()
          refetch()
        }}
        initialData={editPlaybookData}
        recipeId={editPlaybookId || undefined}
      />

      {/* View Playbook Modal */}
      <ViewPlaybookModal
        open={showViewModal}
        onClose={() => {
          setShowViewModal(false)
          setSelectedPlaybook(null)
        }}
        recipe={selectedPlaybook}
        suggestions={selectedPlaybookSuggestions}
        executions={(selectedPlaybookExecutions as any)?.items || (selectedPlaybookExecutions as any) || []}
        executionsLoading={executionsLoading}
        onEdit={() => {
          setShowViewModal(false)
          handleEditClick(selectedPlaybook)
        }}
        onExecute={() => {
          setShowViewModal(false)
          handleRunPlaybook(selectedPlaybook)
        }}
        onShare={!selectedPlaybook?.is_system && !selectedPlaybook?.is_marketplace_item
          ? () => handleShareToMarketplace(selectedPlaybook)
          : undefined
        }
        onViewExecution={(executionId) => {
          if (!selectedPlaybook) return
          setShowViewModal(false)
          onExecuteRecipe?.(0, {
            recipeExecutionId: executionId,
            recipeId: selectedPlaybook.template_id,
            recipeSteps: (selectedPlaybook.steps || []).map((s: any) => ({
              step_id: s.step_id,
              order: s.order,
              prompt_template: s.prompt_template || '',
              agent_id: s.agent_id ?? 0,
            })),
            recipeName: selectedPlaybook.name,
          })
        }}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Playbook?</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &quot;{selectedPlaybook?.name}&quot;? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>

          <div className="flex justify-end gap-4 mt-6">
            <Button
              variant="outline"
              onClick={() => {
                setShowDeleteDialog(false)
                setSelectedPlaybook(null)
              }}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeletePlaybook}
              disabled={deleteMutation.isLoading}
            >
              {deleteMutation.isLoading ? 'Deleting...' : 'Delete Playbook'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
