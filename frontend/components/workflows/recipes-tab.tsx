'use client'

import { useState, useRef, useEffect } from 'react'
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
  Eye,
  Clock,
  Star,
  Share2,
  Loader2,
  Lightbulb,
  Play,
  Bot,
  Zap,
  Wrench,
  BarChart3
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'
import {
  useWorkflowRecipes,
  useDeleteRecipe,
  useRecordRecipeUsage,
  useExecuteRecipe,
  useSubmitRecipeToMarketplace,
  useRecipeSuggestions,
  useRecipeExecutions
} from '@/hooks/use-recipe-api'
import { useToast } from '@/hooks/use-toast'
import { CreateRecipeModal } from './create-recipe-modal'
import { ViewRecipeModal } from './view-recipe-modal'
import { RecipeSuggestionsPanel } from './recipe-suggestions-panel'

interface RecipeExecutionInfo {
  recipeExecutionId: string
  recipeId: string
  recipeSteps: Array<{ step_id: string; order: number; prompt_template: string; agent_id: number }>
  recipeName: string
}

interface RecipesTabProps {
  searchTerm?: string
  externalCreateOpen?: boolean
  onCreateModalClosed?: () => void
  onUseRecipe: (recipe: any) => void
  onExecuteRecipe?: (workflowId: number, recipeExecInfo?: RecipeExecutionInfo) => void
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

export function RecipesTab({
  searchTerm: externalSearchTerm,
  externalCreateOpen,
  onCreateModalClosed,
  onUseRecipe,
  onExecuteRecipe,
}: RecipesTabProps) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showViewModal, setShowViewModal] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [selectedRecipe, setSelectedRecipe] = useState<any>(null)
  const [sharingRecipeId, setSharingRecipeId] = useState<string | null>(null)
  const [cookingRecipeId, setCookingRecipeId] = useState<string | null>(null)
  const [editRecipeData, setEditRecipeData] = useState<any>(null)
  const [editRecipeId, setEditRecipeId] = useState<string | null>(null)

  // Sync external create modal open state
  useEffect(() => {
    if (externalCreateOpen) {
      setEditRecipeData(null)
      setEditRecipeId(null)
      setShowCreateModal(true)
    }
  }, [externalCreateOpen])

  // Fetch recipes with filtering
  const { data: recipesData, isLoading, error, refetch } = useWorkflowRecipes({
    search: externalSearchTerm || undefined,
    limit: 100
  })

  const deleteMutation = useDeleteRecipe()
  const recordUsageMutation = useRecordRecipeUsage()
  const executeRecipeMutation = useExecuteRecipe()
  const submitToMarketplaceMutation = useSubmitRecipeToMarketplace()
  const { toast } = useToast()
  const cookingRef = useRef(false)

  // Fetch suggestions for the selected recipe when viewing
  const { data: selectedRecipeSuggestions } = useRecipeSuggestions(
    showViewModal ? selectedRecipe?.template_id || selectedRecipe?.recipe_id : undefined
  )

  // Fetch recent executions for the selected recipe when viewing
  const selectedRecipeId = showViewModal ? selectedRecipe?.template_id || selectedRecipe?.recipe_id : undefined
  const { data: selectedRecipeExecutions, isLoading: executionsLoading } = useRecipeExecutions(
    selectedRecipeId,
    { limit: 5 }
  )

  const recipes = (recipesData as any)?.items || []

  const handleDeleteRecipe = async () => {
    if (!selectedRecipe) return
    try {
      await deleteMutation.mutateAsync(selectedRecipe.template_id || selectedRecipe.id)
      setShowDeleteDialog(false)
      setSelectedRecipe(null)
      refetch()
    } catch (error) {
      console.error('Error deleting recipe:', error)
    }
  }

  const handleCookRecipe = async (recipe: any) => {
    if (cookingRef.current) return
    cookingRef.current = true
    const recipeId = recipe.template_id || recipe.id?.toString()
    setCookingRecipeId(recipeId)
    try {
      const result: any = await executeRecipeMutation.mutateAsync({ recipeId })
      toast({
        title: 'Recipe Started',
        description: `"${recipe.name}" is now cooking.`,
        variant: 'default',
      })
      if (onExecuteRecipe && result?.recipe_execution_id) {
        onExecuteRecipe(0, {
          recipeExecutionId: result.recipe_execution_id,
          recipeId: recipeId,
          recipeSteps: (recipe.steps || []).map((s: any, i: number) => ({
            step_id: s.step_id || `step-${i + 1}`,
            order: s.order || i + 1,
            prompt_template: s.prompt_template || '',
            agent_id: s.agent_id,
          })),
          recipeName: recipe.name,
        })
      }
    } catch (error: any) {
      toast({
        title: 'Execution Failed',
        description: error?.message || 'Failed to start recipe execution',
        variant: 'destructive',
      })
    } finally {
      setCookingRecipeId(null)
      cookingRef.current = false
    }
  }

  const handleViewClick = (recipe: any) => {
    setSelectedRecipe(recipe)
    setShowViewModal(true)
  }

  const handleEditClick = (recipe: any) => {
    const backendConfig = recipe.execution_config || {}
    const backendSchedule = recipe.schedule_config || {}

    const formSteps = (recipe.steps || []).map((step: any, idx: number) => ({
      step_id: step.step_id || `step-${idx + 1}`,
      order: step.order ?? idx + 1,
      agent_id: step.agent_id != null ? String(step.agent_id) : '',
      prompt_template: step.prompt_template || '',
      pass_to: step.pass_to,
      error_handling: step.error_handling || 'stop',
    }))

    const initialData = {
      name: recipe.name || '',
      description: recipe.description || '',
      inputs: typeof recipe.inputs === 'string' ? recipe.inputs : JSON.stringify(recipe.inputs || {}, null, 2),
      outputs: typeof recipe.outputs === 'string' ? recipe.outputs : JSON.stringify(recipe.outputs || {}, null, 2),
      steps: formSteps,
      execution_config: {
        mode: backendConfig.mode || 'sequential',
        max_retries: backendConfig.max_retries ?? 3,
        retry_delay: backendConfig.retry_delay ?? 1000,
        backoff_strategy: backendConfig.backoff_strategy || 'exponential',
        timeout_per_step: (backendConfig.per_step_timeout ?? 120) * 1000,
        total_timeout: (backendConfig.total_timeout ?? 600) * 1000,
        quality_threshold: backendConfig.quality_threshold ?? 0.7,
        auto_learning: backendConfig.auto_learn ?? true,
        parallel_limit: backendConfig.parallel_limit ?? 5,
        memory_isolation: backendConfig.memory_isolation || 'shared',
      },
      schedule_config: {
        type: backendSchedule.type || 'manual',
        cron_expression: backendSchedule.cron_expression || '',
        trigger_config: backendSchedule.trigger_config || {},
      },
    }

    setEditRecipeData(initialData)
    setEditRecipeId(recipe.template_id || recipe.id?.toString())
    setShowCreateModal(true)
  }

  const handleDeleteClick = (recipe: any) => {
    setSelectedRecipe(recipe)
    setShowDeleteDialog(true)
  }

  const handleShareToMarketplace = async (recipe: any) => {
    setSharingRecipeId(recipe.template_id || recipe.id?.toString())
    try {
      const result: any = await submitToMarketplaceMutation.mutateAsync({
        recipe_id: recipe.template_id,
        category: recipe.category,
        icon: recipe.icon
      })

      if (result.auto_approved) {
        toast({ title: 'Published to Marketplace', description: 'Your recipe is now live in the marketplace!', variant: 'default' })
      } else {
        toast({ title: 'Submitted for Approval', description: 'Your recipe has been submitted and is awaiting approval.', variant: 'default' })
      }
      refetch()
    } catch (error: any) {
      toast({ title: 'Submission Failed', description: error?.message || 'Failed to submit recipe to marketplace', variant: 'destructive' })
    } finally {
      setSharingRecipeId(null)
    }
  }

  const closeCreateModal = () => {
    setShowCreateModal(false)
    setEditRecipeData(null)
    setEditRecipeId(null)
    onCreateModalClosed?.()
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading recipes...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-[hsl(var(--destructive))] mx-auto mb-4" />
          <p className="text-[hsl(var(--destructive))]">Error loading recipes</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Recipes Grid — 4 per row */}
      {recipes.length === 0 ? (
        <div className="text-center py-12">
          <GitBranch className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">No recipes found</p>
          <p className="text-xs text-muted-foreground mt-1">Create your first recipe to get started</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
          {recipes.map((recipe: any, index: number) => {
            const steps = recipe.steps || []
            const agentIds = [...new Set(steps.map((s: any) => s.agent_id).filter(Boolean))] as number[]
            const stepCount = steps.length
            const qualityScore = recipe.quality_score
            const qualityPct = qualityScore != null ? Math.round(qualityScore * 100) : null
            const tools = recipe.required_tools || []
            const isCooking = cookingRecipeId === (recipe.template_id || recipe.id?.toString())
            const isSharing = sharingRecipeId === (recipe.template_id || recipe.id?.toString())

            return (
              <motion.div
                key={recipe.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.05 }}
              >
                <Card className="glass-card hover:border-primary/30 transition-all duration-300 h-full flex flex-col group">
                  {/* Card Header — icon, name, badges */}
                  <CardHeader className="pb-3">
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 rounded-xl bg-primary/20 border border-primary/30 flex items-center justify-center shrink-0 text-xl">
                        {recipe.icon || '🍳'}
                      </div>
                      <div className="flex-1 min-w-0">
                        <CardTitle className="text-sm font-semibold leading-tight truncate">{recipe.name}</CardTitle>
                        <p className="text-xs text-muted-foreground mt-1 line-clamp-2 leading-relaxed">
                          {recipe.description || 'No description'}
                        </p>
                      </div>
                    </div>

                    {/* Badges row */}
                    <div className="flex flex-wrap gap-1.5 mt-3">
                      {recipe.marketplace_category && (
                        <Badge variant="outline" className="text-[10px] h-5 bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20">
                          {recipe.marketplace_category}
                        </Badge>
                      )}
                      {recipe.is_system && (
                        <Badge variant="outline" className="text-[10px] h-5 bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/20">
                          System
                        </Badge>
                      )}
                      {recipe.learning_data?.latest_suggestions?.length > 0 && (
                        <Badge
                          variant="outline"
                          className="text-[10px] h-5 bg-primary/10 text-primary border-primary/20 cursor-pointer hover:bg-primary/20"
                          onClick={(e) => { e.stopPropagation(); handleViewClick(recipe) }}
                        >
                          <Lightbulb className="w-2.5 h-2.5 mr-0.5" />
                          {recipe.learning_data.latest_suggestions.length}
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
                            className={`h-full rounded-full transition-all duration-500 ${
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
                        <div className="text-lg font-bold text-foreground">{recipe.use_count || 0}</div>
                        <div className="text-[10px] text-muted-foreground">Runs</div>
                      </div>
                    </div>

                    {/* Agent avatars */}
                    {agentIds.length > 0 && (
                      <div className="flex items-center gap-1.5">
                        {agentIds.slice(0, 4).map((aid) => {
                          const step = steps.find((s: any) => s.agent_id === aid)
                          return (
                            <div
                              key={aid}
                              className={`w-7 h-7 rounded-lg bg-gradient-to-br ${agentColor(aid)} flex items-center justify-center border border-border/10`}
                              title={`Agent ${aid}`}
                            >
                              <Bot className="w-3.5 h-3.5 text-foreground" />
                            </div>
                          )
                        })}
                        {agentIds.length > 4 && (
                          <div className="w-7 h-7 rounded-lg bg-secondary/80 border border-border/10 flex items-center justify-center text-[10px] text-muted-foreground font-medium">
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
                                className="w-7 h-7 rounded-lg bg-secondary/80 border border-border/10 flex items-center justify-center"
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
                    <div className="flex gap-1.5 pt-1 border-t border-border/5">
                      <Button
                        className="flex-1 bg-primary/90 hover:bg-primary/80 text-primary-foreground text-xs h-8 transition-all duration-200"
                        size="sm"
                        onClick={() => handleCookRecipe(recipe)}
                        disabled={isCooking}
                      >
                        {isCooking ? (
                          <Loader2 className="w-3 h-3 mr-1.5 animate-spin" />
                        ) : (
                          <Play className="w-3 h-3 mr-1.5" />
                        )}
                        {isCooking ? 'Starting...' : 'Cook'}
                      </Button>
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={() => handleViewClick(recipe)} title="View">
                        <Eye className="w-3.5 h-3.5" />
                      </Button>
                      {!recipe.is_system && (
                        <>
                          {!recipe.is_marketplace_item && (
                            <Button
                              variant="ghost" size="sm" className="h-8 w-8 p-0"
                              onClick={() => handleShareToMarketplace(recipe)}
                              disabled={isSharing}
                              title="Share"
                            >
                              {isSharing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Share2 className="w-3.5 h-3.5" />}
                            </Button>
                          )}
                          <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={() => handleEditClick(recipe)} title="Edit">
                            <Edit className="w-3.5 h-3.5" />
                          </Button>
                          <Button
                            variant="ghost" size="sm" className="h-8 w-8 p-0 text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))]/80"
                            onClick={() => handleDeleteClick(recipe)}
                            title="Delete"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </div>
      )}

      {/* Create/Edit Recipe Modal (4-step wizard) */}
      <CreateRecipeModal
        open={showCreateModal}
        onClose={closeCreateModal}
        onSave={() => {
          closeCreateModal()
          refetch()
        }}
        initialData={editRecipeData}
        recipeId={editRecipeId || undefined}
      />

      {/* View Recipe Modal */}
      <ViewRecipeModal
        open={showViewModal}
        onClose={() => {
          setShowViewModal(false)
          setSelectedRecipe(null)
        }}
        recipe={selectedRecipe}
        suggestions={selectedRecipeSuggestions}
        executions={(selectedRecipeExecutions as any)?.items || (selectedRecipeExecutions as any) || []}
        executionsLoading={executionsLoading}
        onEdit={() => {
          setShowViewModal(false)
          handleEditClick(selectedRecipe)
        }}
        onExecute={() => {
          setShowViewModal(false)
          handleCookRecipe(selectedRecipe)
        }}
        onShare={!selectedRecipe?.is_system && !selectedRecipe?.is_marketplace_item
          ? () => handleShareToMarketplace(selectedRecipe)
          : undefined
        }
        onViewExecution={(executionId) => {
          if (!selectedRecipe) return
          setShowViewModal(false)
          onExecuteRecipe?.(0, {
            recipeExecutionId: executionId,
            recipeId: selectedRecipe.template_id,
            recipeSteps: (selectedRecipe.steps || []).map((s: any) => ({
              step_id: s.step_id,
              order: s.order,
              prompt_template: s.prompt_template || '',
              agent_id: s.agent_id ?? 0,
            })),
            recipeName: selectedRecipe.name,
          })
        }}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Recipe?</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &quot;{selectedRecipe?.name}&quot;? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>

          <div className="flex justify-end gap-4 mt-6">
            <Button
              variant="outline"
              onClick={() => {
                setShowDeleteDialog(false)
                setSelectedRecipe(null)
              }}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteRecipe}
              disabled={deleteMutation.isLoading}
            >
              {deleteMutation.isLoading ? 'Deleting...' : 'Delete Recipe'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
