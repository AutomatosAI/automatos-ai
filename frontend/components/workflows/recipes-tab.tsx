'use client'

import { useState, useRef } from 'react'
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
  Play
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  useWorkflowRecipes,
  useDeleteRecipe,
  useRecordRecipeUsage,
  useExecuteRecipe,
  useSubmitRecipeToMarketplace,
  useRecipeSuggestions
} from '@/hooks/use-recipe-api'
import { useToast } from '@/hooks/use-toast'
import { CreateRecipeModal } from './create-recipe-modal'
import { ViewRecipeModal } from './view-recipe-modal'
import { RecipeSuggestionsPanel } from './recipe-suggestions-panel'

interface RecipesTabProps {
  onUseRecipe: (recipe: any) => void
  onExecuteRecipe?: (workflowId: number) => void
  onOpenCreateModal?: () => void
}

export function RecipesTab({ onUseRecipe, onExecuteRecipe, onOpenCreateModal }: RecipesTabProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showViewModal, setShowViewModal] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [selectedRecipe, setSelectedRecipe] = useState<any>(null)
  const [sharingRecipeId, setSharingRecipeId] = useState<string | null>(null)
  const [cookingRecipeId, setCookingRecipeId] = useState<string | null>(null)
  const [editRecipeData, setEditRecipeData] = useState<any>(null)
  const [editRecipeId, setEditRecipeId] = useState<string | null>(null)

  // Fetch recipes with filtering
  const { data: recipesData, isLoading, error, refetch } = useWorkflowRecipes({
    category: selectedCategory !== 'all' ? selectedCategory : undefined,
    difficulty: selectedDifficulty !== 'all' ? selectedDifficulty : undefined,
    search: searchTerm || undefined,
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

  const handleUseRecipe = async (recipe: any) => {
    try {
      // Record usage
      await recordUsageMutation.mutateAsync(recipe.template_id || recipe.id)
      // Notify parent component with full recipe object
      onUseRecipe(recipe)
    } catch (error) {
      console.error('Error recording recipe usage:', error)
      // Still pass recipe even if recording fails
      onUseRecipe(recipe)
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
      // Navigate to ExecutionKitchen via parent
      if (onExecuteRecipe && result?.workflow_id) {
        onExecuteRecipe(result.workflow_id)
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
    // Transform recipe data to CreateRecipeModal form format
    // Backend stores different field names than the form expects
    const backendConfig = recipe.execution_config || {}
    const backendSchedule = recipe.schedule_config || {}

    // Map steps: backend stores agent_id as integer, form expects string
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
        // Backend stores seconds as per_step_timeout, form expects ms as timeout_per_step
        timeout_per_step: (backendConfig.per_step_timeout ?? 120) * 1000,
        total_timeout: (backendConfig.total_timeout ?? 600) * 1000,
        quality_threshold: backendConfig.quality_threshold ?? 0.7,
        // Backend stores as auto_learn, form expects auto_learning
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
        toast({
          title: 'Published to Marketplace',
          description: 'Your recipe is now live in the marketplace!',
          variant: 'default'
        })
      } else {
        toast({
          title: 'Submitted for Approval',
          description: 'Your recipe has been submitted and is awaiting approval.',
          variant: 'default'
        })
      }
      refetch()
    } catch (error: any) {
      toast({
        title: 'Submission Failed',
        description: error?.message || 'Failed to submit recipe to marketplace',
        variant: 'destructive'
      })
    } finally {
      setSharingRecipeId(null)
    }
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
          <AlertTriangle className="h-8 w-8 text-red-400 mx-auto mb-4" />
          <p className="text-red-400">Error loading recipes</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="flex-1 flex gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search recipes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 bg-secondary/50 border-secondary"
            />
          </div>
          
          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger className="w-[180px] bg-secondary/50 border-secondary">
              <SelectValue placeholder="Category" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              <SelectItem value="Support">Support</SelectItem>
              <SelectItem value="Data Processing">Data Processing</SelectItem>
              <SelectItem value="Content">Content</SelectItem>
              <SelectItem value="Security">Security</SelectItem>
              <SelectItem value="Development">Development</SelectItem>
            </SelectContent>
          </Select>

          <Select value={selectedDifficulty} onValueChange={setSelectedDifficulty}>
            <SelectTrigger className="w-[180px] bg-secondary/50 border-secondary">
              <SelectValue placeholder="Difficulty" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Levels</SelectItem>
              <SelectItem value="beginner">Beginner</SelectItem>
              <SelectItem value="intermediate">Intermediate</SelectItem>
              <SelectItem value="advanced">Advanced</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button
            onClick={() => setShowCreateModal(true)}
            className="bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
          >
            <Plus className="w-4 h-4 mr-2" />
            Create Recipe
          </Button>
        </motion.div>
      </div>

      {/* Recipes Grid */}
      {recipes.length === 0 ? (
        <div className="text-center py-12">
          <GitBranch className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">No recipes found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {recipes.map((recipe: any, index: number) => (
            <motion.div
              key={recipe.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card className="glass-card card-glow hover:border-primary/30 transition-all duration-300 h-full flex flex-col">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-2xl">{recipe.icon || '📋'}</span>
                        <CardTitle className="text-lg">{recipe.name}</CardTitle>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {recipe.description}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        {recipe.marketplace_category && (
                          <Badge variant="outline" className="text-xs">
                            {recipe.marketplace_category}
                          </Badge>
                        )}
                        {recipe.is_system && (
                          <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                            System
                          </Badge>
                        )}
                        {recipe.learning_data?.suggestions?.length > 0 && (
                          <Badge
                            variant="outline"
                            className="text-xs bg-orange-500/10 text-orange-400 border-orange-500/20 cursor-pointer hover:bg-orange-500/20 transition-colors"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleViewClick(recipe)
                            }}
                          >
                            <Lightbulb className="w-3 h-3 mr-1" />
                            {recipe.learning_data.suggestions.length} suggestion{recipe.learning_data.suggestions.length !== 1 ? 's' : ''}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col justify-between space-y-4">
                  {/* Recipe Stats */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Used {recipe.use_count} times</span>
                      {recipe.quality_score && (
                        <span>Quality: {(recipe.quality_score * 100).toFixed(0)}%</span>
                      )}
                    </div>
                    
                    {/* Recommended Agents */}
                    {recipe.recommended_agents && recipe.recommended_agents.length > 0 && (
                      <div>
                        <div className="text-xs text-muted-foreground mb-1 flex items-center">
                          <Users className="w-3 h-3 mr-1" />
                          Recommended Agents
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {recipe.recommended_agents.slice(0, 3).map((agent: string, idx: number) => (
                            <Badge key={idx} variant="outline" className="text-xs">
                              {agent}
                            </Badge>
                          ))}
                          {recipe.recommended_agents.length > 3 && (
                            <span className="text-xs text-muted-foreground">
                              +{recipe.recommended_agents.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    <Button
                      className="flex-1 bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
                      size="sm"
                      onClick={() => handleCookRecipe(recipe)}
                      disabled={cookingRecipeId === (recipe.template_id || recipe.id?.toString())}
                    >
                      {cookingRecipeId === (recipe.template_id || recipe.id?.toString()) ? (
                        <Loader2 className="w-3 h-3 mr-2 animate-spin" />
                      ) : (
                        <Play className="w-3 h-3 mr-2" />
                      )}
                      {cookingRecipeId === (recipe.template_id || recipe.id?.toString()) ? 'Starting...' : 'Cook'}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleViewClick(recipe)}
                    >
                      <Eye className="w-3 h-3" />
                    </Button>
                    {!recipe.is_system && (
                      <>
                        {!recipe.is_marketplace_item && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleShareToMarketplace(recipe)}
                            disabled={sharingRecipeId === (recipe.template_id || recipe.id?.toString())}
                            title="Share to Marketplace"
                          >
                            {sharingRecipeId === (recipe.template_id || recipe.id?.toString()) ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <Share2 className="w-3 h-3" />
                            )}
                          </Button>
                        )}
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleEditClick(recipe)}
                        >
                          <Edit className="w-3 h-3" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleDeleteClick(recipe)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      )}

      {/* Create/Edit Recipe Modal (4-step wizard) */}
      <CreateRecipeModal
        open={showCreateModal}
        onClose={() => {
          setShowCreateModal(false)
          setEditRecipeData(null)
          setEditRecipeId(null)
        }}
        onSave={() => {
          setShowCreateModal(false)
          setEditRecipeData(null)
          setEditRecipeId(null)
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
      />

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Recipe?</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{selectedRecipe?.name}"? This action cannot be undone.
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


