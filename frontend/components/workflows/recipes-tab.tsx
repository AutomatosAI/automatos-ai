'use client'

import { useState } from 'react'
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
  Share2
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  useWorkflowRecipes,
  useCreateRecipe,
  useUpdateRecipe,
  useDeleteRecipe,
  useRecordRecipeUsage,
  useSubmitRecipeToMarketplace
} from '@/hooks/use-recipe-api'
import { useToast } from '@/hooks/use-toast'

interface RecipesTabProps {
  onUseRecipe: (recipe: any) => void
  onOpenCreateModal?: () => void
}

export function RecipesTab({ onUseRecipe, onOpenCreateModal }: RecipesTabProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [showViewModal, setShowViewModal] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [selectedRecipe, setSelectedRecipe] = useState<any>(null)
  const [recipeForm, setRecipeForm] = useState<any>({
    recipe_id: '',
    name: '',
    description: '',
    goal: '',  // NEW: Workflow goal
    context: {},  // NEW: Workflow context
    category: 'Development',
    difficulty: 'intermediate',
    tags: [],
    icon: '📋',
    recipe_definition: {
      steps: [],
      agents: [],
      config: {}
    },
    recommended_agents: [],
    estimated_time: '',
    is_public: true,
    is_featured: false
  })

  // Fetch recipes with filtering
  const { data: recipesData, isLoading, error, refetch } = useWorkflowRecipes({
    category: selectedCategory !== 'all' ? selectedCategory : undefined,
    difficulty: selectedDifficulty !== 'all' ? selectedDifficulty : undefined,
    search: searchTerm || undefined,
    limit: 100
  })

  const createMutation = useCreateRecipe()
  const updateMutation = useUpdateRecipe()
  const deleteMutation = useDeleteRecipe()
  const recordUsageMutation = useRecordRecipeUsage()
  const submitToMarketplaceMutation = useSubmitRecipeToMarketplace()
  const { toast } = useToast()

  const recipes = recipesData?.items || []

  const handleCreateRecipe = async () => {
    try {
      await createMutation.mutateAsync({
        ...recipeForm,
        created_by: 'user'
      })
      setShowCreateModal(false)
      resetForm()
      refetch()
    } catch (error) {
      console.error('Error creating recipe:', error)
    }
  }

  const handleUpdateRecipe = async () => {
    if (!selectedRecipe) return
    try {
      await updateMutation.mutateAsync({
        recipeId: selectedRecipe.recipe_id,
        recipeData: recipeForm
      })
      setShowEditModal(false)
      setSelectedRecipe(null)
      resetForm()
      refetch()
    } catch (error) {
      console.error('Error updating recipe:', error)
    }
  }

  const handleDeleteRecipe = async () => {
    if (!selectedRecipe) return
    try {
      await deleteMutation.mutateAsync(selectedRecipe.recipe_id)
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
      await recordUsageMutation.mutateAsync(recipe.recipe_id)
      // Notify parent component with full recipe object
      onUseRecipe(recipe)
    } catch (error) {
      console.error('Error recording recipe usage:', error)
      // Still pass recipe even if recording fails
      onUseRecipe(recipe)
    }
  }

  const handleViewClick = (recipe: any) => {
    setSelectedRecipe(recipe)
    setShowViewModal(true)
  }

  const handleEditClick = (recipe: any) => {
    setSelectedRecipe(recipe)
    setRecipeForm({
      recipe_id: recipe.recipe_id,
      name: recipe.name,
      description: recipe.description,
      category: recipe.category,
      difficulty: recipe.difficulty,
      tags: recipe.tags || [],
      icon: recipe.icon || '📋',
      recipe_definition: recipe.recipe_definition || { steps: [], agents: [], config: {} },
      recommended_agents: recipe.recommended_agents || [],
      estimated_time: recipe.estimated_time || '',
      is_public: recipe.is_public,
      is_featured: recipe.is_featured
    })
    setShowEditModal(true)
  }

  const handleDeleteClick = (recipe: any) => {
    setSelectedRecipe(recipe)
    setShowDeleteDialog(true)
  }

  const handleShareToMarketplace = async (recipe: any) => {
    try {
      const result = await submitToMarketplaceMutation.mutateAsync({
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
    } catch (error: any) {
      toast({
        title: 'Submission Failed',
        description: error?.message || 'Failed to submit recipe to marketplace',
        variant: 'destructive'
      })
    }
  }

  const resetForm = () => {
    setRecipeForm({
      recipe_id: '',
      name: '',
      description: '',
      category: 'Development',
      difficulty: 'intermediate',
      tags: [],
      icon: '📋',
      recipe_definition: {
        steps: [],
        agents: [],
        config: {}
      },
      recommended_agents: [],
      estimated_time: '',
      is_public: true,
      is_featured: false
    })
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

        <Button 
          onClick={() => setShowCreateModal(true)}
          className="bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
        >
          <Plus className="w-4 h-4 mr-2" />
          Create Recipe
        </Button>
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
                        <Badge variant="outline" className="text-xs">
                          {recipe.category}
                        </Badge>
                        <Badge variant="outline" className="text-xs capitalize">
                          {recipe.difficulty}
                        </Badge>
                        {recipe.is_system && (
                          <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                            System
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
                      {recipe.estimated_time && <span>{recipe.estimated_time}</span>}
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
                      onClick={() => handleUseRecipe(recipe)}
                    >
                      <Plus className="w-3 h-3 mr-2" />
                      Use
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

      {/* Create Recipe Modal */}
      <Dialog open={showCreateModal} onOpenChange={setShowCreateModal}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Create Workflow Recipe</DialogTitle>
            <DialogDescription>
              Create a reusable recipe that others can use to quickly start workflows
            </DialogDescription>
          </DialogHeader>
          
          <RecipeForm
            form={recipeForm}
            onChange={setRecipeForm}
            onSubmit={handleCreateRecipe}
            onCancel={() => {
              setShowCreateModal(false)
              resetForm()
            }}
            isLoading={createMutation.isPending}
            submitLabel="Create Recipe"
          />
        </DialogContent>
      </Dialog>

      {/* View Recipe Modal */}
      <Dialog open={showViewModal} onOpenChange={setShowViewModal}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-3xl">{selectedRecipe?.icon || '📋'}</span>
                <div>
                  <DialogTitle className="text-2xl">{selectedRecipe?.name}</DialogTitle>
                  <p className="text-sm text-muted-foreground mt-1">{selectedRecipe?.recipe_id}</p>
                </div>
              </div>
              {!selectedRecipe?.is_system && (
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => {
                    setShowViewModal(false)
                    handleEditClick(selectedRecipe)
                  }}
                >
                  <Edit className="w-3 h-3 mr-2" />
                  Edit
                </Button>
              )}
            </div>
          </DialogHeader>
          
          {selectedRecipe && (
            <div className="space-y-6">
              {/* Badges */}
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="text-xs">{selectedRecipe.category}</Badge>
                <Badge variant="outline" className="text-xs capitalize">{selectedRecipe.difficulty}</Badge>
                {selectedRecipe.is_system && (
                  <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                    System Recipe
                  </Badge>
                )}
                {selectedRecipe.is_featured && (
                  <Badge variant="outline" className="text-xs bg-yellow-500/10 text-yellow-400 border-yellow-500/20">
                    <Star className="w-3 h-3 mr-1" />
                    Featured
                  </Badge>
                )}
              </div>

              {/* Description */}
              <div>
                <h3 className="text-sm font-semibold mb-2">Description</h3>
                <p className="text-sm text-muted-foreground">{selectedRecipe.description}</p>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-4 p-4 bg-secondary/30 rounded-lg">
                <div>
                  <div className="text-xs text-muted-foreground mb-1">Used</div>
                  <div className="text-lg font-semibold">{selectedRecipe.use_count} times</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground mb-1">Success Rate</div>
                  <div className="text-lg font-semibold">{selectedRecipe.success_rate || 0}%</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground mb-1">
                    <Clock className="w-3 h-3 inline mr-1" />
                    Duration
                  </div>
                  <div className="text-lg font-semibold">{selectedRecipe.estimated_time || 'N/A'}</div>
                </div>
              </div>

              {/* Recommended Agents */}
              {selectedRecipe.recommended_agents && selectedRecipe.recommended_agents.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold mb-2 flex items-center">
                    <Users className="w-4 h-4 mr-2" />
                    Recommended Agents ({selectedRecipe.recommended_agents.length})
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedRecipe.recommended_agents.map((agent: string, idx: number) => (
                      <Badge key={idx} variant="outline">{agent}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Tags */}
              {selectedRecipe.tags && selectedRecipe.tags.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold mb-2">Tags</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedRecipe.tags.map((tag: string, idx: number) => (
                      <Badge key={idx} variant="secondary" className="text-xs">#{tag}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Recipe Definition */}
              {selectedRecipe.recipe_definition && (
                <div>
                  <h3 className="text-sm font-semibold mb-2 flex items-center">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Workflow Steps
                  </h3>
                  {selectedRecipe.recipe_definition.steps && selectedRecipe.recipe_definition.steps.length > 0 ? (
                    <div className="space-y-2">
                      {selectedRecipe.recipe_definition.steps.map((step: any, idx: number) => (
                        <div key={idx} className="flex items-start p-3 bg-secondary/30 rounded-lg">
                          <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/20 text-primary text-xs font-semibold mr-3 flex-shrink-0">
                            {idx + 1}
                          </div>
                          <div className="flex-1">
                            <div className="font-medium text-sm">{step.name || step}</div>
                            {step.type && <div className="text-xs text-muted-foreground mt-1">Type: {step.type}</div>}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No steps defined</p>
                  )}
                </div>
              )}

              {/* Metadata */}
              <div className="pt-4 border-t border-secondary text-xs text-muted-foreground space-y-1">
                <div>Created by: {selectedRecipe.created_by}</div>
                <div>Version: {selectedRecipe.version}</div>
                {selectedRecipe.created_at && (
                  <div>Created: {new Date(selectedRecipe.created_at).toLocaleDateString()}</div>
                )}
                {selectedRecipe.last_used_at && (
                  <div>Last used: {new Date(selectedRecipe.last_used_at).toLocaleDateString()}</div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-4 border-t border-secondary">
                <Button
                  className="flex-1 bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
                  onClick={() => {
                    setShowViewModal(false)
                    handleUseRecipe(selectedRecipe)
                  }}
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Use This Recipe
                </Button>
                {!selectedRecipe?.is_system && !selectedRecipe?.is_marketplace_item && (
                  <Button
                    variant="outline"
                    onClick={() => {
                      setShowViewModal(false)
                      handleShareToMarketplace(selectedRecipe)
                    }}
                    disabled={submitToMarketplaceMutation.isPending}
                  >
                    <Share2 className="w-4 h-4 mr-2" />
                    Share to Marketplace
                  </Button>
                )}
                <Button
                  variant="outline"
                  onClick={() => setShowViewModal(false)}
                >
                  Close
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Edit Recipe Modal */}
      <Dialog open={showEditModal} onOpenChange={setShowEditModal}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Recipe</DialogTitle>
            <DialogDescription>
              Update recipe information and configuration
            </DialogDescription>
          </DialogHeader>
          
          <RecipeForm
            form={recipeForm}
            onChange={setRecipeForm}
            onSubmit={handleUpdateRecipe}
            onCancel={() => {
              setShowEditModal(false)
              setSelectedRecipe(null)
              resetForm()
            }}
            isLoading={updateMutation.isPending}
            submitLabel="Update Recipe"
            isEdit
          />
        </DialogContent>
      </Dialog>

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
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? 'Deleting...' : 'Delete Recipe'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

// Recipe Form Component
interface RecipeFormProps {
  form: any
  onChange: (form: any) => void
  onSubmit: () => void
  onCancel: () => void
  isLoading: boolean
  submitLabel: string
  isEdit?: boolean
}

function RecipeForm({ form, onChange, onSubmit, onCancel, isLoading, submitLabel, isEdit }: RecipeFormProps) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="recipe_id">Recipe ID *</Label>
          <Input
            id="recipe_id"
            value={form.recipe_id}
            onChange={(e) => onChange({ ...form, recipe_id: e.target.value })}
            placeholder="my-custom-recipe"
            disabled={isEdit}
            className="bg-secondary/50 border-secondary"
          />
          <p className="text-xs text-muted-foreground mt-1">
            Unique identifier (lowercase, hyphens only)
          </p>
        </div>
        
        <div>
          <Label htmlFor="icon">Icon</Label>
          <Input
            id="icon"
            value={form.icon}
            onChange={(e) => onChange({ ...form, icon: e.target.value })}
            placeholder="📋"
            className="bg-secondary/50 border-secondary"
          />
        </div>
      </div>

      <div>
        <Label htmlFor="name">Recipe Name *</Label>
        <Input
          id="name"
          value={form.name}
          onChange={(e) => onChange({ ...form, name: e.target.value })}
          placeholder="My Workflow Recipe"
          className="bg-secondary/50 border-secondary"
        />
      </div>

      <div>
        <Label htmlFor="description">Description *</Label>
        <Textarea
          id="description"
          value={form.description}
          onChange={(e) => onChange({ ...form, description: e.target.value })}
          placeholder="Describe what this recipe does..."
          className="bg-secondary/50 border-secondary min-h-[80px]"
        />
      </div>

      <div>
        <Label htmlFor="goal">Workflow Goal (Optional)</Label>
        <Input
          id="goal"
          value={form.goal || ''}
          onChange={(e) => onChange({ ...form, goal: e.target.value })}
          placeholder="e.g., Review PR #123 for security vulnerabilities"
          className="bg-secondary/50 border-secondary"
        />
        <p className="text-xs text-muted-foreground mt-1">
          High-level objective - overrides description if provided
        </p>
      </div>

      <div>
        <Label htmlFor="context">Workflow Context (Optional JSON)</Label>
        <Textarea
          id="context"
          value={typeof form.context === 'string' ? form.context : JSON.stringify(form.context || {}, null, 2)}
          onChange={(e) => {
            try {
              const parsed = JSON.parse(e.target.value)
              onChange({ ...form, context: parsed })
            } catch {
              // Invalid JSON, keep as string temporarily
              onChange({ ...form, context: e.target.value })
            }
          }}
          placeholder={'{\n  "codegraph_project": "my-app",\n  "pr_number": 123,\n  "git_url": "https://github.com/..."\n}'}
          className="bg-secondary/50 border-secondary min-h-[100px] font-mono text-xs"
        />
        <p className="text-xs text-muted-foreground mt-1">
          Additional context for execution (JSON format). Useful for CodeGraph integration, PR reviews, etc.
        </p>
        <div className="mt-2 p-2 bg-orange-500/10 border border-orange-500/20 rounded text-xs text-orange-300 flex items-start gap-2">
          <span className="text-sm">⚡</span>
          <span><strong>Pro Tip:</strong> Use <code className="bg-black/30 px-1 py-0.5 rounded">codegraph_project</code> in context to give agents access to indexed code.</span>
        </div>
      </div>

      <div>
        <Label htmlFor="tags">Tags (Optional)</Label>
        <Input
          id="tags"
          value={Array.isArray(form.tags) ? form.tags.join(', ') : ''}
          onChange={(e) => {
            const tagArray = e.target.value.split(',').map(t => t.trim()).filter(Boolean)
            onChange({ ...form, tags: tagArray })
          }}
          placeholder="project:automatos-ai, type:pr-review, automated"
          className="bg-secondary/50 border-secondary"
        />
        <p className="text-xs text-muted-foreground mt-1">
          Comma-separated tags for organization and filtering
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <Label htmlFor="category">Category *</Label>
          <Select 
            value={form.category}
            onValueChange={(value) => onChange({ ...form, category: value })}
          >
            <SelectTrigger className="bg-secondary/50 border-secondary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Support">Support</SelectItem>
              <SelectItem value="Data Processing">Data Processing</SelectItem>
              <SelectItem value="Content">Content</SelectItem>
              <SelectItem value="Security">Security</SelectItem>
              <SelectItem value="Development">Development</SelectItem>
              <SelectItem value="Marketing">Marketing</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="difficulty">Difficulty *</Label>
          <Select 
            value={form.difficulty}
            onValueChange={(value) => onChange({ ...form, difficulty: value })}
          >
            <SelectTrigger className="bg-secondary/50 border-secondary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="beginner">Beginner</SelectItem>
              <SelectItem value="intermediate">Intermediate</SelectItem>
              <SelectItem value="advanced">Advanced</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="estimated_time">Estimated Time</Label>
          <Input
            id="estimated_time"
            value={form.estimated_time}
            onChange={(e) => onChange({ ...form, estimated_time: e.target.value })}
            placeholder="5-10 minutes"
            className="bg-secondary/50 border-secondary"
          />
        </div>
      </div>

      <div className="flex justify-end gap-4 pt-4 border-t border-secondary">
        <Button variant="outline" onClick={onCancel} disabled={isLoading}>
          Cancel
        </Button>
        <Button 
          onClick={onSubmit} 
          disabled={isLoading || !form.recipe_id || !form.name || !form.description}
          className="bg-gray-800 border border-orange-400/50 hover:border-orange-400 hover:bg-gray-700 text-white transition-all duration-200"
        >
          {isLoading ? 'Saving...' : submitLabel}
        </Button>
      </div>
    </div>
  )
}

