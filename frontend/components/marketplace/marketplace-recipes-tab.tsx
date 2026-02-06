'use client'

import { useState } from 'react'
import { FileText, Download, Loader2, Bot, Zap, CheckCircle, ArrowRight, ChefHat, Check, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useToast } from '@/hooks/use-toast'
import { useUser } from '@clerk/nextjs'
import { useInstallRecipeFromMarketplace } from '@/hooks/use-recipe-api'
import { ViewRecipeModal } from '@/components/workflows/view-recipe-modal'

interface MarketplaceRecipesTabProps {
  searchQuery: string
}

export function MarketplaceRecipesTab({ searchQuery }: MarketplaceRecipesTabProps) {
  const [selectedType, setSelectedType] = useState('all')
  const [installingRecipeId, setInstallingRecipeId] = useState<number | null>(null)
  const [selectedRecipe, setSelectedRecipe] = useState<any>(null)
  const [showViewModal, setShowViewModal] = useState(false)
  const [approvingId, setApprovingId] = useState<number | null>(null)
  const [deletingId, setDeletingId] = useState<number | null>(null)
  const { toast } = useToast()
  const { user } = useUser()
  const queryClient = useQueryClient()
  const installMutation = useInstallRecipeFromMarketplace()

  // Admin check (same pattern as agents tab)
  const isAdmin = user?.emailAddresses?.[0]?.emailAddress?.includes('automatos.app') || false

  const handleApprove = async (e: React.MouseEvent, recipeId: number) => {
    e.stopPropagation()
    setApprovingId(recipeId)
    try {
      await apiClient.post(`/api/marketplace/items/${recipeId}/approve`)
      toast({ title: 'Recipe approved and published to marketplace!' })
      queryClient.invalidateQueries({ queryKey: ['marketplaceRecipes'] })
    } catch (error: any) {
      toast({
        title: 'Failed to approve recipe',
        description: error?.message || 'An error occurred',
        variant: 'destructive',
      })
    } finally {
      setApprovingId(null)
    }
  }

  const handleDeleteRecipe = async (e: React.MouseEvent, recipeId: number) => {
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this marketplace recipe?')) return
    setDeletingId(recipeId)
    try {
      await apiClient.delete(`/api/marketplace/items/${recipeId}`)
      toast({ title: 'Recipe removed from marketplace' })
      queryClient.invalidateQueries({ queryKey: ['marketplaceRecipes'] })
    } catch (error: any) {
      toast({
        title: 'Failed to delete recipe',
        description: error?.message || 'An error occurred',
        variant: 'destructive',
      })
    } finally {
      setDeletingId(null)
    }
  }

  const { data: recipes = [], isLoading } = useQuery({
    queryKey: ['marketplaceRecipes', selectedType, searchQuery],
    queryFn: async () => {
      const params = new URLSearchParams({
        type: 'recipe',
        ...(searchQuery && { search: searchQuery }),
        limit: '50'
      })
      return apiClient.get(`/api/marketplace/items?${params}`)
    },
  })

  const handleViewRecipe = (recipe: any) => {
    setSelectedRecipe(recipe)
    setShowViewModal(true)
  }

  const handleInstall = async (e: React.MouseEvent, recipeId: number) => {
    e.stopPropagation()
    setInstallingRecipeId(recipeId)
    try {
      const data = await installMutation.mutateAsync(recipeId)

      toast({
        title: 'Recipe Added to Workspace',
        description: data.message || 'Recipe added successfully',
        variant: 'default'
      })

      // Show warnings for missing dependencies (e.g., agents not found in marketplace)
      if (data.warnings && data.warnings.length > 0) {
        data.warnings.forEach((warning: string) => {
          toast({
            title: 'Warning',
            description: warning,
            variant: 'default'
          })
        })
      }

      // Also invalidate marketplace queries for updated install counts
      queryClient.invalidateQueries({ queryKey: ['marketplaceRecipes'] })
    } catch (error: any) {
      toast({
        title: 'Installation Failed',
        description: error?.message || 'Failed to add recipe to workspace',
        variant: 'destructive'
      })
    } finally {
      setInstallingRecipeId(null)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        <Button
          variant={selectedType === 'all' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setSelectedType('all')}
          className={
            selectedType === 'all'
              ? 'bg-primary hover:bg-primary/90 text-primary-foreground'
              : 'border-secondary text-muted-foreground hover:bg-secondary'
          }
        >
          All Recipes
        </Button>
        <Button
          variant={selectedType === 'simple' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setSelectedType('simple')}
          className={
            selectedType === 'simple'
              ? 'bg-primary hover:bg-primary/90 text-primary-foreground'
              : 'border-secondary text-muted-foreground hover:bg-secondary'
          }
        >
          Simple
        </Button>
        <Button
          variant={selectedType === 'complex' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setSelectedType('complex')}
          className={
            selectedType === 'complex'
              ? 'bg-primary hover:bg-primary/90 text-primary-foreground'
              : 'border-secondary text-muted-foreground hover:bg-secondary'
          }
        >
          Complex
        </Button>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-48 glass-card animate-pulse" />
          ))}
        </div>
      ) : recipes.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No recipes found. Recipes will be available soon!</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {recipes.map((recipe: any) => (
            <Card
              key={recipe.id}
              className="glass-card card-glow hover:border-orange-500/30 hover:shadow-lg hover:shadow-orange-500/20 transition-all duration-300 cursor-pointer flex flex-col"
              onClick={() => handleViewRecipe(recipe)}
            >
              <CardHeader className="pb-3">
                {/* Icon INLINE with Title - EXACTLY like Agent */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <ChefHat className="w-10 h-10 text-orange-400 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold text-white line-clamp-1">
                          {recipe.name}
                        </h3>
                        {isAdmin && !recipe.is_approved && (
                          <Badge variant="outline" className="text-xs border-yellow-500/30 text-yellow-400">
                            Pending
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-gray-500">
                        by {recipe.creator_name || 'Unknown'}
                      </p>
                    </div>
                  </div>
                  {/* Admin Controls */}
                  {isAdmin && (
                    <div className="flex gap-1 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
                      {!recipe.is_approved && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-6 w-6 p-0 border-green-500/30 text-green-400 hover:bg-green-500/10"
                          onClick={(e) => handleApprove(e, recipe.id)}
                          disabled={approvingId === recipe.id}
                          title="Approve"
                        >
                          {approvingId === recipe.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-6 w-6 p-0 border-red-500/30 text-red-400 hover:bg-red-500/10"
                        onClick={(e) => handleDeleteRecipe(e, recipe.id)}
                        disabled={deletingId === recipe.id}
                        title="Delete"
                      >
                        {deletingId === recipe.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
                      </Button>
                    </div>
                  )}
                </div>
              </CardHeader>

              <CardContent className="flex-1 flex flex-col gap-3">
                {/* Description — fixed 2-line height */}
                <p className="text-sm text-gray-400 line-clamp-2 min-h-[2.5rem]">
                  {recipe.description || 'No description available'}
                </p>

                {/* Category badge — fixed slot so cards stay even */}
                <div className="min-h-[1.25rem]">
                  {recipe.marketplace_category && (
                    <Badge variant="outline" className="text-xs border-gray-700 text-gray-400 w-fit">
                      {recipe.marketplace_category}
                    </Badge>
                  )}
                </div>

                {/* Stats Row */}
                <div className="flex items-center gap-3 text-xs text-gray-400">
                  <div className="flex items-center gap-1">
                    <FileText className="w-3.5 h-3.5" />
                    <span>{recipe.steps?.length || 0} Steps</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Bot className="w-3.5 h-3.5" />
                    <span>{recipe.steps ? new Set(recipe.steps.map((s: any) => s.agent_id)).size : 0} Agents</span>
                  </div>
                </div>

                {/* Install count */}
                <div className="text-xs text-gray-500">
                  {recipe.install_count >= 1000
                    ? `${(recipe.install_count / 1000).toFixed(1)}k installs`
                    : `${recipe.install_count || 0} installs`}
                </div>

                {/* Action Buttons — pinned to bottom */}
                <div className="flex items-center gap-2 pt-2 border-t border-gray-800 mt-auto">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 border-gray-700 text-gray-300 hover:bg-gray-800"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleViewRecipe(recipe)
                    }}
                  >
                    Details
                  </Button>
                  <Button
                    size="sm"
                    className="flex-1 bg-orange-600 hover:bg-orange-700 text-white"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleInstall(e, recipe.id)
                    }}
                    disabled={installingRecipeId === recipe.id}
                  >
                    {installingRecipeId === recipe.id ? (
                      <>
                        <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                        Adding...
                      </>
                    ) : (
                      <>
                        <Download className="w-3 h-3 mr-1" />
                        Add to Workspace
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* View Recipe Modal */}
      <ViewRecipeModal
        open={showViewModal}
        onClose={() => {
          setShowViewModal(false)
          setSelectedRecipe(null)
        }}
        recipe={selectedRecipe}
        onExecute={(e) => {
          if (selectedRecipe) {
            handleInstall(e as any, selectedRecipe.id)
          }
        }}
      />
    </div>
  )
}
