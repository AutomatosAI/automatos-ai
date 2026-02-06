'use client'

import { useState } from 'react'
import { FileText, Download, Loader2, Bot, Zap, CheckCircle, ArrowRight, ChefHat, MoreVertical, Eye } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
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
              ? 'bg-secondary border-primary/50 text-foreground font-semibold'
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
              ? 'bg-secondary border-primary/50 text-foreground font-semibold'
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
              ? 'bg-secondary border-primary/50 text-foreground font-semibold'
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
              className="glass-card card-glow hover:border-primary/30 hover:shadow-lg hover:shadow-primary/20 transition-all duration-300 cursor-pointer"
              onClick={() => handleViewRecipe(recipe)}
            >
              <CardHeader className="pb-3">
                {/* Icon INLINE with Title - EXACTLY like Agent */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <ChefHat className="w-10 h-10 text-primary shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold text-foreground line-clamp-1">
                          {recipe.name}
                        </h3>
                        {isAdmin && !recipe.is_approved && (
                          <Badge variant="outline" className="text-xs border-yellow-500/30 text-yellow-400">
                            Pending
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        by {recipe.creator_name || 'Unknown'}
                      </p>
                    </div>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={(e) => e.stopPropagation()}>
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleViewRecipe(recipe) }}>
                        <Eye className="w-4 h-4 mr-2" />
                        View Details
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleInstall(e as any, recipe.id) }} disabled={installingRecipeId === recipe.id}>
                        <Download className="w-4 h-4 mr-2" />
                        {installingRecipeId === recipe.id ? 'Adding...' : 'Add to Workspace'}
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </CardHeader>

              <CardContent className="space-y-3">
                {/* Description */}
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {recipe.description || 'No description available'}
                </p>

                {/* Category badge */}
                {recipe.marketplace_category && (
                  <div className="flex flex-col gap-2">
                    <Badge variant="outline" className="text-xs border-border text-muted-foreground w-fit">
                      {recipe.marketplace_category}
                    </Badge>
                  )}
                </div>

                {/* Stats Row */}
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
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
                <div className="text-xs text-muted-foreground pb-2">
                  {recipe.install_count >= 1000
                    ? `${(recipe.install_count / 1000).toFixed(1)}k installs`
                    : `${recipe.install_count || 0} installs`}
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
