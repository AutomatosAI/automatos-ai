'use client'

import { useState } from 'react'
import { FileText, Download, Loader2, Bot, Zap, CheckCircle, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useToast } from '@/hooks/use-toast'
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
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const installMutation = useInstallRecipeFromMarketplace()

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
              className="glass-card card-glow hover:border-orange-500/30 hover:shadow-lg hover:shadow-orange-500/20 transition-all duration-300 cursor-pointer"
              onClick={() => handleViewRecipe(recipe)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/30 flex items-center justify-center">
                    <FileText className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold line-clamp-1 text-white">{recipe.name}</h3>
                    <p className="text-xs text-gray-500">by {recipe.creator_name}</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-gray-400 line-clamp-3">{recipe.description}</p>

                {/* What This Recipe Does - The Shiny Part! */}
                {recipe.steps && recipe.steps.length > 0 && (
                  <div className="bg-orange-500/5 border border-orange-500/20 rounded-lg p-3 space-y-2">
                    <div className="flex items-center gap-2 text-orange-400 text-xs font-semibold">
                      <Zap className="w-3 h-3" />
                      <span>This Recipe Will:</span>
                    </div>
                    <div className="space-y-1">
                      {recipe.steps.slice(0, 2).map((step: any, idx: number) => (
                        <div key={idx} className="flex items-start gap-2 text-xs text-gray-300">
                          <ArrowRight className="w-3 h-3 mt-0.5 text-orange-400 shrink-0" />
                          <span className="line-clamp-1">{step.prompt_template || `Step ${step.order}`}</span>
                        </div>
                      ))}
                      {recipe.steps.length > 2 && (
                        <div className="text-xs text-orange-400 pl-5">
                          +{recipe.steps.length - 2} more automated steps
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Key Stats */}
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-gray-800">
                  <div className="text-center">
                    <div className="text-xs text-gray-500">Steps</div>
                    <div className="text-sm font-semibold text-white">{recipe.steps?.length || 0}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-500">Agents</div>
                    <div className="text-sm font-semibold text-white flex items-center justify-center gap-1">
                      <Bot className="w-3 h-3 text-orange-400" />
                      {recipe.steps ? new Set(recipe.steps.map((s: any) => s.agent_id)).size : 0}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-500">Quality</div>
                    <div className="text-sm font-semibold flex items-center justify-center gap-1">
                      {recipe.quality_score ? (
                        <>
                          <CheckCircle className="w-3 h-3 text-green-400" />
                          <span className="text-green-400">{(recipe.quality_score * 100).toFixed(0)}%</span>
                        </>
                      ) : (
                        <span className="text-gray-500">N/A</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Install Count & Button */}
                <div className="flex items-center justify-between pt-2 border-t border-gray-800">
                  <span className="text-xs text-gray-500 flex items-center gap-1">
                    <Download className="w-3 h-3" />
                    {recipe.install_count >= 1000
                      ? `${(recipe.install_count / 1000).toFixed(1)}k people use this`
                      : `${recipe.install_count} people use this`}
                  </span>
                  <Button
                    size="sm"
                    className="bg-orange-600 hover:bg-orange-700 text-white shadow-lg shadow-orange-900/20"
                    onClick={(e) => handleInstall(e, recipe.id)}
                    disabled={installingRecipeId === recipe.id}
                  >
                    {installingRecipeId === recipe.id ? (
                      <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                    ) : (
                      <Download className="w-3 h-3 mr-1" />
                    )}
                    {installingRecipeId === recipe.id ? 'Adding...' : 'Get It Now'}
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
