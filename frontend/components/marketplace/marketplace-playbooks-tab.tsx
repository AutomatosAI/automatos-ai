'use client'

import { useState } from 'react'
import { FileText, Download, Loader2, Bot, Zap, CheckCircle, ArrowRight, ChefHat, MoreVertical, Eye } from 'lucide-react'
import { PremiumIcon } from '@/components/shared'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Separator } from '@/components/ui/separator'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'
import { useUser } from '@clerk/nextjs'
import { useInstallPlaybookFromMarketplace } from '@/hooks/use-playbook-api'
import { ViewPlaybookModal } from '@/components/workflows/view-playbook-modal'

interface MarketplacePlaybooksTabProps {
  searchQuery: string
}

export function MarketplacePlaybooksTab({ searchQuery }: MarketplacePlaybooksTabProps) {
  const [viewMode, setViewMode] = useViewMode('mp-recipes')
  const [selectedType, setSelectedType] = useState('all')
  const [installingPlaybookId, setInstallingRecipeId] = useState<number | null>(null)
  const [selectedPlaybook, setSelectedRecipe] = useState<any>(null)
  const [showViewModal, setShowViewModal] = useState(false)
  const [approvingId, setApprovingId] = useState<number | null>(null)
  const [deletingId, setDeletingId] = useState<number | null>(null)
  const { user } = useUser()
  const queryClient = useQueryClient()
  const installMutation = useInstallPlaybookFromMarketplace()
  const { data: iconMappings = {} } = useSystemIcons()

  // Admin check (same pattern as agents tab)
  const isAdmin = user?.emailAddresses?.[0]?.emailAddress?.includes('automatos.app') || false

  const handleApprove = async (e: React.MouseEvent, recipeId: number) => {
    e.stopPropagation()
    setApprovingId(recipeId)
    try {
      await apiClient.post(`/api/marketplace/items/${recipeId}/approve`)
      toast('Playbook approved and published to marketplace!')
      queryClient.invalidateQueries({ queryKey: ['marketplacePlaybooks'] })
    } catch (error: any) {
      toast.error('Failed to approve recipe', { description: error?.message || 'An error occurred' })
    } finally {
      setApprovingId(null)
    }
  }

  const handleDeletePlaybook = async (e: React.MouseEvent, recipeId: number) => {
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this marketplace recipe?')) return
    setDeletingId(recipeId)
    try {
      await apiClient.delete(`/api/marketplace/items/${recipeId}`)
      toast('Playbook removed from marketplace')
      queryClient.invalidateQueries({ queryKey: ['marketplacePlaybooks'] })
    } catch (error: any) {
      toast.error('Failed to delete recipe', { description: error?.message || 'An error occurred' })
    } finally {
      setDeletingId(null)
    }
  }

  const { data: recipes = [], isLoading } = useQuery({
    queryKey: ['marketplacePlaybooks', selectedType, searchQuery],
    queryFn: async () => {
      const params = new URLSearchParams({
        type: 'recipe',
        ...(searchQuery && { search: searchQuery }),
        limit: '50'
      })
      return apiClient.get(`/api/marketplace/items?${params}`)
    },
  })

  const handleViewPlaybook = (recipe: any) => {
    setSelectedRecipe(recipe)
    setShowViewModal(true)
  }

  const handleInstall = async (e: React.MouseEvent, recipeId: number) => {
    e.stopPropagation()
    setInstallingRecipeId(recipeId)
    try {
      const data = await installMutation.mutateAsync(recipeId)

      toast('Playbook Added to Workspace', { description: data.message || 'Recipe added successfully' })

      // Show warnings for missing dependencies (e.g., agents not found in marketplace)
      if (data.warnings && data.warnings.length > 0) {
        data.warnings.forEach((warning: string) => {
          toast('Warning', { description: warning })
        })
      }

      // Also invalidate marketplace queries for updated install counts
      queryClient.invalidateQueries({ queryKey: ['marketplacePlaybooks'] })
    } catch (error: any) {
      toast.error('Installation Failed', { description: error?.message || 'Failed to add playbook to workspace' })
    } finally {
      setInstallingRecipeId(null)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent flex-1">
          <Button
            variant={selectedType === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedType('all')}
            className={`whitespace-nowrap flex-shrink-0 ${
              selectedType === 'all'
                ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }`}
          >
            All Playbooks
          </Button>
          <Button
            variant={selectedType === 'simple' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedType('simple')}
            className={`whitespace-nowrap flex-shrink-0 ${
              selectedType === 'simple'
                ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }`}
          >
            Simple
          </Button>
          <Button
            variant={selectedType === 'complex' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedType('complex')}
            className={`whitespace-nowrap flex-shrink-0 ${
              selectedType === 'complex'
                ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }`}
          >
            Complex
          </Button>
        </div>
        <ViewToggle value={viewMode} onChange={setViewMode} />
      </div>

      {isLoading ? (
        viewMode === 'list' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-16 glass-card animate-pulse rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="h-48 glass-card animate-pulse" />
            ))}
          </div>
        )
      ) : recipes.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No playbooks found. Playbooks will be available soon!</p>
        </div>
      ) : viewMode === 'list' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {recipes.map((recipe: any) => (
            <Card
              key={recipe.id}
              className="glass-card card-glow hover:border-primary/20 transition-all cursor-pointer"
              onClick={() => handleViewPlaybook(recipe)}
            >
              <CardContent className="p-3">
                <div className="flex items-center gap-3">
                  {(() => {
                    const premiumIconName = iconMappings[recipe.marketplace_category] || iconMappings['global_recipe'] || null
                    return premiumIconName ? (
                      <PremiumIcon name={premiumIconName} size={36} className="text-primary shrink-0" />
                    ) : (
                      <ChefHat className="w-8 h-8 text-primary shrink-0" />
                    )
                  })()}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm truncate">{recipe.name}</span>
                      {isAdmin && !recipe.is_approved && (
                        <Badge variant="outline" className="text-[10px] border-warning/30 text-warning shrink-0">
                          Pending
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                      <span>{recipe.steps?.length || 0} Steps</span>
                      <span>&middot;</span>
                      <span>{recipe.steps ? new Set(recipe.steps.map((s: any) => s.agent_id)).size : 0} Agents</span>
                      <span>&middot;</span>
                      <span>{recipe.install_count || 0} installs</span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0 shrink-0"
                    onClick={(e) => { e.stopPropagation(); handleInstall(e, recipe.id) }}
                    disabled={installingPlaybookId === recipe.id}
                  >
                    {installingPlaybookId === recipe.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {recipes.map((recipe: any) => (
            <Card
              key={recipe.id}
              className="glass-card card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer"
              onClick={() => handleViewPlaybook(recipe)}
            >
              <CardHeader className="pb-3">
                {/* Icon INLINE with Title - EXACTLY like Agent */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    {(() => {
                      const premiumIconName = iconMappings[recipe.marketplace_category] || iconMappings['global_recipe'] || null
                      return premiumIconName ? (
                        <PremiumIcon name={premiumIconName} size={40} className="text-primary shrink-0" />
                      ) : (
                        <ChefHat className="w-10 h-10 text-primary shrink-0" />
                      )
                    })()}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold text-foreground line-clamp-1">
                          {recipe.name}
                        </h3>
                        {isAdmin && !recipe.is_approved && (
                          <Badge variant="outline" className="text-xs border-warning/30 text-warning">
                            Pending
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        by {recipe.creator_name || 'Unknown'}
                      </p>
                    </div>
                  </div>
                  {isAdmin && (
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={(e) => e.stopPropagation()}>
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleViewPlaybook(recipe) }}>
                          <Eye className="w-4 h-4 mr-2" />
                          View Details
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  )}
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
                  </div>
                )}

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
              <Separator />
              <div className="flex items-center justify-between px-6 py-3">
                <Button variant="ghost" size="sm"
                  onClick={(e) => { e.stopPropagation(); handleViewPlaybook(recipe) }}
                  className="text-muted-foreground hover:text-foreground p-0 h-auto">
                  Details
                </Button>
                <Button size="sm" variant="outline"
                  disabled={installingPlaybookId === recipe.id}
                  onClick={(e) => {
                    e.stopPropagation()
                    handleInstall(e as any, recipe.id)
                  }}>
                  {installingPlaybookId === recipe.id ? 'Adding...' : 'Add to Workspace'}
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* View Recipe Modal */}
      <ViewPlaybookModal
        open={showViewModal}
        onClose={() => {
          setShowViewModal(false)
          setSelectedRecipe(null)
        }}
        recipe={selectedPlaybook}
        onExecute={(e) => {
          if (selectedPlaybook) {
            handleInstall(e as any, selectedPlaybook.id)
          }
        }}
      />
    </div>
  )
}
