'use client'

import { useState } from 'react'
import { FileText, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

interface MarketplaceRecipesTabProps {
  searchQuery: string
}

export function MarketplaceRecipesTab({ searchQuery }: MarketplaceRecipesTabProps) {
  const [selectedType, setSelectedType] = useState('all')

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
            <Card key={recipe.id} className="glass-card card-glow hover:border-primary/20 transition-all duration-300">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/30 flex items-center justify-center">
                    <FileText className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold line-clamp-1">{recipe.name}</h3>
                    <p className="text-xs text-muted-foreground">by {recipe.creator_name}</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground line-clamp-2">{recipe.description}</p>

                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">
                    {recipe.category || 'Workflow'}
                  </Badge>
                  {recipe.metadata?.recipe_type && (
                    <Badge className="text-xs bg-blue-500/20 text-blue-300 border-blue-500/30">
                      {recipe.metadata.recipe_type}
                    </Badge>
                  )}
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-secondary">
                  <span className="text-xs text-muted-foreground">
                    {recipe.install_count >= 1000
                      ? `${(recipe.install_count / 1000).toFixed(1)}k installs`
                      : `${recipe.install_count} installs`}
                  </span>
                  <Button size="sm" variant="outline">
                    <Download className="w-3 h-3 mr-1" />
                    Install
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
