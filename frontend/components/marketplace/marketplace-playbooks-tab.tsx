'use client'

import { useState } from 'react'
import { FileText } from 'lucide-react'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { useSystemRole } from '@/contexts/role-context'
import { useInstallPlaybookFromMarketplace } from '@/hooks/use-playbook-api'
import { ViewPlaybookModal } from '@/components/workflows/view-playbook-modal'
import { useMarketplaceAccess } from './access/marketplace-access'
import { PLAYBOOK_TYPE } from './access/public-shapes'
import { MarketplaceItemModal } from './marketplace-item-modal'
import { PlaybookGridCard, PlaybookListRow, PlaybookTypeFilter } from './marketplace-playbook-cards'

interface MarketplacePlaybooksTabProps {
  searchQuery: string
}

const PLAYBOOKS_TAB = 'recipes'
const PLAYBOOKS_LIMIT = 50

export function MarketplacePlaybooksTab({ searchQuery }: MarketplacePlaybooksTabProps) {
  const [viewMode, setViewMode] = useViewMode('mp-recipes')
  const [selectedType, setSelectedType] = useState('all')
  const [installingPlaybookId, setInstallingRecipeId] = useState<number | null>(null)
  const [selectedPlaybook, setSelectedRecipe] = useState<any>(null)
  const [showViewModal, setShowViewModal] = useState(false)
  const queryClient = useQueryClient()
  const installMutation = useInstallPlaybookFromMarketplace()
  // Signed out on SaaS: the public reads, and every add goes to sign-in (PRD-270 §5.7).
  const { signedIn, requireSignIn, source } = useMarketplaceAccess()
  const { data: iconMappings = {} } = useSystemIcons(signedIn)

  // Admin check (same pattern as agents tab)
  // Admin controls follow the backend's system_role (super_admin ⊇ admin), the
  // same gate the admin routes enforce. The local operator is super_admin, so a
  // fresh local install sees Import from GitHub; a Clerk email domain never did.
  const { isAdmin } = useSystemRole()

  const { data: recipes = [], isLoading } = useQuery({
    queryKey: ['marketplacePlaybooks', selectedType, searchQuery, source.kind],
    queryFn: () => source.items({ type: PLAYBOOK_TYPE, search: searchQuery || undefined, limit: PLAYBOOKS_LIMIT }),
  })

  const handleViewPlaybook = (recipe: any) => {
    setSelectedRecipe(recipe)
    setShowViewModal(true)
  }

  const handleInstall = async (e: React.MouseEvent, recipeId: number) => {
    e.stopPropagation()
    if (requireSignIn(PLAYBOOKS_TAB, recipeId)) return
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
        <PlaybookTypeFilter selected={selectedType} onSelect={setSelectedType} />
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
            <PlaybookListRow key={recipe.id} recipe={recipe} iconMappings={iconMappings} isAdmin={isAdmin}
              installing={installingPlaybookId === recipe.id} onView={handleViewPlaybook} onInstall={handleInstall} />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {recipes.map((recipe: any) => (
            <PlaybookGridCard key={recipe.id} recipe={recipe} iconMappings={iconMappings} isAdmin={isAdmin}
              installing={installingPlaybookId === recipe.id} onView={handleViewPlaybook} onInstall={handleInstall} />
          ))}
        </div>
      )}

      {/* View Recipe Modal; signed out, the item dialog (its steps aren't public) */}
      {!signedIn && showViewModal && selectedPlaybook && (
        <MarketplaceItemModal itemId={selectedPlaybook.id} itemType={PLAYBOOK_TYPE} onClose={() => setShowViewModal(false)} />
      )}
      <ViewPlaybookModal
        open={signedIn && showViewModal}
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
