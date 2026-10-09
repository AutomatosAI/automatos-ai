'use client'

import { useState } from 'react'
import { Bot } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PremiumIcon } from '@/components/shared'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import { useMarketplaceItems, useInstallMarketplaceItem } from '@/hooks/use-marketplace-api'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { AGENT_CATEGORIES as UNIFIED_CATEGORIES } from '@/lib/agent-constants'
import { MarketplaceItemModal } from './marketplace-item-modal'
import { useSystemRole } from '@/contexts/role-context'
import {
  AgentGridCard,
  AgentListRow,
  AgentsLoading,
  normalizeCategory,
  useAgentAdminActions,
  type AgentInstallState,
  type MarketplaceAgent,
} from './marketplace-agent-cards'

// Unified agent categories from shared constants + "All" filter
const MARKETPLACE_CATEGORIES = [
  { id: 'all', name: 'All Categories' },
  ...UNIFIED_CATEGORIES.map(c => ({ id: c.id, name: c.name })),
]

interface MarketplaceAgentsTabProps {
  searchQuery: string
}

function CategoryFilter({ selected, onSelect, iconMappings }: {
  selected: string
  onSelect: (id: string) => void
  iconMappings: Record<string, string>
}) {
  return (
    <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent flex-1">
      {MARKETPLACE_CATEGORIES.map((category) => {
        const premiumName = iconMappings[category.id]
        const active = selected === category.id
        return (
          <Button
            key={category.id}
            variant={active ? 'default' : 'outline'}
            size="sm"
            onClick={() => onSelect(category.id)}
            className={`whitespace-nowrap flex-shrink-0 gap-1.5 ${active
              ? 'bg-secondary border-primary/50 text-foreground font-semibold'
              : 'border-secondary text-muted-foreground hover:bg-secondary'
              }`}
          >
            {premiumName && <PremiumIcon name={premiumName} size={14} />}
            {category.name}
          </Button>
        )
      })}
    </div>
  )
}

function useAgentInstallState(): AgentInstallState {
  const installMutation = useInstallMarketplaceItem()
  const [installingId, setInstallingId] = useState<number | null>(null)
  const [installedIds, setInstalledIds] = useState<Set<number>>(new Set())
  const install = (agentId: number) => {
    setInstallingId(agentId)
    installMutation.mutate(agentId, {
      onSuccess: () => setInstalledIds(prev => new Set([...prev, agentId])),
      onSettled: () => setInstallingId(null),
    })
  }
  return { installingId, installedIds, install }
}

export function MarketplaceAgentsTab({ searchQuery }: MarketplaceAgentsTabProps) {
  const [viewMode, setViewMode] = useViewMode('mp-agents')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null)

  // Admin controls follow the backend's system_role (super_admin ⊇ admin), the
  // same gate the admin routes enforce. The local operator is super_admin, so a
  // fresh local install sees Import from GitHub; a Clerk email domain never did.
  const { isAdmin } = useSystemRole()

  // Fetch all marketplace agents (filter client-side by unified category)
  const { data: rawAgents = [], isLoading, refetch } = useMarketplaceItems({
    type: 'agent',
    search: searchQuery || undefined,
    limit: 100
  })
  const agents = (rawAgents as MarketplaceAgent[]).filter((a) =>
    selectedCategory === 'all' || normalizeCategory(a.category) === selectedCategory
  )

  const { data: iconMappings = {} } = useSystemIcons()
  const admin = useAgentAdminActions(() => refetch())
  const installState = useAgentInstallState()
  const openAgent = (agent: MarketplaceAgent) => setSelectedAgentId(agent.id)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <CategoryFilter selected={selectedCategory} onSelect={setSelectedCategory} iconMappings={iconMappings} />
        <ViewToggle value={viewMode} onChange={setViewMode} />
      </div>

      {isLoading ? (
        <AgentsLoading list={viewMode === 'list'} />
      ) : agents.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No agents found. Try adjusting your search or filters.</p>
        </div>
      ) : viewMode === 'list' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {agents.map((agent) => (
            <AgentListRow key={agent.id} agent={agent} iconMappings={iconMappings} isAdmin={isAdmin} onOpen={openAgent} />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {agents.map((agent) => (
            <AgentGridCard key={agent.id} agent={agent} iconMappings={iconMappings} isAdmin={isAdmin}
              admin={admin} installState={installState} onOpen={openAgent} />
          ))}
        </div>
      )}

      {selectedAgentId && (
        <MarketplaceItemModal itemId={selectedAgentId} itemType="agent" onClose={() => setSelectedAgentId(null)} />
      )}
    </div>
  )
}
