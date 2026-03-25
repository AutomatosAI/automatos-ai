'use client'

import { useState } from 'react'
import {
  Loader2, Download, Bot, Check, Trash2, Zap, MoreVertical
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { PremiumIcon } from '@/components/shared'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ToolLogo } from '@/components/ui/tool-logo'
import { useMarketplaceItems, useInstallMarketplaceItem } from '@/hooks/use-marketplace-api'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { AGENT_CATEGORIES as UNIFIED_CATEGORIES, LEGACY_CATEGORY_MAP } from '@/lib/agent-constants'
import { MarketplaceItemModal } from './marketplace-item-modal'
import { useUser } from '@clerk/nextjs'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'

// Unified agent categories from shared constants + "All" filter
const MARKETPLACE_CATEGORIES = [
  { id: 'all', name: 'All Categories' },
  ...UNIFIED_CATEGORIES.map(c => ({ id: c.id, name: c.name })),
]

/**
 * Normalize a marketplace agent's category to the unified system.
 * Handles legacy Title Case categories and unknown values.
 */
const normalizeCategory = (category: string | null | undefined): string => {
  if (!category) return 'custom'
  // Already a unified ID?
  if (UNIFIED_CATEGORIES.some(c => c.id === category)) return category
  // Legacy mapping?
  return LEGACY_CATEGORY_MAP[category] || 'custom'
}

interface MarketplaceAgent {
  id: number
  name: string
  description: string
  creator_name: string
  category: string
  install_count: number
  is_approved?: boolean
  is_featured?: boolean
  icon?: string
  metadata: {
    agent_type?: string
    model_id?: string
    skills?: string[]
    tools?: number[]
    tool_names?: string[]
    tool_icons?: string[]
  }
}

interface MarketplaceAgentsTabProps {
  searchQuery: string
}

export function MarketplaceAgentsTab({ searchQuery }: MarketplaceAgentsTabProps) {
  const { user } = useUser()
  const [viewMode, setViewMode] = useViewMode('mp-agents')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null)
  const [approvingId, setApprovingId] = useState<number | null>(null)
  const [deletingId, setDeletingId] = useState<number | null>(null)

  // Check if user is admin (you can adjust this check based on your admin logic)
  const isAdmin = user?.emailAddresses?.[0]?.emailAddress?.includes('automatos.app') || false

  // Fetch all marketplace agents (filter client-side by unified category)
  const { data: rawAgents = [], isLoading, refetch } = useMarketplaceItems({
    type: 'agent',
    search: searchQuery || undefined,
    limit: 100
  })

  // Client-side category filtering using normalized categories
  const agents = selectedCategory === 'all'
    ? rawAgents
    : (rawAgents as MarketplaceAgent[]).filter((a: MarketplaceAgent) =>
        normalizeCategory(a.category) === selectedCategory
      )

  // Get system icons
  const { data: iconMappings = {} } = useSystemIcons()

  // Install agent mutation using our marketplace hook
  const installMutation = useInstallMarketplaceItem()

  const handleAgentClick = (agent: MarketplaceAgent) => {
    setSelectedAgentId(agent.id)
  }

  const handleApprove = async (e: React.MouseEvent, agentId: number) => {
    e.stopPropagation()
    setApprovingId(agentId)
    try {
      await apiClient.post(`/api/marketplace/items/${agentId}/approve`)
      toast.success('Agent approved and published to marketplace!')
      refetch()
    } catch (error: any) {
      toast.error('Failed to approve agent', {
        description: error?.message || 'An error occurred'
      })
    } finally {
      setApprovingId(null)
    }
  }

  const handleDelete = async (e: React.MouseEvent, agentId: number) => {
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this marketplace agent?')) {
      return
    }
    setDeletingId(agentId)
    try {
      await apiClient.delete(`/api/marketplace/items/${agentId}`)
      toast.success('Agent removed from marketplace')
      refetch()
    } catch (error: any) {
      toast.error('Failed to delete agent', {
        description: error?.message || 'An error occurred'
      })
    } finally {
      setDeletingId(null)
    }
  }

  const formatInstallCount = (count: number) => {
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`
    if (count >= 1000) return `${(count / 1000).toFixed(1)}k`
    return count.toString()
  }

  return (
    <div className="space-y-6">
      {/* Category Filter Buttons */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent flex-1">
          {MARKETPLACE_CATEGORIES.map((category) => {
            const premiumName = iconMappings[category.id]
            return (
              <Button
                key={category.id}
                variant={selectedCategory === category.id ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedCategory(category.id)}
                className={`whitespace-nowrap flex-shrink-0 gap-1.5 ${selectedCategory === category.id
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
        <ViewToggle value={viewMode} onChange={setViewMode} />
      </div>

      {/* Agents Grid - 4 columns like Agent Management page */}
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
              <div key={i} className="h-64 glass-card animate-pulse" />
            ))}
          </div>
        )
      ) : (agents as MarketplaceAgent[]).length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No agents found. Try adjusting your search or filters.</p>
        </div>
      ) : viewMode === 'list' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {(agents as MarketplaceAgent[]).map((agent: MarketplaceAgent) => {
            const normalized = normalizeCategory(agent.category)
            const premiumIconName = iconMappings[normalized] || iconMappings[agent.category] || null
            const catDef = UNIFIED_CATEGORIES.find(c => c.id === normalized)
            return (
              <Card
                key={agent.id}
                className="glass-card hover:border-primary/20 transition-all cursor-pointer"
                onClick={() => handleAgentClick(agent)}
              >
                <CardContent className="p-3">
                  <div className="flex items-center gap-3">
                    {premiumIconName ? (
                      <PremiumIcon name={premiumIconName} size={36} className="shrink-0" />
                    ) : (
                      <Bot className="w-9 h-9 text-primary shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-sm truncate">{agent.name}</span>
                        {isAdmin && !agent.is_approved && (
                          <Badge variant="outline" className="text-[10px] border-[hsl(var(--warning))]/30 text-[hsl(var(--warning))] shrink-0">
                            Pending
                          </Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                        <span>{catDef?.name || agent.category}</span>
                        <span>&middot;</span>
                        <span>{formatInstallCount(agent.install_count)} installs</span>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 shrink-0"
                      onClick={(e) => { e.stopPropagation(); handleAgentClick(agent) }}
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {agents.map((agent: MarketplaceAgent) => (
            <Card
              key={agent.id}
              className="glass-card card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer"
              onClick={() => handleAgentClick(agent)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    {(() => {
                      const normalized = normalizeCategory(agent.category)
                      const premiumIconName = iconMappings[normalized] || iconMappings[agent.category] || null
                      return premiumIconName ? (
                        <PremiumIcon name={premiumIconName} size={40} className="shrink-0" />
                      ) : (
                        <Bot className="w-10 h-10 text-primary shrink-0" />
                      )
                    })()}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold text-foreground line-clamp-1">
                          {agent.name}
                        </h3>
                        {isAdmin && !agent.is_approved && (
                          <Badge variant="outline" className="text-xs border-[hsl(var(--warning))]/30 text-[hsl(var(--warning))]">
                            Pending
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        by {agent.creator_name}
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
                        {!agent.is_approved && (
                          <DropdownMenuItem
                            onClick={(e) => { e.stopPropagation(); handleApprove(e as any, agent.id) }}
                            disabled={approvingId === agent.id}
                            title="Approve"
                          >
                            <Check className="w-4 h-4 mr-2" />
                            {approvingId === agent.id ? 'Approving...' : 'Approve'}
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuItem
                          className="text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))] hover:bg-[hsl(var(--destructive))]/10 focus:text-[hsl(var(--destructive))] focus:bg-[hsl(var(--destructive))]/10"
                          onClick={(e) => { e.stopPropagation(); handleDelete(e as any, agent.id) }}
                          disabled={deletingId === agent.id}
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          {deletingId === agent.id ? 'Deleting...' : 'Delete'}
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  )}
                </div>
              </CardHeader>

              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {agent.description}
                </p>

                {/* Category badge */}
                <div className="flex flex-col gap-2">
                  <Badge variant="outline" className="text-xs border-border text-muted-foreground w-fit">
                    {UNIFIED_CATEGORIES.find(c => c.id === normalizeCategory(agent.category))?.name || agent.category}
                  </Badge>
                  {agent.metadata.model_id && (
                    <Badge className="text-xs bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/30 w-fit">
                      <Zap className="w-3 h-3 mr-1" />
                      {agent.metadata.model_id.split('/').pop()?.substring(0, 15)}
                    </Badge>
                  )}
                </div>

                {/* Tools Preview - matching agent roster */}
                {agent.metadata.tool_names && agent.metadata.tool_names.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-3 border-t border-border/30">
                    {agent.metadata.tool_names.slice(0, 5).map((toolName, idx) => (
                      <div key={idx} title={toolName}>
                        <ToolLogo
                          name={toolName}
                          logo={agent.metadata.tool_icons?.[idx]}
                          size={24}
                          showBackground={true}
                          className="bg-secondary/30 border border-border/30"
                        />
                      </div>
                    ))}
                    {agent.metadata.tool_names.length > 5 && (
                      <div className="bg-secondary/30 px-1.5 h-[24px] flex items-center justify-center rounded-md border border-border/30 text-[10px] text-muted-foreground">
                        +{agent.metadata.tool_names.length - 5}
                      </div>
                    )}
                  </div>
                )}

                {/* Install count */}
                <div className="text-xs text-muted-foreground pb-2">
                  {formatInstallCount(agent.install_count)} installs
                </div>

              </CardContent>
              <Separator />
              <div className="flex items-center justify-between px-6 py-3">
                <Button variant="ghost" size="sm"
                  onClick={(e) => { e.stopPropagation(); handleAgentClick(agent) }}
                  className="text-muted-foreground hover:text-foreground p-0 h-auto">
                  Details
                </Button>
                <Button size="sm" variant="outline"
                  onClick={(e) => { e.stopPropagation(); handleAgentClick(agent) }}>
                  Add to Workspace
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Detail Modal */}
      {selectedAgentId && (
        <MarketplaceItemModal
          itemId={selectedAgentId}
          onClose={() => setSelectedAgentId(null)}
        />
      )}
    </div>
  )
}
