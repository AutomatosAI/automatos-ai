'use client'

import { useState } from 'react'
import {
  Loader2, Download, Bot, Check, Trash2, Zap, MoreVertical, Eye,
  UserCircle, Headphones, Terminal, Share2, Calculator, ShoppingBag, PenTool, Users, BarChart3, Wrench
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ToolLogo } from '@/components/ui/tool-logo'
import { useMarketplaceItems, useInstallMarketplaceItem } from '@/hooks/use-marketplace-api'
import { MarketplaceItemModal } from './marketplace-item-modal'
import { useUser } from '@clerk/nextjs'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'

// Agent categories matching US-006
const AGENT_CATEGORIES = [
  { id: 'all', name: 'All Categories' },
  { id: 'Personal Assistant', name: 'Personal Assistant' },
  { id: 'Customer Support', name: 'Customer Support' },
  { id: 'DevOps', name: 'DevOps' },
  { id: 'Social Media', name: 'Social Media' },
  { id: 'Accounting', name: 'Accounting' },
  { id: 'E-commerce', name: 'E-commerce' },
  { id: 'Content Creation', name: 'Content Creation' },
  { id: 'HR', name: 'HR' },
  { id: 'Data Analysis', name: 'Data Analysis' },
  { id: 'Custom', name: 'Custom' },
]

// Category to icon mapping - SIMPLE clean icons only!
const getCategoryIcon = (category: string) => {
  const iconMap: Record<string, any> = {
    'Personal Assistant': UserCircle,
    'Customer Support': Headphones,
    'DevOps': Terminal,
    'Social Media': Share2,
    'Accounting': Calculator,
    'E-commerce': ShoppingBag,
    'Content Creation': PenTool,
    'HR': Users,
    'Data Analysis': BarChart3,
    'Custom': Bot,  // Simple bot, no jellyfish!
    'specialized': Bot,
    'general': Wrench,
  }
  return iconMap[category] || Bot  // Default to simple Bot
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
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null)
  const [approvingId, setApprovingId] = useState<number | null>(null)
  const [deletingId, setDeletingId] = useState<number | null>(null)

  // Check if user is admin (you can adjust this check based on your admin logic)
  const isAdmin = user?.emailAddresses?.[0]?.emailAddress?.includes('automatos.app') || false

  // Fetch marketplace agents using our marketplace hook
  const { data: agents = [], isLoading, refetch } = useMarketplaceItems({
    type: 'agent',
    category: selectedCategory !== 'all' ? selectedCategory : undefined,
    search: searchQuery || undefined,
    limit: 100
  })

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
      <div className="flex flex-wrap gap-2">
        {AGENT_CATEGORIES.map((category) => (
          <Button
            key={category.id}
            variant={selectedCategory === category.id ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory(category.id)}
            className={
              selectedCategory === category.id
                ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }
          >
            {category.name}
          </Button>
        ))}
      </div>

      {/* Agents Grid - 4 columns like Agent Management page */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-64 glass-card animate-pulse" />
          ))}
        </div>
      ) : agents.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No agents found. Try adjusting your search or filters.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {agents.map((agent: MarketplaceAgent) => (
            <Card
              key={agent.id}
              className="glass-card card-glow hover:border-primary/30 hover:shadow-lg hover:shadow-primary/20 transition-all duration-300 cursor-pointer"
              onClick={() => handleAgentClick(agent)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    {(() => {
                      const IconComponent = getCategoryIcon(agent.category)
                      return <IconComponent className="w-10 h-10 text-primary shrink-0" />
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
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={(e) => e.stopPropagation()}>
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleAgentClick(agent) }}>
                        <Eye className="w-4 h-4 mr-2" />
                        View Details
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleAgentClick(agent) }}>
                        <Download className="w-4 h-4 mr-2" />
                        Add to Workspace
                      </DropdownMenuItem>
                      {isAdmin && !agent.is_approved && (
                        <DropdownMenuItem
                          onClick={(e) => { e.stopPropagation(); handleApprove(e as any, agent.id) }}
                          disabled={approvingId === agent.id}
                          title="Approve"
                        >
                          <Check className="w-4 h-4 mr-2" />
                          {approvingId === agent.id ? 'Approving...' : 'Approve'}
                        </DropdownMenuItem>
                      )}
                      {isAdmin && (
                        <DropdownMenuItem
                          className="text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))] hover:bg-[hsl(var(--destructive))]/10 focus:text-[hsl(var(--destructive))] focus:bg-[hsl(var(--destructive))]/10"
                          onClick={(e) => { e.stopPropagation(); handleDelete(e as any, agent.id) }}
                          disabled={deletingId === agent.id}
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          {deletingId === agent.id ? 'Deleting...' : 'Delete'}
                        </DropdownMenuItem>
                      )}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </CardHeader>

              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {agent.description}
                </p>

                {/* Category badge */}
                <div className="flex flex-col gap-2">
                  <Badge variant="outline" className="text-xs border-border text-muted-foreground w-fit">
                    {agent.category}
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
                  <div className="flex flex-wrap gap-2 pt-3 border-t border-white/5">
                    {agent.metadata.tool_names.slice(0, 5).map((toolName, idx) => (
                      <div key={idx} title={toolName}>
                        <ToolLogo
                          name={toolName}
                          logo={agent.metadata.tool_icons?.[idx]}
                          size={24}
                          showBackground={true}
                          className="bg-secondary/30 border border-white/5"
                        />
                      </div>
                    ))}
                    {agent.metadata.tool_names.length > 5 && (
                      <div className="bg-secondary/30 px-1.5 h-[24px] flex items-center justify-center rounded-md border border-white/5 text-[10px] text-muted-foreground">
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
