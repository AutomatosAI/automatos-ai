'use client'

import { useState } from 'react'
import { Download, Bot, Check, Trash2, Zap, MoreVertical } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { PremiumIcon } from '@/components/shared'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ToolLogo } from '@/components/ui/tool-logo'
import { AGENT_CATEGORIES as UNIFIED_CATEGORIES, LEGACY_CATEGORY_MAP } from '@/lib/agent-constants'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'

const TOOL_PREVIEW_COUNT = 5

export interface MarketplaceAgent {
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

export interface AgentAdminActions {
  approvingId: number | null
  deletingId: number | null
  approve: (agentId: number) => void
  remove: (agentId: number) => void
}

export interface AgentInstallState {
  installingId: number | null
  installedIds: Set<number>
  install: (agentId: number) => void
}

type IconMappings = Record<string, string>

/**
 * Normalize a marketplace agent's category to the unified system.
 * Handles legacy Title Case categories and unknown values.
 */
export const normalizeCategory = (category: string | null | undefined): string => {
  if (!category) return 'custom'
  if (UNIFIED_CATEGORIES.some(c => c.id === category)) return category
  return LEGACY_CATEGORY_MAP[category] || 'custom'
}

export const formatInstallCount = (count: number) => {
  if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`
  return count.toString()
}

/** Approve / delete for marketplace agents. The routes need ?type=agent: playbooks share id values. */
export function useAgentAdminActions(onChanged: () => void): AgentAdminActions {
  const [approvingId, setApprovingId] = useState<number | null>(null)
  const [deletingId, setDeletingId] = useState<number | null>(null)

  async function approve(agentId: number) {
    setApprovingId(agentId)
    try {
      await apiClient.post(`/api/marketplace/items/${agentId}/approve?type=agent`)
      toast.success('Agent approved and published to marketplace!')
      onChanged()
    } catch (error: any) {
      toast.error('Failed to approve agent', { description: error?.message || 'An error occurred' })
    } finally {
      setApprovingId(null)
    }
  }

  async function remove(agentId: number) {
    if (!confirm('Are you sure you want to delete this marketplace agent?')) return
    setDeletingId(agentId)
    try {
      await apiClient.delete(`/api/marketplace/items/${agentId}?type=agent`)
      toast.success('Agent removed from marketplace')
      onChanged()
    } catch (error: any) {
      toast.error('Failed to delete agent', { description: error?.message || 'An error occurred' })
    } finally {
      setDeletingId(null)
    }
  }

  return { approvingId, deletingId, approve, remove }
}

function AgentIcon({ agent, iconMappings, size }: { agent: MarketplaceAgent; iconMappings: IconMappings; size: 36 | 40 }) {
  const premiumIconName = iconMappings[normalizeCategory(agent.category)] || iconMappings[agent.category] || null
  if (premiumIconName) return <PremiumIcon name={premiumIconName} size={size} className="shrink-0" />
  return <Bot className={`${size === 36 ? 'w-8 h-8' : 'w-10 h-10'} text-primary shrink-0`} />
}

function PendingBadge({ small }: { small?: boolean }) {
  return (
    <Badge variant="outline" className={`${small ? 'text-[10px] shrink-0' : 'text-xs'} border-[hsl(var(--warning))]/30 text-[hsl(var(--warning))]`}>
      Pending
    </Badge>
  )
}

export function AgentsLoading({ list }: { list: boolean }) {
  if (list) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {[...Array(6)].map((_, i) => <div key={i} className="h-16 glass-card animate-pulse rounded-xl" />)}
      </div>
    )
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {[...Array(8)].map((_, i) => <div key={i} className="h-64 glass-card animate-pulse" />)}
    </div>
  )
}

export function AgentListRow({ agent, iconMappings, isAdmin, onOpen }: {
  agent: MarketplaceAgent
  iconMappings: IconMappings
  isAdmin: boolean
  onOpen: (agent: MarketplaceAgent) => void
}) {
  const catDef = UNIFIED_CATEGORIES.find(c => c.id === normalizeCategory(agent.category))
  return (
    <Card className="glass-card card-glow hover:border-primary/20 transition-all cursor-pointer" onClick={() => onOpen(agent)}>
      <CardContent className="p-3">
        <div className="flex items-center gap-3">
          <AgentIcon agent={agent} iconMappings={iconMappings} size={36} />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm truncate">{agent.name}</span>
              {isAdmin && !agent.is_approved && <PendingBadge small />}
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
              <span>{catDef?.name || agent.category}</span>
              <span>&middot;</span>
              <span>{formatInstallCount(agent.install_count)} installs</span>
            </div>
          </div>
          <Button variant="ghost" size="sm" className="h-8 w-8 p-0 shrink-0"
            onClick={(e) => { e.stopPropagation(); onOpen(agent) }}>
            <Download className="w-4 h-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function AgentCardAdminMenu({ agent, admin }: { agent: MarketplaceAgent; admin: AgentAdminActions }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 w-8 p-0" onClick={(e) => e.stopPropagation()}>
          <MoreVertical className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {!agent.is_approved && (
          <DropdownMenuItem
            onClick={(e) => { e.stopPropagation(); admin.approve(agent.id) }}
            disabled={admin.approvingId === agent.id}
            title="Approve"
          >
            <Check className="w-4 h-4 mr-2" />
            {admin.approvingId === agent.id ? 'Approving...' : 'Approve'}
          </DropdownMenuItem>
        )}
        <DropdownMenuItem
          className="text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))] hover:bg-[hsl(var(--destructive))]/10 focus:text-[hsl(var(--destructive))] focus:bg-[hsl(var(--destructive))]/10"
          onClick={(e) => { e.stopPropagation(); admin.remove(agent.id) }}
          disabled={admin.deletingId === agent.id}
        >
          <Trash2 className="w-4 h-4 mr-2" />
          {admin.deletingId === agent.id ? 'Deleting...' : 'Delete'}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function AgentToolsPreview({ agent }: { agent: MarketplaceAgent }) {
  const toolNames = agent.metadata.tool_names || []
  if (toolNames.length === 0) return null
  return (
    <div className="flex flex-wrap gap-2 pt-3 border-t border-border/30">
      {toolNames.slice(0, TOOL_PREVIEW_COUNT).map((toolName, idx) => (
        <div key={idx} title={toolName}>
          <ToolLogo name={toolName} logo={agent.metadata.tool_icons?.[idx]} size={24} showBackground={true}
            className="bg-secondary/30 border border-border/30" />
        </div>
      ))}
      {toolNames.length > TOOL_PREVIEW_COUNT && (
        <div className="bg-secondary/30 px-1.5 h-[24px] flex items-center justify-center rounded-md border border-border/30 text-[10px] text-muted-foreground">
          +{toolNames.length - TOOL_PREVIEW_COUNT}
        </div>
      )}
    </div>
  )
}

function AgentCardFooter({ agent, installState, onOpen }: {
  agent: MarketplaceAgent
  installState: AgentInstallState
  onOpen: (agent: MarketplaceAgent) => void
}) {
  return (
    <div className="flex items-center justify-between px-6 py-3">
      <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); onOpen(agent) }}
        className="text-muted-foreground hover:text-foreground p-0 h-auto">
        Details
      </Button>
      {installState.installedIds.has(agent.id) ? (
        <Button size="sm" variant="secondary" className="bg-secondary/50 hover:bg-secondary border border-white/10">
          <Check className="w-3 h-3 mr-2" />
          Added
        </Button>
      ) : (
        <Button size="sm" variant="outline" disabled={installState.installingId === agent.id}
          onClick={(e) => { e.stopPropagation(); installState.install(agent.id) }}>
          {installState.installingId === agent.id ? 'Adding...' : 'Add to Workspace'}
        </Button>
      )}
    </div>
  )
}

export function AgentGridCard({ agent, iconMappings, isAdmin, admin, installState, onOpen }: {
  agent: MarketplaceAgent
  iconMappings: IconMappings
  isAdmin: boolean
  admin: AgentAdminActions
  installState: AgentInstallState
  onOpen: (agent: MarketplaceAgent) => void
}) {
  const categoryName = UNIFIED_CATEGORIES.find(c => c.id === normalizeCategory(agent.category))?.name || agent.category
  return (
    <Card className="glass-card card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer" onClick={() => onOpen(agent)}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <AgentIcon agent={agent} iconMappings={iconMappings} size={40} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-foreground line-clamp-1">{agent.name}</h3>
                {isAdmin && !agent.is_approved && <PendingBadge />}
              </div>
              <p className="text-xs text-muted-foreground">by {agent.creator_name}</p>
            </div>
          </div>
          {isAdmin && <AgentCardAdminMenu agent={agent} admin={admin} />}
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground line-clamp-2">{agent.description}</p>
        <div className="flex flex-col gap-2">
          <Badge variant="outline" className="text-xs border-border text-muted-foreground w-fit">{categoryName}</Badge>
          {agent.metadata.model_id && (
            <Badge className="text-xs bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/30 w-fit">
              <Zap className="w-3 h-3 mr-1" />
              {agent.metadata.model_id.split('/').pop()?.substring(0, 15)}
            </Badge>
          )}
        </div>
        <AgentToolsPreview agent={agent} />
        <div className="text-xs text-muted-foreground pb-2">{formatInstallCount(agent.install_count)} installs</div>
      </CardContent>
      <Separator />
      <AgentCardFooter agent={agent} installState={installState} onOpen={onOpen} />
    </Card>
  )
}
