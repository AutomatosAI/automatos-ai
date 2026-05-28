'use client'

import { Bot } from 'lucide-react'
import { PremiumIcon } from '@/components/shared'
import { useAssignableAgents } from '@/hooks/use-agent-api'
import { cn } from '@/lib/utils'

interface BoardAgentSidebarProps {
  selectedAgentId: number | null
  onSelectAgent: (agentId: number | null) => void
  className?: string
}

const BADGE_COLORS: Record<string, string> = {
  lead: 'bg-warning/20 text-warning border-warning/30',
  int: 'bg-info/20 text-info border-info/30',
  spc: 'bg-agent/20 text-agent border-agent/30',
}

function getRoleBadge(role: string | undefined): { label: string; color: string } | null {
  if (!role) return null
  const lower = role.toLowerCase()
  if (lower.includes('lead') || lower.includes('manager') || lower.includes('founder'))
    return { label: 'LEAD', color: BADGE_COLORS.lead }
  if (lower.includes('intern') || lower.includes('junior'))
    return { label: 'INT', color: BADGE_COLORS.int }
  return { label: 'SPC', color: BADGE_COLORS.spc }
}

export function BoardAgentSidebar({ selectedAgentId, onSelectAgent, className }: BoardAgentSidebarProps) {
  const { data: rawAgents, isLoading } = useAssignableAgents()
  const agents = (rawAgents as any[]) ?? []

  return (
    <div className={cn('w-[200px] shrink-0 flex flex-col border-r border-border/30', className)}>
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border/30">
        <div className="w-2 h-2 rounded-full bg-muted-foreground" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Agents
        </h3>
        <span className="text-[10px] font-medium text-muted-foreground bg-secondary/50 px-1.5 py-0.5 rounded-full">
          {agents.length}
        </span>
      </div>

      {/* Agent list */}
      <div className="flex-1 overflow-y-auto py-1">
        {isLoading ? (
          <div className="space-y-2 px-3 py-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-12 bg-secondary/30 rounded-lg animate-pulse" />
            ))}
          </div>
        ) : agents.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <Bot className="w-6 h-6 mb-1 opacity-30" />
            <p className="text-[10px]">No agents</p>
          </div>
        ) : (
          agents.map((agent: any) => {
            const isSelected = selectedAgentId === agent.id
            const badge = getRoleBadge(agent.role)
            const isWorking = agent.status === 'active' || agent.last_heartbeat_status === 'working'

            return (
              <button
                key={agent.id}
                onClick={() => onSelectAgent(isSelected ? null : agent.id)}
                className={cn(
                  'w-full flex items-center gap-2 px-3 py-2 text-left transition-colors',
                  isSelected
                    ? 'bg-primary/10 border-l-2 border-l-primary'
                    : 'hover:bg-secondary/20 border-l-2 border-l-transparent',
                )}
              >
                <div className="w-8 h-8 flex items-center justify-center shrink-0">
                  {agent.premium_icon ? (
                    <PremiumIcon name={agent.premium_icon} size={22} />
                  ) : (
                    <Bot className="w-4 h-4 text-primary" />
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-medium truncate">{agent.name}</span>
                    {badge && (
                      <span className={cn('text-[8px] font-bold px-1 py-0 rounded border', badge.color)}>
                        {badge.label}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <div className={cn(
                      'w-1.5 h-1.5 rounded-full',
                      isWorking ? 'bg-[hsl(var(--success))]' : 'bg-muted-foreground/30',
                    )} />
                    <span className="text-[10px] text-muted-foreground uppercase">
                      {isWorking ? 'Working' : 'Idle'}
                    </span>
                  </div>
                </div>
              </button>
            )
          })
        )}
      </div>
    </div>
  )
}
