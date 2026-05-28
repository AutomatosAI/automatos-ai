'use client'

import { Plus, Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { pillBase, inactiveStyle } from './chat-mode-styles'
import { MAX_PINNED } from '@/hooks/use-pinned-agents'

interface PinAgentPickerAgent {
  id: number
  name: string
  agent_type: string
  status: string
}

interface PinAgentPickerProps {
  agents: PinAgentPickerAgent[]
  pinnedIds: number[]
  onPin: (agentId: number) => void
  onUnpin: (agentId: number) => void
}

export function PinAgentPicker({
  agents,
  pinnedIds,
  onPin,
  onUnpin,
}: PinAgentPickerProps) {
  const activeAgents = agents.filter((a) => a.status === 'active')
  const atMax = pinnedIds.length >= MAX_PINNED

  const handleClick = (agentId: number) => {
    if (pinnedIds.includes(agentId)) {
      onUnpin(agentId)
    } else if (!atMax) {
      onPin(agentId)
    }
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          title="Pin an agent"
          className={cn(pillBase, inactiveStyle)}
        >
          <Plus className="h-4 w-4 md:h-3.5 md:w-3.5 text-primary" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="center" className="w-56">
        <DropdownMenuLabel className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-2 py-1.5">
          Pin an Agent
        </DropdownMenuLabel>
        {activeAgents.length === 0 && (
          <div className="px-2 py-1.5 text-xs text-muted-foreground">
            No agents available
          </div>
        )}
        {activeAgents.map((agent) => {
          const isPinned = pinnedIds.includes(agent.id)
          const isDisabled = !isPinned && atMax

          return (
            <DropdownMenuItem
              key={agent.id}
              disabled={isDisabled}
              onSelect={() => handleClick(agent.id)}
              className="flex items-center gap-2"
            >
              <span className="flex-1 truncate">{agent.name}</span>
              <span className="text-[10px] text-muted-foreground uppercase tracking-wide">
                {agent.agent_type}
              </span>
              {isPinned && (
                <Check className="h-3.5 w-3.5 text-primary shrink-0" />
              )}
            </DropdownMenuItem>
          )
        })}
        {atMax && (
          <div className="px-2 py-1.5 text-xs text-muted-foreground text-center">
            Max {MAX_PINNED} pins
          </div>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
