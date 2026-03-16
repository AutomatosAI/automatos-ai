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

const pillBase = [
  'inline-flex items-center justify-center rounded-full',
  'min-h-[44px] min-w-[44px] px-3 py-2 md:min-h-0 md:min-w-0 md:px-3 md:py-1.5',
  'gap-2 text-xs font-medium',
  'backdrop-blur transition-colors',
  'shadow-[0_0_18px_rgba(249,115,22,0.10)]',
].join(' ')

const inactiveStyle = 'bg-black/10 text-foreground/90 hover:bg-orange-500/10'

export function PinAgentPicker({
  agents,
  pinnedIds,
  onPin,
  onUnpin,
}: PinAgentPickerProps) {
  const activeAgents = agents.filter((a) => a.status === 'active')
  const atMax = pinnedIds.length >= 6

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
          <Plus className="h-4 w-4 md:h-3.5 md:w-3.5 text-orange-400" />
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
                <Check className="h-3.5 w-3.5 text-orange-400 shrink-0" />
              )}
            </DropdownMenuItem>
          )
        })}
        {atMax && (
          <div className="px-2 py-1.5 text-xs text-muted-foreground text-center">
            Max 6 pins
          </div>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
