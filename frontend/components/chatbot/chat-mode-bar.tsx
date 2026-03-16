'use client'

import { Code2, Target } from 'lucide-react'
import { cn } from '@/lib/utils'

interface PinnedAgent {
  id: number
  name: string
  agent_type: string
  status: string
}

interface ChatModeBarProps {
  isCodeActive: boolean
  isMissionActive: boolean
  onCodeClick: () => void
  onMissionClick: () => void
  pinnedAgentIds?: number[]
  agents?: PinnedAgent[]
  selectedAgentId?: number | null
  onAgentSelect?: (agentId: number | null) => void
}

const pillBase = [
  'inline-flex items-center justify-center rounded-full',
  'min-h-[44px] min-w-[44px] px-3 py-2 md:min-h-0 md:min-w-0 md:px-3 md:py-1.5',
  'gap-2 text-xs font-medium',
  'backdrop-blur transition-colors',
  'shadow-[0_0_18px_rgba(249,115,22,0.10)]',
].join(' ')

const activeStyle = 'bg-orange-500/20 ring-1 ring-orange-500/50 text-foreground/90'
const inactiveStyle = 'bg-black/10 text-foreground/90 hover:bg-orange-500/10'

const agentActiveStyle = 'bg-primary/20 ring-1 ring-primary/50 text-foreground/90'

const iconClass = 'h-4 w-4 md:h-3.5 md:w-3.5 text-orange-400'
const labelClass = 'hidden md:inline'

export function ChatModeBar({
  isCodeActive,
  isMissionActive,
  onCodeClick,
  onMissionClick,
  pinnedAgentIds = [],
  agents = [],
  selectedAgentId,
  onAgentSelect,
}: ChatModeBarProps) {
  const handleAgentClick = (agentId: number) => {
    if (!onAgentSelect) return
    onAgentSelect(selectedAgentId === agentId ? null : agentId)
  }

  return (
    <div className="flex flex-wrap justify-center gap-3 md:gap-2">
      {/* Code mode */}
      <button
        type="button"
        onClick={onCodeClick}
        title="Code"
        className={cn(pillBase, isCodeActive ? activeStyle : inactiveStyle)}
      >
        <Code2 className={iconClass} />
        <span className={labelClass}>Code</span>
      </button>

      {/* Mission mode */}
      <button
        type="button"
        onClick={onMissionClick}
        title="Mission"
        className={cn(pillBase, isMissionActive ? activeStyle : inactiveStyle)}
      >
        <Target className={iconClass} />
        <span className={labelClass}>Mission</span>
      </button>

      {/* Pinned agent shortcuts */}
      {pinnedAgentIds.map((agentId) => {
        const agent = agents.find((a) => a.id === agentId)
        if (!agent || agent.status !== 'active') return null

        const isSelected = selectedAgentId === agentId
        const initial = agent.name.charAt(0).toUpperCase()

        return (
          <button
            key={agentId}
            type="button"
            onClick={() => handleAgentClick(agentId)}
            title={agent.name}
            className={cn(pillBase, isSelected ? agentActiveStyle : inactiveStyle)}
          >
            <span className="rounded-full bg-primary/20 w-5 h-5 text-[10px] font-bold flex items-center justify-center">
              {initial}
            </span>
            <span className={labelClass}>{agent.name}</span>
          </button>
        )
      })}
    </div>
  )
}
