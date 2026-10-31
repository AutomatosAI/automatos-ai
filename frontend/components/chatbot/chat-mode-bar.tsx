'use client'

import { Code2, FileText, Mic, Target } from 'lucide-react'
import { useHostVoice } from '@/components/voice/voice-provider'
import { type VoicePressOptions, useVoicePress } from '@/components/voice/voice-press'
import { cn } from '@/lib/utils'
import { pillBase, inactiveStyle } from './chat-mode-styles'

interface PinnedAgent {
  id: number
  name: string
  agent_type: string
  status: string
}

interface ChatModeBarProps {
  isCodeActive: boolean
  isPlanActive: boolean
  isMissionActive: boolean
  onCodeClick: () => void
  onPlanClick: () => void
  onMissionClick: () => void
  pinnedAgentIds?: number[]
  agents?: PinnedAgent[]
  selectedAgentId?: number | null
  onAgentSelect?: (agentId: number | null) => void
}

const activeStyle = 'bg-primary/20 ring-1 ring-primary/50 text-foreground/90'

// Live is the voice presence — brand warning-orange, like the orb.
const liveActiveStyle = 'bg-warning/20 ring-1 ring-warning/50 text-foreground/90'

const agentActiveStyle = 'bg-primary/20 ring-1 ring-primary/50 text-foreground/90'

const iconClass = 'h-4 w-4 md:h-3.5 md:w-3.5 text-primary'
const labelClass = 'hidden md:inline'
const LIVE_LABEL = 'Live'
const LIVE_PRESS: VoicePressOptions = Object.freeze({ mode: 'live', press: 'toggle' })

/**
 * Live voice (PRD-207 S5), on the host voice provider (theme voice): pressed while the tab's session is a live call;
 * a press starts Auto Live on the conversation on screen, or ends the call. When live-status says Live can't start
 * now, the chip says why (its title) and still tries: the mint is the truth, and the strip shows its refusal in
 * plain words. Nothing fences it on the AI label (the product owner's policy, 23 Oct 2026). The press is the one
 * every voice control shares (components/voice/voice-press.ts), as a toggle that starts Live.
 */
function LiveChip() {
  const voice = useHostVoice()
  const { active, handlers } = useVoicePress(voice, LIVE_PRESS)
  return (
    <button
      type="button"
      onClick={handlers.onClick}
      title={voice.available.reason ?? LIVE_LABEL}
      aria-pressed={active}
      data-live-available={voice.available.live}
      className={cn(pillBase, active ? liveActiveStyle : inactiveStyle)}
    >
      <Mic className={cn(iconClass, active && 'text-warning')} />
      <span className={labelClass}>{LIVE_LABEL}</span>
    </button>
  )
}

export function ChatModeBar({
  isCodeActive,
  isPlanActive,
  isMissionActive,
  onCodeClick,
  onPlanClick,
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
      {/* Live voice (PRD-207) */}
      <LiveChip />

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

      {/* Plan mode */}
      <button
        type="button"
        onClick={onPlanClick}
        title="Plan"
        className={cn(pillBase, isPlanActive ? activeStyle : inactiveStyle)}
      >
        <FileText className={iconClass} />
        <span className={labelClass}>Plan</span>
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
