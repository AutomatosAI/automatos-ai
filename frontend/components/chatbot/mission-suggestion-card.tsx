'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Target, Rocket, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useCreateMission } from '@/hooks/use-missions-api'
import { useMissionStore } from '@/stores/mission-store'
import { toast } from 'sonner'

interface MissionSuggestionCardProps {
  goal: string
  complexity: string
  agentId?: number | string
  chatId: string
  recentMessages?: Array<{ role: string; content: string }>
}

export function MissionSuggestionCard({
  goal,
  complexity,
  agentId,
  chatId,
  recentMessages,
}: MissionSuggestionCardProps) {
  const [dismissed, setDismissed] = useState(false)
  const router = useRouter()
  const createMission = useCreateMission()
  const setActivePlanningMissionId = useMissionStore((s) => s.setActivePlanningMissionId)

  if (dismissed) return null

  const goalText = goal.length > 150 ? `${goal.slice(0, 150)}...` : goal

  const complexityLabel: Record<string, string> = {
    organ: 'Multi-step',
    organism: 'Complex multi-agent',
  }

  const handleLaunch = () => {
    createMission.mutate(
      {
        goal,
        config: {
          source: 'chat',
          chat_id: chatId,
          ...(recentMessages && recentMessages.length > 0
            ? { context_messages: recentMessages.slice(-5) }
            : {}),
        },
      },
      {
        onSuccess: (mission) => {
          setActivePlanningMissionId(mission.id)
          toast.success('Mission created — review the plan', {
            action: {
              label: 'View',
              onClick: () => router.push(`/missions/${mission.id}` as any),
            },
          })
        },
        onError: (err) => {
          toast.error(`Failed to create mission: ${err.message}`)
        },
      }
    )
  }

  return (
    <div className="bg-card/50 backdrop-blur border border-orange-500/20 rounded-xl p-4 space-y-3 max-w-md">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Target className="w-4 h-4 text-orange-500" />
          <span className="text-xs font-medium text-orange-400">
            Mission Suggested
          </span>
          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-orange-500/10 text-orange-400 font-medium">
            {complexityLabel[complexity] ?? complexity}
          </span>
        </div>
        <button
          onClick={() => setDismissed(true)}
          className="text-muted-foreground hover:text-foreground transition-colors p-0.5"
          aria-label="Dismiss suggestion"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Description */}
      <p className="text-sm text-foreground/80 leading-snug">
        This task could benefit from a <strong>Multi-Agent Mission</strong> — multiple agents working together with verification.
      </p>

      {/* Goal preview */}
      <p className="text-xs text-muted-foreground italic">&quot;{goalText}&quot;</p>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          className="border-orange-500/30 text-orange-400 hover:bg-orange-500/10"
          onClick={handleLaunch}
          disabled={createMission.isLoading}
        >
          <Rocket className="w-3.5 h-3.5 mr-1.5" />
          {createMission.isLoading ? 'Creating...' : 'Launch Mission'}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-foreground"
          onClick={() => setDismissed(true)}
        >
          No thanks
        </Button>
      </div>
    </div>
  )
}
