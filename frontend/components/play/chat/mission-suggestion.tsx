'use client'

import { MissionCreatedCard } from '@/components/chatbot/mission-created-card'
import { MissionSuggestionCard } from '@/components/chatbot/mission-suggestion-card'
import { useMissionStore } from '@/stores/mission-store'
import type { ChatMessage } from '@/types'

import { type MissionSuggestion, recentMessages } from './suggestion'
import styles from './turn.module.css'

/**
 * PRD-260 §4 — a Mission Auto suggests, as the core chat handles it
 * (components/chatbot/chat.tsx): a 'mission-suggestion' data part becomes the
 * core's MissionSuggestionCard (goal, complexity, agentId, chatId and the
 * recent messages; it creates the Mission itself), shown only while no Mission
 * is being planned (useMissionStore's activePlanningMissionId), and the core's
 * MissionCreatedCard once one is. Play only frames them in its column: the
 * cards and their Mission creation are the core's.
 */

export const SUGGESTION_LABEL = 'Mission suggestion'
export const MISSION_CREATED_LABEL = 'Mission created'

export interface PlayMissionSuggestionProps {
  readonly suggestion: MissionSuggestion | null
  /** The conversation the Mission comes from ('' before the backend named a new chat). */
  readonly chatId: string
  readonly messages: readonly ChatMessage[]
}

export function PlayMissionSuggestion({ suggestion, chatId, messages }: PlayMissionSuggestionProps) {
  const activePlanningMissionId = useMissionStore((s) => s.activePlanningMissionId)
  if (activePlanningMissionId) {
    return (
      <section className={styles.aside} aria-label={MISSION_CREATED_LABEL}>
        <MissionCreatedCard missionId={activePlanningMissionId} />
      </section>
    )
  }
  if (!suggestion) return null
  return (
    <section className={styles.aside} aria-label={SUGGESTION_LABEL}>
      <MissionSuggestionCard
        goal={suggestion.goal}
        complexity={suggestion.complexity}
        agentId={suggestion.agentId}
        chatId={chatId}
        recentMessages={recentMessages(messages)}
      />
    </section>
  )
}
