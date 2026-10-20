'use client'

import {
  type HostChatMessage,
  type HostMissionSuggestion,
  MissionCreatedCard,
  MissionSuggestionCard,
} from '@automatos/theme-sdk'

import { recentMessages } from './reply'
import styles from './turn.module.css'

/**
 * PRD-260 §4 — a Mission Auto suggests, as the core chat handles it
 * (components/chatbot/chat.tsx): a 'mission-suggestion' data part becomes the
 * core's MissionSuggestionCard (goal, complexity, agentId, chatId and the
 * recent messages; it creates the Mission itself), shown only while no Mission
 * is being planned (the chat's planningMissionId), and the core's
 * MissionCreatedCard once one is. Play only frames them in its column: the
 * cards and their Mission creation are the core's, through the SDK
 * (PRD-261 A-14a: cards.*, and the suggestion and the Mission on
 * useChatController's view model).
 */

export const SUGGESTION_LABEL = 'Mission suggestion'
export const MISSION_CREATED_LABEL = 'Mission created'

export interface PlayMissionSuggestionProps {
  readonly suggestion: HostMissionSuggestion | null
  /** The Mission the chat is planning, once one is. */
  readonly planningMissionId: string | null
  /** The conversation the Mission comes from ('' before the backend named a new chat). */
  readonly chatId: string
  readonly messages: readonly HostChatMessage[]
}

export function PlayMissionSuggestion({ suggestion, planningMissionId, chatId, messages }: PlayMissionSuggestionProps) {
  if (planningMissionId) {
    return (
      <section className={styles.aside} aria-label={MISSION_CREATED_LABEL}>
        <MissionCreatedCard missionId={planningMissionId} />
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
