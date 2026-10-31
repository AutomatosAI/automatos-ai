'use client'

/**
 * PRD-268 §6.12.4 (US-W01b) — the chat's MissionSuggestionCard, telling the pinned suggestion it is open.
 *
 * chat.tsx imports this under the card's own name, so the Chat component (over the length limit) is untouched. It
 * draws the same card and registers it in mission-suggestion-presence.ts from mount until the card is answered (its
 * Mission launched, or dismissed by X or "No thanks") or unmounted; meanwhile the suggestion pinned above the
 * composer hides.
 */
import { useEffect, useState } from 'react'
import {
  MissionSuggestionCard as Card,
  type MissionSuggestionCardProps,
} from '@/components/chatbot/mission-suggestion-card'
import { markMissionSuggestionOpen } from './mission-suggestion-presence'

export function MissionSuggestionCard({ onAnswered, ...props }: MissionSuggestionCardProps) {
  const [answered, setAnswered] = useState(false)

  useEffect(() => (answered ? undefined : markMissionSuggestionOpen()), [answered])

  const answer = () => {
    setAnswered(true)
    onAnswered?.()
  }
  return <Card {...props} onAnswered={answer} />
}
