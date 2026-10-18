import type { ChatMessage } from '@/types'

import { messageText } from './message-text'

/**
 * PRD-260 §4, PRD-261 A-11a (M28) — a Mission Auto suggests (a
 * 'mission-suggestion' data part from useChat's onData), read as
 * components/chatbot/chat.tsx reads it, checked field by field because it
 * comes off the wire, and the recent messages the core's
 * MissionSuggestionCard sends as the Mission's context. Pure.
 */

/** chat.tsx sends the newest five messages as the Mission's context. */
export const CONTEXT_MESSAGES_KEPT = 5

/** The data of a 'mission-suggestion' part, as chat.tsx keeps it. */
export type MissionSuggestion = {
  readonly goal: string
  readonly complexity: string
  readonly agentId?: number | string
}

type Fields = Readonly<Record<string, unknown>>

function recordOf(value: unknown): Fields | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Fields) : null
}

function agentIdOf(value: unknown): number | string | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  return typeof value === 'string' && value ? value : undefined
}

/** The suggestion a forwarded onData part carries; null for any other part (or one without a goal). */
export function suggestionFromData(part: unknown): MissionSuggestion | null {
  const forwarded = recordOf(part)
  const data = recordOf(forwarded?.data)
  if (forwarded?.type !== 'mission-suggestion' || data === null) return null
  const goal = typeof data.goal === 'string' ? data.goal.trim() : ''
  if (!goal) return null
  const complexity = typeof data.complexity === 'string' ? data.complexity : ''
  const agentId = agentIdOf(data.agent_id)
  return agentId === undefined ? { goal, complexity } : { goal, complexity, agentId }
}

/** The newest messages as {role, content}, as chat.tsx builds recentMessages. */
export function recentMessages(messages: readonly ChatMessage[]): Array<{ role: string; content: string }> {
  return messages.slice(-CONTEXT_MESSAGES_KEPT).map((message) => ({ role: message.role, content: messageText(message) }))
}
