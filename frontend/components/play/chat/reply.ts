import type { HostChatMessage, HostDeliverable, MissionSuggestionMessage } from '@automatos/theme-sdk'

import { NO_REPLY, type ReplyProgress } from '../../../lib/mascot/events'

/**
 * PRD-260 §4, PRD-261 A-14a — the conversation as Play's chat reads it, on the
 * SDK's messages (useChatController's view model, lib/theme-sdk/chat.ts): the
 * newest reply, the reply being written as Auto reads it, the input of its
 * running tool call, the Deliverables a reply shows, and the recent messages a
 * Mission is planned from. The same readings the core's lib/chat/message-text.ts
 * and lib/chat/suggestion.ts make of the core's messages (copied into Play,
 * docs/theme-host/sdk-surface-map.md; lib/__tests__/prd261-port-chat.test.ts
 * holds each to the core's). Pure. (ES2020: no findLast.)
 */

/** chat.tsx sends the newest five messages as the Mission's context (the core's CONTEXT_MESSAGES_KEPT). */
export const CONTEXT_MESSAGES_KEPT = 5

function newest<T>(items: readonly T[], keep: (item: T) => boolean): T | undefined {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (keep(items[index])) return items[index]
  }
  return undefined
}

/**
 * The conversation as the chat hook holds it: the SDK's messages keep a reply
 * you stopped (P260-FX-30), which the hook itself dropped. Auto, the caption
 * and what follows the conversation read this one, as they read the hook's.
 */
export function liveMessages(messages: readonly HostChatMessage[]): readonly HostChatMessage[] {
  return messages.some((message) => message.stopped) ? messages.filter((message) => !message.stopped) : messages
}

/** The newest reply from Auto. */
export function latestReply(messages: readonly HostChatMessage[]): HostChatMessage | undefined {
  return newest(messages, (message) => message.role === 'assistant')
}

/** The input of the newest tool call still running on the reply (its query, page or title). */
export function runningToolInput(reply: HostChatMessage | undefined): unknown {
  return newest(reply?.toolCalls ?? [], (call) => call.state === 'running')?.input
}

/**
 * The reply being written, for Auto (lib/mascot/events.ts): the newest message
 * when it is a reply, whether its words have started, whether one of its tool
 * calls still runs, how long the model's reasoning on it is and whether a cap
 * ended its turn. A turn still awaiting its reply has none.
 */
export function replyProgress(messages: readonly HostChatMessage[]): ReplyProgress {
  const last = messages[messages.length - 1]
  if (last?.role !== 'assistant') return NO_REPLY
  return {
    id: last.id,
    hasText: last.text.trim().length > 0,
    toolRunning: last.toolCalls.some((call) => call.state === 'running'),
    reasoningChars: last.reasoningChars,
    limitReached: Boolean(last.limitReached),
  }
}

/** The newest messages as {role, content}, as chat.tsx builds the Mission suggestion's recentMessages. */
export function recentMessages(messages: readonly HostChatMessage[]): MissionSuggestionMessage[] {
  return messages.slice(-CONTEXT_MESSAGES_KEPT).map((message) => ({ role: message.role, content: message.text }))
}

/** `known` with the new ones from `found` after it; one per id, the first one seen wins. Never mutates. */
export function mergeDeliverables(
  known: readonly HostDeliverable[],
  found: readonly HostDeliverable[],
): readonly HostDeliverable[] {
  const fresh = found.filter(
    (ref, index) => !known.some((seen) => seen.id === ref.id) && found.findIndex((other) => other.id === ref.id) === index,
  )
  return fresh.length > 0 ? [...known, ...fresh] : known
}

/** The Deliverables a reply shows: the ones it made live, then the ones its done receipts link to. */
export function replyDeliverables(message: HostChatMessage): readonly HostDeliverable[] {
  return mergeDeliverables(message.deliverables, message.linkedDeliverables)
}
