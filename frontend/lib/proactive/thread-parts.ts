/**
 * PRD-268 §6.5.1, §6.12.4, K4(4) (US-W01b) — an Auto-thread message's cards, and what `Message` still draws.
 *
 * A background post carries its card's data after its text part (services/chat_parts.py): `brief@1` (the day's
 * brief) and `suggestion@1` (a suggestion's id). The thread draws each as a card (components/proactive/thread-cards.tsx)
 * and hands `Message`, which is never touched, a copy without those parts. A message whose brief is drawn also loses
 * its text part, because the card holds the prose. A `brief@1` part that isn't a Brief@1 draws no card, so its text
 * stays. Every other part reaches `Message` as it was, which still drops the types it doesn't know. A message with no
 * card part comes back as the same object.
 */
import type { ChatMessage } from '@/types'
import type { Brief } from './types'
import { BRIEF_PART, SUGGESTION_PART } from './types'
import { briefOfParts, isProactiveId, recordOf } from './wire'

const CARD_PARTS: ReadonlySet<unknown> = new Set([BRIEF_PART, SUGGESTION_PART])
const TEXT_PART = 'text'

export interface ThreadParts {
  /** The brief to draw as a card, or null. */
  readonly brief: Brief | null
  /** The suggestions to draw as cards, by id, each once. */
  readonly suggestionIds: readonly string[]
  /** What `Message` draws: the message itself, or a copy without the parts drawn as cards. */
  readonly message: ChatMessage
}

type MessageWithSource = ChatMessage & { source?: unknown; content?: unknown }

/** The suggestion each `suggestion@1` part names: its own id, else the message's `source.suggestion_id`. */
function suggestionIdsOf(parts: readonly unknown[], source: unknown): string[] {
  const fallback = recordOf(source)?.suggestion_id
  const ids = parts
    .map(recordOf)
    .filter((part) => part?.type === SUGGESTION_PART)
    .map((part) => (isProactiveId(part?.suggestion_id) ? part?.suggestion_id : fallback))
    .filter(isProactiveId)
  return ids.filter((id, index) => ids.indexOf(id) === index)
}

/** The message's cards, and the copy `Message` draws in its place. */
export function threadPartsOf(message: ChatMessage): ThreadParts {
  const parts: readonly unknown[] = message.role === 'assistant' && Array.isArray(message.parts) ? message.parts : []
  if (!parts.some((part) => CARD_PARTS.has(recordOf(part)?.type))) return { brief: null, suggestionIds: [], message }
  const brief = briefOfParts(parts)
  const kept = parts.filter((part) => {
    const type = recordOf(part)?.type
    return !CARD_PARTS.has(type) && !(brief !== null && type === TEXT_PART)
  })
  const copy: MessageWithSource = { ...message, parts: kept as ChatMessage['parts'] }
  const shown: MessageWithSource = brief !== null && 'content' in message ? { ...copy, content: '' } : copy
  return { brief, suggestionIds: suggestionIdsOf(parts, (message as MessageWithSource).source), message: shown }
}
