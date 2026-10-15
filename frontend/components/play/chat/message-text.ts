import { upsertTaskCard } from '@/lib/chat/tool-calls'
import { NO_REPLY, type ReplyProgress } from '@/lib/mascot/events'
import type { ChatMessage, TaskCardData } from '@/types'

/**
 * PRD-260 §4 — the core's chat messages (types/chat.ts ChatMessage, as
 * useChat and getChatMessages hand them over) read for Play's chat: the words
 * of a message, the conversation's name, the newest reply, the lines a reply
 * shows above its answer, the tickets it filed or checked, a failed turn the
 * server saved, who wrote a reply when the turn didn't, and the reply being
 * written as Auto reads it. Pure.
 * (The frontend compiles against ES2020, so no findLast: the newest is found
 * from the end by hand.)
 */

export const NEW_CHAT = 'New chat'
const TITLE_CHARS = 60
/** A failed turn the backend saved (consumers/chatbot/service.py _persist_turn_error) carries this origin. */
const TURN_ERROR_ORIGIN = 'turn_error'
/** The core puts this in front of a saved turn error's words; the snag box has its own mark. */
const WARNING_PREFIX = /^\s*⚠️?\s*/u
/** A saved turn error with no words of its own (the Play reference's SAVED_ERROR). */
export const SAVED_ERROR = 'This answer failed.'

/** A part as it arrives: useChat's own and the stored ones share `type`. */
type LoosePart = {
  readonly type?: unknown
  readonly text?: unknown
  readonly narration?: unknown
  readonly card?: unknown
}

function partsOf(message: Pick<ChatMessage, 'parts'>): readonly LoosePart[] {
  return (message.parts ?? []) as readonly LoosePart[]
}

function newest<T>(items: readonly T[], keep: (item: T) => boolean): T | undefined {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (keep(items[index])) return items[index]
  }
  return undefined
}

/** The text parts joined, or the message's content when it has none (useChat keeps both). */
export function messageText(message: ChatMessage): string {
  const text = partsOf(message)
    .map((part) => (part.type === 'text' && typeof part.text === 'string' ? part.text : ''))
    .filter(Boolean)
    .join('\n')
  if (text) return text
  const content = (message as unknown as { content?: unknown }).content
  return typeof content === 'string' ? content : ''
}

/** The row's persisted provenance (messages.source), which getChatMessages keeps on the message. */
function storedOrigin(message: ChatMessage): unknown {
  const source = (message as unknown as { source?: unknown }).source
  return typeof source === 'object' && source !== null ? (source as { origin?: unknown }).origin : undefined
}

/**
 * A failed turn as the server saved it (a reopened conversation, or a turn that ended after the
 * page let go): its words without the core's ⚠️, for the snag box. Null for any other message.
 */
export function savedTurnError(message: ChatMessage): string | null {
  if (storedOrigin(message) !== TURN_ERROR_ORIGIN) return null
  return messageText(message).replace(WARNING_PREFIX, '').trim() || SAVED_ERROR
}

/**
 * Who wrote a reply when the turn itself didn't ("Auto · background", PRD-205): the label
 * getChatMessages puts in metadata.source, as Studio's badge shows it. None for a saved turn
 * error, whose label ("Error") the snag box already says.
 */
export function provenanceOf(message: ChatMessage): string | null {
  const label: unknown = message.metadata?.source
  if (typeof label !== 'string' || !label.trim() || storedOrigin(message) === TURN_ERROR_ORIGIN) return null
  return label.trim()
}

/** The conversation's name: its stored title, else its first message, else "New chat". */
export function chatTitle(messages: readonly ChatMessage[], stored?: string | null): string {
  if (stored?.trim()) return stored.trim()
  const first = messages.find((message) => message.role === 'user')
  const words = first ? messageText(first).replace(/\s+/g, ' ').trim() : ''
  if (!words) return NEW_CHAT
  return words.length <= TITLE_CHARS ? words : `${words.slice(0, TITLE_CHARS - 1).trimEnd()}…`
}

/** The newest assistant message, if any. */
export function latestReply(messages: readonly ChatMessage[]): ChatMessage | undefined {
  return newest(messages, (message) => message.role === 'assistant')
}

/** What Auto said before calling tools (the narration parts), then the progress from inside a tool. */
export function activityLines(message: ChatMessage): string[] {
  const narration = partsOf(message).flatMap((part) =>
    part.type === 'narration' && typeof part.narration === 'string' ? [part.narration] : [],
  )
  return [...narration, ...(message.progress ?? [])].filter((line) => line.trim().length > 0)
}

/** A stored ticket card as the backend saves it (orchestrator/services/ticket_cards.py): an object with its numeric id. */
function isTaskCard(card: unknown): card is TaskCardData {
  return typeof card === 'object' && card !== null && typeof (card as { id?: unknown }).id === 'number'
}

/**
 * The tickets a reply filed or checked, as Studio's message shows them: its
 * stored `task_card` parts (PRD-239 S2), then the cards the stream put on it
 * (PRD-238 S6), one per ticket, the newest data winning (the core's upsertTaskCard).
 */
export function taskCardsOf(message: Pick<ChatMessage, 'parts' | 'taskCards'>): TaskCardData[] {
  const stored = partsOf(message).flatMap((part) => (part.type === 'task_card' && isTaskCard(part.card) ? [part.card] : []))
  return [...stored, ...(message.taskCards ?? [])].reduce<TaskCardData[]>((cards, card) => upsertTaskCard(cards, card), [])
}

/** The input of the newest tool call still running on the reply, or undefined. */
export function runningToolInput(reply: ChatMessage | undefined): unknown {
  return newest(reply?.toolCalls ?? [], (call) => call.state === 'running')?.input
}

/**
 * The reply being written, for Auto (lib/mascot/events.ts): the newest message
 * when it is a reply (useChat appends the empty reply as it sends), whether its
 * words have started, and whether one of its tool calls still runs. A turn still
 * awaiting its reply has none, so the previous turn's words never end thinking.
 */
export function replyProgress(messages: readonly ChatMessage[]): ReplyProgress {
  const last = messages[messages.length - 1]
  if (last?.role !== 'assistant') return NO_REPLY
  return {
    id: last.id,
    hasText: messageText(last).trim().length > 0,
    toolRunning: (last.toolCalls ?? []).some((call) => call.state === 'running'),
  }
}
