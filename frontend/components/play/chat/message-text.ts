import type { ChatMessage } from '@/types'

/**
 * PRD-260 §4 — the core's chat messages (types/chat.ts ChatMessage, as
 * useChat and getChatMessages hand them over) read for Play's chat: the words
 * of a message, the conversation's name, the newest reply, the lines a reply
 * shows above its answer. Pure. (The frontend compiles against ES2020, so no
 * findLast: the newest is found from the end by hand.)
 */

export const NEW_CHAT = 'New chat'
const TITLE_CHARS = 60

/** A part as it arrives: useChat's own and the stored ones share `type`. */
type LoosePart = { readonly type?: unknown; readonly text?: unknown; readonly narration?: unknown }

function partsOf(message: ChatMessage): readonly LoosePart[] {
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

/** The input of the newest tool call still running on the reply, or undefined. */
export function runningToolInput(reply: ChatMessage | undefined): unknown {
  return newest(reply?.toolCalls ?? [], (call) => call.state === 'running')?.input
}
