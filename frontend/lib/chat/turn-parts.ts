/**
 * PRD-268 §6.3, §6.6, K12, K13 (US-W03a) — what a chat turn left the person to act on, beside its text.
 *
 * The backend builds the turn's parts once its tool loop has ended (consumers/chatbot/turn_parts.py):
 * `{type: 'reply_chips@1', replies}`, the two or three short answers Auto offered as taps, and one
 * `{type: 'follow_up@1', follow_up_id, at}` for each follow-up the turn scheduled. Live, they come once,
 * before the answer, on a `tool-data` frame under `turn_parts`, and go on the reply being streamed as its
 * `replyChips` and `followUps` (as the receipts do: ./receipts.ts). Reloaded, the saved message carries them
 * after its text part. Either way they are read here off the wire, with checks: a part that is not what it
 * says is left out, never thrown on, and the rest are kept. The chips stay text, and one that reads as a
 * slash command is no reply (the composer runs those). The voice and the theme chats read a message's text
 * parts only (./message-text.ts), so neither says the chips. Nothing here mutates what it reads.
 */
import { recordOf } from '@/lib/proactive/wire'
import type { ChatMessage, TurnFollowUp } from '@/types'

export const REPLY_CHIPS_PART = 'reply_chips@1'
export const FOLLOW_UP_PART = 'follow_up@1'
/** The `tool-data` key the backend puts the turn's parts under (consumers/chatbot/turn_parts.py LIVE_KEY). */
export const LIVE_KEY = 'turn_parts'
/** The composer runs a message that starts with this as a command (`/mission …`): a chip never does. */
const SLASH_COMMAND = '/'

export type LiveTurnParts = Pick<ChatMessage, 'replyChips' | 'followUps'>

/** The replies a `reply_chips@1` part offers, trimmed, as text; undefined for any other value. */
function chipsOfPart(value: unknown): string[] | undefined {
  const part = recordOf(value)
  if (part?.type !== REPLY_CHIPS_PART || !Array.isArray(part.replies)) return undefined
  const replies: readonly unknown[] = part.replies
  if (!replies.every((reply) => typeof reply === 'string')) return undefined
  return (replies as readonly string[])
    .map((reply) => reply.trim())
    .filter((reply) => reply !== '' && !reply.startsWith(SLASH_COMMAND))
}

/** The follow-up a `follow_up@1` part names: a row id and a time; undefined for any other value. */
function followUpOfPart(value: unknown): TurnFollowUp | undefined {
  const part = recordOf(value)
  if (part?.type !== FOLLOW_UP_PART) return undefined
  const { follow_up_id: followUpId, at } = part
  if (typeof followUpId !== 'number' || !Number.isSafeInteger(followUpId) || followUpId <= 0) return undefined
  if (typeof at !== 'string' || Number.isNaN(Date.parse(at))) return undefined
  return { followUpId, at }
}

/** The turn's parts in a list of parts, live or saved: the first chips part with a reply left, and each follow-up once. */
export function turnPartsOf(parts: unknown): LiveTurnParts {
  const list: readonly unknown[] = Array.isArray(parts) ? parts : []
  const replyChips = list.map(chipsOfPart).find((chips) => chips !== undefined && chips.length > 0)
  const named = list.map(followUpOfPart).filter((followUp): followUp is TurnFollowUp => followUp !== undefined)
  const followUps = named.filter((followUp, index) => named.findIndex((other) => other.followUpId === followUp.followUpId) === index)
  return {
    ...(replyChips && replyChips.length > 0 ? { replyChips } : {}),
    ...(followUps.length > 0 ? { followUps } : {}),
  }
}

/** What a `tool-data` part forwarded by useChat's data callback sets on the live reply, when it carries the
 * turn's parts; undefined for any other part, and for a frame with none that can be read. */
export function liveTurnPartsFromToolData(part: unknown): LiveTurnParts | undefined {
  const forwarded = recordOf(part)
  if (forwarded?.type !== 'tool-data') return undefined
  const live = turnPartsOf(recordOf(forwarded.data)?.[LIVE_KEY])
  return live.replyChips || live.followUps ? live : undefined
}

/** `messages` with the turn's live parts on the reply being streamed (the last assistant message). Never mutates. */
export function withTurnPartsOnTheReply(messages: ChatMessage[], live: LiveTurnParts): ChatMessage[] {
  const at = messages.findLastIndex((m) => m.role === 'assistant')
  return at < 0 ? messages : messages.map((m, i) => (i === at ? { ...m, ...live } : m))
}

type WithTurnParts = Pick<ChatMessage, 'role' | 'parts' | 'replyChips' | 'followUps'>

/** The replies an assistant message offers: live from the frame, else from its saved part; undefined when none. */
export function replyChipsOfMessage(message: WithTurnParts): string[] | undefined {
  if (message.role !== 'assistant') return undefined
  return message.replyChips ?? turnPartsOf(message.parts).replyChips
}

/** The follow-ups an assistant message's turn scheduled: live from the frame, else from its saved parts. */
export function followUpsOfMessage(message: WithTurnParts): TurnFollowUp[] {
  if (message.role !== 'assistant') return []
  return message.followUps ?? turnPartsOf(message.parts).followUps ?? []
}
