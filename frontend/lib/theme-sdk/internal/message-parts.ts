/**
 * PRD-268 §6.6, §6.11.2, K17 (SDK 1.4.0) — a message's parts as the SDK chat area hands them to a theme
 * (HostChatMessage.knownParts, lib/theme-sdk/chat.ts), read off the core message by internal/chat-view.ts.
 *
 * The parts a theme can read are the ones core posts:
 * - `text`: the words, on any message;
 * - `brief@1`: the day's brief (services/brief/model.py), checked as the host's thread checks it (lib/proactive/wire.ts
 *   briefOf);
 * - `suggestion@1`: a suggestion's id (services/proactive/delivery.py): the part's own id, else the message's
 *   `source.suggestion_id`, as the host's thread card reads it (lib/proactive/thread-parts.ts);
 * - `reply_chips@1`: `{type, replies}`, the two or three short answers Auto offered (PRD-268 US-C12,
 *   consumers/chatbot/turn_parts.py). They are read by the web chat's own reader (lib/chat/turn-parts.ts
 *   replyChipsOfMessage): a reply being streamed has them from its `turn_parts` frame before any part is saved, a
 *   reloaded one from its saved part. The chips a theme gets are the ones the host's ReplyChips draws from them
 *   (components/proactive/reply-chips.tsx replyChipsOf: trimmed, at most 24 characters, no repeats, the first
 *   three), and the part only while two or more remain. It stands where the saved message has its chips part, else
 *   (a live reply, not saved yet) last, as the saved parts end with it; once per message.
 *
 * brief@1, suggestion@1 and reply_chips@1 count only on an assistant message, as the host draws them. A part of any
 * other type, or one that isn't what its type says, is left out, never thrown on; the rest keep the message's order.
 * Everything handed back is a frozen copy; nothing here mutates what it reads.
 */
import { REPLY_CHIPS_MIN, replyChipsOf } from '@/components/proactive/reply-chips'
import { REPLY_CHIPS_PART, replyChipsOfMessage, turnPartsOf } from '@/lib/chat/turn-parts'
import { BRIEF_PART, SUGGESTION_PART } from '@/lib/proactive/types'
import { briefOf, isProactiveId, recordOf } from '@/lib/proactive/wire'
import type { ChatMessage } from '@/types'

import type { HostBriefPart, HostMessagePart, HostReplyChipsPart, HostSuggestionPart, HostTextPart } from '../chat'
import type { Brief } from './proactive-shapes'
import { frozenCopy } from './query-view'

type Fields = NonNullable<ReturnType<typeof recordOf>>

/** What every part of one message is read with. */
interface MessageReading {
  readonly assistant: boolean
  readonly source: unknown
  /** The message's chips, read once (replyChipsPart): every saved chips part of it stands for them. */
  readonly chips: HostReplyChipsPart | null
}

interface PartReader {
  readonly read: (part: Fields, reading: MessageReading) => HostMessagePart | null
  /** Only Auto's messages carry it, as the host draws it. */
  readonly assistantOnly: boolean
}

export const TEXT_PART = 'text'
const ASSISTANT = 'assistant'
const NO_PARTS: readonly HostMessagePart[] = Object.freeze([])

function textPart(part: Fields): HostTextPart | null {
  const { text } = part
  return typeof text === 'string' ? Object.freeze<HostTextPart>({ type: TEXT_PART, text }) : null
}

function briefPart(part: Fields): HostBriefPart | null {
  const brief = briefOf(part.brief)
  return brief === null ? null : Object.freeze<HostBriefPart>({ type: BRIEF_PART, brief: frozenCopy(brief) as Brief })
}

function suggestionPart(part: Fields, { source }: MessageReading): HostSuggestionPart | null {
  const own = part.suggestion_id
  const id = isProactiveId(own) ? own : recordOf(source)?.suggestion_id
  return isProactiveId(id) ? Object.freeze<HostSuggestionPart>({ type: SUGGESTION_PART, suggestion_id: id }) : null
}

/** The chips the host's ReplyChips draws under the message, from the web chat's reader: live, else saved. */
function replyChipsPart(message: ChatMessage): HostReplyChipsPart | null {
  const chips = replyChipsOf(replyChipsOfMessage(message) ?? [])
  if (chips.length < REPLY_CHIPS_MIN) return null
  return Object.freeze<HostReplyChipsPart>({ type: REPLY_CHIPS_PART, replies: Object.freeze([...chips]) })
}

/** A saved chips part that reads as chips (the same reader) stands for the message's chips; a malformed one is left out. */
function savedChipsPart(part: Fields, { chips }: MessageReading): HostReplyChipsPart | null {
  return turnPartsOf([part]).replyChips ? chips : null
}

/** The reader of each part type a theme may read. */
const READERS: ReadonlyMap<string, PartReader> = new Map<string, PartReader>([
  [TEXT_PART, { read: textPart, assistantOnly: false }],
  [BRIEF_PART, { read: briefPart, assistantOnly: true }],
  [SUGGESTION_PART, { read: suggestionPart, assistantOnly: true }],
  [REPLY_CHIPS_PART, { read: savedChipsPart, assistantOnly: true }],
])

/** One part as a theme reads it, or null when its type is unknown, not this message's to carry, or it is malformed. */
function knownPart(value: unknown, reading: MessageReading): HostMessagePart | null {
  const part = recordOf(value)
  const type = part?.type
  const reader = typeof type === 'string' ? READERS.get(type) : undefined
  if (part === null || reader === undefined || (reader.assistantOnly && !reading.assistant)) return null
  return reader.read(part, reading)
}

/** The message's chips once: where its saved chips part stands, else last (a live reply's, not saved yet). */
function withChipsOnce(read: readonly HostMessagePart[], chips: HostReplyChipsPart | null): readonly HostMessagePart[] {
  if (chips === null) return read
  const at = read.indexOf(chips)
  return at < 0 ? [...read, chips] : read.filter((part, index) => part !== chips || index === at)
}

/** The message's parts of the types a theme can read, checked and frozen, in the message's order. */
export function hostParts(message: ChatMessage): readonly HostMessagePart[] {
  const parts: readonly unknown[] = Array.isArray(message.parts) ? message.parts : []
  const reading: MessageReading = {
    assistant: message.role === ASSISTANT,
    source: (message as ChatMessage & { source?: unknown }).source,
    chips: replyChipsPart(message),
  }
  const read = parts.map((part) => knownPart(part, reading)).filter((part): part is HostMessagePart => part !== null)
  const known = withChipsOnce(read, reading.chips)
  return known.length === 0 ? NO_PARTS : Object.freeze(known)
}
