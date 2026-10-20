/**
 * PRD-261 A-11b — the SDK chat area's copies (lib/theme-sdk/chat.ts): the host's chat controller
 * (lib/chat-controller) read into the frozen, theme-neutral shapes chat.ts declares. Pure, apart
 * from the bound answered/closed callbacks of an approval card.
 *
 * A conversation handed to a theme carries the controller's own view and the `adopt` callback under
 * a module-private symbol, so only a conversation the SDK handed out opens a stream: a theme can
 * read `key`, `chatId` and `title`, and cannot forge the rest.
 */
import type { ChatApprovals, ChatControllerView, ChatConversationView } from '@/lib/chat-controller/view-model'
import type { FiledApproval } from '@/lib/chat/approvals'
import { deliverablesFromReceipts } from '@/lib/chat/deliverables'
import {
  activityLines,
  chatTitle,
  latestReply,
  messageText,
  provenanceOf,
  savedTurnError,
  taskCardsOf,
} from '@/lib/chat/message-text'
import { receiptsOf } from '@/lib/chat/receipts'
import type { Chat, ChatMessage, ToolCall } from '@/types'

import type { HostChatApproval, HostChatConversation, HostChatMessage, HostChatSummary, HostToolCall } from '../chat'

/** What the SDK keeps on a conversation it handed out. */
export interface OpenedConversation {
  readonly view: ChatConversationView
  /** The backend named this new chat: the screen keeps it. */
  readonly adopt: (chatId: string) => void
}

/** The messages as a theme draws them, and the conversation's title. */
export interface HostMessages {
  readonly title: string
  readonly list: readonly HostChatMessage[]
}

/** What the copies read off the controller's view model. */
export type ControllerReading = Pick<ChatControllerView, 'messages' | 'streaming' | 'isStopped' | 'deliverablesOf' | 'approvals'>

const OPENED: unique symbol = Symbol('automatos.theme-sdk.chat.opened')
const NO_LINES: readonly string[] = Object.freeze([])

type WithOpened = HostChatConversation & { readonly [OPENED]?: OpenedConversation }

/** The conversation a theme holds, with the controller's view and `adopt` out of its reach. */
export function hostConversation(view: ChatConversationView, adopt: (chatId: string) => void): HostChatConversation {
  const conversation = { key: view.key, chatId: view.chatId, title: view.title }
  Object.defineProperty(conversation, OPENED, { value: Object.freeze({ view, adopt }), enumerable: false })
  return Object.freeze(conversation)
}

/** What the SDK kept on a conversation it handed out; null for anything else (no argument, options, a forgery). */
export function openedConversationOf(value: unknown): OpenedConversation | null {
  if (typeof value !== 'object' || value === null) return null
  return (value as WithOpened)[OPENED] ?? null
}

/** Something shaped like a conversation (it has a `key`) that the SDK did not hand out. */
export function isForgedConversation(value: unknown): boolean {
  return typeof value === 'object' && value !== null && 'key' in value && openedConversationOf(value) === null
}

/** An approval card as a theme places it: its data for the host's card, and its two outcomes bound. */
export function hostApproval(card: FiledApproval, chatId: string, approvals: ChatApprovals): HostChatApproval {
  const { key } = card
  return Object.freeze({
    key,
    kind: card.kind,
    replyId: card.replyId,
    status: card.status,
    filedAt: card.filedAt,
    chatId,
    data: card.data,
    answered: () => approvals.answered(key),
    closed: () => approvals.closed(key),
  })
}

function hostError(message: ChatMessage): HostChatMessage['error'] {
  const { error } = message
  return error ? Object.freeze({ message: error.message, code: error.code ?? null }) : null
}

function hostToolCall(call: ToolCall): HostToolCall {
  const { toolCallId, toolName, state, input, error, durationMs, summary } = call
  return Object.freeze({
    toolCallId,
    toolName,
    state,
    input,
    error,
    durationMs,
    summary,
    skipped: call.skipped === true,
    waiting: call.waiting === true,
  })
}

/** What a reply carries beyond its words and cards: its tool calls, who wrote it, a saved failure, a cap, its reasoning. */
type ReplyDetails = Pick<
  HostChatMessage,
  'toolCalls' | 'provenance' | 'savedError' | 'limitReached' | 'reasoningChars' | 'withdrawOffer' | 'linkedDeliverables'
>

function hostWithdrawOffer(offer: ChatMessage['withdrawOffer']): HostChatMessage['withdrawOffer'] {
  if (!offer) return null
  const requests = Object.freeze(offer.requests.map((request) => Object.freeze({ ...request })))
  return Object.freeze({ message: offer.message, requests })
}

function replyDetails(message: ChatMessage): ReplyDetails {
  return {
    toolCalls: Object.freeze((message.toolCalls ?? []).map(hostToolCall)),
    provenance: provenanceOf(message),
    savedError: savedTurnError(message),
    limitReached: message.limitReached?.message || null,
    reasoningChars: message.reasoning?.length ?? 0,
    withdrawOffer: hostWithdrawOffer(message.withdrawOffer),
    linkedDeliverables: Object.freeze(deliverablesFromReceipts(receiptsOf(message))),
  }
}

function hostMessage(message: ChatMessage, core: ControllerReading, chatId: string, writingId: string | null): HostChatMessage {
  const { id } = message
  return Object.freeze({
    ...replyDetails(message),
    id,
    role: message.role,
    text: messageText(message),
    activity: Object.freeze(activityLines(message)),
    error: hostError(message),
    writing: id === writingId,
    stopped: core.isStopped(id),
    receipts: receiptsOf(message) ?? null,
    above: message.receiptsAbove ?? NO_LINES,
    deliverables: core.deliverablesOf(id),
    approvals: Object.freeze(core.approvals.of(id).map((card) => hostApproval(card, chatId, core.approvals))),
    taskCards: Object.freeze(taskCardsOf(message)),
  })
}

/** The conversation as shown (a stopped reply kept), each message copied for a theme, and its title. */
export function hostMessages(core: ControllerReading, view: ChatConversationView): HostMessages {
  const reply = core.streaming ? latestReply(core.messages) : undefined
  const writingId = reply?.id ?? null
  const chatId = view.chatId ?? ''
  return Object.freeze({
    title: chatTitle(core.messages, view.title),
    list: Object.freeze(core.messages.map((message) => hostMessage(message, core, chatId, writingId))),
  })
}

function hostSummary(chat: Chat): HostChatSummary {
  return Object.freeze({ id: chat.id, title: chat.title, updatedAt: chat.updatedAt, preview: chat.lastMessagePreview ?? null })
}

/** A list of conversations, copied for a theme. */
export function hostSummaries(chats: readonly Chat[]): readonly HostChatSummary[] {
  return Object.freeze(chats.map(hostSummary))
}
