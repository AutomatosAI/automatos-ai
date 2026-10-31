import type { FiledApproval } from '@/lib/chat/approvals'
import type { DeliverableRef } from '@/lib/chat/deliverables'
import type { MissionSuggestion } from '@/lib/chat/suggestion'
import type { ChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'
import { CHAT_HREF } from '@/lib/ticket-links'
import type { Chat, ChatMessage } from '@/types'

/**
 * PRD-261 A-11a (M28) — the chat controller's view model: what a theme's chat
 * screen reads and what it calls, whatever it looks like. The conversation on
 * screen (deep link, cold start, the active pointer's conversation loaded), the
 * stream (useChatWithReceipts), Stop keeping the reply being written
 * (P260-FX-30), Try again, and a turn's details (its Deliverables, a Mission
 * suggestion, the approval cards it raised). Built from the PRD-260 chat's
 * hooks; types and pure builders only, the hooks are in conversation.ts,
 * turn.ts, approvals.ts and controller.ts.
 */

export const CHAT_ID_PARAM = 'chatId'
export const CONVERSATION_GONE = 'That conversation is no longer available'

export interface ChatConversationView {
  /** The chat screen's key: it changes only when another conversation is shown. */
  readonly key: string
  /** null for a new chat until the backend names it. */
  readonly chatId: string | null
  readonly title: string | null
  readonly messages: ChatMessage[]
  /** The server is still producing a reply for it (the page reloaded mid-turn). */
  readonly awaitingReply: boolean
}

/** Words for the message box; a new serial puts them in a fresh box, which takes focus. */
export type ChatDraft = { readonly text: string; readonly serial: number }

/**
 * A theme's own link to a new chat with words waiting in its message box
 * (`/chat?<param>=<words>`): read once, then dropped from the URL. The core's
 * chat has no such link; without one the parameter is ignored.
 */
export interface ChatDraftLink {
  readonly param: string
  /** The words the link asks for, or null (none, or nothing safe). */
  readonly read: (params: Pick<URLSearchParams, 'get'> | null) => string | null
}

export interface ChatConversationOptions {
  /** A count of New chat presses: a press while a new chat is on screen shows a fresh one. */
  readonly freshDrafts?: number
  /** Pass a module-level constant: a new object every render reads the link again. */
  readonly draftLink?: ChatDraftLink
}

export interface ChatConversation {
  /** null while the session hydrates, a link or a cold start settles, a conversation loads, or the pointer has just moved off it. */
  readonly view: ChatConversationView | null
  /** The backend named the new chat on screen: keep showing it. Call before the session store hears. */
  readonly adopt: (chatId: string) => void
  /** A link's words for the new chat's message box, until the chat screen takes them. */
  readonly draft: ChatDraft | null
  /** The chat screen put the draft in its box: later new chats start empty. */
  readonly takeDraft: () => void
}

/**
 * What a turn tells the theme beyond its messages, for a mascot or any other
 * reaction: a Deliverable is new, a card waits for the owner, none waits any
 * more, or the newest reply failed in words (an error part).
 */
export type ChatTurnEvent =
  | { readonly type: 'deliverable-added' }
  | { readonly type: 'approval-waiting' }
  | { readonly type: 'approval-settled' }
  | { readonly type: 'turn-failed'; readonly code: string | null }

/** The approvals a reply raised, and what the owner does with them on their cards. */
export interface ChatApprovals {
  /** The cards a reply raised and the owner hasn't closed. */
  readonly of: (replyId: string) => readonly FiledApproval[]
  /** The open cards whose reply is no longer in the conversation: shown after it. */
  readonly loose: readonly FiledApproval[]
  /** A card's approve or deny went through. */
  readonly answered: (key: string) => void
  /** A card closed (its X, or the Mission card itself after a decision). */
  readonly closed: (key: string) => void
  /** A card still waits for the owner. */
  readonly waiting: boolean
}

export interface ChatTurnOptions {
  /**
   * The conversation on screen ('' for a new chat), whose approval cards show: read when the chat screen
   * mounts (it is keyed by the conversation); a new chat's name comes through onChatNamed.
   */
  readonly chatId: string
  /** The conversation as the screen shows it (useChat's, with the replies you stopped kept). */
  readonly messages: readonly ChatMessage[]
  /** The reply is still being written. */
  readonly streaming: boolean
  /** The turn's events (a mascot's dispatch, mapped). */
  readonly onEvent?: (event: ChatTurnEvent) => void
}

export interface ChatTurn {
  /** Pass to useChatWithReceipts' onData beside the theme's own. */
  readonly onData: (part: unknown) => void
  /** A new message or Try again: the turn's suggestion goes, as in chat.tsx. */
  readonly onSend: () => void
  /** Pass to useChatWithReceipts' onChatIdUpdate: the backend named the new chat, and its cards go with the name. */
  readonly onChatNamed: (chatId: string) => void
  readonly suggestion: MissionSuggestion | null
  /**
   * The Deliverables a reply made live this session. A stored reply shows no Deliverable card:
   * receipts don't link Deliverables yet (receipts.py, decision F8, backend).
   */
  readonly deliverablesOf: (replyId: string) => readonly DeliverableRef[]
  /** The approvals the conversation's replies raised live this page session (chat.tsx's widgets are live-only too). */
  readonly approvals: ChatApprovals
}

export interface ChatStreamOptions {
  readonly view: ChatConversationView
  /** The backend named this new chat: the screen keeps it (useChatConversation's adopt). */
  readonly onChatIdAssigned?: (chatId: string) => void
}

/** The stream and the conversation the screen shows. */
export interface ChatStream {
  /** useChatWithReceipts as it is: useChat's own messages, status, errorCode, sendMessage, reload, stop. */
  readonly chat: ChatWithReceipts
  /** useChat's messages with the replies you stopped kept (P260-FX-30). */
  readonly messages: readonly ChatMessage[]
  /** Keep the reply being written, then stop the chat (which also cancels the turn on the server). */
  readonly stop: () => void
  /** Stop ended this reply. */
  readonly isStopped: (replyId: string) => boolean
  /** Hand the stream's data parts and the new chat's name on: call from an effect (useStreamHandlers). */
  readonly connect: (handlers: ChatStreamHandlers) => void
}

export interface ChatStreamHandlers {
  readonly onData: (part: unknown) => void
  readonly onChatNamed: (chatId: string) => void
}

/** Send, Try again and Stop, each with the turn's own bookkeeping. */
export interface ChatActions {
  /** Send the words; false (nothing sent) while a turn runs. */
  readonly send: (text: string) => boolean
  /** Ask the owner's last message again; false (nothing asked) while a turn runs. */
  readonly retry: () => boolean
  /** Keep the reply being written, then stop. */
  readonly stop: () => void
}

export interface ChatControllerOptions extends ChatStreamOptions {
  /** The theme's own reading of each data part (a mascot's), before the turn's. */
  readonly onData?: (part: unknown) => void
  readonly onEvent?: (event: ChatTurnEvent) => void
}

/** Everything a theme's chat screen needs for one conversation. */
export interface ChatControllerView extends ChatActions {
  /** The conversation as shown: the stopped replies kept, a voice session's words in progress at the end. */
  readonly messages: readonly ChatMessage[]
  /** useChat's own messages and the voice session's words (a stopped reply it dropped is not here). */
  readonly liveMessages: readonly ChatMessage[]
  readonly status: ChatWithReceipts['status']
  readonly errorCode: string | null
  readonly streaming: boolean
  readonly isStopped: (replyId: string) => boolean
  readonly suggestion: MissionSuggestion | null
  readonly deliverablesOf: (replyId: string) => readonly DeliverableRef[]
  readonly approvals: ChatApprovals
}

/** `/chat` with the query as it is, less the named parameters. */
export function chatUrlWithout(params: URLSearchParams | null, ...names: readonly string[]): string {
  const next = new URLSearchParams(params?.toString() ?? '')
  names.forEach((name) => next.delete(name))
  const query = next.toString()
  return query ? `${CHAT_HREF}?${query}` : CHAT_HREF
}

export function draftView(serial: number): ChatConversationView {
  return { key: `draft-${serial}`, chatId: null, title: null, messages: [], awaitingReply: false }
}

export function storedView(chatId: string, chat: Chat, messages: ChatMessage[]): ChatConversationView {
  return { key: chatId, chatId, title: chat.title ?? null, messages, awaitingReply: Boolean(chat.turnInFlight) }
}
