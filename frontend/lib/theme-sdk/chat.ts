'use client'

/**
 * PRD-261 A-11b — SDK area `chat`: a theme's chat screen on the host's chat controller (A-11a,
 * lib/chat-controller), the conversation lists, New chat and the receipt lines. The names are the
 * ones A-10c planned (surface/chat.json, docs/theme-host/sdk-surface-map.md); the bodies are thin
 * wrappers, so every theme streams, stops, retries and files approvals exactly as Play does.
 *
 * useChatController is called twice, as Play's chat is built (play-chat.tsx, play-chat-view.tsx):
 *
 * - `useChatController()` in the chat page: the conversation on screen (a `?chatId=` link, a cold
 *   start on the newest conversation or a new chat, New chat pressed again), or null while it
 *   opens; with a theme's own draft link, the words waiting for the message box.
 * - `useChatController(conversation)` in a child keyed by `conversation.key`: that conversation's
 *   stream and turn as a view model (messages with their text, tool calls, receipts, Deliverables and
 *   approval cards, and what else a reply carries: who wrote it, a failure the server saved, a cap,
 *   a withdraw offer, PRD-261 A-14a; status; Send, Try again and Stop, which keeps the partial reply,
 *   P260-FX-30). Keying the child is what starts the stream afresh for another conversation, as
 *   Play's PlayChatView is keyed.
 *
 * `liveReceiptsOf(part)` reads the turn's receipts off one of onData's parts (the receipts frame the
 * backend sends after the tool loop) exactly as the controller reads them for the reply, so a theme
 * can react the moment they arrive (Play's Auto celebrates a done one, and a refused one is oops).
 *
 * A component calls it one way for its whole life: switching throws, instead of React's hook-order
 * error. Only a conversation the first call handed out opens a stream (it carries the host's own
 * state under a private symbol); a hand-made object is refused. Everything returned is a frozen,
 * theme-neutral copy: no store, no query client, no raw core message.
 *
 * Send and Try again start no turn on a page without the host's AI label (PRD-264 EUAI-11b,
 * internal/ai-label-gate.ts): they return false, as while a turn runs. A label a theme hides is still
 * on screen, drawn in the host's own layer, so it never stops a turn.
 */
import { useMemo, useRef } from 'react'

import { useConversationList as useCoreConversationList } from '@/hooks/use-conversation-list'
import { useRecentChats as useCoreRecentChats } from '@/hooks/use-recent-chats'
import { useStartChat as useCoreStartChat } from '@/hooks/use-start-chat'
import { useChatController as useCoreChatController } from '@/lib/chat-controller/controller'
import { useChatConversation } from '@/lib/chat-controller/conversation'
import type { ChatControllerView, ChatConversationView } from '@/lib/chat-controller/view-model'
import {
  LIVE_KEY,
  liveReceiptsFromToolData,
  movedSomethingOnTheBoard as coreMovedSomethingOnTheBoard,
  receiptLine as coreReceiptLine,
  safeLink,
} from '@/lib/chat/receipts'
import { CREDITS_EXHAUSTED_CODE } from '@/lib/credits'
import { useChatSessionStore } from '@/stores/chat-session-store'
import { useMissionStore } from '@/stores/mission-store'
import { useFreshDraftStore } from '@/stores/fresh-draft-store'

import type { WithdrawOffer } from './host-cards/chat-cards'
import { gatedByAiLabel } from './internal/ai-label-gate'
import {
  type OpenedConversation,
  hostApproval,
  hostConversation,
  hostMessages,
  hostSummaries,
  isForgedConversation,
  openedConversationOf,
} from './internal/chat-view'
import { isInternalHref } from './internal/href'

export type HostChatRole = 'system' | 'user' | 'assistant'
export type HostChatStatus = 'idle' | 'streaming' | 'error'

/** What one call of a turn did, written by the platform from the call that ran (PRD-256). */
export interface HostReceipt {
  readonly action: string
  readonly kind: 'read' | 'write'
  /** waiting: an ask for the owner's click, its approval card raised. */
  readonly status: 'done' | 'refused' | 'skipped' | 'waiting'
  readonly subject: string
  readonly effect: string
  readonly link: string | null
  readonly reason: string | null
}

/** The turn's receipts as one stream part carries them (liveReceiptsOf), before the reply shows them. */
export interface HostLiveReceipts {
  readonly receipts: readonly HostReceipt[]
  /** The lines the platform puts above the reply (a refused write, work not done). */
  readonly above: readonly string[]
  /** The model the frame names; null when it names none. */
  readonly model: string | null
}

/** A Deliverable a reply made live this page session. */
export interface HostDeliverable {
  readonly id: string
  readonly title: string
  readonly kind: 'document' | 'image' | 'video'
}

/** A Mission the turn suggests (the core's mission-suggestion part). */
export interface HostMissionSuggestion {
  readonly goal: string
  readonly complexity: string
  readonly agentId?: number | string
}

/** A ticket the reply filed or checked, for cards.TaskCard. */
export interface HostTaskCard {
  readonly id: number
  readonly number?: string | null
  readonly title: string
  readonly status: string
  readonly assigned_agent: string
  readonly runtime?: string | null
  readonly last_tool?: string | null
  readonly files_touched?: number
  readonly exit_reason?: string | null
  readonly denials?: number
  readonly started_at?: string | null
  readonly completed_at?: string | null
}

/** One tool call of a reply, as its tool lines show it (PRD-238 S3). */
export interface HostToolCall {
  readonly toolCallId: string
  readonly toolName: string
  readonly state: 'running' | 'completed' | 'error'
  /** The call's input as the model sent it (a query, a page, a title), unchecked. */
  readonly input?: unknown
  readonly error?: string
  readonly durationMs?: number
  /** The server's one-line result headline, never a raw payload. */
  readonly summary?: string
  /** The loop de-duplicated this call: it never ran. */
  readonly skipped: boolean
  /** An ask for the owner's click: it waits on its card, it did not fail. */
  readonly waiting: boolean
}

/** A reply that failed in words. */
export interface HostChatError {
  readonly message: string
  readonly code: string | null
}

/**
 * An approval a reply raised: a tool approval (a grant) or a Mission plan. The host draws it
 * (cards.ToolApproval, cards.MissionApproval: protected surfaces); the theme only places it.
 */
export interface HostChatApproval {
  readonly key: string
  readonly kind: 'tool' | 'mission'
  /** The reply that raised it. */
  readonly replyId: string
  /** waiting for the owner; answered (the card shows the outcome); closed (gone). */
  readonly status: 'waiting' | 'answered' | 'closed'
  readonly filedAt: Date
  /** The conversation it belongs to ('' for a new chat the backend has not named yet). */
  readonly chatId: string
  /** The card's own data, checked off the wire; the host's card reads it. */
  readonly data: unknown
  /** The card's decision went through. */
  readonly answered: () => void
  /** The card closed. */
  readonly closed: () => void
}

/** One message as a theme draws it. */
export interface HostChatMessage {
  readonly id: string
  readonly role: HostChatRole
  /** The words: the text parts, or the content when there are none. */
  readonly text: string
  /** The reply's activity lines (narration, a long tool's progress). */
  readonly activity: readonly string[]
  readonly error: HostChatError | null
  /** This reply is the one being written now. */
  readonly writing: boolean
  /** Stop ended this reply; its words so far stay (P260-FX-30). */
  readonly stopped: boolean
  /** The turn's receipts; null for a message from before PRD-256. */
  readonly receipts: readonly HostReceipt[] | null
  /** The lines the platform puts above the reply (a refused write, work not done). */
  readonly above: readonly string[]
  readonly deliverables: readonly HostDeliverable[]
  /** The approval cards this reply raised that the owner has not closed. */
  readonly approvals: readonly HostChatApproval[]
  readonly taskCards: readonly HostTaskCard[]
  /** The reply's tool calls, in the order they started. */
  readonly toolCalls: readonly HostToolCall[]
  /** Who wrote the reply when the turn didn't ("Auto · background"); null for a reply of the turn. */
  readonly provenance: string | null
  /** A failed turn the server saved, in its words: the reply is that failure, never an answer. */
  readonly savedError: string | null
  /** A cap that ended the reply's turn, in words; null when none did. */
  readonly limitReached: string | null
  /** The length of the model's reasoning on this reply: its growth is the model thinking. */
  readonly reasoningChars: number
  /** The owner said no while a request still waits on a yes: place it in cards.WithdrawOfferCard. */
  readonly withdrawOffer: WithdrawOffer | null
  /** The Deliverables the reply's done receipts link to, each id checked by the host. */
  readonly linkedDeliverables: readonly HostDeliverable[]
}

/** A conversation the chat page shows. Pass it back to useChatController in a child keyed by `key`. */
export interface HostChatConversation {
  /** Changes only when another conversation is shown. */
  readonly key: string
  /** null for a new chat until the backend names it. */
  readonly chatId: string | null
  readonly title: string | null
}

/** Words for the message box; a new serial puts them in a fresh box. */
export interface HostChatDraft {
  readonly text: string
  readonly serial: number
}

/** A theme's own link to a new chat with words waiting (`/chat?<param>=<words>`), read once. */
export interface HostChatDraftLink {
  readonly param: string
  /** The words the link asks for, or null (none, or nothing safe). */
  readonly read: (params: Pick<URLSearchParams, 'get'> | null) => string | null
}

export interface HostChatOnScreenOptions {
  /** Pass a module-level constant: a new object every render reads the link again. */
  readonly draftLink?: HostChatDraftLink
}

export interface HostChatOnScreen {
  /** null while the session loads, a link or a cold start settles, or a conversation loads. */
  readonly conversation: HostChatConversation | null
  /** A draft link's words, until the chat screen takes them. */
  readonly draft: HostChatDraft | null
  /** The words are in the box: later new chats start empty. */
  readonly takeDraft: () => void
}

/** What a turn tells the theme beyond its messages (a mascot's cues). */
export type HostChatTurnEvent =
  | { readonly type: 'deliverable-added' }
  | { readonly type: 'approval-waiting' }
  | { readonly type: 'approval-settled' }
  | { readonly type: 'turn-failed'; readonly code: string | null }

export interface HostChatControllerOptions {
  /** Each data part of the stream as it arrives, unchecked, before the turn reads it. */
  readonly onData?: (part: unknown) => void
  readonly onEvent?: (event: HostChatTurnEvent) => void
}

/** One conversation's chat: what the screen draws and what it calls. */
export interface HostChatController {
  readonly conversation: HostChatConversation
  /** The conversation's title, or its first words, or "New chat". */
  readonly title: string
  /** The conversation as shown: a reply you stopped stays. */
  readonly messages: readonly HostChatMessage[]
  readonly status: HostChatStatus
  readonly streaming: boolean
  readonly errorCode: string | null
  /**
   * The workspace's credits are used up: the send was refused with `credits_exhausted`
   * (cards.TrialExhaustedBanner). The name predates PRD-265, when the trial was the only credit.
   */
  readonly trialExhausted: boolean
  readonly suggestion: HostMissionSuggestion | null
  /** The Mission the chat is planning, once one is (cards.MissionCreatedCard). */
  readonly planningMissionId: string | null
  /** Open approval cards whose reply left the conversation (Try again, a failed request): shown after it. */
  readonly looseApprovals: readonly HostChatApproval[]
  /** A card still waits for the owner. */
  readonly approvalWaiting: boolean
  /** Sends the words; false (nothing sent) while a turn runs, or on a page with no host AI label (PRD-264). */
  readonly send: (text: string) => boolean
  /** Asks the owner's last message again; false while a turn runs, or on a page with no host AI label. */
  readonly retry: () => boolean
  /** Keeps the reply being written, then stops the turn (here and on the server). */
  readonly stop: () => void
}

/** A conversation in a list. */
export interface HostChatSummary {
  readonly id: string
  readonly title: string
  readonly updatedAt: string
  /** The latest message's words, cut by the server; null when none. */
  readonly preview: string | null
}

export type HostConversationListState =
  | { readonly status: 'loading' }
  | { readonly status: 'failed'; readonly retry: () => void }
  | { readonly status: 'loaded'; readonly chats: readonly HostChatSummary[] }

export interface HostRecentChats {
  readonly chats: readonly HostChatSummary[]
  /** The conversation the chat shows now; null for a new chat. */
  readonly activeChatId: string | null
}

export interface HostStartChat {
  /** A fresh new chat, then the chat page; nothing until `ready`. */
  readonly startChat: () => void
  readonly ready: boolean
}

const MODE_CHANGED =
  'Automatos theme SDK: useChatController is called with a conversation, or without one, for the life of a component. Key the conversation screen by conversation.key.'
const NOT_HANDED_OUT =
  'Automatos theme SDK: useChatController opens only a conversation that useChatController() handed out.'
const LOADING: HostConversationListState = Object.freeze({ status: 'loading' })
const NO_LINES: readonly string[] = Object.freeze([])

/** The conversation on screen, as a theme may hold it. */
function useConversationOnScreen(options: HostChatOnScreenOptions | undefined): HostChatOnScreen {
  const freshDrafts = useFreshDraftStore((s) => s.freshDrafts)
  const { view, adopt, draft, takeDraft } = useChatConversation({ freshDrafts, draftLink: options?.draftLink })
  const conversation = useMemo(() => (view ? hostConversation(view, adopt) : null), [view, adopt])
  return useMemo(() => Object.freeze({ conversation, draft, takeDraft }), [conversation, draft, takeDraft])
}

/** The stream's messages and approval cards copied for a theme, each copy remade only when its source changes. */
function useHostTurn(core: ChatControllerView, view: ChatConversationView) {
  const { messages, streaming, isStopped, deliverablesOf, approvals } = core
  const shown = useMemo(
    () => hostMessages({ messages, streaming, isStopped, deliverablesOf, approvals }, view),
    [messages, streaming, isStopped, deliverablesOf, approvals, view],
  )
  const chatId = view.chatId ?? ''
  const { loose } = approvals
  const looseApprovals = useMemo(
    () => Object.freeze(loose.map((card) => hostApproval(card, chatId, approvals))),
    [loose, chatId, approvals],
  )
  return { shown, looseApprovals }
}

/** One conversation's stream and turn, as a theme's view model. */
function useOpenConversation(
  conversation: HostChatConversation,
  opened: OpenedConversation,
  options: HostChatControllerOptions | undefined,
): HostChatController {
  const core = useCoreChatController({
    view: opened.view,
    onChatIdAssigned: opened.adopt,
    onData: options?.onData,
    onEvent: options?.onEvent,
  })
  const planningMissionId = useMissionStore((s) => s.activePlanningMissionId)
  const { shown, looseApprovals } = useHostTurn(core, opened.view)
  const { status, streaming, errorCode, suggestion, stop } = core
  const send = useMemo(() => gatedByAiLabel(core.send), [core.send])
  const retry = useMemo(() => gatedByAiLabel(core.retry), [core.retry])
  const approvalWaiting = core.approvals.waiting
  return useMemo(
    () =>
      Object.freeze({
        conversation,
        title: shown.title,
        messages: shown.list,
        status,
        streaming,
        errorCode,
        trialExhausted: errorCode === CREDITS_EXHAUSTED_CODE,
        suggestion,
        planningMissionId,
        looseApprovals,
        approvalWaiting,
        send,
        retry,
        stop,
      }),
    [conversation, shown, status, streaming, errorCode, suggestion, planningMissionId, looseApprovals, approvalWaiting, send, retry, stop],
  )
}

/** The conversation on screen (no argument), or one conversation's chat (the conversation it handed out). */
export function useChatController(conversation: HostChatConversation, options?: HostChatControllerOptions): HostChatController
export function useChatController(options?: HostChatOnScreenOptions): HostChatOnScreen
export function useChatController(
  first?: HostChatConversation | HostChatOnScreenOptions,
  options?: HostChatControllerOptions,
): HostChatController | HostChatOnScreen {
  const opened = openedConversationOf(first)
  const withConversation = useRef(opened !== null)
  if (isForgedConversation(first)) throw new Error(NOT_HANDED_OUT)
  if (withConversation.current !== (opened !== null)) throw new Error(MODE_CHANGED)
  // One way for the component's life (checked above), so the same hooks run on every render.
  if (opened !== null) {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    return useOpenConversation(first as HostChatConversation, opened, options)
  }
  // eslint-disable-next-line react-hooks/rules-of-hooks
  return useConversationOnScreen(first as HostChatOnScreenOptions | undefined)
}

/** The caller's conversations, newest first (at most `limit`, the host's default when omitted). */
export function useConversationList(limit?: number): HostConversationListState {
  const state = useCoreConversationList(limit)
  const chats = state.status === 'loaded' ? state.chats : null
  const retry = state.status === 'failed' ? state.retry : null
  return useMemo(() => {
    if (chats) return Object.freeze({ status: 'loaded' as const, chats: hostSummaries(chats) })
    if (retry) return Object.freeze({ status: 'failed' as const, retry })
    return LOADING
  }, [chats, retry])
}

/** The latest conversations for a sidebar, and the one on screen. */
export function useRecentChats(limit?: number): HostRecentChats {
  const recent = useCoreRecentChats(limit)
  const activeChatId = useChatSessionStore((s) => s.session.activeChatId)
  return useMemo(() => Object.freeze({ chats: hostSummaries(recent), activeChatId }), [recent, activeChatId])
}

/** New chat: a fresh draft in the host's chat session, then the chat page. */
export function useStartChat(): HostStartChat {
  const { startChat, ready } = useCoreStartChat()
  return useMemo(() => Object.freeze({ startChat, ready }), [startChat, ready])
}

/** One receipt in the host's plain words ("#0422: moved to Done"), the same in every theme. */
export function receiptLine(receipt: HostReceipt): string {
  return coreReceiptLine(receipt)
}

/** A done receipt's page, only when it is a path in this app that HostLink takes; else null. */
export function receiptLink(receipt: HostReceipt): string | null {
  const link = receipt.status === 'done' ? safeLink(receipt.link) : null
  return link !== null && isInternalHref(link) ? link : null
}

/** Whether a done write touched a card on the board (named by its number). */
export function movedSomethingOnTheBoard(receipts: readonly HostReceipt[]): boolean {
  return coreMovedSomethingOnTheBoard([...receipts])
}

/** The model a receipts frame names (a string), else null. */
function frameModel(part: unknown): string | null {
  const frame = (part as { data?: Record<string, unknown> } | null)?.data?.[LIVE_KEY]
  const model = (frame as { model?: unknown } | null | undefined)?.model
  return typeof model === 'string' ? model : null
}

/**
 * The turn's receipts on one of onData's parts: the receipts frame on a `tool-data` part, read as the
 * host reads it for the reply (only real receipts; the lines above the reply; the model it names).
 * null for any other part, and for a frame without a list.
 */
export function liveReceiptsOf(part: unknown): HostLiveReceipts | null {
  const live = liveReceiptsFromToolData(part)
  const receipts = live?.receipts
  if (live === undefined || receipts === undefined) return null
  return Object.freeze({
    receipts: Object.freeze([...receipts]),
    above: live.receiptsAbove ? Object.freeze([...live.receiptsAbove]) : NO_LINES,
    model: frameModel(part),
  })
}
