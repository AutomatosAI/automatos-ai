/**
 * PRD-261 A-11b — SDK area `chat` (frontend/lib/theme-sdk/chat.ts): a theme's chat screen on the host's chat
 * controller, the conversation lists, New chat and the receipt lines. Everything returned is a frozen,
 * theme-neutral copy.
 *
 * useChatController is called twice, the way Play's chat is built: without an argument in the chat page (the
 * conversation on screen, or null while it opens), then with that conversation in a child keyed by
 * `conversation.key` (its stream and turn). A component calls it one way for its whole life; only a conversation
 * the first call handed out opens a stream.
 *
 * The approval cards on a message (`approvals`, and `looseApprovals` after the conversation) are drawn by the
 * host: pass one to `cards.ToolApproval` or `cards.MissionApproval` (protected surfaces, A-12c).
 */

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
  /** The workspace's trial is spent: the send was refused (cards.TrialExhaustedBanner). */
  readonly trialExhausted: boolean
  readonly suggestion: HostMissionSuggestion | null
  /** The Mission the chat is planning, once one is (cards.MissionCreatedCard). */
  readonly planningMissionId: string | null
  /** Open approval cards whose reply left the conversation (Try again, a failed request): shown after it. */
  readonly looseApprovals: readonly HostChatApproval[]
  /** A card still waits for the owner. */
  readonly approvalWaiting: boolean
  /** Sends the words; false (nothing sent) while a turn runs. */
  readonly send: (text: string) => boolean
  /** Asks the owner's last message again; false while a turn runs. */
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

/** The conversation on screen, then one conversation's chat (the conversation it handed out, in a keyed child). */
export declare function useChatController(conversation: HostChatConversation, options?: HostChatControllerOptions): HostChatController
export declare function useChatController(options?: HostChatOnScreenOptions): HostChatOnScreen

/** The caller's conversations, newest first (at most `limit`, the host's default when omitted). */
export declare function useConversationList(limit?: number): HostConversationListState

/** The latest conversations for a sidebar, and the one on screen. */
export declare function useRecentChats(limit?: number): HostRecentChats

/** New chat: a fresh draft in the host's chat session, then the chat page. */
export declare function useStartChat(): HostStartChat

/** One receipt in the host's plain words ("#0422: moved to Done"), the same in every theme. */
export declare function receiptLine(receipt: HostReceipt): string

/** A done receipt's page, only when it is a path in this app; else null. */
export declare function receiptLink(receipt: HostReceipt): string | null

/** Whether a done write touched a card on the board (named by its number). */
export declare function movedSomethingOnTheBoard(receipts: readonly HostReceipt[]): boolean
