/**
 * PRD-260 D5, D6; PRD-261 A-19a — the core chat's status and its newest reply, as any theme's
 * chat reads them. Pure.
 *
 * `useChat` (lib/chat/hooks.ts, under useChatWithReceipts) has a status of 'idle' | 'streaming' |
 * 'error' (an awaited detached reply reads as 'streaming'). The words, the model's reasoning and a
 * cap that ended the turn never reach its data callback: useChat writes them onto the newest reply,
 * so they are read from it (lib/chat/message-text.ts replyProgress).
 *
 * These lived in builtin Play's mascot library until A-19a moved them here; the core's chat
 * reading and the history refresh import them from here. Play, now an installed package from the
 * theme kit, keeps its own copy in its mascot library and imports nothing from core.
 */

export type ChatStatus = 'idle' | 'streaming' | 'error'

/**
 * The newest reply, as far as a mascot cares: which one it is, whether its words have started,
 * whether one of its tool calls is still running, how much the model has reasoned on it, and
 * whether a cap ended its turn.
 */
export type ReplyProgress = {
  readonly id: string | null
  readonly hasText: boolean
  readonly toolRunning: boolean
  /** The length of the reply's reasoning: its growth is the model thinking. */
  readonly reasoningChars: number
  /** A cap (limit_reached) ended the reply's turn. */
  readonly limitReached: boolean
}

/** No reply yet: the newest message is not one, or a turn still awaits its reply. */
export const NO_REPLY: ReplyProgress = { id: null, hasText: false, toolRunning: false, reasoningChars: 0, limitReached: false }
