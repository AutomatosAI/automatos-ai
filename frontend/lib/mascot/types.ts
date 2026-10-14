import type { Receipt } from '@/types'

import type { ToolActivity } from './tools'

/**
 * PRD-260 D6 — Auto's states and the events that move him, ported from
 * Automatos Play's lib/mascot/types.ts. The stream events the reducer reads
 * are re-declared below (Play's TYPES only: its stream parser is not ported,
 * D5); lib/mascot/events.ts makes them from the core chat's onData parts and
 * status. A receipt is the core's own (types/chat.ts).
 */

export const MASCOT_STATES = [
  'hello',
  'idle',
  'listening',
  'thinking',
  'working',
  'searching',
  'reading',
  'celebrate',
  'oops',
  'sleeping',
  'lost',
  'loading',
] as const

export type MascotName = (typeof MASCOT_STATES)[number]

/** Play once, then return to idle (or to what was queued meanwhile). */
export const ONE_SHOTS: ReadonlySet<MascotName> = new Set<MascotName>(['hello', 'celebrate'])

/** Swap at once instead of waiting for the clip boundary. */
export const URGENT: ReadonlySet<MascotName> = new Set<MascotName>(['oops', 'hello'])

/**
 * Priority when several things are true at once (higher wins):
 * oops > celebrate > listening (waiting) > searching = reading > working > thinking
 * > listening (typing) > idle. hello sits with celebrate; sleeping and lost sit
 * below idle, so any activity wakes Auto up.
 */
export const RANK = {
  oops: 8,
  oneShot: 7,
  waiting: 6,
  tool: 5,
  working: 4,
  thinking: 3,
  typing: 2,
  idle: 1,
  rest: 0,
} as const

/** A state on screen, with the priority of the reason it is there. */
export type Pose = { readonly mascot: MascotName; readonly rank: number }

export type OpenTool = { readonly toolCallId: string; readonly activity: ToolActivity }

export type MascotState = {
  /** What Auto shows now (the player swaps to it at the next clip boundary). */
  readonly mascot: MascotName
  readonly rank: number
  /** What plays when the current one-shot ends (null: idle). */
  readonly queued: Pose | null
  /** Tool calls started and not yet ended in this turn. */
  readonly openTools: readonly OpenTool[]
  /** A message was sent and the turn hasn't finished or failed. */
  readonly turnActive: boolean
  /** This turn's receipts include a done receipt: celebrate on finish. */
  readonly celebrateOnFinish: boolean
  /** The backend is unreachable: Auto sleeps until backend-up. */
  readonly backendDown: boolean
  /** An approval card waits for the owner's answer: Auto rests on listening instead of idle. */
  readonly approvalWaiting: boolean
  /**
   * This turn has moved Auto past thinking: its words started or a tool ran. Once
   * no card waits, a turn still running goes back to its tool, or to thinking
   * only while this is false.
   */
  readonly pastThinking: boolean
}

/**
 * Events from the page, not the stream. app-open means the first open in this
 * browser session (the caller checks, lib/mascot/session.ts). clip-ended names
 * the clip that ended, so a stale one can't end a newer one-shot. request-failed
 * is a failed call. wake is any input after the idle timeout (it ends an idle
 * nap, not the backend's). approval-waiting: an approval card in the chat waits
 * for the owner's answer; approval-settled: every card is answered or closed.
 */
export type ClientEvent =
  | { readonly type: 'app-open' }
  | { readonly type: 'input-focus' }
  | { readonly type: 'input-blur' }
  | { readonly type: 'send' }
  | { readonly type: 'idle-timeout' }
  | { readonly type: 'wake' }
  | { readonly type: 'backend-down' }
  | { readonly type: 'backend-up' }
  | { readonly type: 'not-found' }
  | { readonly type: 'loading-start' }
  | { readonly type: 'loading-end' }
  | { readonly type: 'deliverable-added' }
  | { readonly type: 'request-failed' }
  | { readonly type: 'clip-ended'; readonly state: MascotName }
  | ApprovalEvent

/** The approval cards in Play's chat (T-005c): Auto listens while one waits for the owner. */
export type ApprovalEvent = { readonly type: 'approval-waiting' } | { readonly type: 'approval-settled' }

// --- The stream events (re-declared from Play's lib/stream/types.ts) ----------------------

export type JsonObject = Readonly<Record<string, unknown>>

export type TextEvent = { readonly type: 'text'; readonly text: string }
export type ChatIdEvent = { readonly type: 'chat-id'; readonly chatId: string }
export type AgentAssignedEvent = {
  readonly type: 'agent-assigned'
  readonly agentId: number | null
  readonly agentName: string
  readonly routeMethod: string | null
}
export type ReasoningEvent = { readonly type: 'reasoning'; readonly delta: string }
export type ProgressEvent = { readonly type: 'progress'; readonly text: string }
export type NarrationEvent = { readonly type: 'narration'; readonly text: string; readonly retracted: boolean }
export type ToolStartEvent = {
  readonly type: 'tool-start'
  readonly toolCallId: string
  readonly toolName: string
  readonly input: JsonObject
}
export type ToolEndEvent = {
  readonly type: 'tool-end'
  readonly toolCallId: string
  readonly toolName: string
  readonly success: boolean
  readonly error: string | null
  readonly durationMs: number | null
  readonly summary: string | null
  readonly skipped: boolean
  /** An ask: Auto waits for the owner. success is false, but it is not a failure. */
  readonly waiting: boolean
}
export type ReceiptsEvent = {
  readonly type: 'receipts'
  readonly receipts: readonly Receipt[]
  readonly model: string | null
  /** Lines the reply shows above its text. */
  readonly above: readonly string[]
}
export type WorkflowUpdateEvent = {
  readonly type: 'workflow-update'
  readonly workflowId: string
  readonly status: string
  readonly currentStep: string | null
  readonly progress: number | null
}
export type MemoryInjectedEvent = {
  readonly type: 'memory-injected'
  readonly memories: readonly unknown[]
  readonly totalMatched: number
}
export type LimitReachedEvent = {
  readonly type: 'limit_reached'
  readonly limit: string
  readonly value: number
  readonly message: string
}
export type ToolPermissionDeniedEvent = {
  readonly type: 'tool-permission-denied'
  readonly toolName: string
  readonly agentId: number | null
  readonly reason: string | null
}
export type AttachmentUnavailableEvent = {
  readonly type: 'attachment_unavailable'
  readonly attachments: readonly unknown[]
}
export type UsageEvent = {
  readonly type: 'usage'
  readonly promptTokens: number
  readonly completionTokens: number
  readonly totalTokens: number
}
export type FinishEvent = { readonly type: 'finish'; readonly finishReason: string | null }
/** The turn failed: an e: line, a failed request, or a blocked turn (errorCode). */
export type ErrorEvent = { readonly type: 'error'; readonly message: string; readonly code: string | null }
/** A frame type this version doesn't know. Normal: the backend adds new ones. */
export type UnknownEvent = { readonly type: 'unknown'; readonly eventType: string }

export type StreamEvent =
  | TextEvent
  | ChatIdEvent
  | AgentAssignedEvent
  | ReasoningEvent
  | ProgressEvent
  | NarrationEvent
  | ToolStartEvent
  | ToolEndEvent
  | ReceiptsEvent
  | WorkflowUpdateEvent
  | MemoryInjectedEvent
  | LimitReachedEvent
  | ToolPermissionDeniedEvent
  | AttachmentUnavailableEvent
  | UsageEvent
  | FinishEvent
  | ErrorEvent
  | UnknownEvent

export type MascotEvent = StreamEvent | ClientEvent
