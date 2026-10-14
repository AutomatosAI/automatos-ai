import { LIVE_KEY, liveReceiptsFromToolData } from '@/lib/chat/receipts'

import type { JsonObject, MascotEvent } from './types'

/**
 * PRD-260 D5, D6 — the core chat as Auto's events. Pure.
 *
 * `useChat` (lib/chat/hooks.ts, under useChatWithReceipts) hands its data
 * callback `{type, data}` parts: 'tool-start', 'tool-end', 'tool-data' (the
 * turn's receipts ride one under `receipts`), 'memory-injected',
 * 'workflow-update', 'mission-suggestion' and a few that never move Auto. Its
 * status is 'idle' | 'streaming' | 'error' (an awaited detached reply reads as
 * 'streaming'), and its errorCode names a blocked turn ('trial_exhausted').
 * These parts come off the wire, so every field is checked here: a part that
 * isn't what it says maps to nothing. The words themselves never reach onData:
 * the start of the answer is read from the newest reply (mascotEventsFromReply).
 */

export type ChatStatus = 'idle' | 'streaming' | 'error'

/** useChat's own name for a tool-start without one. */
const UNNAMED_TOOL = 'tool'

type Part = { readonly type?: unknown; readonly data?: unknown }

function recordOf(value: unknown): JsonObject | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as JsonObject) : null
}

function textOf(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function numberOf(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function toolCallIdOf(data: JsonObject): string | null {
  const id = textOf(data.toolCallId)
  return id ? id : null
}

function toolStart(data: JsonObject): MascotEvent[] {
  const toolCallId = toolCallIdOf(data)
  if (toolCallId === null) return []
  const toolName = textOf(data.toolName) || UNNAMED_TOOL
  return [{ type: 'tool-start', toolCallId, toolName, input: recordOf(data.input) ?? {} }]
}

/** A skipped call is no failure, and an ask (waiting) is Auto waiting for the owner (lib/chat/tool-calls.ts). */
function toolEnd(data: JsonObject): MascotEvent[] {
  const toolCallId = toolCallIdOf(data)
  if (toolCallId === null) return []
  return [{
    type: 'tool-end',
    toolCallId,
    toolName: textOf(data.toolName) || UNNAMED_TOOL,
    success: data.success === true,
    error: textOf(data.error),
    durationMs: numberOf(data.durationMs),
    summary: textOf(data.summary),
    skipped: data.skipped === true,
    waiting: data.waiting === true,
  }]
}

/** The turn's receipts, read by lib/chat/receipts.ts exactly as the reply reads them. */
function receipts(part: Part, data: JsonObject): MascotEvent[] {
  const live = liveReceiptsFromToolData(part)
  const list = live?.receipts
  if (live === undefined || list === undefined) return []
  const model = textOf(recordOf(data[LIVE_KEY])?.model)
  return [{ type: 'receipts', receipts: list, model, above: live.receiptsAbove ?? [] }]
}

function memoryInjected(data: JsonObject): MascotEvent[] {
  const memories = Array.isArray(data.memories) ? data.memories : []
  return [{ type: 'memory-injected', memories, totalMatched: numberOf(data.totalMatched) ?? memories.length }]
}

function workflowUpdate(data: JsonObject): MascotEvent[] {
  return [{
    type: 'workflow-update',
    workflowId: textOf(data.workflowId) ?? '',
    status: textOf(data.status) ?? '',
    currentStep: textOf(data.currentStep),
    progress: numberOf(data.progress),
  }]
}

/**
 * What one of useChat's onData parts means for Auto. A mission suggestion, the
 * usage, the agent's info, a stored memory, a tool-data part that carries no
 * receipts and any part this version doesn't know map to nothing.
 */
export function mascotEventsFromData(part: unknown): MascotEvent[] {
  const forwarded = recordOf(part) as Part | null
  const data = recordOf(forwarded?.data)
  if (forwarded === null || data === null) return []
  switch (forwarded.type) {
    case 'tool-start':
      return toolStart(data)
    case 'tool-end':
      return toolEnd(data)
    case 'tool-data':
      return receipts(forwarded, data)
    case 'memory-injected':
      return memoryInjected(data)
    case 'workflow-update':
      return workflowUpdate(data)
    default:
      return []
  }
}

/** The message on Auto's error event when the turn was blocked (errorCode names why). */
export const BLOCKED_TURN = 'The turn was blocked'
/** The message on Auto's error event when the chat's status turned to error. */
export const FAILED_TURN = 'The turn failed'

/**
 * What a change of useChat's status means for Auto: into 'streaming' starts the
 * turn (a send; also a detached reply being awaited), into 'error' fails it, and
 * 'streaming' to 'idle' finishes it — or fails it, when an errorCode says the turn
 * was blocked (a failure and the end of the stream can land in one render).
 */
export function mascotEventsFromStatus(previous: ChatStatus, next: ChatStatus, errorCode: string | null): MascotEvent[] {
  if (previous === next) return []
  if (next === 'streaming') return [{ type: 'send' }]
  if (next === 'error') return [{ type: 'error', message: FAILED_TURN, code: errorCode }]
  if (previous !== 'streaming') return []
  if (errorCode !== null) return [{ type: 'error', message: BLOCKED_TURN, code: errorCode }]
  return [{ type: 'finish', finishReason: null }]
}

/**
 * The newest reply, as far as Auto cares: which one it is, whether its words
 * have started, and whether one of its tool calls is still running. useChat
 * writes the words into the reply instead of handing text parts to onData, so
 * the start of the answer is read from the reply itself.
 */
export type ReplyProgress = {
  readonly id: string | null
  readonly hasText: boolean
  readonly toolRunning: boolean
}

export const NO_REPLY: ReplyProgress = { id: null, hasText: false, toolRunning: false }

/** The reducer only reads that a text event came: its words are on the reply. */
const ANSWER_STARTED: MascotEvent = { type: 'text', text: '' }

/**
 * What a change of the newest reply means for Auto (Play's spec: thinking when
 * the first words arrive → idle). While the chat streams, the reply's words
 * starting with no tool running on it is one text event. Its later words, words
 * that start while a tool runs (the tool's state wins, and its end settles
 * Auto), and any status other than streaming map to nothing.
 */
export function mascotEventsFromReply(previous: ReplyProgress, next: ReplyProgress, status: ChatStatus): MascotEvent[] {
  if (status !== 'streaming' || !next.hasText || next.toolRunning) return []
  const alreadyStarted = previous.id === next.id && previous.hasText
  return alreadyStarted ? [] : [ANSWER_STARTED]
}
