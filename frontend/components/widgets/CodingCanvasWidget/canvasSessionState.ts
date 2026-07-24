/**
 * Canvas session UI state reducer
 * PRD-170 S3
 *
 * Pure fold of the validated canvas event stream (see `canvasEvents.ts`) into
 * the state the session panel renders: ordered turns beside the file tree, a
 * live session status, a monotonically-increasing `treeRefreshTick` the panel
 * watches to re-fetch the tree on agent file edits, and the pending permission
 * requests S4 turns into approval cards.
 *
 * Kept pure (no React, no I/O) so it is exhaustively unit-testable; the panel
 * feeds it events from the SSE stream and renders the result.
 */

import {
  CanvasEventType,
  assistantText,
  toolCallData,
  isTreeMutatingEvent,
  type CanvasEventEnvelope,
} from './canvasEvents'

export type CanvasSessionStatus =
  | 'idle'
  | 'starting'
  | 'running'
  | 'stopped'
  | 'failed'

export interface CanvasTurnItem {
  kind: 'text' | 'tool_call' | 'file_edit' | 'user'
  text?: string
  toolName?: string
  path?: string | null
}

export interface CanvasPermissionRequest {
  requestId: string | null
  toolName: string
  path: string | null
}

export interface CanvasSessionUiState {
  status: CanvasSessionStatus
  error: string | null
  turns: CanvasTurnItem[]
  pendingPermissions: CanvasPermissionRequest[]
  /** Bumped on every file-edit event; the panel re-fetches the tree when it changes. */
  treeRefreshTick: number
  /** Count of completed turns (result markers) — for "N turns" UI. */
  completedTurns: number
}

export const initialCanvasSessionState: CanvasSessionUiState = {
  status: 'idle',
  error: null,
  turns: [],
  pendingPermissions: [],
  treeRefreshTick: 0,
  completedTurns: 0,
}

function statusFromEvent(raw: unknown): CanvasSessionStatus | null {
  switch (raw) {
    case 'starting':
      return 'starting'
    case 'running':
      return 'running'
    case 'stopped':
      return 'stopped'
    case 'failed':
      return 'failed'
    default:
      return null
  }
}

/**
 * Fold one validated canvas event into the session UI state. Always returns a
 * NEW state object (immutable) so React re-renders cleanly.
 */
export function reduceCanvasEvent(
  state: CanvasSessionUiState,
  ev: CanvasEventEnvelope
): CanvasSessionUiState {
  switch (ev.event_type) {
    case CanvasEventType.SessionStatus: {
      const next = statusFromEvent(ev.data.status)
      if (next === null) return state
      const error =
        next === 'failed' && typeof ev.data.error === 'string'
          ? ev.data.error
          : next === 'failed'
            ? state.error
            : null
      return { ...state, status: next, error }
    }

    case CanvasEventType.AssistantText: {
      const text = assistantText(ev)
      if (!text) return state
      return { ...state, turns: [...state.turns, { kind: 'text', text }] }
    }

    case CanvasEventType.ToolCall: {
      const { toolName, path } = toolCallData(ev)
      return {
        ...state,
        turns: [...state.turns, { kind: 'tool_call', toolName, path }],
      }
    }

    case CanvasEventType.FileEdit: {
      const { toolName, path } = toolCallData(ev)
      const bumped = isTreeMutatingEvent(ev)
        ? state.treeRefreshTick + 1
        : state.treeRefreshTick
      return {
        ...state,
        turns: [...state.turns, { kind: 'file_edit', toolName, path }],
        treeRefreshTick: bumped,
      }
    }

    case CanvasEventType.PermissionRequest: {
      const req: CanvasPermissionRequest = {
        requestId:
          typeof ev.data.request_id === 'string' ? ev.data.request_id : null,
        toolName:
          typeof ev.data.tool_name === 'string' ? ev.data.tool_name : '',
        path: typeof ev.data.path === 'string' ? ev.data.path : null,
      }
      return {
        ...state,
        pendingPermissions: [...state.pendingPermissions, req],
      }
    }

    case CanvasEventType.TurnComplete: {
      return { ...state, completedTurns: state.completedTurns + 1 }
    }

    case CanvasEventType.ToolResult:
    default:
      return state
  }
}

/**
 * Optimistically append the user's prompt as a turn (PRD-203 C·S7) so the
 * composer echoes it immediately; the agent's streamed reply follows via the
 * SSE event fold. Pure + immutable — empty/whitespace prompts are a no-op.
 */
export function appendUserPrompt(
  state: CanvasSessionUiState,
  text: string
): CanvasSessionUiState {
  const trimmed = text.trim()
  if (!trimmed) return state
  return { ...state, turns: [...state.turns, { kind: 'user', text: trimmed }] }
}

/** Resolve (drop) a pending permission request once approved/denied (S4). */
export function resolvePermission(
  state: CanvasSessionUiState,
  requestId: string | null
): CanvasSessionUiState {
  return {
    ...state,
    pendingPermissions: state.pendingPermissions.filter(
      (p) => p.requestId !== requestId
    ),
  }
}
