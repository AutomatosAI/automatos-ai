'use client'

/**
 * useCanvasSession — the Code Canvas session controller (PRD-170 S3/S4)
 *
 * Wires the tested pure modules to the live surface:
 *   - starts / stops the workspace's headless SDK session (S1 proxy);
 *   - opens the SSE event stream (S3) and folds each validated canvas event
 *     through `reduceCanvasEvent` for the streamed-turns panel;
 *   - turns `permission.request` events into approval cards (`diffApproval`
 *     model) and POSTs the human decision to `/canvas/decision` (S4);
 *   - live-refreshes the file tree when the session writes a file (S3 AC).
 *
 * Reuses the W10 board-SSE transport (`fetch` + `parseSSEFrames` — native
 * EventSource can't carry the hybrid-auth Authorization header) and the shared
 * `apiClient`. All event-shape and decision LOGIC lives in the unit-tested pure
 * modules; this hook is the thin React/effect shell around them.
 */

import { useCallback, useEffect, useReducer, useState } from 'react'
import { apiClient } from '@/lib/api-client'
import { parseSSEFrames } from '@/hooks/use-board-event-stream'
import {
  parseCanvasEvent,
  CanvasEventType,
  type CanvasEventEnvelope,
  acceptsCanvasEvent,
} from './canvasEvents'
import {
  reduceCanvasEvent,
  resolvePermission,
  appendUserPrompt,
  initialCanvasSessionState,
  type CanvasSessionUiState,
} from './canvasSessionState'
import {
  addEditCard,
  addPermissionCard,
  decide as decideCard,
  setAutoAccept as setAutoAcceptState,
  initialDiffApprovalState,
  type DiffApprovalState,
  type ApprovalDecision,
} from './diffApproval'

const RECONNECT_MIN_MS = 1000
const RECONNECT_MAX_MS = 30000

type SessionAction =
  | { kind: 'event'; ev: CanvasEventEnvelope }
  | { kind: 'resolve'; requestId: string; decision: ApprovalDecision }
  | { kind: 'autoAccept'; enabled: boolean }
  | { kind: 'sent'; prompt: string }

interface CombinedState {
  ui: CanvasSessionUiState
  approvals: DiffApprovalState
}

const initialCombined: CombinedState = {
  ui: initialCanvasSessionState,
  approvals: initialDiffApprovalState,
}

/**
 * Fold a canvas event into BOTH the turn state and the approval-card queue. A
 * permission.request becomes an approval card (a diff card when the event
 * carries old/new content, else a bare permission card). Kept as a reducer so
 * every state change is a single, immutable transition.
 */
function sessionReducer(state: CombinedState, action: SessionAction): CombinedState {
  if (action.kind === 'resolve') {
    // Drop the pending request from the turn panel AND flip the card's status
    // (approved/denied) so its badge and buttons reflect the decision.
    return {
      ui: resolvePermission(state.ui, action.requestId),
      approvals: decideCard(state.approvals, action.requestId, action.decision),
    }
  }

  if (action.kind === 'autoAccept') {
    return {
      ui: state.ui,
      approvals: setAutoAcceptState(state.approvals, action.enabled),
    }
  }

  if (action.kind === 'sent') {
    // Optimistically echo the user's prompt as a turn; the agent's streamed
    // reply follows via the SSE 'event' fold.
    return { ui: appendUserPrompt(state.ui, action.prompt), approvals: state.approvals }
  }

  const ev = action.ev
  const ui = reduceCanvasEvent(state.ui, ev)

  if (ev.event_type === CanvasEventType.PermissionRequest) {
    const requestId = typeof ev.data.request_id === 'string' ? ev.data.request_id : ''
    const toolName = typeof ev.data.tool_name === 'string' ? ev.data.tool_name : ''
    if (!requestId) return { ui, approvals: state.approvals }
    const hasDiff =
      typeof ev.data.old_content === 'string' || typeof ev.data.new_content === 'string'
    const approvals = hasDiff
      ? addEditCard(state.approvals, {
          requestId,
          toolName,
          path: typeof ev.data.path === 'string' ? ev.data.path : '',
          oldContent: typeof ev.data.old_content === 'string' ? ev.data.old_content : '',
          newContent: typeof ev.data.new_content === 'string' ? ev.data.new_content : '',
        })
      : addPermissionCard(state.approvals, requestId, toolName)
    return { ui, approvals }
  }

  return { ui, approvals: state.approvals }
}

export interface CanvasSessionController {
  /** PRD-235 W2 S3: true when this Canvas follows a ticket's Claude Code session (no SDK start/stop). */
  external: boolean
  taskId: string | number | null
  ui: CanvasSessionUiState
  approvals: DiffApprovalState
  starting: boolean
  startError: string | null
  /** Monotonic tick the panel watches to re-fetch the file tree on agent edits. */
  treeRefreshTick: number
  start: () => Promise<void>
  stop: () => Promise<void>
  send: (prompt: string) => Promise<void>
  decide: (requestId: string, decision: ApprovalDecision) => Promise<void>
  setAutoAccept: (enabled: boolean) => Promise<void>
}

export function useCanvasSession(
  workspaceId: string | undefined,
  options: { taskId?: string | number | null } = {},
): CanvasSessionController {
  const [state, dispatch] = useReducer(sessionReducer, initialCombined)
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)

  const start = useCallback(async () => {
    if (!workspaceId) return
    setStarting(true)
    setStartError(null)
    try {
      await apiClient.request(`/api/workspaces/${workspaceId}/canvas/sessions`, {
        method: 'POST',
      })
    } catch (err: unknown) {
      setStartError(err instanceof Error ? err.message : 'Failed to start canvas session')
    } finally {
      setStarting(false)
    }
  }, [workspaceId, options.taskId])

  const stop = useCallback(async () => {
    if (!workspaceId) return
    try {
      await apiClient.request(`/api/workspaces/${workspaceId}/canvas/sessions`, {
        method: 'DELETE',
      })
    } catch {
      // Best-effort — the stream/heartbeat will reflect the terminal state.
    }
  }, [workspaceId])

  const send = useCallback(
    async (prompt: string) => {
      if (!workspaceId) return
      const trimmed = prompt.trim()
      if (!trimmed) return
      // Optimistically echo the prompt; the streamed reply arrives over SSE.
      dispatch({ kind: 'sent', prompt: trimmed })
      try {
        await apiClient.post(`/api/workspaces/${workspaceId}/canvas/message`, {
          prompt: trimmed,
        })
      } catch {
        // Best-effort — the SSE stream / session status reflects any failure;
        // the user can retry from the composer.
      }
    },
    [workspaceId]
  )

  const decide = useCallback(
    async (requestId: string, decision: ApprovalDecision) => {
      if (!workspaceId) return
      // Optimistically resolve the card so the UI reacts immediately; the server
      // decision is what actually applies/reverts in the session.
      dispatch({ kind: 'resolve', requestId, decision })
      try {
        // PRD-235 W2 S3: a ticket's Claude Code session is answered on the ticket, not the worker.
        if (options.taskId != null && options.taskId !== '') {
          await apiClient.post(`/api/v1/tasks/${options.taskId}/session-decision`, { request_id: requestId, approved: decision === 'approve' })
        } else {
          await apiClient.post(`/api/workspaces/${workspaceId}/canvas/decision`, {
            request_id: requestId,
            approved: decision === 'approve',
          })
        }
      } catch {
        // A failed decision leaves the session paused; the pending card returning
        // (on reconnect the server re-emits) lets the user retry.
      }
    },
    [workspaceId]
  )

  const setAutoAccept = useCallback(
    async (enabled: boolean) => {
      if (!workspaceId) return
      // Mirror the toggle into the client approvals state so subsequent edit
      // cards land pre-approved with the "auto-accepted" badge, matching the
      // server-side gate.
      dispatch({ kind: 'autoAccept', enabled })
      try {
        await apiClient.post(`/api/workspaces/${workspaceId}/canvas/auto-accept`, { enabled })
      } catch {
        // Revert the optimistic toggle if the server rejected it.
        dispatch({ kind: 'autoAccept', enabled: !enabled })
      }
    },
    [workspaceId]
  )

  // ── SSE subscription (reuses the W10 board-stream transport) ──────────────
  useEffect(() => {
    if (!workspaceId || typeof window === 'undefined') return

    let cancelled = false
    let controller: AbortController | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let attempt = 0

    const connect = async () => {
      if (cancelled) return
      controller = new AbortController()
      try {
        const headers = await apiClient.getAuthHeaders()
        const url = `${apiClient.getBaseUrl()}/api/workspaces/${workspaceId}/canvas/events`
        const response = await fetch(url, {
          method: 'GET',
          headers: { ...headers, Accept: 'text/event-stream' },
          signal: controller.signal,
        })
        if (!response.ok || !response.body) {
          throw new Error(`canvas stream failed: ${response.status}`)
        }
        attempt = 0
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        while (!cancelled) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const { events, rest } = parseSSEFrames(buffer)
          buffer = rest
          for (const frame of events) {
            const parsed = parseCanvasEvent(frame.data)
            if (parsed && acceptsCanvasEvent(parsed, options.taskId)) dispatch({ kind: 'event', ev: parsed })
          }
        }
      } catch (err) {
        if (cancelled || (err instanceof DOMException && err.name === 'AbortError')) {
          return
        }
        // Real drop — fall through to backoff reconnect.
      } finally {
        controller = null
      }
      if (!cancelled) {
        const delay = Math.min(RECONNECT_MIN_MS * 2 ** attempt, RECONNECT_MAX_MS)
        attempt += 1
        reconnectTimer = setTimeout(connect, delay)
      }
    }

    void connect()
    return () => {
      cancelled = true
      if (controller) controller.abort()
      if (reconnectTimer) clearTimeout(reconnectTimer)
    }
  }, [workspaceId])

  return {
    // PRD-235 W2 S3: a ticket-rooted Canvas follows an external (Claude Code) session.
    external: options.taskId != null && options.taskId !== '',
    taskId: options.taskId ?? null,
    ui: state.ui,
    approvals: state.approvals,
    starting,
    startError,
    treeRefreshTick: state.ui.treeRefreshTick,
    start,
    stop,
    send,
    decide,
    setAutoAccept,
  }
}
