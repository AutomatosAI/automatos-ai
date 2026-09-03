/**
 * Canvas session event schema (frontend mirror)
 * PRD-170 S3
 *
 * The single source of truth for the canvas event vocabulary lives on the
 * worker (`services/workspace-worker/canvas_events.py`). This module MIRRORS it
 * for the session panel, and `canvasEvents.test.ts` asserts the two stay in
 * lockstep — a renamed or dropped event type on either side fails the drift
 * gate rather than silently swallowing a turn in the UI.
 *
 * Keep `CANVAS_EVENT_SCHEMA_VERSION` and `CANVAS_EVENT_TYPES` byte-aligned with
 * the Python literals.
 */

// Mirrors canvas_events.CANVAS_EVENT_SCHEMA_VERSION.
export const CANVAS_EVENT_SCHEMA_VERSION = 1

// Mirrors canvas_events.EVENT_* — the CLOSED vocabulary.
export const CanvasEventType = {
  SessionStatus: 'canvas.session.status',
  AssistantText: 'canvas.assistant.text',
  ToolCall: 'canvas.tool.call',
  ToolResult: 'canvas.tool.result',
  FileEdit: 'canvas.file.edit',
  PermissionRequest: 'canvas.permission.request',
  TurnComplete: 'canvas.turn.complete',
} as const

export type CanvasEventTypeValue =
  (typeof CanvasEventType)[keyof typeof CanvasEventType]

// The set the validator accepts. Frozen so tests can compare membership.
export const CANVAS_EVENT_TYPES: ReadonlySet<string> = new Set(
  Object.values(CanvasEventType)
)

export interface CanvasEventEnvelope {
  schema_version: number
  event_type: CanvasEventTypeValue
  workspace_id: string
  data: Record<string, unknown>
  timestamp: string
}

/**
 * Validate + narrow a raw pub/sub payload into a CanvasEventEnvelope.
 *
 * Returns `null` for anything that is not a well-formed canvas event of the
 * CURRENT schema version and a KNOWN event type. A future/legacy schema version
 * or a drifted event name is rejected (not rendered), which is exactly what the
 * drift gate asserts.
 */
export function parseCanvasEvent(raw: unknown): CanvasEventEnvelope | null {
  if (raw === null || typeof raw !== 'object') return null
  const obj = raw as Record<string, unknown>

  if (obj.schema_version !== CANVAS_EVENT_SCHEMA_VERSION) return null

  const eventType = obj.event_type
  if (typeof eventType !== 'string' || !CANVAS_EVENT_TYPES.has(eventType)) {
    return null
  }

  if (typeof obj.workspace_id !== 'string' || obj.workspace_id.length === 0) {
    return null
  }
  if (typeof obj.timestamp !== 'string' || obj.timestamp.length === 0) {
    return null
  }
  if (obj.data === null || typeof obj.data !== 'object') return null

  return {
    schema_version: CANVAS_EVENT_SCHEMA_VERSION,
    event_type: eventType as CanvasEventTypeValue,
    workspace_id: obj.workspace_id,
    data: obj.data as Record<string, unknown>,
    timestamp: obj.timestamp,
  }
}

/** True when a canvas event should trigger a file-tree refresh. */
export function isTreeMutatingEvent(ev: CanvasEventEnvelope): boolean {
  return ev.event_type === CanvasEventType.FileEdit
}

// ---- Narrowed data accessors (defensive; the wire is validated upstream) ----

export function assistantText(ev: CanvasEventEnvelope): string {
  const t = ev.data.text
  return typeof t === 'string' ? t : ''
}

export interface ToolCallData {
  toolName: string
  toolId: string | null
  path: string | null
}

export function toolCallData(ev: CanvasEventEnvelope): ToolCallData {
  return {
    toolName: typeof ev.data.tool_name === 'string' ? ev.data.tool_name : '',
    toolId: typeof ev.data.tool_id === 'string' ? ev.data.tool_id : null,
    path: typeof ev.data.path === 'string' ? ev.data.path : null,
  }
}

/**
 * PRD-235 W2 S3 — which events a Canvas should show. A Canvas rooted at a ticket's
 * session shows only that ticket's Claude Code events (they carry `data.task_id`);
 * a workspace-rooted Canvas shows the SDK session's events, which carry none.
 */
export function acceptsCanvasEvent(ev: CanvasEventEnvelope, taskId?: string | number | null): boolean {
  const evTask = ev.data?.task_id
  if (taskId == null || taskId === '') return evTask == null
  return evTask != null && String(evTask) === String(taskId)
}
