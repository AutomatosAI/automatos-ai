/**
 * PRD-205 S7 -- client bridge for background->chat delivery ("Auto speaks").
 *
 * Server-side producers (the watcher, scheduled tasks) post assistant
 * messages after the HTTP turn is gone and emit a `chat_changed` NOTIFY on
 * the existing board_events lane. Every frame on that lane arrives under the
 * SSE event name `board_changed` (the stream wraps ALL payloads), so chat
 * payloads are distinguished by `data.event === "chat_changed"` -- the
 * stream hook routes on the PAYLOAD, never on a new SSE event name.
 *
 * The chat UI is useState-based (no react-query, per PRD-205 section 9), so
 * the decoupled bridge is a window CustomEvent: the always-mounted stream
 * hook dispatches `automatos:chat-changed` and interested surfaces (the open
 * chat, history lists) subscribe. Everything here is dependency-free so the
 * routing, filtering, merging, and source-mapping logic is unit-testable.
 */

import { useEffect, useRef } from 'react'

/** Window event fired when a background message lands in one of the viewer's chats. */
export const CHAT_CHANGED_EVENT = 'automatos:chat-changed'

export interface ChatChangedDetail {
  chatId: string
}

/** A `chat_changed` payload as emitted by services/board_events.notify_chat_event. */
export interface ChatChangedPayload {
  chatId: string
  userId: number
}

/**
 * The viewer's integer `users.id`, learned from chat API responses
 * (`/api/chat/history` rows and `GET /api/chat/{id}` both carry `userId`,
 * always the caller's own -- both endpoints are user-scoped). Needed because
 * `chat_changed` payloads address a user by integer id while Clerk only
 * gives the client a subject string. Module-level cache: written on every
 * chat API response, read by the stream hook's per-user filter.
 */
let viewerUserId: number | null = null

export function rememberViewerUserId(candidate: unknown): void {
  if (typeof candidate === 'number' && Number.isInteger(candidate)) {
    viewerUserId = candidate
  }
}

export function getViewerUserId(): number | null {
  return viewerUserId
}

/** Test hygiene only -- the cache is process-global inside vitest. */
export function resetViewerUserIdForTests(): void {
  viewerUserId = null
}

/**
 * Parse one decoded SSE `data` value into a chat-changed payload.
 * Returns null for anything else (board payloads, connect frames, garbage).
 * Pure and total -- no I/O.
 */
export function parseChatChangedPayload(data: unknown): ChatChangedPayload | null {
  if (typeof data !== 'object' || data === null) return null
  const record = data as Record<string, unknown>
  if (record.event !== 'chat_changed') return null
  const chatId = record.chat_id
  const userId = record.user_id
  if (typeof chatId !== 'string' || chatId.length === 0) return null
  if (typeof userId !== 'number' || !Number.isInteger(userId)) return null
  return { chatId, userId }
}

/**
 * Per-user delivery filter: the SSE lane is workspace-filtered server-side;
 * the client drops events addressed to OTHER users in the workspace. When
 * the viewer's id is not yet known (no chat API response seen), deliver --
 * every listener is self-scoped (open-chat id match, user-scoped history
 * refetch), so a false positive only costs one cheap refetch.
 */
export function shouldDeliverChatEvent(
  eventUserId: number,
  viewer: number | null
): boolean {
  return viewer === null || eventUserId === viewer
}

/** Dispatch the window-level chat-changed event (no-op outside the browser). */
export function dispatchChatChanged(chatId: string): void {
  if (typeof window === 'undefined') return
  window.dispatchEvent(
    new CustomEvent<ChatChangedDetail>(CHAT_CHANGED_EVENT, { detail: { chatId } })
  )
}

/**
 * Route one decoded SSE payload from the board stream. Returns true when the
 * payload was a `chat_changed` event (handled here -- dispatched to
 * listeners, or dropped by the per-user filter), false when it belongs to
 * the board (the caller keeps its existing cache invalidation).
 */
export function routeChatChangedPayload(data: unknown): boolean {
  const payload = parseChatChangedPayload(data)
  if (payload === null) return false
  if (shouldDeliverChatEvent(payload.userId, getViewerUserId())) {
    dispatchChatChanged(payload.chatId)
  }
  return true
}

/**
 * Subscribe to chat-changed events for the component's lifetime. The handler
 * lives in a ref so callers may pass a fresh closure every render without
 * re-subscribing.
 */
export function useChatChangedListener(
  onChatChanged: (detail: ChatChangedDetail) => void
): void {
  const handlerRef = useRef(onChatChanged)
  handlerRef.current = onChatChanged

  useEffect(() => {
    if (typeof window === 'undefined') return
    const listener = (event: Event) => {
      const detail = (event as CustomEvent<ChatChangedDetail>).detail
      if (detail && typeof detail.chatId === 'string') handlerRef.current(detail)
    }
    window.addEventListener(CHAT_CHANGED_EVENT, listener)
    return () => window.removeEventListener(CHAT_CHANGED_EVENT, listener)
  }, [])
}

// ---------------------------------------------------------------------------
// Merge + persisted-source mapping (pure)
// ---------------------------------------------------------------------------

interface SourceCarrier {
  metadata?: { source?: unknown } | null
}

/** True when a message carries the persisted background source object. */
export function hasBackgroundSource(message: unknown): boolean {
  const source = (message as SourceCarrier | null | undefined)?.metadata?.source
  return typeof source === 'object' && source !== null
}

function createdAtOf(message: unknown): string {
  const value = (message as { createdAt?: unknown } | null | undefined)?.createdAt
  return typeof value === 'string' ? value : ''
}

/**
 * Merge freshly fetched messages into the live (useState) list after a
 * chat-changed event. Contract:
 *  - Append-only: existing entries -- including any in-flight assistant
 *    placeholder -- are never removed, altered, or reordered.
 *  - Only fetched messages carrying a persisted background `source` are
 *    eligible: in-turn messages keep client-generated ids locally, so
 *    merging them by id would duplicate the whole conversation, and the
 *    chat-changed event only ever signals a background post.
 *  - De-duplicated by id against the current list (and within the fetch).
 *  - Additions are appended in `createdAt` order.
 *  - Returns the SAME array reference when there is nothing to add, so a
 *    `setState(prev => merge(prev, fetched))` skips the re-render.
 */
export function mergeBackgroundMessages<T extends { id: string }>(
  current: T[],
  fetched: T[]
): T[] {
  const knownIds = new Set(current.map((message) => message.id))
  const additions: T[] = []
  for (const message of fetched) {
    if (!hasBackgroundSource(message)) continue
    if (knownIds.has(message.id)) continue
    knownIds.add(message.id)
    additions.push(message)
  }
  if (additions.length === 0) return current
  additions.sort((a, b) => createdAtOf(a).localeCompare(createdAtOf(b)))
  return [...current, ...additions]
}

/**
 * PRD-205 S3->S7: lift a message row's persisted `source` column (the
 * background-author provenance from GET /api/chat/{id}/messages) into
 * `metadata.source`, where the existing badge slot in message.tsx renders
 * it. Rows without a source pass through untouched; never mutates input.
 */
export function attachPersistedSource<T extends object>(row: T): T {
  const source = (row as { source?: unknown }).source
  if (typeof source !== 'object' || source === null) return row
  const metadata = (row as { metadata?: object | null }).metadata
  return { ...row, metadata: { ...(metadata ?? {}), source } } as T
}
