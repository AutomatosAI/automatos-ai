/**
 * PRD-239 S7 — chat from the Canvas.
 *
 * A ticket's Claude Code session is a read-only mirror in the Canvas (the host
 * never types into the session). A message typed beside it therefore goes
 * through the chat lane: the chat page selects the ticket's agent and sends the
 * text as an ordinary turn, which files a follow-up ticket that resumes the
 * session. The panel and the chat page talk through one window event so the
 * Canvas widget needs no knowledge of the chat's state.
 */

export const CANVAS_COMPOSE_EVENT = 'automatos:canvas-compose'

export interface CanvasComposeDetail {
  taskId: string | number
  text: string
  /** Set by the listener that took the message (the chat page). */
  handled?: boolean
}

/** Dispatch a message from the Canvas. Returns true when the chat took it. */
export function dispatchCanvasCompose(taskId: string | number, text: string): boolean {
  if (typeof window === 'undefined') return false
  const detail: CanvasComposeDetail = { taskId, text, handled: false }
  window.dispatchEvent(new CustomEvent<CanvasComposeDetail>(CANVAS_COMPOSE_EVENT, { detail }))
  return detail.handled === true
}

/** Listen for Canvas messages; the handler's message is marked handled. Returns the unsubscribe. */
export function onCanvasCompose(handler: (detail: CanvasComposeDetail) => void): () => void {
  if (typeof window === 'undefined') return () => {}
  const listener = (event: Event) => {
    const detail = (event as CustomEvent<CanvasComposeDetail>).detail
    if (!detail || typeof detail.text !== 'string' || !detail.text.trim()) return
    detail.handled = true
    handler(detail)
  }
  window.addEventListener(CANVAS_COMPOSE_EVENT, listener)
  return () => window.removeEventListener(CANVAS_COMPOSE_EVENT, listener)
}
