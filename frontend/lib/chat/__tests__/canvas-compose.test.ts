/**
 * PRD-239 S7 — a message typed beside a ticket's session reaches the chat page
 * through one window event, and the Canvas learns whether anyone took it.
 */
import { describe, it, expect, vi } from 'vitest'
import { CANVAS_COMPOSE_EVENT, dispatchCanvasCompose, onCanvasCompose } from '../canvas-compose'

describe('canvas compose event', () => {
  it('is handled by a listener and reports it', () => {
    const seen: Array<{ taskId: string | number; text: string }> = []
    const off = onCanvasCompose(({ taskId, text }) => seen.push({ taskId, text }))
    expect(dispatchCanvasCompose(95, 'Hey Bob, what next?')).toBe(true)
    expect(seen).toEqual([{ taskId: 95, text: 'Hey Bob, what next?' }])
    off()
    expect(dispatchCanvasCompose(95, 'anyone?')).toBe(false)
  })

  it('ignores empty text and foreign payloads', () => {
    const handler = vi.fn()
    const off = onCanvasCompose(handler)
    expect(dispatchCanvasCompose(1, '   ')).toBe(false)
    window.dispatchEvent(new CustomEvent(CANVAS_COMPOSE_EVENT, { detail: null }))
    expect(handler).not.toHaveBeenCalled()
    off()
  })
})
