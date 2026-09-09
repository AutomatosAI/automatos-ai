/**
 * PRD-239 S7 — the terminal wire: control messages, and a connect that keeps
 * retrying while the host has not yet learned the grant.
 */
import { describe, it, expect, vi } from 'vitest'
import { connectWithRetry, encodeInput, encodeResize, retryDelays } from '../terminal-protocol'

describe('terminal protocol', () => {
  it('encodes resize as JSON text and input as bytes', () => {
    expect(JSON.parse(encodeResize(100.7, 30))).toEqual({ type: 'resize', cols: 100, rows: 30 })
    expect(JSON.parse(encodeResize(0, 0))).toEqual({ type: 'resize', cols: 2, rows: 2 })
    expect(Array.from(encodeInput('ls\r'))).toEqual([108, 115, 13])
  })

  it('spreads retries evenly inside the budget', () => {
    expect(retryDelays(20_000)).toHaveLength(13)
    expect(retryDelays(1_000)).toEqual([])
  })

  it('retries until the host accepts the grant, then hands back the open socket', async () => {
    let attempts = 0
    const factory = vi.fn((url: string) => {
      attempts += 1
      const ws: any = { url, binaryType: 'blob', onopen: null, onerror: null, onclose: null }
      queueMicrotask(() => (attempts < 3 ? ws.onclose?.() : ws.onopen?.()))
      return ws as WebSocket
    })
    const sleeps: number[] = []
    const ws = await connectWithRetry('ws://127.0.0.1:1/terminal?token=t', 20_000, factory, async (ms) => {
      sleeps.push(ms)
    })
    expect(attempts).toBe(3)
    expect(sleeps).toEqual([1500, 1500])
    expect(ws.binaryType).toBe('arraybuffer')
  })

  it('gives up after the budget with the last error', async () => {
    const factory = vi.fn((url: string) => {
      const ws: any = { url, onopen: null, onerror: null, onclose: null }
      queueMicrotask(() => ws.onerror?.())
      return ws as WebSocket
    })
    await expect(connectWithRetry('ws://x', 3_000, factory, async () => {})).rejects.toThrow('refused')
    expect(factory).toHaveBeenCalledTimes(3)
  })
})
