import { describe, expect, it } from 'vitest'
import { connectWithRetry, describeLaunch, encodeInput, encodeResize, retryDelays } from './terminal-protocol'

describe('terminal protocol (PRD-239 S7)', () => {
  it('encodes a resize as a text control frame with sane bounds', () => {
    expect(JSON.parse(encodeResize(120.7, 40))).toEqual({ type: 'resize', cols: 120, rows: 40 })
    expect(JSON.parse(encodeResize(0, 1))).toEqual({ type: 'resize', cols: 2, rows: 2 })
  })

  it('encodes keystrokes as UTF-8 bytes', () => {
    expect(Array.from(encodeInput('ls\r'))).toEqual([108, 115, 13])
    expect(Array.from(encodeInput('é'))).toEqual([0xc3, 0xa9])
  })

  it('spreads the retry budget in fixed steps', () => {
    expect(retryDelays(20_000)).toEqual(Array(13).fill(1500))
    expect(retryDelays(1000)).toEqual([])
  })

  it('retries until the host has learned the grant, then gives up honestly', async () => {
    let attempts = 0
    const factory = (() => {
      attempts += 1
      const ws = { binaryType: '' } as unknown as WebSocket & { onopen?: () => void; onclose?: () => void }
      queueMicrotask(() => (attempts < 3 ? ws.onclose?.() : ws.onopen?.()))
      return ws
    }) as unknown as (url: string) => WebSocket
    const ws = await connectWithRetry('ws://127.0.0.1:1/terminal?token=t', 6000, factory, async () => {})
    expect(ws).toBeTruthy()
    expect(attempts).toBe(3)

    const refusing = (() => {
      const ws = { binaryType: '' } as unknown as WebSocket & { onclose?: () => void }
      queueMicrotask(() => ws.onclose?.())
      return ws
    }) as unknown as (url: string) => WebSocket
    await expect(connectWithRetry('ws://127.0.0.1:1/terminal?token=t', 3000, refusing, async () => {})).rejects.toThrow('refused')
  })
})

describe('describeLaunch (PRD-239 S7 v2)', () => {
  it('names the agent and the folder for a launched session, else just the folder', () => {
    expect(describeLaunch({ kind: 'claude', session_id: 's', agent_name: 'Bob' }, '/repo', 'x')).toBe('Bob · Claude Code · /repo')
    expect(describeLaunch({ kind: 'claude', session_id: 's' }, null, 'x')).toBe('Claude Code')
    expect(describeLaunch(null, '/repo', 'x')).toBe('/repo')
    expect(describeLaunch(undefined, null, 'the host’s folder')).toBe('the host’s folder')
  })
})
