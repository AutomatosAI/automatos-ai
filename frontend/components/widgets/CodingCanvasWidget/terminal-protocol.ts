/**
 * PRD-239 S7 — the wire between the Canvas terminal and the CLI host.
 *
 * Binary frames carry bytes both ways (keystrokes in, PTY output out); text
 * frames carry JSON control messages. The grant reaches the host on its next
 * heartbeat, so the first connect may be refused for a few seconds — retry
 * inside a budget instead of failing on the first attempt.
 */

export interface ResizeMessage {
  type: 'resize'
  cols: number
  rows: number
}

export function encodeResize(cols: number, rows: number): string {
  return JSON.stringify({ type: 'resize', cols: Math.max(2, Math.floor(cols)), rows: Math.max(2, Math.floor(rows)) } satisfies ResizeMessage)
}

export function encodeInput(data: string): Uint8Array {
  return new TextEncoder().encode(data)
}

/** The waits between connection attempts inside `budgetMs` (pure). */
export function retryDelays(budgetMs: number, stepMs = 1500): number[] {
  const delays: number[] = []
  let spent = 0
  while (spent + stepMs <= budgetMs) {
    delays.push(stepMs)
    spent += stepMs
  }
  return delays
}

export type SocketFactory = (url: string) => WebSocket

function attempt(url: string, factory: SocketFactory): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    let settled = false
    const ws = factory(url)
    ws.binaryType = 'arraybuffer'
    ws.onopen = () => {
      settled = true
      resolve(ws)
    }
    const fail = () => {
      if (settled) return
      settled = true
      reject(new Error('the host refused the connection'))
    }
    ws.onerror = fail
    ws.onclose = fail
  })
}

/** Connect, retrying while the host has not yet learned the grant. */
export async function connectWithRetry(
  url: string,
  budgetMs: number,
  factory: SocketFactory = (u) => new WebSocket(u),
  sleep: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
): Promise<WebSocket> {
  const delays = retryDelays(budgetMs)
  let lastError: Error | null = null
  for (let i = 0; i <= delays.length; i += 1) {
    try {
      return await attempt(url, factory)
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))
      if (i < delays.length) await sleep(delays[i])
    }
  }
  throw lastError ?? new Error('could not connect to the terminal')
}
