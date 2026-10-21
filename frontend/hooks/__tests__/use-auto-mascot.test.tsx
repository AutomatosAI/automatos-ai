/**
 * PRD-260 D5, D6 — useAutoMascot: Auto driven by the core chat's onData parts,
 * status and newest reply (through lib/mascot/events.ts: its words, its
 * reasoning and a cap on its turn), the page's events, hello once per session,
 * the backend's state and the idle nap.
 */
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type AutoMascotOptions, useAutoMascot } from '@/hooks/use-auto-mascot'
import { LIVE_KEY } from '@/lib/chat/receipts'
import { NO_REPLY, type ReplyProgress } from '@/lib/mascot/events'
import { memoryStorage } from '@/lib/looks/__tests__/helpers'
import { receipt } from '@/lib/mascot/__tests__/helpers'
import { CREDITS_EXHAUSTED_CODE } from '@/lib/credits'

/** A short nap for the tests. */
const NAP_MS = 1000

const toolStart = (toolCallId: string, toolName: string) => ({ type: 'tool-start', data: { toolCallId, toolName, input: {} } })
const toolEnd = (toolCallId: string) => ({ type: 'tool-end', data: { toolCallId, toolName: 'x', success: true } })
const doneReceipts = { type: 'tool-data', data: { [LIVE_KEY]: { receipts: [receipt('done')] } } }

function mount(initialProps: AutoMascotOptions = {}) {
  return renderHook((props: AutoMascotOptions) => useAutoMascot(props), { initialProps })
}

afterEach(() => {
  vi.useRealTimers()
})

describe('useAutoMascot: the chat', () => {
  it('thinks when the chat streams, follows the tools, and celebrates a turn that did something', () => {
    const { result, rerender } = mount({ status: 'idle' })
    expect(result.current.mascot).toBe('idle')
    rerender({ status: 'streaming' })
    expect(result.current.mascot).toBe('thinking')
    act(() => result.current.onData(toolStart('c1', 'platform_web_search')))
    expect(result.current.mascot).toBe('searching')
    act(() => result.current.onData(toolEnd('c1')))
    act(() => result.current.onData(doneReceipts))
    rerender({ status: 'idle' })
    expect(result.current.mascot).toBe('celebrate')
    act(() => result.current.onClipEnded('celebrate'))
    expect(result.current.mascot).toBe('idle')
  })

  it('a Deliverable made mid-turn and its done receipt: one celebration, then idle, and the next send thinks (P260-FX-04)', () => {
    const { result, rerender } = mount({ status: 'idle' })
    rerender({ status: 'streaming' })
    act(() => result.current.dispatch({ type: 'deliverable-added' }))
    expect(result.current.mascot).toBe('celebrate')
    act(() => result.current.onData(doneReceipts))
    rerender({ status: 'idle' })
    act(() => result.current.onClipEnded('celebrate'))
    expect(result.current.mascot).toBe('idle')
    rerender({ status: 'streaming' })
    expect(result.current.mascot).toBe('thinking')
  })

  it('thinks at once when it mounts on an awaited reply', () => {
    expect(mount({ status: 'streaming' }).result.current.mascot).toBe('thinking')
  })

  it("hello on the first open of an awaited reply, then back to the reply's thinking (P260-FX-04)", () => {
    const session = memoryStorage()
    const { result } = mount({ status: 'streaming', session: () => session })
    act(() => {
      result.current.greet()
    })
    expect(result.current.mascot).toBe('hello')
    act(() => result.current.onClipEnded('hello'))
    expect(result.current.mascot).toBe('thinking')
  })

  it('plays oops when the turn fails or is blocked', () => {
    const failed = mount({ status: 'idle' })
    failed.rerender({ status: 'streaming' })
    failed.rerender({ status: 'error' })
    expect(failed.result.current.mascot).toBe('oops')

    const blocked = mount({ status: 'idle' })
    blocked.rerender({ status: 'streaming' })
    blocked.rerender({ status: 'idle', errorCode: CREDITS_EXHAUSTED_CODE })
    expect(blocked.result.current.mascot).toBe('oops')
  })

  it("ignores the chat's other parts", () => {
    const { result, rerender } = mount({ status: 'idle' })
    rerender({ status: 'streaming' })
    const before = result.current.state
    act(() => result.current.onData({ type: 'mission-suggestion', data: { goal: 'Plan' } }))
    act(() => result.current.onData({ type: 'data-usage', data: { totalTokens: 3 } }))
    expect(result.current.state).toBe(before)
  })
})

describe("useAutoMascot: the reply's words (T-003b)", () => {
  const reply = (hasText: boolean, toolRunning = false): ReplyProgress => ({ ...NO_REPLY, id: 'r2', hasText, toolRunning })

  it('send → thinking → first text → idle → tool-start → working → tool-end → idle → finish', () => {
    const { result, rerender } = mount({ status: 'idle', reply: NO_REPLY })
    rerender({ status: 'streaming', reply: reply(false) })
    expect(result.current.mascot).toBe('thinking')
    rerender({ status: 'streaming', reply: reply(true) })
    expect(result.current.mascot).toBe('idle')
    act(() => result.current.onData(toolStart('c1', 'query_database')))
    rerender({ status: 'streaming', reply: reply(true, true) })
    expect(result.current.mascot).toBe('working')
    act(() => result.current.onData(toolEnd('c1')))
    rerender({ status: 'streaming', reply: reply(true) })
    expect(result.current.mascot).toBe('idle')
    rerender({ status: 'idle', reply: reply(true) })
    expect(result.current.mascot).toBe('idle')
    expect(result.current.state.turnActive).toBe(false)
  })

  it("words that start while a tool runs keep the tool's state", () => {
    const { result, rerender } = mount({ status: 'idle', reply: NO_REPLY })
    rerender({ status: 'streaming', reply: reply(false) })
    act(() => result.current.onData(toolStart('c1', 'platform_web_search')))
    rerender({ status: 'streaming', reply: reply(false, true) })
    rerender({ status: 'streaming', reply: reply(true, true) })
    expect(result.current.mascot).toBe('searching')
  })

  it('thinks until the words come: an empty reply, or the previous reply already on screen, leaves him thinking', () => {
    const previousReply: ReplyProgress = { ...NO_REPLY, id: 'r1', hasText: true }
    const { result, rerender } = mount({ status: 'idle', reply: previousReply })
    rerender({ status: 'streaming', reply: previousReply })
    expect(result.current.mascot).toBe('thinking')
    rerender({ status: 'streaming', reply: reply(false) })
    expect(result.current.mascot).toBe('thinking')
  })
})

describe("useAutoMascot: the model's reasoning and a cap on the turn (P260-FX-16)", () => {
  const at = (fields: Partial<ReplyProgress> = {}): ReplyProgress => ({ ...NO_REPLY, id: 'r2', ...fields })

  /** A turn streaming on reply r2, with query_database running on it: Auto works. */
  function working() {
    const mounted = mount({ status: 'idle', reply: NO_REPLY })
    mounted.rerender({ status: 'streaming', reply: at() })
    act(() => mounted.result.current.onData(toolStart('c1', 'query_database')))
    mounted.rerender({ status: 'streaming', reply: at({ toolRunning: true }) })
    expect(mounted.result.current.mascot).toBe('working')
    return mounted
  }

  it("tool-start → tool-end (idle) → the reply's reasoning grows (thinking) → its words start (idle); growth while the tool runs changes nothing", () => {
    const { result, rerender } = working()
    rerender({ status: 'streaming', reply: at({ toolRunning: true, reasoningChars: 30 }) })
    expect(result.current.mascot).toBe('working')
    act(() => result.current.onData(toolEnd('c1')))
    rerender({ status: 'streaming', reply: at({ reasoningChars: 30 }) })
    expect(result.current.mascot).toBe('idle')
    rerender({ status: 'streaming', reply: at({ reasoningChars: 80 }) })
    expect(result.current.mascot).toBe('thinking')
    rerender({ status: 'streaming', reply: at({ reasoningChars: 80, hasText: true }) })
    expect(result.current.mascot).toBe('idle')
  })

  it('working → a cap → finish ends on oops', () => {
    const { result, rerender } = working()
    rerender({ status: 'streaming', reply: at({ toolRunning: true, limitReached: true }) })
    expect(result.current.mascot).toBe('oops')
    rerender({ status: 'idle', reply: at({ limitReached: true }) })
    expect(result.current.mascot).toBe('oops')
    expect(result.current.state.turnActive).toBe(false)
  })

  it('a cap that lands with the end of the stream still ends on oops, and the next send thinks', () => {
    const { result, rerender } = working()
    rerender({ status: 'idle', reply: at({ limitReached: true }) })
    expect(result.current.mascot).toBe('oops')
    expect(result.current.state.turnActive).toBe(false)
    rerender({ status: 'streaming', reply: NO_REPLY })
    expect(result.current.mascot).toBe('thinking')
  })
})

describe("useAutoMascot: the page's events", () => {
  it('listens while the message box has focus', () => {
    const { result } = mount()
    act(() => result.current.dispatch({ type: 'input-focus' }))
    expect(result.current.mascot).toBe('listening')
    act(() => result.current.dispatch({ type: 'input-blur' }))
    expect(result.current.mascot).toBe('idle')
  })

  it('says hello once per browser session', () => {
    const session = memoryStorage()
    const first = mount({ session: () => session })
    let played = false
    act(() => {
      played = first.result.current.greet()
    })
    expect(played).toBe(true)
    expect(first.result.current.mascot).toBe('hello')

    const again = mount({ session: () => session })
    act(() => {
      played = again.result.current.greet()
    })
    expect(played).toBe(false)
    expect(again.result.current.mascot).toBe('idle')
  })

  it('sleeps while the backend is down and wakes when it is up', () => {
    const { result, rerender } = mount({ backend: 'up' })
    expect(result.current.mascot).toBe('idle')
    rerender({ backend: 'down' })
    expect(result.current.mascot).toBe('sleeping')
    rerender({ backend: 'up' })
    expect(result.current.mascot).toBe('idle')
  })

  it('naps after the idle time without input, and wakes on input', () => {
    vi.useFakeTimers()
    const { result } = mount({ idleMs: NAP_MS })
    act(() => {
      vi.advanceTimersByTime(NAP_MS)
    })
    expect(result.current.mascot).toBe('sleeping')
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }))
    })
    expect(result.current.mascot).toBe('idle')
  })
})
