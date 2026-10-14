/**
 * PRD-260 D5, D6 — useAutoMascot: Auto driven by the core chat's onData parts
 * and status (through lib/mascot/events.ts), the page's events, hello once per
 * session, the backend's state and the idle nap.
 */
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type AutoMascotOptions, useAutoMascot } from '@/hooks/use-auto-mascot'
import { LIVE_KEY } from '@/lib/chat/receipts'
import { memoryStorage } from '@/lib/looks/__tests__/helpers'
import { receipt } from '@/lib/mascot/__tests__/helpers'
import { TRIAL_EXHAUSTED_CODE } from '@/lib/trial'

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

  it('thinks at once when it mounts on an awaited reply', () => {
    expect(mount({ status: 'streaming' }).result.current.mascot).toBe('thinking')
  })

  it('plays oops when the turn fails or is blocked', () => {
    const failed = mount({ status: 'idle' })
    failed.rerender({ status: 'streaming' })
    failed.rerender({ status: 'error' })
    expect(failed.result.current.mascot).toBe('oops')

    const blocked = mount({ status: 'idle' })
    blocked.rerender({ status: 'streaming' })
    blocked.rerender({ status: 'idle', errorCode: TRIAL_EXHAUSTED_CODE })
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
