/**
 * PRD-260 P260-FX-28 (M91) — Auto naps after three minutes without input, and any
 * input wakes him (D6, hooks/use-idle-timeout.ts). Ported from Automatos Play's
 * idle-timeout test, which the port had dropped. A regression net: if input stopped
 * re-arming the timer, Auto would nap while the owner types. Fake timers; the
 * callbacks are spies.
 */
import { act, fireEvent, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { IDLE_MS, useIdleTimeout } from '@/hooks/use-idle-timeout'

/** Play's mascot spec: sleeping after 3 minutes without input. */
const THREE_MINUTES_MS = 180_000
/** How long before the nap the owner presses a key. */
const KEY_BEFORE_NAP_MS = 1_000

function advance(ms: number): void {
  act(() => {
    vi.advanceTimersByTime(ms)
  })
}

describe('useIdleTimeout (M91)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('naps at IDLE_MS without input, not a millisecond before', () => {
    const onIdle = vi.fn()
    renderHook(() => useIdleTimeout(onIdle, vi.fn()))
    expect(IDLE_MS).toBe(THREE_MINUTES_MS)

    advance(IDLE_MS - 1)
    expect(onIdle).not.toHaveBeenCalled()
    advance(1)
    expect(onIdle).toHaveBeenCalledTimes(1)
  })

  it('starts the full IDLE_MS again on a key pressed just before the nap', () => {
    const onIdle = vi.fn()
    renderHook(() => useIdleTimeout(onIdle, vi.fn()))

    advance(IDLE_MS - KEY_BEFORE_NAP_MS)
    fireEvent.keyDown(window, { key: 'a' })
    advance(IDLE_MS - 1)
    expect(onIdle).not.toHaveBeenCalled()
    advance(1)
    expect(onIdle).toHaveBeenCalledTimes(1)
  })

  it('wakes once, on the first input after the nap', () => {
    const onIdle = vi.fn()
    const onActive = vi.fn()
    renderHook(() => useIdleTimeout(onIdle, onActive))

    fireEvent.pointerMove(window)
    expect(onActive).not.toHaveBeenCalled()
    advance(IDLE_MS)
    expect(onIdle).toHaveBeenCalledTimes(1)

    fireEvent.pointerMove(window)
    expect(onActive).toHaveBeenCalledTimes(1)
    fireEvent.keyDown(window, { key: 'a' })
    expect(onActive).toHaveBeenCalledTimes(1)
  })
})
