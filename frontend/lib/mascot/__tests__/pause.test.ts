/**
 * PRD-261 A-13b (F10, WCAG 2.2.2) — the Pause Auto flag: one per browser, kept
 * in localStorage, heard by every Auto on the page and in other tabs, and kept
 * for the page load when localStorage is blocked.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { isMascotPaused, PAUSE_KEY, type PauseStorage, setMascotPaused, subscribeMascotPause } from '@/lib/mascot/pause'

const blocked: PauseStorage = () => {
  throw new DOMException('blocked', 'SecurityError')
}

afterEach(() => {
  window.localStorage.removeItem(PAUSE_KEY)
  vi.restoreAllMocks()
})

describe('the Pause Auto flag', () => {
  it('is off until the user pauses him', () => {
    expect(isMascotPaused()).toBe(false)
  })

  it('is kept in localStorage, so a remount or a reload still finds him paused', () => {
    setMascotPaused(true)
    expect(window.localStorage.getItem(PAUSE_KEY)).toBe('1')
    expect(isMascotPaused()).toBe(true)
    setMascotPaused(false)
    expect(window.localStorage.getItem(PAUSE_KEY)).toBeNull()
    expect(isMascotPaused()).toBe(false)
  })

  it('tells every subscriber when it changes here, until they unsubscribe', () => {
    const first = vi.fn()
    const second = vi.fn()
    const stopFirst = subscribeMascotPause(first)
    const stopSecond = subscribeMascotPause(second)
    setMascotPaused(true)
    expect(first).toHaveBeenCalledTimes(1)
    expect(second).toHaveBeenCalledTimes(1)
    stopFirst()
    setMascotPaused(false)
    expect(first).toHaveBeenCalledTimes(1)
    expect(second).toHaveBeenCalledTimes(2)
    stopSecond()
  })

  it('hears another tab pause him (a storage event for its key, or a cleared store), and ignores other keys', () => {
    const onChange = vi.fn()
    const stop = subscribeMascotPause(onChange)
    window.dispatchEvent(new StorageEvent('storage', { key: 'automatos-play.looks' }))
    expect(onChange).not.toHaveBeenCalled()
    window.localStorage.setItem(PAUSE_KEY, '1')
    window.dispatchEvent(new StorageEvent('storage', { key: PAUSE_KEY }))
    expect(onChange).toHaveBeenCalledTimes(1)
    expect(isMascotPaused()).toBe(true)
    window.dispatchEvent(new StorageEvent('storage', { key: null }))
    expect(onChange).toHaveBeenCalledTimes(2)
    stop()
    window.dispatchEvent(new StorageEvent('storage', { key: PAUSE_KEY }))
    expect(onChange).toHaveBeenCalledTimes(2)
  })

  it('lasts for this page load when localStorage is blocked, and says so once', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    expect(isMascotPaused(blocked)).toBe(false)
    setMascotPaused(true, blocked)
    expect(isMascotPaused(blocked)).toBe(true)
    setMascotPaused(false, blocked)
    expect(isMascotPaused(blocked)).toBe(false)
    expect(warn).toHaveBeenCalledTimes(1)
  })
})
