/**
 * PRD-261 A-10a — SDK areas `notices` and `events`. A notice goes to the host's own sonner (the
 * Toaster components/providers.tsx mounts), checked field by field; a theme never imports sonner.
 * Host events reach a theme by the SDK's short names, with the detail as it came.
 */
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const toast = vi.hoisted(() => ({
  message: vi.fn(() => 'id-message'),
  success: vi.fn(() => 'id-success'),
  info: vi.fn(() => 'id-info'),
  warning: vi.fn(() => 'id-warning'),
  error: vi.fn(() => 'id-error'),
  dismiss: vi.fn(),
}))

vi.mock('sonner', () => ({ toast }))

import { GLOBAL_SEARCH_OPEN_EVENT } from '@/lib/global-search-event'
import { HOST_EVENT_NAMES, onHostEvent, openGlobalSearch, useHostEvent, type HostEventName } from '../events'
import { dismissNotice, notify, type Notice } from '../notices'

afterEach(() => {
  Object.values(toast).forEach((fn) => fn.mockClear())
})

describe('PRD-261 A-10a — notify', () => {
  it("shows a message in the host's toaster by default and returns its id", () => {
    expect(notify({ title: 'Saved' })).toBe('id-message')
    expect(toast.message).toHaveBeenCalledWith('Saved', {})
  })

  it('sends each kind to its own toast, with description, duration and action', () => {
    const onClick = vi.fn()
    expect(notify({ kind: 'success', title: 'Done', description: 'All three', durationMs: 4000, action: { label: 'Undo', onClick } })).toBe('id-success')
    const [title, options] = toast.success.mock.calls[0] as unknown as [string, { description: string; duration: number; action: { label: string; onClick: () => void } }]
    expect(title).toBe('Done')
    expect(options).toMatchObject({ description: 'All three', duration: 4000, action: { label: 'Undo' } })
    options.action.onClick()
    expect(onClick).toHaveBeenCalledTimes(1)
    notify({ kind: 'error', title: 'Failed' })
    notify({ kind: 'warning', title: 'Careful' })
    notify({ kind: 'info', title: 'FYI' })
    expect([toast.error, toast.warning, toast.info].map((fn) => fn.mock.calls.length)).toEqual([1, 1, 1])
  })

  it.each<[string, unknown]>([
    ['no notice', null],
    ['no title', { title: '   ' }],
    ['an unknown kind', { kind: 'toString', title: 'x' }],
    ['a non-text description', { title: 'x', description: 7 }],
    ['a duration that is not positive', { title: 'x', durationMs: 0 }],
    ['an action without onClick', { title: 'x', action: { label: 'Go' } }],
  ])('refuses %s, showing nothing', (_case, notice) => {
    expect(() => notify(notice as Notice)).toThrow(TypeError)
    expect(Object.values(toast).every((fn) => fn.mock.calls.length === 0)).toBe(true)
  })

  it('dismisses one notice, or all', () => {
    dismissNotice('id-1')
    dismissNotice()
    expect(toast.dismiss.mock.calls).toEqual([['id-1'], [undefined]])
  })
})

describe('PRD-261 A-10a — host events', () => {
  const WINDOW_EVENTS: Readonly<Record<HostEventName, string>> = {
    'board-changed': 'automatos:board-changed',
    'chat-changed': 'automatos:chat-changed',
  }

  it('names the events a theme may follow', () => {
    expect([...HOST_EVENT_NAMES].sort()).toEqual(Object.keys(WINDOW_EVENTS).sort())
    expect(Object.isFrozen(HOST_EVENT_NAMES)).toBe(true)
  })

  it.each(Object.entries(WINDOW_EVENTS))("hands %s's detail to the handler until unsubscribed", (name, type) => {
    const handler = vi.fn()
    const stop = onHostEvent(name as HostEventName, handler)
    window.dispatchEvent(new CustomEvent(type, { detail: { id: 7 } }))
    stop()
    window.dispatchEvent(new CustomEvent(type, { detail: { id: 8 } }))
    expect(handler.mock.calls).toEqual([[{ id: 7 }]])
  })

  it('refuses an event the host does not send, and a missing handler', () => {
    expect(() => onHostEvent('global-search-open' as HostEventName, () => undefined)).toThrow(TypeError)
    expect(() => onHostEvent('board-changed', undefined as unknown as () => void)).toThrow(TypeError)
  })

  it('follows an event for a component lifetime, calling the latest handler', () => {
    const first = vi.fn()
    const second = vi.fn()
    const { rerender, unmount } = renderHook(({ handler }) => useHostEvent('chat-changed', handler), { initialProps: { handler: first } })
    rerender({ handler: second })
    window.dispatchEvent(new CustomEvent('automatos:chat-changed', { detail: 'one' }))
    unmount()
    window.dispatchEvent(new CustomEvent('automatos:chat-changed', { detail: 'two' }))
    expect(first).not.toHaveBeenCalled()
    expect(second.mock.calls).toEqual([['one']])
  })

  it("opens the host's global search", () => {
    const opened = vi.fn()
    window.addEventListener(GLOBAL_SEARCH_OPEN_EVENT, opened)
    openGlobalSearch()
    window.removeEventListener(GLOBAL_SEARCH_OPEN_EVENT, opened)
    expect(opened).toHaveBeenCalledTimes(1)
  })
})
