/**
 * PRD-246 follow-up — a card grid opens as a LIST on a phone (Gerard, 2026-09-18),
 * the toggle stays, and a choice the user made is remembered per page and wins.
 */
import { renderHook, act } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useViewMode } from '@/hooks/use-view-mode'

function phone(matches: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })) as unknown as typeof window.matchMedia
}

describe('useViewMode', () => {
  beforeEach(() => localStorage.clear())

  it('opens as a grid on a desktop', () => {
    phone(false)
    const { result } = renderHook(() => useViewMode('t-desktop'))
    expect(result.current[0]).toBe('grid')
  })

  it('opens as a list on a phone when the user has not chosen', () => {
    phone(true)
    const { result } = renderHook(() => useViewMode('t-phone'))
    expect(result.current[0]).toBe('list')
  })

  it("keeps the user's grid choice on a phone, per page", () => {
    phone(true)
    localStorage.setItem('automatos-view-t-chosen', 'grid')
    const { result } = renderHook(() => useViewMode('t-chosen'))
    expect(result.current[0]).toBe('grid')
  })

  it('remembers a toggle', () => {
    phone(false)
    const { result } = renderHook(() => useViewMode('t-remember'))
    act(() => result.current[1]('list'))
    expect(result.current[0]).toBe('list')
    expect(localStorage.getItem('automatos-view-t-remember')).toBe('list')
  })
})
